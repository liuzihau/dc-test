"""Prove overlapping one-hop DCache gradients, without multi-step BPTT.

The reference explicitly recomputes a fresh producer/consumer pair for each
target loss. The producer's incoming cache is fixed to its original numerical
value. This is intentionally independent of the surrogate implementation.
"""

import copy

import pytest
import torch

from recurrent_gradients import AdjacentCacheGradients


class ToyDenoiser(torch.nn.Module):
  """Shared backbone plus two layers' K/V projections and final feedback."""

  def __init__(self, detach_writer=False):
    super().__init__()
    self.detach_writer = detach_writer
    generator = torch.Generator().manual_seed(97)

    def parameter(*shape):
      return torch.nn.Parameter(
        0.25 * torch.randn(*shape, generator=generator, dtype=torch.float64))

    self.backbone = parameter(3, 3)
    self.writer = parameter(4, 3, 3)
    self.reader = parameter(4, 3, 3)
    self.final_read = parameter(3, 3)
    self.final_write = parameter(3, 3)
    self.head = parameter(3, 2)

  def forward(self, value, cache=None, final=None, source="all"):
    pre_hidden = value @ self.backbone
    if cache is not None and source != "none":
      entries = range(4) if source == "all" else (0, 2)
      for index in entries:
        pre_hidden = pre_hidden + 0.4 * cache[index] @ self.reader[index]
    if final is not None:
      pre_hidden = pre_hidden + 0.3 * final.detach() @ self.final_read
    hidden = torch.tanh(pre_hidden)
    writer_input = hidden.detach() if self.detach_writer else hidden
    cache = [writer_input @ self.writer[index] for index in range(4)]
    final = torch.sin(hidden @ self.final_write)
    prediction = hidden @ self.head
    return prediction, cache, final, hidden


def example_states():
  generator = torch.Generator().manual_seed(127)
  values = torch.randn(5, 3, 3, generator=generator, dtype=torch.float64)
  targets = torch.randn(5, 3, 2, generator=generator, dtype=torch.float64)
  return values, targets


def reconstruction(prediction, target):
  return (prediction - target).square().mean()


def one_hop_run(model, values, targets, weights, sources=None, identity=False):
  helper = AdjacentCacheGradients(enabled=True)
  sources = sources or ["all"] * len(values)
  previous_cache = None
  previous_final = None
  histories = []
  losses = []
  for step, (value, target) in enumerate(zip(values, targets)):
    cache_input = (
      None if previous_cache is None else helper.consume(previous_cache))
    prediction, cache, final, hidden = model(
      value, cache_input, previous_final, source=sources[step])
    hidden.retain_grad()
    final.retain_grad()
    loss = reconstruction(prediction, target)
    if identity and cache_input is not None:
      shuffled = [entry.roll(1, dims=0) for entry in cache_input]
      wrong_prediction, _, _, _ = model(
        value, shuffled, previous_final, source=sources[step])
      wrong_loss = reconstruction(wrong_prediction, target)
      # A large margin deliberately makes this hinge active in every example.
      # Match the canonical identity objective: the corrupted branch is a
      # fixed reference, not a target that can improve by becoming worse.
      loss = loss + 0.1 * torch.relu(loss - wrong_loss.detach() + 10.0)
    histories.append((cache, final, hidden))
    losses.append(loss)
    previous_cache, previous_final = cache, final
  scalar = sum(weight * loss for weight, loss in zip(weights, losses))
  scalar = scalar / sum(weights)
  return helper.attach(scalar), scalar, histories


def explicit_two_state_reference(model, values, targets, weights,
                                 sources=None, identity=False):
  sources = sources or ["all"] * len(values)
  numeric_cache = []
  numeric_final = []
  with torch.no_grad():
    cache, final = None, None
    for step, value in enumerate(values):
      _, cache, final, _ = model(value, cache, final, source=sources[step])
      numeric_cache.append(cache)
      numeric_final.append(final)

  losses = []
  for target_step, (value, target) in enumerate(zip(values, targets)):
    if target_step == 0:
      incoming_cache, previous_final = None, None
    else:
      producer = target_step - 1
      producer_input = numeric_cache[producer - 1] if producer else None
      producer_final = numeric_final[producer - 1] if producer else None
      _, incoming_cache, _, _ = model(
        values[producer], producer_input, producer_final,
        source=sources[producer])
      previous_final = numeric_final[producer]
    prediction, _, _, _ = model(
      value, incoming_cache, previous_final, source=sources[target_step])
    loss = reconstruction(prediction, target)
    if identity and incoming_cache is not None:
      shuffled = [entry.roll(1, dims=0) for entry in incoming_cache]
      wrong_prediction, _, _, _ = model(
        value, shuffled, previous_final, source=sources[target_step])
      wrong_loss = reconstruction(wrong_prediction, target)
      loss = loss + 0.1 * torch.relu(loss - wrong_loss.detach() + 10.0)
    losses.append(loss)
  return sum(weight * loss for weight, loss in zip(weights, losses)) / sum(weights)


@pytest.mark.parametrize("sources", [
  ["all"] * 5,
  ["all", "none", "partial", "all", "none"],
])
@pytest.mark.parametrize("identity", [False, True])
def test_five_state_gradients_match_explicit_overlapping_two_state_graphs(
    sources, identity):
  actual_model = ToyDenoiser()
  reference_model = copy.deepcopy(actual_model)
  values, targets = example_states()
  weights = [0.05, 0.10, 0.20, 1.0, 0.70]
  augmented, original, _ = one_hop_run(
    actual_model, values, targets, weights, sources, identity)
  reference = explicit_two_state_reference(
    reference_model, values, targets, weights, sources, identity)

  torch.testing.assert_close(augmented.detach(), original.detach(), rtol=0, atol=0)
  torch.testing.assert_close(augmented.detach(), reference.detach(), rtol=0, atol=1e-14)
  augmented.backward()
  reference.backward()
  for (name, actual), (reference_name, expected) in zip(
      actual_model.named_parameters(), reference_model.named_parameters()):
    assert name == reference_name
    if expected.grad is None:
      assert actual.grad is None or torch.count_nonzero(actual.grad) == 0, name
    else:
      assert actual.grad is not None, name
      torch.testing.assert_close(
        actual.grad, expected.grad, rtol=1e-10, atol=1e-12, msg=name)


@pytest.mark.parametrize("target_step", [1, 2, 3, 4])
def test_each_loss_reaches_exactly_its_adjacent_producer(target_step):
  values, targets = example_states()
  weights = [float(step == target_step) for step in range(5)]
  augmented, _, histories = one_hop_run(
    ToyDenoiser(), values, targets, weights)
  augmented.backward()

  for step, (_, final, hidden) in enumerate(histories):
    nonzero_gradient = hidden.grad is not None and torch.count_nonzero(hidden.grad) > 0
    assert bool(nonzero_gradient) == (step in (target_step - 1, target_step))
    # The previous final-state route is detached independently of DCache.
    assert final.grad is None or torch.count_nonzero(final.grad) == 0


def test_naive_cache_replacement_still_leaks_gradient_across_multiple_steps():
  model = ToyDenoiser()
  values, targets = example_states()
  cache, final = None, None
  histories = []
  for value in values[:3]:
    prediction, cache, final, hidden = model(value, cache, final)
    hidden.retain_grad()
    histories.append(hidden)
  reconstruction(prediction, targets[2]).backward()
  # Replacing the Python `cache` variable does not sever its autograd history.
  assert histories[0].grad is not None
  assert torch.count_nonzero(histories[0].grad) > 0


def test_consumed_cache_has_identical_values_but_is_a_detached_leaf():
  producer = torch.tensor([2.0, -1.0], requires_grad=True)
  cache = [producer.square(), producer.sin()]
  helper = AdjacentCacheGradients(enabled=True)
  consumed = helper.consume(cache)
  assert len(consumed) == len(cache)
  for original, leaf in zip(cache, consumed):
    torch.testing.assert_close(original, leaf, rtol=0, atol=0)
    assert leaf.is_leaf and leaf.requires_grad
    assert leaf.grad_fn is None
  loss = helper.attach(consumed[0].sum() + consumed[1].sum())
  loss.backward()
  torch.testing.assert_close(producer.grad, 2 * producer.detach() + producer.detach().cos())


def test_unused_cache_and_source_dropout_do_not_add_spurious_gradients():
  producer = torch.tensor(2.0, requires_grad=True)
  unrelated = torch.tensor(3.0, requires_grad=True)
  helper = AdjacentCacheGradients(enabled=True)
  helper.consume([producer.square(), producer.sin()])
  loss = unrelated.square()
  augmented = helper.attach(loss)
  torch.testing.assert_close(augmented.detach(), loss.detach(), rtol=0, atol=0)
  augmented.backward()
  assert producer.grad is None or producer.grad == 0
  torch.testing.assert_close(unrelated.grad, torch.tensor(6.0))


def test_two_consumers_of_same_source_accumulate_their_one_hop_contributions():
  producer = torch.tensor(2.0, requires_grad=True)
  helper = AdjacentCacheGradients(enabled=True)
  source = [producer.square()]
  correct = helper.consume(source)[0]
  corrupt = helper.consume(source)[0]
  loss = correct * 3.0 + torch.relu(correct - corrupt * 0.5 + 10.0) * 0.1
  helper.attach(loss).backward()
  torch.testing.assert_close(producer.grad, torch.tensor(12.2))


def test_empty_helper_preserves_normal_backward():
  value = torch.tensor(2.0, requires_grad=True)
  helper = AdjacentCacheGradients(enabled=True)
  cache = []
  assert helper.consume(cache) == []
  original = value.square()
  augmented = helper.attach(original)
  torch.testing.assert_close(augmented.detach(), original.detach(), rtol=0, atol=0)
  augmented.backward()
  torch.testing.assert_close(value.grad, torch.tensor(4.0))


def test_disabled_helper_preserves_existing_cache_connectivity():
  value = torch.tensor(2.0, requires_grad=True)
  cache = [value.square()]
  helper = AdjacentCacheGradients(enabled=False)
  consumed = helper.consume(cache)
  assert consumed is cache
  original = consumed[0] * 3.0
  assert helper.attach(original) is original
  original.backward()
  torch.testing.assert_close(value.grad, torch.tensor(12.0))


def test_no_grad_evaluation_does_not_create_trainable_cache_or_surrogates():
  helper = AdjacentCacheGradients(enabled=True)
  with torch.no_grad():
    cache = [torch.tensor([2.0, 3.0])]
    consumed = helper.consume(cache)
    assert consumed is cache
    assert not consumed[0].requires_grad
    loss = consumed[0].sum()
    assert helper.attach(loss) is loss
    assert not loss.requires_grad


def test_detached_writer_input_matches_legacy_writer_gradients_exactly():
  """The bridge neither loses nor doubles the writer gradient already in V2."""
  actual_model = ToyDenoiser(detach_writer=True)
  legacy_model = copy.deepcopy(actual_model)
  values, targets = example_states()
  weights = [0.05, 0.10, 0.20, 1.0, 0.70]
  actual, _, _ = one_hop_run(actual_model, values, targets, weights)

  cache, final = None, None
  legacy_losses = []
  for value, target in zip(values, targets):
    prediction, cache, final, _ = legacy_model(value, cache, final)
    legacy_losses.append(reconstruction(prediction, target))
  legacy = sum(weight * loss for weight, loss in zip(weights, legacy_losses)) / sum(weights)

  torch.testing.assert_close(actual.detach(), legacy.detach(), rtol=0, atol=0)
  actual.backward()
  legacy.backward()
  assert torch.count_nonzero(legacy_model.writer.grad) > 0
  for (name, actual_parameter), (_, legacy_parameter) in zip(
      actual_model.named_parameters(), legacy_model.named_parameters()):
    if legacy_parameter.grad is None:
      assert actual_parameter.grad is None or torch.count_nonzero(actual_parameter.grad) == 0
    else:
      torch.testing.assert_close(
        actual_parameter.grad, legacy_parameter.grad,
        rtol=1e-10, atol=1e-12, msg=name)


def test_no_grad_never_requests_vjp_even_with_recorded_training_edges(monkeypatch):
  helper = AdjacentCacheGradients(enabled=True)
  value = torch.tensor(2.0, requires_grad=True)
  consumed = helper.consume([value.square()])
  loss = consumed[0].sum()

  def forbidden_vjp(*args, **kwargs):
    raise AssertionError("Evaluation must not request autograd.grad")

  monkeypatch.setattr(torch.autograd, "grad", forbidden_vjp)
  with torch.no_grad():
    assert helper.attach(loss) is loss


def test_single_attach_and_no_late_boundaries_enforce_one_shot_gradient_rule():
  helper = AdjacentCacheGradients(enabled=True)
  value = torch.tensor(2.0, requires_grad=True)
  consumed = helper.consume([value.square()])
  loss = consumed[0].sum()
  helper.attach(loss)
  with pytest.raises(RuntimeError, match="only once"):
    helper.attach(loss)
  with pytest.raises(RuntimeError, match="after attaching"):
    helper.consume([value.square()])
