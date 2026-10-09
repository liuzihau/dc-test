"""Target-mask-only auxiliary prediction, independent of recurrent training."""

import copy

import pytest
import torch
import torch.nn.functional as F

from neighbor_prediction import NeighborPredictionHeads, neighbor_prediction_loss


def make_heads(hidden_size=4, vocab_size=10):
    torch.manual_seed(113)
    return NeighborPredictionHeads(hidden_size, vocab_size).float()


def call_loss(heads, hidden, clean, state, attention=None, **kwargs):
    if attention is None:
        attention = torch.ones_like(clean)
    return neighbor_prediction_loss(
        heads, hidden, clean, state, attention, mask_index=0, **kwargs)


def directional_reference(heads, direction, hidden, target):
    logits = heads.heads[direction](hidden)
    # MASK is an input status, never a candidate clean prediction.
    logits = torch.cat((torch.full_like(logits[:, :1], -torch.inf),
                        logits[:, 1:]), dim=-1)
    return F.cross_entropy(logits, target)


def test_clean_sources_predict_masked_neighbor_in_correct_direction():
    heads = make_heads()
    hidden = torch.tensor(
        [[[1., 2., 3., 4.], [2., 4., 1., 3.], [4., 1., 3., 2.]]],
        dtype=torch.float32, requires_grad=True)
    clean = torch.tensor([[1, 2, 3]])
    state = torch.tensor([[1, 0, 3]])
    result = call_loss(heads, hidden, clean, state)

    # Source at C predicts its PREVIOUS neighbor B; A predicts its NEXT B.
    expected_prev = directional_reference(
        heads, 'prev', hidden[:, 2], clean[:, 1])
    expected_next = directional_reference(
        heads, 'next', hidden[:, 0], clean[:, 1])
    torch.testing.assert_close(result['prev_loss'], expected_prev)
    torch.testing.assert_close(result['next_loss'], expected_next)
    torch.testing.assert_close(result['loss'], (expected_prev + expected_next) / 2)
    assert result['prev_count'].item() == 1
    assert result['next_count'].item() == 1
    result['loss'].backward()
    assert hidden.grad[:, 0].abs().sum() > 0
    assert hidden.grad[:, 2].abs().sum() > 0
    assert hidden.grad[:, 1].abs().sum() == 0


def test_source_mask_status_does_not_change_pair_eligibility():
    heads = make_heads()
    hidden = torch.randn(1, 3, 4, dtype=torch.float32)
    clean = torch.tensor([[1, 2, 3]])
    clean_source = call_loss(heads, hidden, clean, torch.tensor([[1, 0, 3]]))
    all_masked = call_loss(heads, hidden, clean, torch.zeros_like(clean))
    assert clean_source['prev_count'].item() == 1
    assert clean_source['next_count'].item() == 1
    # Newly masked targets add pairs; they do not invalidate existing sources.
    assert all_masked['prev_count'].item() == 2
    assert all_masked['next_count'].item() == 2


@pytest.mark.parametrize('masked_index,expected_prev,expected_next', [
    (0, 1, 0), (2, 0, 1),
])
def test_sequence_edges_never_wrap(masked_index, expected_prev, expected_next):
    heads = make_heads()
    hidden = torch.randn(1, 3, 4, dtype=torch.float32)
    clean = torch.tensor([[1, 2, 3]])
    state = clean.clone()
    state[:, masked_index] = 0
    result = call_loss(heads, hidden, clean, state)
    assert result['prev_count'].item() == expected_prev
    assert result['next_count'].item() == expected_next
    absent = 'prev' if expected_prev == 0 else 'next'
    assert result[f'{absent}_loss'].item() == 0
    assert result[f'{absent}_accuracy'].item() == 0


def test_original_special_ids_and_padding_exclude_both_pair_endpoints():
    heads = make_heads()
    hidden = torch.randn(1, 7, 4, dtype=torch.float32)
    clean = torch.tensor([[1, 2, 8, 3, 4, 9, 5]])
    state = torch.zeros_like(clean)
    attention = torch.tensor([[1, 1, 1, 1, 1, 1, 0]])
    result = call_loss(
        heads, hidden, clean, state, attention, excluded_token_ids=(8, 9))
    assert result['prev_count'].item() == 2
    assert result['next_count'].item() == 2
    expected_prev = directional_reference(
        heads, 'prev', hidden[0, [1, 4]], clean[0, [0, 3]])
    expected_next = directional_reference(
        heads, 'next', hidden[0, [0, 3]], clean[0, [1, 4]])
    torch.testing.assert_close(result['prev_loss'], expected_prev)
    torch.testing.assert_close(result['next_loss'], expected_next)


def test_ignore_first_excludes_first_position_as_source_and_target():
    heads = make_heads()
    hidden = torch.randn(1, 3, 4, dtype=torch.float32)
    clean = torch.tensor([[1, 2, 3]])
    result = call_loss(
        heads, hidden, clean, torch.zeros_like(clean), ignore_first=True)
    assert result['prev_count'].item() == 1
    assert result['next_count'].item() == 1
    torch.testing.assert_close(
        result['prev_loss'],
        directional_reference(heads, 'prev', hidden[:, 2], clean[:, 1]))
    torch.testing.assert_close(
        result['next_loss'],
        directional_reference(heads, 'next', hidden[:, 1], clean[:, 2]))


@pytest.mark.parametrize('empty_case', ['clean', 'padding', 'special', 'length_one'])
def test_empty_eligible_pairs_have_zero_loss_and_present_zero_gradients(empty_case):
    heads = make_heads()
    length = 1 if empty_case == 'length_one' else 3
    hidden = torch.randn(1, length, 4, dtype=torch.float32, requires_grad=True)
    clean = torch.ones(1, length, dtype=torch.long)
    state = clean if empty_case == 'clean' else torch.zeros_like(clean)
    attention = torch.zeros_like(clean) if empty_case == 'padding' else None
    excluded = (1,) if empty_case == 'special' else ()
    projection_calls = []
    handles = [module.register_forward_hook(
        lambda module, inputs, output: projection_calls.append(True))
        for module in heads.heads.values()]
    result = call_loss(
        heads, hidden, clean, state, attention, excluded_token_ids=excluded)
    for handle in handles:
        handle.remove()
    assert not projection_calls
    for name in ('loss', 'prev_loss', 'next_loss', 'prev_count', 'next_count',
                 'prev_accuracy', 'next_accuracy'):
        assert torch.is_tensor(result[name]), name
        assert torch.isfinite(result[name]).all(), name
        assert result[name].item() == 0, name
    result['loss'].backward()
    for parameter in [hidden] + list(heads.parameters()):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


def test_mask_logit_is_excluded_from_cross_entropy_and_accuracy():
    heads = make_heads()
    with torch.no_grad():
        for direction in ('prev', 'next'):
            linear = heads.heads[direction][-1]
            linear.weight.zero_()
            linear.weight[0, 0] = 1.e6
            linear.weight[2, 0] = 2.0
    hidden = torch.tensor([[[1., 0., 0., 0.]] * 3], dtype=torch.float32)
    clean = torch.tensor([[1, 2, 3]])
    result = call_loss(heads, hidden, clean, torch.tensor([[1, 0, 3]]))
    assert result['prev_accuracy'].item() == 1
    assert result['next_accuracy'].item() == 1
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    for direction in ('prev', 'next'):
        assert torch.count_nonzero(heads.heads[direction][-1].weight.grad[0]) == 0


@pytest.mark.parametrize('chunk_size', [1, 2, 5])
def test_checkpointed_chunks_match_dense_loss_and_input_parameter_vjps(chunk_size):
    reference_heads = make_heads()
    checkpointed_heads = copy.deepcopy(reference_heads)
    torch.manual_seed(151)
    reference_hidden = torch.randn(
        2, 7, 4, dtype=torch.float32, requires_grad=True)
    checkpointed_hidden = reference_hidden.detach().clone().requires_grad_(True)
    clean = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [1, 8, 3, 4, 5, 6, 7]])
    state = torch.tensor([[0, 2, 0, 0, 5, 0, 7], [0, 0, 0, 4, 0, 0, 0]])
    attention = torch.ones_like(clean)
    attention[1, -1] = 0
    reference = call_loss(
        reference_heads, reference_hidden, clean, state, attention,
        excluded_token_ids=(8,), chunk_size=1000, checkpoint_chunks=False)
    actual = call_loss(
        checkpointed_heads, checkpointed_hidden, clean, state, attention,
        excluded_token_ids=(8,), chunk_size=chunk_size, checkpoint_chunks=True)
    for name in reference:
        torch.testing.assert_close(actual[name], reference[name], rtol=1e-6, atol=1e-6)
    reference_inputs = [reference_hidden] + list(reference_heads.parameters())
    actual_inputs = [checkpointed_hidden] + list(checkpointed_heads.parameters())
    expected_vjp = torch.autograd.grad(reference['loss'], reference_inputs, retain_graph=True)
    actual_vjp = torch.autograd.grad(actual['loss'], actual_inputs, retain_graph=True)
    for actual_grad, expected_grad in zip(actual_vjp, expected_vjp):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)
    # AdjacentCacheGradients requests a VJP before the ordinary backward.
    actual['loss'].backward()
    for parameter, expected_grad in zip(actual_inputs, expected_vjp):
        torch.testing.assert_close(parameter.grad, expected_grad, rtol=1e-5, atol=1e-6)


def test_directional_losses_are_means_not_sums_over_eligible_pairs():
    heads = make_heads()
    hidden = torch.randn(1, 3, 4, dtype=torch.float32)
    clean = torch.tensor([[1, 2, 3]])
    state = torch.tensor([[1, 0, 3]])
    single = call_loss(heads, hidden, clean, state)
    duplicated = call_loss(
        heads, hidden.repeat(3, 1, 1), clean.repeat(3, 1), state.repeat(3, 1))
    for name in ('loss', 'prev_loss', 'next_loss', 'prev_accuracy', 'next_accuracy'):
        torch.testing.assert_close(duplicated[name], single[name])
    assert duplicated['prev_count'].item() == 3 * single['prev_count'].item()
    assert duplicated['next_count'].item() == 3 * single['next_count'].item()


def test_no_grad_evaluation_matches_training_loss_without_building_graph():
    heads = make_heads()
    hidden = torch.randn(1, 3, 4, requires_grad=True)
    clean = torch.tensor([[1, 2, 3]])
    state = torch.tensor([[1, 0, 3]])
    training = call_loss(heads, hidden, clean, state, checkpoint_chunks=True)
    with torch.inference_mode():
        evaluation = call_loss(heads, hidden, clean, state, checkpoint_chunks=True)
    for name in training:
        torch.testing.assert_close(evaluation[name], training[name])
        assert not evaluation[name].requires_grad
