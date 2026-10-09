import torch
from types import SimpleNamespace

from models.dit import (
  DDiTBlock,
  DIT,
  Rotary,
  apply_denoising_rope_2d,
)


def make_block(enabled=True, gate_enabled=False, gate_init=0.1):
  block = DDiTBlock(
    n=2,
    dim=8,
    n_heads=2,
    adaLN=False,
    dropout=0.0,
    block_size=2,
    attn_backend='sdpa',
    step_memory_enabled=enabled,
    dc_spatial_rope_dim=2,
    dc_temporal_rope_dim=2,
    step_memory_gate_enabled=gate_enabled,
    step_memory_gate_init=gate_init)
  block.eval()
  return block


def rotary_for(x):
  return Rotary(dim=4)(x)


def test_disabled_step_memory_preserves_vanilla_output():
  torch.manual_seed(0)
  block = make_block(enabled=False)
  current = torch.randn(2, 2, 8)
  unused_previous_kv = torch.randn(2, 2, 2, 2, 4)

  vanilla = block(current, rotary_for(current), c=None, sample_mode=True)
  memory_argument_is_ignored = block(
    current, rotary_for(current), c=None, sample_mode=True,
    previous_step_kv=unused_previous_kv)

  torch.testing.assert_close(
    memory_argument_is_ignored, vanilla, rtol=0, atol=0)
  assert not any(name.startswith('dc_') for name, _ in block.named_parameters())


def test_previous_cache_backpropagates_to_previous_forward():
  torch.manual_seed(1)
  block = make_block(enabled=True)
  source = torch.randn(2, 2, 8, requires_grad=True)
  current = torch.randn(2, 2, 8, requires_grad=True)

  _, previous_kv = block(
    source, rotary_for(source), c=None, sample_mode=True,
    return_step_kv=True)
  output = block(
    current, rotary_for(current), c=None, sample_mode=True,
    previous_step_kv=previous_kv)
  output.square().mean().backward()

  assert source.grad is not None
  assert torch.count_nonzero(source.grad) > 0
  assert block.dc_qkv.weight.grad is not None
  assert block.dc_attn_out.weight.grad is not None


def test_block_returns_raw_entry_kv_before_normal_attention():
  torch.manual_seed(2)
  block = make_block(enabled=True)
  x = torch.randn(2, 5, 8)
  _, step_kv = block(
    x, rotary_for(x), c=None, sample_mode=True,
    return_step_kv=True)

  assert step_kv.shape == (2, 2, 2, 2, 4)

  with torch.no_grad():
    raw_qkv = block.dc_qkv(block.dc_norm(x[:, -2:]))
    raw_qkv = raw_qkv.reshape(2, 2, 3, 2, 4)
    expected_kv = raw_qkv[:, :, 1:]

  torch.testing.assert_close(step_kv, expected_kv)


def test_denoising_attention_runs_before_normal_attention():
  torch.manual_seed(5)
  block = make_block(enabled=True)
  x = torch.randn(1, 2, 8)
  call_order = []
  hooks = [
    block.dc_qkv.register_forward_hook(
      lambda module, inputs, output: call_order.append('dc')),
    block.attn_qkv.register_forward_hook(
      lambda module, inputs, output: call_order.append('normal')),
  ]
  try:
    block(x, rotary_for(x), c=None, sample_mode=True)
  finally:
    for hook in hooks:
      hook.remove()
  assert call_order == ['dc', 'normal']


def test_temporal_rope_distinguishes_previous_and_current_roles():
  torch.manual_seed(3)
  keys = torch.randn(2, 2, 2, 4)
  positions = torch.arange(2)
  previous = apply_denoising_rope_2d(
    keys, positions, temporal_position=0, spatial_dim=2)
  current = apply_denoising_rope_2d(
    keys, positions, temporal_position=1, spatial_dim=2)

  torch.testing.assert_close(previous[..., :2], current[..., :2])
  assert not torch.allclose(previous[..., 2:], current[..., 2:])


def test_source_mask_removes_disallowed_kv_before_softmax():
  torch.manual_seed(4)
  block = make_block(enabled=True)
  current = torch.randn(2, 2, 8)
  previous_a = torch.randn(2, 2, 2, 2, 4)
  previous_b = torch.randn(2, 2, 2, 2, 4)
  current_only = torch.full((2, 2), 2, dtype=torch.int8)
  cache_only = torch.full((2, 2), 1, dtype=torch.int8)

  current_a = block(
    current, rotary_for(current), c=None, sample_mode=True,
    previous_step_kv=previous_a,
    step_memory_source_mask=current_only)
  current_b = block(
    current, rotary_for(current), c=None, sample_mode=True,
    previous_step_kv=previous_b,
    step_memory_source_mask=current_only)
  cache_a = block(
    current, rotary_for(current), c=None, sample_mode=True,
    previous_step_kv=previous_a,
    step_memory_source_mask=cache_only)
  cache_b = block(
    current, rotary_for(current), c=None, sample_mode=True,
    previous_step_kv=previous_b,
    step_memory_source_mask=cache_only)

  torch.testing.assert_close(current_a, current_b, rtol=0, atol=0)
  assert not torch.allclose(cache_a, cache_b)


def test_step_memory_gate_uses_requested_effective_initial_value():
  block = make_block(enabled=True, gate_enabled=True, gate_init=0.1)

  torch.testing.assert_close(
    torch.tanh(block.step_memory_gate), torch.tensor(0.1))


def test_full_dit_supports_base_training_and_recurrent_sampling_shapes():
  config = SimpleNamespace(
    block_size=2,
    model=SimpleNamespace(
      causal_attention=False,
      length=4,
      hidden_size=8,
      cond_dim=4,
      n_heads=2,
      n_blocks=2,
      dropout=0.0,
      tie_word_embeddings=False,
      attn_backend='sdpa'),
    algo=SimpleNamespace(
      parameterization='subs',
      cross_attn=True),
    loader=SimpleNamespace(eval_batch_size=2),
    sampling=SimpleNamespace(kv_cache=False),
    step_memory=SimpleNamespace(
      enabled=True,
      spatial_rope_dim=2,
      temporal_rope_dim=2))
  model = DIT(config, vocab_size=11)
  model.eval()
  sigma = torch.zeros(2)

  doubled_training_input = torch.randint(0, 11, (2, 8))
  training_logits = model(doubled_training_input, sigma)
  assert training_logits.shape == (2, 4, 11)

  sampling_input = torch.randint(0, 11, (2, 4))
  first_logits, first_kv = model(
    sampling_input, sigma, sample_mode=True, return_step_kv=True)
  second_logits, second_kv = model(
    sampling_input, sigma, sample_mode=True,
    previous_step_kv=first_kv, return_step_kv=True)

  assert first_logits.shape == second_logits.shape == (2, 4, 11)
  assert len(first_kv) == len(second_kv) == 2
  assert first_kv[0].shape == second_kv[0].shape == (2, 2, 2, 2, 4)


def test_full_dit_returns_shifted_cache_m2_through_m_l_plus_1():
  config = SimpleNamespace(
    block_size=2,
    model=SimpleNamespace(
      causal_attention=False,
      length=2,
      hidden_size=8,
      cond_dim=4,
      n_heads=2,
      n_blocks=2,
      dropout=0.0,
      tie_word_embeddings=False,
      attn_backend='sdpa'),
    algo=SimpleNamespace(parameterization='subs', cross_attn=False),
    loader=SimpleNamespace(eval_batch_size=1),
    sampling=SimpleNamespace(kv_cache=False),
    step_memory=SimpleNamespace(
      enabled=True, spatial_rope_dim=2, temporal_rope_dim=2))
  model = DIT(config, vocab_size=11).eval()
  tokens = torch.randint(0, 11, (1, 2))
  captured = {}

  def capture_m2(module, inputs, output):
    captured['m2_qkv'] = output.detach()

  def capture_m3(module, inputs, output):
    captured['m3_kv'] = output.detach()

  hooks = [
    model.blocks[1].dc_qkv.register_forward_hook(capture_m2),
    model.dc_final_writer.kv.register_forward_hook(capture_m3),
  ]
  try:
    _, cache = model(
      tokens, torch.zeros(1), sample_mode=True, return_step_kv=True)
  finally:
    for hook in hooks:
      hook.remove()

  expected_m2 = captured['m2_qkv'].reshape(1, 2, 3, 2, 4)[:, :, 1:]
  expected_m3 = captured['m3_kv'].reshape(1, 2, 2, 2, 4)
  assert len(cache) == 2
  torch.testing.assert_close(cache[0], expected_m2)
  torch.testing.assert_close(cache[1], expected_m3)


def test_detached_backbone_cache_still_trains_final_writer():
  torch.manual_seed(6)
  config = SimpleNamespace(
    block_size=2,
    model=SimpleNamespace(
      causal_attention=False, length=2, hidden_size=8, cond_dim=4,
      n_heads=2, n_blocks=2, dropout=0.0, tie_word_embeddings=False,
      attn_backend='sdpa'),
    algo=SimpleNamespace(parameterization='subs', cross_attn=False),
    loader=SimpleNamespace(eval_batch_size=1),
    sampling=SimpleNamespace(kv_cache=False),
    step_memory=SimpleNamespace(
      enabled=True, spatial_rope_dim=2, temporal_rope_dim=2))
  model = DIT(config, vocab_size=11)
  torch.nn.init.normal_(model.output_layer.linear.weight, std=0.02)
  tokens = torch.randint(0, 11, (1, 2))
  _, first_cache = model(
    tokens, torch.zeros(1), sample_mode=True, return_step_kv=True,
    detach_cache_backbone=True)
  output = model(
    tokens, torch.zeros(1), sample_mode=True,
    previous_step_kv=first_cache)
  output.square().mean().backward()
  assert model.dc_final_writer.kv.weight.grad is not None
  assert torch.count_nonzero(model.dc_final_writer.kv.weight.grad) > 0
