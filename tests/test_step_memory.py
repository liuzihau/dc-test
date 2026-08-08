import torch

from models.dit import DDiTBlock, Rotary


def make_block():
  block = DDiTBlock(
    n=2,
    dim=8,
    n_heads=2,
    adaLN=False,
    dropout=0.0,
    block_size=2,
    attn_backend='sdpa')
  block.eval()
  return block


def rotary_for(x):
  return Rotary(dim=4)(x)


def test_zero_gate_preserves_vanilla_output():
  torch.manual_seed(0)
  block = make_block()
  source = torch.randn(2, 2, 8)
  current = torch.randn(2, 2, 8)

  _, previous_qkv = block(
    source, rotary_for(source), c=None, sample_mode=True,
    return_step_qkv=True)
  vanilla = block(current, rotary_for(current), c=None, sample_mode=True)
  memory_conditioned = block(
    current, rotary_for(current), c=None, sample_mode=True,
    previous_step_qkv=previous_qkv)

  torch.testing.assert_close(memory_conditioned, vanilla, rtol=0, atol=0)


def test_undetached_cache_backpropagates_to_previous_forward():
  torch.manual_seed(1)
  block = make_block()
  with torch.no_grad():
    block.step_memory_gate.fill_(0.2)

  source = torch.randn(2, 2, 8, requires_grad=True)
  current = torch.randn(2, 2, 8, requires_grad=True)
  _, previous_qkv = block(
    source, rotary_for(source), c=None, sample_mode=True,
    return_step_qkv=True)
  output = block(
    current, rotary_for(current), c=None, sample_mode=True,
    previous_step_qkv=previous_qkv)
  output.square().mean().backward()

  assert source.grad is not None
  assert torch.count_nonzero(source.grad) > 0
  assert block.step_memory_gate.grad is not None


def test_returned_memory_is_only_the_active_block():
  block = make_block()
  x = torch.randn(2, 5, 8)
  _, step_qkv = block(
    x, rotary_for(x), c=None, sample_mode=True,
    return_step_qkv=True)
  assert step_qkv.shape == (2, 2, 3, 2, 4)
