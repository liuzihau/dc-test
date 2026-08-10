import torch

from rollout_utils import build_rollout_mask_counts


def test_early_rollout_reveals_one_and_keeps_one_mask():
  counts = build_rollout_mask_counts(
    block_size=16,
    num_forwards=2,
    final_mask_count=15,
    device=torch.device('cpu'))
  assert counts == [16, 15]


def test_mature_rollout_is_strict_and_reaches_one_mask():
  torch.manual_seed(0)
  counts = build_rollout_mask_counts(
    block_size=16,
    num_forwards=5,
    final_mask_count=1,
    device=torch.device('cpu'))

  assert counts[0] == 16
  assert counts[-1] == 1
  assert len(counts) == 5
  assert all(current > following >= 1
             for current, following in zip(counts, counts[1:]))
