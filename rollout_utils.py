import torch


def build_rollout_mask_counts(
    block_size, num_forwards, final_mask_count, device):
  """Sample a valid, strictly decreasing rollout mask-count path."""
  if not 2 <= num_forwards <= block_size:
    raise ValueError('num_forwards must be between 2 and block_size')
  max_final = block_size - (num_forwards - 1)
  final_mask_count = max(1, min(int(final_mask_count), max_final))
  if num_forwards == 2:
    return [block_size, final_mask_count]

  candidates = torch.arange(
    final_mask_count + 1, block_size, device=device)
  needed = num_forwards - 2
  selected = candidates[
    torch.randperm(candidates.numel(), device=device)[:needed]]
  intermediate = selected.sort(descending=True).values.tolist()
  return [block_size, *intermediate, final_mask_count]
