"""Masked-target-only spatial auxiliary prediction, not a joint token decoder."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class NeighborPredictionHeads(nn.Module):
  """Two independent normalized vocabulary projections; no backbone input changes."""
  def __init__(self, hidden_size, vocab_size):
    super().__init__()
    self.heads = nn.ModuleDict({
      direction: nn.Sequential(nn.LayerNorm(hidden_size),
                               nn.Linear(hidden_size, vocab_size, bias=False))
      for direction in ('prev', 'next')})


def neighbor_prediction_loss(
    heads, hidden, clean, state, attention_mask, mask_index,
    excluded_token_ids=(), ignore_first=False, chunk_size=128,
    checkpoint_chunks=True):
  """Mean CE over eligible pairs PER DIRECTION and PER MICROBatch.

  Source may be clean or masked. Target must be masked. Both endpoints must
  be valid, non-special positions in the original clean tokens. Masked EOS is
  thus still excluded. There is no wraparound and no unobserved document-ID
  inference. DDP/accumulation average these microbatch means, like the existing
  training recipe; this is not a global-pair-count-weighted optimizer loss.

  Checkpoint scalar chunk losses (non-reentrant for local VJPs) so ten extra
  full-vocabulary activation arrays are not retained over five states.
  """
  if hidden.shape[:2] != clean.shape or state.shape != clean.shape or attention_mask.shape != clean.shape:
    raise ValueError('Neighbor tensors require matching [batch, sequence] dimensions')
  if chunk_size < 1:
    raise ValueError('Neighbor chunk_size must be positive')
  valid = attention_mask.bool().clone()
  for token in set(excluded_token_ids) | {mask_index}:
    if token is not None:
      valid &= clean.ne(int(token))
  if ignore_first and clean.shape[1]:
    valid[:, 0] = False
  masked = state.eq(mask_index)
  results = {}
  for direction, source, target in (
      ('prev', slice(1, None), slice(None, -1)),
      ('next', slice(None, -1), slice(1, None))):
    head = heads.heads[direction]
    pairs = valid[:, source] & valid[:, target] & masked[:, target]
    inputs = hidden[:, source][pairs]
    targets = clean[:, target][pairs]
    count = targets.numel()
    # Keep all parameters connected even on ranks/microbatches with no pairs.
    # Indexing one element avoids reducing the whole vocabulary matrix.
    zero = hidden.sum() * 0.0 + sum(p.reshape(-1)[0] * 0.0 for p in head.parameters())
    loss_sum, correct_sum = zero, hidden.new_zeros((), dtype=torch.float32)

    def chunk_loss(value, labels, projection=head):
      value = value.to(projection[1].weight.dtype)
      logits = projection(value).float()
      forbidden = torch.arange(logits.shape[-1], device=logits.device).eq(mask_index)
      logits = logits.masked_fill(forbidden[None], -torch.inf)
      return (F.cross_entropy(logits, labels, reduction='sum'),
              logits.detach().argmax(-1).eq(labels).float().sum())

    for start in range(0, count, chunk_size):
      value, labels = inputs[start:start + chunk_size], targets[start:start + chunk_size]
      if checkpoint_chunks and torch.is_grad_enabled():
        ce, correct = checkpoint(chunk_loss, value, labels, use_reentrant=False,
                                 preserve_rng_state=False)
      else:
        ce, correct = chunk_loss(value, labels)
      loss_sum = loss_sum + ce
      correct_sum = correct_sum + correct
    results[direction + '_loss'] = loss_sum / max(count, 1)
    results[direction + '_count'] = hidden.new_tensor(count, dtype=torch.float32)
    results[direction + '_accuracy'] = correct_sum / max(count, 1)
  results['loss'] = (results['prev_loss'] + results['next_loss']) / 2
  return results
