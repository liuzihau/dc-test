"""Exactly one temporal gradient hop through overlapping recurrent states.

Ordinary truncated BPTT with disjoint windows would miss some adjacent edges.
Replacing a cache variable would instead preserve every historical edge. This
helper keeps independent local forward graphs, obtains each consumer's direct
cache-input cotangent, and delivers it to the preceding producer exactly once.
It changes first-order parameter gradients, never the numerical objective.
"""

from typing import List, Optional, Tuple

import torch


class AdjacentCacheGradients:
  """Bridge all neighboring graphs without connecting their input leaves.

  Call ``consume`` once at each boundary, supplying connected producer K/V.
  Use its returned leaves as the following forward's incoming K/V. Finally,
  call ``attach`` on the complete, weighted objective (including any local
  identity auxiliary), before the usual single optimizer backward.

  The direct cotangents are computed BEFORE adding bridges and are detached.
  Thus backward into a producer's incoming leaves cannot recursively reach an
  earlier producer. No model parameters or optimizer gradients are modified
  by the internal autograd.grad call. Second-order differentiation is not a
  supported interpretation of this deliberately truncated training rule.
  """

  def __init__(self, enabled: bool = True):
    self.enabled = bool(enabled)
    self._edges: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
    self._attached = False
    self.cotangent_norm: Optional[torch.Tensor] = None

  @property
  def num_edges(self) -> int:
    return len(self._edges)

  def consume(self, cache):
    if not self.enabled or not torch.is_grad_enabled() or cache is None:
      return cache
    if self._attached:
      raise RuntimeError('Cannot add a cache boundary after attaching gradients')
    if not cache:
      return cache
    pairs = []
    for source in cache:
      if not isinstance(source, torch.Tensor) or not source.is_floating_point():
        raise TypeError('DCache entries must be floating-point tensors')
      # The consumer does not mutate cache tensors, so a detached view avoids
      # a large copy. The independent autograd leaf is the temporal cut.
      leaf = source.detach().requires_grad_(True)
      pairs.append((source, leaf))
    self._edges.append(pairs)
    return [leaf for _, leaf in pairs]

  def attach(self, loss: torch.Tensor) -> torch.Tensor:
    if not self.enabled or not torch.is_grad_enabled():
      return loss
    if self._attached:
      raise RuntimeError('Adjacent gradients may be attached only once')
    self._attached = True
    if loss.numel() != 1:
      raise ValueError('Adjacent gradients require a scalar objective')
    self.cotangent_norm = loss.detach().new_zeros((), dtype=torch.float32)
    if not self._edges or not loss.requires_grad:
      return loss
    pairs = [pair for edge in self._edges for pair in edge]
    gradients = torch.autograd.grad(
      loss, [leaf for _, leaf in pairs], retain_graph=True,
      create_graph=False, allow_unused=True)
    # Accumulate outside BF16 to avoid low-precision reduction overflow.
    # Keep float64 for precise analytical/reference tests.
    accumulator_dtype = (
      torch.float64 if loss.dtype == torch.float64 else torch.float32)
    bridge = loss.new_zeros((), dtype=accumulator_dtype)
    squared_norm = loss.detach().new_zeros((), dtype=torch.float32)
    for (source, _), gradient in zip(pairs, gradients):
      if gradient is None:
        continue
      cotangent = gradient.detach()
      squared_norm = squared_norm + cotangent.float().square().sum()
      if source.requires_grad:
        bridge = bridge + (
          (source - source.detach()).to(accumulator_dtype)
          * cotangent.to(accumulator_dtype)).sum()
    self.cotangent_norm = squared_norm.sqrt()
    if not bool(torch.isfinite(self.cotangent_norm)):
      raise FloatingPointError('Nonfinite direct adjacent-cache gradient')
    # The bridge is zero in value, but its source derivative is the frozen
    # direct consumer cotangent. Do not use this augmented loss to recompute
    # cotangents: that would introduce the forbidden additional temporal hops.
    return loss + bridge
