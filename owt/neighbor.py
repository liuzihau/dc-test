"""Memory-bounded NP with the same token weighting and denominator as BD3."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from zebra.neighbor import NeighborHeads, validate_mechanisms


def initialize_heads(heads, initialization):
    """Match BD3's zero final vocabulary projection without changing shared RNG."""
    if initialization not in ('random', 'zero'):
        raise ValueError('NP initialization must be random or zero')
    if initialization == 'zero':
        for head in heads.heads:
            torch.nn.init.zeros_(head[-1].weight)
            torch.nn.init.zeros_(head[-1].bias)


def neighbor_terms(heads, hidden, clean, noisy, valid, mask_id, boundary_ids,
                   weights, denominator, chunk_size=128, ignore_first=True):
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    content = valid.bool().clone()
    for token in set(boundary_ids) | {mask_id}:
        content &= clean.ne(token)
    targets = content & noisy.eq(mask_id)
    if ignore_first:
        targets[:, 0] = False
    length = clean.shape[1]
    weights = torch.broadcast_to(weights, clean.shape)
    terms, counts = {}, {}
    for offset, head in zip(heads.offsets, heads.heads):
        distance = abs(offset)
        # Keep every head connected even when this rank has no eligible pairs.
        zero = hidden.sum() * 0.0 + sum(p.reshape(-1)[0] * 0.0 for p in head.parameters())
        if distance >= length:
            terms[offset], counts[offset] = zero, clean.new_zeros(())
            continue
        source = slice(0, length-distance) if offset > 0 else slice(distance, length)
        target = slice(distance, length) if offset > 0 else slice(0, length-distance)
        eligible = targets[:, target].clone()
        for shift in range(distance+1):
            eligible &= content[:, shift:length-distance+shift]
        h = hidden[:, source][eligible]
        y = clean[:, target][eligible]
        w = weights[:, target][eligible]
        loss = zero

        def chunk_ce(features, labels, factors, projection=head):
            logits = projection(features)
            # Preserve float64 in the numerical gradient check, use FP32 CE
            # with BF16 training. No dense [batch,length,vocab] aux tensors.
            if logits.dtype in (torch.float16, torch.bfloat16):
                logits = logits.float()
            forbidden = torch.arange(logits.shape[-1], device=logits.device).eq(mask_id)
            logits = logits.masked_fill(forbidden, -torch.inf)
            return (F.cross_entropy(logits, labels, reduction='none') * factors).sum()

        for start in range(0, y.numel(), chunk_size):
            inputs = (h[start:start+chunk_size], y[start:start+chunk_size], w[start:start+chunk_size])
            if torch.is_grad_enabled():
                loss = loss + checkpoint(chunk_ce, *inputs, use_reentrant=False,
                                         preserve_rng_state=False)
            else:
                loss = loss + chunk_ce(*inputs)
        terms[offset] = loss / denominator.clamp_min(1)
        counts[offset] = eligible.sum()
    return terms, counts
