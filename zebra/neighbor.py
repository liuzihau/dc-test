"""Independent neighbor heads and target-indexed auxiliary loss."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class NeighborHeads(nn.Module):
    def __init__(self, hidden_size, vocab_size, offsets, hidden_layers=0):
        super().__init__()
        if any(isinstance(x, bool) or int(x) != x for x in offsets):
            raise ValueError('NP offsets must be integers')
        self.offsets = tuple(int(x) for x in offsets)
        if not self.offsets or 0 in self.offsets or len(set(self.offsets)) != len(self.offsets):
            raise ValueError('NP offsets must be distinct, nonzero integers')
        if hidden_layers < 0:
            raise ValueError('NP hidden_layers must be nonnegative')
        self.heads = nn.ModuleList()
        for _ in self.offsets:
            layers = []
            for _ in range(hidden_layers):
                layers.extend([nn.Linear(hidden_size, hidden_size), nn.GELU()])
            layers.append(nn.Linear(hidden_size, vocab_size))
            self.heads.append(nn.Sequential(*layers))


def neighbor_terms(heads, hidden, x0, xt, valid, target_eligible, mask_id,
                   boundary_ids, token_weights, denominator):
    """Sum CE at source i predicting target i+offset, using the main denominator.

    A source can be clean or masked. Both endpoints and every position between
    them must be valid non-boundaries. Target must be masked AND eligible.
    No ground-truth target is supplied as model input. Zero pairs is graph-safe.
    """
    content = valid.bool().clone()
    for token_id in boundary_ids:
        content &= x0.ne(token_id)
    target_mask = xt.eq(mask_id) & target_eligible.bool() & content
    length = x0.shape[1]
    terms, counts = {}, {}
    for offset, head in zip(heads.offsets, heads.heads):
        distance = abs(offset)
        if distance >= length:
            # Keep every head in the DDP graph, even in an empty-pair batch.
            terms[offset] = head(hidden[:, :1]).sum() * 0.0
            counts[offset] = x0.new_zeros(())
            continue
        source = slice(0, length-distance) if offset > 0 else slice(distance, length)
        target = slice(distance, length) if offset > 0 else slice(0, length-distance)
        pair = target_mask[:, target].clone()
        for j in range(distance+1):
            pair &= content[:, j:length-distance+j]
        logits = head(hidden[:, source]).float()
        logits[..., mask_id] = -1000000.0  # same excluded MASK class as MDM
        ce = F.cross_entropy(logits.transpose(1, 2), x0[:, target], reduction='none')
        weights = torch.broadcast_to(token_weights, x0.shape)[:, target]
        terms[offset] = (ce * weights * pair).sum() / denominator.clamp_min(1)
        counts[offset] = pair.sum()
    return terms, counts


def validate_mechanisms(config):
    for name in ('tt', 'ea', 'rm'):
        if config[name]['enabled']:
            raise NotImplementedError(f'{name.upper()} port is pending; refusing to silently ignore it')
    np_config = config['np']
    if np_config['enabled']:
        offsets, weights = np_config['offsets'], np_config['weights']
        if len(offsets) != len(weights) or not offsets:
            raise ValueError('NP requires one weight per offset')
        if any(not math.isfinite(float(w)) or float(w) < 0 for w in weights):
            raise ValueError('NP weights must be finite and nonnegative')
