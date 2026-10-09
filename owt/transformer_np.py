"""Task-specific full-sequence transformer processing and bounded NP losses."""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from owt.runtime import install
install()
from models.dit import DDiTBlock, LayerNorm, modulate_fused

SOURCE_POLICIES = ('target_only', 'masked_source')


@contextmanager
def isolated_rng(seed, device):
    """Seed only CPU and the tensor's CUDA device; restore both afterwards."""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(int(seed))
        if devices:
            torch.cuda.default_generators[devices[0]].manual_seed(int(seed))
        yield


class NeighborTransformer(nn.Module):
    """One bidirectional BD3 block plus its own noise-conditioned final norm."""
    def __init__(self, config):
        super().__init__()
        dim, cond = config.model.hidden_size, config.model.cond_dim
        self.block = DDiTBlock(n=config.model.length, dim=dim,
            n_heads=config.model.n_heads, cond_dim=cond, adaLN=True,
            dropout=config.model.dropout, block_size=config.block_size,
            attn_backend=config.model.attn_backend, max_seqlen=config.model.length)
        self.norm = LayerNorm(dim)
        self.modulation = nn.Linear(cond, 2 * dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, hidden, cos, sin, condition, mask=None):
        hidden = self.block(hidden, (cos, sin), c=condition, causal=False, mask=mask)
        hidden = self.norm(hidden)
        if condition is not None:
            if condition.shape[0] != hidden.shape[0]:
                raise ValueError('Full-sequence NP requires one noise condition per row')
            shift, scale = self.modulation(condition)[:, None].chunk(2, dim=2)
            hidden = modulate_fused(hidden, shift, scale)
        return hidden


def process_branch(branch, hidden, rotary, condition, seed, mask=None,
                   use_checkpoint=True):
    """Replay private dropout identically during checkpoint recomputation."""
    def compute(x, cos, sin, c):
        with isolated_rng(seed, x.device):
            return branch(x, cos, sin, c, mask=mask)

    if torch.is_grad_enabled() and use_checkpoint:
        # Upstream TorchScript fused functions wrap checkpoint's internal
        # early-stop exception in RuntimeError. Complete recomputation instead.
        with set_checkpoint_early_stop(False):
            return checkpoint(compute, hidden, rotary[0], rotary[1], condition,
                              use_reentrant=False, preserve_rng_state=False)
    return compute(hidden, rotary[0], rotary[1], condition)


def transformer_neighbor_terms(heads, features, clean, noisy, valid, mask_id,
                               boundary_ids, weights, denominator, policy,
                               chunk_size=128, ignore_first=True):
    """Use full-sequence branch features, then select adjacent prediction pairs.

    Clean labels enter only CE and eligibility, never transformer processing.
    Statistics have the existing source_pairs.csv schema (5 mask bins x 7).
    """
    if policy not in SOURCE_POLICIES or chunk_size < 1:
        raise ValueError('Invalid transformer NP source policy or chunk size')
    if tuple(heads.offsets) != (-1, 1) or clean.shape[1] < 2:
        raise ValueError('Transformer NP requires adjacent offsets -1,+1 and length >=2')
    content = valid.bool().clone()
    for token in set(boundary_ids) | {mask_id}:
        content &= clean.ne(token)
    targets = content & noisy.eq(mask_id)
    if ignore_first:
        targets[:, 0] = False
    factors = torch.broadcast_to(weights, clean.shape)
    rates = (noisy.eq(mask_id) & valid.bool()).sum(1) / valid.sum(1).clamp_min(1)
    bins = (rates * 5).long().clamp(max=4)
    terms, statistics = {}, {}
    for offset, head in zip(heads.offsets, heads.heads):
        hidden = features[offset]
        source = slice(0, -1) if offset > 0 else slice(1, None)
        target = slice(1, None) if offset > 0 else slice(0, -1)
        eligible = targets[:, target] & content[:, source]
        masked = eligible & noisy[:, source].eq(mask_id)
        selected = masked if policy == 'masked_source' else eligible
        # Includes every processing/projection parameter in empty-pair DDP graphs.
        loss = hidden.sum() * 0 + sum(p.reshape(-1)[0] * 0 for p in head.parameters())
        h = hidden[:, source][selected]
        y, w = clean[:, target][selected], factors[:, target][selected]

        def chunk_ce(x, labels, scale, projection=head):
            logits = projection(x)
            if logits.dtype in (torch.float16, torch.bfloat16):
                logits = logits.float()
            logits = logits.masked_fill(torch.arange(logits.shape[-1], device=logits.device).eq(mask_id), -torch.inf)
            return (F.cross_entropy(logits, labels, reduction='none') * scale).sum()

        for start in range(0, y.numel(), chunk_size):
            inputs = (h[start:start+chunk_size], y[start:start+chunk_size], w[start:start+chunk_size])
            if torch.is_grad_enabled():
                loss = loss + checkpoint(chunk_ce, *inputs, use_reentrant=False, preserve_rng_state=False)
            else:
                loss = loss + chunk_ce(*inputs)
        terms[offset] = loss / denominator.clamp_min(1)
        counts = torch.stack([eligible.sum(1), masked.sum(1), selected.sum(1),
                              (selected & noisy[:, source].eq(mask_id)).sum(1)], 1).double()
        masses = torch.stack([(factors[:, target] * m).sum(1)
                              for m in (eligible, selected, masked)], 1).double()
        rows = torch.cat((counts, masses), 1).detach()
        statistics[offset] = torch.stack([rows[bins.eq(i)].sum(0) for i in range(5)])
    return terms, statistics
