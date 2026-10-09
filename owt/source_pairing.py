"""Adjacent masked-source NP and an exact per-row pair-count control."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

POLICIES = ('masked_source', 'matched_pair_count')
STAT_NAMES = ('eligible', 'masked_source', 'selected', 'selected_masked',
              'eligible_weight_mass', 'selected_weight_mass', 'masked_weight_mass')


def pairing_terms(heads, hidden, clean, noisy, valid, mask_id, boundary_ids,
                  weights, denominator, policy, generator=None, chunk_size=128,
                  ignore_first=True):
    """Preserve original target weighting/denominator; filter source eligibility.

    The control uniformly selects exactly k original pairs per row/direction,
    where k is that canvas's number of masked-source pairs. Row-constant noise
    weights make its selected weight mass exactly match the masked-source arm.
    A private CPU generator preserves the main corruption/dropout RNG streams.
    """
    if policy not in POLICIES or chunk_size < 1:
        raise ValueError('Unsupported source policy or invalid chunk size')
    if set(heads.offsets) != {-1, 1}:
        raise ValueError('This registered source experiment requires offsets -1,+1')
    if policy == 'matched_pair_count' and generator is None:
        raise ValueError('The pair-count control requires its private CPU generator')
    content = valid.bool().clone()
    for token in set(boundary_ids) | {mask_id}:
        content &= clean.ne(token)
    targets = content & noisy.eq(mask_id)
    if ignore_first:
        targets[:, 0] = False
    factors = torch.broadcast_to(weights, clean.shape)
    if policy == 'matched_pair_count' and not torch.equal(factors, factors[:, :1].expand_as(factors)):
        raise ValueError('Exact weight-mass control requires row-constant noise weights')
    rates = (noisy.eq(mask_id) & valid.bool()).sum(1) / valid.sum(1).clamp_min(1)
    bins = (rates * 5).long().clamp(max=4)
    terms, statistics = {}, {}
    for offset, head in zip(heads.offsets, heads.heads):
        source = slice(0, -1) if offset > 0 else slice(1, None)
        target = slice(1, None) if offset > 0 else slice(0, -1)
        eligible = targets[:, target] & content[:, source]
        masked = eligible & noisy[:, source].eq(mask_id)
        selected = masked.clone()
        if policy == 'matched_pair_count':
            selected.zero_()
            for row in range(clean.shape[0]):
                candidates = eligible[row].nonzero().flatten()
                k = int(masked[row].sum())
                if k:
                    chosen = torch.randperm(candidates.numel(), generator=generator)[:k]
                    selected[row, candidates[chosen.to(candidates.device)]] = True
        zero = hidden.sum() * 0 + sum(p.reshape(-1)[0] * 0 for p in head.parameters())
        h, y, w = hidden[:, source][selected], clean[:, target][selected], factors[:, target][selected]
        loss = zero

        def chunk_ce(features, labels, factors, projection=head):
            logits = projection(features)
            if logits.dtype in (torch.float16, torch.bfloat16):
                logits = logits.float()
            forbidden = torch.arange(logits.shape[-1], device=logits.device).eq(mask_id)
            logits = logits.masked_fill(forbidden, -torch.inf)
            return (F.cross_entropy(logits, labels, reduction='none') * factors).sum()

        for start in range(0, y.numel(), chunk_size):
            inputs = (h[start:start+chunk_size], y[start:start+chunk_size], w[start:start+chunk_size])
            if torch.is_grad_enabled():
                loss = loss + checkpoint(chunk_ce, *inputs, use_reentrant=False, preserve_rng_state=False)
            else:
                loss = loss + chunk_ce(*inputs)
        terms[offset] = loss / denominator.clamp_min(1)
        counts = torch.stack([eligible.sum(1), masked.sum(1), selected.sum(1),
                             (selected & noisy[:, source].eq(mask_id)).sum(1)], 1).double()
        masses = torch.stack([(factors[:, target] * mask).sum(1)
                              for mask in (eligible, selected, masked)], 1).double()
        row_stats = torch.cat((counts, masses), 1).detach()
        statistics[offset] = torch.stack([row_stats[bins.eq(i)].sum(0) for i in range(5)])
    return terms, statistics


def private_pair_generator(seed, calls, rank):
    return torch.Generator(device='cpu').manual_seed(int(seed) + int(calls)*1000003 + int(rank)*69069)
