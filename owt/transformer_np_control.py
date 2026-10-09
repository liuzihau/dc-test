"""Isolated B: native transformer branches with matched source-pair exposure."""
import copy

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from owt.model import Diffusion, Loss
from owt.source_pairing import POLICIES, private_pair_generator
from owt.transformer_np import process_branch
from owt.transformer_np_model import TransformerNPMDM

def matched_transformer_terms(heads, features, clean, noisy, valid, mask_id, boundary_ids,
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
        hidden = features[offset]
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



class MatchedTransformerNPMDM(TransformerNPMDM):
    def __init__(self, config, tokenizer):
        if (config.mechanisms.np.get('source_policy') != 'matched_pair_count'
                or config.mechanisms.np.get('pair_selection_seed') != 271828):
            raise ValueError('B requires matched_pair_count and the registered private seed271828')
        # Parent validates A policies. Restore the B policy before any loss,
        # metadata, signature or checkpoint is produced; parameter RNG is identical.
        local = copy.deepcopy(config)
        local.mechanisms.np.source_policy = 'masked_source'
        super().__init__(local, tokenizer)
        self.config.mechanisms.np.source_policy = 'matched_pair_count'
        self.np_config.source_policy = 'matched_pair_count'

    def resume_signature(self):
        result = super().resume_signature()
        result['pair_selection_seed'] = int(self.np_config.pair_selection_seed)
        return result

    def _loss(self, x0, attention_mask, **kwargs):
        if not self.training:
            # Normal validation never executes auxiliary transformer blocks.
            return Diffusion._loss(self, x0, attention_mask, **kwargs)
        if x0.shape[1] != self.num_tokens:
            raise ValueError('Transformer NP requires packed model-length sequences')
        captured = {}

        def capture_branch_input(module, inputs, keywords):
            if 'hidden' in captured:
                raise RuntimeError('Expected exactly one corrupted-canvas backbone forward')
            captured.update(hidden=inputs[0], rotary=inputs[1],
                            condition=keywords.get('c'), mask=keywords.get('mask'))

        handles = [
            self.noise.register_forward_hook(lambda module, inputs, output: captured.update(weight=-output[0])),
            self.backbone.register_forward_pre_hook(lambda module, inputs: captured.update(noisy=inputs[0])),
            self.backbone.blocks[-1].register_forward_pre_hook(capture_branch_input, with_kwargs=True),
        ]
        try:
            result = Diffusion._loss(self, x0, attention_mask, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        features = {}
        with torch.autocast(device_type=x0.device.type, dtype=torch.bfloat16, enabled=x0.is_cuda):
            for index, (offset, branch) in enumerate(zip(self.np_config.offsets, self.backbone.neighbor_branches)):
                seed = (int(self.np_config.branch_dropout_seed) + self.branch_calls * 1000003
                        + int(self.global_rank) * 69069 + index * 104729)
                features[offset] = process_branch(branch, captured['hidden'], captured['rotary'],
                    captured['condition'], seed, mask=captured['mask'],
                    use_checkpoint=bool(self.np_config.branch_checkpoint))
            terms, statistics = matched_transformer_terms(
                self.backbone.neighbor_heads, features, x0, captured['noisy'],
                attention_mask, self.mask_index, self.boundary_ids, captured['weight'],
                attention_mask.sum(), self.np_config.source_policy,
                generator=private_pair_generator(self.np_config.pair_selection_seed, self.branch_calls, self.global_rank),
                chunk_size=int(self.np_config.chunk_size), ignore_first=self.ignore_bos)
        self.branch_calls += 1
        self._last_pair_statistics = statistics
        total = float(self.config.objective.current_weight) * result.loss
        for offset, weight in zip(self.np_config.offsets, self.np_config.weights):
            total = total + float(weight) * terms[offset]
        self._last_components = dict(main_elbo=result.loss.detach(), objective=total.detach(),
                                    np_prev=terms[-1].detach(), np_next=terms[1].detach())
        return Loss(loss=total, nlls=result.nlls, token_mask=result.token_mask)
