"""A plus independent distance-two branches; existing A/B files stay immutable."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from owt.model import OWTMDM, Diffusion, Loss
from owt.transformer_np import NeighborTransformer, isolated_rng, process_branch
from owt.transformer_np_model import TransformerNPMDM
from models.ema import ExponentialMovingAverage

OFFSETS = (-1, 1, -2, 2)

def distance_neighbor_terms(heads, features, clean, noisy, valid, mask_id,
                               boundary_ids, weights, denominator, policy,
                               chunk_size=128, ignore_first=True):
    """Use native features for masked endpoints separated by one or two tokens.

    Clean labels enter only CE and eligibility, never transformer processing.
    Each offset records seven statistics in each of five mask-fraction bins.
    """
    if policy != 'masked_source' or chunk_size < 1:
        raise ValueError('Distance NP requires both-masked policy and a positive chunk size')
    if tuple(heads.offsets) != OFFSETS or clean.shape[1] < 3:
        raise ValueError('Distance NP requires ordered offsets -1,+1,-2,+2 and length >=3')
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
        distance, length = abs(offset), clean.shape[1]
        source = slice(0, length-distance) if offset > 0 else slice(distance, length)
        target = slice(distance, length) if offset > 0 else slice(0, length-distance)
        eligible = targets[:, target].clone()
        # Every intervening token must be valid content; no boundary crossing.
        # Intervening tokens may be revealed: only the two endpoints need masks.
        for shift in range(distance+1):
            eligible &= content[:, shift:length-distance+shift]
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


class DistanceTransformerNPMDM(TransformerNPMDM):
    def __init__(self, config, tokenizer):
        np_config = config.mechanisms.np
        if (not np_config.enabled or np_config.get('initialization') != 'zero'
                or tuple(np_config.offsets) != OFFSETS or np_config.hidden_layers != 0
                or len(np_config.weights) != 4):
            raise ValueError('Distance NP requires four independent zero projections ordered -1,+1,-2,+2')
        if (np_config.get('source_policy') != 'masked_source'
                or np_config.get('transformer_blocks') != 1
                or np_config.get('branch_point') != 'before_last_block'
                or np_config.get('shared_vocabulary_projection') is not False):
            raise ValueError('Distance NP must preserve A architecture and both-masked rule')
        if any(not math.isfinite(float(w)) or float(w) < 0 for w in np_config.weights):
            raise ValueError('Distance weights must be finite and nonnegative')
        if config.algo.cross_attn or config.algo.parameterization != 'subs':
            raise ValueError('Distance NP forbids clean-input cross-attention')
        OWTMDM.__init__(self, config, tokenizer)
        if self.backbone.causal or not self.backbone.adaLN or not self.backbone.blocks:
            raise ValueError('Distance NP requires bidirectional noise-conditioned attention')
        # Construct in A's exact order/seed, then append the two new blocks.
        # OWTMDM creates zero readouts privately too: shared RNG is unchanged.
        with isolated_rng(int(config.seed)+200003, 'cpu'):
            self.backbone.neighbor_branches = nn.ModuleList(
                [NeighborTransformer(config) for _ in OFFSETS])
        if self.ema:
            self.ema = ExponentialMovingAverage(self._get_parameters(), decay=config.training.ema)
        self.branch_calls = 0
        self._last_pair_statistics = None

    def resume_signature(self):
        result = super().resume_signature()
        result['architecture'] = 'independent_transformer_np_distance2_v1'
        result['content_span_exclusion'] = True
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
            terms, statistics = distance_neighbor_terms(
                self.backbone.neighbor_heads, features, x0, captured['noisy'],
                attention_mask, self.mask_index, self.boundary_ids, captured['weight'],
                attention_mask.sum(), self.np_config.source_policy,
                chunk_size=int(self.np_config.chunk_size), ignore_first=self.ignore_bos)
        self.branch_calls += 1
        self._last_pair_statistics = statistics
        total = float(self.config.objective.current_weight) * result.loss
        for offset, weight in zip(self.np_config.offsets, self.np_config.weights):
            total = total + float(weight) * terms[offset]
        self._last_components = dict(main_elbo=result.loss.detach(), objective=total.detach(),
                                    np_prev=terms[-1].detach(), np_next=terms[1].detach(),
                                    np_prev2=terms[-2].detach(), np_next2=terms[2].detach())
        return Loss(loss=total, nlls=result.nlls, token_mask=result.token_mask)
