"""Isolated transformer-NP model; existing live trainer files stay unchanged."""
import math

import torch
from torch import nn

from owt.model import OWTMDM, Diffusion, Loss
from models.ema import ExponentialMovingAverage
from owt.transformer_np import (NeighborTransformer, SOURCE_POLICIES,
                                isolated_rng, process_branch, transformer_neighbor_terms)


class TransformerNPMDM(OWTMDM):
    def __init__(self, config, tokenizer):
        np_config = config.mechanisms.np
        if (not np_config.enabled or np_config.get('initialization') != 'zero'
                or tuple(np_config.offsets) != (-1, 1) or np_config.hidden_layers != 0):
            raise ValueError('Transformer NP requires zero independent projections for -1,+1')
        if (np_config.get('source_policy') not in SOURCE_POLICIES
                or np_config.get('transformer_blocks') != 1
                or np_config.get('branch_point') != 'before_last_block'
                or np_config.get('shared_vocabulary_projection') is not False):
            raise ValueError('Unsupported transformer NP architecture or source policy')
        if any(not math.isfinite(float(w)) or float(w) <= 0 for w in np_config.weights):
            raise ValueError('Each transformer NP direction needs a positive finite weight')
        if config.algo.cross_attn or config.algo.parameterization != 'subs':
            raise ValueError('Transformer NP requires full-sequence masked diffusion without clean-input cross-attention')
        super().__init__(config, tokenizer)
        if self.backbone.causal or not self.backbone.adaLN or not self.backbone.blocks:
            raise ValueError('Transformer NP requires a bidirectional noise-conditioned backbone')
        with isolated_rng(int(config.seed) + 200003, 'cpu'):
            self.backbone.neighbor_branches = nn.ModuleList(
                [NeighborTransformer(config) for _ in self.np_config.offsets])
        if self.ema:
            self.ema = ExponentialMovingAverage(self._get_parameters(), decay=config.training.ema)
        self.branch_calls = 0
        self._last_pair_statistics = None

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
            terms, statistics = transformer_neighbor_terms(
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
                                    np_prev=terms[-1].detach(), np_next=terms[1].detach())
        return Loss(loss=total, nlls=result.nlls, token_mask=result.token_mask)

    def resume_signature(self):
        return dict(architecture='independent_transformer_np_v1',
                    source_policy=self.np_config.source_policy,
                    offsets=list(self.np_config.offsets), weights=list(self.np_config.weights),
                    initialization=self.np_config.initialization,
                    dropout_seed=int(self.np_config.branch_dropout_seed),
                    branch_checkpoint=bool(self.np_config.branch_checkpoint),
                    branch_point=self.np_config.branch_point,
                    main_weight=float(self.config.objective.current_weight),
                    model=dict(hidden_size=self.config.model.hidden_size,
                               cond_dim=self.config.model.cond_dim,
                               n_blocks=self.config.model.n_blocks,
                               n_heads=self.config.model.n_heads,
                               length=self.config.model.length,
                               dropout=self.config.model.dropout))

    @torch.no_grad()
    def diagnostic_forward(self, noisy, sigma):
        """Main scores plus full-sequence native auxiliary features, no labels.

        Normal validation/inference stays main-only. Future collectors can
        apply independent projections in chunks to these branch features.
        """
        if self.training:
            raise ValueError('Diagnostic forward requires eval mode')
        captured = {}

        def capture(module, inputs, keywords):
            captured.update(hidden=inputs[0], rotary=inputs[1],
                            condition=keywords.get('c'), mask=keywords.get('mask'))

        handle = self.backbone.blocks[-1].register_forward_pre_hook(capture, with_kwargs=True)
        try:
            main = self.forward(noisy, sigma=sigma)
        finally:
            handle.remove()
        force_fp32 = getattr(self.backbone, 'force_fp32_eval', False)
        with torch.autocast(device_type=noisy.device.type, dtype=torch.bfloat16,
                            enabled=noisy.is_cuda and not force_fp32):
            features = {offset: process_branch(branch, captured['hidden'], captured['rotary'],
                captured['condition'], seed=0, mask=captured['mask'], use_checkpoint=False)
                for offset, branch in zip(self.np_config.offsets, self.backbone.neighbor_branches)}
        return main, features

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint['transformer_np'] = dict(calls=self.branch_calls, signature=self.resume_signature())

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint.get('transformer_np', {})
        if state.get('signature') != self.resume_signature() or not isinstance(state.get('calls'), int) or state['calls'] < 0:
            raise ValueError('Resume requires identical transformer NP architecture, loss and private RNG policy')
        super().on_load_checkpoint(checkpoint)
        self.branch_calls = state['calls']
