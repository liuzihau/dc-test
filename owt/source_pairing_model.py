"""Source-policy experiment, isolated from the pinned baseline model files."""
import torch
from owt.model import OWTMDM, Diffusion, Loss
from owt.source_pairing import POLICIES, pairing_terms, private_pair_generator


class SourcePairingMDM(OWTMDM):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        if self.np_config.get('source_policy') not in POLICIES:
            raise ValueError('A registered source policy is required')
        if not self.np_config.enabled or self.np_config.initialization != 'zero':
            raise ValueError('Source experiments require zero-initialized NP')
        self.pair_calls = 0
        self._last_pair_statistics = None

    def _loss(self, x0, attention_mask, **kwargs):
        if not self.training:
            return super()._loss(x0, attention_mask, **kwargs)
        if x0.shape[1] != self.num_tokens:
            raise ValueError('Source NP requires packed model-length sequences')
        captured = {}
        handles = [
            self.noise.register_forward_hook(lambda module, inputs, output: captured.update(weight=-output[0])),
            self.backbone.register_forward_pre_hook(lambda module, inputs: captured.update(noisy=inputs[0])),
            self.backbone.output_layer.linear.register_forward_pre_hook(lambda module, inputs: captured.update(hidden=inputs[0])),
        ]
        try:
            result = Diffusion._loss(self, x0, attention_mask, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        generator = private_pair_generator(self.np_config.pair_selection_seed, self.pair_calls, self.global_rank)
        with torch.autocast(device_type=x0.device.type, dtype=torch.bfloat16, enabled=x0.is_cuda):
            terms, statistics = pairing_terms(
                self.backbone.neighbor_heads, captured['hidden'], x0, captured['noisy'],
                attention_mask, self.mask_index, self.boundary_ids, captured['weight'],
                attention_mask.sum(), self.np_config.source_policy, generator,
                chunk_size=int(self.np_config.chunk_size), ignore_first=self.ignore_bos)
        self.pair_calls += 1
        self._last_pair_statistics = statistics
        total = float(self.config.objective.current_weight) * result.loss
        for offset, weight in zip(self.np_config.offsets, self.np_config.weights):
            total = total + float(weight) * terms[offset]
        self._last_components = dict(main_elbo=result.loss.detach(), objective=total.detach(),
                                    np_prev=terms[-1].detach(), np_next=terms[1].detach())
        return Loss(loss=total, nlls=result.nlls, token_mask=result.token_mask)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint['source_pairing'] = dict(calls=self.pair_calls, policy=self.np_config.source_policy,
                                            seed=int(self.np_config.pair_selection_seed))

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint.get('source_pairing')
        if (not state or state['policy'] != self.np_config.source_policy
                or state['seed'] != int(self.np_config.pair_selection_seed)):
            raise ValueError('Resume requires identical source policy/private sampling seed')
        super().on_load_checkpoint(checkpoint)
        self.pair_calls = int(state['calls'])
