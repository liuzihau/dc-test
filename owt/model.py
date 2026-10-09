"""Original BD3 MDM loss with an optional, independent neighbor objective."""
from owt.runtime import install
install()

import torch
from torch.utils.data import DataLoader, DistributedSampler
from diffusion import Diffusion, Loss
from models.ema import ExponentialMovingAverage
import dataloader

from owt.neighbor import NeighborHeads, initialize_heads, neighbor_terms, validate_mechanisms


class OWTMDM(Diffusion):
    def __init__(self, config, tokenizer):
        validate_mechanisms(config.mechanisms)
        if config.algo.name != 'mdlm' or config.block_size != config.model.length:
            raise ValueError('OWT pilot requires full-sequence BD3 MDLM pretraining')
        if config.objective.kind != 'elbo':
            raise ValueError('This aligned pilot requires the original BD3 ELBO')
        super().__init__(config, tokenizer)
        self.np_config = config.mechanisms.np
        if self.np_config.enabled:
            # Preserve shared-backbone initialization AND the next RNG draw.
            with torch.random.fork_rng(devices=[]):
                # Heads are constructed on CPU. torch.manual_seed would also
                # reseed CUDA, which a CPU-only fork_rng does not restore.
                torch.random.default_generator.manual_seed(int(config.seed) + 100003)
                self.backbone.neighbor_heads = NeighborHeads(
                    config.model.hidden_size, self.vocab_size,
                    self.np_config.offsets, self.np_config.hidden_layers)
                initialize_heads(self.backbone.neighbor_heads,
                                 self.np_config.get('initialization', 'random'))
            if self.ema:
                self.ema = ExponentialMovingAverage(self._get_parameters(), decay=config.training.ema)
        self.boundary_ids = tokenizer.all_special_ids
        self._last_components = None

    def _loss(self, x0, attention_mask, **kwargs):
        if not self.training or not self.np_config.enabled:
            result = super()._loss(x0, attention_mask, **kwargs)
            if self.training:
                total = result.loss * float(self.config.objective.current_weight)
                self._last_components = dict(main_elbo=result.loss.detach(), objective=total.detach(),
                                            np_prev=result.loss.detach()*0, np_next=result.loss.detach()*0)
                return Loss(loss=total, nlls=result.nlls, token_mask=result.token_mask)
            return result
        if x0.shape[1] != self.num_tokens:
            raise ValueError('NP expects already packed model-length sequences')
        captured = {}
        handles = [
            self.noise.register_forward_hook(lambda module, inputs, output: captured.update(weight=-output[0])),
            self.backbone.register_forward_pre_hook(lambda module, inputs: captured.update(noisy=inputs[0])),
            self.backbone.output_layer.linear.register_forward_pre_hook(
                lambda module, inputs: captured.update(hidden=inputs[0])),
        ]
        try:
            result = super()._loss(x0, attention_mask, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        with torch.autocast(device_type=x0.device.type, dtype=torch.bfloat16, enabled=x0.is_cuda):
            terms, _ = neighbor_terms(
                self.backbone.neighbor_heads, captured['hidden'], x0, captured['noisy'],
                attention_mask, self.mask_index, self.boundary_ids, captured['weight'],
                attention_mask.sum(), chunk_size=int(self.np_config.chunk_size),
                ignore_first=self.ignore_bos)
        total = float(self.config.objective.current_weight) * result.loss
        for offset, weight in zip(self.np_config.offsets, self.np_config.weights):
            total = total + float(weight) * terms[offset]
        self._last_components = dict(main_elbo=result.loss.detach(), objective=total.detach(),
                                    np_prev=terms[-1].detach(), np_next=terms[1].detach())
        return Loss(loss=total, nlls=result.nlls, token_mask=result.token_mask)

    def training_step(self, batch, batch_idx):
        result = self._loss(batch['input_ids'], batch['attention_mask'])
        self.metrics.train_nlls.update(result.nlls.detach(), result.token_mask)
        # The local callback averages every microbatch and synchronizes once per
        # optimizer update. Components are never inferred from scaled outputs.
        return result.loss

    def on_validation_model_zero_grad(self):
        # Original BD3 toggles sanity_checking to skip a resumed validation.
        # Every scheduled 500-update validation is real in this experiment.
        self.zero_grad()

    def on_load_checkpoint(self, checkpoint):
        saved = checkpoint['hyper_parameters']['config']
        for section, key in (('loader', 'batch_size'), ('loader', 'global_batch_size'),
                             ('trainer', 'devices'), ('trainer', 'accumulate_grad_batches')):
            if saved[section][key] != self.config[section][key]:
                raise ValueError(f'Resume requires identical {section}.{key}')
        super().on_load_checkpoint(checkpoint)

    def on_train_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        loaders = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            old = dl.sampler
            if not isinstance(old, DistributedSampler):
                raise ValueError('OWT launcher requires explicit DDP')
            sampler = dataloader.FaultTolerantDistributedSampler(
                dl.dataset, num_replicas=old.num_replicas, rank=old.rank,
                shuffle=old.shuffle, seed=old.seed, drop_last=old.drop_last)
            sampler.set_epoch(self.trainer.current_epoch)
            cursor = 0
            if self.fast_forward_batches is not None:
                cursor = self.fast_forward_batches * dl.batch_size
                if cursor >= sampler.num_samples:
                    raise ValueError('This pilot resume supports in-epoch optimizer boundaries only')
                sampler.load_state_dict(dict(epoch=self.fast_forward_epochs, counter=cursor))
            print(f'OWT data rank {old.rank}: sampler seed={old.seed}, '
                  f'epoch={sampler.epoch}, completed rows on this rank={cursor}', flush=True)
            loaders.append(DataLoader(
                dl.dataset, batch_size=dl.batch_size, sampler=sampler,
                num_workers=dl.num_workers, pin_memory=dl.pin_memory,
                persistent_workers=dl.num_workers > 0, collate_fn=dl.collate_fn,
                drop_last=dl.drop_last, worker_init_fn=dl.worker_init_fn,
                generator=dl.generator))
        self.trainer.fit_loop._combined_loader.flattened = loaders
        # Lightning 2.5 already built a live iterator before this hook.
        fetcher = self.trainer.fit_loop._data_fetcher
        fetcher.teardown()
        iter(fetcher)
