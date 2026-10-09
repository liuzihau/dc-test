"""Small extension of author DiffLM: canonical training positions and optional NP.

    The upstream generation routine is retained, including its bookkeeping
    permutation with original position IDs. Auxiliary heads are training-only.
"""
import math
import time
import torch

from difflm import DiffLM
from models.ema import ExponentialMovingAverage
from trainer_base import Loss

from zebra.neighbor import NeighborHeads, neighbor_terms, validate_mechanisms


class ZebraMDM(DiffLM):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        validate_mechanisms(config.mechanisms)
        if config.algo.diffusion_attn_mode != 'full' or config.algo.ar_noise:
            raise ValueError('Canonical reasoning adapter requires full-attention MDM')
        if config.algo.diffusion_shuffle or config.algo.shifted_logits:
            raise ValueError('Canonical reasoning adapter requires unshuffled, unshifted inputs')
        self.objective = config.objective
        if self.objective.kind not in ('elbo', 'masked_ce'):
            raise ValueError('objective.kind must be elbo or masked_ce')
        if not math.isfinite(float(self.objective.current_weight)) or self.objective.current_weight < 0:
            raise ValueError('objective.current_weight must be finite and nonnegative')
        self.np_config = config.mechanisms.np
        if self.np_config.enabled:
            # Identical shared backbone initialization AND subsequent RNG stream.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(config.seed) + 100003)
                self.backbone.neighbor_heads = NeighborHeads(
                    config.model.hidden_size, self.vocab_size,
                    self.np_config.offsets, self.np_config.hidden_layers)
            # Parent initialized EMA before the optional heads existed.
            if config.training.ema > 0:
                self.ema = ExponentialMovingAverage(
                    self._get_parameters(), decay=config.training.ema)
        self.boundary_ids = sorted(set(tokenizer.all_special_ids) | {
            tokenizer.convert_tokens_to_ids(x) for x in self.np_config.boundary_tokens})

    def _sort_indices(self, indices, **kwargs):
        # Flags alone still group clean/masked tokens in author DiffLM.
        return torch.arange(indices.shape[1], device=indices.device).expand_as(indices)

    def _get_shuffle_settings(self):
        # The author's sampler packs selected positions internally while carrying
        # original position IDs; training and loss validation stay canonical.
        if getattr(self, '_author_generation', False):
            return True, True
        return False, False

    def generate_completions(self, *args, **kwargs):
        layout = self.config.get(
            'reasoning_generation_layout',
            self.config.get('zebra_generation_layout', 'author'))
        if layout == 'canonical':
            return self._generate_canonical_completions(*args, **kwargs)
        self._author_generation = True
        try:
            return super().generate_completions(*args, **kwargs)
        finally:
            self._author_generation = False

    @torch.no_grad()
    def _generate_canonical_completions(self, completion_batch, num_steps=None,
                                        return_stats=False):
        """Top-prob decoding without ever permuting physical token slots."""
        num_steps, kv_cache, trim_masked_tokens, unmask_policy = \
            self._validate_sampling_configs(num_steps)
        if self.config.algo.diffusion_attn_mode != "full":
            raise ValueError("Canonical generation currently requires full attention")
        if kv_cache or trim_masked_tokens:
            raise ValueError("Canonical generation requires untrimmed, cache-free forwards")
        if unmask_policy != "topp":
            raise ValueError("Canonical generation currently implements top-probability order")
        if int(self.config.sampling.get("n_latent_tokens", 0)) != 0:
            raise ValueError("Canonical generation does not support latent-token reordering")
        if int(self.config.sampling.get("topk_candidate_min", 0)) != 0 or \
                int(self.config.sampling.get("topk_candidate_max", 0)) != 0:
            raise ValueError("Canonical generation requires unrestricted candidates")

        input_ids = completion_batch["input_ids"].to(self.device)
        is_solution = completion_batch["loss_mask"].to(self.device).bool()
        batch, length = input_ids.shape
        if length != self.num_tokens:
            raise ValueError(f"Expected sequence length {self.num_tokens}, got {length}")
        x = torch.where(is_solution, self.mask_index, input_ids)
        identity = torch.arange(length, device=self.device).expand(batch, -1)
        processed = torch.zeros_like(is_solution)
        schedule = self._tokens_unmasked_per_step(num_steps)
        schedule += [1] * (self.num_tokens - sum(schedule))
        if sum(schedule) != self.num_tokens:
            raise RuntimeError("Reveal schedule must process every physical position once")

        noise_scale = float(self.config.sampling.get("noise_scale", 1.0))
        greedy_tokens = bool(self.config.sampling.get("greedy_tokens", False))
        start = time.perf_counter()
        self.backbone.reset_kv_cache()
        processed_count = 0
        for k in schedule:
            log_p_x0 = self.backbone.forward_sample(
                zt=x, sort_idx=identity, attn_mode="full",
                cutoffs=processed_count, kv_cache=False,
                last_k_start=max(0, processed_count - k),
                curr_k_start=processed_count,
                curr_k_end=self.num_tokens, mask_cutoffs=None)
            if self.config.sampling.use_float64:
                log_p_x0 = log_p_x0.to(torch.float64)
            log_p_x0[:, :, self.mask_index] = self.neg_infinity

            confidence = self._compute_confidence_scores(log_p_x0, "topp")
            confidence = confidence.masked_fill(processed, self.neg_infinity)
            # Match author timing: consume fixed problem positions before
            # decoding solutions, but never move or modify those positions.
            undecoded_problem = (~is_solution) & (~processed)
            confidence = torch.where(
                undecoded_problem,
                confidence.new_full((), -self.neg_infinity), confidence)
            selected = confidence.topk(k, dim=-1).indices
            selected_logits = torch.gather(
                log_p_x0, 1,
                selected[:, :, None].expand(-1, -1, log_p_x0.shape[-1]))
            if greedy_tokens:
                values = selected_logits.argmax(-1)
            else:
                noise = torch.distributions.Gumbel(0, 1).sample(
                    selected_logits.shape).to(self.device)
                values = (selected_logits + noise * noise_scale).argmax(-1)
            selected_solution = torch.gather(is_solution, 1, selected)
            original_values = torch.gather(input_ids, 1, selected)
            values = torch.where(selected_solution, values, original_values)
            x.scatter_(1, selected, values)
            processed.scatter_(1, selected, True)
            processed_count += k

        self.backbone.reset_kv_cache()
        duration = time.perf_counter() - start
        print(f"Canonical sampling duration: {duration} seconds")
        if return_stats:
            return x, {"duration": duration}
        return x

    def on_train_epoch_end(self):
        # Upstream validation resets its shared metric collection before this
        # hook; computing epoch train NLL then yields empty-state NaNs. The
        # per-update component CSV is the authoritative training trace here.
        pass

    def on_load_checkpoint(self, checkpoint):
        from zebra.batch_policy import migrate_batch_counters
        if self.config.mode == 'train':
            migrate_batch_counters(checkpoint, self.config)
        super().on_load_checkpoint(checkpoint)

    def _loss(self, x0, valid_tokens, current_accumulation_step=None,
              train_mode=False, loss_mask=None):
        if not train_mode:
            # Common author ELBO validation, independent of auxiliary objective.
            return super()._loss(x0, valid_tokens, current_accumulation_step,
                                 train_mode=False, loss_mask=loss_mask)
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        if self.T != 0:
            raise ValueError('This baseline uses continuous-time corruption')
        dalpha, alpha = self.noise(t)
        alpha = alpha.unsqueeze(-1)
        xt = self.q_xt(x0, alpha, loss_mask=loss_mask)
        sigma = self._sigma_from_alphat(alpha)
        positions = self._sort_indices(xt)
        hidden = []
        handle = None
        if self.np_config.enabled:
            # Exact feature consumed by the normal LM projection, after its norm.
            handle = self.backbone.output_layer.linear.register_forward_pre_hook(
                lambda module, inputs: hidden.append(inputs[0]))
        try:
            log_probs = self.forward(xt, sigma=sigma, sort_idx=positions)
        finally:
            if handle is not None:
                handle.remove()
        effective = valid_tokens if loss_mask is None else valid_tokens * loss_mask
        denominator = effective.sum()
        per_token = self.nll_per_token(log_probs, xt, x0, alpha, dalpha,
                                      low_var=False, train_mode=True)
        current_nll = (per_token * effective).sum() / denominator.clamp_min(1)
        if self.objective.kind == 'elbo':
            current = current_nll
            token_weights = -dalpha / (1-alpha)
            aux_denominator = denominator
        else:
            masked = effective * xt.eq(self.mask_index)
            aux_denominator = masked.sum()
            ce = -log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            current = (ce * masked).sum() / aux_denominator.clamp_min(1)
            token_weights = torch.ones_like(alpha)
        total = self.objective.current_weight * current
        logs = {'current': current, 'current_elbo': current_nll}
        if self.np_config.enabled:
            with torch.autocast(device_type=x0.device.type, dtype=torch.bfloat16,
                                enabled=x0.is_cuda):
                terms, counts = neighbor_terms(
                    self.backbone.neighbor_heads, hidden.pop(), x0, xt, valid_tokens,
                    effective, self.mask_index, self.boundary_ids, token_weights,
                    aux_denominator)
            for offset, weight in zip(self.np_config.offsets, self.np_config.weights):
                total = total + float(weight) * terms[offset]
                label = f'np_{"prev" if offset < 0 else "next"}_{abs(offset)}'
                logs[label] = terms[offset]
                logs[label + '_pairs'] = counts[offset].float()
        logs['objective'] = total
        if self._trainer is not None:
            for key, value in logs.items():
                self.log('components/' + key, value.detach(), on_step=True,
                         on_epoch=False, sync_dist=True)
        # Report main-head ELBO as train NLL, not main+NP total.
        return Loss(total, current_nll * denominator,
                    x0.new_zeros((), dtype=torch.float32), denominator)
