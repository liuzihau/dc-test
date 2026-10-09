import time

import numpy as np
import torch
import torch.nn.functional as F

import trainer_base
import utils
from mdlm import MDLM


class DiffLM(MDLM):
  """
    Based on EsoLM
  """
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    # Initialize pad_index similar to mask_index
    if (not hasattr(tokenizer, 'pad_token')
        or tokenizer.pad_token is None):
      # If pad_token doesn't exist, use a default value
      # Note: pad_token_id might still exist even if pad_token is None
      if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        self.pad_index = tokenizer.pad_token_id
      else:
        # Fallback: use eos_token_id if available, otherwise 0
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
          self.pad_index = tokenizer.eos_token_id
        else:
          self.pad_index = 0
    else:
      self.pad_index = tokenizer.pad_token_id

    print(f"pad_index: {self.pad_index}")
    
    self.alpha_0 = config.algo.alpha_0
    self.noise = trainer_base.LogLinear(self.alpha_0)

    self.ar_noise = config.algo.get('ar_noise', False)
    self.next_token_prediction = config.algo.get('next_token_prediction', False)
    self.mtp_window_size = config.algo.get('mtp_window_size', -1)
    self.mtp_mode = config.algo.get('mtp_mode', 'contiguous')
    
    # New shuffle controls (None = use legacy behavior based on ar_noise)
    self.shuffle_clean_tokens = config.algo.get('shuffle_clean_tokens', None)
    self.shuffle_masked_tokens = config.algo.get('shuffle_masked_tokens', None)

    # If True, exclude padding tokens when selecting masking threshold in ar_noise mode
    self.exclude_padding_from_mask_threshold = config.algo.get(
      'exclude_padding_from_mask_threshold', True)

  def _get_shuffle_settings(self):
    """Get effective shuffle settings, with backward compatibility.
    
    Returns:
      (shuffle_clean, shuffle_masked): tuple of booleans
      
    Legacy behavior (when shuffle_clean_tokens and shuffle_masked_tokens are None):
      - ar_noise=False: shuffle both (True, True)
      - ar_noise=True: shuffle neither (False, False)
    """
    # Get base shuffle from diffusion_shuffle config (defaults to True)
    base_shuffle = self.config.algo.get('diffusion_shuffle', True)
    
    # For backward compatibility: ar_noise forces shuffle to False
    # (only when new configs are not explicitly set)
    if self.ar_noise and self.shuffle_clean_tokens is None and self.shuffle_masked_tokens is None:
      base_shuffle = False
    
    # Get explicit settings or fall back to base
    shuffle_clean = self.shuffle_clean_tokens if self.shuffle_clean_tokens is not None else base_shuffle
    shuffle_masked = self.shuffle_masked_tokens if self.shuffle_masked_tokens is not None else base_shuffle
    
    return shuffle_clean, shuffle_masked

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False, train_mode=False):

    """
    Compute the loss per token for the DiffLM model.

    ar_noise: If True, mask rightmost tokens instead of random tokens.
      We uniformly sample an integer in [1, num_tokens] and mask that many tokens.
      This forces NTP loss for VALIDATION, but we may use MTP for training.
    next_token_prediction: If True, we only compute loss on the first masked token.
    mtp_window_size: If > 0, only compute loss on first mtp_window_size masked tokens and restrict attention to window
    mtp_mode: If 'random', shuffle the masked tokens (except the first one)
    low_var: whether to use the low-variance loss. Only applicable for diffusion; Only relevent for training.
      Validation always uses ELBO loss.
    """
    log_p_theta = log_x_theta.gather(
      dim=-1,
      index=x0[:, :, None])[:, :, 0]
    
    # Determine effective window size for loss masking
    # - next_token_prediction=True overrides mtp_window_size to 1
    # - mtp_window_size > 0 uses that window size
    # - otherwise, use all masked positions (window_size = -1)
    # - When masks are shuffled, window concept doesn't apply (use all masked positions)
    _, shuffle_masked = self._get_shuffle_settings()
    
    if not train_mode:
      # for validation loss, we use configs that reflect the decoding process
      # - When masks are shuffled, the window concept doesn't apply
      # - for ar_noise without shuffled masks, we always generate with ntp = True
      if shuffle_masked:
        loss_window_size = -1
      elif self.ar_noise:
        loss_window_size = 1
      else:
        loss_window_size = -1
    else:
      # During training: 
      # - When masks are shuffled with ar_noise, use all masked positions (window doesn't make sense)
      # - next_token_prediction overrides mtp_window_size
      if self.ar_noise and shuffle_masked:
        loss_window_size = -1
      elif self.next_token_prediction:
        loss_window_size = 1
      elif self.mtp_window_size > 0:
        loss_window_size = self.mtp_window_size
      else:
        loss_window_size = -1

    # carry-over unmasking
    masked_positions = (xt == self.mask_index)  # [batch_size, seq_len]
    has_masked = masked_positions.any(dim=1)  # [batch_size]
    first_masked_idx = masked_positions.long().argmax(dim=1)  # [batch_size]
    
    if loss_window_size > 0:
      # Only compute loss on the first loss_window_size masked positions per sample
      batch_size = xt.shape[0]
      positions = torch.arange(self.num_tokens, device=xt.device).unsqueeze(0).expand(batch_size, -1)
      loss_mask = (
        (positions >= first_masked_idx.unsqueeze(1)) & 
        (positions < first_masked_idx.unsqueeze(1) + loss_window_size) &
        masked_positions  # Only include actual masked positions
      ).float()
      
      # Only apply where there actually is a masked token
      loss_mask = loss_mask * has_masked.unsqueeze(1).float()
    else:
      # Original behavior: mask all masked positions
      loss_mask = masked_positions.float()

    log_p_theta = log_p_theta * loss_mask
 
    if self.ar_noise:
      num_loss_tokens = loss_mask.sum(dim=1)[:, None] # shape: (batch_size, 1)
      num_loss_tokens = num_loss_tokens.clamp(min=1)
      result = -self.num_tokens / num_loss_tokens * log_p_theta
    else:
      if low_var:
        result = -log_p_theta
      else:
        result = dalpha_t / (1 - alpha_t) * log_p_theta
    
    return result

  def q_xt(self, x, alpha_t, loss_mask=None):
    """Computes the noisy sample xt.
    
    If ar_noise is True, masks the rightmost solution tokens instead of random tokens.
    The number of masked tokens is randomized based on the number of solution tokens.
    
    If loss_mask is provided, only solution tokens (where loss_mask=1) can be masked.
    Problem tokens (where loss_mask=0) are never masked.
    For ar_noise=True, loss_mask is assumed to be contiguous (all 0s then all 1s).

    Args:
      x: int torch.Tensor with shape (batch_size, seq_len), input. 
      alpha_t: torch.Tensor with shape (batch_size, 1), noise level.
      loss_mask: optional torch.Tensor with shape (batch_size, seq_len), 
                 1 for solution tokens, 0 for problem tokens.
    """
    batch_size, seq_len = x.shape
    
    # If no loss_mask, treat all tokens as solution tokens
    if loss_mask is None:
      loss_mask = torch.ones_like(x)
    
    if self.ar_noise:
      # Mask rightmost solution tokens
      # Find solution start
      solution_start = loss_mask.long().argmax(dim=1)  # (batch_size,)

      # Count valid threshold positions
      if self.exclude_padding_from_mask_threshold:
        # Exclude padding tokens from threshold selection
        is_padding = (x == self.pad_index)  # (batch_size, seq_len)
        valid_threshold = loss_mask.bool() & (~is_padding)  # (batch_size, seq_len)
        num_valid = valid_threshold.sum(dim=1)  # (batch_size,)
      else:
        # Treat all solution tokens equally (default)
        num_valid = loss_mask.sum(dim=1)  # (batch_size,)

      # Sample random integer in [0, num_valid - 1]
      # - offset=0 means threshold=solution_start (all solution tokens masked) - INCLUDED
      # - We never sample offset >= num_valid (which would mask no solution tokens) - EXCLUDED
      rand_offset = torch.floor(torch.rand(batch_size, device=x.device) * num_valid.float()).long()
      mask_threshold = solution_start + rand_offset
            
      # Mask everything from threshold to the end
      indices = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
      move_indices = indices >= mask_threshold.unsqueeze(1)
      xt = torch.where(move_indices, self.mask_index, x)
      return xt
    else:
      # Random masking of solution tokens only
      rand_mask = torch.rand(*x.shape, device=x.device) < 1 - alpha_t
      move_indices = rand_mask & (loss_mask == 1)
      xt = torch.where(move_indices, self.mask_index, x)
      return xt

  def _sort_indices(
    self, indices, shuffle_clean=False, shuffle_masked=False, loss_mask=None):
    """Sort indices to place clean tokens before masked tokens.
    
    Args:
        indices: Token indices with mask_index for masked positions
        shuffle_clean: Whether to shuffle clean (unmasked) tokens
        shuffle_masked: Whether to shuffle masked tokens
        loss_mask: If provided, problem tokens (loss_mask=0) are always sorted first
    """
    batch_size, seq_len = indices.shape
    device = indices.device
    
    masked = (indices == self.mask_index)
    
    # Compute offsets for clean tokens
    if shuffle_clean:
      clean_offsets = torch.rand(batch_size, seq_len, device=device) * 0.9
    else:
      clean_offsets = torch.linspace(0, 0.9, seq_len, device=device).unsqueeze(0)
    
    # Compute offsets for masked tokens
    if shuffle_masked:
      masked_offsets = torch.rand(batch_size, seq_len, device=device) * 0.9
    else:
      masked_offsets = torch.linspace(0, 0.9, seq_len, device=device).unsqueeze(0)
    
    # Combine based on masked status
    offsets = torch.where(masked, masked_offsets, clean_offsets)
    
    # If loss_mask provided, problem tokens (loss_mask=0) always come first with sorted offsets
    # this logic assumes problem tokens are never masked
    if loss_mask is not None:
      if self.config.algo.get('shuffle_problem_tokens', False):
        problem_offsets = torch.rand(batch_size, seq_len, device=device) * 0.9 - 1.0
      else:
        problem_offsets = torch.linspace(0, 0.9, seq_len, device=device).unsqueeze(0) - 1.0
      offsets = torch.where(loss_mask.bool(), offsets, problem_offsets)

    sort_idx = (masked.float() + offsets).argsort(descending=False)
    return sort_idx

  def _loss(self, x0, valid_tokens,
            current_accumulation_step=None, train_mode=False,
            loss_mask=None):
    diffusion_loss, sort_idx, cutoffs = self.nll(
      x0, None, current_accumulation_step, train_mode, loss_mask=loss_mask)
    valid_tokens_diffusion = torch.gather(
      valid_tokens, dim=1, index=sort_idx)

    # Combine valid_tokens with loss_mask if provided
    # When loss_mask is provided, we only count solution tokens in the denominator.
    # The loss on problem tokens is naturally 0 (they're never masked), but we must
    # also exclude them from num_diffusion to avoid diluting the per-token loss.
    if loss_mask is not None:
      loss_mask_diffusion = torch.gather(
        loss_mask, dim=1, index=sort_idx)
      effective_mask = valid_tokens_diffusion * loss_mask_diffusion
    else:
      effective_mask = valid_tokens_diffusion

    # Position-based loss monitoring (4 quartiles by position in masked span)
    bucket_losses, bucket_counts = None, None
    if self.config.algo.get('log_position_losses', False):
      bucket_losses, bucket_counts = self._compute_position_losses(
        diffusion_loss, cutoffs, effective_mask, train_mode=train_mode)

    diffusion_loss_no_reduce = diffusion_loss.clone().detach()
    diffusion_loss = (
      diffusion_loss * effective_mask).sum()
    num_diffusion = effective_mask.sum()
    diffusion_loss_per_token = diffusion_loss / num_diffusion

    return trainer_base.Loss(
        loss=diffusion_loss_per_token,
        nlls=diffusion_loss_per_token * num_diffusion,
        reconstruction_loss=torch.tensor(0.0).to(x0.device),
        num_tokens=num_diffusion,
        bucket_losses=bucket_losses,
        bucket_counts=bucket_counts)

  def _compute_position_losses(self, loss, cutoffs, effective_mask, train_mode=False):
    """Compute losses bucketed by position within the masked span (4 quartiles).

    This monitoring helps analyze whether tokens with more "latent context"
    (masked tokens to their left in causal attention) have lower loss.

    Args:
        loss: [B, L] per-token losses in sorted order (clean first, masked last)
        cutoffs: [B] number of clean tokens per sample
        effective_mask: [B, L] valid loss positions (1 where loss should count)
        train_mode: If True, also log instantaneous losses per bucket

    Returns:
        tuple: (bucket_losses, bucket_counts) where both are dicts mapping
               bucket index (0-3) to sum of losses / count of tokens
    """
    batch_size, seq_len = loss.shape
    device = loss.device

    # Positions in sequence
    positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, L]

    # Expand cutoffs for broadcasting
    cutoffs_expanded = cutoffs.unsqueeze(1)  # [B, 1]
    num_masked = (seq_len - cutoffs).float().unsqueeze(1)  # [B, 1]

    # Only compute for masked positions (pos >= cutoffs)
    is_masked = positions >= cutoffs_expanded  # [B, L]
    masked_pos = (positions - cutoffs_expanded).float()  # [B, L]

    # Relative position within masked span: 0.0 = first masked, 1.0 = last masked
    # Avoid div by zero when all tokens are clean or only 1 masked token
    relative_pos = masked_pos / (num_masked - 1).clamp(min=1)  # [B, L]
    relative_pos = relative_pos.clamp(0, 1)

    # Bucket assignment (0, 1, 2, 3) for quartiles
    bucket = (relative_pos * 4).long().clamp(max=3)  # [B, L]

    # Compute loss per bucket
    bucket_losses = {}
    bucket_counts = {}
    for b in range(4):
      bucket_mask = (bucket == b) & is_masked & (effective_mask > 0)
      bucket_loss = (loss * bucket_mask).sum()
      bucket_count = bucket_mask.sum()

      bucket_losses[b] = bucket_loss.detach()
      bucket_counts[b] = bucket_count.detach()

      # Log instantaneous losses during training
      if train_mode and bucket_count > 0:
        avg_loss = bucket_loss / bucket_count
        self.log(f'trainer/loss_bucket_{b}', avg_loss.item(),
                 on_step=True, on_epoch=False, sync_dist=True)

    return bucket_losses, bucket_counts

  def nll(self, x0, output_tokens,
          current_accumulation_step=None, train_mode=False,
          loss_mask=None):
    del output_tokens
    t = self._sample_t(x0.shape[0],
                       current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)
    
    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)

    xt = self.q_xt(x0, alpha_t, loss_mask=loss_mask)
    # sort inputs and targets before passing to the model
    shuffle_clean, shuffle_masked = self._get_shuffle_settings()
    sort_idx = self._sort_indices(
      xt, shuffle_clean=shuffle_clean, shuffle_masked=shuffle_masked, loss_mask=loss_mask)
    xt = torch.gather(xt, dim=1, index=sort_idx)
    x0 = torch.gather(x0, dim=1, index=sort_idx)
    
    # Compute mask_cutoffs for attention windowing
    # After sorting, clean tokens are first, then masked tokens
    # mask_cutoffs limits attention to positions < mask_cutoffs
    mask_cutoffs = None
    # cutoffs = number of clean tokens per sample (always compute for position logging)
    cutoffs = (xt != self.mask_index).sum(dim=1)  # [batch_size]
    if self.mtp_window_size > 0:
      # mask_cutoffs = min(cutoffs + mtp_window_size, seq_len)
      mask_cutoffs = torch.clamp(cutoffs + self.mtp_window_size, max=self.num_tokens)
      
      # For "random" mtp_mode, shuffle the masked tokens (except the first one)
      # This allows the model to predict a random span of tokens instead of contiguous
      if self.mtp_mode == 'random':
        # Fixed positions: clean tokens (0 to cutoffs-1) + first masked token (cutoffs)
        # So positions 0 to cutoffs (inclusive) are fixed
        positions = torch.arange(self.num_tokens, device=xt.device).unsqueeze(0)  # (1, L)
        fixed_mask = positions <= cutoffs.unsqueeze(1)  # (B, L)
        
        # Compute shuffle permutation and apply to xt, x0, sort_idx
        gather_idx = self._compute_batch_shuffle_perm(fixed_mask)
        xt = torch.gather(xt, dim=1, index=gather_idx)
        x0 = torch.gather(x0, dim=1, index=gather_idx)
        sort_idx = torch.gather(sort_idx, dim=1, index=gather_idx)

    # pass sort_idx into the model to also sort pos. embeddings
    # _process_model_output performs zero-masking trick
    log_x_theta = self.forward(xt, sigma=sigma, sort_idx=sort_idx, mask_cutoffs=mask_cutoffs)
    # nll_per_token performs carry-over loss masking
    return self.nll_per_token(
      log_x_theta=log_x_theta,
      xt=xt,
      x0=x0,
      alpha_t=alpha_t,
      dalpha_t=dalpha_t,
      low_var=train_mode and self.loss_type == 'low_var',
      train_mode=train_mode), sort_idx, cutoffs
  
  def _sample_t(self, n, accum_step):
    if accum_step is not None:
      # During training
      batch_dim = n
      n = self.config.loader.global_batch_size
    _eps_t = torch.rand(n, device=self.device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=self.device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if accum_step is not None:
      t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
      t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
      t = t.chunk(self.trainer.accumulate_grad_batches)[
        accum_step]
      # corner case for the last datapoint
      t = t[:batch_dim]
    return t

  def _tokens_unmasked_per_step(self, num_steps: int):
    # ------------------------
    # Deterministic schedule
    # ------------------------
    if self.config.sampling.get("fixed_tokens_per_step", False):
        total = int(self.num_tokens)
        base, extra = divmod(total, num_steps)
        schedule = [(base + 1) if i < extra else base for i in range(num_steps)]
        return schedule

    # ------------------------
    # Stochastic (original) schedule
    # ------------------------
    remaining_tokens = int(self.num_tokens)
    num_tokens_to_unmask = []
    dt = 1.0 / num_steps

    # Assumes a log-linear schedule (as in the original).
    for t in np.linspace(1.0, dt, num_steps):
        _, alpha_t = self.noise(t)
        _, alpha_s = self.noise(t - dt)
        p = (alpha_s - alpha_t) / (1.0 - alpha_t)
        p = float(np.clip(p, 0.0, 1.0))  # robust to tiny numerical drift
        if remaining_tokens <= 0:
            break
        n_unmask = np.random.binomial(remaining_tokens, p)
        if n_unmask != 0:
            num_tokens_to_unmask.append(int(n_unmask))
            remaining_tokens -= int(n_unmask)

    # Flush any leftover tokens
    if remaining_tokens != 0:
        num_tokens_to_unmask.append(int(remaining_tokens))

    return num_tokens_to_unmask


  def _validate_sampling_configs(self, num_steps=None):
    """Validate and resolve sampling configs, returning resolved values."""
    if num_steps is None:
      num_steps = self.config.sampling.steps

    attn_mode = self.config.algo.diffusion_attn_mode
    kv_cache = self.config.sampling.get('kv_cache', None)
    trim_masked_tokens = self.config.sampling.get('trim_masked_tokens', None)
    unmask_policy = self.config.sampling.get('unmask_policy', None)

    if unmask_policy not in ['uniform', 'latent_tokens', None]:
      if trim_masked_tokens:
        raise ValueError("trim_masked_tokens is not supported for unmask_policy != uniform or latent_tokens")
      # Force trim_masked_tokens to False for non-uniform unmask policies
      trim_masked_tokens = False

    if kv_cache is None:
      kv_cache = attn_mode in ['causal', 'causal_context']
      print(f"kv_cache defaulted to {kv_cache}")
    if trim_masked_tokens is None:
      trim_masked_tokens = attn_mode in ['causal', 'causal_output', 'solo_causal', 'solo_full']
      print(f"trim_masked_tokens defaulted to {trim_masked_tokens}")

    if attn_mode in ['full', 'causal_output', 'solo_full']:
      assert not kv_cache, f"kv cache is not supported for attn_mode = {attn_mode}"

    if attn_mode in ['full', 'causal_context']:
      assert not trim_masked_tokens, "trim_masked_tokens is not supported for attn_mode = full or causal_context"

    return num_steps, kv_cache, trim_masked_tokens, unmask_policy

  @torch.no_grad()
  def generate_samples(self, num_samples, num_steps=None, return_stats=False):
    """
    Generate samples from the model. Supports causal, causal_context, causal_output, and full attention patterns.
    """
    # TODO double check all this
    num_steps, kv_cache, trim_masked_tokens, unmask_policy = self._validate_sampling_configs(num_steps)
    attn_mode = self.config.algo.diffusion_attn_mode
    noise_scale = self.config.sampling.get('noise_scale', 1.0)

    n_latent_tokens = self.config.sampling.get("n_latent_tokens", 0)
    if n_latent_tokens > 0:
      assert attn_mode in ['causal', 'causal_output', 'solo_causal', 'solo_full']

    unmask_k_tokens = self._tokens_unmasked_per_step(num_steps)
    num_diffusion_tokens = sum(unmask_k_tokens)
    assert num_diffusion_tokens == self.num_tokens

    _, shuffle_masked = self._get_shuffle_settings()

    if unmask_policy != 'uniform':
      assert shuffle_masked, "confidence-based decoding does not support predetermined schedules"
    
    if shuffle_masked:
      # Random permutation for shuffled masked tokens
      sort_idx = torch.rand(
        num_samples, self.num_tokens).argsort(
          descending=False).to(self.device)
    else:
      # Deterministic left-to-right order
      sort_idx = torch.arange(
        self.num_tokens, device=self.device).unsqueeze(0).expand(
          num_samples, -1).clone()

    x = self.prior_sample(num_samples, self.num_tokens)
    x = torch.gather(x, dim=1, index=sort_idx)

    unmask_k_tokens = unmask_k_tokens + [1] * (
      self.num_tokens - num_diffusion_tokens)      
    assert sum(unmask_k_tokens) == self.num_tokens


    unmasked_tokens = 0
    start = time.perf_counter()
    self.backbone.reset_kv_cache()

    for i, k in enumerate(unmask_k_tokens):

      n_latent_tokens = self.config.sampling.get("n_latent_tokens", 0)
      if unmask_policy == 'uniform':
        # OK to use all other tokens as latent tokens
        n_latent_tokens = min(n_latent_tokens, self.num_tokens - unmasked_tokens - k)
      else:
        n_latent_tokens = min(n_latent_tokens, (self.num_tokens - unmasked_tokens - k))
        # Ensure at least topk_candidate_min candidates remain for topp selection
        topk_candidate_min = self.config.sampling.get("topk_candidate_min", 0)
        if topk_candidate_min > 0:
          remaining = self.num_tokens - unmasked_tokens - k
          n_latent_tokens = min(n_latent_tokens, remaining - topk_candidate_min)
          n_latent_tokens = max(0, n_latent_tokens)

      # Latent token modulation: rearrange to place sampled latent tokens before k tokens to decode
      if unmask_policy == 'uniform':
        x_fwd, sort_idx_fwd, latent_reorder_indices = self._latent_reorder(
          x, sort_idx, unmasked_tokens, k, n_latent_tokens)
      else:
        # we will modulate differently for adaptive orders
        x_fwd, sort_idx_fwd = x, sort_idx

      # --- Token range bookkeeping for forward pass ---
      # - last_k_start: Index marking the end of already-encoded (clean) tokens from the previous step.
      #   - Tokens in [:last_k_start] are already encoded and cached.
      #   - Tokens in [last_k_start:unmasked_tokens] will be processed and cached in this pass.
      # - cutoffs: Marks the current boundary of clean context (used for attention mask construction).
      # - masked_tokens_end: End index for masked tokens to use in this forward pass.
      #   - Using masked_tokens_end < num_tokens can speed up execution, but is not always allowed.
      # -----------------------------------------------

      last_k_start = 0 if i == 0 else (unmasked_tokens - unmask_k_tokens[i-1])
      cutoffs = unmasked_tokens

      masked_tokens_end = self.num_tokens
      if trim_masked_tokens:
        masked_tokens_end = unmasked_tokens + n_latent_tokens + k

      # Compute mask_cutoffs for attention windowing during inference
      # This restricts attention so tokens can only attend to positions < mask_cutoffs
      mask_cutoffs = None
      if self.mtp_window_size > 0:
        mask_cutoffs = min(unmasked_tokens + self.mtp_window_size, self.num_tokens)

      log_p_x0 = self.backbone.forward_sample(
        zt=x_fwd,
        sort_idx=sort_idx_fwd,
        attn_mode=attn_mode,
        cutoffs=cutoffs,
        kv_cache=kv_cache,
        last_k_start=last_k_start,
        curr_k_start=unmasked_tokens,  # also last_k_end
        curr_k_end=masked_tokens_end,
        mask_cutoffs=mask_cutoffs)

      if self.config.sampling.use_float64:
        log_p_x0 = log_p_x0.to(torch.float64)
      log_p_x0[:, :, self.mask_index] = self.neg_infinity
      if self.config.sampling.p_nucleus < 1:
        log_p_x0 = utils.top_k_top_p_filtering(
          log_p_x0, top_p=self.config.sampling.p_nucleus)

      # When kv_cache is enabled, forward pass does not return log_p_x0[:unmasked_tokens]
      # Pad to full length for convenience
      if kv_cache:
        # pad log_p_x0 to full length
        log_p_pad = torch.zeros((log_p_x0.shape[0], unmasked_tokens, log_p_x0.shape[-1])).to(self.device)
        log_p_x0 = torch.cat([log_p_pad, log_p_x0], dim=1)

      greedy_tokens = self.config.sampling.get('greedy_tokens', False)

      if unmask_policy == 'uniform':
        log_p_x0_k = log_p_x0[:, unmasked_tokens+n_latent_tokens:unmasked_tokens+n_latent_tokens+k, :]
        # sample from categorical distrs (or greedy argmax)
        if greedy_tokens:
          y = log_p_x0_k.argmax(-1)
        else:
          noise_slice = torch.distributions.Gumbel(0, 1).sample(log_p_x0_k.shape).to(self.device)
          y = (log_p_x0_k + noise_slice * noise_scale).argmax(-1)
        # unmask tokens (sort_idx remains unchanged for rightmost masking)
        x[:, unmasked_tokens:unmasked_tokens+k] = y
        unmasked_tokens += k
      else:
        confidence_scores = self._compute_confidence_scores(log_p_x0, unmask_policy)
        confidence_scores[:, :unmasked_tokens+n_latent_tokens] = self.neg_infinity

        # Limit candidates to first topk_candidate_max positions if specified
        topk_candidate_max = self.config.sampling.get("topk_candidate_max", 0)
        if topk_candidate_max > 0:
          candidate_end = unmasked_tokens + n_latent_tokens + topk_candidate_max
          if candidate_end < self.num_tokens:
            confidence_scores[:, candidate_end:] = self.neg_infinity

        topk_indices = confidence_scores.topk(k, dim=-1).indices
        topk_indices_ = topk_indices[:, :, None].expand(-1, -1, log_p_x0.shape[-1])
        log_p_x0_k = torch.gather(log_p_x0, dim=1, index=topk_indices_)

        # sample from categorical distrs (or greedy argmax)
        if greedy_tokens:
          y = log_p_x0_k.argmax(-1)
        else:
          noise_slice = torch.distributions.Gumbel(0, 1).sample(log_p_x0_k.shape).to(self.device)
          y = (log_p_x0_k + noise_slice * noise_scale).argmax(-1)

        # Write y to x at topk_indices positions (where predictions came from)
        x.scatter_(1, topk_indices, y)

        # Rearrange both x and sort_idx so decoded positions are at [unmasked:unmasked+k]
        sort_idx, suffix_perm = self._move_decoded_indices(
            sort_idx, unmasked_tokens, topk_indices, return_perm=True)
        x = torch.cat([x[:, :unmasked_tokens],
                       torch.gather(x[:, unmasked_tokens:], 1, suffix_perm)], dim=1)
        unmasked_tokens += k

        # avoid using the same positions as latents repeatedly
        perm = torch.cat([
            torch.arange(unmasked_tokens, device=x.device),
            unmasked_tokens + torch.randperm(x.shape[1] - unmasked_tokens, device=x.device)
        ])
        x = x[:, perm]
        sort_idx = sort_idx[:, perm]


    self.backbone.reset_kv_cache()
    sort_idx_reversed = utils.get_reverse_indices(sort_idx)
    x = torch.gather(x, dim=1, index=sort_idx_reversed)

    end = time.perf_counter()
    duration = end - start
    print(f'Sampling duration: {duration} seconds')

    if return_stats:
      return x, {'duration': duration}
    else:
      return x


  @torch.no_grad()
  def generate_completions(self, completion_batch, num_steps=None, return_stats=False):
    """
    Generate completions for problems in the batch.
    """
    num_steps, kv_cache, trim_masked_tokens, unmask_policy = self._validate_sampling_configs(num_steps)
    attn_mode = self.config.algo.diffusion_attn_mode
    noise_scale = self.config.sampling.get('noise_scale', 1.0)

    n_latent_tokens = self.config.sampling.get("n_latent_tokens", 0)
    if n_latent_tokens > 0:
      assert attn_mode in ['causal', 'causal_output', 'solo_causal', 'solo_full']
    
    input_ids = completion_batch['input_ids'].to(self.device)
    is_solution = completion_batch['loss_mask'].to(self.device).bool()
    num_samples = input_ids.shape[0]
    
    # initialize x and sort_idx based on shuffle settings
    # For completions: problem tokens are clean (is_solution=False), solution tokens are masked (is_solution=True)
    shuffle_clean, shuffle_masked = self._get_shuffle_settings()

    if unmask_policy != 'uniform':
      assert shuffle_masked, "confidence-based decoding does not support predetermined schedules"
      assert not trim_masked_tokens, "confidence-based decoding does not support trimming masked tokens"
    
    # Compute offsets for problem tokens (clean)
    if shuffle_clean:
      problem_offsets = torch.rand(num_samples, self.num_tokens, device=self.device) * 0.9
    else:
      problem_offsets = torch.linspace(0, 0.9, self.num_tokens, device=self.device).unsqueeze(0)
    
    # Compute offsets for solution tokens (masked) - add 1.0 to come after problem tokens
    if shuffle_masked:
      solution_offsets = torch.rand(num_samples, self.num_tokens, device=self.device) * 0.9 + 1.0
    else:
      solution_offsets = torch.linspace(0, 0.9, self.num_tokens, device=self.device).unsqueeze(0) + 1.0
    
    offsets = torch.where(is_solution, solution_offsets, problem_offsets)
    sort_idx = offsets.argsort(dim=1, descending=False)
      
    # x is in sorted order; is_solution and input_ids stay in original order
    x = torch.where(is_solution, self.mask_index, input_ids)
    x = torch.gather(x, dim=1, index=sort_idx)

    unmask_k_tokens = self._tokens_unmasked_per_step(num_steps)
    unmask_k_tokens = unmask_k_tokens + [1] * (
      self.num_tokens - sum(unmask_k_tokens))      
    assert sum(unmask_k_tokens) == self.num_tokens

    unmasked_tokens = 0
    start = time.perf_counter()
    self.backbone.reset_kv_cache()

    for i, k in enumerate(unmask_k_tokens):

      n_latent_tokens = self.config.sampling.get("n_latent_tokens", 0)
      n_latent_tokens = min(n_latent_tokens, self.num_tokens - unmasked_tokens - k)

      if unmask_policy == 'uniform':
        # OK to use all other tokens as latent tokens
        n_latent_tokens = min(n_latent_tokens, self.num_tokens - unmasked_tokens - k)
      else:
        # for adaptive orders, we need to leave enough for the adaptive logic to choose from
        n_latent_tokens = min(n_latent_tokens, (self.num_tokens - unmasked_tokens - k))
        # Ensure at least topk_candidate_min candidates remain for topp selection
        topk_candidate_min = self.config.sampling.get("topk_candidate_min", 0)
        if topk_candidate_min > 0:
          remaining = self.num_tokens - unmasked_tokens - k
          n_latent_tokens = min(n_latent_tokens, remaining - topk_candidate_min)
          n_latent_tokens = max(0, n_latent_tokens)

      # Latent token modulation: rearrange to place sampled latent tokens before k tokens to decode
      if unmask_policy == 'uniform':
        x_fwd, sort_idx_fwd, latent_reorder_indices = self._latent_reorder(
          x, sort_idx, unmasked_tokens, k, n_latent_tokens)
      else:
        # we will modulate differently for adaptive orders
        x_fwd, sort_idx_fwd = x, sort_idx

      # --- Token range bookkeeping for forward pass ---
      last_k_start = 0 if i == 0 else (unmasked_tokens - unmask_k_tokens[i-1])
      cutoffs = unmasked_tokens
      
      masked_tokens_end = self.num_tokens
      if trim_masked_tokens:
        masked_tokens_end = unmasked_tokens + n_latent_tokens + k

      # Compute mask_cutoffs for attention windowing during inference
      mask_cutoffs = None
      if self.mtp_window_size > 0:
        mask_cutoffs = min(unmasked_tokens + self.mtp_window_size, self.num_tokens)

      log_p_x0 = self.backbone.forward_sample(
        zt=x_fwd,  
        sort_idx=sort_idx_fwd, 
        attn_mode=attn_mode,
        cutoffs=cutoffs,
        kv_cache=kv_cache,
        last_k_start=last_k_start,
        curr_k_start=unmasked_tokens,  # also last_k_end
        curr_k_end=masked_tokens_end,
        mask_cutoffs=mask_cutoffs)
      
      if self.config.sampling.use_float64:
        log_p_x0 = log_p_x0.to(torch.float64)
      log_p_x0[:, :, self.mask_index] = self.neg_infinity
      if self.config.sampling.p_nucleus < 1:
        log_p_x0 = utils.top_k_top_p_filtering(
          log_p_x0, top_p=self.config.sampling.p_nucleus)

      if kv_cache:
        # pad log_p_x0 to full length
        log_p_pad = torch.zeros((log_p_x0.shape[0], unmasked_tokens, log_p_x0.shape[-1])).to(self.device)
        log_p_x0 = torch.cat([log_p_pad, log_p_x0], dim=1)

      greedy_tokens = self.config.sampling.get('greedy_tokens', False)

      if unmask_policy == 'uniform':
        # Extract predictions for the k tokens (after latent tokens in x_fwd)
        # Note: log_p_x0_k predictions correspond to x[:, unmasked_tokens:unmasked_tokens+k]
        log_p_x0_k = log_p_x0[:, unmasked_tokens+n_latent_tokens:unmasked_tokens+n_latent_tokens+k, :]

        # Get original indices for positions unmasked_tokens:unmasked_tokens+k via sort_idx
        orig_indices_k = sort_idx[:, unmasked_tokens:unmasked_tokens+k]
        is_solution_k = torch.gather(is_solution, dim=1, index=orig_indices_k)
        input_ids_k = torch.gather(input_ids, dim=1, index=orig_indices_k)

        # sample from categorical distrs (or greedy argmax)
        if greedy_tokens:
          y = log_p_x0_k.argmax(-1)
        else:
          noise_slice = torch.distributions.Gumbel(0, 1).sample(log_p_x0_k.shape).to(self.device)
          y = (log_p_x0_k + noise_slice * noise_scale).argmax(-1)
        # keep problem tokens the same (overwrite sampled values with actual problem tokens)
        y = torch.where(is_solution_k, y, input_ids_k)

        # Write back to original x
        x[:, unmasked_tokens:unmasked_tokens+k] = y
        unmasked_tokens += k
      else:
        # Get is_solution in sorted order for confidence scoring
        is_solution_sorted = torch.gather(is_solution, dim=1, index=sort_idx)

        confidence_scores = self._compute_confidence_scores(log_p_x0, unmask_policy)
        pos_infinity = -self.neg_infinity

        # Mask decoded positions (never re-select)
        confidence_scores[:, :unmasked_tokens] = self.neg_infinity

        # Mask latent zone for solution tokens only (problem tokens should not be treated as latent)
        positions = torch.arange(self.num_tokens, device=confidence_scores.device).unsqueeze(0)
        in_latent = (positions >= unmasked_tokens) & (positions < unmasked_tokens + n_latent_tokens)
        latent_solution = in_latent & is_solution_sorted
        confidence_scores = torch.where(latent_solution, self.neg_infinity, confidence_scores)

        # Limit candidates to first topk_candidate_max positions if specified
        topk_candidate_max = self.config.sampling.get("topk_candidate_max", 0)
        if topk_candidate_max > 0:
          candidate_end = unmasked_tokens + n_latent_tokens + topk_candidate_max
          if candidate_end < self.num_tokens:
            confidence_scores[:, candidate_end:] = self.neg_infinity

        # Undecoded problem tokens have highest priority (selected before solution tokens)
        # and ignores candidate/latent token restrictions
        undecoded_problem = (positions >= unmasked_tokens) & (~is_solution_sorted)
        confidence_scores = torch.where(undecoded_problem, pos_infinity, confidence_scores)

        topk_indices = confidence_scores.topk(k, dim=-1).indices
        topk_indices_ = topk_indices[:, :, None].expand(-1, -1, log_p_x0.shape[-1])
        log_p_x0_k = torch.gather(log_p_x0, dim=1, index=topk_indices_)

        # Get original indices for topk positions via sort_idx
        orig_indices_k = torch.gather(sort_idx, dim=1, index=topk_indices)
        is_solution_k = torch.gather(is_solution, dim=1, index=orig_indices_k)
        input_ids_k = torch.gather(input_ids, dim=1, index=orig_indices_k)

        # sample from categorical distrs (or greedy argmax)
        if greedy_tokens:
          y = log_p_x0_k.argmax(-1)
        else:
          noise_slice = torch.distributions.Gumbel(0, 1).sample(log_p_x0_k.shape).to(self.device)
          y = (log_p_x0_k + noise_slice * noise_scale).argmax(-1)
        y = torch.where(is_solution_k, y, input_ids_k)

        # Write y to x at topk_indices positions (where predictions came from)
        x.scatter_(1, topk_indices, y)

        # Rearrange both x and sort_idx so decoded positions are at [unmasked:unmasked+k]
        sort_idx, suffix_perm = self._move_decoded_indices(
            sort_idx, unmasked_tokens, topk_indices, return_perm=True)
        x = torch.cat([x[:, :unmasked_tokens],
                       torch.gather(x[:, unmasked_tokens:], 1, suffix_perm)], dim=1)
        unmasked_tokens += k

        # avoid using the same positions as latents repeatedly
        perm = torch.cat([
            torch.arange(unmasked_tokens, device=x.device),
            unmasked_tokens + torch.randperm(x.shape[1] - unmasked_tokens, device=x.device)
        ])
        x = x[:, perm]
        sort_idx = sort_idx[:, perm]

    self.backbone.reset_kv_cache()
    sort_idx_reversed = utils.get_reverse_indices(sort_idx)
    x = torch.gather(x, dim=1, index=sort_idx_reversed)

    end = time.perf_counter()
    duration = end - start
    print(f'Sampling duration: {duration} seconds')

    if return_stats:
      return x, {'duration': duration}
    else:
      return x

  def _debug_generation_state(self, x, sort_idx, unmasked_tokens, step_idx, 
                                is_solution=None, input_ids=None, sample_idx=0):
    """
    Debug helper to print the state of sequence generation.
    
    Args:
      x: Current tokens in sorted order (batch_size, seq_len)
      sort_idx: Current sorting indices (batch_size, seq_len)
      unmasked_tokens: Number of tokens unmasked so far
      step_idx: Current step index
      is_solution: Boolean mask for solution tokens in sorted order (batch_size, seq_len)
      input_ids: Original input ids in sorted order (batch_size, seq_len)
      sample_idx: Which sample in batch to display (default 0)
    """
    sort_idx_reversed = utils.get_reverse_indices(sort_idx)
    x_original_order = torch.gather(x, dim=1, index=sort_idx_reversed)
    
    x_sample = x[sample_idx].cpu().tolist()
    x_orig_sample = x_original_order[sample_idx].cpu().tolist()
    sort_idx_sample = sort_idx[sample_idx].cpu().tolist()
    
    print(f"\n{'='*60}")
    print(f"Step {step_idx}, sample {sample_idx}")
    print(f"{'='*60}")
    
    # Show x in sorted order with mask indicator
    print(f"\nx (sorted order):")
    sorted_str = []
    for i, tok in enumerate(x_sample):
      if tok == self.mask_index:
        sorted_str.append("[M]")
      else:
        sorted_str.append(str(tok))
    print(" ".join(sorted_str))
    # if len(sorted_str) > 50:
    #   print(f"  ... ({len(sorted_str) - 50} more tokens)")
    
    # Show x in original order
    print(f"\nx (original order):")
    orig_str = []
    for i, tok in enumerate(x_orig_sample):
      if tok == self.mask_index:
        orig_str.append("[M]")
      else:
        orig_str.append(str(tok))
    print(" ".join(orig_str))

    # Show sort_idx mapping
    print(f"\nsort_idx: {sort_idx_sample}")
    
    # Show is_solution if provided
    # if is_solution is not None:
    #   is_sol_sample = is_solution[sample_idx].cpu().tolist()
    #   n_problem = sum(1 for s in is_sol_sample if not s)
    #   n_solution = sum(1 for s in is_sol_sample if s)
    #   print(f"\nis_solution: {n_problem} problem tokens, {n_solution} solution tokens")
    #   # Show boundary
    #   first_solution_idx = next((i for i, s in enumerate(is_sol_sample) if s), -1)
    #   print(f"First solution token at sorted position: {first_solution_idx}")
  
    
    print(f"{'='*60}\n")


  # TODO things to handle post-call:
  # forbid decoding latent tokens
  # forbid decoding clean tokens (:unmasked_tokens)
  # problem tokens first
  def _compute_confidence_scores(self, log_p_x0, unmask_policy):
    p_x0 = F.softmax(log_p_x0, dim=-1)
    if unmask_policy == "uniform":
      confidence_scores = torch.linspace(1, 0, log_p_x0.shape[1], device=log_p_x0.device)
      confidence_scores = confidence_scores.repeat(log_p_x0.shape[0], 1)
    elif unmask_policy == "random":
      confidence_scores = torch.rand_like(p_x0.max(dim=-1).values)
    elif unmask_policy == "topp":
      confidence_scores = p_x0.max(dim=-1).values
    elif unmask_policy == "entropy":
      entropy = (p_x0 * (p_x0 + 1e-10).log()).sum(dim=-1)
      confidence_scores = entropy
    elif unmask_policy == "topp_margin":
      confidence_scores = p_x0.max(dim=-1).values - p_x0.topk(2, dim=-1).values[:, :, 1]
    else:
      raise ValueError(f"Invalid decoding order: {unmask_policy}")

    return confidence_scores


  def _latent_reorder(self, x, sort_idx, unmasked_tokens, k, n_latent_tokens):
    """
    Rearrange sequence to place sampled latent tokens before the k tokens to decode.
    
    Args:
        x: (batch_size, seq_len) token indices
        sort_idx: (batch_size, seq_len) sorting indices
        unmasked_tokens: number of already unmasked tokens
        k: number of tokens to decode in this step
        n_latent_tokens: number of latent tokens to sample and place before k
        
    Returns:
        x_fwd: rearranged x for forward pass
        sort_idx_fwd: rearranged sort_idx for forward pass
        latent_reorder_indices: (seq_len,) indices used for reordering, or None if no reordering
    """
    if n_latent_tokens <= 0:
      return x, sort_idx, None
    
    remaining_count = self.num_tokens - unmasked_tokens - k
    if remaining_count <= 0:
      return x, sort_idx, None
    
    # Sample n_latent_tokens indices from [unmasked_tokens+k, self.num_tokens)
    perm = torch.randperm(remaining_count, device=x.device)
    latent_offsets = perm[:n_latent_tokens]
    latent_indices = unmasked_tokens + k + latent_offsets
    
    # Use the remaining as non-latent offsets
    non_latent_offsets = perm[n_latent_tokens:]
    non_latent_indices = unmasked_tokens + k + non_latent_offsets
    
    # Build reorder indices:
    # [:unmasked_tokens] + latent_indices + [unmasked_tokens:unmasked_tokens+k] + non_latent_indices
    latent_reorder_indices = torch.cat([
      torch.arange(unmasked_tokens, device=x.device),
      latent_indices,
      torch.arange(unmasked_tokens, unmasked_tokens + k, device=x.device),
      non_latent_indices
    ])
    
    x_fwd = x[:, latent_reorder_indices]
    sort_idx_fwd = sort_idx[:, latent_reorder_indices]

    # print(self.num_tokens, unmasked_tokens, k, n_latent_tokens)
    
    return x_fwd, sort_idx_fwd, latent_reorder_indices

  def _latent_reorder_inverse(self, x_fwd, sort_idx_fwd, latent_reorder_indices):
    """
    Inverse of _latent_reorder: restore original ordering from reordered tensors.
    
    Args:
        x_fwd: (batch_size, seq_len) reordered token indices
        sort_idx_fwd: (batch_size, seq_len) reordered sorting indices
        latent_reorder_indices: (seq_len,) indices used for reordering, or None
        
    Returns:
        x: restored x in original order
        sort_idx: restored sort_idx in original order
    """
    if latent_reorder_indices is None:
      return x_fwd, sort_idx_fwd
    
    inverse_indices = torch.argsort(latent_reorder_indices)
    x = x_fwd[:, inverse_indices]
    sort_idx = sort_idx_fwd[:, inverse_indices]
    
    return x, sort_idx

  def _move_decoded_indices(self, sorted_idx: torch.Tensor, s: int, topk_indices: torch.Tensor,
                              return_perm: bool = False):
    """
    Args:
        sorted_idx: (n, L) LongTensor. Each row is a permutation / list of ids.
        s: int. Only positions >= s (the suffix) may be rearranged.
        topk_indices: (n, k) LongTensor, ABSOLUTE positions to move for each row.
                      Guaranteed that all values are >= s.
        return_perm: If True, also return the suffix permutation for applying to other tensors.

    Returns:
        out: (n, L) LongTensor with:
            out[:, :s]    == sorted_idx[:, :s]
            out[:, s:s+k] == values taken from the specified positions (in original order)
            out[:, s+k:]  == remaining suffix values (in original order)
        perm (optional): (n, L-s) LongTensor, the permutation applied to the suffix.
    """
    n, L = sorted_idx.shape
    _, k = topk_indices.shape
    Ls = L - s

    prefix = sorted_idx[:, :s]            # (n, s)
    suffix = sorted_idx[:, s:]            # (n, Ls)

    # Absolute position grid for the suffix
    abs_pos_suffix = torch.arange(s, L, device=sorted_idx.device).unsqueeze(0).expand(n, Ls)  # (n, Ls)

    # Membership mask: which suffix positions are in topk_indices
    in_topk = (abs_pos_suffix.unsqueeze(-1) == topk_indices.unsqueeze(1)).any(dim=-1)  # (n, Ls)

    # Build an "order in topk" map for absolute positions.
    # Default to a large number for non-topk positions.
    INF = k + Ls + 1
    order_map = torch.full((n, L), INF, device=sorted_idx.device, dtype=torch.long)  # (n, L)

    # For each row, set order_map[row, pos] = index in topk_indices[row]
    topk_order = torch.arange(k, device=sorted_idx.device).expand(n, k)              # (n, k)
    order_map.scatter_(1, topk_indices, topk_order)                                  # fill absolute positions

    # Pull orders for suffix positions
    order_in_topk = torch.gather(order_map, 1, abs_pos_suffix)                       # (n, Ls)

    # Composite key:
    #  - topk items:   key = order_in_topk (0..k-1)  -> appear first in specified order
    #  - non-topk:     key = k + pos                -> then keep original order
    pos = torch.arange(Ls, device=sorted_idx.device).expand(n, Ls)
    key = torch.where(in_topk, order_in_topk, k + pos)

    perm = torch.argsort(key, dim=1)                  # (n, Ls)
    new_suffix = torch.gather(suffix, 1, perm)        # (n, Ls)
    out = torch.cat([prefix, new_suffix], dim=1)      # (n, L)

    if return_perm:
      return out, perm
    return out

  def _compute_batch_shuffle_perm(self, fixed_mask):
      """
      Compute permutation indices for shuffling free positions.
      Each sample in the batch can have different fixed positions.

      Args:
          fixed_mask: (B, L) boolean tensor - True means position is fixed
      Returns:
          gather_idx: (B, L) tensor where output[b, i] = input[b, gather_idx[b, i]]
      """
      B, L = fixed_mask.shape
      device = fixed_mask.device

      gather_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L).clone()

      for b in range(B):
          free_positions = (~fixed_mask[b]).nonzero(as_tuple=True)[0]
          n_free = len(free_positions)
          if n_free <= 1:
              continue

          perm = torch.randperm(n_free, device=device)
          shuffled_positions = free_positions[perm]
          gather_idx[b, free_positions] = shuffled_positions

      return gather_idx

