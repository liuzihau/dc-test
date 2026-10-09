import time
import torch

import trainer_base
import utils


class MDLM(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _process_model_output(self, model_output, xt, sigma):
    del xt, sigma
    # zero-masking probabilities
    model_output[:, :, self.mask_index] = self.neg_infinity
    # Normalize the model_output such that x.exp() is
    # a probability distribution over vocab_size.
    model_output = model_output.log_softmax(-1)
    return model_output

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    log_p_theta = log_x_theta.gather(
      dim=-1,
      index=x0[:, :, None])[:, :, 0]
    # carry-over unmasking
    loss_mask = xt == self.mask_index
    log_p_theta = log_p_theta * loss_mask
    if low_var:
      return -log_p_theta
    else:
      return dalpha_t / (1 - alpha_t) * log_p_theta


class NoShuffleMDLM(MDLM):
  """MDLM without token shuffling - processes tokens in original sequence order.

  This variant is simpler than DiffLM:
  - No sort_idx reordering
  - Only supports full bidirectional attention
  - Left-to-right unmasking during generation
  - Compatible with shifted_logits

  Use this when you want to match diffusion-vs-ar's approach where tokens
  stay in their original order and position i-1 predicts position i.
  """

  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    # Validate configuration
    attn_mode = config.algo.get('diffusion_attn_mode', 'full')
    assert attn_mode == 'full', (
      f"NoShuffleMDLM only supports diffusion_attn_mode='full', got '{attn_mode}'"
    )
    # Time reweighting mode: 'none' (ELBO), 'linear' (diffusion-vs-ar style)
    self.time_reweighting = config.algo.get('time_reweighting', 'none')
    self.num_diffusion_steps = config.algo.get('num_diffusion_steps', 20)

  def q_xt_discrete(self, x, t, loss_mask=None):
    """Computes noisy sample using discrete time masking (diffusion-vs-ar style).

    Masking probability is (t+1)/T where t is in [0, T-1].

    Args:
      x: int torch.Tensor with shape (batch_size, seq_len), input.
      t: torch.Tensor with shape (batch_size,), discrete timestep in [0, T-1].
      loss_mask: optional torch.Tensor with shape (batch_size, seq_len),
                 1 for solution tokens, 0 for problem tokens.
    Returns:
      xt: masked input
      t_mask: boolean mask indicating which positions were masked
    """
    if loss_mask is None:
      loss_mask = torch.ones_like(x)

    # Masking probability: (t+1)/T, matching diffusion-vs-ar trainer.py:58
    T = self.num_diffusion_steps
    mask_prob = (t + 1).float() / T  # [batch_size]
    mask_prob = mask_prob[:, None]   # [batch_size, 1]

    # Random masking of solution tokens only
    u = torch.rand_like(x, dtype=torch.float)
    t_mask = (u < mask_prob) & (loss_mask == 1)
    xt = x.masked_fill(t_mask, self.mask_index)
    return xt, t_mask

  def q_xt(self, x, alpha_t, loss_mask=None):
    """Computes the noisy sample xt with optional loss_mask (continuous time).

    If loss_mask is provided, only solution tokens (where loss_mask=1) can be masked.
    Problem tokens (where loss_mask=0) are never masked.

    Args:
      x: int torch.Tensor with shape (batch_size, seq_len), input.
      alpha_t: torch.Tensor with shape (batch_size, 1), noise level.
      loss_mask: optional torch.Tensor with shape (batch_size, seq_len),
                 1 for solution tokens, 0 for problem tokens.
    """
    # If no loss_mask, treat all tokens as solution tokens (can be masked)
    if loss_mask is None:
      loss_mask = torch.ones_like(x)

    # Random masking of solution tokens only
    rand_mask = torch.rand(*x.shape, device=x.device) < 1 - alpha_t
    move_indices = rand_mask & (loss_mask == 1)
    xt = torch.where(move_indices, self.mask_index, x)
    return xt

  def _loss(self, x0, valid_tokens,
            current_accumulation_step=None, train_mode=False,
            loss_mask=None):
    """Compute loss with proper normalization matching diffusion-vs-ar.

    Key differences from base _loss:
    1. Passes loss_mask to nll() so only solution tokens are masked
    2. Normalizes by ACTUALLY-MASKED tokens (not all solution tokens)
    3. Optionally applies focal-loss-like token reweighting
    4. Supports linear time reweighting (diffusion-vs-ar style)
    """
    if self.time_reweighting == 'linear':
      # Use discrete time and linear weighting (diffusion-vs-ar style)
      loss, t_mask, time_weight = self.nll_discrete(
        x0, loss_mask=loss_mask, train_mode=train_mode)
    else:
      # Use continuous time with ELBO weighting
      loss = self.nll(x0, None, current_accumulation_step, train_mode, loss_mask=loss_mask)
      t_mask = loss != 0
      time_weight = None
    # loss shape: [B, L], zeroed at non-masked positions

    # Token reweighting (focal loss) - matches diffusion-vs-ar trainer.py:139-140
    # Formula: alpha * (1 - exp(-loss))^gamma * loss
    if self.config.algo.get('token_reweighting', False):
      alpha = self.config.algo.get('focal_alpha', 0.25)
      gamma = self.config.algo.get('focal_gamma', 1.0)
      # Only apply to non-zero (actually masked) positions
      nonzero_mask = loss > 0
      loss = torch.where(
        nonzero_mask,
        alpha * (1 - torch.exp(-loss)) ** gamma * loss,
        loss
      )

    # Apply time weighting if using linear reweighting
    # diffusion-vs-ar trainer.py:150: (loss * weight).sum() / loss_mask.sum()
    if time_weight is not None:
      # time_weight is [B, 1], broadcast to [B, L]
      loss = loss * time_weight

    # Normalize by ACTUALLY-MASKED tokens (not all solution tokens)
    # This matches diffusion-vs-ar trainer.py:150: loss.sum() / loss_mask.sum()
    # where loss_mask is t_mask (tokens masked at this timestep)
    actually_masked = t_mask.float()
    num_tokens = actually_masked.sum()
    nlls = loss.sum()
    token_nll = nlls / num_tokens

    return trainer_base.Loss(
      loss=token_nll,
      nlls=nlls,
      reconstruction_loss=torch.tensor(0.0).to(x0.device),
      num_tokens=num_tokens)

  def nll_discrete(self, x0, loss_mask=None, train_mode=False):
    """Training forward pass with discrete time (diffusion-vs-ar style).

    Uses discrete timesteps t in [0, T-1] and masking probability (t+1)/T.
    Returns raw CE loss without ELBO weighting (for use with linear time reweighting).

    Returns:
      loss: [B, L] per-token CE loss, zeroed at non-masked positions
      t_mask: [B, L] boolean mask of masked positions
      time_weight: [B, 1] linear time weight (T - t)
    """
    batch_size = x0.shape[0]
    T = self.num_diffusion_steps

    # Sample discrete timestep t in [0, T-1]
    t = torch.randint(0, T, (batch_size,), device=x0.device)

    # Create noisy input using discrete masking
    xt, t_mask = self.q_xt_discrete(x0, t, loss_mask=loss_mask)

    # Forward pass (sigma is not used with time_conditioning=False)
    sigma = torch.zeros(batch_size, 1, device=x0.device)
    log_x_theta = self.forward(xt, sigma=sigma, sort_idx=None)
    utils.print_nans(log_x_theta, 'model_output')

    # Compute raw CE loss (no ELBO weighting)
    log_p_theta = log_x_theta.gather(
      dim=-1, index=x0[:, :, None])[:, :, 0]
    # Zero out non-masked positions
    loss = -log_p_theta * t_mask.float()

    # Linear time weight: (T - t), matching diffusion-vs-ar trainer.py:145
    time_weight = (T - t).float()[:, None]  # [B, 1]

    return loss, t_mask, time_weight

  def nll(self, x0, output_tokens,
          current_accumulation_step=None, train_mode=False,
          loss_mask=None):
    """Training forward pass without token shuffling (continuous time).

    Unlike DiffLM, this does NOT reorder tokens by sort_idx.
    Tokens stay in their original sequence order.
    """
    del output_tokens
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      t += (1 / self.T)

    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)

    # Create noisy input (masks some solution tokens)
    xt = self.q_xt(x0, alpha_t, loss_mask=loss_mask)

    # NO sort_idx - process in original order
    log_x_theta = self.forward(xt, sigma=sigma, sort_idx=None)
    utils.print_nans(log_x_theta, 'model_output')

    return self.nll_per_token(
      log_x_theta=log_x_theta,
      xt=xt,
      x0=x0,
      alpha_t=alpha_t,
      dalpha_t=dalpha_t,
      low_var=train_mode and self.loss_type == 'low_var')

  def _tokens_unmasked_per_step(self, num_steps):
    """Compute how many tokens to unmask at each step."""
    tokens_per_step = self.num_tokens // num_steps
    remainder = self.num_tokens % num_steps
    # Distribute remainder across first few steps
    unmask_k_tokens = [tokens_per_step + (1 if i < remainder else 0)
                       for i in range(num_steps)]
    return unmask_k_tokens

  @torch.no_grad()
  def generate_samples(self, num_samples, num_steps=None, return_stats=False):
    """Generate samples without token shuffling.

    Tokens stay in their original positions. Unmasking order is based on
    model confidence (lowest confidence tokens are unmasked first to allow
    refinement, or highest confidence for greedy decoding).
    """
    if num_steps is None:
      num_steps = self.config.sampling.steps

    # Check for dvar-compatible decoding mode
    decoding_strategy = self.config.sampling.get('decoding_strategy', None)
    if decoding_strategy:
      return self._generate_samples_dvar(
        num_samples, num_steps, decoding_strategy, return_stats)

    unmask_k_tokens = self._tokens_unmasked_per_step(num_steps)
    assert sum(unmask_k_tokens) == self.num_tokens

    # Initialize all tokens as masked
    x = self.prior_sample(num_samples, self.num_tokens)

    start = time.perf_counter()

    for k in unmask_k_tokens:
      # Forward pass with NO sort_idx
      log_p_x0 = self.backbone.forward_sample(
        zt=x,
        sort_idx=None,  # No shuffling - tokens stay in original positions
        attn_mode='full',
        cutoffs=None,
        kv_cache=False)

      # Zero out mask token probability
      log_p_x0[:, :, self.mask_index] = self.neg_infinity

      # Find currently masked positions
      is_masked = (x == self.mask_index)  # [B, L]

      # Get max log prob for each position (confidence)
      max_log_p, best_tokens = log_p_x0.max(dim=-1)  # [B, L]

      # Set confidence of non-masked positions to very high (won't be selected)
      confidence = max_log_p.clone()
      confidence[~is_masked] = float('inf')

      # Select k positions with highest confidence among masked positions
      # (most confident predictions get unmasked)
      _, topk_indices = confidence.topk(k, dim=-1, largest=False)  # lowest = most confident after masking trick

      # Actually we want highest confidence, let me fix this
      confidence[~is_masked] = float('-inf')
      _, topk_indices = confidence.topk(k, dim=-1, largest=True)  # [B, k]

      # Unmask selected positions with sampled tokens
      # Use Gumbel-max for sampling instead of argmax for diversity
      for b in range(num_samples):
        positions = topk_indices[b]
        log_p_pos = log_p_x0[b, positions, :]
        noise = torch.distributions.Gumbel(0, 1).sample(log_p_pos.shape).to(self.device)
        sampled_tokens = (log_p_pos + noise).argmax(-1)
        x[b, positions] = sampled_tokens

    end = time.perf_counter()
    duration = end - start
    print(f'Sampling duration: {duration} seconds')

    if return_stats:
      return x, {'duration': duration}
    return x

  @torch.no_grad()
  def generate_completions(self, completion_batch, num_steps=None, return_stats=False):
    """Generate completions without token shuffling.

    Problem tokens (loss_mask=0) are kept as-is, solution tokens (loss_mask=1)
    are masked and unmasked based on model confidence.
    """
    if num_steps is None:
      num_steps = self.config.sampling.steps

    # Check for dvar-compatible decoding mode
    decoding_strategy = self.config.sampling.get('decoding_strategy', None)
    if decoding_strategy:
      return self._generate_completions_dvar(
        completion_batch, num_steps, decoding_strategy, return_stats)

    input_ids = completion_batch['input_ids'].to(self.device)
    loss_mask = completion_batch['loss_mask'].to(self.device).bool()
    num_samples = input_ids.shape[0]

    # Initialize: problem tokens keep their values, solution tokens are masked
    x = torch.where(loss_mask, self.mask_index, input_ids)

    # Count solution tokens (positions to unmask)
    num_solution_tokens = loss_mask.sum(dim=1)  # [batch_size]
    max_solution_tokens = num_solution_tokens.max().item()

    if max_solution_tokens == 0:
      # No solution tokens to generate
      if return_stats:
        return x, {'duration': 0.0}
      return x

    # Compute unmask schedule based on solution token count
    tokens_per_step = max(1, max_solution_tokens // num_steps)
    print(max_solution_tokens, num_steps, tokens_per_step)
    unmask_k_tokens = []
    remaining = max_solution_tokens
    while remaining > 0:
      k = min(tokens_per_step, remaining)
      unmask_k_tokens.append(k)
      remaining -= k
    print(f'unmask_k_tokens: {unmask_k_tokens}')

    # Ablation flags
    greedy_tokens = self.config.sampling.get('greedy_tokens', False)
    stochastic_positions = self.config.sampling.get('stochastic_positions', False)
    position_noise_scale = self.config.sampling.get('position_noise_scale', 0.5)

    start = time.perf_counter()
    total_steps = len(unmask_k_tokens)

    for step_idx, k in enumerate(unmask_k_tokens):
      # Forward pass with NO sort_idx
      log_p_x0 = self.backbone.forward_sample(
        zt=x,
        sort_idx=None,
        attn_mode='full',
        cutoffs=None,
        kv_cache=False)

      log_p_x0[:, :, self.mask_index] = self.neg_infinity

      # Find currently masked solution positions
      is_masked_solution = loss_mask & (x == self.mask_index)  # [B, L]

      # Get max log prob for each position (confidence)
      max_log_p, _ = log_p_x0.max(dim=-1)  # [B, L]

      # Set confidence of non-target positions to -inf (won't be selected)
      confidence = max_log_p.clone()
      confidence[~is_masked_solution] = float('-inf')

      # Ablation: stochastic position selection (like DVAR)
      if stochastic_positions:
        # Compute rate similar to DVAR: decays from 1 to 0 over steps
        rate = 1.0 - (step_idx / total_steps)
        gumbel = -torch.log(-torch.log(torch.rand_like(confidence) + 1e-8) + 1e-8)
        # Only add noise to valid positions (not -inf)
        valid_mask = confidence > float('-inf')
        confidence = torch.where(valid_mask, confidence + position_noise_scale * rate * gumbel, confidence)

      # Select k positions with highest confidence among masked solution positions
      _, topk_indices = confidence.topk(k, dim=-1, largest=True)  # [B, k]

      # Unmask selected positions with sampled tokens
      for b in range(num_samples):
        # Only unmask up to the number of remaining masked positions
        num_masked = is_masked_solution[b].sum().item()
        num_to_unmask = min(k, num_masked)
        if num_to_unmask == 0:
          continue

        positions = topk_indices[b, :num_to_unmask]
        log_p_pos = log_p_x0[b, positions, :]

        # Ablation: greedy vs Gumbel token selection
        if greedy_tokens:
          sampled_tokens = log_p_pos.argmax(-1)
        else:
          noise = torch.distributions.Gumbel(0, 1).sample(log_p_pos.shape).to(self.device)
          sampled_tokens = (log_p_pos + noise).argmax(-1)
        x[b, positions] = sampled_tokens

    end = time.perf_counter()
    duration = end - start
    print(f'Completion duration: {duration} seconds')

    if return_stats:
      return x, {'duration': duration}
    return x

  @torch.no_grad()
  def _generate_completions_dvar(self, completion_batch, num_steps, decoding_strategy, return_stats=False):
    """Generate completions using diffusion-vs-ar compatible decoding.

    This method matches the exact decoding algorithm from the diffusion-vs-ar codebase:
    1. Greedy argmax for token selection (not Gumbel sampling)
    2. Re-masking paradigm: predict all tokens, then re-mask low-confidence ones
    3. Stochastic position selection with decaying noise

    Args:
      completion_batch: dict with 'input_ids' and 'loss_mask'
      num_steps: number of diffusion steps (typically 20 for dvar)
      decoding_strategy: string like "stochastic0.5-linear" or "deterministic-linear"
      return_stats: whether to return timing statistics
    """
    import math

    # Parse decoding strategy: "stochastic0.5-linear" → (0.5, "linear")
    mode, schedule = decoding_strategy.split("-")
    if mode.startswith("stochastic"):
      noise_scale = float(mode.replace("stochastic", ""))
      stochastic = True
    else:
      noise_scale = 0.0
      stochastic = False

    # Ablation flags (can override decoding_strategy settings)
    use_greedy_tokens = self.config.sampling.get('greedy_tokens', True)  # DVAR default: True
    use_stochastic_positions = self.config.sampling.get('stochastic_positions', stochastic)

    input_ids = completion_batch['input_ids'].to(self.device)
    loss_mask = completion_batch['loss_mask'].to(self.device).bool()

    # Initial state: mask all solution positions (maskable positions)
    x_t = torch.where(loss_mask, self.mask_index, input_ids)
    init_maskable_mask = loss_mask.clone()

    start = time.perf_counter()

    for t in range(num_steps - 1, -1, -1):
      # Forward pass
      log_p_x0 = self.backbone.forward_sample(
        zt=x_t,
        sort_idx=None,
        attn_mode='full',
        cutoffs=None,
        kv_cache=False)

      log_p_x0[:, :, self.mask_index] = self.neg_infinity
      log_scores = log_p_x0.log_softmax(dim=-1)

      # Token selection: greedy (DVAR default) or Gumbel sampling (ablation)
      if use_greedy_tokens:
        x0_scores, x0 = log_scores.max(dim=-1)
      else:
        # Gumbel sampling for tokens
        gumbel_tokens = torch.distributions.Gumbel(0, 1).sample(log_scores.shape).to(self.device)
        x0 = (log_scores + gumbel_tokens).argmax(dim=-1)
        x0_scores, _ = log_scores.max(dim=-1)  # Still use max for confidence

      # Keep non-masked positions unchanged (problem tokens stay as-is)
      x0 = torch.where(x_t == self.mask_index, x0, x_t)

      if t > 0:
        # Compute mask rate based on schedule
        if schedule == "linear":
          rate = t / num_steps
        elif schedule == "cosine":
          rate = math.cos((num_steps - t) / num_steps * math.pi * 0.5)
        else:
          raise ValueError(f"Unknown schedule: {schedule}")

        # Compute cutoff: how many tokens to re-mask
        cutoff_len = (init_maskable_mask.sum(1, keepdim=True).float() * rate).long()

        # Set non-maskable scores high so they won't be selected for remasking
        scores_for_topk = x0_scores.clone()
        scores_for_topk[~init_maskable_mask] = 1000.0

        # Stochastic selection with decaying noise (can be disabled via ablation)
        if use_stochastic_positions and noise_scale > 0:
          gumbel = -torch.log(-torch.log(torch.rand_like(scores_for_topk) + 1e-8) + 1e-8)
          scores_for_topk = scores_for_topk + (noise_scale * rate) * gumbel

        # Select positions to re-mask (lowest confidence)
        lowest_k_mask = self._topk_masking(scores_for_topk, cutoff_len)
        x_t = torch.where(lowest_k_mask, self.mask_index, x0)
      else:
        x_t = x0

    end = time.perf_counter()
    duration = end - start
    print(f'Completion duration (dvar mode): {duration} seconds')

    if return_stats:
      return x_t, {'duration': duration}
    return x_t

  def _topk_masking(self, scores, cutoff_len):
    """Select positions with lowest scores up to cutoff_len.

    Matches diffusion-vs-ar's topk_masking function.

    Args:
      scores: [B, L] confidence scores (already noisy if stochastic)
      cutoff_len: [B, 1] number of positions to mask per sample

    Returns:
      mask: [B, L] True for positions to re-mask
    """
    # Clamp cutoff_len to valid range
    cutoff_len = cutoff_len.clamp(min=0, max=scores.shape[1] - 1)
    sorted_scores = scores.sort(dim=-1)[0]
    cutoff = sorted_scores.gather(dim=-1, index=cutoff_len)
    return scores < cutoff

  @torch.no_grad()
  def _generate_samples_dvar(self, num_samples, num_steps, decoding_strategy, return_stats=False):
    """Generate samples using diffusion-vs-ar compatible decoding.

    This method matches the exact decoding algorithm from the diffusion-vs-ar codebase
    for unconditional generation (all tokens are maskable).

    Args:
      num_samples: number of samples to generate
      num_steps: number of diffusion steps (typically 20 for dvar)
      decoding_strategy: string like "stochastic0.5-linear" or "deterministic-linear"
      return_stats: whether to return timing statistics
    """
    import math

    # Parse decoding strategy: "stochastic0.5-linear" → (0.5, "linear")
    mode, schedule = decoding_strategy.split("-")
    if mode.startswith("stochastic"):
      noise_scale = float(mode.replace("stochastic", ""))
      stochastic = True
    else:
      noise_scale = 0.0
      stochastic = False

    # Initialize all tokens as masked
    x_t = self.prior_sample(num_samples, self.num_tokens)
    init_maskable_mask = torch.ones_like(x_t, dtype=torch.bool)

    start = time.perf_counter()

    for t in range(num_steps - 1, -1, -1):
      # Forward pass
      log_p_x0 = self.backbone.forward_sample(
        zt=x_t,
        sort_idx=None,
        attn_mode='full',
        cutoffs=None,
        kv_cache=False)

      log_p_x0[:, :, self.mask_index] = self.neg_infinity
      log_scores = log_p_x0.log_softmax(dim=-1)

      # Greedy token selection (key difference from esolm's Gumbel sampling)
      x0_scores, x0 = log_scores.max(dim=-1)

      # Keep non-masked positions unchanged
      x0 = torch.where(x_t == self.mask_index, x0, x_t)

      if t > 0:
        # Compute mask rate based on schedule
        if schedule == "linear":
          rate = t / num_steps
        elif schedule == "cosine":
          rate = math.cos((num_steps - t) / num_steps * math.pi * 0.5)
        else:
          raise ValueError(f"Unknown schedule: {schedule}")

        # Compute cutoff: how many tokens to re-mask
        cutoff_len = (init_maskable_mask.sum(1, keepdim=True).float() * rate).long()

        # All tokens are maskable in unconditional generation
        scores_for_topk = x0_scores.clone()

        # Stochastic selection with decaying noise
        if stochastic and noise_scale > 0:
          gumbel = -torch.log(-torch.log(torch.rand_like(scores_for_topk) + 1e-8) + 1e-8)
          scores_for_topk = scores_for_topk + (noise_scale * rate) * gumbel

        # Select positions to re-mask (lowest confidence)
        lowest_k_mask = self._topk_masking(scores_for_topk, cutoff_len)
        x_t = torch.where(lowest_k_mask, self.mask_index, x0)
      else:
        x_t = x0

    end = time.perf_counter()
    duration = end - start
    print(f'Sampling duration (dvar mode): {duration} seconds')

    if return_stats:
      return x_t, {'duration': duration}
    return x_t

