import time

import torch

import trainer_base
import utils


class DiffuParallel(trainer_base.TrainerBase):
  def __init__(self, config, tokenizer):
    vocab_size = tokenizer.vocab_size
    if (not hasattr(tokenizer, 'mask_token')
        or tokenizer.mask_token is None):
      self.mask_index = vocab_size
      vocab_size += 1
    else:
      self.mask_index = tokenizer.mask_token_id
    super().__init__(config, tokenizer,
                     vocab_size=vocab_size)

    self.loss_type = config.algo.loss_type
    self.save_hyperparameters()
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert not self.config.algo.time_conditioning
    assert self.config.prior.type == 'none'

  def _process_model_input(self, x0, valid_tokens):
    # TODO
    input_tokens = x0[:, :-1]
    output_tokens = x0[:, 1:]
    valid_tokens = valid_tokens[:, 1:]
    return input_tokens, output_tokens, valid_tokens

  def nll(self, input_tokens, output_tokens,
          current_accumulation_step, train_mode):
    del train_mode, current_accumulation_step
    
    output = self.backbone(input_tokens, None)
    output[:, :, self.mask_index] = self.neg_infinity
    output = output.log_softmax(-1)
    nll = - output.gather(
      -1, output_tokens[:, :, None])[:, :, 0]
    return nll

  def _shuffle_indices(self, indices, shuffle_alpha: float):
    #shuffle_alpha: 0.0 means no shuffling, 1.0 means full shuffling
    B, L = indices.shape
    device = indices.device
    base = torch.linspace(0, 1.0, L, device=device).unsqueeze(0).expand(B, -1)
    noise = torch.rand(B, L, device=device)
    offsets = torch.lerp(base, noise, shuffle_alpha)
    offsets[:, 0] = -0.1
    sort_idx = offsets.argsort(dim=1)
    return sort_idx

  def _get_shuffle_alpha(self, train_mode):
    if not self.config.algo.diffusion_shuffle:
      return 0.0
    elif not train_mode:
      return 1.0
    else:
      shuffle_warmup = self.config.algo.get('shuffle_warmup', 0)
      if shuffle_warmup > 0:
        return min(1.0, self.trainer.global_step / shuffle_warmup)
      else:
        return 1.0

  def _loss(self, x0, valid_tokens,
            current_accumulation_step=None,
            train_mode=False,
            loss_mask=None):

    shuffle_alpha = self._get_shuffle_alpha(train_mode)
    
    # Log shuffle_alpha during training for monitoring
    if train_mode:
      self.log('trainer/shuffle_alpha', shuffle_alpha, on_step=True, on_epoch=False)

    sort_idx = self._shuffle_indices(x0, shuffle_alpha=shuffle_alpha)
    x0 = torch.gather(x0, dim=1, index=sort_idx)
    valid_tokens = torch.gather(valid_tokens, dim=1, index=sort_idx)
    if loss_mask is not None:
      loss_mask = torch.gather(loss_mask, dim=1, index=sort_idx)
    seq_len = x0.shape[1]

    # TODO maybe it would be easier to add an extra BOS
    # so that we don't need to provide rotary_range

    input_tokens = x0[:, :-1]
    output_tokens = x0[:, 1:]
    sort_idx_source = sort_idx[:, :-1]  # Position of each input token
    sort_idx_target = sort_idx[:, 1:]   # Position to predict
    valid_tokens = valid_tokens[:, 1:]
    if loss_mask is not None:
      loss_mask = loss_mask[:, 1:]

    with torch.amp.autocast('cuda', dtype=torch.float32):
      # assuming esolm_dit
      model_output = self.backbone(
        input_tokens, None, 
        sort_idx_source=sort_idx_source,
        sort_idx_target=sort_idx_target,
        rotary_range=seq_len)

    model_output[:, :, self.mask_index] = self.neg_infinity
    model_output = model_output.log_softmax(-1)
    log_p_theta = model_output.gather(
      dim=-1,
      index=output_tokens[:, :, None])[:, :, 0]

    # Apply next_token_prediction masking if enabled
    # Keep only a single random position per sequence and scale by num_tokens
    next_token_prediction = self.config.algo.get('next_token_prediction', False)
    if next_token_prediction and train_mode:
      batch_size, seq_len_minus_1 = log_p_theta.shape
      # Sample a random position for each sequence (uniformly across valid positions)
      # Note: valid_tokens marks which positions are valid for loss computation
      # We sample uniformly from [0, seq_len_minus_1) and then mask
      random_positions = torch.randint(
        0, seq_len_minus_1, (batch_size,), device=log_p_theta.device)
      # Create a mask that is 1 only at the random position for each sequence
      position_mask = torch.zeros_like(log_p_theta)
      position_mask.scatter_(1, random_positions.unsqueeze(1), 1.0)
      # Apply mask and scale by num_tokens to keep expected loss scale same
      log_p_theta = log_p_theta * position_mask * self.num_tokens

    if not train_mode or self.loss_type == 'elbo':
      loss = -log_p_theta
    elif self.loss_type == 'low_var':
      loss = -log_p_theta
    else:
      raise ValueError(f"Invalid loss type: {self.loss_type}")

    # Combine valid_tokens with loss_mask if provided
    if loss_mask is not None:
      effective_mask = valid_tokens * loss_mask
    else:
      effective_mask = valid_tokens

    loss_no_reduce = loss.clone().detach()
    loss = (
      loss * effective_mask).sum()
    num_tokens = effective_mask.sum()
    loss_per_token = loss / num_tokens
    
    return trainer_base.Loss(
        loss=loss_per_token,
        nlls=loss_per_token * num_tokens,
        reconstruction_loss=torch.tensor(0.0).to(x0.device),
        num_tokens=num_tokens)

  @torch.no_grad()
  def generate_samples(self, num_samples, return_stats=False, **kwargs):
    """
    Generate samples autoregressively in a permuted order.
    
    At step i (0-indexed), we have generated tokens 0..i (total i+1 tokens).
    - Input: x[:, :i+1] (the tokens generated so far)
    - sort_idx_source[:i+1]: source positions for each input token
    - sort_idx_target[1:i+2]: target positions (what to predict at each step)
    - We take the prediction from the last position to get token i+1
    """
    num_pred_tokens = self.num_tokens - 1
    x = torch.zeros(
      (num_samples, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    seq_len = x.shape[1]

    # Create sort_idx - determines the order of generation
    # sort_idx[0] = 0 always (BOS stays first)
    # sort_idx[1:] is the permuted order for the rest
    shuffle_alpha = self._get_shuffle_alpha(train_mode=False)
    sort_idx = self._shuffle_indices(x, shuffle_alpha=shuffle_alpha)

    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((num_samples, num_pred_tokens, self.vocab_size))
             .to(self.device))
    if self.config.sampling.use_float64:
      noise = noise.to(torch.float64)
    kv_cache = self.config.sampling.get('kv_cache', False)
    assert not kv_cache, "kv cache not supported yet for DiffuParallel"
    self.backbone.reset_kv_cache()
    
    start = time.perf_counter()
    for i in range(num_pred_tokens):
      # At step i, we have i+1 tokens (indices 0..i)
      # We want to predict the token at position i+1 (in permuted order)
      
      # Source positions: where each input token comes from
      # For inputs 0..i, source positions are sort_idx[0..i]
      curr_sort_idx_source = sort_idx[:, :i+1]
      
      # Target positions: what position to predict at each step
      # For inputs 0..i, target positions are sort_idx[1..i+1]
      curr_sort_idx_target = sort_idx[:, 1:i+2]
      
      output = self.backbone(
        x[:, :i + 1], None, 
        sort_idx_source=curr_sort_idx_source, 
        sort_idx_target=curr_sort_idx_target, 
        rotary_range=seq_len)
      output[:, :, self.mask_index] = self.neg_infinity
      output = output.log_softmax(-1)
      
      # Take prediction from the last position
      y = (output[:, -1, :] + noise[:, i, :]).argmax(-1)
      x[:, i + 1] = y
    self.backbone.reset_kv_cache()

    # Reverse the permutation to get tokens in original order
    sort_idx_reversed = utils.get_reverse_indices(sort_idx)
    x = torch.gather(x, dim=1, index=sort_idx_reversed)
    
    end = time.perf_counter()
    duration = end - start
    print(f'Sampling duration: {duration} seconds')
    
    if return_stats:
      return x, {'duration': duration}
    else:
      return x

