import time

import torch

import trainer_base


class AR(trainer_base.TrainerBase):
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
    self.save_hyperparameters()
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert not self.config.algo.time_conditioning
    assert self.config.prior.type == 'none'

  def _process_model_input(self, x0, valid_tokens):
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

  @torch.no_grad()
  def generate_samples(self, num_samples, return_stats=False, **kwargs):
    # precompute token buffer
    num_pred_tokens = self.num_tokens - 1
    x = torch.zeros(
      (num_samples, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((num_samples, num_pred_tokens, self.vocab_size))
             .to(self.device))
    if self.config.sampling.use_float64:
      noise = noise.to(torch.float64)
    kv_cache = self.config.sampling.kv_cache
    self.backbone.reset_kv_cache()
    
    start = time.perf_counter()
    for i in range(num_pred_tokens):
      output = self.backbone(
        x[:, :i + 1], sigma=None, x0=None, kv_cache=kv_cache)
      output[:, :, self.mask_index] = self.neg_infinity
      output = output.log_softmax(-1)
      y = (output[:, -1, :] + noise[:, i, :]).argmax(-1)
      x[:, i + 1] = y
    self.backbone.reset_kv_cache()
    
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
    
    For each position, if the token is part of the problem (loss_mask == 0),
    use the ground truth token. Otherwise, sample from the model.
    This handles variable-length problems naturally - each sample transitions
    from "copying problem tokens" to "generating solution tokens" at its own pace.
    
    Args:
      completion_batch: dict with 'input_ids' and 'loss_mask'
        - input_ids: (batch_size, seq_len) problem + solution tokens
        - loss_mask: (batch_size, seq_len) 0 for problem, 1 for solution
      num_steps: ignored for AR (kept for API consistency with diffusion models)
      return_stats: whether to return timing statistics
    
    Returns:
      x: (batch_size, seq_len) generated sequences
      stats: dict with 'duration' (if return_stats=True)
    """
    del num_steps  # AR generates token-by-token, doesn't use num_steps
    input_ids = completion_batch['input_ids'].to(self.device)
    is_solution = completion_batch['loss_mask'].to(self.device).bool()
    
    batch_size, seq_len = input_ids.shape
    num_pred_tokens = seq_len - 1
    
    # Initialize with input_ids - problem tokens are already in place
    x = input_ids.clone()
    
    # Precompute noise for all positions
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((batch_size, num_pred_tokens, self.vocab_size))
             .to(self.device))
    if self.config.sampling.use_float64:
      noise = noise.to(torch.float64)
    
    kv_cache = self.config.sampling.kv_cache
    self.backbone.reset_kv_cache()
    
    start = time.perf_counter()
    
    for i in range(num_pred_tokens):
      output = self.backbone(
        x[:, :i + 1], sigma=None, x0=None, kv_cache=kv_cache)
      output[:, :, self.mask_index] = self.neg_infinity
      output = output.log_softmax(-1)
      
      # Sample next token
      sampled = (output[:, -1, :] + noise[:, i, :]).argmax(-1)
      
      # For positions where loss_mask == 0 (problem tokens), use ground truth
      # For positions where loss_mask == 1 (solution tokens), use sampled
      # loss_mask[:, i+1] tells us if position i+1 is a solution token
      x[:, i + 1] = torch.where(is_solution[:, i + 1], sampled, input_ids[:, i + 1])
    
    self.backbone.reset_kv_cache()
    
    end = time.perf_counter()
    duration = end - start
    print(f'Completion duration: {duration} seconds')
    
    if return_stats:
      return x, {'duration': duration}
    else:
      return x

  def _process_sigma(self, sigma):
    del sigma
    return None

  def _loss(self, x0, valid_tokens,
            current_accumulation_step=None,
            train_mode=False,
            loss_mask=None):
    # Process inputs (shift by 1 for AR: input is x[:-1], target is x[1:])
    input_tokens, output_tokens, valid_tokens = self._process_model_input(
      x0, valid_tokens)
    
    # Get per-token NLL
    loss = self.nll(input_tokens, output_tokens,
                    current_accumulation_step, train_mode)
    assert loss.ndim == 2
    
    # Apply loss_mask if provided (shift by 1 same as valid_tokens)
    if loss_mask is not None:
      _, _, loss_mask = self._process_model_input(x0, loss_mask)
      effective_mask = valid_tokens * loss_mask
    else:
      effective_mask = valid_tokens
    
    nlls = (loss * effective_mask).sum()
    num_tokens = effective_mask.sum()
    token_nll = nlls / num_tokens

    return trainer_base.Loss(
        loss=token_nll,
        nlls=nlls,
        reconstruction_loss=torch.tensor(0).to(x0.device),
        num_tokens=num_tokens)

