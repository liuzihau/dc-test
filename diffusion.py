import itertools
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from einops import rearrange
from tqdm import tqdm
from collections import OrderedDict

import dataloader
import metrics
import models
import noise_schedule
from rollout_utils import build_rollout_mask_counts
import utils
import numpy as np
import itertools

def _sample_categorical(categorical_probs):
  gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
  samples = (categorical_probs / gumbel_norm).argmax(dim=-1)
  return samples

def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.tokenizer = tokenizer
    self.vocab_size = self.tokenizer.vocab_size
    self.sampler = self.config.algo.sampler
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.cross_attn = self.config.algo.cross_attn
    self.ignore_bos = self.config.algo.ignore_bos
    self.mdlm_loss_scale = self.config.algo.mdlm_loss_scale
    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
    if hasattr(self.config, 'algo'):
      self.parameterization = self.config.algo.parameterization
    else:
      self.parameterization = self.config.parameterization
    if hasattr(self.config, 'block_size'):
      self.block_size = self.config.block_size
    else:
      self.block_size = self.config.model.length
    if self.parameterization == 'ar':
      self.block_size = 1
    if self.config.algo.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.algo.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.algo.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
      #  egenerate mask if pretrained model uses flex attention mask
      # and current model uses sdpa mask
      if getattr(self.backbone.config, 'attn_backend', None) == 'flex' and \
        self.config.model.attn_backend == 'sdpa':
        self.backbone.config.attn_backend = 'sdpa'
        for i in self.backbone.backbone.blocks:
          i.attn_backend = 'sdpa'
        self.backbone.backbone.gen_mask(self.config.model.length, self.block_size, attn_backend='sdpa')
    else:
      raise ValueError(f'Unknown backbone: {self.config.algo.backbone}')

    self.T = self.config.algo.T
    self.num_tokens = self.config.model.length

    self.noise = noise_schedule.get_noise(self.config)
    self.metrics = metrics.Metrics(config)

    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        self._get_parameters(),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.var_min = self.config.algo.var_min
    if self.var_min:
      self.register_buffer('sampling_eps_min', torch.tensor(
        self.config.training.sampling_eps_min))
      self.register_buffer('sampling_eps_max', torch.tensor(
        self.config.training.sampling_eps_max))
      
    self.time_conditioning = self.config.algo.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()

  def _get_parameters(self):
    parameters = [self.backbone.parameters(),
                  self.noise.parameters()]
    return itertools.chain(* parameters)

  def on_validation_model_zero_grad(self) -> None:
    '''
    Small hack to avoid first validation on resume. 
    This will NOT work if the gradient accumulation step should be performed at this point.
    '''
    super().on_validation_model_zero_grad()
    if self.trainer.ckpt_path is not None and getattr(self, '_restarting_skip_val_flag', True):
        self.trainer.sanity_checking = True
        self._restarting_skip_val_flag = False

  def _validate_configuration(self):
    if self.config.mode == 'sample_eval' and \
        self.config.sampling.first_hitting:
      assert self.config.loader.eval_batch_size == 1
    assert self.config.algo.backbone in {
      'dit', 'ar', 'hf_dit'}
    if self.config.algo.parameterization == 'ar':
      assert not self.config.algo.time_conditioning
    if self.config.sampling.kv_cache:
      assert self.config.algo.name in {'ar', 'bd3lm'}

    rollout_config = getattr(
      getattr(self.config, 'step_memory', {}), 'rollout', {})
    pretrain_config = getattr(
      getattr(self.config, 'step_memory', {}), 'pretrain', {})
    if bool(getattr(pretrain_config, 'enabled', False)):
      assert self.config.step_memory.enabled
      assert self.config.step_memory.use_previous_kv
      assert self.config.algo.name == 'mdlm'
      assert self.config.algo.backbone == 'dit'
      assert self.block_size == self.config.model.length
      assert not self.config.algo.cross_attn
      assert self.config.noise.type == 'loglinear'
      assert not self.config.sampling.kv_cache
      assert not bool(getattr(rollout_config, 'enabled', False))
      assert float(pretrain_config.teacher_token_probability) == 1.0, (
        'The first shifted-DCache pretraining trial is fully teacher forced')
    if bool(getattr(rollout_config, 'enabled', False)):
      assert self.config.step_memory.enabled
      assert self.config.algo.name == 'bd3lm'
      assert self.config.algo.backbone == 'dit'
      assert self.block_size > 1
      assert not self.config.sampling.kv_cache, (
        'Training rollout builds its clean prefix directly and must not use '
        'the inference completed-prefix cache')
      
    if self.parameterization in {'sedd'}:
      assert self.time_conditioning
    
    if self.config.mode == 'sample_eval':
      assert self.config.model.attn_backend != 'flex', 'FlexAttention mask not supported at inference.'
    if self.config.model.attn_backend == 'flex':
      assert self.config.algo.name == 'bd3lm', 'Custom FlexAttention mask only supported for BD3LM.'
      
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs) 
    self.metrics.to(*args, **kwargs)
    if hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'sdpa':
      self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(*args, **kwargs)
    elif hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'flex':
      self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(self.device)
    if hasattr(self, 'sampling_eps_min') and torch.is_tensor(self.sampling_eps_min):
      self.sampling_eps_min = self.sampling_eps_min.to(*args, **kwargs)
      self.sampling_eps_max = self.sampling_eps_max.to(*args, **kwargs)
    return self

  def _replace_ckpt_keys(self, checkpoint):
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
      new_state_dict[k.replace('_orig_mod.', '')] = v
    checkpoint['state_dict'] = new_state_dict
    return checkpoint

  def on_load_checkpoint(self, checkpoint):
    print('Loading checkpoint at', checkpoint['global_step'])
    self._restarting_skip_val_flag = True

    # for models compiled with `torch.compile`
    if '_orig_mod.' in list(checkpoint['state_dict'].keys())[0]:
      checkpoint = self._replace_ckpt_keys(checkpoint)

    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
      current_parameters = [
        (name, parameter) for name, parameter in self.named_parameters()
        if parameter.requires_grad]
      if len(self.ema.shadow_params) != len(current_parameters):
        current_parameter_names = {name for name, _ in current_parameters}
        old_parameter_names = [
          name for name in checkpoint['state_dict']
          if name in current_parameter_names
          or name.endswith('step_memory_gate')]
        if len(old_parameter_names) != len(self.ema.shadow_params):
          raise RuntimeError(
            'Cannot map checkpoint EMA parameters onto the current model')
        old_ema_by_name = dict(zip(
          old_parameter_names, self.ema.shadow_params))
        self.ema.shadow_params = [
          old_ema_by_name.get(name, parameter.detach().clone())
          for name, parameter in current_parameters]
    if 'sampling_eps_min' in checkpoint.keys():
      self.sampling_eps_min = checkpoint['sampling_eps_min']
      self.sampling_eps_max = checkpoint['sampling_eps_max']
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    if hasattr(self, 'sampling_eps_min'):
      checkpoint['sampling_eps_min'] = self.sampling_eps_min
      checkpoint['sampling_eps_max'] = self.sampling_eps_max
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(self._get_parameters())

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    # cause of overfitting for block size 1?
    if self.parameterization == 'ar':
      return None
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma, sample_mode=False, store_kv=False,
              previous_step_kv=None, return_step_kv=False,
              detach_cache_backbone=False):
    """Returns log score."""
    sigma = self._process_sigma(sigma)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      if self.config.algo.name in {'bd3lm', 'mdlm'}:
        if self.config.algo.backbone == 'hf_dit':
          if previous_step_kv is not None or return_step_kv:
            raise NotImplementedError(
              'step_memory is currently implemented for the native dit backbone')
          if self.config.algo.name == 'bd3lm':
            backbone_output = self.backbone(
              x, sigma, store_kv=store_kv, sample_mode=sample_mode)
          else:
            backbone_output = self.backbone(x, sigma)
        else:
          backbone_output = self.backbone(
            x, sigma,
            store_kv=store_kv,
            sample_mode=sample_mode,
            previous_step_kv=previous_step_kv,
            return_step_kv=return_step_kv,
            detach_cache_backbone=detach_cache_backbone)
        if return_step_kv:
          logits, next_step_kv = backbone_output
        else:
          logits = backbone_output
      elif self.config.algo.name == 'ar':
        if self.config.algo.backbone == 'hf_dit':
          logits = self.backbone(x, None)     
        else:
          logits = self.backbone(x, sigma, sample_mode=sample_mode, store_kv=store_kv)
        logits[:, :, self.mask_index] = self.neg_infinity
        logits = logits.log_softmax(-1)
      else:
        logits = self.backbone(x, sigma)

    if self.cross_attn:
      x = x[:, :self.config.model.length]
    if self.parameterization == 'subs':
      scores = self._subs_parameterization(logits=logits, xt=x)
    elif self.parameterization == 'sedd':
      scores = self._sedd_parameterization(logits=logits, xt=x, sigma=sigma)
    else:
      scores = logits
    if return_step_kv:
      return scores, next_step_kv
    return scores

  def _rollout_mask_counts(self):
    """Return a strictly decreasing mask-count trajectory for this step."""
    config = self.config.step_memory.rollout
    curriculum_steps = max(1, int(config.curriculum_steps))
    progress = min(float(self.global_step) / curriculum_steps, 1.0)

    forwards_float = (
      float(config.forwards_start)
      + progress * (float(config.forwards_end) - float(config.forwards_start)))
    num_forwards = int(round(forwards_float))
    num_forwards = max(2, min(num_forwards, self.block_size))

    final_ratio = (
      float(config.final_mask_ratio_start)
      + progress * (
        float(config.final_mask_ratio_end)
        - float(config.final_mask_ratio_start)))
    final_mask_count = int(round(final_ratio * self.block_size))
    return build_rollout_mask_counts(
      self.block_size, num_forwards, final_mask_count, self.device)

  @torch.no_grad()
  def _rollout_transition(
      self, state, target, model_log_probs, reveal_count):
    """Reveal uniformly selected masks using the configured 85/15 mixture."""
    masked = state.eq(self.mask_index)
    random_scores = torch.rand(state.shape, device=state.device)
    random_scores = random_scores.masked_fill(~masked, -1.0)
    selected_positions = random_scores.topk(
      reveal_count, dim=-1).indices
    reveal_mask = torch.zeros_like(masked)
    reveal_mask.scatter_(1, selected_positions, True)

    selected_log_probs = model_log_probs[reveal_mask]
    probabilities = selected_log_probs.exp()
    nucleus_p = float(self.config.step_memory.rollout.nucleus_p)
    if nucleus_p < 1.0:
      sorted_probs, sorted_indices = probabilities.sort(
        dim=-1, descending=True)
      keep = sorted_probs.cumsum(dim=-1) <= nucleus_p
      keep[:, 0] = True
      sorted_probs = sorted_probs * keep
      probabilities.zero_().scatter_(-1, sorted_indices, sorted_probs)
      probabilities /= probabilities.sum(dim=-1, keepdim=True)
    sampled_tokens = torch.multinomial(probabilities, num_samples=1).squeeze(-1)

    teacher_probability = float(
      self.config.step_memory.rollout.teacher_token_probability)
    use_teacher = torch.rand(
      sampled_tokens.shape, device=state.device) < teacher_probability
    revealed_tokens = torch.where(
      use_teacher, target[reveal_mask], sampled_tokens)

    next_state = state.clone()
    next_state[reveal_mask] = revealed_tokens
    return next_state

  def _step_memory_rollout_loss(self, x0, attention_mask):
    """Auxiliary recurrent rollout on one clean-prefix/active-block pair."""
    x0, _, attention_mask = self._maybe_sub_sample(x0, attention_mask)
    num_blocks = x0.shape[1] // self.block_size
    if num_blocks < 2:
      raise ValueError('Step-memory rollout requires a non-initial target block')

    block_attention = rearrange(
      attention_mask[:, :num_blocks * self.block_size],
      'b (g s) -> b g s', s=self.block_size)
    valid_blocks = block_attention.bool().all(dim=-1).all(dim=0)
    valid_blocks[0] = False
    candidate_blocks = valid_blocks.nonzero(as_tuple=False).flatten()
    if candidate_blocks.numel() == 0:
      raise ValueError('No fully valid non-initial block for step-memory rollout')
    target_block_index = candidate_blocks[
      torch.randint(candidate_blocks.numel(), (), device=x0.device)].item()
    start = target_block_index * self.block_size
    end = start + self.block_size

    clean_prefix = x0[:, :start]
    target = x0[:, start:end]
    target_attention = attention_mask[:, start:end].bool()
    state = torch.full_like(target, self.mask_index)
    previous_step_kv = None
    losses = []
    mask_counts = self._rollout_mask_counts()
    use_previous_kv = bool(getattr(
      self.config.step_memory, 'use_previous_kv', True))
    detach_between_steps = bool(getattr(
      self.config.step_memory, 'detach_between_steps', True))

    for forward_index, mask_count in enumerate(mask_counts):
      actual_mask_count = int(state[0].eq(self.mask_index).sum().item())
      if actual_mask_count != mask_count:
        raise RuntimeError(
          f'Rollout expected {mask_count} masks, found {actual_mask_count}')

      masked_ratio = state.eq(self.mask_index).float().mean(dim=-1, keepdim=True)
      sigma = self._sigma_from_p(masked_ratio)
      model_input = torch.cat((clean_prefix, state), dim=-1)
      model_output, current_step_kv = self.forward(
        model_input,
        sigma=sigma,
        sample_mode=True,
        previous_step_kv=(previous_step_kv if use_previous_kv else None),
        return_step_kv=True,
        detach_cache_backbone=detach_between_steps)
      active_log_probs = model_output[:, -self.block_size:]

      remaining = state.eq(self.mask_index) & target_attention
      target_log_probs = torch.gather(
        active_log_probs, -1, target[:, :, None]).squeeze(-1)
      losses.append(
        -(target_log_probs * remaining).sum() / remaining.sum().clamp_min(1))

      if forward_index + 1 == len(mask_counts):
        break

      reveal_count = mask_count - mask_counts[forward_index + 1]
      state = self._rollout_transition(
        state, target, active_log_probs, reveal_count)
      previous_step_kv = current_step_kv

    rollout_loss = torch.stack(losses).mean()
    metrics_out = {
      'num_forwards': torch.tensor(
        float(len(mask_counts)), device=x0.device),
      'final_mask_count': torch.tensor(
        float(mask_counts[-1]), device=x0.device),
    }
    return rollout_loss, metrics_out

  def _ensure_nested_transition(self, x0, attention_mask, s_time):
    """Build teacher-forced nested states x_s and x_t with 1 <= masks_t < masks_s.

    The s state follows the ordinary MDLM Bernoulli corruption except for the
    rare sample with fewer than two maskable positions, where two positions are
    forced masked so that a valid recurrent transition exists. Given t < s,
    each s-mask remains masked with probability t/s; boundary corrections then
    guarantee at least one reveal and at least one remaining prediction target.
    """
    eligible = attention_mask.bool().clone()
    if self.ignore_bos:
      eligible[:, 0] = False
    if (eligible.sum(dim=-1) < 2).any():
      raise ValueError(
        'Shifted-DCache pretraining needs at least two maskable tokens')

    _, s_probability = self.noise(s_time)
    s_mask = (torch.rand_like(x0, dtype=torch.float32)
              <= s_probability) & eligible
    for batch_index in range(x0.shape[0]):
      if int(s_mask[batch_index].sum()) < 2:
        candidates = eligible[batch_index].nonzero(
          as_tuple=False).flatten()
        chosen = candidates[torch.randperm(
          candidates.numel(), device=x0.device)[:2]]
        s_mask[batch_index, chosen] = True

    min_ratio = float(self.config.step_memory.pretrain.min_mask_ratio)
    random_fraction = torch.rand_like(s_time)
    t_time = min_ratio + random_fraction * (s_time - min_ratio)
    t_time = torch.minimum(t_time, s_time)
    _, t_probability = self.noise(t_time)
    remain_probability = (t_probability / s_probability).clamp(0.0, 1.0)
    t_mask = s_mask & (
      torch.rand_like(x0, dtype=torch.float32) <= remain_probability)

    for batch_index in range(x0.shape[0]):
      s_positions = s_mask[batch_index].nonzero(
        as_tuple=False).flatten()
      t_count = int(t_mask[batch_index].sum())
      if t_count == 0:
        keep = s_positions[torch.randint(
          s_positions.numel(), (), device=x0.device)]
        t_mask[batch_index, keep] = True
      elif t_count == s_positions.numel():
        reveal = s_positions[torch.randint(
          s_positions.numel(), (), device=x0.device)]
        t_mask[batch_index, reveal] = False

    x_s = torch.where(s_mask, self.mask_index, x0)
    x_t = torch.where(t_mask, self.mask_index, x0)
    return x_s, x_t, t_time, s_mask, t_mask

  def _recurrent_pretrain_state_loss(
      self, x0, state, attention_mask, time, previous_step_kv,
      return_step_kv):
    """Evaluate one explicitly constructed MDLM state and optional cache."""
    loss_scale, probability = self.noise(time)
    sigma = self._sigma_from_p(probability[:, 0].unsqueeze(-1))
    output = self.forward(
      state,
      sigma=sigma,
      sample_mode=True,
      previous_step_kv=previous_step_kv,
      return_step_kv=return_step_kv,
      detach_cache_backbone=bool(
        self.config.step_memory.detach_between_steps))
    if return_step_kv:
      model_output, next_step_kv = output
    else:
      model_output = output
      next_step_kv = None
    target_log_probability = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    nlls = loss_scale * target_log_probability * attention_mask
    token_nll = nlls.sum() / attention_mask.sum()
    return Loss(
      loss=token_nll, nlls=nlls, token_mask=attention_mask), next_step_kv

  def _step_memory_pretrain_loss(self, x0, attention_mask):
    """Three-pass full-mask -> s -> t shifted-DCache pretraining objective."""
    x0, _, attention_mask = self._maybe_sub_sample(x0, attention_mask)
    attention_mask = attention_mask.to(dtype=torch.float32)
    batch_size = x0.shape[0]
    min_ratio = float(self.config.step_memory.pretrain.min_mask_ratio)
    s_time = self._sample_t(
      x0.shape, x0.device, min_ratio, 1.0,
      block_size=self.config.model.length)
    x_s, x_t, t_time, s_mask, t_mask = self._ensure_nested_transition(
      x0, attention_mask, s_time)

    eligible = attention_mask.bool().clone()
    if self.ignore_bos:
      eligible[:, 0] = False
    full_state = torch.where(eligible, self.mask_index, x0)
    full_time = torch.ones(
      (batch_size, 1), device=x0.device, dtype=torch.float32)

    full_loss, full_cache = self._recurrent_pretrain_state_loss(
      x0, full_state, attention_mask, full_time,
      previous_step_kv=None, return_step_kv=True)
    s_loss, s_cache = self._recurrent_pretrain_state_loss(
      x0, x_s, attention_mask, s_time,
      previous_step_kv=full_cache, return_step_kv=True)
    t_loss, _ = self._recurrent_pretrain_state_loss(
      x0, x_t, attention_mask, t_time,
      previous_step_kv=s_cache, return_step_kv=False)

    config = self.config.step_memory.pretrain
    full_weight = float(config.full_loss_weight)
    s_weight = float(config.s_loss_weight)
    t_weight = float(config.t_loss_weight)
    weight_sum = full_weight + s_weight + t_weight
    if weight_sum <= 0:
      raise ValueError('Shifted-DCache pretraining loss weights must sum positive')
    total_loss = (
      full_weight * full_loss.loss
      + s_weight * s_loss.loss
      + t_weight * t_loss.loss) / weight_sum
    diagnostics = {
      'loss_full': full_loss.loss.detach(),
      'loss_s': s_loss.loss.detach(),
      'loss_t': t_loss.loss.detach(),
      's_mask_ratio': s_mask.float().sum() / eligible.float().sum(),
      't_mask_ratio': t_mask.float().sum() / eligible.float().sum(),
      'revealed_tokens': (s_mask & ~t_mask).float().sum(dim=-1).mean(),
      'remaining_masks': t_mask.float().sum(dim=-1).mean(),
    }
    return total_loss, s_loss, diagnostics
    
  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()
    self.metrics.reset()
    assert self.metrics.train_nlls.nll.mean_value == 0
    assert self.metrics.train_nlls.nll.weight == 0

  def training_step(self, batch, batch_idx):
    del batch_idx
    pretrain_config = getattr(self.config.step_memory, 'pretrain', {})
    if bool(getattr(pretrain_config, 'enabled', False)):
      total_loss, s_loss, diagnostics = self._step_memory_pretrain_loss(
        batch['input_ids'], batch['attention_mask'])
      self.metrics.train_nlls.update(s_loss.nlls, s_loss.token_mask)
      for name, value in diagnostics.items():
        self.log(
          f'trainer/{name}', value, on_step=True, on_epoch=False,
          sync_dist=True)
      self.log(
        name='trainer/loss', value=total_loss.detach(), on_step=True,
        on_epoch=False, sync_dist=True)
      return total_loss

    losses = self._loss(batch['input_ids'],
                        batch['attention_mask'])
    self.metrics.train_nlls.update(losses.nlls, losses.token_mask)
    total_loss = losses.loss

    rollout_config = getattr(self.config.step_memory, 'rollout', {})
    if bool(getattr(rollout_config, 'enabled', False)):
      rollout_loss, rollout_metrics = self._step_memory_rollout_loss(
        batch['input_ids'], batch['attention_mask'])
      rollout_weight = float(getattr(rollout_config, 'weight', 0.1))
      total_loss = total_loss + rollout_weight * rollout_loss
      self.log('trainer/base_loss', losses.loss.detach(), on_step=True,
               on_epoch=False, sync_dist=True)
      self.log('trainer/rollout_loss', rollout_loss.detach(), on_step=True,
               on_epoch=False, sync_dist=True)
      self.log('trainer/rollout_forwards', rollout_metrics['num_forwards'],
               on_step=True, on_epoch=False, sync_dist=True)
      self.log('trainer/rollout_final_masks',
               rollout_metrics['final_mask_count'],
               on_step=True, on_epoch=False, sync_dist=True)
    self.log(name='trainer/loss',
             value=total_loss.detach(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return total_loss

  def on_validation_epoch_start(self):
    self.metrics.reset()
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.eval()
    self.backbone.eval()
    self.noise.eval()
    assert self.metrics.valid_nlls.nll.mean_value == 0
    assert self.metrics.valid_nlls.nll.weight == 0
    self.sampling_eps = self.config.training.sampling_eps

  def on_validation_epoch_end(self):
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k,  value=v.compute(), on_step=False,
              on_epoch=True, sync_dist=True)
    if self.ema:
      self.ema.restore(self._get_parameters())
    if self.var_min and not self.trainer.sanity_checking:
      self._clipped_schedule_search()
      self.log('sampling_eps_min',
               self.sampling_eps_min,
               on_epoch=True,
               on_step=False,
               sync_dist=True)
      self.log('sampling_eps_max',
               self.sampling_eps_max,
               on_epoch=True,
               on_step=False,
               sync_dist=True)
  
  def _check_val_sampling_intvl(self, sampling_eps_min, sampling_eps_max):
    """Checks if the current sampling interval is valid for reporting likelihood."""
    if (sampling_eps_min == 1e-3 \
        and sampling_eps_max == 1 \
        and not (self.block_size == 1 and self.config.training.eval_nll)):
      return True # elbo
    elif (self.block_size == 1 and sampling_eps_min >= 1):
      return True # nll (block size 1)
    return False # not a valid elbo (biased estimate)
      
  def validation_step(self, batch, batch_idx):
    """Evaluate a fixed corruption for fair curves across separate runs.

    The validation RNG is restored after every batch, so validation does not
    perturb the subsequent training stream. With the same seed, rank, and
    validation batch, vanilla MDLM and the DCache s-pass receive the same
    sampled time and (apart from the nested-transition boundary safeguard) the
    same corruption mask.
    """
    input_device = batch['input_ids'].device
    cuda_devices = []
    if input_device.type == 'cuda':
      cuda_devices = [input_device.index]
    trainer = getattr(self, '_trainer', None)
    global_rank = 0 if trainer is None else trainer.global_rank
    validation_seed = (
      int(self.config.seed) + 1_000_003 * int(global_rank) + int(batch_idx))
    with torch.random.fork_rng(devices=cuda_devices):
      torch.random.default_generator.manual_seed(validation_seed)
      if input_device.type == 'cuda':
        torch.cuda.manual_seed(validation_seed)
      if bool(getattr(self.config.step_memory.pretrain, 'enabled', False)):
        return self._step_memory_validation_step(batch)
      return self._standard_validation_step(batch)

  def _step_memory_validation_step(self, batch):
    """Validate the cache-assisted s-state used for the matched comparison."""
    total_loss, s_loss, diagnostics = self._step_memory_pretrain_loss(
      batch['input_ids'], batch['attention_mask'])
    token_mask = s_loss.token_mask.clone()
    if self.ignore_bos:
      token_mask[:, 0] = 0
    comparable_s_loss = (
      s_loss.nlls * token_mask).sum() / token_mask.sum().clamp_min(1)
    self.metrics.valid_nlls.update(s_loss.nlls, token_mask)
    batch_size = batch['input_ids'].shape[0]
    self.log('val/loss_s', comparable_s_loss, on_step=False, on_epoch=True,
             sync_dist=True, batch_size=batch_size)
    self.log('val/loss_total', total_loss, on_step=False, on_epoch=True,
             sync_dist=True, batch_size=batch_size)
    self.log('val/loss_full', diagnostics['loss_full'], on_step=False,
             on_epoch=True, sync_dist=True, batch_size=batch_size)
    self.log('val/loss_t', diagnostics['loss_t'], on_step=False,
             on_epoch=True, sync_dist=True, batch_size=batch_size)
    return total_loss

  def _standard_validation_step(self, batch):
    if self.var_min:
      for noise_clip_start in self.metrics.valid_vars.keys():
        sampling_eps_min, sampling_eps_max = noise_clip_start
        if self._check_val_sampling_intvl(sampling_eps_min, sampling_eps_max) == True:
          # compute and record nelbo
          losses_clip = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            sampling_eps_min=sampling_eps_min,
                            sampling_eps_max=sampling_eps_max)
          losses = Loss(
            nlls=losses_clip.nlls.clone(),
            token_mask=losses_clip.token_mask,
            loss=losses_clip.loss.clone())
        elif len(self.metrics.valid_vars[noise_clip_start]) < 100:
          # elbo from clipped schedule (biased estimate)
          losses_clip = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            sampling_eps_min=sampling_eps_min,
                            sampling_eps_max=sampling_eps_max)
        if len(self.metrics.valid_vars[noise_clip_start]) < 100:
          # only report variance over 100 batches
          nlls = losses_clip.nlls
          self.metrics.valid_vars[noise_clip_start].append(
            nlls.reshape(
              nlls.shape[0], -1, self.block_size).mean(-1))
    elif self.block_size == 1:
      # nll
      losses = self._loss(batch['input_ids'],
                          batch['attention_mask'],
                          sampling_eps_min=1,
                          sampling_eps_max=1)
    else:
      # nelbo
      losses = self._loss(batch['input_ids'],
                          batch['attention_mask'],
                          sampling_eps_min=1e-3,
                          sampling_eps_max=1)
    self.metrics.valid_nlls.update(losses.nlls, losses.token_mask)
    return losses.loss

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      self._get_parameters(),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {'scheduler': scheduler,
                      'interval': 'step',
                      'monitor': 'val/loss',
                      'name': 'trainer/lr'}
    return [optimizer], [scheduler_dict]
  
  def _resample_q_xt(
      self, x, xt, move_indices, p, block_size, sampling_eps_min, sampling_eps_max):
    """Resamples x_t if the percentage of masked tokens is outside the bounds
    defined by sampling_eps_min and sampling_eps_max."""
    perc_masked = (xt == self.mask_index).float().sum(-1) / block_size
    while (perc_masked < sampling_eps_min).any() or \
      (perc_masked > sampling_eps_max).any():
      # if a bound is epsilon, don't resample
      if sampling_eps_min == 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked > sampling_eps_max)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max == 1:
        regen_idx = (perc_masked < sampling_eps_min)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked < sampling_eps_min) | (perc_masked > sampling_eps_max)
      regen_idx = regen_idx.repeat_interleave(block_size,dim=-1)
      move_indices[regen_idx] = (torch.rand(
        * x.shape, device=x.device) < p)[regen_idx]
      xt = torch.where(move_indices, self.mask_index, x)
      xt = xt.reshape(xt.shape[0], -1, block_size)
      perc_masked = (xt == self.mask_index).float().sum(-1) / block_size
    return xt
  
  def q_xt(
      self, x, p, block_size=None, sampling_eps_min=None, sampling_eps_max=None):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      p: float torch.Tensor with shape (batch_size, 1).
      block_size: int, block size.
      sampling_eps_min: float, minimum percentage of masked tokens.
      sampling_eps_max: float, maximum percentage of masked tokens.
    """
    if block_size is None:
      block_size = self.block_size
  
    move_indices = torch.rand(
      * x.shape, device=x.device) <= p
    xt = torch.where(move_indices, self.mask_index, x)

    if block_size == 1 and sampling_eps_min == 1.0:
      return torch.full_like(x, self.mask_index)

    # no need to resample for bounds 1e-3, 1
    if self.config.training.resample and \
      not (sampling_eps_min == 1e-3 and sampling_eps_max == 1.0):
      xt = xt.reshape(xt.shape[0], -1, block_size)
      xt = self._resample_q_xt(x,
                               xt,
                               move_indices,
                               p,
                               block_size,
                               sampling_eps_min,
                               sampling_eps_max)
      xt = xt.reshape(xt.shape[0], -1)
    return xt

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64, device=self.device)

  @torch.no_grad()
  def _nucleus_sample(self, p_x0):
    p = self.config.sampling.nucleus_p
    if p == 1.0:
      return p_x0
    p_x0_ = p_x0[:, -self.block_size:].clone()
    sorted_probs, sorted_indices = p_x0_.sort(dim=-1, descending=True)
    cum_probs = sorted_probs.cumsum(dim=-1)
    nucleus_mask = cum_probs <= p
    nucleus_mask[..., 0] = 1
    sorted_probs = sorted_probs * nucleus_mask
    p_x0_.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
    p_x0_ /= p_x0_.sum(-1, keepdim=True)
    p_x0[:, -self.block_size:] = p_x0_
    return p_x0

  @torch.no_grad()
  def _ddpm_caching_update(self, x, t, dt, p_x0=None,
                           previous_step_kv=None):
    _, move_chance_t = self.noise(t)
    _, move_chance_s = self.noise(t - dt)
    sigma_t = self._sigma_from_p(move_chance_t)
    move_chance_t = move_chance_t[:, None]
    move_chance_s = move_chance_s[:, None]
    mask_prob = move_chance_s / move_chance_t

    next_step_kv = previous_step_kv
    if p_x0 is None:
      use_step_memory = self.config.step_memory.enabled
      use_previous_kv = bool(getattr(
        self.config.step_memory, 'use_previous_kv', True))
      if self.config.sampling.kv_cache:
        model_output = self.forward(
          x[:, -self.block_size:], sigma_t, sample_mode=True,
          previous_step_kv=(
            previous_step_kv
            if use_step_memory and use_previous_kv else None),
          return_step_kv=use_step_memory)
      else:   
        model_output = self.forward(
          x, sigma_t, sample_mode=True,
          previous_step_kv=(
            previous_step_kv
            if use_step_memory and use_previous_kv else None),
          return_step_kv=use_step_memory)
      if use_step_memory:
        p_x0, next_step_kv = model_output
      else:
        p_x0 = model_output
      p_x0 = p_x0.to(torch.float64)
      if not self.config.sampling.kv_cache:
        p_x0 = p_x0[:, -self.block_size:]
      p_x0 = p_x0.exp()
      p_x0 = self._nucleus_sample(p_x0)

    if self.config.sampling.first_hitting:
      x_block = _sample_categorical(p_x0)
      # randomly and uniformly select an index in the block (among masked tokens)
      num_masked = (x[:, -self.block_size:] == self.mask_index).sum(-1)
      ind = torch.randint(0, num_masked, (x_block.shape[0],))
      ind = (x[:, -self.block_size:] == self.mask_index).nonzero()[ind, 1]
      mask = (torch.arange(self.block_size, device=x.device) == ind[:, None]).to(x_block.dtype)
      x_block = x_block * mask + x[:, -self.block_size:] * (1 - mask)
    else:
      q_xs = p_x0 * (1 - mask_prob)
      q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)
      x_block = _sample_categorical(q_xs)
    copy_flag = (x[:, -self.block_size:] != self.mask_index).to(x.dtype)
    x_block =  copy_flag * x[:, -self.block_size:] + (1 - copy_flag) * x_block
    x_new = torch.cat((x[:, :-self.block_size], x_block), dim=-1)

    # compute kv cache if all tokens in a block are sampled
    if self.config.sampling.kv_cache and self.mask_index not in x_block:
      _ = self.forward(x_block, sigma_t, sample_mode=True, store_kv=True)

    if not torch.allclose(x_new, x):
      return None, x_new, next_step_kv
    else:
      return p_x0, x_new, next_step_kv

  @torch.no_grad()
  def _ar_sampler(self, bsz, context_len=1024):
    # reset kvs
    if self.config.sampling.kv_cache:
      self.backbone.reset_kv_cache()

    with torch.amp.autocast('cuda', dtype=torch.float32):
      # precompute token buffer
      num_pred_tokens = self.num_tokens - 1
      x = torch.zeros(
        (bsz, num_pred_tokens + 1),
        dtype=torch.long,
        device=self.device)
      x[:, 0] = self.tokenizer.bos_token_id
      stop = False
      for i in tqdm(range(num_pred_tokens)):
        # need to sample a gumbel for each token
        # to save memory in variable-length sampling
        noise = (torch.distributions.Gumbel(0, 1)
                .sample((bsz, self.vocab_size))
                .to(self.device))
        next_logits = self.forward(
          x[:, :i + 1][:, -context_len:],
          None,
          store_kv=self.config.sampling.kv_cache)[:, -1:].to(torch.float64)
    
        next_logits = next_logits.exp()
        next_logits = self._nucleus_sample(next_logits).log()
        y = (next_logits[:, -1] + noise).argmax(-1)
        # check if we need to resample (or stop sampling for variable-length sampling)
        if (i+1) > 256:
          stop, x_out = self._check_stop_conds(x[:, :i+1])
          if stop:
            x = x_out
        if (stop and not self.config.sampling.var_length) \
          or (stop and x.shape[-1] == 1):
          return None
        elif stop:
          break
        x[:, i + 1] = y
      return x
  
  @torch.no_grad()
  def _sample(
    self, seqlen=None, num_steps=None, eps=1e-5, batch_size_per_gpu=None):
    """Generate samples from the model."""
    if seqlen is None:
      seqlen = self.config.model.length
    if batch_size_per_gpu is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size
    samples = []
    if self.parameterization == 'ar':
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          num_tries += 1
          sample_i = self._ar_sampler(batch_size_per_gpu)
          if num_tries > 10:
            raise ValueError('Sampling failed.')
        samples.append(sample_i)
        self.metrics.gen_nfes.append(self.config.model.length)
      samples = torch.cat(samples, dim=0) 
      return self.tokenizer.batch_decode(samples)
    if self.sampler == 'semi_ar':
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          num_tries += 1
          sample_i, nfes = self._semi_ar_sampler(
            n_samples=batch_size_per_gpu,
            num_strides=(seqlen // self.block_size), 
            num_steps=num_steps,
            seqlen=seqlen)
          if num_tries > 10:
            raise ValueError('Sampling failed.')
        samples.append(sample_i)
        self.metrics.nfes.update(nfes)
        self.metrics.gen_nfes.append(nfes)
    else:
      nfes = num_steps
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          sample_i = self._analytic_sampler(
            n_samples=batch_size_per_gpu,
            num_steps=num_steps,
            seqlen=seqlen,
            eps=eps)
          num_tries += 1
          if num_tries > 10 and sample_i is None:
            raise ValueError('Sampling failed.')
        samples.append(sample_i)
        self.metrics.nfes.update(nfes)
        self.metrics.gen_nfes.append(nfes)
    samples = torch.cat(samples, dim=0) 
    return self.tokenizer.batch_decode(samples)

  def _sigma_from_p(self, p):
    return torch.min(- torch.log(1 - p), self.noise.sigma_max)

  def restore_model_and_sample(self, num_steps, eps=1e-5, seqlen=None):
    """Generate samples from the model."""
    if self.ema:  
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
    self.backbone.eval()
    self.noise.eval()
    samples = self._sample(
      seqlen=seqlen,
      batch_size_per_gpu=self.config.loader.eval_batch_size,
      num_steps=num_steps,
      eps=eps)
    self.metrics.record_generative_perplexity(
      samples,
      self.config.model.length,
      self.config.loader.eval_batch_size,
      self.device)
    return samples

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma).to(torch.float64)
    if self.config.sampling.nucleus_p == 1.0:
      return model_output.exp()
    model_output = model_output - model_output.logsumexp(-1, keepdim=True)
    model_output = self._nucleus_sample(model_output.exp())
    return model_output

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, dt):
    sigma_t = self._sigma_from_p(self.noise(t)[1])
    sigma_s = self._sigma_from_p(self.noise(t - dt)[1])
    dsigma = sigma_t - sigma_s
    score = self.get_score(x, sigma_t)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)


  def _denoiser_update(self, x, t):
    sigma = self._sigma_from_p(self.noise(t)[1])
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples


  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(
      self, batch_dims, device, sampling_eps_min, sampling_eps_max, block_size=None):
    if block_size is None:
      block_size = self.block_size
    n = batch_dims[-1]
    num_blocks = n // block_size
    _eps_b = torch.rand((batch_dims[0], num_blocks), device=device)

    # antithetic sampling along blocks & batches (for uniform sampling)
    if self.antithetic_sampling:
      offset_b = torch.arange(batch_dims[0] * num_blocks, device=device) / (batch_dims[0] * num_blocks)
      offset_b = offset_b.view(batch_dims[0], num_blocks)
      _eps_b = (_eps_b / (batch_dims[0] * num_blocks) + offset_b) % 1
    t = _eps_b
    if block_size != self.config.model.length:
      t = t.repeat_interleave(block_size, dim=-1)

    # nll
    if sampling_eps_max >= 1 and sampling_eps_min >= 1:
      return torch.ones_like(t)
    t = t * (sampling_eps_max - sampling_eps_min) + sampling_eps_min
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.num_tokens:
      assert seqlen == 2 * self.num_tokens
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.num_tokens)
      end = start + self.num_tokens
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation ppl, since the val
      # examples will all start and end with BOS/EOS
      if self.config.data.insert_train_special == True:
        input_tokens[:, 0] = self.tokenizer.bos_token_id
        output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    
    return input_tokens, output_tokens, new_attention_mask

  def _forward_pass_diffusion(self, x0, t=None, sampling_eps_min=None, sampling_eps_max=None):
    if t is None:
      t = self._sample_t(x0.shape,
                         x0.device,
                         sampling_eps_min,
                         sampling_eps_max)

    loss_scale, p = self.noise(t)
    sigma = self._sigma_from_p(p[:,0].unsqueeze(-1))
    dsigma = - loss_scale * torch.expm1(sigma) # used for sedd

    # below is needed to reproduce mdlm/sedd numbers with models from sahoo et al
    # (numerical imprecision computing probs under loglinear schedule)
    if self.mdlm_loss_scale:
      sigma, dsigma = self.noise.total_noise(t), self.noise.rate_noise(t)
      p = 1 - torch.exp(-sigma)
      loss_scale = - (dsigma / torch.expm1(sigma))

    xt = self.q_xt(x0,
                   p,
                   sampling_eps_min=sampling_eps_min,
                   sampling_eps_max=sampling_eps_max)
    if sampling_eps_min is not None and sampling_eps_min > 0.5:
      loss_scale = - torch.ones_like(loss_scale)
    if self.ignore_bos:
      xt[:, 0] = x0[:, 0]
    
    x_input = xt
    if self.cross_attn:
      x_input = torch.cat((xt, x0), dim=-1)

    model_output = self.forward(x_input, sigma=sigma)
    utils.print_nans(model_output, 'model_output')

    if self.parameterization == 'sedd':
      return dsigma * self._score_entropy(
        model_output, sigma, xt, x0)

    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    loss = loss_scale * log_p_theta
    return loss

  def _loss(self, x0, attention_mask, t=None, sampling_eps_min=None, sampling_eps_max=None):
    if sampling_eps_min is None and hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = self.sampling_eps_min
      sampling_eps_max = self.sampling_eps_max
    elif not hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = 1e-3
      sampling_eps_max = 1.0
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)
    if self.parameterization == 'ar':
      output = self.forward(input_tokens, None)
      loss = - output.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
    else:
      loss = self._forward_pass_diffusion(
        input_tokens,
        sampling_eps_min=sampling_eps_min,
        sampling_eps_max=sampling_eps_max,)
    
    if self.ignore_bos and not self.training:
      attention_mask[:, 0] = 0
      
    nlls = (loss * attention_mask)
    token_nll = nlls.sum() / attention_mask.sum()
    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _clipped_schedule_search(self):
    # collect losses per batch across devices and sum them per interval
    best_var = float('inf')
    for (eps_min, eps_max), var in self.metrics.valid_vars.items():
      all_vars = torch.tensor(0., device=self.device)
      for i in range(len(var)):
        agg_var = var[i].to(self.device)
        agg_var = self.all_gather(agg_var)
        all_vars += agg_var.var()
      if all_vars < best_var:
        best_var = all_vars
        sampling_eps_min_best = eps_min
        sampling_eps_max_best = eps_max
      self.log(f'valid_var_{round(eps_min, 2)} - {round(eps_max, 2)}',
                all_vars / len(var),
                on_epoch=True,
                on_step=False,
                sync_dist=True)
    if self.config.algo.fix_clipping == False:
      self.sampling_eps_min.fill_(sampling_eps_min_best)
      self.sampling_eps_max.fill_(sampling_eps_max_best)

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  @torch.no_grad
  def _analytic_sampler(
    self, n_samples, num_steps, seqlen, eps=1e-5): 
    x = self._sample_prior(
      n_samples,
      seqlen).to(self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    for i in tqdm(range(num_steps), desc='step'):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      x = self._analytic_update(x=x, t=t, dt=dt)
    # denoising step 
    t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                  device=self.device)
    x = self._denoiser_update(x=x, t=t)
    
    stop, x = self._check_stop_conds(x)
    if stop:
      return None
    return x

  @torch.no_grad
  def _semi_ar_sampler(
    self, n_samples, num_steps, num_strides, seqlen, context_size=1024):
    if seqlen is None:
      seqlen = self.config.model.length
    sampling_steps = 0
          
    mdlm_semi_ar = self.config.algo.name == 'mdlm' and self.config.model.length > self.block_size
    if mdlm_semi_ar:
      # sliding window of length 512 for mdlm semi-ar decoding
      num_strides = self.config.model.length // 512
      num_strides -= 1

    ones = torch.ones((n_samples,1), dtype=self.dtype,
                      device=self.device)
    
    # reset kvs
    if self.config.sampling.kv_cache:
      self.backbone.reset_kv_cache(eval_batch_size=self.config.loader.eval_batch_size)

    for stride_num in tqdm(range(num_strides)):
      # Step memory is local to one active block. Completed-block cache has a
      # separate lifetime and is intentionally not reset here.
      previous_step_kv = None
      # sample next block
      if stride_num == 0:
        x_accum = self._sample_prior(n_samples, self.block_size).to(self.device)
        x_accum[:, 0] = self.tokenizer.bos_token_id
      else:
        if mdlm_semi_ar:
          x = self._sample_prior(n_samples, 512).to(self.device)
        else:
          x = self._sample_prior(n_samples, self.block_size).to(self.device)
        x_accum = torch.cat((x_accum, x), dim=1)

      # compute logits in a sliding window (context passed to model can't exceed context_size)
      end_idx = (stride_num + 1) * self.block_size
      start_idx = max(end_idx - context_size, 0)
      fwd_idx = torch.arange(start_idx, end_idx)
      if mdlm_semi_ar and stride_num > 0: # MDLM
        fwd_idx = torch.arange(512*(stride_num), (512*(stride_num))+self.block_size)

      dt = 1 / num_steps
      p_x0_cache = None
      timesteps = torch.linspace(1, 0, num_steps, device=self.device)
      t = 1
      for i in range(num_steps):
        if self.mask_index not in x_accum:
          break

        # faster (equivalent) sampler from zheng et al (2025)
        if self.config.sampling.first_hitting:
          u = np.random.rand()
          num_masked = (x_accum[:, fwd_idx] == self.mask_index).sum(-1).item()
          t *= u**(1 / num_masked)
              
        elif not self.config.sampling.first_hitting:
          t = timesteps[i]

        p_x0_cache, x_next, previous_step_kv = self._ddpm_caching_update(
            x=x_accum[:, fwd_idx],
            t=t * ones,
            dt=dt,
            p_x0=p_x0_cache,
            previous_step_kv=previous_step_kv)
        if p_x0_cache is None:
          sampling_steps += 1
       
        x_accum[:, fwd_idx] = x_next

      # check if we need to resample (or stop sampling for variable-length sampling)
      if x_accum.shape[1] > 256:
        stop, x_accum = self._check_stop_conds(x_accum)
        if (stop and not self.config.sampling.var_length) \
          or (stop and x.shape[-1] == 1):
          return None, None
        elif stop:
          break
    return x_accum, sampling_steps
  
  def _compute_entropy(self, x):
    _, counts = torch.unique(x, return_counts=True, sorted=False)
    entropy = torch.special.entr(counts.float() / counts.sum()).sum()
    return entropy
  
  def _check_stop_conds(self, x):
    """Check if sampling should stop based on 1) eos, 2) entropy, or 3) likelihood.
    Entropy/likelihood evaluated on last 256 token-block.
    
    Args:
      x: torch.Tensor, current sample.
    Returns:
      stop: bool, whether to stop sampling.
      x: torch.Tensor, sample (potentially truncated for variable-length sampling).
    """
    stop = False # stop sampling?
    truncate_idx = None # truncate sample? (variable-length sampling only)

    # CRITERION 2: always stop sampling if entropy is low
    entropy = self._compute_entropy(x[:, -256:])
    if entropy < 4:
      stop = True

    # for variable length sampling, check if we should stop
    # sampling, and where to truncate the sample
    if self.config.sampling.var_length:
      # CRITERION 1: stop at sampled EOS token
      if len(torch.where(x == self.tokenizer.eos_token_id)[0]) > 1:
        stop = True
        eos_idx = torch.where(x == self.tokenizer.eos_token_id)
        if len(eos_idx[0]) > 1:
          truncate_idx = min(eos_idx[1][1]+1, x.shape[1])

      # CRITERION 2: stop if entropy/likelihood is low
      if entropy < 4:
        stop = True
        truncate_idx = x.shape[1] - 256

    # truncate sample (variable-length sampling only)
    if truncate_idx is not None:
      x = x[:, :truncate_idx]
      if x.ndim == 1:
        x = x.unsqueeze(0)

    return stop, x
