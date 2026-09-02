import itertools
import typing
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


@dataclass
class DcachehoopingOutput:
  """Processed model output plus recurrent workspace diagnostics."""
  scores: torch.FloatTensor
  editable_log_probs: typing.Optional[torch.FloatTensor]
  step_kv: list
  final_hidden: torch.FloatTensor
  confidence_logits: typing.Optional[torch.FloatTensor]


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
    objective_matched_config = getattr(
      getattr(self.config, 'training', {}),
      'objective_matched_multistate', {})
    objective_matched_enabled = bool(getattr(
      objective_matched_config, 'enabled', False))
    dcachehooping_config = getattr(self.config, 'dcachehooping', {})
    dcachehooping_enabled = bool(getattr(
      dcachehooping_config, 'enabled', False))
    assert not (
      objective_matched_enabled
      and bool(getattr(pretrain_config, 'enabled', False))), (
        'Objective-matched vanilla and DCache pretraining are mutually '
        'exclusive')
    if dcachehooping_enabled:
      assert bool(getattr(pretrain_config, 'enabled', False)), (
        'Dcachehooping extends DCache-v2 pretraining')
      assert self.config.step_memory.enabled
      assert self.config.step_memory.use_previous_kv
      assert self.config.step_memory.detach_between_steps, (
        'Dcachehooping requires detached recurrent sources')
      assert not objective_matched_enabled
      latent_dropout = float(
        dcachehooping_config.latent_dropout_probability)
      latent_mask_probability = float(
        dcachehooping_config.latent_mask_probability)
      assert 0 <= latent_dropout <= 1
      assert 0 <= latent_mask_probability <= 1
      assert float(dcachehooping_config.latent_mask_loss_weight) >= 0
      tentative_config = dcachehooping_config.tentative
      confidence_config = dcachehooping_config.confidence
      status_config = getattr(
        dcachehooping_config, 'status_embedding', {})
      status_enabled = bool(getattr(status_config, 'enabled', True))
      assert 0 <= float(tentative_config.batch_probability) <= 1
      if bool(getattr(
          dcachehooping_config, 'exclusive_auxiliary_routes', True)):
        tentative_probability = (
          float(tentative_config.batch_probability)
          if bool(tentative_config.enabled) else 0.0)
        assert latent_mask_probability + tentative_probability <= 1, (
          'Exclusive Dcachehooping auxiliary probabilities must sum to <= 1')
      assert float(tentative_config.loss_weight) >= 0
      assert float(confidence_config.loss_weight) >= 0
      assert not bool(confidence_config.enabled) or bool(
        tentative_config.enabled), (
          'Confidence supervision requires tentative proposals')
      assert not bool(tentative_config.enabled) or status_enabled, (
        'Tentative proposals require the token-status embedding')
      assert latent_mask_probability == 0 or status_enabled, (
        'Latent-mask robustness requires the token-status embedding')
      assert 0 <= float(
        dcachehooping_config.identity_final_probability) <= 1
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
        'Local DCache pretraining is fully teacher forced')
      step_size_min = float(pretrain_config.step_size_min)
      step_size_max = float(pretrain_config.step_size_max)
      max_t0 = float(pretrain_config.max_t0_mask_ratio)
      assert 0 < step_size_min <= step_size_max
      assert max_t0 < 1.0
      assert 3 * step_size_max <= max_t0, (
        'Largest DCache step leaves no valid interval for trajectory center')
      weights = [
        float(pretrain_config.full_loss_weight),
        float(pretrain_config.t0_loss_weight),
        float(pretrain_config.t1_loss_weight),
        float(pretrain_config.t2_loss_weight),
        float(pretrain_config.t3_loss_weight),
      ]
      assert all(weight >= 0 for weight in weights)
      assert sum(weights) > 0
      source_config = pretrain_config.source_dropout
      assert 0 <= float(source_config.cache_only_probability) <= 1
      assert 0 <= float(source_config.current_only_probability) <= 1
      assert (
        float(source_config.cache_only_probability)
        + float(source_config.current_only_probability) <= 1)
      assert int(source_config.warmup_steps) >= 1
      identity_config = pretrain_config.identity
      assert 0 <= float(identity_config.batch_probability) <= 1
      assert float(identity_config.margin) >= 0
      assert float(identity_config.weight) >= 0
    if objective_matched_enabled:
      assert not self.config.step_memory.enabled, (
        'Objective-matched control must use the vanilla backbone')
      assert not self.config.step_memory.use_previous_kv, (
        'Objective-matched control must explicitly disable previous K/V')
      assert self.config.algo.name == 'mdlm'
      assert self.config.algo.backbone == 'dit'
      assert self.block_size == self.config.model.length
      assert not self.config.algo.cross_attn
      assert self.config.noise.type == 'loglinear'
      assert not self.config.sampling.kv_cache
      assert not bool(getattr(rollout_config, 'enabled', False))
      assert float(pretrain_config.teacher_token_probability) == 1.0, (
        'Objective-matched multi-state training is fully teacher forced')
      assert not bool(pretrain_config.source_dropout.enabled), (
        'Source dropout is a cache-only treatment and must be disabled')
      assert not bool(pretrain_config.identity.enabled), (
        'Cache identity loss must be disabled for the vanilla control')
      step_size_min = float(pretrain_config.step_size_min)
      step_size_max = float(pretrain_config.step_size_max)
      max_t0 = float(pretrain_config.max_t0_mask_ratio)
      assert 0 < step_size_min <= step_size_max
      assert max_t0 < 1.0
      assert 3 * step_size_max <= max_t0, (
        'Largest local step leaves no valid trajectory-center interval')
      weights = [
        float(pretrain_config.full_loss_weight),
        float(pretrain_config.t0_loss_weight),
        float(pretrain_config.t1_loss_weight),
        float(pretrain_config.t2_loss_weight),
        float(pretrain_config.t3_loss_weight),
      ]
      assert all(weight >= 0 for weight in weights)
      assert sum(weights) > 0
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
              detach_cache_backbone=False,
              step_memory_source_mask=None,
              previous_final_hidden=None, token_status=None,
              return_dcachehooping=False,
              return_editable_log_probs=False,
              return_confidence_logits=False):
    """Returns log score."""
    if (return_editable_log_probs or return_confidence_logits) \
        and not return_dcachehooping:
      raise ValueError(
        'Optional Dcachehooping outputs require return_dcachehooping=true')
    sigma = self._process_sigma(sigma)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      if self.config.algo.name in {'bd3lm', 'mdlm'}:
        if self.config.algo.backbone == 'hf_dit':
          if (previous_step_kv is not None or return_step_kv
              or previous_final_hidden is not None or return_dcachehooping):
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
            detach_cache_backbone=detach_cache_backbone,
            step_memory_source_mask=step_memory_source_mask,
            previous_final_hidden=previous_final_hidden,
            token_status=token_status,
            return_dcachehooping=return_dcachehooping,
            return_confidence_logits=return_confidence_logits)
        if return_dcachehooping:
          logits = backbone_output.logits
          next_step_kv = backbone_output.step_kv
          final_hidden = backbone_output.final_hidden
          confidence_logits = backbone_output.confidence_logits
        elif return_step_kv:
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
    editable_log_probs = None
    if return_editable_log_probs:
      editable_logits = logits.clone()
      editable_logits[:, :, self.mask_index] = self.neg_infinity
      editable_log_probs = editable_logits.log_softmax(dim=-1)
    if self.parameterization == 'subs':
      scores = self._subs_parameterization(logits=logits, xt=x)
    elif self.parameterization == 'sedd':
      scores = self._sedd_parameterization(logits=logits, xt=x, sigma=sigma)
    else:
      scores = logits
    if return_dcachehooping:
      return DcachehoopingOutput(
        scores=scores,
        editable_log_probs=editable_log_probs,
        step_kv=next_step_kv,
        final_hidden=final_hidden,
        confidence_logits=confidence_logits)
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

  def _recurrent_pretrain_state_loss(
      self, x0, state, attention_mask, time, previous_step_kv,
      return_step_kv, step_memory_source_mask=None):
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
        self.config.step_memory.detach_between_steps),
      step_memory_source_mask=step_memory_source_mask)
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
      loss=token_nll, nlls=nlls, token_mask=attention_mask), \
      next_step_kv, model_output

  def _independent_pretrain_state_loss(
      self, x0, state, attention_mask, time):
    """Evaluate one explicit teacher-forced state with vanilla MDLM only.

    This deliberately does not pass any step-memory arguments to `forward`.
    Combined with the configuration invariant that step memory is disabled,
    this guarantees the objective-matched control cannot write or consume a
    denoising cache.
    """
    loss_scale, probability = self.noise(time)
    sigma = self._sigma_from_p(probability[:, 0].unsqueeze(-1))
    model_output = self.forward(state, sigma=sigma, sample_mode=True)
    target_log_probability = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    nlls = loss_scale * target_log_probability * attention_mask
    token_nll = nlls.sum() / attention_mask.sum()
    return Loss(
      loss=token_nll, nlls=nlls, token_mask=attention_mask)

  def _sample_local_step_trajectory(self, x0, attention_mask):
    """Sample exact nested masks for full -> t0 -> t1 -> t2 -> t3.

    Mask ratios follow the agreed centered construction:

      k ~ U(k_min, k_max)
      x ~ U(1.5k, max_t0 - 1.5k)
      (t0, t1, t2, t3) = x + (1.5k, 0.5k, -0.5k, -1.5k)

    Integer mask counts are corrected at sequence boundaries so t0 is never
    fully masked, t3 keeps at least one prediction target, and every adjacent
    transition reveals at least one clean teacher token.
    """
    config = self.config.step_memory.pretrain
    eligible = attention_mask.bool().clone()
    if self.ignore_bos:
      eligible[:, 0] = False
    eligible_counts = eligible.sum(dim=-1)
    if (eligible_counts < 5).any():
      raise ValueError(
        'Local DCache trajectories need at least five maskable positions')

    batch_size = x0.shape[0]
    k_min = float(config.step_size_min)
    k_max = float(config.step_size_max)
    max_t0 = float(config.max_t0_mask_ratio)
    k = k_min + torch.rand(
      (batch_size, 1), device=x0.device) * (k_max - k_min)
    x_min = 1.5 * k
    x_max = max_t0 - 1.5 * k
    if (x_max < x_min).any():
      raise ValueError(
        'Local trajectory has an empty center interval; reduce step size')
    center = x_min + torch.rand_like(k) * (x_max - x_min)
    offsets = torch.tensor(
      [1.5, 0.5, -0.5, -1.5], device=x0.device)[None]
    sampled_ratios = center + k * offsets

    masks = [torch.zeros_like(eligible) for _ in range(4)]
    realized_ratios = torch.empty_like(sampled_ratios)
    mask_counts = torch.empty(
      (batch_size, 4), dtype=torch.long, device=x0.device)
    for batch_index in range(batch_size):
      candidates = eligible[batch_index].nonzero(
        as_tuple=False).flatten()
      count = int(candidates.numel())
      raw_counts = torch.round(
        sampled_ratios[batch_index] * count).to(torch.long)
      corrected = torch.empty_like(raw_counts)
      corrected[0] = raw_counts[0].clamp(min=4, max=count - 1)
      corrected[1] = raw_counts[1].clamp(
        min=3, max=int(corrected[0]) - 1)
      corrected[2] = raw_counts[2].clamp(
        min=2, max=int(corrected[1]) - 1)
      corrected[3] = raw_counts[3].clamp(
        min=1, max=int(corrected[2]) - 1)
      permutation = candidates[torch.randperm(
        count, device=x0.device)]
      for state_index in range(4):
        masks[state_index][
          batch_index, permutation[:corrected[state_index]]] = True
      mask_counts[batch_index] = corrected
      realized_ratios[batch_index] = corrected.float() / count

    states = [
      torch.where(mask, self.mask_index, x0) for mask in masks]
    return {
      'states': states,
      'masks': masks,
      'ratios': realized_ratios,
      'sampled_ratios': sampled_ratios,
      'mask_counts': mask_counts,
      'step_size': k,
      'center': center,
      'eligible': eligible,
    }

  def _source_dropout_mask(self, masked_positions):
    """Sample joint/cache-only/current-only modes for masked queries."""
    config = self.config.step_memory.pretrain.source_dropout
    if not self.training or not bool(config.enabled):
      return None, {
        'cache_only_probability': torch.tensor(
          0.0, device=masked_positions.device),
        'cache_only_fraction': torch.tensor(
          0.0, device=masked_positions.device),
        'current_only_fraction': torch.tensor(
          0.0, device=masked_positions.device),
      }
    trainer = getattr(self, '_trainer', None)
    step = 0 if trainer is None else int(trainer.global_step)
    warmup_steps = max(1, int(config.warmup_steps))
    progress = min(float(step) / warmup_steps, 1.0)
    cache_probability = float(config.cache_only_probability) * progress
    current_probability = float(config.current_only_probability)
    if cache_probability + current_probability > 1.0:
      raise ValueError('Step-memory source probabilities must sum to <= 1')

    random_values = torch.rand(
      masked_positions.shape, device=masked_positions.device)
    source_mask = torch.zeros_like(masked_positions, dtype=torch.int8)
    cache_only = masked_positions & (random_values < cache_probability)
    current_only = (
      masked_positions
      & (random_values >= cache_probability)
      & (random_values < cache_probability + current_probability))
    source_mask[cache_only] = 1
    source_mask[current_only] = 2
    denominator = masked_positions.float().sum().clamp_min(1)
    return source_mask, {
      'cache_only_probability': torch.tensor(
        cache_probability, device=masked_positions.device),
      'cache_only_fraction': cache_only.float().sum() / denominator,
      'current_only_fraction': current_only.float().sum() / denominator,
    }

  @staticmethod
  def _per_example_raw_masked_nll(model_output, x0, token_mask):
    target_log_probability = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    token_mask = token_mask.to(target_log_probability.dtype)
    return -(target_log_probability * token_mask).sum(dim=-1) / \
      token_mask.sum(dim=-1).clamp_min(1)

  @staticmethod
  def _shuffle_cache_across_batch(cache):
    batch_size = cache[0].shape[0]
    if batch_size < 2:
      raise ValueError('Shuffled-cache identity loss requires batch size >= 2')
    shift = int(torch.randint(
      1, batch_size, (), device=cache[0].device).item())
    return [entry.roll(shifts=shift, dims=0) for entry in cache]

  def _step_memory_gate_mean(self):
    gates = [
      torch.tanh(block.step_memory_gate)
      for block in self.backbone.blocks
      if getattr(block, 'step_memory_gate', None) is not None]
    if not gates:
      return torch.tensor(1.0, device=self.device)
    return torch.stack(gates).mean()

  def _step_memory_pretrain_loss(self, x0, attention_mask):
    """Five-forward local-trajectory DCache-v2 pretraining objective."""
    x0, _, attention_mask = self._maybe_sub_sample(x0, attention_mask)
    attention_mask = attention_mask.to(dtype=torch.float32)
    batch_size = x0.shape[0]
    trajectory = self._sample_local_step_trajectory(x0, attention_mask)
    eligible = trajectory['eligible']
    full_state = torch.where(eligible, self.mask_index, x0)
    full_time = torch.ones(
      (batch_size, 1), device=x0.device, dtype=torch.float32)

    full_loss, previous_cache, _ = self._recurrent_pretrain_state_loss(
      x0, full_state, attention_mask, full_time,
      previous_step_kv=None, return_step_kv=True)
    config = self.config.step_memory.pretrain
    state_weights = [
      float(config.t0_loss_weight),
      float(config.t1_loss_weight),
      float(config.t2_loss_weight),
      float(config.t3_loss_weight),
    ]
    state_losses = []
    source_masks = []
    source_diagnostics = []
    t2_cache = None
    t3_output = None
    for state_index, (state, mask) in enumerate(zip(
        trajectory['states'], trajectory['masks'])):
      source_mask = None
      dropout_diagnostics = {
        'cache_only_probability': torch.tensor(0.0, device=x0.device),
        'cache_only_fraction': torch.tensor(0.0, device=x0.device),
        'current_only_fraction': torch.tensor(0.0, device=x0.device),
      }
      if state_index >= 2:
        source_mask, dropout_diagnostics = self._source_dropout_mask(mask)
      return_cache = state_index < 3
      state_loss, next_cache, model_output = \
        self._recurrent_pretrain_state_loss(
          x0, state, attention_mask,
          trajectory['ratios'][:, state_index:state_index + 1],
          previous_step_kv=previous_cache,
          return_step_kv=return_cache,
          step_memory_source_mask=source_mask)
      state_losses.append(state_loss)
      if state_index == 3:
        t3_output = model_output
      else:
        del model_output
      source_masks.append(source_mask)
      source_diagnostics.append(dropout_diagnostics)
      if state_index == 2:
        t2_cache = next_cache
      previous_cache = next_cache

    full_weight = float(config.full_loss_weight)
    weight_sum = full_weight + sum(state_weights)
    if weight_sum <= 0:
      raise ValueError('Shifted-DCache pretraining loss weights must sum positive')
    base_loss = (
      full_weight * full_loss.loss
      + sum(weight * loss.loss for weight, loss in zip(
        state_weights, state_losses))) / weight_sum

    identity_config = config.identity
    identity_loss = torch.zeros((), device=x0.device)
    identity_correct_nll = torch.zeros((), device=x0.device)
    identity_shuffled_nll = torch.zeros((), device=x0.device)
    identity_applied = torch.zeros((), device=x0.device)
    apply_identity = (
      self.training
      and bool(identity_config.enabled)
      and batch_size >= 2
      and float(torch.rand((), device=x0.device))
      < float(identity_config.batch_probability))
    if apply_identity:
      shuffled_cache = self._shuffle_cache_across_batch(t2_cache)
      with torch.no_grad():
        shuffled_output = self.forward(
          trajectory['states'][3],
          sigma=self._sigma_from_p(trajectory['ratios'][:, 3:4]),
          sample_mode=True,
          previous_step_kv=shuffled_cache,
          return_step_kv=False,
          step_memory_source_mask=source_masks[3])
      identity_mask = trajectory['masks'][3]
      if source_masks[3] is not None:
        identity_mask = identity_mask & source_masks[3].ne(2)
      valid_examples = identity_mask.any(dim=-1)
      if valid_examples.any():
        identity_applied.fill_(1.0)
        correct_per_example = self._per_example_raw_masked_nll(
          t3_output, x0, identity_mask)[valid_examples]
        shuffled_per_example = self._per_example_raw_masked_nll(
          shuffled_output, x0, identity_mask)[valid_examples]
        identity_correct_nll = correct_per_example.mean()
        identity_shuffled_nll = shuffled_per_example.mean()
        identity_loss = torch.relu(
          float(identity_config.margin)
          + correct_per_example
          - shuffled_per_example.detach()).mean()
      del shuffled_output
    total_loss = base_loss + float(identity_config.weight) * identity_loss

    cache_only_fraction = torch.stack([
      item['cache_only_fraction'] for item in source_diagnostics[2:]]).mean()
    current_only_fraction = torch.stack([
      item['current_only_fraction'] for item in source_diagnostics[2:]]).mean()
    cache_only_probability = source_diagnostics[2][
      'cache_only_probability']
    diagnostics = {
      'loss_full': full_loss.loss.detach(),
      'loss_t0': state_losses[0].loss.detach(),
      'loss_t1': state_losses[1].loss.detach(),
      'loss_t2': state_losses[2].loss.detach(),
      'loss_t3': state_losses[3].loss.detach(),
      'mask_ratio_t0': trajectory['ratios'][:, 0].mean(),
      'mask_ratio_t1': trajectory['ratios'][:, 1].mean(),
      'mask_ratio_t2': trajectory['ratios'][:, 2].mean(),
      'mask_ratio_t3': trajectory['ratios'][:, 3].mean(),
      'step_size': trajectory['step_size'].mean(),
      'revealed_tokens': (
        trajectory['mask_counts'][:, :-1]
        - trajectory['mask_counts'][:, 1:]).float().mean(),
      'remaining_masks': trajectory['mask_counts'][:, 3].float().mean(),
      'source_cache_only_probability': cache_only_probability,
      'source_cache_only_fraction': cache_only_fraction,
      'source_current_only_fraction': current_only_fraction,
      'identity_loss': identity_loss.detach(),
      'identity_applied': identity_applied,
      'identity_correct_nll': identity_correct_nll.detach(),
      'identity_shuffled_nll': identity_shuffled_nll.detach(),
      'identity_gain': (
        identity_shuffled_nll - identity_correct_nll).detach(),
      'gate_mean': self._step_memory_gate_mean().detach(),
      'loss_base': base_loss.detach(),
    }
    # t2 has the largest trajectory weight and a twice-warmed cache. Keep it
    # as the validation/training reference Loss used by the metric accumulator.
    return total_loss, state_losses[2], diagnostics

  @staticmethod
  def _shuffle_hidden_across_batch(hidden):
    batch_size = hidden.shape[0]
    if batch_size < 2:
      raise ValueError('Shuffled latent identity loss requires batch size >= 2')
    shift = int(torch.randint(
      1, batch_size, (), device=hidden.device).item())
    return hidden.roll(shifts=shift, dims=0)

  def _dcachehooping_status(self, state, tentative_mask=None):
    """Return mask=0, committed=1, tentative=2 status identifiers."""
    status_config = getattr(
      self.config.dcachehooping, 'status_embedding', {})
    if not bool(getattr(status_config, 'enabled', True)):
      return None
    status = state.ne(self.mask_index).long()
    if tentative_mask is not None:
      status = status.masked_fill(tentative_mask, 2)
    return status

  def _dcachehooping_state_loss(
      self, x0, state, attention_mask, time, previous_step_kv,
      previous_final_hidden, return_step_kv, token_status,
      step_memory_source_mask=None, loss_token_mask=None,
      return_editable_log_probs=False,
      return_confidence_logits=False):
    """Evaluate one state while returning both recurrent workspace sources."""
    loss_scale, probability = self.noise(time)
    sigma = self._sigma_from_p(probability[:, 0].unsqueeze(-1))
    output = self.forward(
      state,
      sigma=sigma,
      sample_mode=True,
      previous_step_kv=previous_step_kv,
      return_step_kv=return_step_kv,
      detach_cache_backbone=bool(
        self.config.step_memory.detach_between_steps),
      step_memory_source_mask=step_memory_source_mask,
      previous_final_hidden=previous_final_hidden,
      token_status=token_status,
      return_dcachehooping=True,
      return_editable_log_probs=return_editable_log_probs,
      return_confidence_logits=return_confidence_logits)
    target_log_probability = torch.gather(
      output.scores, -1, x0[:, :, None]).squeeze(-1)
    if loss_token_mask is None:
      loss_token_mask = attention_mask
    loss_token_mask = loss_token_mask.to(target_log_probability.dtype)
    nlls = loss_scale * target_log_probability * loss_token_mask
    token_nll = nlls.sum() / loss_token_mask.sum().clamp_min(1)
    return Loss(
      loss=token_nll, nlls=nlls, token_mask=loss_token_mask), output

  @staticmethod
  def _masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1)

  @staticmethod
  def _distributed_metric_ratio(numerator, denominator):
    """Return an exact DDP-global diagnostic ratio without gradient use."""
    numerator = numerator.detach().float().clone()
    denominator = denominator.detach().float().clone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
      torch.distributed.all_reduce(
        numerator, op=torch.distributed.ReduceOp.SUM)
      torch.distributed.all_reduce(
        denominator, op=torch.distributed.ReduceOp.SUM)
    return numerator / denominator.clamp_min(1), denominator

  @staticmethod
  def _sample_synchronized_uniform(device):
    """Sample one uniform scalar shared by every DDP rank."""
    random_value = torch.rand((), device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
      torch.distributed.broadcast(random_value, src=0)
    return float(random_value)

  @classmethod
  def _sample_synchronized_event(cls, probability, device):
    """Sample one batch event shared by every DDP rank."""
    return cls._sample_synchronized_uniform(device) < float(probability)

  def _dcachehooping_tentative_losses(
      self, x0, candidate_tokens, tentative_mask, output, eligible):
    """Direct correction CE and RemeDi-style token confidence targets."""
    target_log_probability = torch.gather(
      output.editable_log_probs, -1, x0[:, :, None]).squeeze(-1)
    tentative_loss = -self._masked_mean(
      target_log_probability, tentative_mask)

    after_tokens = output.editable_log_probs.argmax(dim=-1)
    candidate_correct = candidate_tokens.eq(x0) & tentative_mask
    candidate_wrong = ~candidate_tokens.eq(x0) & tentative_mask
    after_correct = after_tokens.eq(x0)
    before_accuracy, tentative_count = self._distributed_metric_ratio(
      candidate_correct.float().sum(), tentative_mask.float().sum())
    after_accuracy, _ = self._distributed_metric_ratio(
      (after_correct & tentative_mask).float().sum(),
      tentative_mask.float().sum())
    wrong_fix_rate, tentative_wrong_count = \
      self._distributed_metric_ratio(
        (after_correct & candidate_wrong).float().sum(),
        candidate_wrong.float().sum())
    correct_keep_rate, _ = self._distributed_metric_ratio(
      (after_correct & candidate_correct).float().sum(),
      candidate_correct.float().sum())

    confidence_loss = torch.zeros((), device=x0.device)
    confidence_brier = confidence_loss.clone()
    tentative_confidence_correct = confidence_loss.clone()
    tentative_confidence_wrong = confidence_loss.clone()
    if output.confidence_logits is not None:
      # Confidence means "the current/proposed token can be trusted". Clean
      # visible tokens are positive, incorrect tentative tokens are negative,
      # and masks receive the detached probability of their clean target.
      visible = candidate_tokens.ne(self.mask_index)
      confidence_target = torch.ones_like(
        output.confidence_logits, dtype=torch.float32)
      confidence_target = torch.where(
        tentative_mask,
        candidate_tokens.eq(x0).to(confidence_target.dtype),
        confidence_target)
      masked = ~visible
      masked_soft_target = target_log_probability.detach().exp()
      confidence_target = torch.where(
        masked, masked_soft_target, confidence_target)
      confidence_mask = eligible.bool()
      confidence_bce = F.binary_cross_entropy_with_logits(
        output.confidence_logits,
        confidence_target.to(output.confidence_logits.dtype),
        reduction='none')
      confidence_loss = self._masked_mean(
        confidence_bce, confidence_mask)
      confidence_probability = output.confidence_logits.sigmoid()
      confidence_brier = self._masked_mean(
        (confidence_probability - confidence_target) ** 2,
        confidence_mask)
      tentative_confidence_correct, _ = self._distributed_metric_ratio(
        (confidence_probability * candidate_correct).sum(),
        candidate_correct.float().sum())
      tentative_confidence_wrong, _ = self._distributed_metric_ratio(
        (confidence_probability * candidate_wrong).sum(),
        candidate_wrong.float().sum())
    return tentative_loss, confidence_loss, {
      'tentative_count': tentative_count,
      'tentative_wrong_count': tentative_wrong_count,
      'tentative_before_accuracy': before_accuracy.detach(),
      'tentative_after_accuracy': after_accuracy.detach(),
      'tentative_wrong_fix_rate': wrong_fix_rate.detach(),
      'tentative_correct_keep_rate': correct_keep_rate.detach(),
      'confidence_brier': confidence_brier.detach(),
      'confidence_correct_mean': tentative_confidence_correct.detach(),
      'confidence_wrong_mean': tentative_confidence_wrong.detach(),
    }

  def _dcachehooping_pretrain_loss(self, x0, attention_mask):
    """DCache-v2 plus detached final-state memory and direct correction."""
    x0, _, attention_mask = self._maybe_sub_sample(x0, attention_mask)
    attention_mask = attention_mask.to(dtype=torch.float32)
    eligible = attention_mask.bool().clone()
    if self.ignore_bos:
      eligible[:, 0] = False
    batch_size = x0.shape[0]
    trajectory = self._sample_local_step_trajectory(x0, attention_mask)
    full_state = torch.where(eligible, self.mask_index, x0)
    full_time = torch.ones(
      (batch_size, 1), device=x0.device, dtype=torch.float32)
    config = self.config.step_memory.pretrain
    hooping_config = self.config.dcachehooping
    tentative_config = hooping_config.tentative

    # A single categorical draw preserves both marginal supervision rates but
    # prevents the two extra trainable transformer graphs from being live at
    # once. This is important on 24 GiB GPUs and avoids attaching two distinct
    # auxiliary objectives to an arbitrary small subset of examples.
    if self.training and bool(getattr(
        hooping_config, 'exclusive_auxiliary_routes', True)):
      auxiliary_draw = self._sample_synchronized_uniform(x0.device)
      tentative_probability = (
        float(tentative_config.batch_probability)
        if bool(tentative_config.enabled) else 0.0)
      apply_tentative = auxiliary_draw < tentative_probability
      apply_latent_mask = (
        tentative_probability <= auxiliary_draw
        < tentative_probability + float(
          hooping_config.latent_mask_probability))
    else:
      apply_latent_mask = (
        self.training and self._sample_synchronized_event(
          hooping_config.latent_mask_probability, x0.device))
      apply_tentative = (
        bool(tentative_config.enabled)
        and (not self.training or self._sample_synchronized_event(
          tentative_config.batch_probability, x0.device)))

    drop_latent = (
      self.training and self._sample_synchronized_event(
        hooping_config.latent_dropout_probability, x0.device))

    full_loss, full_output = self._dcachehooping_state_loss(
      x0, full_state, attention_mask, full_time,
      previous_step_kv=None,
      previous_final_hidden=None,
      return_step_kv=True,
      token_status=self._dcachehooping_status(full_state))
    previous_cache = full_output.step_kv
    previous_hidden = full_output.final_hidden.detach()

    state_weights = [
      float(config.t0_loss_weight),
      float(config.t1_loss_weight),
      float(config.t2_loss_weight),
      float(config.t3_loss_weight),
    ]
    state_losses = []
    source_masks = []
    source_diagnostics = []
    t2_output = None
    t2_cache = None
    t2_hidden = None
    t3_output = None
    for state_index, (state, mask) in enumerate(zip(
        trajectory['states'], trajectory['masks'])):
      source_mask = None
      dropout_diagnostics = {
        'cache_only_probability': torch.tensor(0.0, device=x0.device),
        'cache_only_fraction': torch.tensor(0.0, device=x0.device),
        'current_only_fraction': torch.tensor(0.0, device=x0.device),
      }
      if state_index >= 2:
        source_mask, dropout_diagnostics = self._source_dropout_mask(mask)
      return_cache = state_index < 3
      state_loss, state_output = self._dcachehooping_state_loss(
        x0, state, attention_mask,
        trajectory['ratios'][:, state_index:state_index + 1],
        previous_step_kv=previous_cache,
        previous_final_hidden=(None if drop_latent else previous_hidden),
        return_step_kv=return_cache,
        token_status=self._dcachehooping_status(state),
        step_memory_source_mask=source_mask,
        # Only t2 proposes tokens, and only on a tentative-route batch.
        return_editable_log_probs=(apply_tentative and state_index == 2))
      state_losses.append(state_loss)
      source_masks.append(source_mask)
      source_diagnostics.append(dropout_diagnostics)
      if state_index == 2:
        t2_output = state_output
        t2_cache = state_output.step_kv
        t2_hidden = state_output.final_hidden.detach()
      if state_index == 3:
        t3_output = state_output
      previous_cache = state_output.step_kv
      previous_hidden = state_output.final_hidden.detach()

    full_weight = float(config.full_loss_weight)
    weight_sum = full_weight + sum(state_weights)
    if weight_sum <= 0:
      raise ValueError('Dcachehooping base loss weights must sum positive')
    base_loss = (
      full_weight * full_loss.loss
      + sum(weight * loss.loss for weight, loss in zip(
        state_weights, state_losses))) / weight_sum

    zero = torch.zeros((), device=x0.device)
    latent_mask_loss = zero
    latent_mask_applied = zero.clone()
    if apply_latent_mask:
      latent_mask_applied.fill_(1.0)
      lexical_mask_state = torch.where(eligible, self.mask_index, x0)
      latent_mask_loss_struct, _ = self._dcachehooping_state_loss(
        x0, lexical_mask_state, attention_mask,
        trajectory['ratios'][:, 2:3],
        previous_step_kv=None,
        previous_final_hidden=t2_hidden,
        return_step_kv=False,
        # Keep the logical state even though the lexical identity is hidden.
        token_status=self._dcachehooping_status(
          trajectory['states'][2]),
        loss_token_mask=trajectory['masks'][2] & eligible)
      latent_mask_loss = latent_mask_loss_struct.loss

    tentative_loss = zero
    confidence_loss = zero
    tentative_applied = zero.clone()
    tentative_metrics = {
      name: zero.clone() for name in [
        'tentative_count', 'tentative_wrong_count',
        'tentative_before_accuracy', 'tentative_after_accuracy',
        'tentative_wrong_fix_rate', 'tentative_correct_keep_rate',
        'confidence_brier', 'confidence_correct_mean',
        'confidence_wrong_mean']}
    if apply_tentative:
      tentative_applied.fill_(1.0)
      tentative_mask = (
        trajectory['masks'][2]
        & ~trajectory['masks'][3]
        & eligible)
      candidate_tokens = t2_output.editable_log_probs.detach().argmax(dim=-1)
      tentative_state = trajectory['states'][3].clone()
      tentative_state[tentative_mask] = candidate_tokens[tentative_mask]
      tentative_status = self._dcachehooping_status(
        tentative_state, tentative_mask=tentative_mask)
      _, tentative_output = self._dcachehooping_state_loss(
        x0, tentative_state, attention_mask,
        trajectory['ratios'][:, 3:4],
        previous_step_kv=t2_cache,
        previous_final_hidden=t2_hidden,
        return_step_kv=False,
        token_status=tentative_status,
        return_editable_log_probs=True,
        return_confidence_logits=bool(
          hooping_config.confidence.enabled))
      tentative_loss, confidence_loss, tentative_metrics = \
        self._dcachehooping_tentative_losses(
          x0, tentative_state, tentative_mask,
          tentative_output, eligible)
      if not bool(hooping_config.confidence.enabled):
        confidence_loss = zero

    identity_config = config.identity
    identity_loss = zero
    identity_correct_nll = zero
    identity_shuffled_nll = zero
    identity_applied = zero.clone()
    identity_final_source = zero.clone()
    apply_identity = (
      self.training
      and bool(identity_config.enabled)
      and batch_size >= 2
      and self._sample_synchronized_event(
        identity_config.batch_probability, x0.device))
    if apply_identity:
      shuffle_final = (
        not drop_latent
        and self._sample_synchronized_event(
          hooping_config.identity_final_probability, x0.device))
      identity_final_source.fill_(float(shuffle_final))
      identity_cache = t2_cache
      identity_hidden = None if drop_latent else t2_hidden
      if shuffle_final:
        identity_hidden = self._shuffle_hidden_across_batch(t2_hidden)
      else:
        identity_cache = self._shuffle_cache_across_batch(t2_cache)
      with torch.no_grad():
        shuffled_output = self.forward(
          trajectory['states'][3],
          sigma=self._sigma_from_p(trajectory['ratios'][:, 3:4]),
          sample_mode=True,
          previous_step_kv=identity_cache,
          return_step_kv=False,
          step_memory_source_mask=source_masks[3],
          previous_final_hidden=identity_hidden,
          token_status=self._dcachehooping_status(
            trajectory['states'][3]))
      identity_mask = trajectory['masks'][3] & eligible
      if source_masks[3] is not None:
        identity_mask = identity_mask & source_masks[3].ne(2)
      valid_examples = identity_mask.any(dim=-1)
      if valid_examples.any():
        identity_applied.fill_(1.0)
        correct_per_example = self._per_example_raw_masked_nll(
          t3_output.scores, x0, identity_mask)[valid_examples]
        shuffled_per_example = self._per_example_raw_masked_nll(
          shuffled_output, x0, identity_mask)[valid_examples]
        identity_correct_nll = correct_per_example.mean()
        identity_shuffled_nll = shuffled_per_example.mean()
        identity_loss = torch.relu(
          float(identity_config.margin)
          + correct_per_example
          - shuffled_per_example.detach()).mean()

    # DDP uses find_unused_parameters=false. Auxiliary routes are sampled, so
    # keep every optional parameter in every autograd graph with zero gradient
    # on batches where its route is absent. This avoids a reduction failure on
    # the next update without changing the objective.
    optional_parameter_anchor = sum(
      parameter.sum() * 0.0
      for name, parameter in self.backbone.named_parameters()
      if 'dcachehooping_' in name)
    total_loss = (
      base_loss
      + float(hooping_config.latent_mask_loss_weight) * latent_mask_loss
      + float(tentative_config.loss_weight) * tentative_loss
      + float(hooping_config.confidence.loss_weight) * confidence_loss
      + float(identity_config.weight) * identity_loss
      + optional_parameter_anchor)

    cache_only_fraction = torch.stack([
      item['cache_only_fraction'] for item in source_diagnostics[2:]]).mean()
    current_only_fraction = torch.stack([
      item['current_only_fraction'] for item in source_diagnostics[2:]]).mean()
    diagnostics = {
      'loss_full': full_loss.loss.detach(),
      'loss_t0': state_losses[0].loss.detach(),
      'loss_t1': state_losses[1].loss.detach(),
      'loss_t2': state_losses[2].loss.detach(),
      'loss_t3': state_losses[3].loss.detach(),
      'mask_ratio_t0': trajectory['ratios'][:, 0].mean(),
      'mask_ratio_t1': trajectory['ratios'][:, 1].mean(),
      'mask_ratio_t2': trajectory['ratios'][:, 2].mean(),
      'mask_ratio_t3': trajectory['ratios'][:, 3].mean(),
      'step_size': trajectory['step_size'].mean(),
      'revealed_tokens': (
        trajectory['mask_counts'][:, :-1]
        - trajectory['mask_counts'][:, 1:]).float().mean(),
      'remaining_masks': trajectory['mask_counts'][:, 3].float().mean(),
      'source_cache_only_probability': source_diagnostics[2][
        'cache_only_probability'],
      'source_cache_only_fraction': cache_only_fraction,
      'source_current_only_fraction': current_only_fraction,
      'latent_dropped': torch.tensor(
        float(drop_latent), device=x0.device),
      'latent_mask_applied': latent_mask_applied,
      'latent_mask_loss': latent_mask_loss.detach(),
      'tentative_applied': tentative_applied,
      'tentative_loss': tentative_loss.detach(),
      'confidence_loss': confidence_loss.detach(),
      'identity_loss': identity_loss.detach(),
      'identity_applied': identity_applied,
      'identity_final_source': identity_final_source,
      'identity_correct_nll': identity_correct_nll.detach(),
      'identity_shuffled_nll': identity_shuffled_nll.detach(),
      'identity_gain': (
        identity_shuffled_nll - identity_correct_nll).detach(),
      'gate_mean': self._step_memory_gate_mean().detach(),
      'loss_base': base_loss.detach(),
      'loss_total': total_loss.detach(),
    }
    diagnostics.update(tentative_metrics)
    return total_loss, state_losses[2], diagnostics

  def _objective_matched_multistate_loss(self, x0, attention_mask):
    """Five-state vanilla control matched to the DCache base objective.

    The full, t0, t1, t2, and t3 states use the exact same trajectory sampler
    and normalized token-loss weights as DCache-v2. Every forward is
    independent: the vanilla backbone has no DCache parameters, and no cache
    is written or supplied between states.
    """
    if self.config.step_memory.enabled:
      raise RuntimeError(
        'Objective-matched vanilla loss requires step memory to be disabled')
    x0, _, attention_mask = self._maybe_sub_sample(x0, attention_mask)
    attention_mask = attention_mask.to(dtype=torch.float32)
    batch_size = x0.shape[0]
    trajectory = self._sample_local_step_trajectory(x0, attention_mask)
    eligible = trajectory['eligible']
    full_state = torch.where(eligible, self.mask_index, x0)
    full_time = torch.ones(
      (batch_size, 1), device=x0.device, dtype=torch.float32)

    full_loss = self._independent_pretrain_state_loss(
      x0, full_state, attention_mask, full_time)
    state_losses = []
    for state_index, state in enumerate(trajectory['states']):
      state_losses.append(self._independent_pretrain_state_loss(
        x0,
        state,
        attention_mask,
        trajectory['ratios'][:, state_index:state_index + 1]))

    config = self.config.step_memory.pretrain
    full_weight = float(config.full_loss_weight)
    state_weights = [
      float(config.t0_loss_weight),
      float(config.t1_loss_weight),
      float(config.t2_loss_weight),
      float(config.t3_loss_weight),
    ]
    weight_sum = full_weight + sum(state_weights)
    if weight_sum <= 0:
      raise ValueError(
        'Objective-matched multi-state loss weights must sum positive')
    total_loss = (
      full_weight * full_loss.loss
      + sum(weight * loss.loss for weight, loss in zip(
        state_weights, state_losses))) / weight_sum

    diagnostics = {
      'loss_full': full_loss.loss.detach(),
      'loss_t0': state_losses[0].loss.detach(),
      'loss_t1': state_losses[1].loss.detach(),
      'loss_t2': state_losses[2].loss.detach(),
      'loss_t3': state_losses[3].loss.detach(),
      'mask_ratio_t0': trajectory['ratios'][:, 0].mean(),
      'mask_ratio_t1': trajectory['ratios'][:, 1].mean(),
      'mask_ratio_t2': trajectory['ratios'][:, 2].mean(),
      'mask_ratio_t3': trajectory['ratios'][:, 3].mean(),
      'step_size': trajectory['step_size'].mean(),
      'revealed_tokens': (
        trajectory['mask_counts'][:, :-1]
        - trajectory['mask_counts'][:, 1:]).float().mean(),
      'remaining_masks': trajectory['mask_counts'][:, 3].float().mean(),
      'loss_weight_sum': torch.tensor(weight_sum, device=x0.device),
      'num_forwards': torch.tensor(5.0, device=x0.device),
      'loss_base': total_loss.detach(),
    }
    # Match DCache reporting: t2 has the largest local-state loss weight.
    return total_loss, state_losses[2], diagnostics
    
  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()
    self.metrics.reset()
    assert self.metrics.train_nlls.nll.mean_value == 0
    assert self.metrics.train_nlls.nll.weight == 0

  def training_step(self, batch, batch_idx):
    del batch_idx
    pretrain_config = getattr(self.config.step_memory, 'pretrain', {})
    objective_matched_config = getattr(
      self.config.training, 'objective_matched_multistate', {})
    dcache_pretrain = bool(getattr(pretrain_config, 'enabled', False))
    objective_matched = bool(getattr(
      objective_matched_config, 'enabled', False))
    if dcache_pretrain or objective_matched:
      if dcache_pretrain:
        if bool(getattr(self.config.dcachehooping, 'enabled', False)):
          total_loss, reference_loss, diagnostics = \
            self._dcachehooping_pretrain_loss(
              batch['input_ids'], batch['attention_mask'])
        else:
          total_loss, reference_loss, diagnostics = \
            self._step_memory_pretrain_loss(
              batch['input_ids'], batch['attention_mask'])
      else:
        total_loss, reference_loss, diagnostics = \
          self._objective_matched_multistate_loss(
            batch['input_ids'], batch['attention_mask'])
      self.metrics.train_nlls.update(
        reference_loss.nlls, reference_loss.token_mask)
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
      objective_matched_config = getattr(
        self.config.training, 'objective_matched_multistate', {})
      if bool(getattr(objective_matched_config, 'enabled', False)):
        return self._objective_matched_validation_step(batch)
      return self._standard_validation_step(batch)

  def _step_memory_validation_step(self, batch):
    """Validate the twice-warmed t2 state and all trajectory components."""
    if bool(getattr(self.config.dcachehooping, 'enabled', False)):
      total_loss, reference_loss, diagnostics = \
        self._dcachehooping_pretrain_loss(
          batch['input_ids'], batch['attention_mask'])
    else:
      total_loss, reference_loss, diagnostics = \
        self._step_memory_pretrain_loss(
          batch['input_ids'], batch['attention_mask'])
    token_mask = reference_loss.token_mask.clone()
    if self.ignore_bos:
      token_mask[:, 0] = 0
    comparable_t2_loss = (
      reference_loss.nlls * token_mask).sum() / token_mask.sum().clamp_min(1)
    self.metrics.valid_nlls.update(reference_loss.nlls, token_mask)
    batch_size = batch['input_ids'].shape[0]
    # Preserve loss_s/loss_t aliases so existing plotting tools keep working:
    # s is the main twice-warmed t2 state and t is the final t3 state.
    self.log('val/loss_s', comparable_t2_loss, on_step=False, on_epoch=True,
             sync_dist=True, batch_size=batch_size)
    self.log('val/loss_t2', comparable_t2_loss, on_step=False, on_epoch=True,
             sync_dist=True, batch_size=batch_size)
    self.log('val/loss_total', total_loss, on_step=False, on_epoch=True,
             sync_dist=True, batch_size=batch_size)
    self.log('val/loss_full', diagnostics['loss_full'], on_step=False,
             on_epoch=True, sync_dist=True, batch_size=batch_size)
    self.log('val/loss_t0', diagnostics['loss_t0'], on_step=False,
             on_epoch=True, sync_dist=True, batch_size=batch_size)
    self.log('val/loss_t1', diagnostics['loss_t1'], on_step=False,
             on_epoch=True, sync_dist=True, batch_size=batch_size)
    self.log('val/loss_t3', diagnostics['loss_t3'], on_step=False,
             on_epoch=True, sync_dist=True, batch_size=batch_size)
    self.log('val/loss_t', diagnostics['loss_t3'], on_step=False,
             on_epoch=True, sync_dist=True, batch_size=batch_size)
    for state_index in range(4):
      self.log(
        f'val/mask_ratio_t{state_index}',
        diagnostics[f'mask_ratio_t{state_index}'],
        on_step=False, on_epoch=True, sync_dist=True,
        batch_size=batch_size)
    self.log('val/gate_mean', diagnostics['gate_mean'], on_step=False,
             on_epoch=True, sync_dist=True, batch_size=batch_size)
    for name in [
        'loss_base', 'latent_mask_loss', 'tentative_loss',
        'confidence_loss', 'tentative_count', 'tentative_wrong_count',
        'tentative_before_accuracy', 'tentative_after_accuracy',
        'tentative_wrong_fix_rate', 'tentative_correct_keep_rate',
        'confidence_brier', 'confidence_correct_mean',
        'confidence_wrong_mean']:
      if name in diagnostics:
        self.log(
          f'val/{name}', diagnostics[name], on_step=False, on_epoch=True,
          sync_dist=True, batch_size=batch_size)
    return total_loss

  def _objective_matched_validation_step(self, batch):
    """Validate B on the same deterministic local trajectory as DCache."""
    total_loss, reference_loss, diagnostics = \
      self._objective_matched_multistate_loss(
        batch['input_ids'], batch['attention_mask'])
    token_mask = reference_loss.token_mask.clone()
    if self.ignore_bos:
      token_mask[:, 0] = 0
    comparable_t2_loss = (
      reference_loss.nlls * token_mask).sum() / token_mask.sum().clamp_min(1)
    self.metrics.valid_nlls.update(reference_loss.nlls, token_mask)
    batch_size = batch['input_ids'].shape[0]
    self.log('val/loss_s', comparable_t2_loss, on_step=False, on_epoch=True,
             sync_dist=True, batch_size=batch_size)
    self.log('val/loss_t2', comparable_t2_loss, on_step=False, on_epoch=True,
             sync_dist=True, batch_size=batch_size)
    self.log('val/loss_total', total_loss, on_step=False, on_epoch=True,
             sync_dist=True, batch_size=batch_size)
    self.log('val/loss_full', diagnostics['loss_full'], on_step=False,
             on_epoch=True, sync_dist=True, batch_size=batch_size)
    self.log('val/loss_t0', diagnostics['loss_t0'], on_step=False,
             on_epoch=True, sync_dist=True, batch_size=batch_size)
    self.log('val/loss_t1', diagnostics['loss_t1'], on_step=False,
             on_epoch=True, sync_dist=True, batch_size=batch_size)
    self.log('val/loss_t3', diagnostics['loss_t3'], on_step=False,
             on_epoch=True, sync_dist=True, batch_size=batch_size)
    self.log('val/loss_t', diagnostics['loss_t3'], on_step=False,
             on_epoch=True, sync_dist=True, batch_size=batch_size)
    for state_index in range(4):
      self.log(
        f'val/mask_ratio_t{state_index}',
        diagnostics[f'mask_ratio_t{state_index}'],
        on_step=False, on_epoch=True, sync_dist=True,
        batch_size=batch_size)
    self.log('val/num_forwards', diagnostics['num_forwards'],
             on_step=False, on_epoch=True, sync_dist=True,
             batch_size=batch_size)
    self.log('val/loss_weight_sum', diagnostics['loss_weight_sum'],
             on_step=False, on_epoch=True, sync_dist=True,
             batch_size=batch_size)
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
                           previous_step_kv=None,
                           previous_final_hidden=None):
    """Apply one denoising update and advance recurrent model state.

    ``p_x0`` is reused when the sampled state did not change.  In that case
    there is deliberately no model forward, so neither recurrent source may
    advance: both returned memories remain the inputs from the last real
    forward.
    """
    _, move_chance_t = self.noise(t)
    _, move_chance_s = self.noise(t - dt)
    sigma_t = self._sigma_from_p(move_chance_t)
    move_chance_t = move_chance_t[:, None]
    move_chance_s = move_chance_s[:, None]
    mask_prob = move_chance_s / move_chance_t

    next_step_kv = previous_step_kv
    next_final_hidden = previous_final_hidden
    if p_x0 is None:
      use_step_memory = self.config.step_memory.enabled
      use_previous_kv = bool(getattr(
        self.config.step_memory, 'use_previous_kv', True))
      use_final_state = bool(getattr(
        getattr(self.config, 'dcachehooping', {}), 'enabled', False))
      forward_x = (
        x[:, -self.block_size:]
        if self.config.sampling.kv_cache else x)
      if use_final_state:
        model_output = self.forward(
          forward_x,
          sigma_t,
          sample_mode=True,
          previous_step_kv=(
            previous_step_kv
            if use_step_memory and use_previous_kv else None),
          return_step_kv=use_step_memory,
          previous_final_hidden=previous_final_hidden,
          return_dcachehooping=True)
        p_x0 = model_output.scores
        next_step_kv = model_output.step_kv
        # Sampling runs under no_grad, but detaching here documents and
        # enforces the same recurrent boundary used during pretraining.
        next_final_hidden = model_output.final_hidden.detach()
      elif self.config.sampling.kv_cache:
        model_output = self.forward(
          forward_x, sigma_t, sample_mode=True,
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
      if not use_final_state:
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
      masked = x[:, -self.block_size:] == self.mask_index
      num_masked = masked.sum(-1)
      if torch.any(num_masked == 0):
        raise RuntimeError(
          'First-hitting update requires at least one mask in every sample')
      random_rank = torch.floor(
        torch.rand(x_block.shape[0], device=x.device) * num_masked).long()
      rank_at_position = masked.long().cumsum(-1) - 1
      mask = (
        masked & rank_at_position.eq(random_rank[:, None])).to(x_block.dtype)
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
      return None, x_new, next_step_kv, next_final_hidden
    else:
      return p_x0, x_new, next_step_kv, next_final_hidden

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
      previous_final_hidden = None
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

        (p_x0_cache,
         x_next,
         previous_step_kv,
         previous_final_hidden) = self._ddpm_caching_update(
            x=x_accum[:, fwd_idx],
            t=t * ones,
            dt=dt,
            p_x0=p_x0_cache,
            previous_step_kv=previous_step_kv,
            previous_final_hidden=previous_final_hidden)
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
