import json
import os

import fsspec
import hydra
import lightning as L
import numpy as np
import omegaconf
import rich.syntax
import rich.tree
import torch

import algo
import dataloader
import utils
import datetime


# Try to import dataflux, but make it optional for clusters without dataflux support
try:
    from dataflux_pytorch.lightning import DatafluxLightningCheckpoint
    DATAFLUX_AVAILABLE = True
except ImportError:
    DATAFLUX_AVAILABLE = False
    print("Warning: dataflux_pytorch not available, skipping dataflux checkpoint integration")

# Fix for PyTorch 2.6+ checkpoint loading
# PyTorch 2.6 changed torch.load default to weights_only=True for security
# Since we're loading our own trusted checkpoints, we monkey-patch torch.load
# to default to weights_only=False when not explicitly specified
# This allows loading checkpoints with arbitrary Python objects (OmegaConf, typing, etc.)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # Only override if weights_only is not explicitly set
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


def _is_gcs_path(path):
  """Check if a path is a Google Cloud Storage path."""
  if path is None:
    return False
  return path.startswith('gs://') or path.startswith('gcs://')

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(diffusion_model, config, tokenizer):
  if 'hf' in config.algo.backbone:
    return diffusion_model(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion_model.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


def _generate_samples(diffusion_model, config, logger,
                      tokenizer):
  logger.info('Starting Sample Eval.')
  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  model.metrics.gen_ppl.reset()
  model.metrics.sample_entropy.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  all_samples = []
  batch_times = []
  for _ in range(config.sampling.num_sample_batches):
    if config.sampling.semi_ar:
      # TODO add timing stats
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)
      text_samples = intermediate_samples[-1]
      # Note: Samples generated using semi-ar method
      # need to to be processed before computing generative perplexity
      # since these samples contain numerous <|endoftext|> tokens
      # and diffusion.compute_generative_perplexity() discards
      # any text after the first EOS token.
    else:
      samples, stats = model.restore_model_and_sample(
        num_steps=config.sampling.steps,
        return_stats=True)
      model.metrics.record_entropy(samples)
      text_samples = model.tokenizer.batch_decode(samples)

      if config.eval.compute_generative_perplexity:
        model.metrics.record_generative_perplexity(
          text_samples, config.model.length, model.device)

      all_samples.extend(list(text_samples))
    batch_time = stats['duration']
    batch_times.append(batch_time)
    if isinstance(text_samples, list):
      n_samples = len(text_samples)
    else:
      n_samples = getattr(text_samples, 'shape', [0])[0] if hasattr(text_samples, 'shape') else 0
    print(f"Batch sampled in {batch_time:.4f} seconds, {n_samples} samples")
  generative_ppl = 0.
  entropy = 0.
  if not config.sampling.semi_ar:
    generative_ppl = model.metrics.gen_ppl.compute().item()
    entropy = model.metrics.sample_entropy.compute().item()
    print('Generative perplexity:', generative_ppl)
    print('Sample entropy:', entropy)
  samples_path = config.eval.generated_samples_path
  if '.' in samples_path:
    parts = samples_path.rsplit('.', 1)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    samples_path = f"{parts[0]}_{ts}.{parts[1]}"
  else:
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    samples_path = f"{samples_path}_{ts}"
  total_time = sum(batch_times)
  with fsspec.open(samples_path, 'w', encoding='utf-8') as f:
    data = {
      'generative_ppl': generative_ppl,
      'entropy': entropy,
      'generated_seqs': all_samples,
      'time_per_batch': total_time / len(batch_times),
    }
    json.dump(data, f, ensure_ascii=False, indent=4)
  print('Samples saved at:', samples_path)
  print(f"Total sampling time: {total_time:.4f} seconds")
  if len(batch_times) > 1:
    avg_time = total_time / len(batch_times)
    print(f"Average batch time: {avg_time:.4f} seconds ({len(batch_times)} batches)")

  # Evaluate samples if evaluation is available for this dataset
  from synthetic_data import evaluate_samples
  eval_metrics = evaluate_samples(samples_path, config.data)
  if eval_metrics is not None:
    # Append evaluation metrics to the samples file
    with fsspec.open(samples_path, 'r', encoding='utf-8') as f:
      data = json.load(f)
    data['eval_metrics'] = eval_metrics
    with fsspec.open(samples_path, 'w', encoding='utf-8') as f:
      json.dump(data, f, ensure_ascii=False, indent=4)
    print('Evaluation metrics added to:', samples_path)


def _eval_ppl(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Perplexity Eval.')

  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  plugins = []
  if DATAFLUX_AVAILABLE and _is_gcs_path(config.checkpointing.save_dir):
    dataflux_ckpt = DatafluxLightningCheckpoint(project_name="YOUR_GCP_PROJECT")
    plugins.append(dataflux_ckpt)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger,
    plugins=plugins)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _complete(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Completions.')
  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  model.metrics.gen_ppl.reset()
  model.metrics.sample_entropy.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides

  # load validation dataset
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True)
  _print_batch(valid_ds, valid_ds, tokenizer)

  all_samples = []
  batch_times = []
  num_sample_batches = config.sampling.num_sample_batches

  # Collect raw token IDs for puzzle evaluation
  all_predicted_ids = []
  all_ground_truth_ids = []

  for batch_idx, batch in enumerate(valid_ds):
    if num_sample_batches != -1 and batch_idx >= num_sample_batches:
      break
    samples, stats = model.restore_model_and_complete(
      batch=batch,
      num_steps=config.sampling.steps,
      return_stats=True)
    
    text_samples = model.tokenizer.batch_decode(samples)
    all_samples.extend(list(text_samples))

    # Collect for puzzle evaluation
    all_predicted_ids.append(samples.cpu().numpy())
    all_ground_truth_ids.append(batch['input_ids'].cpu().numpy())

    batch_time = stats['duration']
    batch_times.append(batch_time)
    if isinstance(text_samples, list):
      n_samples = len(text_samples)
    else:
      n_samples = getattr(text_samples, 'shape', [0])[0] if hasattr(text_samples, 'shape') else 0
    print(f"Batch sampled in {batch_time:.4f} seconds, {n_samples} samples")

  # Evaluate completions if evaluation is available for this dataset
  from synthetic_data import evaluate_completions
  eval_metrics = None
  if all_predicted_ids:
    predicted_ids = np.concatenate(all_predicted_ids, axis=0)
    ground_truth_ids = np.concatenate(all_ground_truth_ids, axis=0)
    
    eval_metrics = evaluate_completions(predicted_ids, ground_truth_ids, config.data)

  samples_path = config.eval.generated_samples_path
  if '.' in samples_path:
    parts = samples_path.rsplit('.', 1)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    samples_path = f"{parts[0]}_{ts}.{parts[1]}"
  else:
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    samples_path = f"{samples_path}_{ts}"
  total_time = sum(batch_times)
  with fsspec.open(samples_path, 'w', encoding='utf-8') as f:
    output_data = {
      'generated_seqs': all_samples,
      'time_per_batch': total_time / len(batch_times),
    }
    if eval_metrics is not None:
      output_data['eval_metrics'] = eval_metrics
    json.dump(output_data, f, ensure_ascii=False, indent=4)
  print('Samples saved at:', samples_path)
  print(f"Total sampling time: {total_time:.4f} seconds")
  if len(batch_times) > 1:
    avg_time = total_time / len(batch_times)
    print(f"Average batch time: {avg_time:.4f} seconds ({len(batch_times)} batches)")


def _train(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      **config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  if config.training.finetune_path != '':
    assert utils.fsspec_exists(config.training.finetune_path)
    model = diffusion_model.load_from_checkpoint(
      config.training.finetune_path,
      tokenizer=tokenizer,
      config=config)
  else:
    model = diffusion_model(config, tokenizer=valid_ds.tokenizer)

  plugins = []
  if DATAFLUX_AVAILABLE and _is_gcs_path(config.checkpointing.save_dir):
    dataflux_ckpt = DatafluxLightningCheckpoint(project_name="YOUR_GCP_PROJECT")
    plugins.append(dataflux_ckpt)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger,
    plugins=plugins)
  
  # Optionally run validation at step 0 before any training
  if config.eval.validate_at_start:
    logger.info('Running validation at step 0 (before training).')
    logger.info('Note this may cause CUDA graph conflicts between validate() and fit()')
    logger.info('To avoid this, set eval.validate_at_start=False')
    trainer.validate(model, valid_ds)
  
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)
  if config.algo.name == 'ar':
    diffusion_model = algo.AR
  elif config.algo.name == 'mdlm':
    diffusion_model = algo.MDLM
  elif config.algo.name == 'noshuffle_mdlm':
    diffusion_model = algo.NoShuffleMDLM
  elif config.algo.name == 'esolm':
    diffusion_model = algo.EsoLM
  elif config.algo.name == 'difflm':
    diffusion_model = algo.DiffLM
  elif config.algo.name == 'duo_base':
    diffusion_model = algo.DUO_BASE
  elif config.algo.name == 'd3pm':
    diffusion_model = algo.D3PMAbsorb
  elif config.algo.name == 'sedd':
    diffusion_model = algo.SEDDAbsorb
  elif config.algo.name == 'duo':
    diffusion_model = algo.DUO
  elif config.algo.name == 'distillation':
    diffusion_model = algo.Distillation
  elif config.algo.name == 'diffuparallel':
    diffusion_model = algo.DiffuParallel
  else:
    raise ValueError(
      f'Invalid algorithm name: {config.algo.name}')
  kwargs = {'diffusion_model': diffusion_model,
            'config': config,
            'tokenizer': tokenizer,
            'logger': logger}
  if config.mode == 'sample_eval':
    _generate_samples(**kwargs)
  elif config.mode == 'ppl_eval':
    _eval_ppl(**kwargs)
  elif config.mode == 'completions':
    _complete(**kwargs)
  else:
    _train(**kwargs)


if __name__ == '__main__':
  main()