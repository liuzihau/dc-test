#!/usr/bin/env python3
"""Isolated BD3 pretraining entrypoint for fresh/resumed MDM and MDM+NP."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from owt.runtime import install, UPSTREAM
install()

import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf
import torch

import dataloader
from owt.model import OWTMDM
from owt.metrics import LocalMetrics, PreclipGradientMetrics


def make_config(args):
    for name, fn in [('cwd', lambda: str(ROOT)), ('device_count', torch.cuda.device_count),
                     ('eval', eval), ('div_up', lambda x, y: (x+y-1)//y)]:
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, fn)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(UPSTREAM/'configs')):
        config = hydra.compose(config_name='config', overrides=[
            'algo=mdlm', 'model=small', 'data=openwebtext-split'])
    OmegaConf.set_struct(config, False)
    recipe = OmegaConf.load(ROOT/'owt/configs'/f'{args.variant}.yaml')
    config.mechanisms = recipe.mechanisms
    config.objective = recipe.objective
    config.seed = 1
    config.data.cache_dir = str(args.data_dir.resolve())
    config.data.insert_train_special = False
    config.data.insert_valid_special = False
    config.data.insert_valid_eos = False
    config.model.attn_backend = 'sdpa'
    config.loader.batch_size = config.loader.eval_batch_size = args.microbatch
    config.loader.global_batch_size = config.loader.eval_global_batch_size = args.global_batch
    config.loader.num_workers = args.workers
    config.trainer.devices = 2
    config.trainer.accumulate_grad_batches = args.global_batch // (2*args.microbatch)
    config.trainer.max_steps = args.steps
    config.trainer.precision = 'bf16-mixed'
    config.trainer.val_check_interval = args.interval * config.trainer.accumulate_grad_batches
    config.trainer.num_sanity_val_steps = 0
    config.checkpointing.save_dir = str(args.run.resolve())
    config.checkpointing.resume_from_ckpt = args.resume is not None
    config.checkpointing.resume_ckpt_path = str(args.resume) if args.resume else None
    config.wandb = None
    config.mode = 'train'
    config.monitoring = dict(validation_examples=args.val_examples, interval=args.interval,
                             fixed_validation_seed=90002, checkpoint_keep=3)
    return config


def verify_cache(config):
    result = {}
    for key, name in [('train', 'openwebtext-train_train'), ('validation', 'openwebtext-valid_validation')]:
        path = Path(config.data.cache_dir)/f'{name}_bs1024_wrapped_specialFalse.dat'
        for meta in ('state.json', 'dataset_info.json'):
            if not (path/meta).is_file():
                raise FileNotFoundError(f'Prepared OWT cache missing: {path/meta}')
        state = json.loads((path/'state.json').read_text())
        for shard in state['_data_files']:
            if not (path/shard['filename']).is_file():
                raise FileNotFoundError(path/shard['filename'])
        result[key] = dict(path=str(path), shards=len(state['_data_files']),
                           state_sha256=hashlib.sha256((path/'state.json').read_bytes()).hexdigest())
    return result


def main():
    sys.argv[0] = str(Path(__file__).resolve())
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['mdm', 'mdm_np', 'mdm_np_zero_init',
                        'mdm_np_zero_init_low_weight'], required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'.cache/huggingface')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--interval', type=int, default=500)
    parser.add_argument('--microbatch', type=int, default=8)
    parser.add_argument('--global-batch', type=int, default=512)
    parser.add_argument('--val-examples', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.global_batch % (2*args.microbatch) or args.val_examples % (2*args.microbatch):
        parser.error('Global batch and validation examples must divide evenly across two GPUs/microbatches')
    if min(args.steps, args.interval, args.workers, args.val_examples) < 1:
        parser.error('Steps, interval, workers and val examples must be positive')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2,3':
        raise RuntimeError('This launcher is restricted to physical CUDA_VISIBLE_DEVICES=2,3')
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Expected exactly the two authorized GPUs')
    args.run = args.run.resolve()
    args.run.mkdir(parents=True, exist_ok=True)
    config = make_config(args)
    cache = verify_cache(config)
    contract = dict(variant=args.variant, seed=config.seed, microbatch=args.microbatch,
                    global_batch=args.global_batch, devices=2, validation_examples=args.val_examples,
                    interval=args.interval, mechanisms=OmegaConf.to_container(config.mechanisms),
                    objective=OmegaConf.to_container(config.objective), cache=cache,
                    upstream_commit='1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7')
    gradient_diagnostics = args.variant == 'mdm_np_zero_init_low_weight'
    if gradient_diagnostics:
        contract['diagnostics'] = 'accumulated_joint_preclip_l2_v1'
    contract_path = args.run/'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise RuntimeError('Run contract changed; use a fresh output directory')
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        contract_path.write_text(json.dumps(contract, indent=2)+'\n')
        OmegaConf.save(config, args.run/'resolved_config.yaml', resolve=True)
    L.seed_everything(config.seed, workers=True)
    tokenizer = dataloader.get_tokenizer(config)
    train, valid = dataloader.get_dataloaders(config, tokenizer)
    # Fixed, non-shuffled monitoring subset; DDP shards this across both GPUs.
    subset = valid.dataset.select(range(args.val_examples))
    valid = torch.utils.data.DataLoader(subset, batch_size=args.microbatch,
        num_workers=args.workers, pin_memory=True,
        generator=torch.Generator().manual_seed(1234))
    model = OWTMDM(config, tokenizer)
    print(f'{args.variant}: params={sum(p.numel() for p in model.parameters()):,}; '
          f'global batch {args.global_batch}=2 x {args.microbatch} x '
          f'{config.trainer.accumulate_grad_batches}; stop={args.steps}; '
          f'validation/checkpoint every {args.interval} optimizer updates', flush=True)
    checkpoint = ModelCheckpoint(dirpath=args.run/'checkpoints',
        filename='step-{step:07d}', auto_insert_metric_name=False,
        every_n_train_steps=args.interval, save_top_k=3, monitor='step', mode='max',
        save_last='link', save_on_train_epoch_end=False)
    callbacks = [LocalMetrics(args.run), checkpoint]
    if gradient_diagnostics:
        callbacks.append(PreclipGradientMetrics(args.run))
    trainer = L.Trainer(accelerator='cuda', devices=2, num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=False), precision='bf16-mixed',
        max_steps=args.steps, max_epochs=-1,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        gradient_clip_val=config.trainer.gradient_clip_val,
        val_check_interval=config.trainer.val_check_interval, check_val_every_n_epoch=None,
        num_sanity_val_steps=0, log_every_n_steps=10, enable_progress_bar=False,
        default_root_dir=str(args.run), callbacks=callbacks,
        logger=CSVLogger(str(args.run), name='lightning_logs'))
    trainer.fit(model, train, valid, ckpt_path=str(args.resume) if args.resume else None)
    if trainer.is_global_zero:
        (args.run/'complete.json').write_text(json.dumps(dict(
            optimizer_step=trainer.global_step, checkpoint=checkpoint.last_model_path), indent=2)+'\n')
        print(f'COMPLETE {args.variant}: {trainer.global_step} optimizer steps', flush=True)


if __name__ == '__main__':
    main()
