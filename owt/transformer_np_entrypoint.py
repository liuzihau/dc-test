"""Fresh aligned 5000-update transformer-NP trial on GPUs 2/3."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from owt.transformer_np_schedule import (ROOT, RUN_ROOT, VARIANTS,
                                         entrypoint_authorization)
if __name__ == '__main__' and '--help' not in sys.argv:
    entrypoint_authorization()

from owt.runtime import install
install()
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf
import torch

import dataloader
from owt.entrypoint import make_config, verify_cache
from owt.source_pairing_metrics import SourcePairMetrics
from owt.transformer_np_metrics import TransformerGradientMetrics, TransformerLocalMetrics
from owt.transformer_np_model import TransformerNPMDM


def main():
    sys.argv[0] = str(Path(__file__).resolve())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--np-weight', type=float, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / '.cache/huggingface')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    selection = entrypoint_authorization()
    if args.variant != selection['selected_variant'] or args.np_weight != selection['np_weight_per_direction']:
        parser.error('Variant/weight differs from the selected trial')
    if args.run.resolve() != ROOT / RUN_ROOT / args.variant:
        parser.error('Run path differs from the isolated selected trial')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2,3' or torch.cuda.device_count() != 2:
        raise RuntimeError('Transformer NP requires exactly physical GPUs 2,3')
    args.steps, args.interval, args.microbatch = 5000, 500, 8
    args.global_batch, args.val_examples, args.workers = 512, 1024, 4
    args.run = args.run.resolve()
    args.run.mkdir(parents=True, exist_ok=True)
    config = make_config(args)
    config.mechanisms.np.weights = [args.np_weight, args.np_weight]
    cache = verify_cache(config)
    reference = json.loads((ROOT / 'outputs/owt/mdm-np-5k/mdm/contract.json').read_text())
    if cache != reference['cache']:
        raise ValueError('Transformer trial must use the exact aligned MDM data caches')
    L.seed_everything(config.seed, workers=True)
    tokenizer = dataloader.get_tokenizer(config)
    train, valid = dataloader.get_dataloaders(config, tokenizer)
    valid = torch.utils.data.DataLoader(valid.dataset.select(range(args.val_examples)),
        batch_size=args.microbatch, num_workers=args.workers, pin_memory=True,
        generator=torch.Generator().manual_seed(1234))
    model = TransformerNPMDM(config, tokenizer)
    parameters = dict(total=sum(p.numel() for p in model.parameters()),
                      neighbor_processing=sum(p.numel() for p in model.backbone.neighbor_branches.parameters()),
                      neighbor_readouts=sum(p.numel() for p in model.backbone.neighbor_heads.parameters()))
    contract = dict(variant=args.variant, seed=config.seed, microbatch=args.microbatch,
        global_batch=args.global_batch, devices=2, validation_examples=args.val_examples,
        interval=args.interval, mechanisms=OmegaConf.to_container(config.mechanisms),
        objective=OmegaConf.to_container(config.objective), cache=cache, parameters=parameters,
        architecture=model.resume_signature(), diagnostics='transformer_np_joint_preclip_l2_v1',
        source_diagnostics='per_direction_maskbin_pair_counts_and_weight_mass_v1',
        upstream_commit='1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7')
    path = args.run / 'contract.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Run contract changed; do not mix architecture or loss settings')
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        path.write_text(json.dumps(contract, indent=2) + '\n')
        OmegaConf.save(config, args.run / 'resolved_config.yaml', resolve=True)
    print(f'{args.variant}: params={parameters}; global batch=512; steps=5000; '
          f'NP weight={args.np_weight} each; policy={config.mechanisms.np.source_policy}', flush=True)
    checkpoint = ModelCheckpoint(dirpath=args.run / 'checkpoints',
        filename='step-{step:07d}', auto_insert_metric_name=False,
        every_n_train_steps=args.interval, save_top_k=3, monitor='step', mode='max',
        save_last='link', save_on_train_epoch_end=False)
    trainer = L.Trainer(accelerator='cuda', devices=2, num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=False), precision='bf16-mixed',
        max_steps=args.steps, max_epochs=-1,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        gradient_clip_val=config.trainer.gradient_clip_val,
        val_check_interval=config.trainer.val_check_interval, check_val_every_n_epoch=None,
        num_sanity_val_steps=0, log_every_n_steps=10, enable_progress_bar=False,
        default_root_dir=str(args.run), callbacks=[TransformerLocalMetrics(args.run),
            checkpoint, SourcePairMetrics(args.run), TransformerGradientMetrics(args.run)],
        logger=CSVLogger(str(args.run), name='lightning_logs'))
    trainer.fit(model, train, valid, ckpt_path=str(args.resume) if args.resume else None)
    if trainer.is_global_zero:
        (args.run / 'complete.json').write_text(json.dumps(dict(
            optimizer_step=trainer.global_step, checkpoint=checkpoint.last_model_path), indent=2) + '\n')


if __name__ == '__main__':
    main()
