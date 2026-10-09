"""Resume the selected MDM or A checkpoint to absolute optimizer step7500."""
import argparse
import json
import os
from pathlib import Path
import sys

from owt.continuation import (ROOT,RUN_ROOT,A,ORIGINALS,verify_selection,
    continuation_config,inspect_checkpoint,tensor_fingerprint,optimizer_fingerprint)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--variant',choices=['mdm',A],required=True)
    args=parser.parse_args()
    selection=verify_selection(args.selection)
    if os.environ.get('OWT_CONTINUATION_AUTHORIZED')!=str(args.selection.resolve()):
        raise ValueError('Launch through continuation_schedule under the GPU lock')
    from owt.runtime import install
    install()
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.strategies import DDPStrategy
    from omegaconf import OmegaConf
    import torch
    import dataloader
    from owt.entrypoint import verify_cache
    from owt.model import OWTMDM
    from owt.transformer_np_model import TransformerNPMDM
    from owt.transformer_np_metrics import TransformerLocalMetrics,TransformerGradientMetrics
    from owt.source_pairing_metrics import SourcePairMetrics
    from owt.continuation_metrics import ContinuationAudit
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3' or torch.cuda.device_count()!=2:
        raise ValueError('Continuation requires physical GPUs2/3')
    run=ROOT/RUN_ROOT/args.variant
    checkpoint=ROOT/selection['checkpoints'][args.variant]['path']
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    audit=inspect_checkpoint(payload,args.variant)
    config=continuation_config(OmegaConf.create(payload['hyper_parameters']['config']),run,checkpoint)
    cache=verify_cache(config)
    original=json.loads((ROOT/ORIGINALS[args.variant]/'contract.json').read_text())
    if cache!=original['cache']:
        raise ValueError('Continuation dataset changed')
    L.seed_everything(config.seed,workers=True)
    tokenizer=dataloader.get_tokenizer(config)
    train,valid=dataloader.get_dataloaders(config,tokenizer)
    valid=torch.utils.data.DataLoader(valid.dataset.select(range(1024)),batch_size=8,
        num_workers=4,pin_memory=True,generator=torch.Generator().manual_seed(1234))
    model=(OWTMDM if args.variant=='mdm' else TransformerNPMDM)(config,tokenizer)
    if args.variant==A and model.resume_signature()!=payload['transformer_np']['signature']:
        raise ValueError('A architecture must remain identical')
    expected=dict(step=5000,model=tensor_fingerprint(sorted(payload['state_dict'].items())),
        optimizer=optimizer_fingerprint(payload['optimizer_states'][0]),
        ema=tensor_fingerprint([(str(i),x) for i,x in enumerate(payload['ema']['shadow_params'])]),
        ema_updates=payload['ema']['num_updates'],scheduler_last_epoch=5000,
        learning_rates=[.0003],fast_forward_batches=160000,fast_forward_epochs=0,
        branch_calls=payload.get('transformer_np',{}).get('calls'))
    del payload
    if int(os.environ.get('LOCAL_RANK','0'))==0:
        run.mkdir(parents=True,exist_ok=True)
        OmegaConf.save(config,run/'resolved_config.yaml',resolve=True)
        contract=dict(original_contract=original,variant=args.variant,resume_step=5000,target_step=7500,
            source_checkpoint=selection['checkpoints'][args.variant],continuation_seed=750001,
            source_header_audit=audit,restoration_expected=expected)
        (run/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    checkpoint_callback=ModelCheckpoint(dirpath=run/'checkpoints',filename='step-{step:07d}',
        auto_insert_metric_name=False,every_n_train_steps=500,save_top_k=3,monitor='step',mode='max',
        save_last='link',save_on_train_epoch_end=False)
    callbacks=[TransformerLocalMetrics(run),ContinuationAudit(run,expected),checkpoint_callback]
    if args.variant==A:
        callbacks += [SourcePairMetrics(run),TransformerGradientMetrics(run)]
    trainer=L.Trainer(accelerator='cuda',devices=2,num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=False),precision='bf16-mixed',
        max_steps=7500,max_epochs=-1,accumulate_grad_batches=32,
        gradient_clip_val=config.trainer.gradient_clip_val,
        val_check_interval=config.trainer.val_check_interval,check_val_every_n_epoch=None,
        num_sanity_val_steps=0,log_every_n_steps=10,enable_progress_bar=False,
        default_root_dir=str(run),callbacks=callbacks,logger=CSVLogger(str(run),name='lightning_logs'))
    trainer.fit(model,train,valid,ckpt_path=str(checkpoint))
    if trainer.is_global_zero:
        if trainer.global_step!=7500:
            raise ValueError('Continuation did not finish step7500')
        (run/'complete.json').write_text(json.dumps(dict(optimizer_step=7500,
            checkpoint=checkpoint_callback.last_model_path,resumed_from=5000),indent=2)+'\n')


if __name__=='__main__': main()
