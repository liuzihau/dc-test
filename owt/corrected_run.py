"""Continue migrated MDM/A checkpoints with distinct PAD/MASK and epoch-safe resume."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import MethodType
import signal
import subprocess
import sys
import time

from owt.continuation import ROOT,A,digest,continuation_config,tensor_fingerprint,optimizer_fingerprint
from owt.research import atomic_write,timestamp

RUN_ROOT=ROOT/'outputs/owt/corrected-5epochs-20261008'
FILES=['owt/corrected_training.py','owt/migrate_corrected_checkpoint.py','owt/corrected_run.py',
       'owt/test_corrected_training.py','owt/corrected_ddp_check.py']

def use_optimizer_validation_interval(trainer,interval=500):
    """Partial accumulation groups must not move validation off optimizer boundaries."""
    def should_validate(loop,data_fetcher):
        return (loop._should_check_val_epoch() and not loop._should_accumulate()
                and loop.global_step>0 and loop.global_step%interval==0)
    trainer.fit_loop.epoch_loop._should_check_val_fx=MethodType(should_validate,trainer.fit_loop.epoch_loop)

def write(path,value):atomic_write(path,json.dumps(value,indent=2)+'\n')

def verify(path):
    protocol=json.loads(path.read_text())
    if protocol['run_root']!=str(RUN_ROOT) or protocol['order']!=['mdm',A]:raise ValueError('Unexpected paired run')
    if protocol['target_epochs']!=5 or protocol['milestones']!=[15000,25000,50000,85100]:raise ValueError('Unexpected budget')
    for filename,sha in protocol['source_sha256'].items():
        if digest(ROOT/filename)!=sha:raise ValueError('Pinned implementation changed: '+filename)
    return protocol

def expected_state(payload):
    return dict(step=int(payload['global_step']),model=tensor_fingerprint(sorted(payload['state_dict'].items())),
        optimizer=optimizer_fingerprint(payload['optimizer_states'][0]),
        ema=tensor_fingerprint([(str(i),x) for i,x in enumerate(payload['ema']['shadow_params'])]),
        ema_updates=payload['ema']['num_updates'],scheduler_last_epoch=payload['lr_schedulers'][0]['last_epoch'],
        cursor=payload['corrected_training']['cursor'],branch_calls=payload.get('transformer_np',{}).get('calls'))

def worker(path,variant,target,source):
    p=verify(path)
    if os.environ.get('OWT_CORRECTED_LOCK_HELD')!=str(path.resolve()):raise ValueError('Use the controller under the GPU lock')
    from owt.runtime import install
    install()
    import torch
    import lightning as L
    from lightning.pytorch.callbacks import Callback,ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.strategies import DDPStrategy
    from omegaconf import OmegaConf
    import dataloader
    from owt.entrypoint import verify_cache
    from owt.corrected_training import CorrectedMDM,CorrectedTransformerNP,TokenizerAdapter
    from owt.transformer_np_metrics import TransformerLocalMetrics,TransformerGradientMetrics
    from owt.source_pairing_metrics import SourcePairMetrics
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3' or torch.cuda.device_count()!=2:raise ValueError('Use physical GPUs2/3')
    run=RUN_ROOT/variant
    payload=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    expected=expected_state(payload)
    config=continuation_config(payload['hyper_parameters']['config'],run,source)
    config.trainer.max_steps=target
    cache=verify_cache(config)
    old_contract=json.loads((ROOT/p['original_contracts'][variant]).read_text())
    if cache!=old_contract['cache']:raise ValueError('Dataset cache changed')
    L.seed_everything(config.seed,workers=True)
    tokenizer=dataloader.get_tokenizer(config)
    # Use the existing packed data before introducing the new reserved MASK ID.
    train,valid=dataloader.get_dataloaders(config,tokenizer)
    if tokenizer.pad_token_id!=50257:raise ValueError('Cached token IDs differ')
    tokenizer.add_special_tokens({'mask_token':'[MASK]'})
    if tokenizer.mask_token_id!=50258 or len(tokenizer)!=50259:raise ValueError('MASK assignment differs')
    tokenizer=TokenizerAdapter(tokenizer)
    valid=torch.utils.data.DataLoader(valid.dataset.select(range(1024)),batch_size=8,
        num_workers=4,pin_memory=True,generator=torch.Generator().manual_seed(1234))
    actual_updates=(len(train.dataset)+511)//512
    if actual_updates!=17020:raise ValueError('Epoch length changed')
    model=(CorrectedMDM if variant=='mdm' else CorrectedTransformerNP)(config,tokenizer)
    class Audit(Callback):
        def __init__(self):self.digests=[];self.handle=None
        def on_train_start(self,trainer,module):
            optimizer=trainer.optimizers[0].state_dict()
            actual=dict(step=trainer.global_step,model=tensor_fingerprint(sorted(module.state_dict().items())),
                optimizer=optimizer_fingerprint(optimizer),
                ema=tensor_fingerprint([(str(i),x) for i,x in enumerate(module.ema.shadow_params)]),
                ema_updates=module.ema.num_updates,
                scheduler_last_epoch=trainer.lr_scheduler_configs[0].scheduler.last_epoch,
                cursor=module._resume_cursor,branch_calls=getattr(module,'branch_calls',None))
            if actual!=expected:raise ValueError('Full-state restoration differs')
            write(run/f'resume-{target}-rank{trainer.global_rank}.json',dict(verified=True,expected=expected,actual=actual))
            def capture(backbone,inputs):
                if module.training and trainer.global_step<expected['step']+3:
                    h=hashlib.sha256()
                    for value in inputs[:2]:h.update(value.detach().float().cpu().contiguous().numpy().tobytes())
                    self.digests.append(h.hexdigest())
            self.handle=module.backbone.register_forward_pre_hook(capture)
        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            if trainer.global_step==expected['step']+3 and not trainer.fit_loop.epoch_loop._should_accumulate():
                write(run/f'first-draws-{target}-rank{trainer.global_rank}.json',dict(resume_step=expected['step'],
                    first_three_updates=trainer.global_step,microbatch_corruption_sha256=self.digests))
                self.handle.remove();self.handle=None
    del payload
    if int(os.environ.get('LOCAL_RANK','0'))==0:
        run.mkdir(parents=True,exist_ok=True)
        OmegaConf.save(config,run/'resolved_config.yaml',resolve=True)
        write(run/f'contract-{target}.json',dict(resume_checkpoint=str(source),target_step=target,
            restoration=expected,cache=cache,steps_per_epoch=actual_updates,pad_id=50257,mask_id=50258,
            old_training_history='Legacy PAD/MASK collision; corrected training begins at step7500'))
    checkpoint=ModelCheckpoint(dirpath=run/'checkpoints',filename='step-{step:07d}',
        auto_insert_metric_name=False,every_n_train_steps=500,save_top_k=3,monitor='step',mode='max',
        save_last='link',save_on_train_epoch_end=False)
    callbacks=[TransformerLocalMetrics(run),Audit(),checkpoint]
    if variant==A:callbacks += [SourcePairMetrics(run),TransformerGradientMetrics(run)]
    trainer=L.Trainer(accelerator='cuda',devices=2,strategy=DDPStrategy(find_unused_parameters=False),
        precision='bf16-mixed',max_steps=target,max_epochs=5,accumulate_grad_batches=32,
        gradient_clip_val=config.trainer.gradient_clip_val,val_check_interval=config.trainer.val_check_interval,
        check_val_every_n_epoch=None,num_sanity_val_steps=0,log_every_n_steps=10,enable_progress_bar=False,
        default_root_dir=str(run),callbacks=callbacks,logger=CSVLogger(str(run),name='lightning_logs'))
    use_optimizer_validation_interval(trainer)
    trainer.fit(model,train,valid,ckpt_path=str(source))
    if trainer.global_step!=target:raise ValueError('Run ended before the requested milestone')
    if target%500:
        trainer.validate(model,valid,verbose=False)
    # All ranks take part: the checkpoint contains each rank's RNG state.
    milestone=run/'milestones'/f'step-{target:07d}.ckpt'
    milestone.parent.mkdir(parents=True,exist_ok=True)
    trainer.save_checkpoint(milestone)
    if trainer.is_global_zero:
        write(run/f'complete-{target}.json',dict(step=target,epoch=trainer.current_epoch,
            checkpoint=str(milestone),resumed_from=expected['step']))

def controller(path):
    p=verify(path)
    RUN_ROOT.mkdir(parents=True,exist_ok=True)
    def status(stage,**details):write(RUN_ROOT/'queue.json',dict(stage=stage,updated_at=timestamp(),
        controller_pid=os.getpid(),target_epochs=5,target_step=85100,**details))
    with (RUN_ROOT/'.controller.lock').open('a') as controller_lock:
        fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (ROOT/'outputs/owt/mdm-np-5k/.queue.lock').open('a') as gpu_lock:
            while True:
                try:fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:status('waiting_for_gpu_lock');time.sleep(30)
            try:
                for target in p['milestones']:
                    for variant in p['order']:
                        run=RUN_ROOT/variant;receipt=run/f'complete-{target}.json'
                        if receipt.exists():
                            if not Path(json.loads(receipt.read_text())['checkpoint']).exists():raise ValueError('Milestone missing')
                            continue
                        previous=[s for s in p['milestones'] if s<target]
                        source=(run/'milestones'/f'step-{previous[-1]:07d}.ckpt') if previous else ROOT/p['checkpoints'][variant]['path']
                        if not previous and digest(source)!=p['checkpoints'][variant]['sha256']:raise ValueError('Migrated checkpoint changed')
                        if not source.exists():raise FileNotFoundError(source)
                        run.mkdir(parents=True,exist_ok=True)
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                            OPENBLAS_NUM_THREADS='2',OWT_CORRECTED_LOCK_HELD=str(path.resolve()))
                        with (run/'train.log').open('a') as log:
                            child=subprocess.Popen([sys.executable,'-u','-m','owt.corrected_run','--protocol',str(path),
                                '--worker','--variant',variant,'--target',str(target),'--source',str(source)],
                                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                            status('training',variant=variant,milestone=target,worker_pid=child.pid,source=str(source))
                            try:
                                if child.wait()!=0 or not receipt.exists():raise RuntimeError('Corrected continuation failed')
                            except BaseException:
                                if child.poll() is None:
                                    os.killpg(child.pid,signal.SIGTERM)
                                    try:child.wait(timeout=30)
                                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                                raise
                status('complete')
            except BaseException as error:status('failed_requires_review',error=str(error));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--variant',choices=['mdm',A]);parser.add_argument('--target',type=int);parser.add_argument('--source',type=Path)
    args=parser.parse_args()
    if args.worker:worker(args.protocol.resolve(),args.variant,args.target,args.source.resolve())
    else:controller(args.protocol.resolve())
