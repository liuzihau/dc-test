"""User-selected MDM-only continuation from7500 to15000 updates."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from owt.continuation import ROOT,digest,verify_selection,continuation_config,tensor_fingerprint,optimizer_fingerprint
from owt.research import atomic_write,read_json,read_csv,timestamp

RUN=ROOT/'outputs/owt/continuation-15000/mdm'
SOURCE=ROOT/'outputs/owt/continuation-7500/mdm'
FILES=['owt/mdm_15000.py','owt/mdm_15000_metrics.py','owt/test_mdm_15000.py']


def write(path,data):atomic_write(path,json.dumps(data,indent=2)+'\n')


def config_for_resume(saved,run,checkpoint):
    config=continuation_config(saved,run,checkpoint)
    config.trainer.max_steps=15000
    return config


def audit_checkpoint(p):
    c=p['hyper_parameters']['config']
    if p['global_step']!=7500 or p['ema']['num_updates']!=7500 or c['mechanisms']['np']['enabled']:
        raise ValueError('Require MDM-only EMA7500 checkpoint')
    if c['loader']['batch_size']!=8 or c['loader']['global_batch_size']!=512 or c['trainer']['devices']!=2 or c['trainer']['accumulate_grad_batches']!=32:
        raise ValueError('Batch settings changed')
    if c['model']['length']!=1024 or c['seed']!=1 or c['trainer']['max_steps']!=7500 or c['lr_scheduler']['num_warmup_steps']!=2500:
        raise ValueError('Model/seed/scheduler changed')
    if c['lr_scheduler']['_target_']!='transformers.get_constant_schedule_with_warmup':raise ValueError('Unexpected scheduler')
    if len(p['optimizer_states'])!=1 or len(p['lr_schedulers'])!=1:raise ValueError('Missing optimizer/scheduler')
    optimizer=p['optimizer_states'][0]
    if {int(v['step']) for v in optimizer['state'].values()}!={7500}:
        raise ValueError('Adam states not aligned to7500')
    if any(not {'exp_avg','exp_avg_sq'}.issubset(v) for v in optimizer['state'].values()):raise ValueError('Missing Adam moments')
    if p['lr_schedulers'][0]['last_epoch']!=7500 or any(g['lr']!=.0003 for g in optimizer['param_groups']):
        raise ValueError('Scheduler/learning rate changed')
    loops=p['loops']['fit_loop']
    if loops['epoch_loop.batch_progress']['current']['completed']!=240000 or loops['epoch_progress']['current']['completed']!=0:
        raise ValueError('Unsupported sampler/epoch cursor')
    return dict(step=7500,model=tensor_fingerprint(sorted(p['state_dict'].items())),
        optimizer=optimizer_fingerprint(optimizer),
        ema=tensor_fingerprint([(str(i),x) for i,x in enumerate(p['ema']['shadow_params'])]),
        ema_updates=7500,scheduler_last_epoch=7500,learning_rates=[.0003],
        fast_forward_batches=240000,fast_forward_epochs=0)


def verify_protocol(path):
    p=read_json(path)
    if not p or p.get('execution_ready') is not True or not p.get('user_instruction') or p['resume_step']!=7500 or p['target_step']!=15000 or p['variant']!='mdm' or p['run']!=str(RUN):
        raise ValueError('Require user-selected baseline-only7500→15000 protocol')
    verify_selection(ROOT/'outputs/research-notes/continuation_7500_selection_20261006_v2.json')
    for name,sha in p['source_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Continuation source changed')
    for name,sha in p['evidence_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Continuation evidence changed')
    if p['checkpoint']['path']!=str((SOURCE/'checkpoints/step-0007500.ckpt').relative_to(ROOT)):
        raise ValueError('Resume must use the completed MDM7500 checkpoint')
    checkpoint=ROOT/p['checkpoint']['path'];stat=checkpoint.stat()
    if stat.st_size!=p['checkpoint']['bytes'] or stat.st_mtime_ns!=p['checkpoint']['mtime_ns']:
        raise ValueError('7500 checkpoint changed')
    return p


def worker(path):
    p=verify_protocol(path)
    if os.environ.get('MDM_15000_LOCK_HELD')!=str(path.resolve()):raise ValueError('Use the GPU-lock controller')
    from owt.runtime import install
    install()
    import torch
    import lightning as L
    from omegaconf import OmegaConf
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.strategies import DDPStrategy
    import dataloader
    from owt.entrypoint import verify_cache
    from owt.model import OWTMDM
    from owt.transformer_np_metrics import TransformerLocalMetrics
    from owt.mdm_15000_metrics import ResumeAudit
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3' or torch.cuda.device_count()!=2:raise ValueError('Use GPUs2/3 only')
    payload=torch.load(ROOT/p['checkpoint']['path'],map_location='cpu',mmap=True,weights_only=False)
    expected=audit_checkpoint(payload)
    config=config_for_resume(payload['hyper_parameters']['config'],RUN,ROOT/p['checkpoint']['path'])
    cache=verify_cache(config)
    previous=read_json(SOURCE/'contract.json')
    if cache!=previous['original_contract']['cache']:raise ValueError('Original data cache changed')
    L.seed_everything(config.seed,workers=True)
    tokenizer=dataloader.get_tokenizer(config);train,valid=dataloader.get_dataloaders(config,tokenizer)
    if 15000*32*8>=(len(train.dataset)+1)//2:raise ValueError('Target crosses the supported first epoch')
    valid=torch.utils.data.DataLoader(valid.dataset.select(range(1024)),batch_size=8,num_workers=4,pin_memory=True,
        generator=torch.Generator().manual_seed(1234))
    model=OWTMDM(config,tokenizer);del payload
    if int(os.environ.get('LOCAL_RANK','0'))==0:
        OmegaConf.save(config,RUN/'resolved_config.yaml',resolve=True)
        write(RUN/'contract.json',dict(previous_contract=previous,resume_step=7500,target_step=15000,cache=cache,
            source_checkpoint=p['checkpoint'],restoration_expected=expected,training_recipe_changed=False))
    ckpt=ModelCheckpoint(dirpath=RUN/'checkpoints',filename='step-{step:07d}',auto_insert_metric_name=False,
        every_n_train_steps=500,save_top_k=3,monitor='step',mode='max',save_last='link',save_on_train_epoch_end=False)
    trainer=L.Trainer(accelerator='cuda',devices=2,num_nodes=1,strategy=DDPStrategy(find_unused_parameters=False),
        precision='bf16-mixed',max_steps=15000,max_epochs=-1,accumulate_grad_batches=32,
        gradient_clip_val=config.trainer.gradient_clip_val,val_check_interval=config.trainer.val_check_interval,
        check_val_every_n_epoch=None,num_sanity_val_steps=0,log_every_n_steps=10,enable_progress_bar=False,
        default_root_dir=str(RUN),callbacks=[TransformerLocalMetrics(RUN),ResumeAudit(RUN,expected),ckpt],
        logger=CSVLogger(str(RUN),name='lightning_logs'))
    trainer.fit(model,train,valid,ckpt_path=str(ROOT/p['checkpoint']['path']))
    if trainer.is_global_zero:
        if trainer.global_step!=15000:raise ValueError('Did not finish15000')
        write(RUN/'complete.json',dict(optimizer_step=15000,resumed_from=7500,checkpoint=ckpt.last_model_path))


def controller(path):
    p=verify_protocol(path);root=RUN.parent;root.mkdir(parents=True,exist_ok=True)
    def status(stage,**details):write(root/'queue.json',dict(stage=stage,updated_at=timestamp(),controller_pid=os.getpid(),
        variant='mdm',resume_step=7500,target_step=15000,**details))
    with (root/'.controller.lock').open('a') as controller_lock:
        fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            with (ROOT/'outputs/owt/mdm-np-5k/.queue.lock').open('a') as lock:
                while True:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                    except BlockingIOError:status('waiting_for_gpu_lock');time.sleep(30)
                verify_protocol(path)
                if digest(ROOT/p['checkpoint']['path'])!=p['checkpoint']['sha256']:raise ValueError('Checkpoint checksum changed')
                if RUN.exists():raise FileExistsError('Preserve previous/partial15000 run')
                RUN.mkdir();(RUN/'local_metrics').mkdir()
                for filename in ('train.csv','validation.csv'):
                    source=SOURCE/'local_metrics'/filename
                    if read_csv(source)[-1]['optimizer_step']!=7500:raise ValueError('Metric prefix incomplete')
                    shutil.copy2(source,RUN/'local_metrics'/filename)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                    MDM_15000_LOCK_HELD=str(path.resolve()))
                with (RUN/'train.log').open('a') as log:
                    child=subprocess.Popen([sys.executable,'-u','-m','owt.mdm_15000','--protocol',str(path.resolve()),'--worker'],
                        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    reviewed=False;status('training',worker_pid=child.pid,startup_verified=False)
                    try:
                        while child.poll() is None:
                            rows=[r for r in read_csv(RUN/'local_metrics/train.csv') if r['optimizer_step']>7500][:3]
                            if not reviewed and len(rows)==3:
                                if [r['optimizer_step'] for r in rows]!=[7501,7502,7503]:raise ValueError('Wrong first update')
                                for row in rows:
                                    if abs(row['objective']-row['main_elbo'])>1e-6 or row['learning_rate']!=.0003:raise ValueError('MDM recipe changed')
                                for rank in (0,1):
                                    if not read_json(RUN/f'resume-rank{rank}.json'):raise ValueError('Rank resume audit missing')
                                reviewed=True;write(RUN/'startup_review.json',dict(restoration_verified=True,steps=[7501,7502,7503],
                                    main_only=True,learning_rate=.0003,optimizer_ema_scheduler_sampler_restored=True))
                                status('training',worker_pid=child.pid,startup_verified=True)
                            time.sleep(10)
                        if child.wait() or (read_json(RUN/'complete.json') or {}).get('optimizer_step')!=15000:
                            raise RuntimeError('MDM continuation failed')
                    except BaseException:
                        if child.poll() is None:
                            os.killpg(child.pid,signal.SIGTERM)
                            try:child.wait(timeout=30)
                            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        raise
                status('complete',next_training_launched=False)
        except BaseException as error:status('failed_requires_review',error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    (worker if args.worker else controller)(args.protocol)
