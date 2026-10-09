"""Prepare and run the agreed clean-source frozen-probe study on GPUs2/3."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/runtime/clean-probe-mpl'))
from owt.continuation import A,ORIGINALS,digest,verify_selection
from owt.research import atomic_write,read_json,timestamp

FILES=['analysis/clean_neighbor_probe.py','analysis/prepare_clean_probe_data.py',
    'analysis/clean_probe_worker.py','analysis/clean_probe_report.py',
    'analysis/run_clean_neighbor_probes.py','analysis/test_clean_neighbor_probe.py']
OUTPUT=ROOT/'outputs/analysis/owt-clean-neighbor-probes-20261007'


def write(path,data):atomic_write(path,json.dumps(data,indent=2,allow_nan=False)+'\n')


def prepare(args):
    from analysis.prepare_clean_probe_data import prepare as prepare_data
    root=args.output.resolve()
    if root.exists():raise FileExistsError('Preserve earlier/partial probe runs')
    root.mkdir(parents=True)
    selection=verify_selection(ROOT/'outputs/research-notes/continuation_7500_selection_20261006_v2.json')
    for v in ('mdm',A):
        if (read_json(ROOT/'outputs/owt/continuation-7500'/v/'complete.json') or {}).get('optimizer_step')!=7500:
            raise ValueError('Both7500 checkpoints must finish')
    sizes=(4,2,2) if args.smoke else (200,50,100)
    data=prepare_data(root/'data',*sizes)
    p=dict(created_at=timestamp(),output=str(root),smoke=args.smoke,checkpoint_steps=[7500] if args.smoke else [7500,5000],
        models=['MDM','A'],offsets=[-2,-1,1,2],source='revealed',target='masked',backbone_frozen=True,
        parameter_state='EMA',layer='before last main transformer block',feature_precision='FP32',head_precision='FP32',
        trained_modules='four new independent Linear(768,50258) heads per model; no transformer or backbone updates',
        data_path=str((root/'data/data.npz').relative_to(ROOT)),data_sha256=data['data_sha256'],
        documents=dict(zip(('train','development','evaluation'),sizes)),
        corruption_seeds=dict(train=[100] if args.smoke else [100,101],development=[200],evaluation=[0] if args.smoke else list(range(10))),
        reveal_rates=[.5] if args.smoke else [.25,.5,.75],probe_seeds=[0] if args.smoke else [0,1,2],
        learning_rates=[.0003] if args.smoke else [.0001,.0003,.001],weight_decay=.001,
        batch_size=512,max_epochs=2 if args.smoke else 20,early_stop_patience=3,early_stop_min_delta=.0001,
        learning_rate_selection='lowest mean development macro CE across MDM and A at7500; reuse unchanged at5000',
        objective='equal-weight mean of four per-head eligible-pair CE averages',
        primary='equal-weight offset mean CE at50% reveal,7500 checkpoint',
        bootstrap_draws=100 if args.smoke else 2000,bootstrap_unit='document, all10 corruption seeds together',
        physical_gpus=[2,3],gpu_lock='outputs/owt/mdm-np-5k/.queue.lock',
        source_sha256={file:digest(ROOT/file) for file in FILES},
        production_source_sha256=selection['source_sha256'],checkpoints={},
        data_receipt=data,autonomous_goal_resumed=False)
    for step in p['checkpoint_steps']:
        p['checkpoints'][str(step)]={}
        for v,model in [('mdm','MDM'),(A,'A')]:
            run=ROOT/ORIGINALS[v] if step==5000 else ROOT/'outputs/owt/continuation-7500'/v
            checkpoint=run/'checkpoints'/f'step-{step:07d}.ckpt'
            p['checkpoints'][str(step)][model]=dict(path=str(checkpoint.relative_to(ROOT)),sha256=digest(checkpoint),
                config=str((run/'resolved_config.yaml').relative_to(ROOT)),step=step)
    write(root/'protocol.json',p)
    write(root/'progress.json',dict(stage='prepared',checkpoint_steps=p['checkpoint_steps'],models=p['models']))
    return p


def launch_pair(p,step,action,seed=0):
    root=Path(p['output']);protocol=root/'protocol.json'
    children=[];logs=[]
    try:
        for model,gpu in [('MDM','2'),('A','3')]:
            run=root/f'step{step}'/model;run.mkdir(parents=True,exist_ok=True)
            log=(run/f'{action}-seed{seed}.log').open('a');logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,CLEAN_PROBE_GPU_LOCK_HELD=str(protocol.resolve()))
            command=[sys.executable,'-u','-m','analysis.clean_probe_worker','--protocol',str(protocol),
                '--model',model,'--step',str(step),'--action',action,'--seed',str(seed)]
            children.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        write(root/'progress.json',dict(stage=action,checkpoint_step=step,seed=seed,
            worker_pids={m:c.pid for m,c in zip(('MDM','A'),children)},updated_at=timestamp()))
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('A probe worker failed; inspect its log')
            time.sleep(5)
        if any(c.returncode for c in children):raise RuntimeError('A probe worker failed; inspect its log')
    except BaseException:
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        for child in children:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise
    finally:
        for log in logs:log.close()


def check_caches(p,step):
    root=Path(p['output'])/f'step{step}'
    for split in ('train','development','evaluation'):
        a=read_json(root/'A/cache'/split/'complete.json');b=read_json(root/'MDM/cache'/split/'complete.json')
        if a['sources']!=b['sources'] or a['width']!=b['width']:raise ValueError('Paired cache shape differs')
        for name in ('targets','documents','levels','source_tokens'):
            if a['files'][name]!=b['files'][name]:raise ValueError('Paired sources/targets differ')


def run(p):
    root=Path(p['output'])
    for file,sha in {**p['source_sha256'],**p['production_source_sha256']}.items():
        if digest(ROOT/file)!=sha:raise ValueError('Pinned source changed')
    from analysis.clean_probe_report import report
    for step in p['checkpoint_steps']:
        launch_pair(p,step,'cache');check_caches(p,step)
        if step==p['checkpoint_steps'][0]:
            launch_pair(p,step,'calibrate')
            candidates=[]
            for lr in p['learning_rates']:
                values=[read_json(root/f'step{step}'/m/'fits'/('calibration-'+format(lr,'.0e'))/'complete.json')['best_development_ce'] for m in ('MDM','A')]
                candidates.append(dict(learning_rate=lr,mean_development_ce=sum(values)/2,model_development_ce=values))
            selected=min(candidates,key=lambda d:d['mean_development_ce'])
            write(root/'selected_hyperparameters.json',dict(**selected,candidates=candidates,
                evaluation_used_for_selection=False,selection_checkpoint=step))
        for seed in p['probe_seeds']:
            launch_pair(p,step,'fit',seed)
            fits=[read_json(root/f'step{step}'/m/'fits'/f'seed-{seed}'/'complete.json') for m in ('MDM','A')]
            if fits[0]['sampler_first_batches']!=fits[1]['sampler_first_batches']:
                raise ValueError('Paired fitting minibatches differ')
        report(root,step,p)
        write(root/f'step{step}'/'complete.json',dict(checkpoint_step=step,models=['MDM','A'],seeds=p['probe_seeds'],
            only_new_heads_trained=True,backbone_frozen=True,paired_sources_and_targets_verified=True))
    write(root/'progress.json',dict(stage='complete',checkpoint_steps=p['checkpoint_steps'],updated_at=timestamp()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--prepared',type=Path)
    args=parser.parse_args()
    p=read_json(args.prepared) if args.prepared else prepare(args)
    if args.prepare_only:print(p['output']);raise SystemExit(0)
    root=Path(p['output'])
    with (root/'.controller.lock').open('a') as controller:
        fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            with (ROOT/p['gpu_lock']).open('a') as gpu_lock:
                try:fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    write(root/'progress.json',dict(stage='waiting_for_gpu_lock'))
                    fcntl.flock(gpu_lock,fcntl.LOCK_EX)
                run(p)
        except BaseException as error:
            write(root/'progress.json',dict(stage='failed',error=str(error),updated_at=timestamp()));raise
