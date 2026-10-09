"""Run the paired CPU diagnostic at5000, then at7500 after training completes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from owt.continuation import ROOT,A,digest
from owt.research import read_json,atomic_write,timestamp

OUT=ROOT/'outputs/analysis/owt-local-top5-mdm-A-20261006'
FILES=['analysis/run_local_top5_pair.py','analysis/local_top5_metrics.py',
    'analysis/plot_local_top5_pair.py','analysis/test_local_top5_metrics.py',
    'analysis/schedule_local_top5_pair.py','analysis/local_denoising_metrics.py']


def status(root,stage,**details):
    atomic_write(root/'queue.json',json.dumps(dict(stage=stage,updated_at=timestamp(),
        controller_pid=os.getpid(),checkpoint_steps=[5000,7500],device='cpu',cpu_threads=2,
        reveal_probabilities=[.25,.5,.75],samples=100,corruption_seeds=10,**details),indent=2)+'\n')


def verify_pins(protocol):
    for file,sha in protocol['source_sha256'].items():
        if digest(ROOT/file)!=sha:raise RuntimeError('Diagnostic source changed: '+file)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'.controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol_path=root/'schedule_protocol.json'
        if protocol_path.exists():protocol=read_json(protocol_path)
        else:
            protocol=dict(created_at=timestamp(),checkpoint_steps=[5000,7500],
                source_sha256={file:digest(ROOT/file) for file in FILES},
                user_instruction='Can we redo the 16-grid test at25%,50%,75% reveal, with the agreed two-level error categories?',
                agreed_setup='MDM and A, EMA/FP32, same100 independent OWT documents and10 seeds; repeat matched7500 checkpoints',
                gpu_training_unchanged=True,probe_head_training=False)
            atomic_write(protocol_path,json.dumps(protocol,indent=2)+'\n')
        try:
            verify_pins(protocol)
            for step in (5000,7500):
                run=root/f'step{step}'
                if (read_json(run/'summary.json') or {}).get('stage')=='complete':continue
                if run.exists():raise RuntimeError('Preserve partial collection; inspect before retry')
                if step==7500:
                    status(root,'waiting_for_both_7500')
                    while not all((read_json(ROOT/'outputs/owt/continuation-7500'/v/'complete.json') or {}).get('optimizer_step')==7500 for v in ('mdm',A)):
                        q=read_json(ROOT/'outputs/owt/continuation-7500/queue.json') or {}
                        if 'failed' in q.get('stage',''):raise RuntimeError('Training continuation failed before matched7500 evaluation')
                        time.sleep(30)
                verify_pins(protocol)
                status(root,'evaluating',checkpoint_step=step,run=str(run.relative_to(ROOT)))
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                    OPENBLAS_NUM_THREADS='2',MPLCONFIGDIR=str(ROOT/'.cache/runtime/local-top5-mpl'))
                with (root/f'step{step}.log').open('a') as log:
                    command=[sys.executable,'-u','-m','analysis.run_local_top5_pair',
                        '--checkpoint-step',str(step),'--output',str(run)]
                    subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                if (read_json(run/'summary.json') or {}).get('stage')!='complete':
                    raise RuntimeError('Diagnostic completion receipt missing')
            status(root,'complete')
        except BaseException as error:
            status(root,'failed',error=str(error));raise


if __name__=='__main__':main()
