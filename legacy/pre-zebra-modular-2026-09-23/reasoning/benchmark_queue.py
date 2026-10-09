"""Bounded fresh benchmark runs; never resume old synthetic/layout checkpoints."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .benchmark import PROTOCOL
from .runner import atomic_json, digest
from .zebra_continuation import verified_checkpoint

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('vanilla', 'mdm', 'mdm_aux', 'both', 'both_aux')
DISPLAY_NAMES = {'vanilla': 'MDM', 'mdm': 'TT', 'mdm_aux': 'TT + NP',
                 'both': 'TT + RM', 'both_aux': 'TT + RM + NP'}
TASKS = ('zebra-benchmark', 'sudoku-benchmark')


def training_command(task, variant, data, run, steps=5000, micro_batch=128):
    command = [sys.executable, '-u', str(ROOT/'scripts/reasoning/run_reasoning.py'),
               'train', '--task', task, '--variant', variant, '--data-dir', str(data),
               '--run-dir', str(run), '--size', 'mini', '--max-steps', str(steps),
               '--global-batch', '128', '--micro-batch', str(micro_batch),
               '--seed', '1', '--eval-seed', '2026', '--val-every', '500',
               '--save-every', '500', '--save-seconds', '1200']
    if variant == 'vanilla':
        command += ['--gradient-mode', 'detached', '--no-robustness']
    else:
        command += ['--gradient-mode', 'adjacent', '--merged-policy', 'current_preserving']
    return command


def report(directory, title='Released-data reconstruction v2 — same test IDs; final checkpoint',
           variants=VARIANTS, display_names=DISPLAY_NAMES, suite=None):
    """Never mix legacy scores; require paired examples and data/protocol hashes."""
    rows, expected = [], {}
    for path in sorted(Path(directory).glob('*/*/generation.json')):
        r = json.loads(path.read_text())
        if r.get('benchmark_protocol') != PROTOCOL:
            raise ValueError('Unverified evaluation protocol: '+str(path))
        task = r['contract']['task']; variant = r['contract']['variant']
        if task not in TASKS or variant not in variants:
            raise ValueError('Unexpected benchmark task/variant')
        if suite is not None and r['contract'].get('suite') != suite:
            raise ValueError('Evaluation belongs to a different architecture suite')
        metrics = r['metrics']
        if (metrics['policy'] != 'top_prob' or metrics['candidate_k'] != 8
                or metrics['token_selection'] != 'paper' or metrics['tokens_per_step'] != 1
                or metrics['memory_condition'] != 'correct' or metrics['seed'] != 2026):
            raise ValueError('Generation settings differ')
        ids = [e['id'] for e in r['examples']]
        signature = (r['contract']['data_sha256'], ids, r['step'])
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Empty/duplicate evaluation IDs')
        if task in expected and signature != expected[task]:
            raise ValueError('Comparisons require same data, test IDs/order and checkpoint step')
        expected[task] = signature
        n = len(ids); successes = sum(bool(e['scores']['valid_solution']) for e in r['examples'])
        p = successes/n; z = 1.96; den = 1+z*z/n
        center = (p+z*z/(2*n))/den
        half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        rows.append(dict(task=task, variant=variant, step=r['step'], examples=n,
                         valid_solution=p, ci95_low=center-half, ci95_high=center+half,
                         strict_sequence_success=sum(bool(e['scores']['strict_sequence_success']) for e in r['examples'])/n))
    if not rows:
        return
    out = Path(directory)/'report'; out.mkdir(exist_ok=True)
    with (out/'accuracy.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for ax, task in zip(axes,TASKS):
        subset=sorted([r for r in rows if r['task']==task], key=lambda r: variants.index(r['variant']))
        vals=[100*r['valid_solution'] for r in subset]
        errors=[[100*(r['valid_solution']-r['ci95_low']) for r in subset],
                [100*(r['ci95_high']-r['valid_solution']) for r in subset]]
        if subset:
            ax.bar([display_names[r['variant']].replace(' + ', '\n+ ') for r in subset],vals,yerr=errors,capsize=4)
        ax.set(title=task, ylabel='Whole-grid accuracy (%)', ylim=(0,100))
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(out/'accuracy.png',dpi=180); plt.close(fig)
    atomic_json(out/'protocol.json',PROTOCOL)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('plan','run','report'))
    p.add_argument('--data-root',type=Path,default=ROOT/'.cache/reasoning')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/reasoning/benchmark-v2')
    p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--micro-batch',type=int,default=128)
    p.add_argument('--gpu',default='0')
    p.add_argument('--hours',type=float,default=10)
    p.add_argument('--deadline-from',type=Path,help='Do not extend an already authorized queue budget')
    a=p.parse_args(argv)
    if a.steps<1 or a.micro_batch<1 or 128%a.micro_batch or not math.isfinite(a.hours) or a.hours<=0:
        raise ValueError('Positive steps/hours and a microbatch dividing 128 required')
    if a.action=='report':
        report(a.output); return
    commands=[]
    for task in TASKS:
        data=a.data_root/(task+'-v2-n20000-v1000-t1000')
        for variant in VARIANTS:
            run=a.output/task/variant
            commands.append((task,variant,data,run,training_command(task,variant,data,run,a.steps,a.micro_batch)))
    if a.action=='plan':
        print(json.dumps(dict(protocol=PROTOCOL,commands=[c[-1] for c in commands]),indent=2)); return
    # Validate both prepared datasets before allocating GPU time to any model.
    from .data import ReasoningDataset
    for task in TASKS:
        for split in ('train','validation','test'):
            d=ReasoningDataset(a.data_root/(task+'-v2-n20000-v1000-t1000'),split)
            if d.task!=task or len(d)!=(20000 if split=='train' else 1000):
                raise ValueError('Require frozen 20k/1k/1k source subsets for this queue')
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=dict(steps=a.steps,micro_batch=a.micro_batch,data_root=str(a.data_root.resolve()),
                    protocol=PROTOCOL)
        cfg=a.output/'queue_config.json'
        if cfg.exists() and json.loads(cfg.read_text())!=config:
            raise ValueError('Queue contract differs')
        atomic_json(cfg,config)
        budget=a.output/'budget.json'
        if not budget.exists():
            deadline=time.time()+a.hours*3600
            if a.deadline_from:
                deadline=min(deadline,json.loads(a.deadline_from.read_text())['deadline_utc_seconds'])
            atomic_json(budget,dict(deadline_utc_seconds=deadline))
        deadline=json.loads(budget.read_text())['deadline_utc_seconds']
        env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=a.gpu
        def execute(command,label):
            remaining=deadline-time.time()
            if remaining<=0:
                raise TimeoutError('Persistent queue time budget exhausted')
            console=a.output/'console'; console.mkdir(exist_ok=True)
            logfile=console/(str(time.time_ns())+'-'+label+'.log')
            atomic_json(a.output/'status.json',dict(status='running',stage=label,command=command,
                        console=str(logfile),deadline_utc_seconds=deadline))
            print(label,flush=True)
            with logfile.open('x') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                try:
                    code=child.wait(timeout=remaining)
                except BaseException:
                    child.terminate()
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
                    raise
                if code: raise RuntimeError('Stage failed: '+label+'; see '+str(logfile))
        try:
            for task,variant,data,run,command in commands:
                if (run/'checkpoints/last.pt').exists():
                    _,receipt=verified_checkpoint(run)
                    if receipt['step']>a.steps: raise ValueError('Checkpoint exceeds requested final step')
                else:
                    receipt=dict(step=0)
                if receipt['step']<a.steps:
                    execute(command,task+'-'+variant+'-train')
                ckpt,_=verified_checkpoint(run,a.steps)
                if not (run/'generation.json').exists():
                    execute([sys.executable,'-u',str(ROOT/'scripts/reasoning/run_reasoning.py'),
                             'evaluate','--checkpoint',str(ckpt),'--data-dir',str(data),
                             '--output',str(run/'generation.json'),'--examples','1000',
                             '--batch-size','32','--seed','2026','--policy','top_prob'],
                            task+'-'+variant+'-evaluate')
                report(a.output)
                completed=[str(a.output/task/v) for v in VARIANTS if (a.output/task/v/'generation.json').exists()]
                execute([sys.executable,str(ROOT/'scripts/reasoning/run_reasoning.py'),
                         'plot','--runs',*completed,'--output',str(a.output/task/'training.png'),
                         '--smooth','60'],task+'-plot')
            atomic_json(a.output/'status.json',dict(status='finished',protocol=PROTOCOL))
        except BaseException as error:
            atomic_json(a.output/'status.json',dict(status='stopped',error=str(error)))
            raise


if __name__=='__main__':
    main()
