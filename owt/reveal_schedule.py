"""Wait without CUDA, then run the user-requested final-checkpoint diagnostic."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from owt.research import ROOT, atomic_write, read_json, record_event


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'outputs/owt/mdm-np-5k')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/analysis/owt-reveal-sweep-5000')
    args=parser.parse_args()
    args.root=args.root.resolve();args.output=args.output.resolve()
    state=args.root/'reveal_sweep_queue.json'
    with (args.root/'.reveal-followup.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        variants=['mdm','mdm_np_zero_init']
        atomic_write(state,json.dumps(dict(stage='waiting_for_zero_completion',queue_pid=os.getpid(),
            prerequisites=[v+':5000' for v in variants],output=str(args.output)),indent=2)+'\n')
        record_event('reveal_sweep_armed_zero_only_v2','Zero-NP versus MDM sweep armed',
            'At the user\'s request, the analysis includes only MDM and zero-init NP. '
            'A CPU-only controller waits for successful 5,000-update completion of those two arms. '
            'It will then run the exact 5x3 mask/reveal-reliability grid on the fixed 1,024-row validation subset, '
            'using frozen EMA and nearest different clean-token replacements. GPU evaluation takes the shared '
            'training lock and uses physical GPU 2; no CUDA context is allocated while waiting. '
            'Layer-update cosines and main/auxiliary errors are collected in the same forward passes, with 15 figures comparing the two arms. '
            'The lower-weight trial remains held until these results are analyzed.')
        try:
            while True:
                done=[read_json(args.root/v/'complete.json') for v in variants]
                if all(d and d.get('optimizer_step')==5000 for d in done):
                    break
                zero=read_json(args.root/'zero_init_queue.json') or {}
                if zero.get('stage') in ['failed','failed_prerequisites']:
                    raise RuntimeError('Zero-init controller failed; do not evaluate an incomplete run')
                time.sleep(60)
            args.output.mkdir(parents=True,exist_ok=True)
            if not (args.output/'summary.json').exists():
                command=[sys.executable,'-u','-m','owt.reveal_sweep','--root',str(args.root),
                         '--output',str(args.output),'--device','cuda','--variants',*variants]
                with (args.output/'eval.log').open('a') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                    atomic_write(state,json.dumps(dict(stage='evaluation',queue_pid=os.getpid(),
                        worker_pid=child.pid,output=str(args.output)),indent=2)+'\n')
                    if child.wait()!=0:
                        raise RuntimeError('Reveal sweep worker failed; inspect eval.log')
            subprocess.run([sys.executable,'-m','analysis.plot_reveal_sweep','--input',
                            str(args.output/'summary.json')],cwd=ROOT,check=True)
            expected={f'layer_cosine_mask{m:03d}_correct{c:03d}.pdf'
                      for m in [100,80,60,40,20] for c in [100,80,60]}
            actual={p.name for p in args.output.glob('layer_cosine_mask*_correct*.pdf')}
            if actual!=expected:
                raise RuntimeError('The layerwise figure grid is incomplete')
            record_event('reveal_sweep_figures_complete_zero_only_v2','All 15 layerwise cosine figures generated',
                'Generated one figure for each mask-ratio x revealed-correctness condition, with MDM '
                'and zero-init NP curves for masked, correct revealed, and wrong revealed content positions. '
                'Also saved a combined 15-page cosine PDF and main-performance/difference heatmaps. '
                'These geometric diagnostics do not establish forgetting or causal computational benefit. '
                'Keep the next training decision on hold for scientific review.',dict(output=str(args.output),figures=sorted(actual)))
            atomic_write(state,json.dumps(dict(stage='complete',queue_pid=os.getpid(),
                output=str(args.output),next_training='held_for_analysis'),indent=2)+'\n')
            print('COMPLETE reveal sweep and 15 figures',flush=True)
        except Exception as exc:
            atomic_write(state,json.dumps(dict(stage='failed',error=str(exc),output=str(args.output)),indent=2)+'\n')
            record_event('reveal_sweep_controller_failed_v1','Reveal sweep needs inspection',str(exc))
            raise


if __name__=='__main__':
    main()
