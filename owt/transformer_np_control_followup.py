"""Wait for this B trial only, then collect/report its registered final cells."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from owt.research import ROOT, atomic_write, read_json
from owt.transformer_np_control_schedule import B, RUN_ROOT, verify_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    verify_selection(args.selection)
    run = ROOT / RUN_ROOT / B
    status = ROOT / RUN_ROOT / 'control_followup.json'
    atomic_write(status, json.dumps(dict(stage='waiting_for_B5000', variant=B, pid=os.getpid()))+'\n')
    while (read_json(run / 'complete.json') or {}).get('optimizer_step') != 5000:
        if (read_json(ROOT / RUN_ROOT / 'control_queue.json') or {}).get('stage') == 'failed_requires_review':
            atomic_write(status, json.dumps(dict(stage='training_failed_requires_review', variant=B))+'\n')
            raise RuntimeError('B training failed; no additional experiment started')
        time.sleep(30)
    verify_selection(args.selection)
    # The trainer writes complete.json just before releasing the shared lock.
    # Allow its controller to exit; collector still enforces lock exclusion.
    time.sleep(30)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='2,3', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MPLCONFIGDIR=str(ROOT / '.cache/runtime/np-control-report-mpl'))
    for module, stage in [('owt.transformer_np_diagnostics', 'collecting_B5000'),
                          ('analysis.owt_transformer_np_report', 'reporting_B5000')]:
        atomic_write(status, json.dumps(dict(stage=stage, variant=B, pid=os.getpid()))+'\n')
        subprocess.run([sys.executable, '-u', '-m', module, '--protocol', str(args.protocol.resolve()),
            '--variant', B], cwd=ROOT, env=env, check=True)
    atomic_write(status, json.dumps(dict(stage='B_final_report_ready_scientific_review_pending',
        variant=B, next_training_launched=False))+'\n')


if __name__ == '__main__':
    main()
