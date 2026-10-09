"""Sequential OWT pilots and a serialized zero-init follow-up on GPUs 2/3."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def require_original_completion(root):
    for variant in ('mdm', 'mdm_np'):
        path = root/variant/'complete.json'
        if not path.is_file() or json.loads(path.read_text()).get('optimizer_step', 0) < 5000:
            raise RuntimeError(f'{variant} has not successfully completed 5,000 updates; '
                               'refusing to start zero-init training')


def low_weight_decision(root, threshold=0.03):
    """Pre-registered practical gate, not a significance test or seed estimate."""
    require_original_completion(root)
    path = root/'mdm_np_zero_init/complete.json'
    if not path.is_file() or json.loads(path.read_text()).get('optimizer_step', 0) != 5000:
        raise RuntimeError('Zero-init NP must successfully complete 5,000 updates first')
    values = {}
    for variant in ('mdm', 'mdm_np_zero_init'):
        with (root/variant/'local_metrics/validation.csv').open() as stream:
            rows = [row for row in csv.DictReader(stream) if row['optimizer_step'] == '5000']
        if not rows:
            raise RuntimeError(f'Missing final 5,000-update validation for {variant}')
        values[variant] = float(rows[-1]['val_nll'])
        if not math.isfinite(values[variant]):
            raise RuntimeError(f'Nonfinite final validation for {variant}')
    delta = values['mdm_np_zero_init']-values['mdm']
    return dict(launch=delta >= threshold, delta_nats=delta,
                practical_threshold_nats=threshold, final_validation=values,
                interpretation='practical next-trial selection, not statistical significance')


def reviewed_low_weight_decision(root, path):
    """Use an explicit scientific selection after the complete diagnostic gate."""
    root, path = Path(root).resolve(), Path(path).resolve()
    selected = json.loads(path.read_text())
    if (selected.get('status') != 'selected_after_all_diagnostics_and_frozen_readout_review'
            or selected.get('variant') != 'mdm_np_zero_init_low_weight'
            or selected.get('target_steps') != 5000
            or selected.get('microbatch') != 8
            or selected.get('physical_gpus') != [2, 3]
            or Path(selected.get('run_root', '')).resolve() != root):
        raise RuntimeError('Invalid reviewed low-weight selection')
    required = ('diagnostic_decision', 'frozen_readout_summary', 'training_decision_pdf', 'training_recipe')
    artifacts = selected.get('artifact_sha256', {})
    if not artifacts or any(selected.get(key) not in artifacts for key in required):
        raise RuntimeError('Missing reviewed decision evidence')
    for filename, expected in artifacts.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Reviewed decision artifact changed: '+filename)
    memo = json.loads(Path(selected['diagnostic_decision']).read_text())
    if (memo.get('status') != 'all_registered_diagnostics_completed_and_scientifically_reviewed'
            or len(memo.get('coverage', {})) != 6
            or set(memo['coverage'].values()) != {'completed_and_reviewed'}):
        raise RuntimeError('Full diagnostic review is incomplete')
    frozen = json.loads(Path(selected['frozen_readout_summary']).read_text())
    if frozen.get('scoring_rows') != 512 or len(frozen.get('cells', [])) != 19:
        raise RuntimeError('Selected frozen readout is incomplete')
    for variant in ('mdm', 'mdm_np_zero_init'):
        complete = root/variant/'complete.json'
        if not complete.exists() or json.loads(complete.read_text()).get('optimizer_step') != 5000:
            raise RuntimeError('Reviewed baseline incomplete: '+variant)
    return dict(launch=True, selection=str(path), selection_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                microbatch=selected['microbatch'],
                rationale=selected['rationale'], final_validation=selected['final_validation'],
                interpretation='explicit scientific selection; no automatic validation threshold')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'smoke', 'followup-zero', 'followup-low-weight'])
    parser.add_argument('--root', type=Path, default=ROOT/'outputs/owt/mdm-np-5k')
    parser.add_argument('--microbatch', type=int, default=8)
    parser.add_argument('--decision', type=Path, help='Reviewed scientific selection for low-weight follow-up')
    args = parser.parse_args()
    if args.action == 'smoke' and args.root == ROOT/'outputs/owt/mdm-np-5k':
        args.root = ROOT/'.cache/runtime/owt/smoke'
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    followup = args.action.startswith('followup-')
    low_weight = args.action == 'followup-low-weight'
    if args.decision and not low_weight:
        parser.error('--decision applies only to the low-weight follow-up')
    reviewed = reviewed_low_weight_decision(args.root, args.decision) if args.decision else None
    if reviewed and args.microbatch != reviewed['microbatch']:
        parser.error('Microbatch differs from reviewed matched training protocol')
    if low_weight and not reviewed and (args.root/'reveal_sweep_required.json').exists():
        raise RuntimeError('User-requested reveal sweep and analysis must precede another training trial')
    variant = 'mdm_np_zero_init_low_weight' if low_weight else 'mdm_np_zero_init'
    state_path = args.root/('low_weight_queue.json' if low_weight else
                           'zero_init_queue.json' if followup else 'current.json')
    # A separate nonblocking lock prevents duplicate followers while the original
    # queue owns .queue.lock. Hold both locks through the entire follow-up run.
    controller_name = ('.low-weight-followup.lock' if low_weight else
                       '.zero-init-followup.lock' if followup else '.original-controller.lock')
    with (args.root/controller_name).open('a') as controller:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if followup:
            state_path.write_text(json.dumps(dict(stage='waiting_for_prior_queue' if low_weight
                                                     else 'waiting_for_original_queue',
                variant=variant, target_steps=5000, queue_pid=os.getpid(),
                prerequisites=['mdm:5000', 'mdm_np_zero_init:5000'] if reviewed else
                    ['mdm:5000', 'mdm_np:5000'] + (['mdm_np_zero_init:5000'] if low_weight else []),
                reviewed_selection=str(args.decision.resolve()) if reviewed else None), indent=2)+'\n')
            print('Waiting for the prior experiment queue lock; no CUDA context allocated.', flush=True)
        with (args.root/'.queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if followup else fcntl.LOCK_NB))
            if followup:
                try:
                    if not reviewed:
                        require_original_completion(args.root)
                    if low_weight:
                        decision = reviewed_low_weight_decision(args.root, args.decision) if reviewed else low_weight_decision(args.root)
                        from owt.research import record_event
                        if reviewed:
                            record_event('reviewed_low_weight_launch_20261001', 'Reviewed low-weight training launch',
                                'All final diagnostics and frozen-readout results reviewed. Launch only the selected fresh zero-init NP recipe with 0.05 per direction, under the shared lock. '+decision['rationale'], decision)
                        else:
                            record_event('low_weight_decision', 'Conditional follow-up decision',
                                'Final zero-init NP minus MDM validation ELBO: '
                                f'{decision["delta_nats"]:.6f} nats/token. '
                                'The pre-registered practical threshold is +0.03 nats/token. '
                                + ('Launch fresh zero-init NP with weights 0.05 per direction.'
                                   if decision['launch'] else
                                   'Do not launch the low-weight trial; review the initialization outcome '
                                   'before selecting a source-visibility experiment.'),
                                decision)
                        if not decision['launch']:
                            state_path.write_text(json.dumps(dict(stage='not_launched',
                                reason='zero_init_outcome_requires_theory_review',
                                decision=decision), indent=2)+'\n')
                            print('LOW-WEIGHT TRIAL DEFERRED', decision, flush=True)
                            return
                except Exception as exc:
                    state_path.write_text(json.dumps(dict(stage='failed_prerequisites', error=str(exc)), indent=2)+'\n')
                    raise
            run_variants(args, state_path, (variant,) if followup else ('mdm', 'mdm_np'))


def run_variants(args, state_path, variants):
    try:
        for variant in variants:
            run = args.root/variant
            run.mkdir(exist_ok=True)
            target = 2 if args.action == 'smoke' else 5000
            complete = run/'complete.json'
            if complete.exists() and json.loads(complete.read_text())['optimizer_step'] >= target:
                print(f'Already complete: {variant}', flush=True)
                continue
            command = [sys.executable, '-u', str(ROOT/'owt/entrypoint.py'),
                       '--variant', variant, '--run', str(run), '--steps', str(target),
                       '--microbatch', str(args.microbatch)]
            if args.action == 'smoke':
                command += ['--global-batch',str(4*args.microbatch),'--interval','1',
                            '--val-examples',str(2*args.microbatch),'--workers','1']
            latest = run/'checkpoints/last.ckpt'
            if latest.is_file():
                command += ['--resume',str(latest.resolve())]
            state_path.write_text(json.dumps(dict(
                stage='train', variant=variant, target_steps=target,
                reviewed_selection=str(args.decision.resolve()) if getattr(args,'decision',None) else None,
                queue_pid=os.getpid(), log=str(run/'train.log')), indent=2)+'\n')
            print('RUN', ' '.join(command), flush=True)
            print('LOG', run/'train.log', flush=True)
            with (run/'train.log').open('a') as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
            if not complete.exists() or json.loads(complete.read_text())['optimizer_step'] != target:
                raise RuntimeError(f'{variant} did not reach {target}')
        if args.action == 'smoke':
            # Exercise actual optimizer/EMA/data-cursor continuation with NP.
            run = args.root/'mdm_np'
            checkpoint = run/'checkpoints/step-0000002.ckpt'
            command = [sys.executable, '-u', str(ROOT/'owt/entrypoint.py'),
                       '--variant','mdm_np','--run',str(run),'--steps','3',
                       '--microbatch',str(args.microbatch), '--global-batch',str(4*args.microbatch),
                       '--interval','1','--val-examples',str(2*args.microbatch),'--workers','1',
                       '--resume',str(checkpoint)]
            with (run/'resume.log').open('a') as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
            result = json.loads((run/'complete.json').read_text())
            if result['optimizer_step'] != 3:
                raise RuntimeError('Smoke continuation did not reach optimizer step 3')
        state_path.write_text(json.dumps(dict(
            stage='complete', variants=list(variants), target_steps=target), indent=2)+'\n')
        print('COMPLETE', args.action, args.root, flush=True)
    except Exception as exc:
        state_path.write_text(json.dumps(dict(stage='failed', error=str(exc)), indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
