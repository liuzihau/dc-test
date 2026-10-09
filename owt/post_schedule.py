"""CPU-only follower: collect all selected supplements after the core sweep."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from owt.research import ROOT, atomic_write, read_json, record_event


def core_ready(root, core):
    queue = read_json(root / 'reveal_sweep_queue.json') or {}
    if queue.get('stage') == 'failed':
        raise RuntimeError('Core sweep failed; inspect its controller before the supplement')
    if queue.get('stage') != 'complete':
        return False
    if Path(queue['output']).resolve() != core.resolve():
        raise RuntimeError('Core controller completed a different output directory')
    summary = read_json(core / 'summary.json')
    if not summary:
        raise RuntimeError('Core controller says complete but summary is missing')
    protocol = summary['protocol']
    if (protocol['variants'] != ['mdm', 'mdm_np_zero_init'] or
            len(protocol['row_ids']) != 1024 or protocol['optimizer_step'] != 5000):
        raise RuntimeError('Core sweep differs from the required two-arm full-cohort run')
    figures = {f'layer_cosine_mask{m:03d}_correct{c:03d}.pdf'
               for m in (100, 80, 60, 40, 20) for c in (100, 80, 60)}
    if not all((core / name).is_file() for name in figures):
        raise RuntimeError('Required 15 core cosine figures are incomplete')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs/owt/mdm-np-5k')
    parser.add_argument('--core', type=Path, default=ROOT / 'outputs/analysis/owt-reveal-sweep-5000')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/analysis/owt-post-diagnostics-5000')
    args = parser.parse_args()
    args.root = args.root.resolve(); args.core = args.core.resolve(); args.output = args.output.resolve()
    state = args.root / 'post_diagnostics_queue.json'
    import os
    with (args.root / '.post-followup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def status(stage, **extra):
            atomic_write(state, json.dumps(dict(stage=stage, queue_pid=os.getpid(),
                core=str(args.core), output=str(args.output), next_training='held_for_all_diagnostics', **extra), indent=2) + '\n')
        status('waiting_for_core_sweep_and_figures')
        record_event('post_diagnostics_follower_armed_20261001', 'Post-5k diagnostic supplements armed',
            'A CPU-only follower waits for the complete two-arm core sweep and all 15 cosine figures. '
            'It then collects the 1,024-row near-clean and paired first-input controls and 128-row same-source '
            'interventions under the shared GPU lock, followed by the exact 16-row CPU zero-init bridge '
            'reusing checked saved MDM observations. It never launches training; error-report review and '
            'the full decision memo remain required.', dict(core=str(args.core), output=str(args.output)))
        try:
            while not core_ready(args.root, args.core):
                time.sleep(60)
            args.output.mkdir(parents=True, exist_ok=True)
            commands = []
            if not (args.output / 'summary.json').exists():
                commands.append(('near_clean_and_source_collection', [sys.executable, '-u', '-m',
                    'owt.post_diagnostics', '--root', str(args.root), '--core', str(args.core),
                    '--output', str(args.output), '--device', 'cuda']))
            bridge = ROOT / 'outputs/analysis/owt-zero-init-mask-profile-5000'
            if not (bridge / 'summary.json').exists():
                commands.append(('exact_pilot_bridge', [sys.executable, '-u', '-m',
                    'analysis.owt_mask_profile', '--root', str(args.root), '--output', str(bridge),
                    '--compare', 'mdm_np_zero_init', '--threads', '2', '--reference',
                    str(ROOT / 'outputs/analysis/owt-mask-profile-5000/summary.json')]))
            commands.append(('bridge_figures', [sys.executable, '-m', 'analysis.plot_owt_mask_profile',
                '--input', str(bridge / 'summary.json')]))
            report = ROOT / 'outputs/analysis/owt-head-complementarity-5000'
            if not (report / 'summary.json').exists():
                commands.append(('paired_error_reports', [sys.executable, '-u', '-m',
                    'analysis.owt_error_report', '--core', str(args.core), '--supplement',
                    str(args.output), '--output', str(report)]))
            for stage, command in commands:
                with (args.output / 'collection.log').open('a') as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    status(stage, worker_pid=child.pid)
                    if child.wait() != 0:
                        raise RuntimeError(f'{stage} failed; inspect {args.output / "collection.log"}')
            supplement = read_json(args.output / 'summary.json')
            pilot = read_json(bridge / 'summary.json')
            if not supplement or supplement['protocol']['preflight'] or not pilot or pilot['comparison_variant'] != 'mdm_np_zero_init':
                raise RuntimeError('Supplement or exact bridge completion evidence is missing')
            error_report = read_json(report / 'summary.json')
            if (not error_report or error_report['preflight'] or error_report['variants'] != ['mdm', 'mdm_np_zero_init']
                    or len(error_report['row_ids']) != 1024 or len(error_report['cells']) != 19
                    or len(error_report['source_interventions']) != 27):
                raise RuntimeError('Full paired error report completion evidence is missing')
            status('diagnostics_complete_waiting_for_scientific_review', bridge=str(bridge), report=str(report))
            record_event('post_diagnostic_collections_complete_20261001', 'Diagnostic supplement and exact bridge collected',
                'The near-clean, paired first-input and same-source diagnostic supplement and exact zero-init '
                'Section 2.1 bridge are collected. Main/auxiliary and cross-model error reports include paired '
                'uncertainty, source/exposure strata and matched context controls. Scientific inspection and '
                'the full decision memo remain required; no next training recipe has been selected or launched.',
                dict(output=str(args.output), bridge=str(bridge), report=str(report)))
        except Exception as exc:
            status('failed', error=str(exc))
            record_event('post_diagnostic_follower_failed_20261001', 'Post-5k supplement needs inspection', str(exc))
            raise


if __name__ == '__main__':
    main()
