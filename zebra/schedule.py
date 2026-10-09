"""Alternating MDM / MDM+NP runs, full-state resumes and separate generation."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = 1499933
STEPS_PER_EPOCH = math.ceil(ROWS / 512)


def milestones(final_epoch):
    return sorted(set(range(3, final_epoch + 1, 3)) | {final_epoch})


def continuation_milestones(start_epoch, final_epoch):
    if start_epoch < 0 or final_epoch <= start_epoch:
        raise ValueError('Require 0 <= start_epoch < final_epoch')
    if start_epoch == 0:
        return milestones(final_epoch)
    return sorted(set(range(start_epoch + 3, final_epoch + 1, 3)) | {final_epoch})


def checkpoint_step(path):
    # Local, trusted full Lightning checkpoint (optimizer, EMA and loop state).
    import torch
    sys.path.insert(0, str(ROOT / 'third_party/reasoning_with_latent_tokens'))
    return int(torch.load(path, map_location='cpu', weights_only=False, mmap=True)['global_step'])


def latest_checkpoint(run):
    candidates = list((run / 'checkpoints').glob('*.ckpt'))
    if not candidates:
        return 0, None
    # Read metadata, not just filenames: interrupted saves must fail visibly.
    return max(((checkpoint_step(p), p) for p in candidates), key=lambda x: x[0])


def atomic_json(path, payload):
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(path)


def run_process(command, log):
    print('RUN', ' '.join(map(str, command)), '\nLOG', log, flush=True)
    with log.open('a') as stream:
        subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT,
                       check=True, cwd=ROOT)


def refresh(root):
    rows = []
    for variant in ('mdm', 'mdm_np'):
        for completion in (root/variant/'generation').glob('epoch-*/complete.json'):
            meta = json.loads(completion.read_text())
            payload = json.loads(Path(meta['samples']).read_text())
            metrics = payload['eval_metrics']
            rows.append(dict(variant=variant, epoch=meta['epoch'], step=meta['step'],
                             accuracy=metrics['puzzle_accuracy'],
                             row_accuracy=metrics.get('mean_row_accuracy'),
                             cell_accuracy=metrics.get('mean_cell_accuracy'),
                             correct=metrics['n_correct_puzzles'], n=metrics['n_total_puzzles']))
    rows.sort(key=lambda r: (r['variant'], r['epoch']))
    with (root/'generation_history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'variant','epoch','step','accuracy','row_accuracy','cell_accuracy','correct','n'])
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for variant, label in [('mdm', 'MDM'), ('mdm_np', 'MDM + NP (0.25 each)')]:
            data = [r for r in rows if r['variant'] == variant]
            ax.plot([r['epoch'] for r in data], [100*r['accuracy'] for r in data], 'o-', label=label)
        largest = max(100*r['accuracy'] for r in rows)
        upper = min(100.0, max(1.0, largest * 1.10))
        ax.set(xlabel='Training epochs', ylabel='Exact puzzle accuracy (%)', ylim=(0, upper))
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root/'accuracy_vs_epoch.png', dpi=180)
        plt.close(fig)
        # Row and cell accuracy live on different scales; separate panels make
        # small differences visible without mixing four overlapping curves.
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, field, title in [
                (axes[0], 'row_accuracy', 'Mean row accuracy'),
                (axes[1], 'cell_accuracy', 'Mean cell accuracy')]:
            available = []
            for variant, label in [('mdm', 'MDM'), ('mdm_np', 'MDM + NP (0.25 each)')]:
                data = [r for r in rows if r['variant'] == variant and r[field] is not None]
                if data:
                    values = [100*r[field] for r in data]
                    available.extend(values)
                    ax.plot([r['epoch'] for r in data], values, 'o-', label=label)
            upper = min(100.0, max(1.0, max(available) * 1.10)) if available else 100.0
            ax.set(title=title, xlabel='Training epochs', ylabel='Accuracy (%)', ylim=(0, upper))
            ax.grid(alpha=.25)
            if ax.lines:
                ax.legend()
        fig.tight_layout()
        fig.savefig(root/'row_cell_accuracy_vs_epoch.png', dpi=180)
        plt.close(fig)
    # Main-head ELBO is comparable; total NP objective has additional terms.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    any_data = False
    for variant, label in [('mdm', 'MDM'), ('mdm_np', 'MDM + NP')]:
        for ax, filename, field, smooth in [
            (axes[0], 'train.csv', 'train_loss', True),
            (axes[1], 'train.csv', 'main_elbo', True),
            (axes[2], 'validation.csv', 'val_nll', False)]:
            path = root/variant/'local_metrics'/filename
            if path.exists():
                data = pd.read_csv(path).drop_duplicates('optimizer_step', keep='last')
                if field in data and not data.empty:
                    values = data[field].rolling(128, min_periods=1).mean() if smooth else data[field]
                    visible = data.optimizer_step >= 5000
                    ax.plot(data.loc[visible, 'optimizer_step'], values.loc[visible], label=label)
                    any_data = True
    for ax, title in zip(axes, ['Train total objective (includes NP)',
                               'Train main ELBO', 'Validation main ELBO']):
        ax.set(title=title, xlabel='Optimizer updates', ylabel='Loss')
        ax.set_xlim(left=5000)
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend()
    if any_data:
        fig.tight_layout()
        fig.savefig(root/'loss_vs_step.png', dpi=180)
    plt.close(fig)


def status(root):
    policy = root/'microbatch_policy.json'
    report = root/'microbatch_benchmark.json'
    if policy.exists():
        if report.exists():
            result = json.loads(report.read_text())
            print('Boundary benchmark:', json.dumps(result))
        else:
            print('Microbatch 128->256 benchmark scheduled AFTER both epoch-3 evaluations.')
    state = root/'current.json'
    if state.exists():
        print(json.dumps(json.loads(state.read_text()), indent=2))
    else:
        print('Queue has not started.')
    for variant in ('mdm', 'mdm_np'):
        for kind in ('train', 'validation'):
            path = root/variant/'local_metrics'/f'{kind}.csv'
            if path.exists():
                with path.open() as stream:
                    rows = [r for r in csv.DictReader(stream) if r.get('optimizer_step')]
                if rows:
                    last = rows[-1]
                    step = int(float(last['optimizer_step']))
                    field = 'main_elbo' if kind == 'train' else 'val_nll'
                    print(f'{variant} {kind}: step {step}, logical epoch {step/STEPS_PER_EPOCH:.3f}, '
                          f'{field}={last.get(field)}')
    refresh(root)
    history = root/'generation_history.csv'
    if history.exists():
        with history.open() as stream:
            rows = list(csv.DictReader(stream))
        for variant in ('mdm', 'mdm_np'):
            data = [r for r in rows if r['variant'] == variant]
            if data:
                last = data[-1]
                print(f"{variant} generation: epoch {last['epoch']}, accuracy {100*float(last['accuracy']):.3f}%")
    print('Plots:', root/'loss_vs_step.png', root/'accuracy_vs_epoch.png',
          root/'row_cell_accuracy_vs_epoch.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'smoke', 'check', 'plot', 'status'])
    parser.add_argument('--root', type=Path, default=ROOT/'outputs/zebra/mdm-np-40ep')
    parser.add_argument('--microbatch', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--variants', nargs='+', choices=['mdm', 'mdm_np'],
                        default=['mdm', 'mdm_np'])
    parser.add_argument('--wait-for-lock', action='store_true')
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.action == 'plot':
        refresh(args.root)
        return
    if args.action == 'status':
        status(args.root)
        return
    contract = dict(rows=ROWS, global_batch=512, devices=2, microbatch=args.microbatch,
                    steps_per_epoch=STEPS_PER_EPOCH, seed=1,
                    recipes={v: hashlib.sha256((ROOT/f'zebra/configs/{v}.yaml').read_bytes()).hexdigest()
                             for v in ('mdm', 'mdm_np')})
    contract_path = args.root/'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise RuntimeError('Run contract changed: use a NEW output root for a new setting')
    if args.action == 'check':
        print(json.dumps(dict(contract, milestones=continuation_milestones(
            args.start_epoch, args.epochs)), indent=2))
        return
    with (args.root/'.queue.lock').open('w') as lock:
        lock_flags = fcntl.LOCK_EX if args.wait_for_lock else fcntl.LOCK_EX | fcntl.LOCK_NB
        fcntl.flock(lock, lock_flags)
        atomic_json(contract_path, contract)
        planned = continuation_milestones(args.start_epoch, args.epochs)
        for epoch in ([0] if args.action == 'smoke' else planned):
            for variant in args.variants:
                run = args.root / variant
                run.mkdir(exist_ok=True)
                recipe = ROOT / f'zebra/configs/{variant}.yaml'
                target = 2 if args.action == 'smoke' else epoch * STEPS_PER_EPOCH
                step, checkpoint = latest_checkpoint(run)
                command = [sys.executable, '-u', ROOT/'zebra/entrypoint.py',
                           '--recipe', recipe, '--run', run,
                           '--microbatch', args.microbatch, '--target-steps', target]
                if step < target:
                    atomic_json(args.root/'current.json', dict(
                        stage='train', variant=variant, target_epoch=epoch,
                        target_steps=target, queue_pid=os.getpid(),
                        log=str(run/f'train-to-{target}.log')))
                    if checkpoint:
                        command.extend(['--resume', checkpoint])
                    if args.action == 'smoke':
                        command.extend(['--smoke', '--workers', '0'])
                    run_process(command, run/f'train-to-{target}.log')
                    step, checkpoint = latest_checkpoint(run)
                    if step != target:
                        raise RuntimeError(f'{variant}: expected step {target}, found {step}')
                generation = run/'generation'/f'epoch-{epoch:03d}'
                generation.mkdir(parents=True, exist_ok=True)
                if (generation/'complete.json').exists():
                    continue
                if step > target:
                    raise RuntimeError(f'Missing evaluation for epoch {epoch}; refusing to use later checkpoint')
                eval_command = [sys.executable, '-u', ROOT/'zebra/entrypoint.py',
                                '--recipe', recipe, '--run', generation, '--stage', 'evaluate',
                                '--resume', checkpoint, '--devices', '1']
                if args.action == 'smoke':
                    eval_command.extend(['--eval-batches','1','--eval-batch-size','4','--workers','0'])
                atomic_json(args.root/'current.json', dict(
                    stage='generation', variant=variant, epoch=epoch,
                    step=step, queue_pid=os.getpid(), log=str(generation/'generation.log')))
                run_process(eval_command, generation/'generation.log')
                samples = sorted(generation.glob('samples_*.json'))
                if not samples:
                    raise RuntimeError('Evaluation exited without a sample file')
                payload = json.loads(samples[-1].read_text())
                expected = 4 if args.action == 'smoke' else 1280
                if payload['eval_metrics']['n_total_puzzles'] != expected:
                    raise RuntimeError('Wrong evaluation sample count')
                atomic_json(generation/'complete.json', dict(
                    epoch=epoch, step=step, checkpoint=str(checkpoint), samples=str(samples[-1])))
                refresh(args.root)
        atomic_json(args.root/'current.json', dict(stage='complete', epochs=args.epochs,
                    variants=args.variants, queue_pid=os.getpid()))
        print('COMPLETE', args.action, args.root, flush=True)


if __name__ == '__main__':
    main()
