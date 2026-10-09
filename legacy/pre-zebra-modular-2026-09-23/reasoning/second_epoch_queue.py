"""Full-state Zebra/Sudoku continuations by exactly one additional epoch.

Original checkpoints/logs/evaluations are immutable. Only the exact epoch budget
changes. No score-dependent stopping, learning-rate restart, or new GPU overlap.
"""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import torch

from .benchmark import PROTOCOL
from .benchmark_queue import ROOT, TASKS, VARIANTS, DISPLAY_NAMES, training_command
from .data import ReasoningDataset
from .runner import atomic_json, digest, load_checkpoint
from .zebra_continuation import verified_checkpoint
from .variants import SPLIT_VARIANTS, SPLIT_LABELS
from .full_epoch_queue import split_training_command

CLI = ROOT / 'scripts/reasoning/run_reasoning.py'
SETTINGS = dict(policy='top_prob', candidate_k=8, token_selection='paper',
                tokens_per_step=1, memory_condition='correct', seed=2026)


def continuation_command(task, variant, data, run, target_epochs=2):
    return training_command(task, variant, data, run) + [
        '--epochs', str(target_epochs), '--validation-protocol', 'both',
        '--validation-examples', '1000', '--eval-batch-size', '64']


def split_continuation_command(task, variant, data, run, devices=2,
                               micro_batch=32, eval_batch=32, robustness=True,
                               target_epochs=2):
    command = split_training_command(task, variant, data, run, 0, devices,
                                     micro_batch, eval_batch, robustness)
    command.remove('--one-epoch')
    return command + ['--epochs', str(target_epochs), '--lr', '0.0003', '--warmup-steps', '1000']


def validate_split_source(source, devices, micro_batch, robustness=True, source_epochs=1):
    """Reject changed geometry/recipe before making a continuation checkpoint."""
    contract = json.loads((Path(source) / 'contract.json').read_text())
    expected = dict(suite='split', epochs=source_epochs, world_size=devices,
                    micro_batch=micro_batch, global_batch=128, lr=0.0003,
                    warmup_steps=1000, precision='bf16', seed=1,
                    validation_protocol='both')
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError('Split continuation must preserve source batch/optimizer contract: '+str(source))
    model = contract['model_config']
    if model['gradient_mode'] != 'adjacent':
        raise ValueError('Expected adjacent-gradient split suite')
    probability = 0.25 if robustness and '_rm' in contract['variant'] else 0.0
    if model['identity_probability'] != probability:
        raise ValueError('Source robustness settings differ; do not change the continuation recipe')
    return contract


def fork_next_epoch(source, destination, target_epochs=2):
    """Retain every tensor/RNG/cursor; explicitly extend only contract.epochs."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Continuation must be separate from the source run')
    contract = json.loads((source / 'contract.json').read_text())
    if (type(target_epochs) is not int or target_epochs < 2
            or contract.get('epochs') != target_epochs - 1
            or contract.get('tail_policy') != 'partial_batch_no_wrap_no_drop'
            or not isinstance(contract.get('world_size'), int) or contract['world_size'] < 1):
        raise ValueError('Expected an exact preceding-epoch checkpoint with a positive world size')
    count = contract['epoch_examples']
    source_epochs = target_epochs - 1
    steps_per_epoch = math.ceil(count / contract['global_batch'])
    step, target_step = source_epochs * steps_per_epoch, target_epochs * steps_per_epoch
    path, receipt = verified_checkpoint(source, step)
    extended = dict(contract, epochs=target_epochs)
    provenance = dict(version=1, source_run=str(source), source_checkpoint=str(path),
                      source_sha256=receipt['sha256'], source_step=step,
                      source_examples=source_epochs*count, target_step=target_step, target_examples=target_epochs*count,
                      source_contract_sha256=digest(source / 'contract.json'),
                      change=f'contract.epochs: {source_epochs} -> {target_epochs} only; preserve model/optimizer/RNG/data cursor')
    if destination.exists():
        if (json.loads((destination / 'continuation_source.json').read_text()) != provenance
                or json.loads((destination / 'contract.json').read_text()) != extended):
            raise ValueError('Existing continuation contract/provenance differs')
        _, current = verified_checkpoint(destination)
        if not step <= current['step'] <= target_step:
            raise ValueError('Continuation checkpoint outside requested epoch budget')
        return provenance
    checkpoint = load_checkpoint(path)
    if (checkpoint['contract'] != contract or checkpoint['step'] != step
            or checkpoint['examples_seen'] != source_epochs*count):
        raise ValueError('Source has not completed exactly the preceding epoch')
    if len(checkpoint['rng_by_rank']) != contract['world_size']:
        raise ValueError('Checkpoint must preserve RNG state for every source rank')
    checkpoint['contract'] = extended
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix='.'+destination.name+'.', dir=destination.parent))
    checkpoints = scratch / 'checkpoints'
    checkpoints.mkdir()
    target = checkpoints / path.name
    torch.save(checkpoint, target)
    with target.open('rb') as stream:
        os.fsync(stream.fileno())
    atomic_json(target.with_suffix('.pt.json'), dict(file=target.name, step=step,
                size=target.stat().st_size, sha256=digest(target), created_ns=time.time_ns()))
    (checkpoints / 'last.pt').symlink_to(target.name)
    atomic_json(scratch / 'contract.json', extended)
    atomic_json(scratch / 'continuation_source.json', provenance)
    verified_checkpoint(scratch, step)
    scratch.rename(destination)
    return provenance


def fork_second_epoch(source, destination):
    """Backward-compatible one-to-two continuation entry point."""
    return fork_next_epoch(source, destination, target_epochs=2)


def validated_generation(run, data, variant, step):
    result = json.loads((Path(run) / 'generation.json').read_text())
    checkpoint, _ = verified_checkpoint(run, step)
    test = ReasoningDataset(data, 'test')
    ids = [r['id'] for r in test.records]
    if (result.get('benchmark_protocol') != PROTOCOL or result['step'] != step
            or Path(result['checkpoint']).resolve() != checkpoint
            or result['contract']['task'] != test.task or result['contract']['variant'] != variant
            or result['contract']['data_sha256'] != digest(Path(data) / 'manifest.json')
            or [r['id'] for r in result['examples']] != ids or len(ids) != 1000
            or result['metrics']['num_examples'] != 1000
            or any(result['metrics'].get(k) != v for k, v in SETTINGS.items())
            or result['arguments']['batch_size'] != 32):
        raise ValueError('Generation checkpoint/test/protocol mismatch: '+str(run))
    return result


def paired_changes(first, second):
    a = {r['id']: bool(r['scores']['valid_solution']) for r in first['examples']}
    b = {r['id']: bool(r['scores']['valid_solution']) for r in second['examples']}
    if not a or list(a) != list(b) or len(a) != len(first['examples']):
        raise ValueError('Paired comparison needs unique, identical test IDs/order')
    gained = sum(b[k] and not a[k] for k in a)
    lost = sum(a[k] and not b[k] for k in a)
    n = gained + lost
    probability = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(gained, lost)+1)) / 2**n) if n else 1.0
    return dict(gained=gained, lost=lost, solved_both=sum(a[k] and b[k] for k in a),
                delta_accuracy_pp=100*(gained-lost)/len(a), paired_exact_p=probability)


def comparison_report(source, output, data, variants=None, labels=None, target_epochs=2):
    variants = VARIANTS if variants is None else variants
    labels = DISPLAY_NAMES if labels is None else labels
    test = ReasoningDataset(data, 'test')
    task = test.task
    gold = {r['id']: r['answer'] for r in test.records}
    rows, paired = [], []
    for variant in variants:
        original = Path(source)/task/variant
        continued = Path(output)/task/variant
        contract = json.loads((original/'contract.json').read_text())
        step = math.ceil(contract['epoch_examples']/contract['global_batch'])
        if contract.get('epochs', 1) != target_epochs-1:
            raise ValueError('Comparison requires the immediately preceding epoch')
        first = validated_generation(original, data, variant, (target_epochs-1)*step)
        results = [(target_epochs-1, first)]
        if (continued/'generation.json').exists():
            second = validated_generation(continued, data, variant, target_epochs*step)
            results.append((target_epochs, second))
            paired.append(dict(variant=variant, **paired_changes(first, second)))
        for epoch, result in results:
            details = result['examples']
            solved = sum(bool(e['scores']['valid_solution']) for e in details)
            correct = total = 0
            for e in details:
                answer, pred = gold[e['id']], e['predicted_answer_slots']
                correct += sum(i < len(pred) and x == pred[i] for i, x in enumerate(answer))
                total += len(answer)
            rows.append(dict(variant=variant, label=labels[variant], epoch=epoch,
                             step=result['step'], examples=len(details), solved=solved,
                             accuracy=solved/len(details), cell_accuracy=correct/total))
    out = Path(output)/'report'/task
    out.mkdir(parents=True, exist_ok=True)
    for name, entries in [('generation_comparison', rows), ('paired_changes', paired)]:
        if entries:
            with (out/(name+'.csv')).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
                writer.writeheader(); writer.writerows(entries)
    atomic_json(out/'comparison.json', dict(rows=rows, paired=paired, generation=SETTINGS,
                test_sha256=digest(Path(data)/'test.jsonl'),
                limit='Same 1k test and decoding seed; one training seed, no score-selected stopping.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, metric, title in zip(axes, ('accuracy', 'cell_accuracy'),
                                  ('Whole-puzzle accuracy (%)', 'Generated cell accuracy (%)')):
        for epoch, offset, color in [(target_epochs-1, -.19, '#4c78a8'), (target_epochs, .19, '#f58518')]:
            selected = [r for r in rows if r['epoch'] == epoch]
            x = [variants.index(r['variant'])+offset for r in selected]
            bars = axis.bar(x, [100*r[metric] for r in selected], .36,
                            label='Epoch '+str(epoch), color=color)
            if metric == 'accuracy':
                axis.bar_label(bars, labels=[str(r['solved'])+'/1000' for r in selected], fontsize=8, padding=3)
        axis.set_xticks(range(len(variants)))
        axis.set_xticklabels([labels[v].replace(' + ', '\n+ ') for v in variants])
        axis.set_ylabel(title); axis.set_ylim(bottom=0); axis.margins(y=.18); axis.legend()
    fig.suptitle(f'{task}: epoch {target_epochs-1} vs {target_epochs} — fixed test and generation protocol')
    fig.tight_layout(); fig.savefig(out/'generation_comparison.png', dpi=180); plt.close(fig)


def acquire_finished_predecessor(source, tasks=TASKS, variants=VARIANTS):
    """Hold predecessor lock after successful completion; never stop its service."""
    status_path = Path(source)/'status.json'
    if not status_path.exists():
        raise ValueError('Missing predecessor queue status')
    status = json.loads(status_path.read_text())
    if status['status'] == 'stopped':
        raise RuntimeError('Predecessor queue stopped: inspect its failure before continuing')
    if status['status'] != 'finished':
        return None
    lock = (Path(source)/'queue.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    try:
        for task in tasks:
            for variant in variants:
                run = Path(source)/task/variant
                state = json.loads((run/'status.json').read_text())
                generation = json.loads((run/'generation.json').read_text())
                if (state['status'] != 'finished' or generation['step'] != state['max_steps']
                        or generation['metrics']['num_examples'] != 1000
                        or generation['contract']['task'] != task
                        or generation['contract']['variant'] != variant):
                    count = 'ten' if len(tasks)*len(variants) == 10 else str(len(tasks)*len(variants))
                    raise ValueError('All '+count+' predecessor training AND evaluation jobs must finish')
    except BaseException:
        lock.close()
        raise
    return lock


def select_schedule(plan, specification):
    """Only select/reorder existing jobs; never alter their training contracts."""
    if set(specification) != {'version', 'reason', 'jobs'} or specification['version'] != 1:
        raise ValueError('Invalid queue schedule schema')
    if not isinstance(specification['reason'], str) or not specification['reason'].strip():
        raise ValueError('Schedule change requires a recorded reason')
    available = {(j['task'], j['variant']): j for j in plan}
    selected, seen = [], set()
    for item in specification['jobs']:
        if not isinstance(item, dict) or set(item) != {'task', 'variant'}:
            raise ValueError('Schedule entries contain only task and variant')
        key = (item['task'], item['variant'])
        if key not in available or key in seen:
            raise ValueError('Unknown or duplicate scheduled job: '+str(key))
        selected.append(available[key]); seen.add(key)
    if not selected:
        raise ValueError('Queue schedule cannot be empty')
    return selected


class SecondEpochQueue:
    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
        self.split = getattr(args, 'suite', 'legacy') == 'split'
        self.tasks = tuple(reversed(TASKS)) if getattr(args, 'task_order', 'source') == 'sudoku-first' else TASKS
        self.variants = SPLIT_VARIANTS if self.split else VARIANTS
        self.labels = SPLIT_LABELS if self.split else DISPLAY_NAMES
        self.target_epochs = getattr(args, 'target_epochs', 2)

    def status(self, state, **fields):
        atomic_json(self.output/'status.json', dict(status=state, jobs=getattr(self, 'scheduled_jobs', len(self.tasks)*len(self.variants)), pid=os.getpid(),
                    updated_ns=time.time_ns(), **fields))

    def execute(self, command, label):
        console = self.output/'console'; console.mkdir(exist_ok=True)
        logfile = console/(str(time.time_ns())+'-'+label+'.log')
        self.status('running', stage=label, command=command, console=str(logfile))
        print(label+': '+str(logfile), flush=True)
        with logfile.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=self.env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait()
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL); child.wait()
                raise
        if code:
            raise RuntimeError('Stage failed: '+label+'; see '+str(logfile))

    def run(self):
        a = self.args
        with (self.output/'queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            config = dict(version=1, source=str(a.after.resolve()), data_root=str(a.data_root.resolve()),
                          tasks=list(self.tasks), variants=list(self.variants), epochs=self.target_epochs, gpu=a.gpu, generation=SETTINGS)
            if self.split:
                config.update(version=2, suite='split', world_size=a.devices,
                              micro_batch=a.micro_batch, eval_batch=a.eval_batch,
                              robustness=not a.no_robustness)
                previous = json.loads((a.after/'queue_config.json').read_text())
                if (previous.get('suite') != 'split' or previous.get('variants') != list(self.variants)
                        or set(previous.get('tasks', [])) != set(self.tasks)
                        or previous.get('world_size') != a.devices
                        or previous.get('micro_batch') != a.micro_batch
                        or previous.get('eval_batch') != a.eval_batch
                        or previous.get('robustness') != (not a.no_robustness)
                        or previous.get('epochs', 1) != self.target_epochs-1):
                    raise ValueError('Predecessor must be the six-control, two-task split queue')
            path = self.output/'queue_config.json'
            if path.exists() and json.loads(path.read_text()) != config:
                raise ValueError('Queue specification changed')
            atomic_json(path, config)
            predecessor = None
            try:
                print(f'Waiting for all epoch-{self.target_epochs-1} training AND generation evaluations: '+str(a.after), flush=True)
                print('No GPU context is created while waiting. Continuation preserves the full optimizer state and LR.', flush=True)
                while predecessor is None:
                    predecessor = acquire_finished_predecessor(a.after, self.tasks, self.variants)
                    if predecessor is None:
                        self.status('waiting', stage=f'all-epoch-{self.target_epochs-1}-training-and-evaluation',
                                    predecessor=str(a.after))
                        time.sleep(30)
                plan = []
                for task in self.tasks:
                    data_path = a.data_root/(task+'-full-v1')
                    data = ReasoningDataset(data_path, 'train')
                    if data.task != task or data.manifest['schema_version'] != 2:
                        raise ValueError('Requires prepared full benchmark dataset')
                    steps_per_epoch = math.ceil(len(data)/128)
                    source_step = (self.target_epochs-1)*steps_per_epoch
                    target = self.target_epochs*steps_per_epoch
                    for variant in self.variants:
                        source = a.after/task/variant
                        destination = self.output/task/variant
                        if self.split:
                            validate_split_source(source, a.devices, a.micro_batch, not a.no_robustness,
                                                  source_epochs=self.target_epochs-1)
                        validated_generation(source, data_path, variant, source_step)
                        fork_next_epoch(source, destination, self.target_epochs)
                        plan.append(dict(task=task, variant=variant, examples=len(data),
                                         source_step=source_step, target_step=target,
                                         data=str(data_path), run=str(destination)))
                    comparison_report(a.after, self.output, data_path, self.variants, self.labels, self.target_epochs)
                atomic_json(self.output/'plan.json', plan)
                schedule_path = getattr(a, 'schedule_file', None)
                if schedule_path:
                    specification = json.loads(schedule_path.read_text())
                    selected = select_schedule(plan, specification)
                    selected_keys = {(j['task'], j['variant']) for j in selected}
                    atomic_json(self.output/'effective_schedule.json', dict(
                        specification=specification, schedule_sha256=digest(schedule_path),
                        selected=selected, skipped=[j for j in plan if (j['task'],j['variant']) not in selected_keys]))
                    plan = selected
                self.scheduled_jobs = len(plan)
                for job in plan:
                    task, variant, target = job['task'], job['variant'], job['target_step']
                    data_path, destination = Path(job['data']), Path(job['run'])
                    command = (split_continuation_command(task, variant, data_path, destination,
                               a.devices, a.micro_batch, a.eval_batch, not a.no_robustness, self.target_epochs)
                               if self.split else continuation_command(task, variant, data_path, destination, self.target_epochs))
                    state_path = destination/'status.json'
                    state = json.loads(state_path.read_text()) if state_path.exists() else {}
                    # A handoff can adopt an already-finished training job that
                    # still needs its evaluation; do not launch redundant ranks.
                    if not (state.get('status') == 'finished' and state.get('step') == target
                            and state.get('examples_seen') == self.target_epochs*job['examples']):
                        self.execute(command, task+'-'+variant+f'-epoch{self.target_epochs}-train')
                    checkpoint, _ = verified_checkpoint(destination, target)
                    state = json.loads((destination/'status.json').read_text())
                    if state['status'] != 'finished' or state['examples_seen'] != self.target_epochs*job['examples']:
                        raise ValueError('Continuation did not consume exactly the requested epochs')
                    if not (destination/'generation.json').exists():
                        self.execute([sys.executable, '-u', str(CLI), 'evaluate',
                            '--checkpoint', str(checkpoint), '--data-dir', str(data_path),
                            '--output', str(destination/'generation.json'), '--examples', '1000',
                            '--batch-size', '32', '--seed', '2026', '--policy', 'top_prob'], task+'-'+variant+f'-epoch{self.target_epochs}-evaluate')
                    validated_generation(destination, data_path, variant, target)
                    comparison_report(a.after, self.output, data_path, self.variants, self.labels, self.target_epochs)
                self.status('finished', variants=list(self.variants), tasks=list(self.tasks),
                            completed_schedule=[dict(task=j['task'],variant=j['variant']) for j in plan],
                            report=str(self.output/'report'))
            except BaseException as error:
                self.status('stopped', error=str(error))
                raise
            finally:
                if predecessor is not None:
                    predecessor.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'report', 'plan'))
    parser.add_argument('--after', type=Path, default=ROOT/'outputs/reasoning/full-epoch-v1')
    parser.add_argument('--data-root', type=Path, default=ROOT/'.cache/reasoning')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/reasoning/second-epoch-v1')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--suite', choices=('legacy', 'split'), default='legacy')
    parser.add_argument('--devices', type=int, default=2)
    parser.add_argument('--micro-batch', type=int, default=32)
    parser.add_argument('--eval-batch', type=int, default=32)
    parser.add_argument('--no-robustness', action='store_true')
    parser.add_argument('--task-order', choices=('source', 'sudoku-first'), default='source')
    parser.add_argument('--target-epochs', type=int, default=2,
                        help='Absolute epoch count; predecessor must finish exactly one fewer epoch')
    parser.add_argument('--schedule-file', type=Path,
                        help='Explicit recorded subset/order of jobs; training contracts remain unchanged')
    args = parser.parse_args(argv)
    if args.target_epochs < 2:
        parser.error('Continuation target must be at least two epochs')
    if args.suite == 'split' and (args.devices != 2 or len(set(args.gpu.split(','))) != 2
            or args.micro_batch < 1 or 128 % (args.devices*args.micro_batch)
            or args.eval_batch < 1):
        parser.error('Split continuation requires two distinct GPUs, positive batches, and global128 divisible by devices*microbatch')
    if args.after.resolve() == args.output.resolve() or args.after.resolve() in args.output.resolve().parents or args.output.resolve() in args.after.resolve().parents:
        parser.error('Output and predecessor must be separate, non-nested directories')
    tasks = tuple(reversed(TASKS)) if args.task_order == 'sudoku-first' else TASKS
    variants = SPLIT_VARIANTS if args.suite == 'split' else VARIANTS
    labels = SPLIT_LABELS if args.suite == 'split' else DISPLAY_NAMES
    if args.action == 'plan':
        summary = dict(after=str(args.after), output=str(args.output), tasks=list(tasks),
                              variants=list(variants), source_epochs=args.target_epochs-1,
                              epochs=args.target_epochs, suite=args.suite,
                              gpu=args.gpu, devices=args.devices if args.suite == 'split' else 1,
                              micro_batch=args.micro_batch if args.suite == 'split' else 128,
                              global_batch=128, learning_rate='1000-step linear warmup then constant 0.0003; no restart',
                              full_state_resume=True, compare_generation=SETTINGS)
        if args.schedule_file:
            summary['scheduled_jobs'] = select_schedule(
                [dict(task=t,variant=v) for t in tasks for v in variants],
                json.loads(args.schedule_file.read_text()))
            summary['schedule_file'] = str(args.schedule_file)
        print(json.dumps(summary, indent=2))
    elif args.action == 'report':
        for task in tasks:
            comparison_report(args.after, args.output, args.data_root/(task+'-full-v1'), variants, labels, args.target_epochs)
    else:
        def interrupted(signum, _frame):
            raise KeyboardInterrupt('Queue interrupted by signal '+str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        SecondEpochQueue(args).run()


if __name__ == '__main__':
    main()
