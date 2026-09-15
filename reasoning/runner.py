"""Small-task experiments, isolated from the historical OpenWebText trainer.

An explicit optimizer-boundary loop avoids Lightning's microbatch/epoch resume
ambiguity. DDP, accumulation, AdamW, RNG and a deterministic data cursor are
checkpointed together. No dataset downloads, GPU selection, or jobs on import.
"""
import argparse
from contextlib import nullcontext
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Subset


TASKS = ('sudoku', 'zebra', 'countdown')
VARIANTS = ('vanilla', 'mdm', 'mdm_aux', 'final', 'dcache', 'both', 'both_aux')
SIZES = {'debug': (32, 4, 2), 'tiny': (384, 12, 3),
         'mini': (512, 8, 6), 'sminy': (768, 12, 6)}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class GlobalExampleStream:
    """Rank-independent epoch permutations indexed by COMPLETED examples.

    Every optimizer update consumes exactly global_batch logical examples.
    A restart never uses the dataloader's prefetch position. Epochs smaller
    than a global batch wrap with a fresh permutation rather than dropping data.
    """
    def __init__(self, size, seed):
        if size < 1:
            raise ValueError('Training split is empty')
        self.size, self.seed = size, seed
        self.epoch, self.order = None, None

    def indices(self, start, count):
        if start < 0 or count < 0:
            raise ValueError('Negative data cursor')
        result = []
        while count:
            epoch, offset = divmod(start, self.size)
            if self.epoch != epoch:
                generator = torch.Generator().manual_seed(self.seed + epoch)
                self.order = torch.randperm(self.size, generator=generator).tolist()
                self.epoch = epoch
            take = min(count, self.size - offset)
            result.extend(self.order[offset:offset + take])
            start, count = start + take, count - take
        return result


def build_model_config(args, dataset):
    size = args.size or ('tiny' if args.task == 'countdown' else 'mini')
    hidden, heads, layers = SIZES[size]
    memory = {'vanilla': 'none', 'mdm': 'none', 'mdm_aux': 'none',
              'final': 'final', 'dcache': 'dcache', 'both': 'both',
              'both_aux': 'both'}[args.variant]
    tokenizer = dataset.tokenizer
    return dict(
        vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id,
        mask_id=tokenizer.mask_id, special_ids=list(tokenizer.special_ids),
        hidden_size=hidden, n_heads=heads, n_layers=layers,
        max_length=dataset.max_length, memory_mode=memory,
        attention_mode='vanilla' if args.variant == 'vanilla' else 'merged',
        neighbors=args.variant.endswith('_aux'), neighbor_weight=args.neighbor_weight,
        gradient_mode=args.gradient_mode,
        trajectory='single' if args.variant == 'vanilla' else 'five',
        weights=[0.05, 0.10, 0.20, 1.00, 0.70],
        kmin=0.025, kmax=0.10, max_mask_ratio=0.9975,
        cache_only_probability=0.0 if args.no_robustness else 0.20,
        current_only_probability=0.0 if args.no_robustness else 0.05,
        source_dropout_warmup_steps=1000,
        final_dropout=0.0 if args.no_robustness else 0.10,
        identity_probability=0.0 if args.no_robustness else 0.25,
        identity_margin=0.05, identity_weight=0.10,
        identity_final_probability=0.50)


class TrainingForward(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch, step, generator):
        return self.model.compute_loss(batch, step=step, generator=generator, training=True)


def move_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def rng_state():
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda'].cpu())


def save_checkpoint(run_dir, model, optimizer, step, contract, rank, world):
    local = rng_state()
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    if rank == 0:
        directory = Path(run_dir) / 'checkpoints'
        directory.mkdir(parents=True, exist_ok=True)
        # Unique names protect the last valid save when revalidating one step.
        name = f'step-{step:09d}-{uuid.uuid4().hex[:8]}.pt'
        path = directory / name
        tmp = directory / ('.' + name + '.tmp')
        torch.save(dict(format_version=1, step=step, model=model.state_dict(),
                        optimizer=optimizer.state_dict(), contract=contract,
                        model_config=model.config, rng_by_rank=states,
                        examples_seen=step * contract['global_batch']), tmp)
        with tmp.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        atomic_json(path.with_suffix('.pt.json'), dict(
            file=name, step=step, sha256=digest(path), size=path.stat().st_size,
            created_ns=time.time_ns()))
        last = directory / 'last.pt'
        if last.exists() and not last.is_symlink():
            raise ValueError('Refusing to replace a non-symlink last.pt')
        link = directory / ('.last-' + uuid.uuid4().hex)
        link.symlink_to(name)
        os.replace(link, last)
        # Only receipts/files created by this task runner are ever rotated.
        receipts = []
        for receipt in directory.glob('step-*.pt.json'):
            item = json.loads(receipt.read_text())
            if Path(item['file']).name != item['file'] or not item['file'].startswith('step-'):
                raise ValueError('Invalid reasoning checkpoint receipt')
            receipts.append((item['step'], item['created_ns'], item, receipt))
        for _, _, item, receipt in sorted(receipts)[:-3]:
            (directory / item['file']).unlink()
            receipt.unlink()
        print(f'Checkpoint committed at optimizer step {step}: {path}', flush=True)
    if world > 1:
        dist.barrier()


def load_checkpoint(path):
    path = Path(path).resolve(strict=True)
    receipt_path = path.with_suffix('.pt.json')
    if not receipt_path.exists():
        raise ValueError('Reasoning checkpoint requires its adjacent .pt.json receipt')
    receipt = json.loads(receipt_path.read_text())
    if receipt['size'] != path.stat().st_size or receipt['sha256'] != digest(path):
        raise ValueError('Incomplete or corrupted reasoning checkpoint')
    # Only load checkpoints produced by this experiment in a trusted workspace.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('format_version') != 1:
        raise ValueError('Not a reasoning checkpoint; OWT checkpoints are incompatible')
    return checkpoint


class MetricWriter:
    """Per-attempt raw logs; union columns without silently losing auxiliaries."""
    def __init__(self, directory, resume_step):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.rows, self.fields = [], ['step']
        atomic_json(self.directory / 'resume_attempt.json', dict(
            version=1, resume_step=resume_step,
            started_ns=time.time_ns(), attempt_id=uuid.uuid4().hex,
            step_convention='step is COMPLETED optimizer updates; keep prior steps <= resume_step'))

    def log(self, step, values):
        row = {'step': step, **values}
        if any(not math.isfinite(float(v)) for v in row.values()):
            raise FloatingPointError('Nonfinite metric; refusing to hide training failure')
        new_fields = [key for key in row if key not in self.fields]
        self.rows.append(row)
        self.fields.extend(new_fields)
        path = self.directory / 'metrics.csv'
        if new_fields or not path.exists():
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=self.fields)
                writer.writeheader()
                writer.writerows(self.rows)
        else:
            with path.open('a', newline='') as stream:
                csv.DictWriter(stream, fieldnames=self.fields).writerow(row)


def validate(model, dataset, args, device):
    """Fixed examples, batch size and random masks across methods and checkpoints."""
    from reasoning.evaluation import evaluate_corruption
    subset = Subset(dataset, range(min(args.validation_examples, len(dataset))))
    loader = DataLoader(subset, batch_size=args.eval_batch_size, shuffle=False)
    state = rng_state()
    model.eval()
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                             enabled=args.precision == 'bf16'):
            metrics, _ = evaluate_corruption(
                model, loader, device, tokenizer=dataset.tokenizer,
                records=dataset.records, ratios=(0.1, 0.3, 0.5, 0.7),
                seed=args.eval_seed, memory_condition='correct', reset_each_ratio=True)
    finally:
        restore_rng(state)
        model.train()
    # Only scalars enter CSV. Structured per-ratio metrics remain in JSON.
    flat = {}
    def visit(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(prefix + '/' + str(key), child)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[prefix] = float(value)
    visit('val', metrics)
    # Explicit macro mean over the FOUR FIXED corruption levels, not an ELBO.
    ratios = list(metrics.get('ratios', {}).values())
    for source, destination in (('conditional_nll', 'conditional_nll'),
                                ('content_conditional_nll', 'content_conditional_nll')):
        available = [value[source] for value in ratios if value.get(source) is not None]
        if available:
            flat['val/' + destination] = sum(available) / len(available)
    metrics['mean_over_ratios_nll'] = flat.get('val/conditional_nll')
    return flat, metrics


def train(args):
    from reasoning.data import ReasoningDataset
    from reasoning.model import ReasoningModel
    world, rank = int(os.environ.get('WORLD_SIZE', '1')), int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use --device cpu only for small smoke tests')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    if world > 1:
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo')
    if args.global_batch % (world * args.micro_batch):
        raise ValueError('global_batch must divide evenly by world_size * micro_batch')
    accumulation = args.global_batch // (world * args.micro_batch)
    if accumulation < 1:
        raise ValueError('Global batch cannot be smaller than a distributed microbatch')
    if args.precision == 'bf16' and device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise ValueError('Requested BF16 is unsupported on this CUDA device')
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = ReasoningDataset(args.data_dir, 'train')
    validation = ReasoningDataset(args.data_dir, 'validation')
    if len(validation) == 0:
        raise ValueError('Training requires a nonempty validation split')
    if dataset.manifest['task'] != args.task:
        raise ValueError('Task and dataset manifest do not match')
    model_config = build_model_config(args, dataset)
    model = ReasoningModel(model_config).to(device)
    contract = dict(task=args.task, variant=args.variant, model_config=model.config,
                    data_sha256=digest(Path(args.data_dir) / 'manifest.json'),
                    global_batch=args.global_batch, micro_batch=args.micro_batch,
                    world_size=world, seed=args.seed, lr=args.lr,
                    weight_decay=args.weight_decay, warmup_steps=args.warmup_steps,
                    grad_clip=args.grad_clip, precision=args.precision,
                    device_type=device.type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=args.weight_decay)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_lock = None
    if rank == 0:
        run_lock = (run_dir / '.training.lock').open('a')
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if world > 1:
        dist.barrier()
    start = 0
    last = run_dir / 'checkpoints/last.pt'
    if args.fresh and ((run_dir / 'contract.json').exists() or last.exists()):
        raise ValueError('--fresh requires a NEW run directory; nothing was overwritten')
    if last.exists():
        checkpoint = load_checkpoint(last)
        if checkpoint['contract'] != contract:
            raise ValueError('Resume contract differs: data/model/optimizer/batch geometry must match')
        start = int(checkpoint['step'])
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        restore_rng(checkpoint['rng_by_rank'][rank])
    elif (run_dir / 'contract.json').exists():
        if json.loads((run_dir / 'contract.json').read_text()) != contract:
            raise ValueError('Existing run directory belongs to a different configuration')
        if any((run_dir / 'checkpoints').glob('step-*')):
            raise ValueError('Checkpoint files exist without last.pt; inspect before restarting')
    if rank == 0:
        atomic_json(run_dir / 'contract.json', contract)
        atomic_json(run_dir / 'launch.json', dict(
            **vars(args), world_size=world, accumulation=accumulation,
            resume_step=start, parameters=sum(p.numel() for p in model.parameters()),
            resolved_device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            torch_version=torch.__version__, started_ns=time.time_ns(),
            reproduction='BD3-derived matched controls; NOT an exact Latent Tokens reproduction'))
    if start >= args.max_steps:
        if rank == 0:
            print(f'Already completed {start} optimizer updates (target {args.max_steps}).')
        if world > 1:
            dist.destroy_process_group()
        if run_lock:
            run_lock.close()
        return
    wrapped = TrainingForward(model)
    if world > 1:
        wrapped = torch.nn.parallel.DistributedDataParallel(
            wrapped, device_ids=[local_rank] if device.type == 'cuda' else None,
            find_unused_parameters=True)
    model.train()
    if rank == 0:
        attempt = run_dir / 'logs' / f'attempt-{time.time_ns()}-{uuid.uuid4().hex[:8]}'
        writer = MetricWriter(attempt, start)
        atomic_json(run_dir / 'status.json', dict(status='running', step=start,
                                                max_steps=args.max_steps, pid=os.getpid(),
                                                updated_ns=time.time_ns()))
        print(f'{args.task}/{args.variant}: {start} -> {args.max_steps} optimizer updates; '
              f'{world} ranks x {args.micro_batch} x {accumulation} = {args.global_batch}.', flush=True)
        print(f'Output: {run_dir}; validation every {args.val_every}; latest 3 checkpoints.', flush=True)
    stream = GlobalExampleStream(len(dataset), args.seed)
    last_save_time = time.monotonic()
    from torch.utils.data import default_collate
    for step in range(start, args.max_steps):
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        summed = {}
        learning_rate = args.lr * min(1.0, (step + 1) / max(args.warmup_steps, 1))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        for micro in range(accumulation):
            offset = step * args.global_batch + (micro * world + rank) * args.micro_batch
            rows = stream.indices(offset, args.micro_batch)
            batch = move_batch(default_collate([dataset[i] for i in rows]), device)
            generator = torch.Generator(device=device).manual_seed(args.seed + 104729 * offset)
            sync = wrapped.no_sync() if world > 1 and micro + 1 < accumulation else nullcontext()
            with sync:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=args.precision == 'bf16'):
                    loss, metrics = wrapped(batch, step, generator)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f'Nonfinite loss at optimizer update {step + 1}')
                (loss / accumulation).backward()
            for key, value in {**metrics, 'loss': loss.detach()}.items():
                value = torch.as_tensor(value, device=device).detach().float()
                if value.numel() != 1:
                    raise ValueError(f'Metric {key} must be scalar')
                summed[key] = summed.get(key, torch.zeros((), device=device)) + value / accumulation
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip,
                                                       error_if_nonfinite=True)
        optimizer.step()
        completed = step + 1
        if completed % args.log_every == 0 or completed == args.max_steps:
            keys = sorted(summed)
            values = torch.stack([summed[key] for key in keys])
            if world > 1:
                dist.all_reduce(values)
                values /= world
            if rank == 0:
                row = {'train/' + key: float(value) for key, value in zip(keys, values)}
                row.update(lr=learning_rate, grad_norm=float(gradient_norm),
                           seconds_per_update=time.monotonic() - tick,
                           examples_seen=completed * args.global_batch)
                writer.log(completed, row)
                atomic_json(run_dir / 'status.json', dict(status='running', step=completed,
                                                        max_steps=args.max_steps, pid=os.getpid(),
                                                        updated_ns=time.time_ns()))
                print(f'Step {completed}/{args.max_steps}: loss={row["train/loss"]:.5f}, '
                      f'{row["seconds_per_update"]:.2f}s/update', flush=True)
        if completed % args.val_every == 0 or completed == args.max_steps:
            if rank == 0:
                values, full = validate(model, validation, args, device)
                writer.log(completed, values)
                atomic_json(run_dir / 'validation' / f'step-{completed:09d}.json', full)
                print(f'Validation step {completed}: {values}', flush=True)
            if world > 1:
                dist.barrier()
        timer_due = torch.tensor(int(rank == 0 and args.save_seconds > 0 and
                                 time.monotonic() - last_save_time >= args.save_seconds), device=device)
        if world > 1:
            dist.broadcast(timer_due, src=0)
        if completed % args.save_every == 0 or completed == args.max_steps or bool(timer_due):
            save_checkpoint(run_dir, model, optimizer, completed, contract, rank, world)
            last_save_time = time.monotonic()
    if rank == 0:
        atomic_json(run_dir / 'status.json', dict(status='finished', step=args.max_steps))
    if world > 1:
        dist.destroy_process_group()
    if run_lock:
        run_lock.close()


def evaluate(args):
    from reasoning.data import ReasoningDataset
    from reasoning.model import ReasoningModel
    from reasoning.evaluation import evaluate_corruption, evaluate_generation
    torch.set_num_threads(args.cpu_threads)
    checkpoint = load_checkpoint(args.checkpoint)
    dataset = ReasoningDataset(args.data_dir, args.split)
    if len(dataset) == 0:
        raise ValueError('Cannot evaluate an empty split')
    if digest(Path(args.data_dir) / 'manifest.json') != checkpoint['contract']['data_sha256']:
        raise ValueError('Dataset manifest differs from training; explicit OOD evaluation needs a separate protocol')
    device = torch.device(args.device)
    model = ReasoningModel(checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    loader = DataLoader(Subset(dataset, range(min(args.examples, len(dataset)))),
                        batch_size=args.batch_size, shuffle=False)
    kwargs = dict(tokenizer=dataset.tokenizer, records=dataset.records,
                  seed=args.seed, memory_condition=args.memory_condition)
    with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=checkpoint['contract']['precision'] == 'bf16'):
        if args.protocol == 'generate':
            metrics, details = evaluate_generation(model, loader, device,
                                                   policy=args.policy, candidate_k=8, **kwargs)
        else:
            metrics, details = evaluate_corruption(
                model, loader, device, reset_each_ratio=args.protocol == 'cold', **kwargs)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an evaluation: {output}')
    atomic_json(output, dict(checkpoint=str(Path(args.checkpoint).resolve()),
                             step=checkpoint['step'], contract=checkpoint['contract'],
                             arguments=vars(args), metrics=metrics, examples=details))
    print(json.dumps(metrics, indent=2))
    print(f'Wrote {output}')


def plot(args):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] /
                                            '.cache/runtime/reasoning/matplotlib'))
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    found, task = False, None
    for run in args.runs:
        run = Path(run)
        if (run / 'contract.json').exists():
            current_task = json.loads((run / 'contract.json').read_text())['task']
            if task is not None and task != current_task:
                raise ValueError('Plot one task at a time; different task vocabularies have incomparable NLL scales')
            task = current_task
        frames = []
        for meta_path in sorted((run / 'logs').glob('attempt-*/resume_attempt.json')):
            meta = json.loads(meta_path.read_text())
            # Completed-step convention: steps strictly AFTER resume are abandoned.
            frames = [frame.loc[frame.step <= meta['resume_step']] for frame in frames]
            path = meta_path.parent / 'metrics.csv'
            if path.exists():
                frames.append(pd.read_csv(path))
        if not frames:
            continue
        frame = pd.concat(frames, ignore_index=True).groupby('step', as_index=False).last()
        label = run.name
        for axis, column in zip(axes, ('train/loss', args.val_metric)):
            if column in frame:
                data = frame[['step', column]].dropna()
                if not data.empty:
                    values = data[column].rolling(args.smooth if axis is axes[0] else 1,
                                                  min_periods=1).mean()
                    axis.plot(data.step, values, label=label)
                    found = True
        output_table = Path(args.output).with_name(Path(args.output).stem + '-' + run.name + '.csv')
        output_table.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_table, index=False)
    if not found:
        raise ValueError('No matching metrics found; inspect logs/attempt-*/metrics.csv column names')
    for axis, title in zip(axes, ('Training objective (not NLL)', args.val_metric)):
        axis.set(xlabel='Completed optimizer updates', ylabel=title)
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=7)
    figure.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f'Wrote {args.output}')


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest='action', required=True)
    prepare = sub.add_parser('prepare', help='Generate clearly labelled pilot data or import normalized JSONL')
    prepare.add_argument('--task', choices=TASKS, required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--train-size', type=int, default=1000)
    prepare.add_argument('--valid-size', type=int, default=100)
    prepare.add_argument('--test-size', type=int, default=100)
    prepare.add_argument('--seed', type=int, default=17)
    source = prepare.add_mutually_exclusive_group()
    source.add_argument('--input-path', help='Normalized records; no silent dataset download')
    source.add_argument('--split-inputs', help='JSON mapping train/validation/test to normalized JSONL paths; preserve splits')
    fit = sub.add_parser('train', help='Fresh or resume own latest full-state checkpoint')
    fit.add_argument('--task', choices=TASKS, required=True)
    fit.add_argument('--variant', choices=VARIANTS, required=True)
    fit.add_argument('--data-dir', required=True)
    fit.add_argument('--run-dir', required=True)
    fit.add_argument('--size', choices=SIZES)
    fit.add_argument('--max-steps', type=int, default=5000)
    fit.add_argument('--global-batch', type=int, default=128)
    fit.add_argument('--micro-batch', type=int, default=8)
    fit.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    fit.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    fit.add_argument('--gradient-mode', choices=('adjacent', 'detached'), default='adjacent')
    fit.add_argument('--neighbor-weight', type=float, default=0.5)
    fit.add_argument('--no-robustness', action='store_true')
    fit.add_argument('--lr', type=float, default=3e-4)
    fit.add_argument('--weight-decay', type=float, default=0.0)
    fit.add_argument('--warmup-steps', type=int, default=1000)
    fit.add_argument('--grad-clip', type=float, default=1.0)
    fit.add_argument('--seed', type=int, default=1)
    fit.add_argument('--eval-seed', type=int, default=2026)
    fit.add_argument('--eval-batch-size', type=int, default=8)
    fit.add_argument('--validation-examples', type=int, default=128)
    fit.add_argument('--val-every', type=int, default=500)
    fit.add_argument('--save-every', type=int, default=500)
    fit.add_argument('--save-seconds', type=float, default=1200)
    fit.add_argument('--log-every', type=int, default=10)
    fit.add_argument('--cpu-threads', type=int, default=4)
    fit.add_argument('--fresh', action='store_true')
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--checkpoint', required=True)
    evaluation.add_argument('--data-dir', required=True)
    evaluation.add_argument('--output', required=True)
    evaluation.add_argument('--split', choices=('validation', 'test'), default='test')
    evaluation.add_argument('--protocol', choices=('generate', 'cold', 'nested'), default='generate')
    evaluation.add_argument('--policy', choices=('uniform', 'top_prob'), default='top_prob')
    evaluation.add_argument('--memory-condition', choices=(
        'correct', 'none', 'shuffle_dcache', 'shuffle_final', 'shuffle_both'), default='correct')
    evaluation.add_argument('--examples', type=int, default=100)
    evaluation.add_argument('--batch-size', type=int, default=8)
    evaluation.add_argument('--seed', type=int, default=2026)
    evaluation.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    evaluation.add_argument('--cpu-threads', type=int, default=4)
    plotting = sub.add_parser('plot')
    plotting.add_argument('--runs', nargs='+', required=True)
    plotting.add_argument('--output', required=True)
    plotting.add_argument('--smooth', type=int, default=60)
    plotting.add_argument('--val-metric', default='val/conditional_nll')
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    for key in ('max_steps', 'global_batch', 'micro_batch', 'cpu_threads', 'val_every',
                'save_every', 'log_every', 'eval_batch_size', 'validation_examples',
                'examples', 'batch_size', 'smooth'):
        if hasattr(args, key) and getattr(args, key) < 1:
            raise ValueError(f'{key} must be positive')
    if args.action == 'prepare':
        from reasoning.data import prepare_dataset
        input_path = args.input_path
        if args.split_inputs:
            sources_path = Path(args.split_inputs).resolve()
            input_path = json.loads(sources_path.read_text())
            input_path = {key: str((sources_path.parent / value).resolve())
                          for key, value in input_path.items()}
        result = prepare_dataset(args.output, args.task, train_size=args.train_size,
                                 valid_size=args.valid_size, test_size=args.test_size,
                                 seed=args.seed, input_path=input_path)
        print(json.dumps(result, indent=2))
    elif args.action == 'train':
        if args.lr <= 0 or args.grad_clip <= 0 or args.weight_decay < 0 or args.warmup_steps < 0:
            raise ValueError('Require positive lr/grad_clip and nonnegative weight_decay/warmup_steps')
        train(args)
    elif args.action == 'evaluate':
        evaluate(args)
    else:
        plot(args)


if __name__ == '__main__':
    main()
