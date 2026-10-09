"""Isolated, restartable paper-informed Zebra baseline; never changes DiT jobs."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import torch
from torch.utils.data import DataLoader, Subset, default_collate

from .data import ReasoningDataset
from .evaluation import evaluate_generation
from .runner import (GlobalExampleStream, MetricWriter, atomic_json, digest,
                     examples_at_step, load_checkpoint, move_batch, restore_rng,
                     rng_state, save_checkpoint, validate)
from .tfw import (PaperMDM, UPSTREAM_COMMIT, corrupt, decode, full_mask_mixture,
                  learning_rate, loss_terms)
from .tasks import score_prediction


def model_config(dataset, debug=False, logit_shift=1, target_region='padded_tail', padding_attention='visible',
                 zebra_encoding='absolute'):
    tokenizer = dataset.tokenizer
    config = dict(family='tfw_gpt2_v1', vocab_size=tokenizer.vocab_size,
                mask_id=tokenizer.mask_id, pad_id=tokenizer.pad_id,
                bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
                n_positions=512, hidden_size=32 if debug else 512,
                n_layers=2 if debug else 6, n_heads=4 if debug else 8,
                dropout=0.1, memory_mode='none')
    if type(logit_shift) is not int or logit_shift not in (0, 1):
        raise ValueError('logit_shift must be 0 or 1')
    # No new key for the historical default: old strict resumes still work.
    if logit_shift == 0:
        config['logit_shift'] = 0
    if target_region != 'padded_tail':
        config['target_region'] = target_region
    if padding_attention != 'visible':
        config['padding_attention'] = padding_attention
    if zebra_encoding != 'absolute':
        c_id, n_id = tokenizer.encode(['c', 'n'])
        config.update(zebra_encoding=zebra_encoding, zebra_public_tokens=dict(
            sep_id=tokenizer.sep_id, c_id=c_id, n_id=n_id, digit_ids=tokenizer.encode(list('012345'))))
    return config


def validate_fork(parent, contract, source):
    """Allow only an explicitly labelled hard-start corruption intervention."""
    old = parent['contract']
    if ('overfit_train_indices' in old or 'fork' in old or old.get('full_mask_probability', 0)
            or old.get('target_region') != 'answer' or old.get('padding_attention') != 'masked'
            or old.get('logit_shift') != 0):
        raise ValueError('Fork source must be the ordinary repaired full-data baseline')
    candidate = {k: v for k, v in contract.items()
                 if k not in ('full_mask_probability', 'mixture_normalizer')}
    if old != candidate:
        raise ValueError('Fork contract differs beyond the allowed full-mask intervention')
    if parent['examples_seen'] != examples_at_step(parent['step'], old):
        raise ValueError('Invalid parent data cursor')
    return dict(checkpoint=str(Path(source).resolve()), sha256=digest(source),
                step=parent['step'], examples_seen=parent['examples_seen'],
                restored='model, optimizer, RNG, LR schedule step, data cursor',
                intervention='full_mask_probability; shared pre-mixture loss denominator')


def generation_report(model, dataset, args, device, step, run, split):
    output = run / 'generation' / f'{split}-step-{step:09d}.json'
    checkpoint_path = (run / 'checkpoints/last.pt').resolve(strict=True)
    checkpoint_hash = digest(checkpoint_path)
    protocol = ('tfw-diagnostic-v1' if model.target_region == 'padded_tail'
                else 'tfw-answer-only-v2')
    if output.exists():
        saved = json.loads(output.read_text())
        if (saved['step'] != step or saved['data_sha256'] != digest(Path(args.data_dir) / 'manifest.json')
                or saved['protocol_version'] != protocol
                or saved['seed'] != args.eval_seed or saved['checkpoint_sha256'] != checkpoint_hash):
            raise ValueError('Existing generation provenance differs')
        return
    count = min(len(dataset), args.generation_examples if split == 'test' else 128)
    loader = DataLoader(Subset(dataset, range(count)), batch_size=args.eval_batch_size, shuffle=False)
    state, training = rng_state(), model.training
    model.eval()
    reports = {}
    try:
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                   enabled=args.precision == 'bf16'):
            for policy in ('upstream_remask', 'paper_monotonic'):
                details = []
                generator = torch.Generator(device=device).manual_seed(args.eval_seed)
                for batch in loader:
                    batch = move_batch(batch, device)
                    prediction = decode(model, batch, generator, steps=50, policy=policy)
                    for row, index in enumerate(batch['record_index'].tolist()):
                        ids = prediction[row, batch['target_mask'][row]].tolist()
                        tokens = dataset.tokenizer.decode(ids)
                        record = dataset.records[index]
                        details.append(dict(id=record['id'], record_index=index, slot_tokens=tokens,
                                            scores=score_prediction(record, tokens)))
                metrics = {key: sum(float(item['scores'][key]) for item in details) / len(details)
                           for key in details[0]['scores']}
                metrics['answer_cell_pad_fraction'] = sum(
                    item['slot_tokens'][:-1].count('[PAD]') for item in details) / sum(
                        len(item['slot_tokens']) - 1 for item in details)
                reports[policy] = dict(metrics=metrics, examples=details, nfe_per_example=50)
            metrics, details = evaluate_generation(
                model, loader, device, tokenizer=dataset.tokenizer, records=dataset.records,
                seed=args.eval_seed, policy='top_prob', candidate_k=8)
            reports['matched_candidate8'] = dict(metrics=metrics, examples=details)
    finally:
        restore_rng(state)
        model.train(training)
    atomic_json(output, dict(step=step, split=split, seed=args.eval_seed, num_examples=count,
                            checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
                            protocol_version=protocol,
                            data_sha256=digest(Path(args.data_dir) / 'manifest.json'),
                            model_config=model.config, reports=reports,
                            note='Three separately labelled decoders; not exact paper reproduction.'))
    print('Generation:', output, {p: r['metrics'].get('valid_solution') for p, r in reports.items()}, flush=True)


def training_sanity_report(model, dataset, indices, args, device, step, run):
    """Memorization diagnostic on TRAIN examples; never a test-set result.

    Full-mask input is constructed from the public layout. The gold answer is
    used only after prediction for scoring, including the greedy rollout.
    """
    state, training = rng_state(), model.training
    model.eval()
    totals = dict(content_tokens=0, correct=0, predicted_pad=0,
                  puzzles=0, one_pass_exact=0, rollout_exact=0, rollout_pad=0)
    predictions = []
    try:
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                   enabled=args.precision == 'bf16'):
            for begin in range(0, len(indices), args.eval_batch_size):
                chosen = indices[begin:begin + args.eval_batch_size]
                batch = move_batch(default_collate([dataset[i] for i in chosen]), device)
                gold, target = batch['input_ids'], batch['target_mask'].bool()
                # Use the same canvas convention as this model's train/decode.
                from .tfw import prediction_mask
                current = gold.masked_fill(prediction_mask(batch, model.target_region), model.mask_id)
                logits = model(current, attention_mask=batch['attention_mask'])['logits'].float()
                predicted = logits.argmax(-1)
                content = target & gold.ne(dataset.tokenizer.eos_id)
                rollout = decode(model, batch, torch.Generator(device=device).manual_seed(args.eval_seed),
                                 steps=50, policy='paper_monotonic', noise=0.)
                totals['content_tokens'] += int(content.sum())
                totals['correct'] += int(((predicted == gold) & content).sum())
                totals['predicted_pad'] += int((predicted.eq(model.pad_id) & content).sum())
                totals['puzzles'] += len(chosen)
                totals['one_pass_exact'] += int(((predicted == gold) | ~content).all(-1).sum())
                totals['rollout_exact'] += int(((rollout == gold) | ~content).all(-1).sum())
                totals['rollout_pad'] += int((rollout.eq(model.pad_id) & content).sum())
                for row, index in enumerate(chosen):
                    predictions.append(dict(train_index=index,
                        gold=gold[row, target[row]].tolist(),
                        one_pass=predicted[row, target[row]].tolist(),
                        rollout=rollout[row, target[row]].tolist()))
    finally:
        restore_rng(state)
        model.train(training)
    result = dict(step=step, split='training_memorization_diagnostic', **totals,
        full_mask_content_accuracy=totals['correct'] / totals['content_tokens'],
        full_mask_pad_fraction=totals['predicted_pad'] / totals['content_tokens'],
        greedy_rollout_exact=totals['rollout_exact'] / totals['puzzles'],
        greedy_rollout_pad_fraction=totals['rollout_pad'] / totals['content_tokens'],
        predictions=predictions, note='Training subset only; NOT held-out reasoning accuracy.')
    atomic_json(run / 'sanity' / f'step-{step:09d}.json', result)
    print(f'Train-subset sanity {step}: full-mask content accuracy='
          f'{result["full_mask_content_accuracy"]:.4f}, greedy exact={result["greedy_rollout_exact"]:.4f}, '
          f'PAD={result["greedy_rollout_pad_fraction"]:.4f}', flush=True)


def full_mask_validation(model, dataset, args, device):
    """Held-out hard-start diagnostic, kept separate from the historic NLL."""
    totals = dict(tokens=0, content_tokens=0, nll_sum=0., content_nll_sum=0.,
                  content_correct=0, content_pad=0, first_content_correct=0)
    loader = DataLoader(Subset(dataset, range(min(len(dataset), args.validation_examples))),
                        batch_size=args.eval_batch_size, shuffle=False)
    state, training = rng_state(), model.training
    model.eval()
    try:
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                   enabled=args.precision == 'bf16'):
            for batch in loader:
                batch = move_batch(batch, device)
                gold, mask = batch['input_ids'], batch['target_mask'].bool()
                logits = model(gold.masked_fill(mask, model.mask_id),
                               attention_mask=batch['attention_mask'])['logits'].float().clone()
                logits[..., model.mask_id] = -torch.inf  # Match common validation convention.
                content = mask & gold.ne(dataset.tokenizer.eos_id)
                predictions = logits.argmax(-1)
                confidence = logits.softmax(-1).amax(-1).masked_fill(~content, -torch.inf)
                chosen = confidence.argmax(-1, keepdim=True)
                totals['first_content_correct'] += int((predictions.gather(1, chosen)
                                                        == gold.gather(1, chosen)).sum())
                totals['tokens'] += int(mask.sum())
                totals['content_tokens'] += int(content.sum())
                totals['nll_sum'] += float(torch.nn.functional.cross_entropy(logits[mask], gold[mask], reduction='sum'))
                totals['content_nll_sum'] += float(torch.nn.functional.cross_entropy(logits[content], gold[content], reduction='sum'))
                totals['content_correct'] += int(((predictions == gold) & content).sum())
                totals['content_pad'] += int((predictions.eq(model.pad_id) & content).sum())
    finally:
        restore_rng(state)
        model.train(training)
    return dict(conditional_nll=totals['nll_sum']/totals['tokens'],
                content_nll=totals['content_nll_sum']/totals['content_tokens'],
                content_accuracy=totals['content_correct']/totals['content_tokens'],
                first_confident_content_accuracy=totals['first_content_correct']/min(len(dataset), args.validation_examples),
                content_pad_fraction=totals['content_pad']/totals['content_tokens'],
                num_examples=min(len(dataset), args.validation_examples), mask_ratio=1.0)


def train(args):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This separate control supports one GPU only')
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if torch.cuda.device_count() != 1:
            raise ValueError('Select exactly one GPU via CUDA_VISIBLE_DEVICES')
        torch.cuda.set_device(0)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = ReasoningDataset(args.data_dir, 'train')
    valid = ReasoningDataset(args.data_dir, 'validation')
    test = ReasoningDataset(args.data_dir, 'test')
    if data.task != 'zebra-benchmark':
        raise ValueError('This launch is restricted to the prepared Zebra benchmark')
    overfit = getattr(args, 'overfit_examples', 0)
    if overfit and (not args.stop_after_steps or overfit > len(data) or overfit < 1):
        raise ValueError('Overfit diagnostics need a valid subset size and explicit step limit')
    indices = sorted(random.Random(args.seed).sample(range(len(data)), overfit)) if overfit else None
    train_data = Subset(data, indices) if overfit else data
    per_epoch = math.ceil(len(train_data) / args.global_batch)
    # Preserve the full-data LR horizon even in the tiny overfit diagnostic.
    total = args.schedule_epochs * math.ceil(len(data) / args.global_batch)
    stop = min(total, args.stop_after_steps or args.probe_epochs * per_epoch)
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / '.training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logit_shift = getattr(args, 'logit_shift', 1)
    target_region = getattr(args, 'target_region', 'padded_tail')
    padding_attention = getattr(args, 'padding_attention', 'visible')
    full_mask_probability = getattr(args, 'full_mask_probability', 0.)
    if (not math.isfinite(full_mask_probability) or not 0 <= full_mask_probability <= 1
            or (full_mask_probability and (target_region != 'answer' or padding_attention != 'masked'))):
        raise ValueError('Full-mask mixture requires probability in [0,1] and repaired answer-only mode')
    zebra_encoding = getattr(args, 'zebra_encoding', 'absolute')
    if zebra_encoding != 'absolute':
        from .zebra_encoding import audit_public_dimensions
        public_audit = {name: audit_public_dimensions(ds) for name, ds in
                        (('train', data), ('validation', valid), ('test', test))}
        atomic_json(run / 'public_layout_audit.json', public_audit)
    model = PaperMDM(model_config(data, args.debug, logit_shift, target_region, padding_attention,
                                 zebra_encoding)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.)
    contract = dict(task=data.task, variant='tfw_gpt2_defaults' if logit_shift else 'tfw_gpt2_no_shift',
                    model_config=model.config,
                    world_size=1, global_batch=args.global_batch, micro_batch=args.micro_batch,
                    epoch_examples=len(train_data), epochs=args.schedule_epochs,
                    tail_policy='partial_batch_no_wrap_no_drop', data_sha256=digest(Path(args.data_dir) / 'manifest.json'),
                    lr=args.lr, scheduler='linear_300epoch_horizon_no_warmup',
                    weight_decay=0., grad_clip=1., seed=args.seed, precision=args.precision,
                    device_type=device.type, corruption_timesteps=64,
                    loss='global_masked_token_mean_times_inverse_discrete_timestep',
                    focal=False, supervise_padding=True, logit_shift=logit_shift, upstream_commit=UPSTREAM_COMMIT)
    if target_region != 'padded_tail' or padding_attention != 'visible':
        contract.update(variant=contract['variant'] + '_answer_only', target_region=target_region,
                        padding_attention=padding_attention, supervise_padding=False)
    if full_mask_probability:
        contract.update(full_mask_probability=full_mask_probability,
                        mixture_normalizer='pre_mixture_global_masked_count')
    if overfit:
        contract['overfit_train_indices'] = indices
        # A tiny subset makes many more data epochs than the full-data LR
        # horizon. Do not let examples_at_step clamp and repeat ONE corruption
        # forever after schedule_epochs tiny epochs.
        contract['epochs'] = math.ceil(total / per_epoch)
        contract['lr_schedule_full_data_epochs'] = args.schedule_epochs
        contract['overfit_cursor_version'] = 2
    last = run / 'checkpoints/last.pt'
    start = 0
    parent = None
    fork_from = getattr(args, 'fork_from', None)
    if fork_from:
        if overfit:
            raise ValueError('Cannot fork a full-data run into a memorization diagnostic')
        if Path(fork_from).resolve().parent == (run / 'checkpoints').resolve():
            raise ValueError('Fork destination must differ from source; use ordinary resume')
        parent = load_checkpoint(fork_from)
        contract['fork'] = validate_fork(parent, contract, fork_from)
    elif last.exists():
        # A fork subsequently resumes its own strict contract without needing
        # its immutable parent checkpoint to remain on disk indefinitely.
        saved_contract = json.loads((run / 'contract.json').read_text())
        if 'fork' in saved_contract:
            contract['fork'] = saved_contract['fork']
    if last.exists():
        checkpoint = load_checkpoint(last)
        if checkpoint['contract'] != contract:
            raise ValueError('TFW resume contract mismatch')
        start = checkpoint['step']
        if checkpoint['examples_seen'] != examples_at_step(start, contract):
            raise ValueError('Invalid data cursor')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        restore_rng(checkpoint['rng_by_rank'][0])
    elif (run / 'contract.json').exists():
        if json.loads((run / 'contract.json').read_text()) != contract:
            raise ValueError('Existing run configuration differs')
        if any((run / 'checkpoints').glob('step-*.pt')):
            raise ValueError('Checkpoint present without last.pt; inspect before restarting')
    if not last.exists() and parent is not None:
        start = parent['step']
        model.load_state_dict(parent['model'], strict=True)
        optimizer.load_state_dict(parent['optimizer'])
        restore_rng(parent['rng_by_rank'][0])
        if stop <= start:
            raise ValueError('Fork stopping step must exceed the parent step')
    del parent
    atomic_json(run / 'contract.json', contract)
    if fork_from and not last.exists():
        # A committed starting checkpoint makes shutdown before the first new
        # save recoverable without silently falling back to random weights.
        save_checkpoint(run, model, optimizer, start, contract, 0, 1)
    atomic_json(run / 'launch.json', dict(**vars(args), resume_step=start,
                parameters=sum(p.numel() for p in model.parameters()), steps_per_epoch=per_epoch,
                total_schedule_steps=total, stop_step=stop, started_ns=time.time_ns()))
    print(f'TFW-informed GPT-2: {start} -> {stop}; scheduler horizon {total}; '
          f'parameters {sum(p.numel() for p in model.parameters()):,}; global={args.global_batch}; '
          f'micro={args.micro_batch}; source data {len(data):,}.', flush=True)
    print(f'Prediction alignment: logit_shift={logit_shift} '
          f'({"upstream previous-position hidden" if logit_shift else "same-position hidden"}).', flush=True)
    # A checkpoint can precede an interrupted epoch-end generation evaluation.
    if start and not overfit and ((start % per_epoch == 0 and not getattr(args, 'no_epoch_end_generation', False)) or
                                  (getattr(args, 'final_generation', False) and start == stop)):
        generation_report(model, test, args, device, start, run, 'test')
    if start >= stop:
        atomic_json(run / 'status.json', dict(status='paused' if start < total else 'finished', step=start, stop_step=stop))
        return
    writer = MetricWriter(run / 'logs' / f'attempt-{time.time_ns()}-{uuid.uuid4().hex[:8]}', start)
    stream = GlobalExampleStream(len(train_data), args.seed)
    last_save = time.monotonic()
    if overfit and start == 0:
        training_sanity_report(model, data, indices, args, device, 0, run)
    for step in range(start, stop):
        tick = time.monotonic()
        lr = learning_rate(step, args.lr, total)
        for group in optimizer.param_groups:
            group['lr'] = lr
        offset = examples_at_step(step, contract)
        count = min(args.global_batch, len(train_data) - offset % len(train_data))
        if examples_at_step(step + 1, contract) != offset + count:
            raise ValueError('Training cursor stopped advancing; refusing to repeat a fixed corruption')
        batch = move_batch(default_collate([train_data[i] for i in stream.indices(offset, count)]), device)
        generator = torch.Generator(device=device).manual_seed(args.seed + 104729 * offset)
        inputs, masked, j, eligible = corrupt(batch, model.mask_id, 64, generator, target_region)
        # Shared with the control: do not accidentally lower the whole loss
        # scale by increasing the denominator when adding full-mask examples.
        denominator = masked.sum().clamp_min(1)
        inputs, masked, j, forced = full_mask_mixture(
            inputs, masked, j, eligible, model.mask_id, generator, full_mask_probability)
        if target_region == 'answer' and bool((masked & batch['input_ids'].eq(model.pad_id)).any()):
            raise ValueError('Answer-only training must never supervise PAD')
        optimizer.zero_grad(set_to_none=True)
        totals = {}
        for begin in range(0, count, args.micro_batch):
            end = min(count, begin + args.micro_batch)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
                logits = model(inputs[begin:end], attention_mask=batch['attention_mask'][begin:end])['logits']
                numerator, terms = loss_terms(logits, batch['input_ids'][begin:end], masked[begin:end],
                                              j[begin:end], batch['target_mask'][begin:end], model.pad_id)
            loss = numerator / denominator
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Nonfinite loss at {step + 1}')
            loss.backward()
            for key, value in terms.items():
                totals[key] = totals.get(key, 0) + float(value)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        completed = step + 1
        if completed % args.log_every == 0 or completed == stop:
            row = {
                'train/loss': totals['weighted_sum'] / float(denominator),
                'train/unweighted_padded_nll': totals['ce_sum'] / max(1, totals['masked_count']),
                'train/unweighted_target_nll': totals['ce_sum'] / max(1, totals['masked_count']),
                'train/answer_nll': totals['answer_ce_sum'] / max(1, totals['answer_count']),
                'train/answer_accuracy': totals['answer_correct'] / max(1, totals['answer_count']),
                'train/answer_predicted_pad_fraction': totals['answer_predicted_pad'] / max(1, totals['answer_count']),
                'train/padding_supervision_fraction': totals['padding_count'] / max(1, totals['masked_count']),
                'train/answer_masked_count': totals['answer_count'],
                'train/forced_full_mask_fraction': float(forced.float().mean()),
                'train/objective_denominator': float(denominator),
                'train/full_answer_mask_fraction': float(((masked & batch['target_mask']).sum(-1)
                    == batch['target_mask'].sum(-1)).float().mean()),
                'train/full_tail_mask_fraction': float((masked.sum(-1) == eligible.sum(-1)).float().mean()),
                'lr': lr, 'grad_norm': float(grad_norm), 'seconds_per_update': time.monotonic() - tick,
                'examples_seen': examples_at_step(completed, contract),
                'epoch_fraction': examples_at_step(completed, contract) / len(train_data)}
            writer.log(completed, row)
            atomic_json(run / 'status.json', dict(status='running', step=completed, stop_step=stop,
                                                schedule_steps=total, pid=os.getpid(), updated_ns=time.time_ns()))
            print(f'Step {completed}/{stop}: weighted loss={row["train/loss"]:.5f}, '
                  f'answer NLL={row["train/answer_nll"]:.4f}, {row["seconds_per_update"]:.2f}s/update', flush=True)
        epoch_end = completed % per_epoch == 0 and not overfit
        if overfit and (completed % args.sanity_every == 0 or completed == stop):
            training_sanity_report(model, data, indices, args, device, completed, run)
        if completed % args.val_every == 0 or completed == stop or epoch_end:
            values, full = validate(model, valid, args, device)
            if target_region == 'answer':
                hard = full_mask_validation(model, valid, args, device)
                full['full_mask_diagnostic'] = hard
                values.update({'val/full_mask/' + key: value for key, value in hard.items()})
            writer.log(completed, values)
            atomic_json(run / 'validation' / f'step-{completed:09d}.json', full)
            print(f'Validation {completed}: common answer NLL={values["val/conditional_nll"]:.5f}', flush=True)
        generation_due = bool(not overfit and args.generation_every and completed % args.generation_every == 0)
        if (completed % args.save_every == 0 or completed == stop or epoch_end or generation_due
                or time.monotonic() - last_save >= args.save_seconds):
            save_checkpoint(run, model, optimizer, completed, contract, 0, 1)
            last_save = time.monotonic()
        if (epoch_end and not getattr(args, 'no_epoch_end_generation', False)) or (getattr(args, 'final_generation', False) and completed == stop and not overfit):
            generation_report(model, test, args, device, completed, run, 'test')
        elif generation_due:
            generation_report(model, valid, args, device, completed, run, 'validation')
    atomic_json(run / 'status.json', dict(status='paused' if stop < total else 'finished',
                step=stop, schedule_steps=total, examples_seen=examples_at_step(stop, contract),
                reason='Requested bounded probe complete; explicit extension required'))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--data-dir', required=True)
    result.add_argument('--run-dir', required=True)
    result.add_argument('--probe-epochs', type=int, default=3)
    result.add_argument('--schedule-epochs', type=int, default=300)
    result.add_argument('--stop-after-steps', type=int)
    result.add_argument('--global-batch', type=int, default=128)
    result.add_argument('--micro-batch', type=int, default=32)
    result.add_argument('--lr', type=float, default=0.001)
    result.add_argument('--logit-shift', type=int, choices=(0, 1), default=1,
                        help='1 preserves the upstream shifted baseline; 0 is a separate same-position ablation')
    result.add_argument('--target-region', choices=('padded_tail', 'answer'), default='padded_tail',
                        help='answer uses public grid+EOS only; legacy padded_tail also supervises outer PAD')
    result.add_argument('--padding-attention', choices=('visible', 'masked'), default='visible',
                        help='masked excludes outer padding keys in all train/eval paths; requires answer targets')
    result.add_argument('--overfit-examples', type=int, default=0,
                        help='TRAIN-only memorization diagnostic; requires explicit --stop-after-steps')
    result.add_argument('--sanity-every', type=int, default=100)
    result.add_argument('--zebra-encoding', choices=('absolute', 'answer_relative', 'typed_coordinates'),
                        default='absolute', help='Opt-in public positional/coordinate representation diagnostic')
    result.add_argument('--full-mask-probability', type=float, default=0.,
                        help='Separate hard-start objective; force this fraction fully masked, keep shared pre-mixture normalization')
    result.add_argument('--fork-from', help='Trusted repaired baseline checkpoint: copy full state into a NEW run, optionally change full-mask probability only')
    result.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16')
    result.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    result.add_argument('--seed', type=int, default=1)
    result.add_argument('--eval-seed', type=int, default=2026)
    result.add_argument('--cpu-threads', type=int, default=4)
    result.add_argument('--validation-examples', type=int, default=1000)
    result.add_argument('--eval-batch-size', type=int, default=32)
    result.add_argument('--val-every', type=int, default=500)
    result.add_argument('--save-every', type=int, default=500)
    result.add_argument('--save-seconds', type=int, default=1200)
    result.add_argument('--log-every', type=int, default=10)
    result.add_argument('--generation-every', type=int, default=5000)
    result.add_argument('--generation-examples', type=int, default=1000)
    result.add_argument('--final-generation', action='store_true',
                        help='Also evaluate the frozen test subset at a non-epoch stopping boundary')
    result.add_argument('--no-epoch-end-generation', action='store_true',
                        help='Skip epoch-boundary generation only; explicit final/periodic evaluation still applies')
    result.add_argument('--debug', action='store_true')
    return result


if __name__ == '__main__':
    args = parser().parse_args()
    for key in ('probe_epochs', 'schedule_epochs', 'global_batch', 'micro_batch', 'cpu_threads',
                'validation_examples', 'eval_batch_size', 'val_every', 'save_every', 'save_seconds',
                'log_every', 'generation_examples', 'sanity_every'):
        if getattr(args, key) < 1:
            raise ValueError(key + ' must be positive')
    if (args.probe_epochs > args.schedule_epochs or args.global_batch % args.micro_batch
            or not math.isfinite(args.lr) or args.lr <= 0 or args.generation_every < 0
            or args.overfit_examples < 0
            or (args.stop_after_steps is not None and args.stop_after_steps < 1)):
        raise ValueError('Invalid epoch, batch, LR, generation or stopping budget')
    train(args)
