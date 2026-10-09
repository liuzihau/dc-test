"""Frozen near-clean, first-input and same-source diagnostics after core sweep.

No torch/CUDA initialization precedes the shared GPU lock. Core evidence stays
immutable. Final runs use only MDM and zero NP; CPU preflight may use MDM alone.
"""
import argparse
import csv
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from owt.post_diagnostic_canvases import anchor_pair, source_pairs
from owt.research import ROOT, atomic_write, read_json, record_event
from owt.reveal_sweep import digest, score_batch

VARIANTS = ['mdm', 'mdm_np_zero_init']


def score_targets(model, clean, cases, device):
    """One main pass and only the selected source's native auxiliary readout."""
    import torch
    inputs = torch.as_tensor(np.stack([c['canvas'] for c in cases]), device=device)
    batch = torch.arange(len(cases), device=device)
    target = torch.tensor([c['target_index'] for c in cases], device=device)
    labels = torch.as_tensor(clean, device=device)[batch, target]
    captured = {}
    handle = None
    if model.np_config.enabled:
        handle = model.backbone.output_layer.linear.register_forward_pre_hook(
            lambda module, args: captured.update(hidden=args[0].detach()))
    try:
        with torch.inference_mode():
            logp = model(inputs, torch.zeros((len(cases), 1), device=device))
            if logp.dtype != torch.float32:
                raise RuntimeError('Selected-target diagnostics require FP32')
            selected = logp[batch, target]
            main_true = selected.gather(-1, labels[:, None]).squeeze(-1)
            top, prediction = selected.max(-1)
            rows = [dict(main_target_ce=float(-v), main_target_correct=int(p == y),
                         main_prediction=int(p), main_true_logp=float(v),
                         main_top1_probability=float(t.exp()))
                    for v, p, y, t in zip(main_true, prediction, labels, top)]
            if model.np_config.enabled:
                hidden = captured['hidden']
                if hidden.dtype != torch.float32:
                    raise RuntimeError('Selected-source hidden state is not FP32')
                heads = dict(zip(model.backbone.neighbor_heads.offsets,
                                 model.backbone.neighbor_heads.heads))
                for direction in (-1, 1):
                    indices = [k for k, c in enumerate(cases) if c['direction'] == direction]
                    if not indices:
                        continue
                    sources = [cases[k]['source_index'] for k in indices]
                    logits = heads[-direction](hidden[indices, sources]).clone()
                    if logits.dtype != torch.float32:
                        raise RuntimeError('Selected-source head is not FP32')
                    logits[:, model.mask_index] = -torch.inf
                    aux = logits.log_softmax(-1)
                    values = aux.gather(-1, labels[indices, None]).squeeze(-1)
                    aux_top, aux_prediction = aux.max(-1)
                    for k, v, p, t in zip(indices, values, aux_prediction, aux_top):
                        rows[k].update(auxiliary_target_ce=float(-v),
                            auxiliary_target_correct=int(p == labels[k]),
                            auxiliary_prediction=int(p), auxiliary_true_logp=float(v),
                            auxiliary_top1_probability=float(t.exp()))
            if not all(np.isfinite(r['main_target_ce']) and
                       np.isfinite(r.get('auxiliary_target_ce', 0)) for r in rows):
                raise FloatingPointError('Nonfinite selected-target diagnostic')
            return rows
    finally:
        if handle is not None:
            handle.remove()


def write_rows(path, rows):
    columns = sorted({k for r in rows for k in r})
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_core_row(row):
    return {k: (v if k == 'variant' or 'sha256' in k else float(v))
            for k, v in row.items() if v != ''}


def summarize_sources(rows, variants, bootstraps=2000):
    lookup = {}
    for row in rows:
        key = tuple(row[k] for k in ('variant', 'row_id', 'mask_ratio', 'direction', 'source_state'))
        if key in lookup:
            raise ValueError('Duplicate selected-source observation')
        lookup[key] = row
    result = []
    for ratio in (.1, .2, .6):
        ids = sorted({r['row_id'] for r in rows if r['mask_ratio'] == ratio})
        if not ids:
            result.append(dict(mask_ratio=ratio, rows=0, unavailable=True))
            continue
        benefits, gaps = {}, {}
        for variant in variants:
            values = []
            for row_id in ids:
                per_direction = []
                for d in (-1, 1):
                    latent = lookup[variant, row_id, ratio, d, 'masked']
                    visible = lookup[variant, row_id, ratio, d, 'revealed']
                    base_latent = lookup['mdm', row_id, ratio, d, 'masked']
                    base_visible = lookup['mdm', row_id, ratio, d, 'revealed']
                    for state, pair, base in [('masked', latent, base_latent), ('revealed', visible, base_visible)]:
                        for field in ('target_index', 'source_index', 'input_sha256', 'clean_sha256'):
                            assert pair[field] == base[field], 'Source intervention model pairing changed'
                        gaps.setdefault((variant, d, state), []).append(pair['main_target_ce'] - base['main_target_ce'])
                    assert latent['target_index'] == visible['target_index']
                    assert latent['source_index'] == visible['source_index']
                    per_direction.append(latent['main_target_ce'] - visible['main_target_ce'])
                values.append(per_direction)
            benefits[variant] = np.asarray(values)
        rng = np.random.default_rng(20261005)
        draws = rng.integers(0, len(ids), size=(bootstraps, len(ids)))
        for variant in variants:
            for direction in ('both', -1, 1):
                b = benefits[variant].mean(1) if direction == 'both' else benefits[variant][:, 0 if direction == -1 else 1]
                reference = benefits['mdm'].mean(1) if direction == 'both' else benefits['mdm'][:, 0 if direction == -1 else 1]
                interaction = b - reference
                item = dict(variant=variant, mask_ratio=ratio, direction=direction, rows=len(ids),
                    source_reveal_benefit=float(b.mean()), np_minus_mdm_benefit=float(interaction.mean()),
                    benefit_row_bootstrap_95=np.quantile(b[draws].mean(1), [.025, .975]).tolist(),
                    interaction_row_bootstrap_95=np.quantile(interaction[draws].mean(1), [.025, .975]).tolist())
                for state in ('masked', 'revealed'):
                    values = (np.mean([gaps[variant, d, state] for d in (-1, 1)], axis=0)
                              if direction == 'both' else np.asarray(gaps[variant, direction, state]))
                    item[state + '_main_ce_gap_vs_mdm'] = float(values.mean())
                result.append(item)
    return result


def run(args):
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif os.environ.get('CUDA_VISIBLE_DEVICES') != '2,3':
        raise RuntimeError('Use only physical CUDA_VISIBLE_DEVICES=2,3')
    import torch
    from datasets import load_from_disk
    from owt.checkpoint import load_ema_model
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda' and torch.cuda.device_count() != 2:
        raise RuntimeError('Expected the two authorized GPUs')
    device = 'cpu' if args.device == 'cpu' else 'cuda:0'
    core = read_json(args.core / 'summary.json')
    if not core:
        raise RuntimeError('Complete the core sweep before the supplement')
    protocol = core['protocol']
    if protocol['anchor'] != 'none' or protocol['seed'] != 20261004:
        raise RuntimeError('Core corruption protocol differs from the selected protocol')
    row_ids = protocol['row_ids'][:args.examples] if args.preflight else protocol['row_ids']
    if not args.preflight and (len(row_ids) != 1024 or protocol['variants'] != VARIANTS):
        raise RuntimeError('Final supplement requires the complete two-arm 1,024-row core sweep')
    source_ids = sorted(np.random.default_rng(20261005).choice(
        row_ids, min(args.source_examples, len(row_ids)), replace=False).tolist())
    with (args.core / 'observations.csv').open() as stream:
        reference = {(r['variant'], int(r['row_id'])): r for r in csv.DictReader(stream)
                     if float(r['mask_ratio']) == .2 and float(r['correct_fraction']) == 1.}
    contracts = {v: read_json(args.root / v / 'contract.json') for v in args.variants}
    for variant in args.variants:
        if read_json(args.root / variant / 'complete.json')['optimizer_step'] != 5000:
            raise RuntimeError('Only completed 5,000-step checkpoints are eligible')
        if contracts[variant]['cache'] != contracts['mdm']['cache']:
            raise RuntimeError('Different prepared data across arms')
    cache = Path(contracts['mdm']['cache']['validation']['path'])
    if hashlib.sha256((cache / 'state.json').read_bytes()).hexdigest() != contracts['mdm']['cache']['validation']['state_sha256']:
        raise RuntimeError('Validation cache manifest changed')
    dataset = load_from_disk(str(cache))
    all_cases, omissions = [], []
    for row_id in source_ids:
        clean = np.asarray(dataset[row_id]['input_ids'], dtype=np.int64)
        for ratio in (.1, .2, .6):
            cases = source_pairs(clean, ratio, row_id)
            if not cases:
                omissions.append(dict(row_id=row_id, mask_ratio=ratio, reason='No eligible masked target with two non-anchor content sources'))
            all_cases.extend(cases)
    meta = dict(row_ids=row_ids, source_row_ids=source_ids, source_omissions=omissions,
        variants=args.variants, parameter_state='EMA', optimizer_step=5000, precision='FP32; TF32 off',
        preflight=args.preflight, core=str(args.core), seed=20261004, selection_seed=20261005,
        near_clean_conditions=['mask10_unclamped', 'mask10_first_clean', 'mask20_first_clean'],
        mask20_unclamped_reference='Reuse immutable core mask20/correct100 observations and arrays',
        source_target_rule='Masked content target with both adjacent content sources; target>=2 so input zero is never toggled',
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in ('owt/post_diagnostics.py', 'owt/post_diagnostic_canvases.py',
                                    'owt/reveal_sweep.py', 'owt/head_diagnostics.py')},
        max_source_sequence_evaluations=len(source_ids) * 3 * 2 * 2 * len(args.variants))
    atomic_write(args.output / 'protocol.json', json.dumps(meta, indent=2) + '\n')
    near_rows, source_rows, provenance = [], [], {}
    started = time.monotonic()
    predictions = args.output / 'head_predictions'
    predictions.mkdir(exist_ok=True)
    for variant in args.variants:
        model, provenance[variant] = load_ema_model(args.root / variant)
        if provenance[variant] != {k: core['provenance'][variant][k] for k in provenance[variant]}:
            raise RuntimeError('Core checkpoint identity changed')
        if model.time_conditioning or model.parameterization != 'subs' or model.backbone.attn_backend != 'sdpa':
            raise RuntimeError('Selected protocol requires time-unconditioned SUBS/SDPA')
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise RuntimeError('Expected FP32 parameters')
        model.backbone.force_fp32_eval = True
        model.to(device)
        # Store the immutable 20% reference alongside newly measured conditions.
        old_file = args.core / 'head_predictions' / f'{variant}_mask020_correct100.npz'
        with np.load(old_file) as saved:
            indices = [int(np.flatnonzero(saved['row_ids'] == i)[0]) for i in row_ids]
            np.savez_compressed(predictions / f'{variant}_mask20_unclamped.npz',
                **{key: saved[key][indices] for key in saved.files})
        for i in row_ids:
            r = reference[variant, i]
            parsed = parse_core_row(r)
            parsed.update(condition='mask20_unclamped', row_id=i, reused_core=True)
            near_rows.append(parsed)
        for ratio, anchored, condition in [(.1, False, 'mask10_unclamped'),
                                          (.1, True, 'mask10_first_clean'),
                                          (.2, True, 'mask20_first_clean')]:
            batches = []
            for start in range(0, len(row_ids), args.microbatch):
                ids = row_ids[start:start + args.microbatch]
                clean = np.asarray([dataset[i]['input_ids'] for i in ids], dtype=np.int64)
                pairs = [anchor_pair(x, ratio, i) for i, x in zip(ids, clean)]
                chosen = [p[int(anchored)] for p in pairs]
                canvas, masked, wrong = [np.stack([p[k] for p in chosen]) for k in range(3)]
                if ratio == .2:
                    for i, x, pair in zip(ids, clean, pairs):
                        assert digest(pair[0][0]) == reference[variant, i]['input_sha256']
                        assert digest(x) == reference[variant, i]['clean_sha256']
                arrays = {}
                values = score_batch(model, clean, canvas, masked, wrong, device,
                                     capture_heads=True, prediction_output=arrays)
                batches.append(arrays)
                for i, x, xt, mask, value in zip(ids, clean, canvas, masked, values):
                    near_rows.append(dict(variant=variant, row_id=i, mask_ratio=ratio,
                        correct_fraction=1., condition=condition, reused_core=False,
                        clean_sha256=digest(x), input_sha256=digest(xt), mask_sha256=digest(mask),
                        actual_masked_positions=int(mask.sum()), **value))
                atomic_write(args.output / 'progress.json', json.dumps(dict(variant=variant,
                    stage=condition, near_observations=len(near_rows), source_observations=len(source_rows),
                    elapsed_seconds=time.monotonic() - started)) + '\n')
            np.savez_compressed(predictions / f'{variant}_{condition}.npz', row_ids=np.asarray(row_ids),
                **{key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]})
            print(variant, condition, 'complete', flush=True)
        for start in range(0, len(all_cases), args.microbatch):
            cases = all_cases[start:start + args.microbatch]
            clean = np.asarray([dataset[c['row_id']]['input_ids'] for c in cases], dtype=np.int64)
            values = score_targets(model, clean, cases, device)
            for c, x, value in zip(cases, clean, values):
                source_rows.append(dict(variant=variant, **{k: v for k, v in c.items() if k != 'canvas'},
                    input_sha256=digest(c['canvas']), clean_sha256=digest(x), target_token=int(x[c['target_index']]), **value))
            atomic_write(args.output / 'progress.json', json.dumps(dict(variant=variant,
                stage='source_intervention', near_observations=len(near_rows), source_observations=len(source_rows),
                elapsed_seconds=time.monotonic() - started)) + '\n')
        del model
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    write_rows(args.output / 'near_clean_observations.csv', near_rows)
    write_rows(args.output / 'source_observations.csv', source_rows)
    artifact = dict(protocol=meta, provenance=provenance,
        source_summary=summarize_sources(source_rows, args.variants),
        elapsed_seconds=time.monotonic() - started,
        near_clean_report='Paired token arrays and row sufficient statistics saved for the combined error report',
        limitations=['Single training seed; paired row intervals do not measure seed uncertainty.',
                    'Frozen input intervention does not isolate a training component or source representation mechanism.',
                    'Preflight artifacts are collection checks, not full-cohort research results.'])
    atomic_write(args.output / 'summary.json', json.dumps(artifact, indent=2) + '\n')
    identity = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:12]
    record_event('post_diagnostics_' + identity, 'Post-5k paired diagnostic collection completed',
        f'Collected near-clean and first-input controls on {len(row_ids)} rows and same-source interventions on '
        f'{len(source_ids)} selected rows for {", ".join(args.variants)}; preflight={args.preflight}. '
        'Core 20%-mask reference evidence is reused and checked. Source-only changes preserve every other input '
        'and input zero remains clean. No training state was changed.', dict(output=str(args.output), protocol=meta))
    print('COMPLETE', args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs/owt/mdm-np-5k')
    parser.add_argument('--core', type=Path, default=ROOT / 'outputs/analysis/owt-reveal-sweep-5000')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--microbatch', type=int, default=4)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--examples', type=int, default=1024)
    parser.add_argument('--source-examples', type=int, default=128)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=VARIANTS)
    args = parser.parse_args()
    if min(args.examples, args.source_examples, args.microbatch) < 1 or args.examples > 1024 or args.source_examples > 128:
        parser.error('Positive sizes required; at most 1,024 cohort and 128 source rows')
    if args.variants not in [VARIANTS, ['mdm']] or (not args.preflight and (args.variants != VARIANTS or args.examples != 1024 or args.source_examples != 128)):
        parser.error('Final runs require both arms and complete cohorts; MDM-only is preflight only')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.post-diagnostics.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / 'summary.json').exists():
            parser.error('Completed evidence exists; use a fresh output directory')
        if args.device == 'cuda':
            with (args.root / '.queue.lock').open('a') as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                run(args)
        else:
            run(args)


if __name__ == '__main__':
    main()
