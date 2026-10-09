"""Frozen evaluation of registered source-policy arms; baseline files stay pinned."""
import argparse
import csv
import fcntl
import json
import os
import time
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np

from analysis.owt_error_report import load_predictions, verify_pair, read_observations
from owt.low_weight_diagnostics import (
    CONDITIONS, prepare_reference_cell, verify_saved_arrays, verified_checkpoint,
    verify_protocol as verify_low_protocol, sha256,
)
from owt.research import ROOT, atomic_write, read_json, record_event
from owt.reveal_sweep import digest, score_batch
from owt.source_pairing_schedule import VARIANTS, verify_authorization

POLICY = dict(zip(VARIANTS, ('masked_source', 'matched_pair_count')))
REFERENCES = ('mdm', 'mdm_np_zero_init', 'mdm_np_zero_init_low_weight')


def verify_protocol(protocol):
    if (protocol.get('variants') != list(VARIANTS)
            or protocol.get('conditions') != [c[0] for c in CONDITIONS]
            or protocol.get('row_ids') != list(range(1024))
            or protocol.get('corruption_seed') != 20261004
            or protocol.get('physical_gpus') != [2, 3]
            or protocol.get('optimizer_step') != 5000):
        raise ValueError('Registered source-arm diagnostic cohort changed')
    for group in ('source_sha256', 'evidence_sha256'):
        for filename, expected in protocol[group].items():
            if sha256(ROOT/filename) != expected:
                raise ValueError('Registered diagnostic dependency changed: '+filename)
    low = read_json(ROOT/protocol['low_weight_protocol'])
    verify_low_protocol(low)
    if (protocol['fixed_fusion'] != low['fixed_fusion']
            or protocol['bootstrap'] != low['bootstrap']
            or protocol['run_root'] != low['run_root']
            or protocol['gpu_lock'] != low['gpu_lock']
            or protocol['parameter_state'] != 'EMA'
            or protocol['precision'] != low['precision']
            or protocol['cpu_preflight_threads'] != 2):
        raise ValueError('Original scoring rows, fixed mixture or bootstrap changed')
    for variant in VARIANTS:
        verify_authorization(ROOT/protocol['source_protocol'],
                             ROOT/protocol['source_review'], variant)


def verify_trial_completed(root, variant):
    if variant not in VARIANTS:
        raise ValueError('Unregistered source-policy arm')
    run = root/variant
    completion = read_json(run/'complete.json') or {}
    if completion.get('optimizer_step') != 5000:
        raise RuntimeError('Source arm must complete exactly5000 before evaluation')
    original = read_json(root/'mdm_np_zero_init/contract.json')
    actual = read_json(run/'contract.json')
    if not actual or not original:
        raise RuntimeError('Matched source-policy contracts required')
    actual = json.loads(json.dumps(actual))
    actual['variant'] = original['variant']
    if (actual.pop('diagnostics', None) != 'accumulated_joint_preclip_l2_v1'
            or actual.pop('source_diagnostics', None) != 'per_direction_maskbin_pair_counts_and_weight_mass_v1'
            or actual['mechanisms']['np'].pop('source_policy', None) != POLICY[variant]
            or actual['mechanisms']['np'].pop('pair_selection_seed', None) != 271828
            or actual != original):
        raise RuntimeError('Source contract differs beyond registered policy and diagnostics')


def prepare_references(root, protocol, label, row_ids=None):
    low_protocol = read_json(ROOT/protocol['low_weight_protocol'])
    clean, ids, canvas, masked, wrong, arrays = prepare_reference_cell(
        root, low_protocol, label, row_ids)
    registered = next(c for c in protocol['low_weight_references'] if c['cell'] == label)
    folder = ROOT/protocol['low_weight_collection']
    full = load_predictions(ROOT/registered['file'])
    if sha256(ROOT/registered['file']) != registered['sha256']:
        raise ValueError('Saved lower-weight predictions changed')
    summary = read_json(folder/'summary.json')
    if summary.get('preflight') or summary.get('optimizer_step') != 5000:
        raise ValueError('Real final lower-weight reference required')
    verified_checkpoint(root/REFERENCES[2], summary['provenance'])
    indices = np.searchsorted(full['row_ids'], ids)
    arrays[REFERENCES[2]] = {k: v[indices] for k, v in full.items()}
    observations = {int(r['row_id']): r for r in read_observations(folder/'observations.csv')
                    if r['condition'] == label and int(r['row_id']) in set(ids.tolist())}
    if len(observations) != len(ids):
        raise ValueError('Lower-weight reference observations missing')
    verify_saved_arrays(arrays[REFERENCES[2]], clean, ids, masked, wrong, observations)
    for row_id, x, inp, mask in zip(ids, clean, canvas, masked):
        record = observations[int(row_id)]
        if (record['clean_sha256'] != digest(x) or record['input_sha256'] != digest(inp)
                or record['mask_sha256'] != digest(mask)):
            raise ValueError('Lower-weight reference input differs')
    for arm in REFERENCES[1:]:
        verify_pair(arrays['mdm'], arrays[arm])
    return clean, ids, canvas, masked, wrong, arrays


def write_cell(output, label, arrays, masked, rows):
    path = output/(label+'.npz')
    np.savez_compressed(path, **arrays, input_masked=masked)
    load_predictions(path)
    return dict(cell=label, prediction_file=path.name,
                predictions_sha256=sha256(path), rows=len(rows))


def collect(args, protocol):
    if args.output.exists():
        raise RuntimeError('Output exists; refuse duplicate or partial overwrite')
    if args.preflight_replay:
        model, provenance = None, None
        evaluated_variant = 'mdm_np_zero_init'
    else:
        verify_trial_completed(args.root, args.variant)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '2,3':
            raise RuntimeError('Final collection requires physical GPUs2,3')
        # Every dependency/input is checked before any tensor/model import.
        for label, _ in CONDITIONS:
            prepare_references(args.root, protocol, label)
        import torch
        from owt.checkpoint import load_ema_model
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if torch.cuda.device_count() != 2:
            raise RuntimeError('Only the authorized GPU pair may be visible')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, provenance = load_ema_model(args.root/args.variant)
        if (model.np_config.source_policy != POLICY[args.variant]
                or model.time_conditioning or model.parameterization != 'subs'
                or model.backbone.attn_backend != 'sdpa'
                or any(p.dtype != torch.float32 for p in model.parameters())):
            raise RuntimeError('Unexpected final source-model route or precision')
        # Source policy changes training loss only; identical parameter keys
        # allow the original strict EMA loader and native readout evaluation.
        model.backbone.force_fp32_eval = True
        model.to('cuda:0')
        evaluated_variant = args.variant
    args.output.mkdir(parents=True)
    atomic_write(args.output/'protocol.json', json.dumps(protocol, indent=2)+'\n')
    observations, cells = [], []
    started = time.monotonic()
    for label, ratio in CONDITIONS:
        cohort = protocol['row_ids'][:1] if args.preflight_replay else None
        clean, ids, canvas, masked, wrong, saved = prepare_references(
            args.root, protocol, label, cohort)
        cell_rows, batches = [], []
        if args.preflight_replay:
            arrays = saved['mdm_np_zero_init']
            for row_id in ids:
                # Replay checked scalars without evaluating the saved model.
                i = int(np.searchsorted(ids, row_id))
                selected = arrays['scored'][i]
                cell_rows.append(dict(variant=evaluated_variant, condition=label,
                    row_id=int(row_id), mask_ratio=ratio, correct_fraction=1.,
                    clean_sha256=digest(clean[i]), input_sha256=digest(canvas[i]),
                    mask_sha256=digest(masked[i]), masked_targets=int(selected.sum()),
                    masked_ce_sum=float(-arrays['main_true_logp'][i, selected].astype(float).sum()),
                    masked_correct_count=int(((arrays['main_prediction'][i] == clean[i]) & selected).sum())))
        else:
            for start in range(0, len(ids), args.microbatch):
                end = min(start+args.microbatch, len(ids))
                predictions = {}
                values = score_batch(model, clean[start:end], canvas[start:end],
                    masked[start:end], wrong[start:end], 'cuda:0', capture_heads=True,
                    prediction_output=predictions)
                batches.append(predictions)
                for i, value in zip(range(start, end), values):
                    cell_rows.append(dict(variant=evaluated_variant, condition=label,
                        row_id=int(ids[i]), mask_ratio=ratio, correct_fraction=1.,
                        clean_sha256=digest(clean[i]), input_sha256=digest(canvas[i]),
                        mask_sha256=digest(masked[i]), **value))
                atomic_write(args.output/'progress.json', json.dumps(dict(
                    condition=label, finished_observations=len(observations)+len(cell_rows),
                    elapsed_seconds=time.monotonic()-started), indent=2)+'\n')
            arrays = dict(row_ids=ids, **{k: np.concatenate([b[k] for b in batches])
                                        for k in batches[0]})
        verify_saved_arrays(arrays, clean, ids, masked, wrong,
                            {r['row_id']: r for r in cell_rows})
        verify_pair(saved['mdm'], arrays)
        for direction in ('left', 'right'):
            if not np.array_equal(arrays[direction+'_source_state'],
                                  saved['mdm_np_zero_init'][direction+'_source_state']):
                raise ValueError('Native source eligibility differs')
        cells.append(write_cell(args.output, label, arrays, masked, cell_rows))
        observations.extend(cell_rows)
        print(label, len(ids), 'rows verified', flush=True)
    with (args.output/'observations.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in observations for k in row}))
        writer.writeheader()
        writer.writerows(observations)
    result = dict(protocol_sha256=sha256(args.protocol), preflight=args.preflight_replay,
        preflight_kind='saved-reference replay, no model calls' if args.preflight_replay else None,
        requested_variant=args.variant, evaluated_variant=evaluated_variant,
        row_ids=ids.tolist(), optimizer_step=5000, parameter_state='EMA',
        precision='saved FP32 reference replay' if args.preflight_replay else protocol['precision'],
        provenance=provenance, observations=len(observations), cells=cells,
        reference_model_forwards=0, new_sequence_evaluations=0 if args.preflight_replay else len(observations),
        native_readout_policy='all original content-eligible sources; no evaluation source filtering',
        elapsed_seconds=time.monotonic()-started)
    atomic_write(args.output/'summary.json', json.dumps(result, indent=2)+'\n')
    if not args.preflight_replay:
        record_event('source_frozen_collection_'+args.variant+'_20261002',
            'Source-policy frozen predictions collected',
            'Five registered conditions evaluated only the completed source-policy EMA. '
            'Saved MDM, standard-zero and lower-weight inputs/scores checked and reused; '
            'native readouts retain all source states for matched evaluation. Scientific review pending.',
            dict(variant=args.variant, collection=str(args.output)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--microbatch', type=int, default=8)
    parser.add_argument('--preflight-replay', action='store_true')
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    verify_protocol(protocol)
    args.root = ROOT/protocol['run_root']
    if not 1 <= args.microbatch <= 8:
        parser.error('Microbatch must be1--8')
    if args.preflight_replay and args.output is None:
        parser.error('Saved-replay preflight needs an explicit fresh output')
    args.output = args.output or ROOT/protocol['outputs'][args.variant]['collection']
    if args.preflight_replay:
        collect(args, protocol)
    else:
        with (args.root/'.queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            collect(args, protocol)


if __name__ == '__main__':
    main()
