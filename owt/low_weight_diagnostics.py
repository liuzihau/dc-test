"""Five registered correct-context canvases for the selected low-weight trial.

Only the new checkpoint is evaluated. Saved MDM/standard-zero observations
are verified and reused. Torch is imported only inside the held GPU lock.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np

from analysis.owt_error_report import load_predictions, verify_pair, read_observations, verify_source_states
from owt.research import ROOT, atomic_write, read_json, record_event
from owt.reveal_corruption import make_canvas
from owt.reveal_sweep import digest, score_batch

VARIANT = 'mdm_np_zero_init_low_weight'
REFERENCE_VARIANTS = ('mdm', 'mdm_np_zero_init')
CONDITIONS = [(f'mask{n:03d}_correct100', n/100) for n in (80, 60, 40, 20)] + [('mask10_unclamped', .1)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_protocol(protocol):
    if (protocol.get('variant') != VARIANT or protocol.get('conditions') != [c[0] for c in CONDITIONS]
            or protocol.get('corruption_seed') != 20261004
            or protocol.get('row_ids') != list(range(1024))):
        raise ValueError('Registered low-weight cohort or corruption protocol changed')
    for field in ('source_sha256', 'decision_sha256'):
        for filename, expected in protocol[field].items():
            if sha256(ROOT/filename) != expected:
                raise ValueError('Registered source/decision changed: '+filename)


def verified_checkpoint(run, expected):
    checkpoint = run/'checkpoints/step-0005000.ckpt'
    stat = checkpoint.stat()
    actual = dict(checkpoint=str(checkpoint), checkpoint_size=stat.st_size,
        checkpoint_mtime_ns=stat.st_mtime_ns, config_sha256=sha256(run/'resolved_config.yaml'),
        parameter_state='EMA', optimizer_step=5000)
    for key, value in actual.items():
        if expected[key] != value:
            raise ValueError('Reference checkpoint identity changed: '+key)


def verify_completed_trial(root):
    completion = read_json(root/VARIANT/'complete.json')
    if not completion or completion.get('optimizer_step') != 5000:
        raise RuntimeError('Low-weight trial must complete exactly5000 before frozen collection')
    original = read_json(root/'mdm_np_zero_init/contract.json')
    low = read_json(root/VARIANT/'contract.json')
    if not original or not low:
        raise RuntimeError('Missing matched run contracts')
    low = json.loads(json.dumps(low))
    if low['mechanisms']['np']['weights'] != [.05, .05] or low.pop('diagnostics', None) != 'accumulated_joint_preclip_l2_v1':
        raise RuntimeError('Unexpected low-weight recipe or logging')
    low['variant'] = original['variant']
    low['mechanisms']['np']['weights'] = original['mechanisms']['np']['weights']
    if low != original:
        raise RuntimeError('Training contract differs beyond the selected weights and read-only logging')


def reference_canvases(clean, ids, ratio, seed, observations):
    """Reconstruct and hash-check every input, not just its masked population."""
    clean = np.asarray(clean, dtype=np.int64)
    if len(clean) != len(ids) or len(set(ids.tolist())) != len(ids):
        raise ValueError('Invalid row population')
    # Correctness=1 needs no replacement sources. A two-token dummy pool leaves
    # the independent mask RNG unchanged and its replacement samples unused.
    cases = [make_canvas(x, ratio, 1., seed, int(i), [0, 1], 50257, [50256]) for i, x in zip(ids, clean)]
    for row_id, x, case in zip(ids, clean, cases):
        record = observations[int(row_id)]
        for key, value in [('clean_sha256', digest(x)), ('input_sha256', digest(case[0])), ('mask_sha256', digest(case[1]))]:
            if record[key] != value:
                raise ValueError('Reference corruption hash differs: '+key)
    return np.stack([c[0] for c in cases]), np.stack([c[1] for c in cases]), np.stack([c[2] for c in cases])


def verify_saved_arrays(arrays, clean, ids, masked, wrong, observations):
    if not np.array_equal(arrays['row_ids'], ids) or not np.array_equal(arrays['clean'], clean):
        raise ValueError('Prediction clean tokens or row identities differ')
    scored = masked.copy(); scored[:, 0] = False
    if not np.array_equal(arrays['scored'], scored):
        raise ValueError('Saved scored target population differs')
    if 'left_prediction' in arrays:
        verify_source_states(arrays, masked, wrong)
    for index, row_id in enumerate(ids):
        record = observations[int(row_id)]
        if (int(float(record['masked_targets'])) != int(scored[index].sum())
                or int(float(record['masked_correct_count'])) != int(((arrays['main_prediction'][index] == clean[index]) & scored[index]).sum())
                or abs(float(record['masked_ce_sum']) + arrays['main_true_logp'][index, scored[index]].astype(float).sum()) > 1e-8):
            raise ValueError('Prediction scores do not reconstruct observed main statistics')


def prepare_reference_cell(root, protocol, label, row_ids=None):
    registered = next(c for c in protocol['reference_inputs'] if c['cell'] == label)
    arrays = {}
    for variant in REFERENCE_VARIANTS:
        path = ROOT/registered['files'][variant]
        if sha256(path) != registered['sha256'][variant]:
            raise ValueError('Registered reference predictions changed')
        arrays[variant] = load_predictions(path)
    verify_pair(arrays['mdm'], arrays['mdm_np_zero_init'])
    ids = arrays['mdm']['row_ids']
    if not np.array_equal(ids, np.asarray(protocol['row_ids'])):
        raise ValueError('Reference monitoring cohort changed')
    if row_ids is not None:
        indices = np.searchsorted(ids, np.asarray(row_ids))
        if not np.array_equal(ids[indices], row_ids):
            raise ValueError('Preflight rows missing from reference')
        arrays = {v: {k: a[indices] for k, a in values.items()} for v, values in arrays.items()}
        ids = np.asarray(row_ids)
    collection = ROOT/registered['collection']
    if sha256(collection/'summary.json') != registered['collection_sha256']:
        raise ValueError('Reference collection summary changed')
    if sha256(collection/registered['observations']) != registered['observations_sha256']:
        raise ValueError('Reference observation collection changed')
    summary = read_json(collection/'summary.json')
    for variant in REFERENCE_VARIANTS:
        verified_checkpoint(root/variant, summary['provenance'][variant])
    ratio = dict(CONDITIONS)[label]
    records = read_observations(collection/registered['observations'])
    rows = {v: {} for v in REFERENCE_VARIANTS}
    for row in records:
        if (row['variant'] in rows and int(row['row_id']) in ids and float(row['mask_ratio']) == ratio
                and float(row.get('correct_fraction', 1.)) == 1.
                and (row.get('condition', label) == label or label.startswith('mask0'))):
            key = int(row['row_id'])
            if key in rows[row['variant']]:
                raise ValueError('Duplicate reference row')
            rows[row['variant']][key] = row
    if any(set(r) != set(ids.tolist()) for r in rows.values()):
        raise ValueError('Reference observations missing registered rows')
    clean = arrays['mdm']['clean'].astype(np.int64)
    canvases, masked, wrong = reference_canvases(clean, ids, ratio, protocol['corruption_seed'], rows['mdm'])
    reference_canvases(clean, ids, ratio, protocol['corruption_seed'], rows['mdm_np_zero_init'])
    for variant in REFERENCE_VARIANTS:
        verify_saved_arrays(arrays[variant], clean, ids, masked, wrong, rows[variant])
    return clean, ids, canvases, masked, wrong, arrays


def collect(args, protocol):
    if args.output.exists():
        raise RuntimeError('Collection output exists; refuse duplicate/partial overwrite')
    if not args.preflight:
        verify_completed_trial(args.root)
    # Reference provenance and all inputs are verified before importing torch.
    reference = {label: prepare_reference_cell(args.root, protocol, label,
        protocol['row_ids'][:1] if args.preflight else None) for label, _ in CONDITIONS}
    if args.preflight:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif os.environ.get('CUDA_VISIBLE_DEVICES') != '2,3':
        raise RuntimeError('Final collection requires authorized CUDA_VISIBLE_DEVICES=2,3')
    import torch
    from owt.checkpoint import load_ema_model
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    if not args.preflight and torch.cuda.device_count() != 2:
        raise RuntimeError('Expected only the authorized GPU pair')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    variant = 'mdm_np_zero_init' if args.preflight else VARIANT
    device = 'cpu' if args.preflight else 'cuda:0'
    model, provenance = load_ema_model(args.root/variant)
    if model.time_conditioning or model.parameterization != 'subs' or model.backbone.attn_backend != 'sdpa':
        raise RuntimeError('Unexpected frozen model route')
    model.backbone.force_fp32_eval = True
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError('FP32 frozen parameters required')
    model.to(device)
    args.output.mkdir(parents=True)
    atomic_write(args.output/'protocol.json', json.dumps(protocol, indent=2)+'\n')
    started = time.monotonic(); observations = []; cells = []
    for label, ratio in CONDITIONS:
        clean, ids, canvases, masked, wrong, saved = reference[label]
        batches = []; cell_rows = []
        for start in range(0, len(ids), args.microbatch):
            stop = start+args.microbatch
            predictions = {}
            values = score_batch(model, clean[start:stop], canvases[start:stop], masked[start:stop], wrong[start:stop],
                device, capture_heads=True, prediction_output=predictions)
            batches.append(predictions)
            for index, value in zip(range(start, min(stop, len(ids))), values):
                cell_rows.append(dict(variant=variant, condition=label, row_id=int(ids[index]), mask_ratio=ratio,
                    correct_fraction=1., input_sha256=digest(canvases[index]), clean_sha256=digest(clean[index]),
                    mask_sha256=digest(masked[index]), **value))
            atomic_write(args.output/'progress.json', json.dumps(dict(condition=label,
                finished_observations=len(observations)+len(cell_rows), elapsed_seconds=time.monotonic()-started), indent=2)+'\n')
        prediction_file=args.output/f'{label}.npz'
        np.savez_compressed(prediction_file, row_ids=ids,
            **{k:np.concatenate([b[k] for b in batches]) for k in batches[0]})
        result=load_predictions(prediction_file)
        verify_saved_arrays(result, clean, ids, masked, wrong, {r['row_id']:r for r in cell_rows})
        verify_pair(saved['mdm'], result)
        if not np.array_equal(saved['mdm_np_zero_init']['left_source_state'], result['left_source_state']) or not np.array_equal(saved['mdm_np_zero_init']['right_source_state'], result['right_source_state']):
            raise ValueError('Native auxiliary eligibility differs from reference')
        observations.extend(cell_rows)
        cells.append(dict(cell=label, prediction_file=prediction_file.name, predictions_sha256=sha256(prediction_file), rows=len(ids)))
        print(label, len(ids), 'rows collected and verified', flush=True)
    with (args.output/'observations.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted({key for row in observations for key in row}))
        writer.writeheader(); writer.writerows(observations)
    artifact=dict(protocol=protocol, protocol_sha256=sha256(args.protocol), preflight=args.preflight,
        evaluated_variant=variant, row_ids=ids.tolist(), parameter_state='EMA', optimizer_step=5000,
        precision='FP32, backbone autocast disabled, TF32 off', device=device, microbatch=args.microbatch,
        provenance=provenance, observations=len(observations), cells=cells, elapsed_seconds=time.monotonic()-started,
        reference_model_forwards=5 if args.preflight else 0)
    atomic_write(args.output/'summary.json', json.dumps(artifact,indent=2)+'\n')
    if not args.preflight:
        record_event('low_weight_frozen_collection_complete_20261001','Low-weight frozen predictions collected',
            'Only the new EMA checkpoint was evaluated on the five registered 1024-row correct-context canvases. '
            'Original MDM and standard-zero evidence was checked and reused. Scientific comparison remains pending.',dict(output=str(args.output)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'outputs/research-notes/low_weight_diagnostic_protocol_20261001.json')
    parser.add_argument('--root',type=Path,default=ROOT/'outputs/owt/mdm-np-5k')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--microbatch',type=int,default=8)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args(); protocol=read_json(args.protocol)
    if not protocol or protocol['variant'] != VARIANT or protocol['conditions'] != [c[0] for c in CONDITIONS]:
        parser.error('Registered five-condition low-weight protocol required')
    verify_protocol(protocol)
    if args.preflight and args.output is None:
        parser.error('Preflight requires a fresh explicit output')
    args.output=args.output or ROOT/protocol['collection_output']
    if not 1 <= args.microbatch <= 8:
        parser.error('Microbatch must be1--8')
    if args.preflight:
        collect(args,protocol)
    else:
        with (args.root/'.queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            collect(args,protocol)


if __name__ == '__main__':
    main()
