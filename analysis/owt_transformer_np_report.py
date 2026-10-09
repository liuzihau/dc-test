"""Native transformer-NP comparisons from verified saved predictions."""
import argparse
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np

from analysis.owt_error_report import bootstrap_weights, load_predictions, read_observations
from analysis.owt_frozen_readout import macro_report
from analysis.owt_source_pairing_report import compare_cell, pair_rows_match
from owt.source_pairing_diagnostics import prepare_references, verified_checkpoint, verify_saved_arrays, sha256
from owt.transformer_np_diagnostics import A, B, RUN_ROOT, verify_protocol, verify_completed
from owt.research import ROOT, atomic_write, read_json, read_csv

LINEAR = 'mdm_np_zero_init_masked_source'
REFERENCE_ROOT = ROOT / 'outputs/owt/mdm-np-5k'
LINEAR_COLLECTION = ROOT / 'outputs/analysis/mdm_np_zero_init_masked_source-frozen-5000'


def require_collection(folder, protocol_path, protocol, variant):
    artifact = read_json(folder / 'summary.json') or {}
    if (artifact.get('preflight') is not False
            or artifact.get('protocol_sha256') != sha256(protocol_path)
            or artifact.get('requested_variant') != variant or artifact.get('evaluated_variant') != variant
            or artifact.get('row_ids') != protocol['row_ids'] or artifact.get('optimizer_step') != 5000
            or artifact.get('parameter_state') != 'EMA' or artifact.get('observations') != 5120
            or artifact.get('reference_model_forwards') != 0 or artifact.get('new_sequence_evaluations') != 5120
            or artifact.get('precision') != protocol['precision']
            or artifact.get('native_transformer_features') is not True
            or artifact.get('block_fp32_assertions') is not True
            or [c.get('cell') for c in artifact.get('cells', [])] != protocol['conditions']):
        raise ValueError('Require all five real native FP32 final cells')
    verified_checkpoint(verify_completed(variant), artifact['provenance'])
    return artifact


def training_evidence(variant):
    validation = {}
    for name in ('mdm', 'mdm_np_zero_init', 'mdm_np_zero_init_low_weight', LINEAR, A, B):
        run = RUN_ROOT / name if name in (A, B) else REFERENCE_ROOT / name
        rows = read_csv(run / 'local_metrics/validation.csv')
        final = next((r for r in rows if r['optimizer_step'] == 5000), None)
        if final is not None:
            validation[name] = final
        elif name != B or variant == B:
            raise ValueError('Missing final validation: ' + name)
    trajectory = {name: read_csv(RUN_ROOT / variant / 'local_metrics' / name)
                  for name in ('train.csv', 'gradient_norms.csv', 'source_pairs.csv')}
    for name, rows in trajectory.items():
        if [r['optimizer_step'] for r in rows] != list(range(1, 5001)):
            raise ValueError('Incomplete native trajectory: ' + name)
        if any(not np.isfinite(v) for r in rows for v in r.values() if isinstance(v, (int, float))):
            raise ValueError('Nonfinite native trajectory')
    for row in trajectory['train.csv']:
        if abs(row['objective'] - row['main_elbo'] - .25 * (row['np_prev'] + row['np_next'])) > 1e-5:
            raise ValueError('Native objective does not reconstruct')
    for row in trajectory['gradient_norms.csv']:
        norm = sum(row[k]**2 for k in ('shared_trunk_l2', 'main_readout_l2',
                   'neighbor_readouts_l2', 'neighbor_processing_l2'))**.5
        if (not np.isclose(norm, row['joint_l2'], rtol=1e-10, atol=1e-10)
                or row['clip_limit'] != 1
                or not np.isclose(row['estimated_clip_multiplier'], min(1, 1/(norm+1e-6)), rtol=1e-10, atol=1e-10)):
            raise ValueError('Native norm/clipping does not reconstruct')
    visible = 0.
    for row in trajectory['source_pairs.csv']:
        for direction in ('prev', 'next'):
            for b in range(5):
                key = f'{direction}_maskbin{b}_'
                if (row[key+'selected'] != row[key+'masked_source']
                        or not 0 <= row[key+'selected_masked'] <= row[key+'selected']
                        or not np.isclose(row[key+'selected_weight_mass'], row[key+'masked_weight_mass'], rtol=2e-6, atol=1e-3)):
                    raise ValueError('Native pair count/mass differs')
                visible += row[key+'selected'] - row[key+'selected_masked']
    if (variant == A and visible != 0) or (variant == B and visible <= 0):
        raise ValueError('Unexpected native source placement')
    reference = REFERENCE_ROOT / LINEAR if variant == A else RUN_ROOT / A
    paired = pair_rows_match(read_csv(reference / 'local_metrics/source_pairs.csv'), trajectory['source_pairs.csv'])
    return dict(primary_validation=validation, visible_selected_pairs=visible,
                matched_supervision_audit=paired, matched_reference=reference.name,
                finite_updates=5000, joint_gradient_partition_verified=True,
                main_only_inference=True, limit='One training seed; task-specific attention and capacity change together.')


def arrays_for(folder, artifact, label, clean, ids, masked, wrong):
    entry = next(c for c in artifact['cells'] if c['cell'] == label)
    path = folder / entry['prediction_file']
    if sha256(path) != entry['predictions_sha256']:
        raise ValueError('Saved predictions changed')
    arrays = load_predictions(path)
    observations = {int(r['row_id']): r for r in read_observations(folder / 'observations.csv') if r['condition'] == label}
    verify_saved_arrays(arrays, clean, ids, masked, wrong, observations)
    return arrays


def plot(output, cells):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4))
    x = np.arange(5)
    for field, label, color in [('mdm_vs_trial_main_all', 'Trial minus MDM', '#245A91'),
            (LINEAR+'_vs_trial_main_all', 'Trial minus linear both-masked NP', '#888888')]:
        m = [c['comparisons'][field]['metrics']['ce_difference'] for c in cells]
        y = np.array([v['value'] for v in m]); ci = np.array([v['row_bootstrap_95'] for v in m])
        ax.errorbar(x, y, yerr=np.stack([y-ci[:, 0], ci[:, 1]-y]), fmt='o-', capsize=3, color=color, label=label)
    ax.axhline(0, color='#555555', ls='--', lw=.8)
    ax.set_xticks(x, ['80', '60', '40', '20', '10']); ax.set_xlabel('Masked fraction (%)')
    ax.set_ylabel('Main conditional CE difference (nats / target)'); ax.legend(fontsize=9); ax.grid(alpha=.2)
    fig.tight_layout()
    for suffix in ('png', 'pdf'): fig.savefig(output / ('native_main_comparison.'+suffix), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--variant', choices=(A, B), required=True)
    args = parser.parse_args(); protocol = read_json(args.protocol); reference = verify_protocol(protocol)
    collection = ROOT / protocol['outputs'][args.variant]['collection']
    output = ROOT / protocol['outputs'][args.variant]['report']
    artifact = require_collection(collection, args.protocol, protocol, args.variant)
    if output.exists(): raise ValueError('Refuse report overwrite')
    ids = np.asarray(protocol['row_ids']); scoring_ids = np.asarray(protocol['fixed_fusion']['scoring_row_ids'])
    indices = np.searchsorted(ids, scoring_ids)
    weights = bootstrap_weights(len(ids), protocol['bootstrap']['draws'], protocol['bootstrap']['full_row_seed'])
    scoring_weights = bootstrap_weights(len(indices), protocol['bootstrap']['draws'], protocol['bootstrap']['scoring_row_seed'])
    linear_artifact = read_json(LINEAR_COLLECTION / 'summary.json')
    other_folder = ROOT / protocol['outputs'][A]['collection']
    other_artifact = require_collection(other_folder, args.protocol, protocol, A) if args.variant == B else None
    training = training_evidence(args.variant)
    output.mkdir(parents=True); cells = []; macro = {}
    for entry in artifact['cells']:
        label = entry['cell']
        clean, refids, canvas, masked, wrong, saved = prepare_references(REFERENCE_ROOT, reference, label)
        if not np.array_equal(ids, refids): raise ValueError('Reference rows changed')
        trial = arrays_for(collection, artifact, label, clean, ids, masked, wrong)
        saved[LINEAR] = arrays_for(LINEAR_COLLECTION, linear_artifact, label, clean, ids, masked, wrong)
        other = arrays_for(other_folder, other_artifact, label, clean, ids, masked, wrong) if other_artifact else None
        comparisons, stats, fixed, frozen = compare_cell(saved, trial, weights, indices, scoring_weights,
            protocol['fixed_fusion']['lambda_weight'], other)
        np.savez_compressed(output / (label+'_row_statistics.npz'), row_ids=ids, **stats)
        for name, values in frozen.items():
            np.savez_compressed(output / (label+'_'+name+'_frozen_row_statistics.npz'), row_ids=scoring_ids, **values)
            macro.setdefault(name, []).append(values)
        cells.append(dict(cell=label, comparisons=comparisons, fixed_fusion=fixed))
    result = dict(protocol_sha256=sha256(args.protocol), preflight=False, variant=args.variant,
        row_ids=ids.tolist(), scoring_row_ids=scoring_ids.tolist(), cells=cells,
        fixed_lambda=protocol['fixed_fusion']['lambda_weight'], fitting_performed=False,
        fixed_fusion_macro={k: macro_report(v, scoring_weights) for k, v in macro.items()},
        paired_source_arm_available=other_artifact is not None, model_forward_passes=0,
        collection_sha256=sha256(collection / 'summary.json'), bootstrap=protocol['bootstrap'],
        limitations=protocol['limitations'], training_evidence=training,
        sign_convention='reference_vs_trial: trial minus reference; count_vs_masked: A minus B')
    plot(output, cells)
    atomic_write(output / 'summary.json', json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Five native cells and complete finite training trajectory verified.', flush=True)


if __name__ == '__main__':
    main()
