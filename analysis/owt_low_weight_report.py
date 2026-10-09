"""Paired low-weight outcomes and fixed original fusion; no fitting/forwards."""
import argparse
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np

from analysis.owt_error_report import bootstrap_weights, pair_report, union_report, load_predictions, verify_pair
from analysis.owt_frozen_readout import row_statistics, summarize_rows, macro_report
from owt.low_weight_diagnostics import CONDITIONS, VARIANT, prepare_reference_cell, sha256, verify_protocol
from owt.research import ROOT, atomic_write, read_json, read_csv, record_event


def final_training_evidence(root):
    """Primary validation point outcomes and descriptive joint-gradient logs."""
    validation = {}
    for variant in ('mdm', 'mdm_np_zero_init', VARIANT):
        rows=read_csv(root/variant/'local_metrics/validation.csv')
        final=next((r for r in rows if r['optimizer_step']==5000),None)
        if final is None:
            raise ValueError('Missing finite final validation: '+variant)
        validation[variant]=final
    gradients=read_csv(root/VARIANT/'local_metrics/gradient_norms.csv')
    training=read_csv(root/VARIANT/'local_metrics/train.csv')
    if ([r['optimizer_step'] for r in gradients] != list(range(1,5001))
            or [r['optimizer_step'] for r in training] != list(range(1,5001))):
        raise ValueError('Complete fresh5000-step loss and joint-gradient logs required')
    for row in training:
        if abs(row['objective']-row['main_elbo']-.05*(row['np_prev']+row['np_next'])) > 1e-5:
            raise ValueError('Logged low-weight objective does not reconstruct')
    for row in gradients:
        norm=sum(row[k]**2 for k in ('shared_trunk_l2','main_readout_l2','neighbor_readouts_l2'))**.5
        if not np.isclose(norm,row['joint_l2'],rtol=1e-10,atol=1e-10):
            raise ValueError('Joint norm partition differs')
    norms={}
    for group in ('all','first128','last128'):
        rows=gradients if group=='all' else gradients[:128] if group=='first128' else gradients[-128:]
        norms[group]=dict(updates=len(rows),clipped_fraction=float(np.mean([r['estimated_clip_multiplier']<1 for r in rows])),
            quantile_probabilities=[.05,.5,.95],
            quantiles={key:np.quantile([r[key] for r in rows],[.05,.5,.95]).tolist()
                for key in ('joint_l2','shared_trunk_l2','main_readout_l2','neighbor_readouts_l2','estimated_clip_multiplier')})
    low=validation[VARIANT]['val_nll']
    return dict(final_validation=validation,low_minus_mdm_validation_nll=low-validation['mdm']['val_nll'],
        low_minus_standard_validation_nll=low-validation['mdm_np_zero_init']['val_nll'],joint_gradient_summary=norms,
        validation_uncertainty='Final monitoring-subset points; no seed or paired validation-row uncertainty.',
        gradient_limit='Accumulated joint group norms, not separate main/auxiliary gradients; references lack matching trajectory logs.')


def compare_cell(mdm, standard, low, weights, scoring_indices, scoring_weights, fixed_lambda):
    verify_pair(mdm, standard); verify_pair(mdm, low)
    clean, selected = low['clean'], low['scored']
    reports = {}; stats = {}
    def pair(name, first, second, first_head, second_head, population):
        report, values, _ = pair_report(clean, first[first_head+'_prediction'], second[second_head+'_prediction'],
            first[first_head+'_true_logp'], second[second_head+'_true_logp'], population, weights)
        reports[name] = report
        stats.update({name+'__'+key: value for key, value in values.items()})
    pair('mdm_vs_low_main', mdm, low, 'main', 'main', selected)
    pair('standard_vs_low_main', standard, low, 'main', 'main', selected)
    pair('mdm_vs_standard_main', mdm, standard, 'main', 'main', selected)
    common = selected & (low['left_prediction'] >= 0) & (low['right_prediction'] >= 0)
    if (not np.array_equal(low['left_source_state'], standard['left_source_state'])
            or not np.array_equal(low['right_source_state'], standard['right_source_state'])):
        raise ValueError('Common native target eligibility differs')
    for direction in ('left', 'right'):
        for group, state in [('all', None), ('masked', 0), ('correct_revealed', 1)]:
            choose = selected & (low[direction+'_prediction'] >= 0)
            if state is not None:
                choose &= low[direction+'_source_state'] == state
            pair('low_main_'+direction+'_'+group, low, low, 'main', direction, choose)
            pair('standard_vs_low_'+direction+'_'+group, standard, low, direction, direction, choose)
        pair('common_low_main_'+direction, low, low, 'main', direction, common)
    pair('common_mdm_vs_low_main', mdm, low, 'main', 'main', common)
    pair('common_standard_vs_low_main', standard, low, 'main', 'main', common)
    for label, values in [('low', low), ('standard', standard)]:
        report, summary = union_report(clean, values, common, weights)
        reports[label+'_union'] = report
        stats.update({label+'_union__'+k:v for k,v in summary.items()})
    fixed = {}
    frozen_stats = {}
    for name, values in [('low', low), ('standard', standard)]:
        rows = row_statistics(mdm, values, scoring_indices, fixed_lambda)
        result, _ = summarize_rows(rows, scoring_weights)
        fixed[name] = result
        frozen_stats[name] = rows
    if not np.array_equal(frozen_stats['low']['targets'], frozen_stats['standard']['targets']):
        raise ValueError('Fixed mixture comparison uses different target population')
    # Pair the two mixtures directly, keeping the same scoring row draws.
    comparison = dict(targets=frozen_stats['low']['targets'],
        low_minus_standard_mixture_ce_sum=frozen_stats['low']['mixture_ce_sum']-frozen_stats['standard']['mixture_ce_sum'])
    fixed['low_vs_standard_mixture'], _ = summarize_rows(comparison, scoring_weights)
    return reports, stats, fixed, frozen_stats


def plot(output, cells, preflight):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5))
    x = np.arange(5)
    def line(ax, metrics, label, color, factor=1):
        values = np.asarray([m['value'] for m in metrics])*factor
        ci = np.asarray([m['row_bootstrap_95'] for m in metrics])*factor
        ax.errorbar(x,values,yerr=np.stack([values-ci[:,0],ci[:,1]-values]),fmt='o-',capsize=3,label=label,color=color)
    for field, label, color in [('mdm_vs_standard_main','Standard zero','#c44e52'),('mdm_vs_low_main','Low weight','#2166ac')]:
        line(axes[0],[c['comparisons'][field]['metrics']['ce_difference'] for c in cells],label,color)
    line(axes[1],[c['comparisons']['standard_vs_low_main']['metrics']['ce_difference'] for c in cells],
         'Low minus standard','#2166ac')
    for variant, color in [('standard','#c44e52'),('low','#2166ac')]:
        line(axes[2],[c['comparisons'][variant+'_union']['metrics']['absolute_rescue'] for c in cells],variant,color,100)
    for ax, title, units in zip(axes,['Main CE minus MDM','Main CE change from standard zero','Either auxiliary rescues main'],
        ['Nats / original masked target','Nats / original masked target','% of common content targets']):
        ax.set_xticks(x,['80','60','40','20','10']); ax.set_xlabel('Masked fraction (%)')
        ax.set_title(title,fontsize=10); ax.set_ylabel(units); ax.axhline(0,color='#555555',ls='--',lw=.8)
        ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle(('PREFLIGHT ONLY — ' if preflight else '')+'Lower auxiliary weight: matched correct-context diagnostics',fontsize=13)
    fig.text(.5,.035,'Whole packed-row 95% intervals; one training seed. Frozen EMA/FP32.\nMain comparisons use all original masked targets; rescue uses the common native content-target population.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.07,right=.985,top=.81,bottom=.24,wspace=.5)
    for suffix in ('pdf','png'): fig.savefig(output/('low_weight_comparison.'+suffix),dpi=180)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'outputs/research-notes/low_weight_diagnostic_protocol_20261001.json')
    parser.add_argument('--collection',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args(); protocol=read_json(args.protocol)
    if not protocol: parser.error('Missing registered protocol')
    verify_protocol(protocol)
    collection=args.collection or ROOT/protocol['collection_output']
    output=args.output or ROOT/protocol['report_output']
    artifact=read_json(collection/'summary.json')
    if not artifact or artifact['preflight'] != args.preflight or artifact['protocol_sha256'] != sha256(args.protocol):
        parser.error('Complete collection with matching protocol/preflight required')
    expected_variant='mdm_np_zero_init' if args.preflight else VARIANT
    ids=np.asarray(artifact['row_ids'])
    if (artifact['evaluated_variant'] != expected_variant or len(artifact['cells']) != 5
            or not args.preflight and not np.array_equal(ids,np.asarray(protocol['row_ids']))):
        parser.error('Unexpected collected model or cohort')
    if output.exists(): parser.error('Report output exists; refuse overwriting completed or partial evidence')
    output.mkdir(parents=True)
    weights=bootstrap_weights(len(ids),protocol['bootstrap']['draws'],protocol['bootstrap']['full_row_seed'])
    scoring_ids=ids if args.preflight else np.asarray(protocol['fixed_fusion']['scoring_row_ids'])
    scoring_indices=np.searchsorted(ids,scoring_ids)
    if not np.array_equal(ids[scoring_indices],scoring_ids): parser.error('Scoring rows missing')
    scoring_weights=bootstrap_weights(len(scoring_ids),protocol['bootstrap']['draws'],protocol['bootstrap']['scoring_row_seed'])
    cells=[]; frozen={name:[] for name in ('low','standard')}
    for label,_ in CONDITIONS:
        entry=next(c for c in artifact['cells'] if c['cell']==label)
        path=collection/entry['prediction_file']
        if sha256(path)!=entry['predictions_sha256']: raise ValueError('Collected predictions changed')
        low=load_predictions(path)
        _,_,_,_,_,saved=prepare_reference_cell(ROOT/protocol['run_root'],protocol,label,ids if args.preflight else None)
        comparisons,stats,fixed,frozen_stats=compare_cell(saved['mdm'],saved['mdm_np_zero_init'],low,
            weights,scoring_indices,scoring_weights,protocol['fixed_fusion']['lambda_weight'])
        np.savez_compressed(output/(label+'_row_statistics.npz'),row_ids=ids,**stats)
        for name in ('low','standard'):
            np.savez_compressed(output/(label+'_'+name+'_frozen_row_statistics.npz'),row_ids=scoring_ids,**frozen_stats[name])
            frozen[name].append(frozen_stats[name])
        cells.append(dict(cell=label,comparisons=comparisons,fixed_fusion=fixed))
    macro={name:macro_report(stats,scoring_weights) for name,stats in frozen.items()}
    mixture_change=[dict(targets=a['targets'],low_minus_standard_mixture_ce_sum=a['mixture_ce_sum']-b['mixture_ce_sum'])
                    for a,b in zip(frozen['low'],frozen['standard'])]
    macro['low_vs_standard_mixture']=macro_report(mixture_change,scoring_weights)
    summary=dict(protocol=str(args.protocol.relative_to(ROOT)),protocol_sha256=sha256(args.protocol),preflight=args.preflight,
        row_ids=ids.tolist(),scoring_row_ids=scoring_ids.tolist(),cells=cells,fixed_fusion_macro=macro,
        fixed_lambda=protocol['fixed_fusion']['lambda_weight'],fitting_performed=False,model_forward_passes=0,
        bootstrap=protocol['bootstrap'],collection_sha256=sha256(collection/'summary.json'),limitations=protocol['limitations'],
        training_evidence=None if args.preflight else final_training_evidence(ROOT/protocol['run_root']))
    plot(output,cells,args.preflight)
    atomic_write(output/'summary.json',json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if not args.preflight:
        record_event('low_weight_paired_reports_complete_20261001','Low-weight paired report ready for scientific review',
            'Five matched conditions compare the new main and native auxiliary predictions with saved MDM/standard-zero evidence. '
            'The original fusion weight is fixed on the original512 scoring rows; no fitting or model forward was performed in reporting.',dict(output=str(output)))
    print('Five matched condition reports and fixed-fusion macros written; scientific review still required.')


if __name__ == '__main__':
    main()
