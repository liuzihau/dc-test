"""Source-placement comparisons from saved scores; no model calls or refitting."""
import argparse
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np

from analysis.owt_error_report import bootstrap_weights, pair_report, union_report, load_predictions, verify_pair
from analysis.owt_frozen_readout import row_statistics, summarize_rows, macro_report
from owt.source_pairing_diagnostics import (
    REFERENCES, POLICY, VARIANTS, verify_protocol, verify_trial_completed,
    prepare_references, sha256, verified_checkpoint,
)
from owt.research import ROOT, atomic_write, read_json, read_csv, record_event


def require_collection(folder, protocol_path, protocol, variant, preflight=False):
    artifact = read_json(folder/'summary.json') or {}
    ids = protocol['row_ids'][:1] if preflight else protocol['row_ids']
    if (artifact.get('preflight') is not preflight
            or artifact.get('protocol_sha256') != sha256(protocol_path)
            or artifact.get('requested_variant') != variant
            or artifact.get('evaluated_variant') != ('mdm_np_zero_init' if preflight else variant)
            or artifact.get('row_ids') != ids
            or artifact.get('optimizer_step') != 5000
            or artifact.get('parameter_state') != 'EMA'
            or artifact.get('observations') != 5*len(ids)
            or artifact.get('reference_model_forwards') != 0
            or artifact.get('new_sequence_evaluations') != (0 if preflight else 5*len(ids))
            or not preflight and artifact.get('precision') != protocol['precision']
            or [c.get('cell') for c in artifact.get('cells', [])] != protocol['conditions']):
        raise ValueError('Registered real final source collection is incomplete')
    if not preflight:
        verify_trial_completed(ROOT/protocol['run_root'], variant)
        verified_checkpoint(ROOT/protocol['run_root']/variant, artifact['provenance'])
    return artifact


def pair_rows_match(first, second):
    """Audit supervision amount across the complete two training trajectories."""
    if len(first) != 5000 or len(second) != 5000:
        raise ValueError('Two complete source-pair trajectories required')
    largest_mass_error = 0.
    for a, b in zip(first, second):
        if a['optimizer_step'] != b['optimizer_step']:
            raise ValueError('Source-pair update alignment differs')
        for direction in ('prev', 'next'):
            for maskbin in range(5):
                key = f'{direction}_maskbin{maskbin}_'
                for field in ('eligible', 'masked_source', 'selected'):
                    if a[key+field] != b[key+field]:
                        raise ValueError('Matched auxiliary count differs: '+key+field)
                for field in ('eligible_weight_mass', 'selected_weight_mass', 'masked_weight_mass'):
                    x, y = a[key+field], b[key+field]
                    largest_mass_error = max(largest_mass_error, abs(x-y))
                    if not np.isclose(x, y, rtol=2e-6, atol=1e-3):
                        raise ValueError('Matched weighted supervision mass differs: '+key+field)
    return dict(updates=5000,counts_exactly_match=True,
                weight_mass_matches_with_rounding=True,max_absolute_mass_difference=largest_mass_error,
                tolerance=dict(relative=2e-6,absolute=1e-3),
                limit='Counts and mass do not establish identical selected targets or gradient directions.')


def training_evidence(root, variant):
    validation = {}
    for name in REFERENCES+(variant,):
        final = next((r for r in read_csv(root/name/'local_metrics/validation.csv')
                      if r['optimizer_step'] == 5000), None)
        if final is None:
            raise ValueError('Missing finite final validation: '+name)
        validation[name] = final
    trajectory = {name: read_csv(root/variant/'local_metrics'/name)
                  for name in ('train.csv', 'gradient_norms.csv', 'source_pairs.csv')}
    for name, rows in trajectory.items():
        if [r['optimizer_step'] for r in rows] != list(range(1, 5001)):
            raise ValueError('Complete finite source trajectory required: '+name)
    for row in trajectory['train.csv']:
        if abs(row['objective']-row['main_elbo']-.25*(row['np_prev']+row['np_next'])) > 1e-5:
            raise ValueError('Source-arm objective does not reconstruct')
    for row in trajectory['gradient_norms.csv']:
        reconstructed = sum(row[k]**2 for k in ('shared_trunk_l2','main_readout_l2','neighbor_readouts_l2'))**.5
        if (not np.isclose(reconstructed,row['joint_l2'],rtol=1e-10,atol=1e-10)
                or row['clip_limit'] != 1
                or not np.isclose(row['estimated_clip_multiplier'],min(1,1/(row['joint_l2']+1e-6)),rtol=1e-10,atol=1e-10)):
            raise ValueError('Source-arm joint norm/clipping does not reconstruct')
    visible_selected = 0.
    bins = {}
    for direction in ('prev','next'):
        for b in range(5):
            key=f'{direction}_maskbin{b}_'
            totals={field:sum(r[key+field] for r in trajectory['source_pairs.csv'])
                    for field in ('eligible','masked_source','selected','selected_masked',
                                  'eligible_weight_mass','selected_weight_mass','masked_weight_mass')}
            for row in trajectory['source_pairs.csv']:
                if (row[key+'selected'] != row[key+'masked_source']
                        or not 0 <= row[key+'selected_masked'] <= row[key+'selected'] <= row[key+'eligible']
                        or POLICY[variant]=='masked_source' and row[key+'selected_masked'] != row[key+'selected']
                        or not np.isclose(row[key+'selected_weight_mass'],row[key+'masked_weight_mass'],rtol=2e-6,atol=1e-3)):
                    raise ValueError('Source policy count/mass logs differ from registered rule')
            bins[key[:-1]]=totals
            visible_selected += totals['selected']-totals['selected_masked']
    if POLICY[variant]=='matched_pair_count' and visible_selected <= 0:
        raise ValueError('Count control never included a visible source')
    other=next(name for name in VARIANTS if name != variant)
    paired=None
    if (root/other/'complete.json').exists():
        verify_trial_completed(root,other)
        other_rows=read_csv(root/other/'local_metrics/source_pairs.csv')
        paired=pair_rows_match(trajectory['source_pairs.csv'],other_rows)
        final=next((r for r in read_csv(root/other/'local_metrics/validation.csv')
                    if r['optimizer_step']==5000),None)
        if final is None:raise ValueError('Other source arm final validation missing')
        validation[other]=final
    g=trajectory['gradient_norms.csv']
    return dict(final_validation=validation,
        trial_minus_reference={name:validation[variant]['val_nll']-validation[name]['val_nll']
                               for name in validation if name!=variant},
        last128_training_means={k:float(np.mean([r[k] for r in trajectory['train.csv'][-128:]]))
                               for k in ('main_elbo','np_prev','np_next')},
        joint_norm_median=float(np.median([r['joint_l2'] for r in g])),
        clipping_fraction=float(np.mean([r['estimated_clip_multiplier']<1 for r in g])),
        source_bins=bins,selected_visible_source_count=visible_selected,
        matched_two_arm_supervision_audit=paired,
        limit='One-seed EMA validation points; training updates are dependent; joint norms do not decompose task-gradient conflict.')


def compare_cell(saved, trial, weights, indices, scoring_weights, fixed_lambda, other=None):
    clean, scored=trial['clean'],trial['scored']
    reports, stats, frozen={}, {}, {}
    def pair(name,first,second,head_a,head_b,population):
        verify_pair(first,second)
        report,values,_=pair_report(clean,first[head_a+'_prediction'],second[head_b+'_prediction'],
            first[head_a+'_true_logp'],second[head_b+'_true_logp'],population,weights)
        reports[name]=report
        stats.update({name+'__'+k:v for k,v in values.items()})
    masked=trial['input_masked']
    adjacent_visible=np.zeros_like(masked)
    adjacent_visible[:,1:]|=~masked[:,:-1]
    adjacent_visible[:,:-1]|=~masked[:,1:]
    for name,reference in saved.items():
        for label,selection in [('all',scored),('adjacent_visible',scored&adjacent_visible),
                                ('no_adjacent_visible',scored&~adjacent_visible)]:
            pair(name+'_vs_trial_main_'+label,reference,trial,'main','main',selection)
    common=scored&(trial['left_prediction']>=0)&(trial['right_prediction']>=0)
    for direction in ('left','right'):
        for reference in REFERENCES[1:]:
            if not np.array_equal(saved[reference][direction+'_source_state'],trial[direction+'_source_state']):
                raise ValueError('Matched native head eligibility differs')
        for group,state in [('all',None),('masked',0),('correct_revealed',1)]:
            selected=scored&(trial[direction+'_prediction']>=0)
            if state is not None:selected&=trial[direction+'_source_state']==state
            pair('trial_main_'+direction+'_'+group,trial,trial,'main',direction,selected)
            for reference in REFERENCES[1:]:
                pair(reference+'_vs_trial_'+direction+'_'+group,saved[reference],trial,direction,direction,selected)
        pair('common_trial_main_'+direction,trial,trial,'main',direction,common)
    for name,arrays in [('trial',trial),*[(k,saved[k]) for k in REFERENCES[1:]]]:
        result,values=union_report(clean,arrays,common,weights)
        reports[name+'_union']=result
        stats.update({name+'_union__'+k:v for k,v in values.items()})
        frozen[name]=row_statistics(saved['mdm'],arrays,indices,fixed_lambda)
    if other is not None:
        # For the count-arm report, negative means masked-source beats count.
        pair('count_vs_masked_main_all',trial,other,'main','main',scored)
        for direction in ('left','right'):
            if not np.array_equal(trial[direction+'_source_state'],other[direction+'_source_state']):
                raise ValueError('Other source-arm eligibility differs')
        result,values=union_report(clean,other,common,weights)
        reports['masked_arm_union']=result
        stats.update({'masked_arm_union__'+k:v for k,v in values.items()})
        frozen['masked_arm']=row_statistics(saved['mdm'],other,indices,fixed_lambda)
    fixed={name:summarize_rows(values,scoring_weights)[0] for name,values in frozen.items()}
    return reports,stats,fixed,frozen


def plot(output,cells,variant,preflight):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    x=np.arange(5)
    for field,label,color in [('mdm_vs_trial_main_all','Trial minus MDM','#245A91'),
                             ('mdm_np_zero_init_vs_trial_main_all','Trial minus standard NP','#B65328'),
                             ('mdm_np_zero_init_low_weight_vs_trial_main_all','Trial minus low weight','#666666')]:
        metrics=[c['comparisons'][field]['metrics']['ce_difference'] for c in cells]
        y=np.array([m['value'] for m in metrics]);ci=np.array([m['row_bootstrap_95'] for m in metrics])
        axes[0].errorbar(x,y,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),fmt='o-',capsize=3,label=label,color=color)
    for field,label in [('trial_union','Trial'),('mdm_np_zero_init_union','Standard NP'),
                        ('mdm_np_zero_init_low_weight_union','Low weight')]:
        axes[1].plot(x,[100*c['comparisons'][field]['metrics']['absolute_rescue']['value'] for c in cells],'o-',label=label)
    for ax in axes:
        ax.set_xticks(x,['80','60','40','20','10']);ax.set_xlabel('Masked fraction (%)')
        ax.axhline(0,color='#555555',ls='--',lw=.8);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Main CE difference (nats / masked target)')
    axes[1].set_ylabel('Either auxiliary rescue (% of common targets)')
    fig.suptitle(('PREFLIGHT REPLAY ONLY: ' if preflight else '')+POLICY[variant])
    fig.text(.5,.025,'One training seed; paired packed-row intervals on main CE. Rescue is not deployable routing.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.94))
    for suffix in ('png','pdf'):fig.savefig(output/('source_pairing_comparison.'+suffix),dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    parser.add_argument('--collection',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--preflight-replay',action='store_true')
    args=parser.parse_args();protocol=read_json(args.protocol);verify_protocol(protocol)
    collection=args.collection or ROOT/protocol['outputs'][args.variant]['collection']
    output=args.output or ROOT/protocol['outputs'][args.variant]['report']
    artifact=require_collection(collection,args.protocol,protocol,args.variant,args.preflight_replay)
    if output.exists():raise RuntimeError('Report output exists; refuse overwrite')
    ids=np.asarray(artifact['row_ids']);scoring_ids=ids if args.preflight_replay else np.asarray(protocol['fixed_fusion']['scoring_row_ids'])
    indices=np.searchsorted(ids,scoring_ids)
    if not np.array_equal(ids[indices],scoring_ids):raise ValueError('Original scoring rows missing')
    weights=bootstrap_weights(len(ids),protocol['bootstrap']['draws'],protocol['bootstrap']['full_row_seed'])
    scoring_weights=bootstrap_weights(len(scoring_ids),protocol['bootstrap']['draws'],protocol['bootstrap']['scoring_row_seed'])
    other_folder=ROOT/protocol['outputs'][VARIANTS[0]]['collection']
    other_artifact=None
    if not args.preflight_replay and args.variant==VARIANTS[1] and (other_folder/'summary.json').exists():
        other_artifact=require_collection(other_folder,args.protocol,protocol,VARIANTS[0])
    if not args.preflight_replay and args.variant==VARIANTS[1] and other_artifact is None:
        raise ValueError('Evaluate the masked-source arm before the matched-count report')
    output.mkdir(parents=True);cells=[];macro={}
    for entry in artifact['cells']:
        label=entry['cell'];path=collection/entry['prediction_file']
        if sha256(path)!=entry['predictions_sha256']:raise ValueError('Trial predictions changed')
        trial=load_predictions(path)
        _,_,_,masked,_,saved=prepare_references(ROOT/protocol['run_root'],protocol,label,ids if args.preflight_replay else None)
        if not np.array_equal(trial['input_masked'],masked):raise ValueError('Trial input mask changed')
        other=None
        if other_artifact:
            other_entry=next(c for c in other_artifact['cells'] if c['cell']==label)
            other_path=other_folder/other_entry['prediction_file']
            if sha256(other_path)!=other_entry['predictions_sha256']:raise ValueError('Masked-arm predictions changed')
            other=load_predictions(other_path)
            if not np.array_equal(other['input_masked'],masked):raise ValueError('Masked-arm input mask changed')
        comparisons,stats,fixed,frozen=compare_cell(saved,trial,weights,indices,scoring_weights,protocol['fixed_fusion']['lambda_weight'],other)
        np.savez_compressed(output/(label+'_row_statistics.npz'),row_ids=ids,**stats)
        for name,values in frozen.items():
            np.savez_compressed(output/(label+'_'+name+'_frozen_row_statistics.npz'),row_ids=scoring_ids,**values)
            macro.setdefault(name,[]).append(values)
        cells.append(dict(cell=label,comparisons=comparisons,fixed_fusion=fixed))
    result=dict(protocol_sha256=sha256(args.protocol),preflight=args.preflight_replay,variant=args.variant,
        row_ids=ids.tolist(),scoring_row_ids=scoring_ids.tolist(),cells=cells,
        fixed_lambda=protocol['fixed_fusion']['lambda_weight'],fitting_performed=False,
        fixed_fusion_macro={k:macro_report(v,scoring_weights) for k,v in macro.items()},
        paired_source_arm_available=other_artifact is not None,
        model_forward_passes=0,collection_sha256=sha256(collection/'summary.json'),
        bootstrap=protocol['bootstrap'],limitations=protocol['limitations'],
        training_evidence=None if args.preflight_replay else training_evidence(ROOT/protocol['run_root'],args.variant))
    plot(output,cells,args.variant,args.preflight_replay)
    atomic_write(output/'summary.json',json.dumps(result,indent=2,allow_nan=False)+'\n')
    if not args.preflight_replay:
        record_event('source_paired_report_'+args.variant+'_20261002','Source-policy paired report ready',
            'Five conditions compare native main/auxiliary scores against saved MDM, standard-zero and low-weight evidence. '
            'Original mixture weight/scoring rows unchanged; no fitting or model calls in reporting. '
            'Scientific review and the matched-count comparison remain required.',dict(variant=args.variant,report=str(output)))
    print('Source-policy report written; scientific review required.')


if __name__=='__main__':
    main()
