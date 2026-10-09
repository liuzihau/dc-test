"""Frozen main-head evaluation over exact masking and revealed-token reliability.

This conditional diagnostic is not a likelihood bound or a generation benchmark.
GPU evaluation holds the training queue lock and uses only physical GPUs 2/3.
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
from owt.research import ROOT, LABELS, atomic_write, read_json, record_event
from owt.reveal_corruption import make_canvas, nearest_different_sources

MASK_RATIOS=[1.,.8,.6,.4,.2]
CORRECT_FRACTIONS=[1.,.8,.6]
COSINE_GROUPS=['masked','correct_revealed','wrong_revealed']


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def exposed_target_mask(masked,wrong,sources):
    exposed=np.zeros_like(masked)
    for k in range(len(masked)):
        copied=sources[k][wrong[k]]
        if (copied<0).any():
            raise ValueError('An incorrect reveal has no valid clean-token source')
        exposed[k,copied[masked[k,copied]]]=True
    return exposed


def score_batch(model,clean,canvases,masked,wrong,device,exposed=None,capture_cosines=False,reference_exposures=None,capture_heads=False,prediction_output=None):
    import torch
    truth=torch.as_tensor(clean,dtype=torch.long,device=device)
    inputs=torch.as_tensor(canvases,dtype=torch.long,device=device)
    scored=masked.copy();scored[:,0]=False  # Original BD3 excludes the first target.
    selected=torch.as_tensor(scored,device=device)
    if not selected.sum(1).all():
        raise ValueError('Every evaluated row must contain a scored masked target')
    captured=[];handles=[];head_features={}
    if capture_cosines:
        def hook(module,args,output):
            if getattr(model.backbone, 'force_fp32_eval', False):
                if args[0].dtype != torch.float32 or output.dtype != torch.float32 or torch.is_autocast_enabled('cuda'):
                    raise RuntimeError('The frozen FP32 diagnostic entered a mixed-precision block')
            captured.append(torch.nn.functional.cosine_similarity(
                args[0].detach().float(),output.detach().float(),dim=-1).clamp(-1,1))
        handles=[block.register_forward_hook(hook) for block in model.backbone.blocks]
    if capture_heads and getattr(getattr(model,'np_config',None),'enabled',False):
        handles.append(model.backbone.output_layer.linear.register_forward_pre_hook(
            lambda module,args:head_features.update(hidden=args[0].detach())))
    try:
        with torch.inference_mode():
            logp=model(inputs,torch.zeros((len(inputs),1),device=device))
            if getattr(getattr(model, 'backbone', None), 'force_fp32_eval', False) and logp.dtype != torch.float32:
                raise RuntimeError('The frozen FP32 diagnostic produced lower-precision log probabilities')
    finally:
        for handle in handles:
            handle.remove()
    with torch.inference_mode():
        nll=-logp.gather(-1,truth[...,None]).squeeze(-1)
        correct=logp.argmax(-1).eq(truth)
        adjacent=np.zeros_like(wrong)
        adjacent[:,1:] |= wrong[:,:-1]
        adjacent[:,:-1] |= wrong[:,1:]
        near=torch.as_tensor(adjacent&scored,device=device)
        copied=torch.as_tensor(exposed&scored if exposed is not None else np.zeros_like(scored),device=device)
        totals=torch.stack([torch.where(selected,nll,0).double().sum(1),
            (correct&selected).sum(1),selected.sum(1),
            torch.where(near,nll,0).double().sum(1),near.sum(1),
            torch.where(copied,nll,0).double().sum(1),copied.sum(1)],dim=1).cpu().numpy()
    if not np.isfinite(totals).all():
        raise FloatingPointError('Nonfinite masked-target metrics')
    rows=[dict(masked_ce_sum=float(v[0]),masked_correct_count=int(v[1]),masked_targets=int(v[2]),
                 adjacent_wrong_ce_sum=float(v[3]),adjacent_wrong_targets=int(v[4]),
                 copied_target_ce_sum=float(v[5]),copied_targets=int(v[6])) for v in totals]
    copied_positions=(exposed&scored) if exposed is not None else np.zeros_like(scored)
    for k,row in enumerate(rows):
        row['copied_target_mask_sha256']=digest(copied_positions[k])
    # At c=1, score the exact target groups that later c<1 canvases will expose.
    # Reuse this forward pass; do not compare different target populations.
    for reliability,positions in (reference_exposures or {}).items():
        membership=torch.as_tensor(positions&scored,device=device)
        with torch.inference_mode():
            sums=torch.where(membership,nll,0).double().sum(1).cpu().numpy()
        tag=f'c{round(reliability*100):03d}'
        for k,row in enumerate(rows):
            row[f'reference_copied_ce_sum_{tag}']=float(sums[k])
            row[f'reference_copied_targets_{tag}']=int((positions[k]&scored[k]).sum())
            row[f'reference_copied_mask_sha256_{tag}']=digest(positions[k]&scored[k])
    if capture_cosines:
        assert len(captured)==len(model.backbone.blocks)
        cosines=torch.stack(captured,dim=1)
        content=clean!=50256;content[:,0]=False
        groups={'masked':masked&content,'correct_revealed':~masked&~wrong&content,
                'wrong_revealed':wrong&content}
        for group,positions in groups.items():
            membership=torch.as_tensor(positions,device=device)
            sums=torch.where(membership[:,None,:],cosines,0).double().sum(-1).cpu().numpy()
            for k,row in enumerate(rows):
                row[f'cosine_{group}_count']=int(positions[k].sum())
                for layer,value in enumerate(sums[k],start=1):
                    row[f'cosine_{group}_layer{layer:02d}_sum']=float(value)
    if capture_heads:
        from owt.head_diagnostics import collect
        with torch.inference_mode():
            predictions=collect(model,logp,head_features.get('hidden'),clean,masked,wrong,rows)
        if prediction_output is not None:
            prediction_output.update(predictions)
    return rows


def summarize_cosines(rows,variants,layers):
    summary=[]
    for variant in variants:
        for ratio in MASK_RATIOS:
            for reliability in CORRECT_FRACTIONS:
                selected=[r for r in rows if (r['variant'],r['mask_ratio'],r['correct_fraction'])
                          ==(variant,ratio,reliability)]
                for group in COSINE_GROUPS:
                    eligible=[r for r in selected if r[f'cosine_{group}_count']>0]
                    n=len(eligible)
                    rng=np.random.default_rng(20261005)
                    draws=rng.integers(0,n,size=(2000,n)) if n else None
                    for layer in range(1,layers+1):
                        values=np.array([r[f'cosine_{group}_layer{layer:02d}_sum']/r[f'cosine_{group}_count']
                                         for r in eligible])
                        mean=float(values.mean()) if n else None
                        interval=[float(v) for v in np.quantile(values[draws].mean(1),[.025,.975])] if n else None
                        summary.append(dict(variant=variant,mask_ratio=ratio,correct_fraction=reliability,
                            token_group=group,layer=layer,mean_cosine=mean,row_bootstrap_95=interval,
                            contributing_rows=n,tokens=sum(r[f'cosine_{group}_count'] for r in eligible)))
    return summary


def summarize_exposure(paired,reliable,base,baseline,ids,counts,draws,reliability):
    tag=f'c{round(reliability*100):03d}'
    key=f'reference_copied_ce_sum_{tag}'
    if reliability!=1. and not all(key in reliable[i] and key in baseline[i] for i in ids):
        return None  # Earlier immutable pilots did not collect matched references.
    copied_counts=np.array([paired[i]['copied_targets'] for i in ids],dtype=float)
    assert np.array_equal(copied_counts,[base[i]['copied_targets'] for i in ids])
    if reliability==1.:
        assert not copied_counts.any()
        clean_copied=base_clean_copied=np.zeros(len(ids))
    else:
        for i in ids:
            assert reliable[i][f'reference_copied_targets_{tag}']==paired[i]['copied_targets']
            assert baseline[i][f'reference_copied_targets_{tag}']==base[i]['copied_targets']
            assert reliable[i][f'reference_copied_mask_sha256_{tag}']==paired[i]['copied_target_mask_sha256']
            assert baseline[i][f'reference_copied_mask_sha256_{tag}']==base[i]['copied_target_mask_sha256']
        clean_copied=np.array([reliable[i][key] for i in ids])
        base_clean_copied=np.array([baseline[i][key] for i in ids])
    copied_sums=np.array([paired[i]['copied_target_ce_sum'] for i in ids])
    base_copied=np.array([base[i]['copied_target_ce_sum'] for i in ids])
    sums=np.array([paired[i]['masked_ce_sum'] for i in ids])
    clean=np.array([reliable[i]['masked_ce_sum'] for i in ids])
    base_sums=np.array([base[i]['masked_ce_sum'] for i in ids])
    base_clean=np.array([baseline[i]['masked_ce_sum'] for i in ids])
    result={}
    groups=[('copied',copied_counts,copied_sums,clean_copied,base_copied,base_clean_copied),
            ('not_copied',counts-copied_counts,sums-copied_sums,clean-clean_copied,
             base_sums-base_copied,base_clean-base_clean_copied)]
    for name,population,wrong,correct,mdm_wrong,mdm_correct in groups:
        n=population.sum()
        item=dict(masked_targets=int(n),fraction_of_masked_targets=float(n/counts.sum()),
                  contributing_rows=int((population>0).sum()),
                  corrupted_ce=float(wrong.sum()/n) if n else None,
                  correct_context_ce_on_same_targets=float(correct.sum()/n) if n else None)
        denominator=population[draws].sum(1)
        valid=denominator>0
        item['bootstrap_draws_with_targets']=int(valid.sum())
        for metric,values in [('delta_from_mdm',wrong-mdm_wrong),
            ('degradation_from_correct_reveals',wrong-correct),
            ('excess_reliability_penalty_vs_mdm',(wrong-correct)-(mdm_wrong-mdm_correct))]:
            item[metric]=float(values.sum()/n) if n else None
            sampled=values[draws].sum(1)[valid]/denominator[valid]
            item[metric+'_row_bootstrap_95']=([float(v) for v in np.quantile(sampled,[.025,.975])]
                                             if len(sampled) else None)
        result[name]=item
    # The exact target partition must reconstruct the primary degradation.
    pooled=sum(g['fraction_of_masked_targets']*g['degradation_from_correct_reveals']
               for g in result.values() if g['masked_targets'])
    assert np.isclose(pooled,(sums-clean).sum()/counts.sum(),rtol=1e-10,atol=1e-10)
    return result


def summarize(rows,variants,bootstraps=2000):
    lookup={}
    for row in rows:
        key=(row['variant'],row['mask_ratio'],row['correct_fraction'])
        population=lookup.setdefault(key,{})
        if row['row_id'] in population:
            raise ValueError('Duplicate diagnostic observation')
        population[row['row_id']]=row
    summary=[]
    for ratio in MASK_RATIOS:
        baseline=lookup[('mdm',ratio,1.)]
        ids=sorted(baseline)
        counts=np.array([baseline[i]['masked_targets'] for i in ids],dtype=float)
        rng=np.random.default_rng(20261005)
        draws=rng.integers(0,len(ids),size=(bootstraps,len(ids)))
        denominators=counts[draws].sum(1)
        for reliability in CORRECT_FRACTIONS:
            base=lookup[('mdm',ratio,reliability)]
            assert set(base)==set(ids)
            base_sum=np.array([base[i]['masked_ce_sum'] for i in ids])
            base_clean=np.array([baseline[i]['masked_ce_sum'] for i in ids])
            for variant in variants:
                paired=lookup[(variant,ratio,reliability)]
                reliable=lookup[(variant,ratio,1.)]
                assert set(paired)==set(reliable)==set(ids)
                for i in ids:
                    assert paired[i]['input_sha256']==base[i]['input_sha256']
                    assert paired[i]['clean_sha256']==base[i]['clean_sha256']
                    assert paired[i]['masked_targets']==base[i]['masked_targets']==baseline[i]['masked_targets']
                    assert paired[i]['mask_sha256']==reliable[i]['mask_sha256']==baseline[i]['mask_sha256']
                    if 'copied_target_mask_sha256' in paired[i]:
                        assert paired[i]['copied_target_mask_sha256']==base[i]['copied_target_mask_sha256']
                    if ratio==1.:
                        assert paired[i]['input_sha256']==reliable[i]['input_sha256']
                        assert paired[i]['masked_ce_sum']==reliable[i]['masked_ce_sum']
                sums=np.array([paired[i]['masked_ce_sum'] for i in ids])
                clean_sums=np.array([reliable[i]['masked_ce_sum'] for i in ids])
                difference=sums-base_sum
                degradation=sums-clean_sums
                interaction=degradation-(base_sum-base_clean)
                total=counts.sum()
                item=dict(variant=variant,mask_ratio=ratio,correct_fraction=reliability,rows=len(ids),
                    masked_targets=int(total),masked_ce=float(sums.sum()/total),
                    masked_accuracy=sum(paired[i]['masked_correct_count'] for i in ids)/total,
                    delta_from_mdm=float(difference.sum()/total),
                    degradation_from_correct_reveals=float(degradation.sum()/total),
                    excess_reliability_penalty_vs_mdm=float(interaction.sum()/total))
                for name,values in [('delta',difference),('degradation',degradation),('interaction',interaction)]:
                    interval=np.quantile(values[draws].sum(1)/denominators,[.025,.975])
                    item[name+'_row_bootstrap_95']=[float(v) for v in interval]
                near_count=sum(paired[i]['adjacent_wrong_targets'] for i in ids)
                item['adjacent_wrong_targets']=near_count
                item['adjacent_wrong_ce']=(sum(paired[i]['adjacent_wrong_ce_sum'] for i in ids)/near_count
                                           if near_count else None)
                copied_count=sum(paired[i]['copied_targets'] for i in ids)
                item['copied_targets']=copied_count
                item['copied_target_ce']=(sum(paired[i]['copied_target_ce_sum'] for i in ids)/copied_count
                                          if copied_count else None)
                item['unexposed_target_ce']=((sums.sum()-sum(paired[i]['copied_target_ce_sum'] for i in ids))
                                             /(total-copied_count) if total>copied_count else None)
                item['matched_exposure_degradation']=summarize_exposure(
                    paired,reliable,base,baseline,ids,counts,draws,reliability)
                summary.append(item)
    return summary


def run(args):
    if args.device=='cpu':
        os.environ['CUDA_VISIBLE_DEVICES']=''
    elif os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':
        raise RuntimeError('GPU evaluation requires physical CUDA_VISIBLE_DEVICES=2,3')
    # No tensor libraries or CUDA calls are made until the serialized GPU lock is held.
    import torch
    from datasets import load_from_disk
    from owt.checkpoint import load_ema_model
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    if args.device=='cuda' and torch.cuda.device_count()!=2:
        raise RuntimeError('Expected exactly the two authorized GPUs')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device='cuda:0' if args.device=='cuda' else 'cpu'
    variants=args.variants
    contracts={name:read_json(args.root/name/'contract.json') for name in variants}
    for name in variants:
        completion=read_json(args.root/name/'complete.json')
        if not completion or completion['optimizer_step']!=5000:
            raise RuntimeError(f'{name} has not completed exactly 5,000 updates')
        if contracts[name]['cache']!=contracts['mdm']['cache']:
            raise RuntimeError('Compared models use different prepared data')
    cache=Path(contracts['mdm']['cache']['validation']['path'])
    if digest(np.frombuffer((cache/'state.json').read_bytes(),dtype=np.uint8))!=contracts['mdm']['cache']['validation']['state_sha256']:
        raise RuntimeError('Validation cache manifest changed')
    dataset=load_from_disk(str(cache))
    row_ids=sorted(np.random.default_rng(args.seed).choice(1024,args.examples,replace=False).tolist())
    source_cache={i:nearest_different_sources(dataset[i]['input_ids'],[50256,50257]) for i in row_ids}
    protocol=dict(mask_ratios=MASK_RATIOS,correct_fractions=CORRECT_FRACTIONS,
        correctness_population='revealed tokens; fixed first token excluded only in first-clean anchor mode',
        mask_count_rule='round(ratio * eligible input positions), exact per row',
        wrong_count_rule='round((1-correctness) * revealed eligible positions), guaranteed different tokens',
        anchor=args.anchor,scored_targets='masked original-clean targets, excluding position zero',
        cosine_definition='cos(h_i before block l, h_i after block l); within each model',
        cosine_populations='masked, correct revealed, wrong revealed; exclude first position and clean EOS',
        cosine_reduction='mean over eligible positions within each row, then equal-weight mean over rows',
        corruption='nearest different clean token in the same packed row; ties go left; EOS/MASK excluded',
        replacements_may_copy_masked_sources=True,seed=args.seed,row_ids=row_ids,
        exposure_reference='At c=1, score the same masked target positions that each c<1 canvas will copy into wrong context; no additional forward pass',
        variants=variants,optimizer_step=5000,
        parameter_state='EMA',precision='FP32; TF32 disabled',device=device,microbatch=args.microbatch,
        full_mask_control='Identical canvas and scores across reliability cells at 100% masking',
        head_diagnostics='Same-target auxiliary CE, accuracy, agreement, five error cases and rescue; same backbone forward pass; row aggregates and compressed prediction arrays',
        head_source_states={'masked':0,'correct_revealed':1,'wrong_revealed':2,'ineligible':255})
    atomic_write(args.output/'protocol.json',json.dumps(protocol,indent=2)+'\n')
    started=time.monotonic();rows=[];provenance={}
    for variant in variants:
        model,provenance[variant]=load_ema_model(args.root/variant)
        if model.time_conditioning or model.parameterization!='subs':
            raise RuntimeError('This protocol is for the current time-unconditioned SUBS MDLM pilot')
        if model.backbone.attn_backend != 'sdpa':
            raise RuntimeError('The FP32 diagnostic requires the SDPA attention backend')
        model.backbone.force_fp32_eval=True
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise RuntimeError('The frozen diagnostic requires FP32 model parameters')
        provenance[variant]['forward_precision']='FP32; backbone BF16 autocast explicitly disabled'
        model.to(device);full_mask_metrics={};full_mask_predictions={}
        for ratio in MASK_RATIOS:
            for reliability in CORRECT_FRACTIONS:
                prediction_batches=[]
                for start in range(0,len(row_ids),args.microbatch):
                    ids=row_ids[start:start+args.microbatch]
                    clean=np.array([dataset[i]['input_ids'] for i in ids],dtype=np.int64)
                    inputs=[make_canvas(x,ratio,reliability,args.seed,i,None,50257,[50256],
                            keep_first=args.anchor=='first-clean',nearest_sources=source_cache[i]) for i,x in zip(ids,clean)]
                    canvases=np.stack([v[0] for v in inputs]);masked=np.stack([v[1] for v in inputs])
                    wrong=np.stack([v[2] for v in inputs])
                    sources=np.stack([source_cache[i] for i in ids])
                    exposed=exposed_target_mask(masked,wrong,sources)
                    references={}
                    if reliability==1.:
                        for reference_c in CORRECT_FRACTIONS[1:]:
                            future=[make_canvas(x,ratio,reference_c,args.seed,i,None,50257,[50256],
                                keep_first=args.anchor=='first-clean',nearest_sources=source_cache[i])
                                for i,x in zip(ids,clean)]
                            assert all(np.array_equal(v[1],masked[k]) for k,v in enumerate(future))
                            references[reference_c]=exposed_target_mask(masked,np.stack([v[2] for v in future]),sources)
                    if ratio==1. and reliability!=1.:
                        values=[]
                        for i,x in zip(ids,canvases):
                            saved,canvas_hash=full_mask_metrics[i]
                            assert digest(x)==canvas_hash
                            values.append(saved.copy())
                        predictions=full_mask_predictions[start]
                    else:
                        predictions={}
                        values=score_batch(model,clean,canvases,masked,wrong,device,exposed,
                                           capture_cosines=True,reference_exposures=references,
                                           capture_heads=True,prediction_output=predictions)
                        if ratio==1.:
                            full_mask_predictions[start]=predictions
                    prediction_batches.append(predictions)
                    for k,(i,value) in enumerate(zip(ids,values)):
                        row=dict(variant=variant,row_id=i,mask_ratio=ratio,correct_fraction=reliability,
                            input_sha256=digest(canvases[k]),clean_sha256=digest(clean[k]),
                            mask_sha256=digest(masked[k]),**inputs[k][3],**value)
                        rows.append(row)
                        if ratio==1. and reliability==1.:
                            full_mask_metrics[i]=(value.copy(),row['input_sha256'])
                    atomic_write(args.output/'progress.json',json.dumps(dict(variant=variant,
                        mask_ratio=ratio,correct_fraction=reliability,finished_observations=len(rows),
                        elapsed_seconds=time.monotonic()-started),indent=2)+'\n')
                print(variant,ratio,reliability,'finished',len(rows),'observations',flush=True)
                prediction_dir=args.output/'head_predictions'
                prediction_dir.mkdir(exist_ok=True)
                prediction_file=prediction_dir/f'{variant}_mask{round(ratio*100):03d}_correct{round(reliability*100):03d}.npz'
                np.savez_compressed(prediction_file,row_ids=np.asarray(row_ids),
                    **{key:np.concatenate([batch[key] for batch in prediction_batches]) for key in prediction_batches[0]})
                atomic_write(args.output/'partial_observations.json',json.dumps(rows)+'\n')
        del model;gc.collect()
        if args.device=='cuda':
            torch.cuda.empty_cache()
    columns=sorted({k for row in rows for k in row})
    with (args.output/'observations.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    summary=summarize(rows,variants)
    layerwise=summarize_cosines(rows,variants,layers=12)
    from owt.head_diagnostics import summarize as summarize_heads
    head_summary=summarize_heads(rows)
    atomic_write(args.output/'head_diagnostics_summary.json',json.dumps(dict(
        summary=head_summary,precision='FP32',parameter_state='EMA',optimizer_step=5000,
        uncertainty='Paired row sufficient statistics are saved; these point estimates do not measure training-seed uncertainty.',
        prediction_arrays='head_predictions/*.npz',
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['owt/head_diagnostics.py','owt/reveal_sweep.py']},
        limitations=['Auxiliary heads are native only to NP arms.','Oracle top1 selection uses labels; it is not deployable performance.']),indent=2)+'\n')
    artifact=dict(protocol=protocol,provenance=provenance,summary=summary,
        layerwise_cosines=layerwise,head_diagnostics=head_summary,
        elapsed_seconds=time.monotonic()-started,limitations=[
        'Conditional masked-token CE, not the production ELBO, a likelihood bound, or generation quality.',
        'One paired training seed; row bootstrap does not estimate seed variability or account for document dependence.',
        'Nearest-token replacements can expose clean tokens from masked source positions; copied-target strata are recorded.',
        'Copied versus not-copied target strata are matched across context conditions, but do not isolate the causal effect of copying: all incorrect reveals change together, and copied token values may occur at other targets.',
        'Synthetic incorrect reveals are outside absorbing-mask training and do not reproduce generation errors.',
        'With anchor=none, masking/corrupting the first input token also departs from its always-clean training convention.',
        'Adjacent-wrong-target strata are exploratory and do not isolate local causal effects.'])
    artifact['limitations'].append('Layer-update cosine measures geometric change, not forgetting, correctness, or useful downstream computation.')
    atomic_write(args.output/'summary.json',json.dumps(artifact,indent=2)+'\n')
    identity=hashlib.sha256(json.dumps(dict(protocol=protocol,provenance=provenance),sort_keys=True).encode()).hexdigest()[:12]
    record_event('reveal_sweep_'+identity,'Masking and reveal-reliability sweep completed',
        f'Evaluated final EMA checkpoints for {", ".join(variants)} on {args.examples} fixed held-out rows '
        'over mask ratios 100/80/60/40/20% and revealed-token correctness 100/80/60%. '
        'All arms used identical corruptions and original clean masked targets. The fully masked row '
        'shares identical inputs and scores across correctness cells. Results include masked CE, accuracy, '
        'paired NP-minus-MDM differences, and excess unreliability penalties with row-bootstrap intervals. '
        'Layerwise before/after-block cosine is recorded for masked, correct revealed, and wrong revealed tokens. '
        'This conditional stress test is not a generation benchmark or a seed-level replication.',artifact)
    print('COMPLETE',args.output,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'outputs/owt/mdm-np-5k')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variants',nargs='+',choices=list(LABELS),default=['mdm','mdm_np_zero_init'])
    parser.add_argument('--examples',type=int,default=1024)
    parser.add_argument('--microbatch',type=int,default=4)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--anchor',choices=['none','first-clean'],default='none')
    parser.add_argument('--seed',type=int,default=20261004)
    args=parser.parse_args()
    if (not 1<=args.examples<=1024 or min(args.microbatch,args.threads)<1 or
        'mdm' not in args.variants or len(set(args.variants))!=len(args.variants)):
        parser.error('Use 1--1024 rows, positive batch/threads, unique variants including MDM')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.sweep.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/'summary.json').exists():
            parser.error('Completed evidence exists here; use a fresh output directory')
        if args.device=='cuda':
            with (args.root/'.queue.lock').open('a') as gpu_lock:
                print('Waiting for training queue lock; no CUDA context allocated.',flush=True)
                fcntl.flock(gpu_lock,fcntl.LOCK_EX)
                run(args)
        else:
            run(args)


if __name__=='__main__':
    main()
