"""Frozen, paired CPU diagnostic of final OWT MDM and selected NP checkpoints.

Uses EMA, full 1,024-token geometry, shared corruption draws, and fixed noise
levels. This exploratory FP32 diagnostic is not production BF16 validation or
an estimate of variability across training seeds. It never constructs a trainer.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/runtime/analysis/matplotlib'))

import numpy as np
import torch
from datasets import load_from_disk
from torch.nn import functional as F

from owt.checkpoint import load_ema_model as load_model, checkpoint_provenance
from owt.research import atomic_write, record_event


def checksum(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def auxiliary_ce(head,hidden,targets,mask_index):
    chunks=[]
    for start in range(0,len(targets),128):
        logits=head(hidden[start:start+128])
        logits[:,mask_index]=-torch.inf
        chunks.append(F.cross_entropy(logits,targets[start:start+128],reduction='none'))
    return torch.cat(chunks) if chunks else hidden.new_empty(0)


def evaluate(model,x,noise_level,seed):
    captured={}
    def raw_output(module,inputs,logits):
        # MASK is the final vocabulary row. Match the main head's normalization.
        captured['ce']=(torch.logsumexp(logits[...,:model.mask_index],dim=-1)
                        -logits.gather(-1,x[...,None]).squeeze(-1)).detach()
    handles=[model.backbone.register_forward_pre_hook(
                 lambda module,inputs:captured.update(noisy=inputs[0].detach().clone())),
             model.backbone.register_forward_hook(raw_output),
             model.backbone.output_layer.linear.register_forward_pre_hook(
                 lambda module,inputs:captured.update(hidden=inputs[0].detach()))]
    try:
        torch.manual_seed(seed)
        with torch.inference_mode():
            losses=model._forward_pass_diffusion(x,t=torch.tensor([[noise_level]]),
                         sampling_eps_min=1e-3,sampling_eps_max=1.0)
            noisy=captured['noisy']
            masked=noisy.eq(model.mask_index)
            assert not masked[:,0].any()
            ce=captured['ce']
            scale=float((losses[masked]/ce[masked])[0])
            torch.testing.assert_close(losses[masked],ce[masked]*scale,rtol=2e-5,atol=2e-5)
            result=dict(noise_level=noise_level,seed=seed,clean_sha256=checksum(x),
                noisy_sha256=checksum(noisy),masked_tokens=int(masked.sum()),
                main_elbo=float(losses[:,1:].mean()),
                main_masked_ce=float(ce[masked].mean()),actual_loss_weight=scale)
            content=x.ne(50256)
            interior=content[:,1:-1]&content[:,:-2]&content[:,2:]&masked[:,1:-1]
            left_masked=masked[:,:-2]
            right_masked=masked[:,2:]
            for label,condition in [('both_visible',~left_masked&~right_masked),
                                    ('mixed',left_masked^right_masked),
                                    ('both_masked',left_masked&right_masked)]:
                select=interior&condition
                result[label+'_count']=int(select.sum())
                result[label+'_main_ce']=float(ce[:,1:-1][select].mean()) if select.any() else None
            paired=interior&(left_masked^right_masked)
            result['paired_source_count']=int(paired.sum())
            if model.np_config.enabled and paired.any():
                hidden=captured['hidden']
                heads=dict(zip(model.backbone.neighbor_heads.offsets,model.backbone.neighbor_heads.heads))
                y=x[:,1:-1][paired]
                left=auxiliary_ce(heads[1],hidden[:,:-2][paired],y,model.mask_index)
                right=auxiliary_ce(heads[-1],hidden[:,2:][paired],y,model.mask_index)
                choose_left=left_masked[paired]
                visible=torch.where(choose_left,right,left)
                latent=torch.where(choose_left,left,right)
                result.update(paired_visible_source_ce=float(visible.mean()),
                    paired_masked_source_ce=float(latent.mean()),
                    paired_self_ce=float(ce[:,1:-1][paired].mean()))
            return result
    finally:
        for handle in handles: handle.remove()


def summarize(rows,comparison='mdm_np'):
    summary=[]
    for level in sorted({row['noise_level'] for row in rows}):
        base={row['row_id']:row for row in rows if row['variant']=='mdm' and row['noise_level']==level}
        other={row['row_id']:row for row in rows if row['variant']==comparison and row['noise_level']==level}
        assert base.keys()==other.keys(), 'Diagnostic arms must cover exactly the same rows'
        ids=sorted(base.keys())
        assert ids
        for index in ids:
            assert base[index]['clean_sha256']==other[index]['clean_sha256']
            assert base[index]['noisy_sha256']==other[index]['noisy_sha256']
            assert base[index]['paired_source_count']==other[index]['paired_source_count']
        difference=np.array([other[index]['main_elbo']-base[index]['main_elbo'] for index in ids])
        rng=np.random.default_rng(20261001)
        samples=difference[rng.integers(0,len(ids),size=(2000,len(ids)))].mean(axis=1)
        item=dict(noise_level=level,rows=len(ids),
                  mdm_main_elbo=float(np.mean([base[i]['main_elbo'] for i in ids])),
                  comparison_main_elbo=float(np.mean([other[i]['main_elbo'] for i in ids])),
                  delta_main_elbo=float(difference.mean()),
                  diagnostic_row_bootstrap_95=[float(v) for v in np.quantile(samples,[.025,.975])],
                  positive_row_differences=int((difference>0).sum()))
        if comparison=='mdm_np':
            item['random_np_main_elbo']=item['comparison_main_elbo']
        pairs=sum(other[i]['paired_source_count'] for i in ids)
        if pairs:
            for name in ['paired_visible_source_ce','paired_masked_source_ce','paired_self_ce']:
                item[name]=sum(other[i][name]*other[i]['paired_source_count']
                               for i in ids if other[i]['paired_source_count']>0)/pairs
            item['paired_source_count']=pairs
        summary.append(item)
    return summary


def reuse_baseline(reference,run,row_ids,levels,dataset):
    """Reuse immutable paired reference observations only after provenance checks."""
    artifact=json.loads(reference.read_text())
    assert artifact['row_ids']==row_ids and artifact['noise_levels']==levels, 'Reference sampling changed'
    expected=checkpoint_provenance(run)
    assert artifact['provenance']['mdm']==expected, 'Reference checkpoint or configuration changed'
    with (reference.parent/'observations.csv').open() as stream:
        selected=[r for r in csv.DictReader(stream) if r['variant']=='mdm']
    rows=[]
    for raw in selected:
        row={key:(value if key in ['variant','clean_sha256','noisy_sha256'] else
                  float(value) if value!='' else None) for key,value in raw.items()}
        row['row_id']=int(row['row_id'])
        row['seed']=int(row['seed'])
        assert all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
        index=row['row_id']; level=row['noise_level']
        assert index in row_ids and level in levels
        assert row['seed']==20261001+10000*index+levels.index(level)
        clean=torch.tensor(dataset[index]['input_ids'],dtype=torch.long).reshape(1,-1)
        assert row['clean_sha256']==checksum(clean), 'Reference validation data changed'
        rows.append(row)
    assert len(rows)==len(row_ids)*len(levels)
    assert {(r['row_id'],r['noise_level']) for r in rows}=={(i,t) for i in row_ids for t in levels}
    return rows,expected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'outputs/owt/mdm-np-5k')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/analysis/owt-zero-init-mask-profile-5000')
    parser.add_argument('--examples',type=int,default=16)
    parser.add_argument('--levels',type=float,nargs='+',default=[.1,.3,.5,.7,.9])
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--compare',choices=['mdm_np','mdm_np_zero_init','mdm_np_zero_init_low_weight'],default='mdm_np_zero_init')
    parser.add_argument('--reference',type=Path,help='Reuse checked MDM observations from an earlier paired summary')
    args=parser.parse_args()
    if (not 1<=args.examples<=1024 or any(not 0<t<1 for t in args.levels)
            or len(set(args.levels))!=len(args.levels) or args.threads<1):
        parser.error('Use 1--1024 examples, unique interior noise levels, and positive threads')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available(), 'This diagnostic must remain CPU only'
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'summary.json').exists():
        parser.error('A completed diagnostic already exists here; use a fresh output directory')
    for variant in ['mdm',args.compare]:
        completion=json.loads((args.root/variant/'complete.json').read_text())
        if completion['optimizer_step']!=5000:
            parser.error('Both compared variants must have completed exactly 5,000 optimizer updates')
    started=time.monotonic()
    row_ids=sorted(np.random.default_rng(20261001).choice(1024,args.examples,replace=False).tolist())
    dataset=load_from_disk(str(ROOT/'.cache/huggingface/openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat'))
    rows=[]; provenance={}
    if args.reference:
        rows,provenance['mdm']=reuse_baseline(args.reference,args.root/'mdm',row_ids,args.levels,dataset)
        print('Reused',len(rows),'provenance-checked MDM observations',flush=True)
    for variant in ([args.compare] if args.reference else ['mdm',args.compare]):
        print('Loading',variant,flush=True)
        model,provenance[variant]=load_model(args.root/variant)
        for index in row_ids:
            x=torch.tensor(dataset[index]['input_ids'],dtype=torch.long).reshape(1,-1)
            for k,level in enumerate(args.levels):
                seed=20261001+10000*index+k
                result=evaluate(model,x,level,seed)
                result.update(variant=variant,row_id=index)
                rows.append(result)
            atomic_write(args.output/'progress.json',json.dumps(dict(
                variant=variant,finished_rows=len(rows),elapsed_seconds=time.monotonic()-started),indent=2)+'\n')
            print(variant,'row',index,'finished observations',len(rows),flush=True)
        del model
        gc.collect()
    summary=summarize(rows,args.compare)
    columns=sorted({key for row in rows for key in row})
    with (args.output/'observations.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    artifact=dict(provenance=provenance,row_ids=row_ids,noise_levels=args.levels,comparison_variant=args.compare,
        corruptions_identical=True,summary=summary,elapsed_seconds=time.monotonic()-started,
        limitations=['One paired training seed; no seed-level uncertainty estimate.',
                    'CPU FP32, evaluation mode, diagnostic noise grid; not production validation replay.',
                    'Bootstrap resamples packed rows; it does not account for document dependence.',
                    'Existing auxiliary readouts do not establish downstream computational use.'])
    atomic_write(args.output/'summary.json',json.dumps(artifact,indent=2)+'\n')
    identity=hashlib.sha256(json.dumps(dict(rows=row_ids,levels=args.levels,
                         provenance=provenance),sort_keys=True).encode()).hexdigest()[:12]
    record_event('cpu_mask_profile_5000_'+identity,'Frozen paired CPU diagnostic completed',
        f'Profiled the final EMA MDM and {args.compare} checkpoints on {args.examples} fixed held-out rows '
        f'and {len(args.levels)} noise levels. Full sequence geometry was preserved and corruption '
        'hashes matched across arms. Outputs include paired main-loss differences and same-target '
        'visible/masked source readouts. This is exploratory CPU FP32 evidence, not a training-seed '
        'replication or causal proof of representation use.',artifact)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
