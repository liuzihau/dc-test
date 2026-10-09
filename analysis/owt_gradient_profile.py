"""Local, CPU-only main/auxiliary gradient diagnostic at a frozen EMA checkpoint.

Measures all shared parameters before task-specific vocabulary readouts, with
dropout disabled and no optimizer update. This is not the accumulated BF16
training gradient, an Adam update, or a causal explanation of the training gap.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import resource
import time

from analysis.owt_mask_profile import ROOT,load_model,checksum
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from datasets import load_from_disk
from owt.neighbor import neighbor_terms
from owt.research import atomic_write,record_event


def source_losses(heads,hidden,clean,noisy,valid,mask_id,boundary_ids,weights,
                  denominator,coefficients,chunk_size=128):
    content=valid.bool()&clean.ne(mask_id)
    for token in boundary_ids: content &= clean.ne(token)
    targets=content&noisy.eq(mask_id)
    targets[:,0]=False
    weights=torch.broadcast_to(weights,clean.shape)
    total=hidden.sum()*0
    visible,latent=total,total
    counts={'visible':0,'masked':0}
    for offset,head in zip(heads.offsets,heads.heads):
        assert abs(offset)==1
        source=slice(0,-1) if offset>0 else slice(1,None)
        target=slice(1,None) if offset>0 else slice(0,-1)
        eligible=targets[:,target]&content[:,source]
        h=hidden[:,source][eligible];y=clean[:,target][eligible]
        factors=weights[:,target][eligible]
        masked_source=noisy[:,source][eligible].eq(mask_id)
        counts['masked']+=int(masked_source.sum())
        counts['visible']+=int((~masked_source).sum())
        def chunk(features,labels,weight,is_masked,projection=head):
            logits=projection(features)
            forbidden=torch.arange(logits.shape[-1],device=logits.device).eq(mask_id)
            logits=logits.masked_fill(forbidden,-torch.inf)
            ce=F.cross_entropy(logits,labels,reduction='none')*weight
            return torch.stack([(ce*(~is_masked)).sum(),(ce*is_masked).sum()])
        for start in range(0,len(y),chunk_size):
            values=checkpoint(chunk,h[start:start+chunk_size],y[start:start+chunk_size],
                              factors[start:start+chunk_size],masked_source[start:start+chunk_size],
                              use_reentrant=False,preserve_rng_state=False)
            visible=visible+float(coefficients[offset])*values[0]/denominator.clamp_min(1)
            latent=latent+float(coefficients[offset])*values[1]/denominator.clamp_min(1)
    return visible,latent,counts


def group_name(name):
    if name.startswith('vocab_embed.'): return 'token_embeddings'
    if name.startswith('blocks.'):
        layer=int(name.split('.')[1])
        return ['blocks_0_3','blocks_4_7','blocks_8_11'][layer//4]
    return 'other_shared'


def gradient_stats(named,main,visible,latent):
    groups={}
    for (name,param),gm,gv,gl in zip(named,main,visible,latent):
        if all(g is None for g in [gm,gv,gl]): continue
        ga=(gv+gl) if gv is not None and gl is not None else gv if gv is not None else gl
        for label in [group_name(name),'all_shared_trunk']:
            row=groups.setdefault(label,dict(main_squared=0,visible_squared=0,masked_squared=0,
                                             auxiliary_squared=0,main_visible_dot=0,main_masked_dot=0,
                                             main_auxiliary_dot=0))
            for key,g in [('main',gm),('visible',gv),('masked',gl),('auxiliary',ga)]:
                if g is not None: row[key+'_squared']+=float(torch.sum(g*g,dtype=torch.float64))
            if gm is not None:
                for key,g in [('visible',gv),('masked',gl),('auxiliary',ga)]:
                    if g is not None: row['main_'+key+'_dot']+=float(torch.sum(gm*g,dtype=torch.float64))
    for row in groups.values():
        for key in ['main','visible','masked','auxiliary']:
            row[key+'_norm']=math.sqrt(row[key+'_squared'])
        for key in ['visible','masked','auxiliary']:
            denominator=row['main_norm']*row[key+'_norm']
            row['main_'+key+'_cosine']=row['main_'+key+'_dot']/denominator if denominator else None
            row[key+'_to_main_norm']=row[key+'_norm']/row['main_norm'] if row['main_norm'] else None
    return groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels',type=float,nargs='+',default=[.1,.5,.9])
    parser.add_argument('--rows',type=int,nargs='+',default=[47,141])
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/analysis/owt-gradient-profile-5000')
    args=parser.parse_args()
    if any(not 0<t<1 for t in args.levels): parser.error('Use interior noise levels')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available()
    started=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=True)
    def progress(stage, **details):
        atomic_write(args.output/'progress.json',json.dumps(dict(
            stage=stage,pid=os.getpid(),elapsed_seconds=time.monotonic()-started,**details),indent=2)+'\n')
    progress('loading_checkpoint')
    model,provenance=load_model(ROOT/'outputs/owt/mdm-np-5k/mdm_np')
    dataset=load_from_disk(str(ROOT/'.cache/huggingface/openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat'))
    clean=torch.tensor([dataset[index]['input_ids'] for index in args.rows],dtype=torch.long)
    named=[(name,p) for name,p in model.backbone.named_parameters()
           if p.requires_grad and not name.startswith(('neighbor_heads.','output_layer.linear.'))]
    params=[p for _,p in named]
    coefficients=dict(zip(model.np_config.offsets,model.np_config.weights))
    results=[]
    for k,level in enumerate(args.levels):
        progress('forward',noise_level=level)
        captured={}
        handles=[model.backbone.register_forward_pre_hook(
                    lambda module,inputs:captured.update(noisy=inputs[0].detach().clone())),
                 model.backbone.output_layer.linear.register_forward_pre_hook(
                    lambda module,inputs:captured.update(hidden=inputs[0])),
                 model.noise.register_forward_hook(
                    lambda module,inputs,output:captured.update(weight=-output[0]))]
        try:
            torch.manual_seed(20261002+k)
            losses=model._forward_pass_diffusion(clean,t=torch.full((len(clean),1),level),
                              sampling_eps_min=1e-3,sampling_eps_max=1.0)
            main_loss=losses.mean()
            valid=torch.ones_like(clean);denominator=valid.sum()
            visible,latent,counts=source_losses(model.backbone.neighbor_heads,captured['hidden'],
                clean,captured['noisy'],valid,model.mask_index,model.boundary_ids,
                captured['weight'],denominator,coefficients)
            with torch.no_grad():
                reference,_=neighbor_terms(model.backbone.neighbor_heads,captured['hidden'],
                    clean,captured['noisy'],valid,model.mask_index,model.boundary_ids,
                    captured['weight'],denominator,chunk_size=128,ignore_first=model.ignore_bos)
                expected=sum(coefficients[offset]*value for offset,value in reference.items())
                torch.testing.assert_close(visible+latent,expected,rtol=2e-5,atol=2e-5)
            gmain=torch.autograd.grad(main_loss,params,retain_graph=True,allow_unused=True)
            progress('main_gradient_complete',noise_level=level)
            gvisible=torch.autograd.grad(visible,params,retain_graph=True,allow_unused=True)
            progress('visible_gradient_complete',noise_level=level)
            glatent=torch.autograd.grad(latent,params,allow_unused=True)
            stats=gradient_stats(named,gmain,gvisible,glatent)
            row=dict(noise_level=level,main_loss=float(main_loss.detach()),
                     weighted_visible_auxiliary=float(visible.detach()),
                     weighted_masked_auxiliary=float(latent.detach()),
                     counts=counts,noisy_sha256=checksum(captured['noisy']),groups=stats)
            results.append(row)
            atomic_write(args.output/'partial_results.json',json.dumps(results,indent=2)+'\n')
            print(json.dumps(dict(noise_level=level,shared=stats['all_shared_trunk'])),flush=True)
            del gmain,gvisible,glatent,losses,main_loss,visible,latent
        finally:
            for handle in handles: handle.remove()
            captured.clear();gc.collect()
        assert all(p.grad is None for p in model.parameters())
    artifact=dict(provenance=provenance,row_ids=args.rows,results=results,
        shared_parameters=sum(p.numel() for p in params),elapsed_seconds=time.monotonic()-started,
        peak_cpu_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        limitations=[f'Frozen EMA, CPU FP32, dropout disabled, {len(args.rows)} rows, {len(args.levels)} noise levels.',
                    'Not accumulated BF16 training gradients or Adam/clipped update directions.',
                    'Local alignment does not identify the cause of training-trajectory differences.'])
    atomic_write(args.output/'summary.json',json.dumps(artifact,indent=2)+'\n')
    progress('complete')
    identity=checksum(clean)[:12]+'_'+'_'.join(str(t) for t in args.levels)
    record_event('shared_trunk_gradient_profile_5000_'+identity,'Frozen shared-trunk gradient diagnostic completed',
        'Measured main, visible-source NP, and masked-source NP gradients over all shared trunk '
        'parameters before the task-specific vocabulary readouts. Used frozen EMA parameters, '
        f'CPU FP32, dropout disabled, {len(args.rows)} fixed held-out rows, and {len(args.levels)} corruption levels. '
        'Auxiliary source-class loss sums matched the production objective. No optimizer update '
        'was performed. Results are local diagnostic evidence, not the actual training updates '
        'or a causal explanation of trajectory differences.',artifact)


if __name__=='__main__':main()
