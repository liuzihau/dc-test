"""Paired main-only EMA/FP32 diagnosis at three reveal rates, without GPU use."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch
import warnings

os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
import numpy as np
from analysis.local_denoising_metrics import make_canvas,visibility_statistics
from analysis.local_top5_metrics import statistics,transition_statistics,summaries,paired_summary,FIELDS,CLASSES
from owt.continuation import ROOT,A,ORIGINALS,digest,verify_selection
from owt.research import atomic_write,timestamp

DATA=ROOT/'outputs/analysis/owt-local-denoising-20261006'
OUTPUT=ROOT/'outputs/analysis/owt-local-top5-mdm-A-5000-20261006'


def write(path,value):
    atomic_write(path,json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_model(variant,selection,checkpoint_step=5000):
    import torch
    from omegaconf import OmegaConf
    from owt.model import OWTMDM
    from owt.transformer_np_model import TransformerNPMDM
    record=selection['checkpoints'][variant]
    run_root=ROOT/ORIGINALS[variant]
    if checkpoint_step==7500:
        run_root=ROOT/'outputs/owt/continuation-7500'/variant
        completion=json.loads((run_root/'complete.json').read_text())
        if completion.get('optimizer_step')!=7500 or completion.get('resumed_from')!=5000:
            raise ValueError('Both continuations must finish before step7500 comparison')
        checkpoint=run_root/'checkpoints/step-0007500.ckpt'
        stat=checkpoint.stat()
        record=dict(path=str(checkpoint.relative_to(ROOT)),sha256=digest(checkpoint),
            bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,optimizer_step=7500)
    checkpoint=ROOT/record['path']
    if digest(checkpoint)!=record['sha256']:raise ValueError('Checkpoint checksum differs')
    payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    config=OmegaConf.load(run_root/'resolved_config.yaml')
    tokenizer=SimpleNamespace(vocab_size=50257,mask_token=None,all_special_ids=[50256])
    with patch('diffusion.metrics.Metrics',return_value=torch.nn.Module()):
        model=(OWTMDM if variant=='mdm' else TransformerNPMDM)(config,tokenizer)
    if payload['global_step']!=checkpoint_step or payload['ema']['num_updates']!=checkpoint_step:
        raise ValueError('Require matched EMA checkpoint step')
    if variant==A and payload['transformer_np']['signature']!=model.resume_signature():
        raise ValueError('A architecture differs')
    model.load_state_dict(payload['state_dict'],strict=True)
    model.ema.load_state_dict(payload['ema'])
    parameters=list(model._get_parameters());shadows=model.ema.shadow_params
    if len(parameters)!=len(shadows) or any(p.shape!=s.shape for p,s in zip(parameters,shadows)):
        raise ValueError('EMA parameter order/shape differs')
    model.ema.copy_to(parameters)
    if any(not torch.equal(p,s) for p,s in zip(parameters,shadows)):
        raise ValueError('EMA swap is incomplete')
    model.ema=None;model.float().eval().requires_grad_(False)
    model.backbone.force_fp32_eval=True
    if model.time_conditioning or model.cross_attn or model.mask_index!=50257:
        raise ValueError('Unexpected main inference geometry')
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Model is not FP32')
    def check_precision(module,inputs,output):
        if inputs[0].dtype!=torch.float32 or output.dtype!=torch.float32:
            raise ValueError('An attention block entered lower precision')
    for block in model.backbone.blocks:block.register_forward_hook(check_precision)
    def reject_auxiliary(*args):raise ValueError('Auxiliary processing ran during main-only diagnosis')
    if variant==A:
        for branch in model.backbone.neighbor_branches:branch.register_forward_pre_hook(reject_auxiliary)
        model.backbone.neighbor_heads.register_forward_pre_hook(reject_auxiliary)
    del payload,parameters,shadows;gc.collect()
    return model,record


def score(model,canvas,clean,probability):
    import torch
    x=torch.as_tensor(canvas,dtype=torch.long);y=torch.as_tensor(clean,dtype=torch.long)
    sigma=torch.full((len(x),1),-np.log1p(-probability),dtype=torch.float32)
    with torch.inference_mode(),warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='.*CUDA is not available.*')
        warnings.filterwarnings('ignore',message='.*target dtype is not supported.*')
        logp=model(x,sigma)
        if logp.dtype!=torch.float32 or logp.shape!=(*x.shape,50258):
            raise ValueError('Unexpected score dtype/shape')
        logp[...,50257]=-torch.inf
        prediction=logp.argmax(-1)
        top5=logp.topk(5,dim=-1).indices
        nll=-logp.gather(-1,y[...,None]).squeeze(-1)
        if not torch.isfinite(nll).all():raise ValueError('Nonfinite correct-token loss')
        return prediction.numpy().astype(np.int32),top5.numpy().astype(np.int32),nll.numpy()


def run(args):
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    selection=verify_selection(ROOT/'outputs/research-notes/continuation_7500_selection_20261006_v2.json')
    receipt=json.loads((DATA/'data_receipt.json').read_text())
    if digest(DATA/'inputs.npz')!=receipt['inputs_sha256']:raise ValueError('Input data changed')
    data=np.load(DATA/'inputs.npz',allow_pickle=False)
    clean=data['clean'][:args.samples];ids=data['document_ids'][:args.samples]
    if clean.shape!=(args.samples,1024) or len(np.unique(ids))!=args.samples:
        raise ValueError('Require independent document rows of length1024')
    if (clean>=50257).any():raise ValueError('Exclude mask/pad collision from ground truth')
    protocol=dict(created_at=timestamp(),models=['MDM','A'],checkpoint_step=args.checkpoint_step,parameter_state='EMA',
        precision='FP32, TF32 disabled; every attention block audited',main_heads_only=True,device='cpu',cpu_threads=2,
        samples=args.samples,length=1024,corruption_seeds=list(range(args.seeds)),reveal_probabilities=args.reveal_levels,
        mask_probability_rule='1 minus reveal probability; Bernoulli draws, not exact counts',
        document_ids=ids.tolist(),data_receipt=receipt,inputs_sha256=digest(DATA/'inputs.npz'),
        local_visibility_offsets=[-2,-1,1,2],visibility='all valid masked centers; no distinct-ID restriction',
        error_eligibility='three masked content positions, pairwise-distinct true IDs; same document/block',
        error_fields=list(FIELDS),transition_classes=list(CLASSES),
        error_subclass_priority=['center_support','neighbor_only_support','no_local_support'],top_k=5,
        bootstrap_unit='document; keep all corruption seeds together',bootstrap_draws=args.bootstraps,
        original_reference_preserved=True,exploratory_data_previously_inspected=True,
        full_run=args.samples==100 and args.seeds==10 and args.reveal_levels==[.25,.5,.75],
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in
            [Path(__file__),ROOT/'analysis/local_top5_metrics.py',ROOT/'analysis/local_denoising_metrics.py',
             ROOT/'analysis/plot_local_top5_pair.py',ROOT/'analysis/test_local_top5_metrics.py']},
        production_source_sha256=selection['source_sha256'])
    write(args.output/'protocol.json',protocol)
    models={};started=time.monotonic()
    for variant,label in [('mdm','MDM'),(A,'A')]:
        write(args.output/'progress.json',dict(stage='loading_checkpoint',model=label,pid=os.getpid()))
        models[label],record=load_model(variant,selection,args.checkpoint_step);protocol.setdefault('checkpoints',{})[label]=record
    write(args.output/'protocol.json',protocol)
    forwards=0;expected=2*args.samples*args.seeds*len(args.reveal_levels)
    draws=np.random.default_rng(20261006).integers(0,args.samples,size=(args.bootstraps,args.samples))
    complete_cells=[]
    for reveal in args.reveal_levels:
        probability=1-reveal;name=f'reveal{round(100*reveal):02d}'
        folder=args.output/name;folder.mkdir()
        vis={k:[] for k in models};counts={k:[] for k in models};tables=[];realized=[]
        for seed in range(args.seeds):
            canvas,masked=make_canvas(clean,ids,seed,probability)
            realized.append(float(masked.sum()/(clean!=50256).sum()))
            pred={k:[] for k in models};top={k:[] for k in models};nll={k:[] for k in models}
            for document in range(args.samples):
                for label,model in models.items():
                    p,t,l=score(model,canvas[document:document+1],clean[document:document+1],probability)
                    pred[label].append(p);top[label].append(t);nll[label].append(l);forwards+=1
                    write(args.output/'progress.json',dict(stage='evaluating',reveal_probability=reveal,
                        corruption_seed=seed,finished_documents= document+(label=='A'),model=label,
                        completed_forwards=forwards,requested_forwards=expected,
                        elapsed_seconds=time.monotonic()-started,pid=os.getpid()))
            arrays=dict(document_ids=ids,clean=clean,canvas=canvas,masked=masked)
            for label in models:
                pred[label]=np.concatenate(pred[label]);top[label]=np.concatenate(top[label]);nll[label]=np.concatenate(nll[label])
                v=visibility_statistics(clean,masked,pred[label],nll[label])
                e=statistics(clean,masked,pred[label],top[label])
                vis[label].append(v);counts[label].append(e)
                arrays.update({label+'_prediction':pred[label],label+'_top5':top[label],label+'_nll':nll[label],
                    label+'_visibility':v,label+'_errors':e})
            table=transition_statistics(clean,masked,pred['MDM'],top['MDM'],pred['A'],top['A'])
            tables.append(table);arrays['paired_transitions']=table
            np.savez_compressed(folder/f'seed-{seed:02d}.npz',**arrays)
            summary=dict(reveal_probability=reveal,mask_probability=probability,samples=args.samples,
                completed_seeds=seed+1,requested_seeds=args.seeds,complete=seed+1==args.seeds,
                realized_mask_probability=float(np.mean(realized)),realized_mask_probability_per_seed=realized,
                models={k:summaries(np.stack(vis[k]),np.stack(counts[k]),draws) for k in models},
                paired=paired_summary(np.stack(vis['MDM']),np.stack(vis['A']),np.stack(tables),draws),
                bootstrap_unit='document; all corruption seeds together',bootstrap_draws=args.bootstraps)
            write(folder/'summary.json',summary)
            from analysis.plot_local_top5_pair import refresh
            refresh(folder)
            print(name,'finished seed',seed,'forwards',forwards,'elapsed',round(time.monotonic()-started),flush=True)
        complete_cells.append(name)
    if torch.cuda.is_initialized():raise ValueError('CPU diagnostic initialized CUDA')
    if models['A'].branch_calls!=0:raise ValueError('An auxiliary training route ran')
    write(args.output/'summary.json',dict(stage='complete',cells=complete_cells,model_forwards=forwards,
        requested_forwards=expected,elapsed_seconds=time.monotonic()-started,main_heads_only=True,
        auxiliary_forward_calls=0,cuda_initialized=False,protocol_sha256=digest(args.output/'protocol.json')))
    write(args.output/'progress.json',dict(stage='complete',completed_forwards=forwards,
        requested_forwards=expected,elapsed_seconds=time.monotonic()-started,pid=os.getpid()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--samples',type=int,default=100)
    parser.add_argument('--seeds',type=int,default=10)
    parser.add_argument('--checkpoint-step',type=int,choices=[5000,7500],default=5000)
    parser.add_argument('--reveal-levels',type=float,nargs='+',default=[.25,.5,.75])
    parser.add_argument('--bootstraps',type=int,default=2000)
    args=parser.parse_args();args.output=args.output.resolve()
    if min(args.samples,args.seeds,args.bootstraps)<1 or any(not 0<r<1 for r in args.reveal_levels):
        parser.error('Require positive counts and reveal rates between0 and1')
    if args.output.exists():raise FileExistsError('Preserve earlier/partial runs: '+str(args.output))
    args.output.mkdir(parents=True)
    try:run(args)
    except BaseException as error:
        write(args.output/'progress.json',dict(stage='failed',error=str(error),pid=os.getpid()));raise
