"""GPU worker for frozen-feature caching, linear probe fitting and evaluation."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import time

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/runtime/clean-probe-mpl'))
import numpy as np
from analysis.clean_neighbor_probe import pairs,load_backbone,shared_features,new_heads,logits,batch_indices,evaluate,macro_loss
from analysis.local_denoising_metrics import make_canvas
from owt.continuation import ROOT,digest
from owt.research import atomic_write,timestamp


def write(path,data):atomic_write(path,json.dumps(data,indent=2,allow_nan=False)+'\n')


def verify_protocol(p):
    for filename,sha in p['source_sha256'].items():
        if digest(ROOT/filename)!=sha:raise ValueError('Probe source changed: '+filename)
    if digest(ROOT/p['data_path'])!=p['data_sha256']:raise ValueError('Probe data changed')


def open_cache(run,split):
    path=run/'cache'/split
    receipt=json.loads((path/'complete.json').read_text())
    result={name:np.load(path/(name+'.npy'),mmap_mode='r',allow_pickle=False)
        for name in ('features','targets','documents','levels','source_tokens')}
    for name in ('targets','documents','levels','source_tokens'):
        if digest(path/(name+'.npy'))!=receipt['files'][name]:raise ValueError('Cache metadata changed')
    if result['features'].shape!=(receipt['sources'],receipt['width']):raise ValueError('Feature shape differs')
    return result,receipt


def cache(p,name,step,run,device):
    import torch
    data=np.load(ROOT/p['data_path'],allow_pickle=False)
    model=load_backbone(p,name,step).to(device)
    if any(parameter.requires_grad for parameter in model.parameters()):raise ValueError('Backbone is not frozen')
    def reject(*args):raise ValueError('A trained auxiliary branch ran')
    if name=='A':
        for branch in model.backbone.neighbor_branches:branch.register_forward_pre_hook(reject)
    audit_input=torch.as_tensor(data['development'][:1],dtype=torch.long,device=device)
    audit_input=audit_input.clone();audit_input[:,100:104]=50257
    full=shared_features(model,audit_input,full_forward=True)
    early=shared_features(model,audit_input)
    if not torch.equal(full,early):raise ValueError('Early feature capture differs from the normal main path')
    del full,early
    from owt.continuation import tensor_fingerprint
    before=tensor_fingerprint(sorted(model.state_dict().items()))
    for split in ('train','development','evaluation'):
        path=run/'cache'/split
        if path.exists():raise FileExistsError('Preserve partial cache for inspection')
        path.mkdir(parents=True)
        clean=data[split];ids=data[split+'_ids'];layouts=[];n=0
        for level,rate in enumerate(p['reveal_rates']):
            for seed in p['corruption_seeds'][split]:
                canvas,masked=make_canvas(clean,ids,seed,1-rate)
                for doc in range(len(clean)):
                    _,pos,target=pairs(clean[doc:doc+1],masked[doc:doc+1])
                    layouts.append((level,seed,doc,canvas[doc],pos,target));n+=len(pos)
        width=int(model.config.model.hidden_size)
        arrays=dict(features=np.lib.format.open_memmap(path/'features.npy',mode='w+',dtype=np.float32,shape=(n,width)),
            targets=np.lib.format.open_memmap(path/'targets.npy',mode='w+',dtype=np.int32,shape=(n,4)),
            documents=np.lib.format.open_memmap(path/'documents.npy',mode='w+',dtype=np.int32,shape=(n,)),
            levels=np.lib.format.open_memmap(path/'levels.npy',mode='w+',dtype=np.int8,shape=(n,)),
            source_tokens=np.lib.format.open_memmap(path/'source_tokens.npy',mode='w+',dtype=np.int32,shape=(n,)))
        cursor=0
        for number,(level,seed,doc,canvas,pos,target) in enumerate(layouts):
            x=torch.as_tensor(canvas[None],dtype=torch.long,device=device)
            hidden=shared_features(model,x)
            values=hidden[0,torch.as_tensor(pos,device=device)].cpu().numpy()
            end=cursor+len(pos)
            arrays['features'][cursor:end]=values;arrays['targets'][cursor:end]=target
            arrays['documents'][cursor:end]=doc;arrays['levels'][cursor:end]=level
            arrays['source_tokens'][cursor:end]=clean[doc,pos];cursor=end
            if number%10==0 or number+1==len(layouts):
                write(run/'progress.json',dict(stage='extracting_frozen_features',split=split,
                    completed_canvases=number+1,requested_canvases=len(layouts),sources=cursor,pid=os.getpid()))
        for array in arrays.values():array.flush()
        if cursor!=n or not np.isfinite(arrays['features']).all():raise ValueError('Incomplete/nonfinite features')
        counts=(arrays['targets']>=0).sum(0).tolist()
        del arrays;gc.collect()
        files={key:digest(path/(key+'.npy')) for key in ('features','targets','documents','levels','source_tokens')}
        write(path/'complete.json',dict(sources=n,width=width,documents=len(clean),document_ids=ids.tolist(),
            offset_pair_counts=counts,files=files,parameter_state='EMA',precision='FP32',layer='before last main block',
            full_forward_feature_equivalence=True,backbone_frozen=True,auxiliary_branches_used=False))
    after=tensor_fingerprint(sorted(model.state_dict().items()))
    if before!=after:raise ValueError('Frozen backbone changed during extraction')
    write(run/'cache_complete.json',dict(backbone_before=before,backbone_after=after,
        checkpoint=p['checkpoints'][str(step)][name],strict_ema_load=True,full_forward_feature_equivalence=True))
    write(run/'progress.json',dict(stage='features_complete',pid=os.getpid()))


def score_cache(heads,run,split,p,device):
    c,r=open_cache(run,split)
    return evaluate(heads,c['features'],c['targets'],c['documents'],c['levels'],c['source_tokens'],
        p['batch_size'],device,r['documents'],len(p['reveal_rates']))


def fit(p,run,device,lr,seed,tag):
    import torch
    c,r=open_cache(run,'train');path=run/'fits'/tag
    if path.exists():raise FileExistsError('Preserve partial probe fit')
    path.mkdir(parents=True)
    heads=new_heads(r['width'],seed=seed).to(device)
    optimizer=torch.optim.AdamW(heads.parameters(),lr=lr,weight_decay=p['weight_decay'])
    valid=[np.flatnonzero(c['targets'][:,k]>=0) for k in range(4)]
    if any(len(x)==0 for x in valid):raise ValueError('Empty fitting population')
    initial={key:value.detach().clone() for key,value in heads.state_dict().items()}
    traces=[];best=float('inf');bad=0;started=time.monotonic()
    for epoch in range(p['max_epochs']):
        heads.train();loss_sum=np.zeros(4);steps=0
        for indices in batch_indices(valid,seed,epoch,p['batch_size']):
            optimizer.zero_grad(set_to_none=True);losses=[]
            if len(traces)<3:traces.append([hashlib.sha256(ix.tobytes()).hexdigest() for ix in indices])
            for k,ix in enumerate(indices):
                h=torch.from_numpy(np.array(c['features'][ix],copy=True)).to(device)
                y=torch.as_tensor(np.asarray(c['targets'][ix,k]),dtype=torch.long,device=device)
                loss=torch.nn.functional.cross_entropy(logits(heads[k],h),y)
                if not torch.isfinite(loss):raise ValueError('Nonfinite probe fitting loss')
                (loss/4).backward();losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(heads.parameters(),1.0)
            optimizer.step();loss_sum+=losses;steps+=1
            if steps%100==0:write(run/'progress.json',dict(stage='fitting_heads',tag=tag,seed=seed,
                epoch=epoch+1,updates=steps,pid=os.getpid()))
        development=score_cache(heads,run,'development',p,device);value=macro_loss(development)
        row=dict(epoch=epoch+1,development_ce=value,training_ce=float((loss_sum/steps).mean()),
            elapsed_seconds=time.monotonic()-started)
        with (path/'learning_curve.csv').open('a',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(row))
            if epoch==0:writer.writeheader()
            writer.writerow(row)
        if value<best-p['early_stop_min_delta']:
            best=value;bad=0
            torch.save(dict(heads={k:v.detach().cpu() for k,v in heads.state_dict().items()},
                epoch=epoch+1,development_ce=value,seed=seed,lr=lr,width=r['width'],offsets=[-2,-1,1,2]),path/'best.pt')
        else:bad+=1
        write(run/'progress.json',dict(stage='fitting_heads',tag=tag,seed=seed,epoch=epoch+1,
            development_ce=value,best_development_ce=best,pid=os.getpid()))
        print('probe',tag,'epoch',epoch+1,'development CE',value,flush=True)
        if bad>=p['early_stop_patience']:break
    if not any(not torch.equal(initial[k],v) for k,v in heads.state_dict().items()):
        raise ValueError('Probe weights did not update')
    del initial
    result=dict(best_development_ce=best,epochs=epoch+1,seed=seed,lr=lr,sampler_first_batches=traces,
        only_new_linear_heads_trained=True,feature_cache_sha256=r['files']['features'],
        max_epochs_reached=epoch+1==p['max_epochs'],best_checkpoint=str(path/'best.pt'))
    write(path/'complete.json',result)
    return path


def execute(p,name,step,run,device,action,seed):
    import torch
    if action=='cache':return cache(p,name,step,run,device)
    if action=='calibrate':
        for lr in p['learning_rates']:fit(p,run,device,lr,0,'calibration-'+format(lr,'.0e'))
        return
    selection=json.loads((Path(p['output'])/'selected_hyperparameters.json').read_text())
    lr=selection['learning_rate'];tag='seed-'+str(seed)
    if step==p['checkpoint_steps'][0] and seed==0:
        source=run/'fits'/('calibration-'+format(lr,'.0e'))
        import shutil
        dest=run/'fits'/tag
        if dest.exists():raise FileExistsError('Preserve completed/partial fit')
        shutil.copytree(source,dest)
        fitted=dest
    else:fitted=fit(p,run,device,lr,seed,tag)
    payload=torch.load(fitted/'best.pt',map_location='cpu',weights_only=True)
    heads=new_heads(payload['width'],seed=seed).to(device);heads.load_state_dict(payload['heads'],strict=True)
    stats=score_cache(heads,run,'evaluation',p,device)
    np.save(fitted/'evaluation_stats.npy',stats)
    write(fitted/'evaluation_complete.json',dict(seed=seed,lr=lr,macro_ce=macro_loss(stats),
        documents=p['documents']['evaluation'],corruption_seeds=p['corruption_seeds']['evaluation'],
        stats_sha256=digest(fitted/'evaluation_stats.npy'),development_best_epoch=payload['epoch']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--model',choices=['MDM','A'],required=True)
    parser.add_argument('--step',type=int,choices=[5000,7500],required=True)
    parser.add_argument('--action',choices=['cache','calibrate','fit'],required=True)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();p=json.loads(args.protocol.read_text());verify_protocol(p)
    if os.environ.get('CLEAN_PROBE_GPU_LOCK_HELD')!=str(args.protocol.resolve()):raise ValueError('Use the GPU-lock controller')
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2','3') or torch.cuda.device_count()!=1:
        raise ValueError('One authorized physical GPU per worker is required')
    device='cuda:0';run=Path(p['output'])/f'step{args.step}'/args.model;run.mkdir(parents=True,exist_ok=True)
    try:execute(p,args.model,args.step,run,device,args.action,args.seed)
    except BaseException as error:
        write(run/'progress.json',dict(stage='failed',error=str(error),pid=os.getpid()));raise
