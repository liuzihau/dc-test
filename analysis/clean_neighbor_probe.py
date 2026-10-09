"""Clean-source eligibility, frozen shared features, and four linear probes."""
import numpy as np

OFFSETS=(-2,-1,1,2)
MASK_ID=50257
SPECIAL_IDS=(50256,50257)


def pairs(clean,masked,block_size=1024,special_ids=SPECIAL_IDS):
    clean=np.asarray(clean);masked=np.asarray(masked,dtype=bool)
    if clean.shape!=masked.shape or clean.ndim!=2:raise ValueError('Canvas geometry differs')
    content=~np.isin(clean,special_ids)
    targets=np.full((*clean.shape,4),-1,dtype=np.int32)
    for k,offset in enumerate(OFFSETS):
        start=max(0,-offset);stop=min(clean.shape[1],clean.shape[1]-offset)
        position=np.arange(start,stop);target=position+offset
        valid=np.ones((len(clean),stop-start),dtype=bool)
        for shift in range(min(0,offset),max(0,offset)+1):valid&=content[:,start+shift:stop+shift]
        valid&=(position//block_size==target//block_size)[None,:]
        valid&=~masked[:,position]&masked[:,target]
        targets[:,position,k]=np.where(valid,clean[:,target],-1)
    row,position=np.nonzero((targets>=0).any(-1))
    return row,position,targets[row,position]


def load_backbone(protocol,model_name,step):
    import gc
    import torch
    from omegaconf import OmegaConf
    from types import SimpleNamespace
    from unittest.mock import patch
    from owt.continuation import ROOT,digest
    from owt.model import OWTMDM
    from owt.transformer_np_model import TransformerNPMDM
    record=protocol['checkpoints'][str(step)][model_name]
    path=ROOT/record['path']
    if digest(path)!=record['sha256']:raise ValueError('Checkpoint changed')
    payload=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    config=OmegaConf.load(ROOT/record['config'])
    with patch('diffusion.metrics.Metrics',return_value=torch.nn.Module()):
        tokenizer=SimpleNamespace(vocab_size=50257,mask_token=None,all_special_ids=[50256])
        model=(OWTMDM if model_name=='MDM' else TransformerNPMDM)(config,tokenizer)
    if payload['global_step']!=step or payload['ema']['num_updates']!=step:
        raise ValueError('EMA/checkpoint step mismatch')
    if model_name=='A' and model.resume_signature()!=payload['transformer_np']['signature']:
        raise ValueError('A architecture changed')
    model.load_state_dict(payload['state_dict'],strict=True)
    model.ema.load_state_dict(payload['ema'])
    parameters=list(model._get_parameters());shadow=model.ema.shadow_params
    if len(parameters)!=len(shadow) or any(p.shape!=s.shape for p,s in zip(parameters,shadow)):
        raise ValueError('EMA order differs')
    model.ema.copy_to(parameters)
    if any(not torch.equal(p,s) for p,s in zip(parameters,shadow)):raise ValueError('EMA load differs')
    model.ema=None;model.float().eval().requires_grad_(False)
    model.backbone.force_fp32_eval=True
    if model.time_conditioning or model.cross_attn:raise ValueError('Unexpected backbone geometry')
    del payload,parameters,shadow;gc.collect()
    return model


def shared_features(model,canvas,full_forward=False):
    import torch
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Backbone must be frozen and in evaluation mode')
    captured={}
    class FeatureReady(Exception):pass
    def capture(module,inputs):
        captured['hidden']=inputs[0].detach().clone()
        if not full_forward:raise FeatureReady()
    hook=model.backbone.blocks[-1].register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            try:model(canvas,torch.zeros((len(canvas),1),device=canvas.device,dtype=torch.float32))
            except FeatureReady:pass
    finally:hook.remove()
    h=captured['hidden']
    if h.dtype!=torch.float32 or h.requires_grad:raise ValueError('Feature precision/gradient differs')
    return h


def new_heads(width,vocabulary=50258,seed=0):
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return torch.nn.ModuleList([torch.nn.Linear(width,vocabulary) for _ in OFFSETS])


def logits(head,features,mask_id=MASK_ID):
    result=head(features)
    if mask_id<result.shape[-1]:result[...,mask_id]=-torch_inf()
    return result


def torch_inf():
    return float('inf')


def batch_indices(valid_indices,seed,epoch,batch_size):
    n=max(len(x) for x in valid_indices)
    generators=[np.random.default_rng(np.random.SeedSequence([20261007,seed,epoch,k])) for k in range(4)]
    for start in range(0,n,batch_size):
        size=min(batch_size,n-start)
        yield [indices[g.integers(0,len(indices),size=size)] for g,indices in zip(generators,valid_indices)]


def evaluate(heads,features,targets,documents,levels,source_tokens,batch_size,device,document_count,level_count):
    import torch
    stats=np.zeros((document_count,level_count,4,6),dtype=np.float64)
    heads.eval()
    with torch.inference_mode():
        for k,head in enumerate(heads):
            index=np.flatnonzero(targets[:,k]>=0)
            for start in range(0,len(index),batch_size):
                ix=index[start:start+batch_size]
                h=torch.from_numpy(np.array(features[ix],copy=True)).to(device)
                y=torch.as_tensor(np.asarray(targets[ix,k]),dtype=torch.long,device=device)
                score=logits(head,h)
                loss=torch.nn.functional.cross_entropy(score,y,reduction='none').cpu().numpy()
                correct=(score.argmax(-1)==y).cpu().numpy()
                repeated=np.asarray(source_tokens[ix])==np.asarray(targets[ix,k])
                values=np.column_stack([np.ones(len(ix)),loss,correct,repeated,
                    np.where(repeated,loss,0),np.where(repeated,correct,0)])
                np.add.at(stats[:,:,k,:],(np.asarray(documents[ix]),np.asarray(levels[ix])),values)
    return stats


def macro_loss(stats):
    total=np.asarray(stats).sum((0,1))
    if (total[:,0]==0).any():raise ValueError('An offset has no eligible pairs')
    return float(np.mean(total[:,1]/total[:,0]))
