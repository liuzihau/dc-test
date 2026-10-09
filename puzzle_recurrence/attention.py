"""Extra attention with age/spatial RoPE and canonically indexed raw KV banks."""
from contextlib import contextmanager
from types import MethodType
import math
import torch
from torch import nn
from torch.nn import functional as F


def rope(x,positions,age,spatial_dim):
    """x: B,H,L,D; positions: B,L; previous/current ages: 0/1."""
    def rotate(value,coords):
        width=value.shape[-1]
        frequencies=10000.**(-torch.arange(0,width,2,device=value.device,dtype=torch.float32)/width)
        angles=coords.float()[...,None]*frequencies
        cos=torch.cat((angles.cos(),angles.cos()),dim=-1)[:,None].to(value.dtype)
        sin=torch.cat((angles.sin(),angles.sin()),dim=-1)[:,None].to(value.dtype)
        first,second=value.chunk(2,dim=-1)
        return value*cos+torch.cat((-second,first),dim=-1)*sin
    spatial,temporal=x.split((spatial_dim,x.shape[-1]-spatial_dim),dim=-1)
    return torch.cat((rotate(spatial,positions),rotate(temporal,torch.full_like(positions,age))),dim=-1)


class ScaleNorm(nn.Module):
    def __init__(self,width):super().__init__();self.width=width;self.weight=nn.Parameter(torch.ones(width))
    def forward(self,value):return F.layer_norm(value.float(),(self.width,),self.weight.float())


class MemoryAttention(nn.Module):
    def __init__(self,width,heads,dropout,gate_init=.1):
        super().__init__();self.heads=heads;self.head_dim=width//heads
        temporal=max(2,self.head_dim//4);temporal-=temporal%2
        self.spatial_dim=self.head_dim-temporal
        if self.spatial_dim<2 or self.spatial_dim%2:raise ValueError('RoPE requires two positive even dimensions')
        if not -1<gate_init<1:raise ValueError('Gate initialization must be between -1 and 1')
        self.norm=ScaleNorm(width)
        self.qkv=nn.Linear(width,3*width,bias=False);self.out=nn.Linear(width,width,bias=False)
        self.dropout=nn.Dropout(dropout);self.gate=nn.Parameter(torch.tensor(math.atanh(gate_init)))

    def forward(self,hidden,previous,positions,valid,previous_valid,source_modes):
        b,l,width=hidden.shape
        qkv=self.qkv(self.norm(hidden)).reshape(b,l,3,self.heads,self.head_dim)
        q,k,v=[qkv[:,:,j].transpose(1,2) for j in range(3)]
        entry=qkv[:,:,1:3] # raw, unrotated KV; writer for the preceding depth
        q=rope(q,positions,1,self.spatial_dim);current_k=rope(k,positions,1,self.spatial_dim)
        # Always allocate two banks, including fully masked history slots in the
        # current-only control. Attention tensor shapes match the recurrent arm.
        if previous is None:
            pk=torch.zeros_like(k);pv=torch.zeros_like(v)
            previous_valid=torch.zeros_like(valid)
        else:
            pk=previous[:,:,0].transpose(1,2);pv=previous[:,:,1].transpose(1,2)
            canonical=torch.arange(l,device=hidden.device).expand(b,-1)
            pk=rope(pk,canonical,0,self.spatial_dim)
        keys=torch.cat((pk,current_k),dim=2);values=torch.cat((pv,v),dim=2)
        mask=torch.cat((previous_valid[:,None,:].expand(-1,l,-1),valid[:,None,:].expand(-1,l,-1)),dim=-1)
        if source_modes is not None:
            mask[:,:,:l] &= source_modes.ne(2)[:,:,None]
            mask[:,:,l:] &= source_modes.ne(1)[:,:,None]
        attended=F.scaled_dot_product_attention(q,keys,values,attn_mask=mask[:,None],dropout_p=0.)
        attended=attended.transpose(1,2).reshape(b,l,width)
        residual=self.dropout(self.out(attended))*self.gate.tanh()
        residual=residual.masked_fill(~valid[:,:,None],0)
        return hidden+residual.to(hidden.dtype),entry


class FinalWriter(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.heads=heads
        self.norm=ScaleNorm(width)
        self.kv=nn.Linear(width,2*width,bias=False)
    def forward(self,hidden):
        b,l,w=hidden.shape
        return self.kv(self.norm(hidden)).reshape(b,l,2,self.heads,w//self.heads)


def gather_bank(bank,indices):
    return bank.gather(1,indices[:,:,None,None,None].expand(-1,-1,*bank.shape[2:]))


class MemoryController:
    """Hook the unmodified author blocks; memory is never an input embedding."""
    def __init__(self,backbone,recurrent):
        self.backbone=backbone;self.recurrent=recurrent;self.active=False
        self._handles=[block.register_forward_pre_hook(self._hook(j),with_kwargs=True) for j,block in enumerate(backbone.blocks)]
        self._handles.append(backbone.blocks[-1].register_forward_hook(self._final_hook))
        self._original_sample=backbone.forward_sample
        self.generation_valid=None;self.generation_previous=None;self.generation_calls=0
        controller=self
        def sample(module,zt,sort_idx,*args,**kwargs):
            if kwargs.get('kv_cache',False):raise ValueError('Recurrent puzzle sampling requires kv_cache=False')
            if controller.generation_valid is None:raise RuntimeError('Start a generation session before forward_sample')
            valid=controller.generation_valid.gather(1,sort_idx)
            previous=controller.generation_previous if controller.recurrent else None
            with controller.run(previous,sort_idx,valid,controller.generation_valid):
                result=controller._original_sample(zt,sort_idx,*args,**kwargs)
                controller.generation_previous=controller.banks() if controller.recurrent else None
            controller.generation_calls+=1
            return result
        backbone.forward_sample=MethodType(sample,backbone)

    @contextmanager
    def run(self,previous,positions,valid,previous_valid=None,modes=None):
        if self.active:raise RuntimeError('Memory contexts cannot nest')
        if positions.shape!=valid.shape:raise ValueError('Positions and validity differ')
        expected=torch.arange(positions.shape[1],device=positions.device).expand_as(positions)
        if not torch.equal(positions.sort(dim=1).values,expected):raise ValueError('Memory requires complete original position IDs')
        self.active=True;self.previous=previous;self.positions=positions;self.valid=valid.bool()
        self.previous_valid=previous_valid.bool() if previous_valid is not None else self.valid.gather(1,positions.argsort(1))
        self.modes=modes;self.entries=[];self.final=None
        try:yield self
        finally:
            self.active=False;self.previous=None;self.entries=[];self.final=None

    def _hook(self,index):
        def before(block,args,kwargs):
            if not self.active:raise RuntimeError('Extra attention must run within an explicit memory context')
            hidden,entry=self.backbone.memory_attention[index](args[0],
                self.previous[index] if self.previous is not None and self.recurrent else None,
                self.positions,self.valid,self.previous_valid,self.modes)
            self.entries.append(entry)
            return (hidden,*args[1:]),kwargs
        return before

    def _final_hook(self,block,args,output):
        if self.active:self.final=output

    def banks(self):
        if self.final is None or len(self.entries)!=len(self.backbone.blocks):raise RuntimeError('Incomplete memory forward')
        inverse=self.positions.argsort(1)
        entries=self.entries[1:]+[self.backbone.memory_final_writer(self.final)]
        canonical_valid=self.valid.gather(1,inverse)
        return [gather_bank(e,inverse).masked_fill(~canonical_valid[:,:,None,None,None],0) for e in entries]

    def parameter_anchor(self):
        # The current-only control includes the identical shifted final writer.
        # Its unused parameters stay in the DDP graph with exactly zero gradient.
        return sum(p.reshape(-1)[0]*0 for p in list(self.backbone.memory_attention.parameters())+
            list(self.backbone.memory_final_writer.parameters()) if p.requires_grad)
