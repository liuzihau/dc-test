"""Three exact nested teacher states, without a full-mask anchor."""
from dataclasses import dataclass
import torch

@dataclass
class Trajectory:
    states: list
    masks: list
    requested_ratios: torch.Tensor
    realized_ratios: torch.Tensor
    counts: torch.Tensor
    center: torch.Tensor
    spacing: torch.Tensor

def sample_trajectory(clean,eligible,mask_id,k_min=.025,k_max=.10,generator=None):
    if not 0<k_min<=k_max<.5:raise ValueError('Require 0 < k_min <= k_max < .5')
    eligible=eligible.bool();n=eligible.sum(1)
    if (n<4).any():raise ValueError('Three strict non-full states need at least four eligible targets per row')
    batch,length=clean.shape
    def rand(shape):return torch.rand(shape,device=clean.device,generator=generator)
    k=k_min+(k_max-k_min)*rand((batch,))
    t=k+.001+(1-2*k-.002)*rand((batch,))
    requested=torch.stack((t+k,t,t-k),dim=1)
    desired=torch.round(requested*n[:,None]).long()
    high=desired[:,0].clamp(min=3).minimum(n-1)
    mid=desired[:,1].clamp(min=2).minimum(high-1)
    low=desired[:,2].clamp(min=1).minimum(mid-1)
    counts=torch.stack((high,mid,low),dim=1)
    # One ranking defines all three masks; ineligible positions sort last.
    order=rand((batch,length)).masked_fill(~eligible,2.).argsort(dim=1)
    rank=torch.empty_like(order).scatter_(1,order,torch.arange(length,device=clean.device).expand(batch,-1))
    masks=[eligible & (rank<counts[:,j,None]) for j in range(3)]
    states=[torch.where(m,mask_id,clean) for m in masks]
    return Trajectory(states,masks,requested,counts/n[:,None],counts,t,k)
