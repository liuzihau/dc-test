"""Three-state attention control and one-hop DCache on the author puzzle model."""
from contextlib import contextmanager
import copy
import torch
from torch import nn
from models.ema import ExponentialMovingAverage
from trainer_base import Loss
from zebra.model import ZebraMDM
from puzzle_recurrence.attention import MemoryAttention,FinalWriter,MemoryController
from puzzle_recurrence.adjacent import AdjacentCacheGradients
from puzzle_recurrence.trajectory import sample_trajectory

SCHEMA='puzzle_three_state_adjacent_kv_v1'

@contextmanager
def preserve_rng():
    cpu=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:yield
    finally:
        torch.set_rng_state(cpu)
        if cuda:torch.cuda.set_rng_state_all(cuda)


class PuzzleTrajectoryMDM(ZebraMDM):
    def __init__(self,config,tokenizer):
        from puzzle_recurrence.settings import validate_config
        validate_config(config)
        from puzzle_recurrence.runtime import retain_compiled_backward_buffers
        retain_compiled_backward_buffers()
        if config.mechanisms.np.enabled:raise ValueError('NP + recurrence is deferred; first run the separate ablations')
        cfg=config.puzzle_recurrence
        if cfg.schema!=SCHEMA or list(cfg.loss_weights)!=[.25,1.,.25]:raise ValueError('Require the selected normalized three-state objective')
        if cfg.final_hidden_feedback or cfg.identity_loss:raise ValueError('This first comparison has no final-hidden or identity route')
        if cfg.gradient_horizon!=1:raise ValueError('Use exactly one temporal gradient hop')
        if not config.mechanisms.tt.enabled or not config.mechanisms.ea.enabled:raise ValueError('Trajectory arms require TT and EA')
        if config.sampling.kv_cache or config.sampling.get('trim_masked_tokens',False) or config.sampling.get('n_latent_tokens',0):
            raise ValueError('Use full-sequence author sampling without prefix cache, trimming, or latent tokens')
        if config.algo.alpha_0!=1. or config.objective.kind!='elbo':raise ValueError('Preserve author alpha_0=1 ELBO')
        # Construct the unchanged shared backbone through the existing adapter.
        baseline=copy.deepcopy(config)
        for key in ('tt','ea','rm'):baseline.mechanisms[key].enabled=False
        super().__init__(baseline,tokenizer)
        self.config=config;self.hparams['config']=config;self.objective=config.objective;self.np_config=config.mechanisms.np
        self.recurrent=bool(config.mechanisms.rm.enabled)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(config.seed)+810013)
            self.backbone.memory_attention=nn.ModuleList([MemoryAttention(config.model.hidden_size,
                config.model.n_heads,config.model.dropout,cfg.gate_init) for _ in self.backbone.blocks])
            self.backbone.memory_final_writer=FinalWriter(config.model.hidden_size,config.model.n_heads)
        self.memory=MemoryController(self.backbone,self.recurrent)
        if self.ema:self.ema=ExponentialMovingAverage(self._get_parameters(),decay=config.training.ema)
        self._forward_valid=None;self._last_trajectory=None;self._last_components=None;self._debug_banks=[]

    def resume_signature(self):
        from omegaconf import OmegaConf
        return dict(schema=SCHEMA,task=self.config.data.name,recurrent=self.recurrent,
            settings=OmegaConf.to_container(self.config.puzzle_recurrence,resolve=True),
            objective=OmegaConf.to_container(self.config.objective,resolve=True),
            model=OmegaConf.to_container(self.config.model,resolve=True),
            train_on_all_tokens=bool(self.config.training.train_on_all_tokens))

    def on_save_checkpoint(self,checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint['puzzle_recurrence_signature']=self.resume_signature()

    def on_load_checkpoint(self,checkpoint):
        if checkpoint.get('puzzle_recurrence_signature')!=self.resume_signature():
            raise ValueError('Resume requires the same task, trajectory, objective and memory policy')
        super().on_load_checkpoint(checkpoint)

    def forward(self,xt,sigma,sort_idx=None,x0=None,mask_cutoffs=None):
        if self.memory.active:return super().forward(xt,sigma,sort_idx,x0,mask_cutoffs)
        positions=sort_idx if sort_idx is not None else torch.arange(xt.shape[1],device=xt.device).expand_as(xt)
        valid=self._forward_valid if self._forward_valid is not None else xt.ne(self.pad_index)
        # A normal single-canvas validation is always reset, with no history.
        with self.memory.run(None,positions,valid):return super().forward(xt,sigma,sort_idx,x0,mask_cutoffs)

    def _source_modes(self,masked,state_index):
        modes=torch.zeros_like(masked,dtype=torch.long)
        cfg=self.config.puzzle_recurrence.source_dropout
        if not self.training or not cfg.enabled or state_index==0:return modes
        ramp=min(1.,float(self.global_step)/max(1,int(cfg.warmup_steps)))
        previous_only=float(cfg.previous_only_probability)*ramp
        current_only=float(cfg.current_only_probability)
        # Same source-policy draws in both matched arms; no additional model RNG.
        draws=torch.rand(masked.shape,device=masked.device)
        modes[masked & (draws<previous_only)]=1
        modes[masked & (draws>=previous_only) & (draws<previous_only+current_only)]=2
        return modes

    def _trajectory_loss(self,x0,valid_tokens,loss_mask=None):
        effective=valid_tokens if loss_mask is None else valid_tokens*loss_mask
        if ((x0==self.mask_index)&effective.bool()).any():raise ValueError('Clean targets contain the reserved MASK class')
        trajectory=sample_trajectory(x0,effective.bool(),self.mask_index,
            self.config.puzzle_recurrence.k_min,self.config.puzzle_recurrence.k_max)
        denominator=effective.sum();positions=torch.arange(x0.shape[1],device=x0.device).expand_as(x0)
        previous=None;bridge=AdjacentCacheGradients(enabled=self.recurrent and self.training and torch.is_grad_enabled())
        losses=[];accuracies=[];ces=[];correct_counts=[];target_counts=[];banks=[];modes=[]
        for j,(state,masked) in enumerate(zip(trajectory.states,trajectory.masks)):
            ratio=trajectory.realized_ratios[:,j:j+1]
            # Author noise has p(mask)=.999*time. Recover time from realized count.
            time=ratio/(1-self.noise.eps)
            dalpha,alpha=self.noise(time)
            source_modes=self._source_modes(masked,j);modes.append(source_modes)
            with self.memory.run(previous,positions,valid_tokens.bool(),valid_tokens.bool(),source_modes):
                scores=self.forward(state,self._sigma_from_alphat(alpha))
                raw=self.memory.banks()
            if self.config.puzzle_recurrence.get('debug_graph',False):
                for value in raw:
                    if value.requires_grad:value.retain_grad()
                banks.append(raw)
            if j<2:previous=bridge.consume(raw) if self.recurrent else None
            per_token=self.nll_per_token(scores,state,x0,alpha,dalpha,low_var=False,train_mode=True)
            loss=(per_token*effective).sum()/denominator.clamp_min(1);losses.append(loss)
            target=masked & effective.bool()
            ce=-scores.gather(-1,x0[:,:,None]).squeeze(-1)
            ces.append((ce*target).sum()/target.sum().clamp_min(1))
            correct=((scores.argmax(-1)==x0)&target).sum();count=target.sum()
            correct_counts.append(correct);target_counts.append(count)
            accuracies.append(correct/count.clamp_min(1))
        weights=list(self.config.puzzle_recurrence.loss_weights);normalizer=sum(weights)
        main=sum(w*l for w,l in zip(weights,losses))/normalizer
        objective=main+self.memory.parameter_anchor()
        objective=bridge.attach(objective)
        self._last_trajectory=dict(requested=trajectory.requested_ratios.detach(),realized=trajectory.realized_ratios.detach(),
            counts=trajectory.counts.detach(),eligible=effective.sum(1).detach(),
            losses=torch.stack(losses).detach(),accuracy=torch.stack(accuracies).detach(),ce=torch.stack(ces).detach(),
            correct=torch.stack(correct_counts).detach(),targets=torch.stack(target_counts).detach(),
            modes=[m.detach() for m in modes],gradient_edges=bridge.num_edges,
            state_order=['t+k','t','t-k'],identity_forwards=0,main_forwards=3)
        self._last_components=dict(current=main.detach(),current_elbo=main.detach(),objective=objective.detach(),
            np_prev_1=main.detach()*0,np_next_1=main.detach()*0)
        self._debug_banks=banks
        if self._trainer is not None:
            for key,value in self._last_components.items():self.log('components/'+key,value,on_step=True,on_epoch=False,sync_dist=True)
        return Loss(objective,main.detach()*denominator,x0.new_zeros((),dtype=torch.float32),denominator)

    def _loss(self,x0,valid_tokens,current_accumulation_step=None,train_mode=False,loss_mask=None):
        if train_mode:return self._trajectory_loss(x0,valid_tokens,loss_mask)
        old=self._forward_valid;self._forward_valid=valid_tokens.bool()
        try:return super()._loss(x0,valid_tokens,current_accumulation_step,False,loss_mask)
        finally:self._forward_valid=old

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._validation_correct=torch.zeros(3,device=self.device,dtype=torch.float64)
        self._validation_targets=torch.zeros_like(self._validation_correct)

    def validation_step(self,batch,batch_idx):
        loss=super().validation_step(batch,batch_idx)
        # Report conditional trajectory diagnostics separately from author ELBO.
        with preserve_rng():
            diagnostic=self._trajectory_loss(batch['input_ids'],batch['attention_mask'],batch.get('loss_mask'))
        self.log('val/trajectory_objective',diagnostic.loss,on_step=False,on_epoch=True,sync_dist=True,
            batch_size=batch['input_ids'].shape[0])
        self._validation_correct+=self._last_trajectory['correct']
        self._validation_targets+=self._last_trajectory['targets']
        return loss

    def on_validation_epoch_end(self):
        correct=self._validation_correct.clone();counts=self._validation_targets.clone()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(correct);torch.distributed.all_reduce(counts)
        for j,name in enumerate(('high','center','low')):
            self.log('val/trajectory_'+name+'_accuracy',correct[j]/counts[j].clamp_min(1),
                on_step=False,on_epoch=True,sync_dist=False)
        super().on_validation_epoch_end()

    def generate_completions(self,completion_batch,*args,**kwargs):
        inputs=completion_batch['input_ids'].to(self.device)
        valid=completion_batch.get('attention_mask',inputs.ne(self.pad_index)).to(self.device).bool()
        self.memory.generation_valid=valid;self.memory.generation_previous=None;self.memory.generation_calls=0
        try:return super().generate_completions(completion_batch,*args,**kwargs)
        finally:
            self.memory.generation_valid=None;self.memory.generation_previous=None
