"""Fixed experiment contracts for Sudoku and Zebra, with no automatic training."""
VARIANTS=('mdm','mdm_np','trajectory_attention','trajectory_recurrent')
SCHEMA='puzzle_three_state_adjacent_kv_v1'

def settings():
    return dict(schema=SCHEMA,loss_weights=[.25,1.,.25],k_min=.025,k_max=.10,gate_init=.1,
        gradient_horizon=1,final_hidden_feedback=False,identity_loss=False,debug_graph=False,
        source_dropout=dict(enabled=True,previous_only_probability=.20,current_only_probability=.05,warmup_steps=1000))

def configure(config,variant):
    from omegaconf import OmegaConf
    if variant not in VARIANTS:raise ValueError('Unknown puzzle ablation')
    OmegaConf.set_struct(config,False)
    trajectory=variant.startswith('trajectory_')
    config.mechanisms.tt.enabled=trajectory;config.mechanisms.ea.enabled=trajectory
    config.mechanisms.rm.enabled=variant=='trajectory_recurrent';config.mechanisms.np.enabled=variant=='mdm_np'
    config.puzzle_recurrence=settings()
    config.puzzle_recurrence_variant=variant
    config.sampling.trim_masked_tokens=False;config.sampling.kv_cache=False
    config.trainer.accumulate_grad_batches=config.loader.global_batch_size//(config.trainer.devices*config.loader.batch_size)
    return config

def validate_config(config):
    m=config.mechanisms;c=config.puzzle_recurrence
    if c.schema!=SCHEMA or list(c.loss_weights)!=[.25,1.,.25]:raise ValueError('Require three state weights 0.25,1,0.25')
    if c.final_hidden_feedback or c.identity_loss or c.gradient_horizon!=1:raise ValueError('Require KV-only, identity-free, one-hop recurrence')
    if not 0<c.k_min<=c.k_max<.5:raise ValueError('Invalid trajectory spacing')
    if not -1<c.gate_init<1:raise ValueError('Invalid residual gate')
    d=c.source_dropout
    if min(d.previous_only_probability,d.current_only_probability)<0 or d.previous_only_probability+d.current_only_probability>1:
        raise ValueError('Invalid source dropout probabilities')
    if d.warmup_steps<0:raise ValueError('Invalid source warmup')
    if bool(m.tt.enabled)!=bool(m.ea.enabled) or (m.rm.enabled and not m.tt.enabled):raise ValueError('Attention and trajectory must be paired')
    if m.np.enabled and m.tt.enabled:raise ValueError('Merged NP and RM training is deferred')
