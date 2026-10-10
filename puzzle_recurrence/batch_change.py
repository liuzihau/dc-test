"""Explicit epoch-boundary device/microbatch changes with fixed global batch."""
import copy

def prepare_batch_change(checkpoint,current,allowed=False,allow_device_change=False):
    metadata=checkpoint['puzzle_data_cursor'];old=metadata['batch_policy']
    if old==current:return checkpoint,None
    if not (allowed or allow_device_change):raise ValueError('Use --allow-microbatch-change for an explicit epoch-boundary change')
    if old['global_batch']!=current['global_batch']:
        raise ValueError('Batch migration requires unchanged global batch')
    resized=old['devices']!=current['devices']
    if resized and not allow_device_change:raise ValueError('Device count change requires --allow-device-change')
    cursor=metadata['cursor']
    if cursor['rows'] or cursor['batches']:raise ValueError('Microbatch change requires a completed data epoch')
    for policy in (old,current):
        if policy['batch']*policy['devices']*policy['accumulation']!=policy['global_batch']:
            raise ValueError('Inconsistent effective batch policy')
    result=dict(checkpoint);hyper=dict(checkpoint['hyper_parameters']);config=copy.deepcopy(hyper['config'])
    # The old Zebra adapter has a different batch-counter migration. Our
    # consumed cursor is already normalized to zero at an epoch boundary.
    # Bypass that older migration; preserve total actual microbatch counters.
    if isinstance(config,dict):
        config['loader']['batch_size']=current['batch'];config['trainer']['accumulate_grad_batches']=current['accumulation']
        config['trainer']['devices']=current['devices']
    else:
        config.loader.batch_size=current['batch'];config.trainer.accumulate_grad_batches=current['accumulation']
        config.trainer.devices=current['devices']
    hyper['config']=config;result['hyper_parameters']=hyper
    receipt=dict(step=int(checkpoint['global_step']),epoch=int(cursor['epoch']),old_policy=old,new_policy=current,
        optimizer_ema_scheduler_preserved=True,global_batch_preserved=True,device_count_changed=resized,
        rng_policy='Existing rank RNG restored; added ranks receive deterministic independent streams' if resized else 'Existing rank RNG restored',
        corruption_stream='RNG restored; grouping changes, so subsequent draws need not match the old microbatch run')
    return result,receipt
