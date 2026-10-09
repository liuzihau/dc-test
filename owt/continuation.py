"""Contracts for the user-selected MDM/A 5000-to-7500 continuations."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = 'mdm_np_zero_init_transformer_masked_source'
DISTANCE = 'mdm_np_zero_init_transformer_distance2_masked_source'
RUN_ROOT = Path('outputs/owt/continuation-7500')
ORIGINALS = {'mdm': Path('outputs/owt/mdm-np-5k/mdm'),
             A: Path('outputs/owt/transformer-np-5k') / A}
FILES = ('owt/continuation.py', 'owt/continuation_entrypoint.py',
         'owt/continuation_metrics.py', 'owt/continuation_schedule.py',
         'owt/test_continuation.py')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_selection(path, verify_checkpoints=False):
    selection = json.loads(Path(path).read_text())
    if (selection.get('execution_ready') is not True or not selection.get('user_instruction')
            or selection.get('order') != ['mdm', A] or selection.get('resume_step') != 5000
            or selection.get('target_step') != 7500 or selection.get('continuation_seed') != 750001
            or selection.get('wait_for_variant') != DISTANCE
            or selection.get('run_root') != str(RUN_ROOT)):
        raise ValueError('Require the explicit MDM/A continuation selection')
    legacy = json.loads((ROOT/'outputs/research-notes/transformer_np_distance2_selection_20261005.json').read_text())
    for filename, expected in legacy['source_sha256'].items():
        if selection['source_sha256'].get(filename) != expected or digest(ROOT/filename) != expected:
            raise ValueError('Existing training dependency changed: '+filename)
    for filename, expected in legacy['evidence_sha256'].items():
        if digest(ROOT/filename) != expected:
            raise ValueError('Active distance-two evidence changed: '+filename)
    if not set(FILES).issubset(selection['source_sha256']):
        raise ValueError('Continuation implementation is not pinned')
    for filename, expected in selection['source_sha256'].items():
        if digest(ROOT/filename) != expected:
            raise ValueError('Continuation source changed: '+filename)
    for filename, expected in selection['evidence_sha256'].items():
        if digest(ROOT/filename) != expected:
            raise ValueError('Continuation reference changed: '+filename)
    for variant, original in ORIGINALS.items():
        record = selection['checkpoints'][variant]
        if record['path'] != str(original/'checkpoints/step-0005000.ckpt'):
            raise ValueError('Resume must use the original step5000 checkpoint')
        p=ROOT/record['path']; stat=p.stat()
        if stat.st_size != record['bytes'] or stat.st_mtime_ns != record['mtime_ns']:
            raise ValueError('Original checkpoint size/time changed')
        if verify_checkpoints and digest(p) != record['sha256']:
            raise ValueError('Original checkpoint hash changed')
    return selection


def continuation_config(saved, run, checkpoint):
    from omegaconf import OmegaConf
    import torch
    # Checkpoints retain Hydra interpolations. A fresh resume process must
    # register the original entrypoint's resolvers before resolving them.
    for name, fn in [('cwd', lambda: str(ROOT)), ('device_count', torch.cuda.device_count),
                     ('eval', eval), ('div_up', lambda x, y: (x+y-1)//y)]:
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, fn)
    config = OmegaConf.create(OmegaConf.to_container(saved, resolve=True))
    OmegaConf.set_struct(config, False)
    config.trainer.max_steps = 7500
    config.checkpointing.save_dir = str(Path(run).resolve())
    config.checkpointing.resume_from_ckpt = True
    config.checkpointing.resume_ckpt_path = str(Path(checkpoint).resolve())
    return config


def inspect_checkpoint(payload, variant):
    if payload.get('global_step') != 5000:
        raise ValueError('Resume checkpoint is not step5000')
    config = payload['hyper_parameters']['config']
    if (config['loader']['batch_size'] != 8 or config['loader']['global_batch_size'] != 512
            or config['trainer']['devices'] != 2 or config['trainer']['accumulate_grad_batches'] != 32
            or config['model']['length'] != 1024 or config['seed'] != 1):
        raise ValueError('Reference batch/model settings changed')
    if config['lr_scheduler'] != {'_target_':'transformers.get_constant_schedule_with_warmup', 'num_warmup_steps':2500}:
        raise ValueError('Unexpected original scheduler')
    optimizer = payload['optimizer_states']
    schedulers = payload['lr_schedulers']
    if len(optimizer)!=1 or len(schedulers)!=1 or schedulers[0]['last_epoch']!=5000:
        raise ValueError('Missing original optimizer or scheduler state')
    if any(group['lr']!=.0003 for group in optimizer[0]['param_groups']):
        raise ValueError('Original learning rate differs from0.0003')
    steps = {int(state['step']) for state in optimizer[0]['state'].values()}
    if steps!={5000} or any(not {'exp_avg','exp_avg_sq'}.issubset(state) for state in optimizer[0]['state'].values()):
        raise ValueError('Incomplete step5000 Adam moments')
    if payload['ema']['num_updates']!=5000:
        raise ValueError('EMA is not aligned to step5000')
    fit=payload['loops']['fit_loop']
    batches=fit['epoch_loop.batch_progress']['current']['completed']
    epoch=fit['epoch_progress']['current']['completed']
    if batches!=160000 or epoch!=0:
        raise ValueError('Expected 5000 x32 microbatches in epoch0')
    np_config=config['mechanisms']['np']
    if variant=='mdm':
        if np_config['enabled'] or 'transformer_np' in payload:
            raise ValueError('MDM reference must have no NP')
    elif variant==A:
        if (not np_config['enabled'] or np_config['offsets']!=[-1,1]
                or np_config['weights']!=[.25,.25] or np_config['source_policy']!='masked_source'
                or payload['transformer_np']['calls']!=160000):
            raise ValueError('A architecture/loss/private RNG changed')
    else:
        raise ValueError('Unknown continuation variant')
    return dict(global_step=5000,epoch=epoch,microbatches=batches,
                sampler_rows_per_rank=batches*8,learning_rate=.0003,
                scheduler_last_epoch=5000,ema_updates=5000,
                optimizer_parameters=len(optimizer[0]['state']),
                rng_policy='matched deterministic new per-rank stream; original per-rank RNG not saved')


def tensor_fingerprint(tensors):
    """Lightweight restoration audit; strict checkpoint loading checks full keys/shapes."""
    import torch
    h=hashlib.sha256()
    for name,tensor in tensors:
        if not torch.is_tensor(tensor):
            h.update(str((name,tensor)).encode()); continue
        h.update(str((name,tuple(tensor.shape),str(tensor.dtype))).encode())
        flat=tensor.detach().reshape(-1)
        if flat.numel():
            sample=flat[[0,flat.numel()//2,flat.numel()-1]].float().cpu().numpy()
            h.update(sample.tobytes())
    return h.hexdigest()


def optimizer_fingerprint(state):
    tensors=[]
    for index,values in sorted(state['state'].items()):
        for key,value in sorted(values.items()):
            tensors.append((f'{index}:{key}',value))
    return tensor_fingerprint(tensors)
