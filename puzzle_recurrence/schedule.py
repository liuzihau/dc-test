"""Three-epoch milestones and trusted checkpoint planning for paired runs."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
TASKS={'sudoku':dict(rows=1804463,steps_per_epoch=3525,epochs=20),
       'zebra':dict(rows=1499933,steps_per_epoch=2930,epochs=40)}

def milestones(epochs,every=3):
    if epochs<1 or every<1:raise ValueError('Epochs and evaluation interval must be positive')
    return sorted(set(range(every,epochs+1,every))|{epochs})

def checkpoint_info(path):
    author=ROOT/'third_party/reasoning_with_latent_tokens'
    if str(author) not in sys.path:sys.path.insert(0,str(author))
    import torch
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    metadata=payload['puzzle_data_cursor']
    return dict(path=str(Path(path).resolve()),step=int(payload['global_step']),cursor=metadata['cursor'],
        batch_policy=metadata['batch_policy'],variant=metadata['variant'],task=metadata.get('task'))

def checkpoints(run):
    paths={p.resolve() for p in (Path(run)/'checkpoints').glob('*.ckpt')}
    # A validation-best checkpoint can have the same update as the periodic
    # resume checkpoint. Prefer the periodic file, with deterministic ties.
    return sorted((checkpoint_info(p) for p in paths),
        key=lambda r:(r['step'],not(r['cursor']['rows'] or r['cursor']['batches']),Path(r['path']).name!='best.ckpt',r['path']))

def latest_checkpoint(run):
    records=checkpoints(run)
    return records[-1] if records else None

def checkpoint_at(run,step):
    return next((r for r in reversed(checkpoints(run)) if r['step']==step),None)

def paired_microbatch(requested,latest):
    existing=[r for r in latest if r is not None]
    if not existing:return requested,None
    policies={r['batch_policy']['batch'] for r in existing}
    if len(policies)>1:raise ValueError('The two arms have different saved microbatch policies')
    old=next(iter(policies))
    if old==requested:return requested,None
    # Align the batch-size change at the same update in both arms. If one is
    # ahead, first let both finish the next milestone at the saved batch size.
    if len(existing)!=2 or existing[0]['step']!=existing[1]['step']:
        return old,'Batch change deferred until both arms reach a matched milestone'
    if any(r['cursor']['rows'] or r['cursor']['batches'] for r in existing):
        return old,'Batch change deferred until a data-epoch boundary'
    return requested,None
