"""Evaluate epoch6 in both arms and attention epoch8, then resume latest states.

Checkpoint filename epochs are zero-based. Evaluation history uses completed
epochs: filename6 -> completed7; filename8 -> completed9.
"""
import argparse
import json
from pathlib import Path
import os
import sys

from puzzle_recurrence.run_pair import ROOT,VARIANTS,initial_checkpoints
from puzzle_recurrence.schedule import TASKS,checkpoint_at,latest_checkpoint


def resume_command(args):
    root=args.output_root.expanduser().resolve()/args.task/'three-state-ablation'
    requested=[(VARIANTS[0],7,args.attention_epoch6),(VARIANTS[1],7,args.recurrent_epoch6),
               (VARIANTS[0],9,args.attention_epoch8)]
    specifications=[]
    for variant,epoch,override in requested:
        if override:
            path=str(override.expanduser().resolve())
        else:
            info=checkpoint_at(root/variant,epoch*TASKS[args.task]['steps_per_epoch'])
            if info is None:
                flag='--attention-epoch8' if epoch==9 else ('--attention-epoch6' if variant==VARIANTS[0] else '--recurrent-epoch6')
                raise FileNotFoundError(f'{variant}: filename epoch{epoch-1} checkpoint missing. Supply its backup with {flag} PATH.')
            path=info['path']
        specifications.append([args.task,variant,path])
    records=initial_checkpoints(specifications,[args.task])[args.task]
    if [r['epoch'] for r in records]!=[7,7,9]:
        raise ValueError('Checkpoint overrides must correspond to filename epochs6,6,8 (completed epochs7,7,9)')
    latest=[latest_checkpoint(root/v) for v in VARIANTS]
    if any(r is None for r in latest):raise FileNotFoundError('Both arms need a latest training checkpoint')
    for variant,info in zip(VARIANTS,latest):
        print(f'Resume {variant}: completed epoch{info["cursor"]["epoch"]}, step{info["step"]}, {info["path"]}',flush=True)
    cmd=[sys.executable,'-u','-m','puzzle_recurrence.run_pair','--tasks',args.task,'--resume',
        '--microbatch',str(args.microbatch),'--allow-microbatch-change','--workers',str(args.workers),
        '--output-root',str(args.output_root.expanduser().resolve()),
        '--attention-gpus',args.attention_gpus,'--recurrent-gpus',args.recurrent_gpus]
    if args.epochs is not None:cmd+=['--epochs',str(args.epochs)]
    if args.dry_run:cmd+=['--dry-run']
    for specification in specifications:cmd+=['--initial-evaluation',*specification]
    return cmd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=list(TASKS),required=True)
    parser.add_argument('--output-root',type=Path,default=ROOT/'outputs')
    parser.add_argument('--attention-epoch6',type=Path,help='Optional backup path for attention filename epoch6')
    parser.add_argument('--recurrent-epoch6',type=Path,help='Optional backup path for recurrent filename epoch6')
    parser.add_argument('--attention-epoch8',type=Path,help='Optional backup path for attention filename epoch8')
    parser.add_argument('--microbatch',type=int,default=64);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--attention-gpus',default='0,1');parser.add_argument('--recurrent-gpus',default='2,3')
    parser.add_argument('--epochs',type=int,help='Total completed epoch budget, not extra epochs')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    cmd=resume_command(args)
    if args.dry_run:print(json.dumps(dict(command=cmd),indent=2),flush=True)
    os.execv(sys.executable,cmd)


if __name__=='__main__':main()
