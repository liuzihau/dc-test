"""Four isolated puzzle ablations. Default action checks configuration, not training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
AUTHOR=ROOT/'third_party/reasoning_with_latent_tokens'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(AUTHOR))
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(name,'2')
os.environ['PYTHONPATH']=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('HF_HUB_CACHE',str(ROOT/'.cache/huggingface/hub'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/runtime/puzzle-recurrence/mpl'))
from puzzle_recurrence.settings import VARIANTS,configure,validate_config

def build_config(args):
    from omegaconf import OmegaConf
    if args.task=='sudoku':from sudoku.entrypoint import make_config
    else:from zebra.entrypoint import make_config
    config,upstream=make_config(args)
    config=configure(config,args.variant)
    recipe=OmegaConf.load(args.recipe)
    config.puzzle_recurrence=recipe.puzzle_recurrence
    config.puzzle_allow_microbatch_change=bool(getattr(args,'allow_microbatch_change',False))
    validate_config(config)
    config.callbacks.trajectory_metrics=dict(_target_='puzzle_recurrence.metrics.TrajectoryMetrics',run=str(args.run))
    return config,upstream

def main():
    sys.argv[0]=str(Path(__file__).resolve())
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['sudoku','zebra'],required=True)
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    parser.add_argument('--stage',choices=['check','train','evaluate'],default='check')
    parser.add_argument('--run',type=Path);parser.add_argument('--recipe',type=Path)
    parser.add_argument('--resume',type=Path);parser.add_argument('--devices',type=int,default=2)
    parser.add_argument('--microbatch',type=int,default=32)
    parser.add_argument('--allow-microbatch-change',action='store_true')
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--target-steps',type=int);parser.add_argument('--checkpoint-interval',type=int)
    parser.add_argument('--eval-batches',type=int,default=10);parser.add_argument('--eval-batch-size',type=int,default=128)
    parser.add_argument('--candidate-window',type=int);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--generation-layout',choices=['author'],default='author')
    args=parser.parse_args()
    args.run=(args.run or ROOT/'outputs'/args.task/'three-state-ablation'/args.variant).resolve()
    args.recipe=(args.recipe or ROOT/'puzzle_recurrence/configs'/args.task/(args.variant+'.yaml')).resolve()
    if args.resume:args.resume=args.resume.resolve()
    args.target_steps=args.target_steps or (70500 if args.task=='sudoku' else 117200)
    args.checkpoint_interval=args.checkpoint_interval or (3525 if args.task=='sudoku' else 2930)
    if args.candidate_window is None:args.candidate_window=8 if args.task=='sudoku' else 0
    if args.stage=='train' and int(os.environ.get('LOCAL_RANK','0'))==0 and args.run.exists() and not args.resume:
        raise FileExistsError('Use a new output directory or an explicit checkpoint; preserve existing trials')
    from zebra.runtime import install_no_cudagraph_compile
    install_no_cudagraph_compile()
    from omegaconf import OmegaConf
    import lightning as L
    config,upstream=build_config(args)
    record=dict(task=args.task,variant=args.variant,stage=args.stage,
        baseline='third_party/reasoning_with_latent_tokens',trajectory=bool(config.mechanisms.tt.enabled),
        extra_attention=bool(config.mechanisms.ea.enabled),recurrent_memory=bool(config.mechanisms.rm.enabled),
        neighbor_prediction=bool(config.mechanisms.np.enabled),
        settings=OmegaConf.to_container(config.puzzle_recurrence,resolve=True),
        training_target_policy='all valid tokens' if config.training.train_on_all_tokens else 'author solution loss mask',
        validation='author single-canvas ELBO plus separate three-state diagnostics for trajectory arms',
        generation='original author sampler; canonical memory keyed by original token IDs',
        recipe_sha256=hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
        run=str(args.run),automatic_next_experiment=False)
    if args.stage=='check':print(json.dumps(record,indent=2));return
    import torch
    from puzzle_recurrence.devices import selected_gpu_ids,gpu_locks
    gpu_ids=selected_gpu_ids(os.environ.get('CUDA_VISIBLE_DEVICES',''))
    if torch.cuda.device_count()!=len(gpu_ids) or len(gpu_ids)<int(config.trainer.devices):
        raise ValueError('Visible GPU count does not match the requested device policy')
    record['physical_gpu_ids']=gpu_ids
    with gpu_locks(ROOT,gpu_ids,int(os.environ.get('LOCAL_RANK','0'))):
        for flag,value in [('--run',args.run),('--recipe',args.recipe),('--resume',args.resume)]:
            if value is not None and flag in sys.argv:sys.argv[sys.argv.index(flag)+1]=str(value)
        args.run.mkdir(parents=True,exist_ok=True);os.chdir(args.run)
        if args.resume:
            header=torch.load(args.resume,map_location='cpu',weights_only=False,mmap=True)
            if header.get('puzzle_data_cursor',{}).get('variant')!=args.variant:
                raise ValueError('Checkpoint belongs to another ablation')
            del header
        if int(os.environ.get('LOCAL_RANK','0'))==0:
            OmegaConf.save(config,args.run/'resolved_config.yaml',resolve=True)
            (args.run/'contract.json').write_text(json.dumps(record,indent=2)+'\n')
        from puzzle_recurrence.factory import model_class
        L.seed_everything(args.seed)
        tokenizer=upstream.dataloader.get_tokenizer(config);logger=upstream.utils.get_logger('puzzle-recurrence')
        operation=upstream._complete if args.stage=='evaluate' else upstream._train
        operation(model_class(config),config,logger,tokenizer)

if __name__=='__main__':main()
