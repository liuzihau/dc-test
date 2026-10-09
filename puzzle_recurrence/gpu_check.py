"""Full-size author-backbone forward/backward checks; zero optimizer updates."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from puzzle_recurrence.entrypoint import ROOT,build_config
from zebra.runtime import install_no_cudagraph_compile
install_no_cudagraph_compile()
from puzzle_recurrence.factory import model_class


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--microbatch',type=int,default=2)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3' or torch.cuda.device_count()!=2:raise ValueError('Set CUDA_VISIBLE_DEVICES=2,3')
    records=[]
    with (ROOT/'outputs/owt/mdm-np-5k/.queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for task in ('sudoku','zebra'):
            for variant in ('trajectory_attention','trajectory_recurrent'):
                cfgargs=SimpleNamespace(task=task,variant=variant,run=args.output.parent,
                    recipe=ROOT/'puzzle_recurrence/configs'/task/(variant+'.yaml'),seed=1,resume=None,workers=0,
                    microbatch=args.microbatch,devices=2,target_steps=1,checkpoint_interval=1,stage='check',eval_batches=1,
                    eval_batch_size=2,candidate_window=8 if task=='sudoku' else 0,generation_layout='author',smoke=False)
                cfg,upstream=build_config(cfgargs);tok=upstream.dataloader.get_tokenizer(cfg)
                with patch('metrics.transformers.AutoTokenizer.from_pretrained',return_value=SimpleNamespace(pad_token='[PAD]',pad_token_id=0)):
                    torch.manual_seed(13);model=model_class(cfg)(cfg,tok).cuda().train()
                with torch.no_grad():model.backbone.output_layer.linear.weight.normal_(std=.02)
                content=[i for i in range(model.vocab_size) if i not in set(tok.all_special_ids)|{model.mask_index,model.pad_index}]
                clean=torch.tensor([[content[j%len(content)] for j in range(cfg.model.length)]]*args.microbatch,device='cuda')
                valid=torch.ones_like(clean);loss_mask=torch.ones_like(clean);loss_mask[:,:cfg.model.length//2]=0
                mask=None if cfg.training.train_on_all_tokens else loss_mask
                torch.cuda.reset_peak_memory_stats()
                loss=model._loss(clean,valid,train_mode=True,loss_mask=mask).loss
                loss.backward();torch.cuda.synchronize()
                assert torch.isfinite(loss) and float(loss)<100
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
                records.append(dict(task=task,variant=variant,microbatch=args.microbatch,length=cfg.model.length,hidden_size=cfg.model.hidden_size,
                    layers=cfg.model.n_blocks,loss=float(loss),forward_passes=model._last_trajectory['main_forwards'],
                    gradient_edges=model._last_trajectory['gradient_edges'],peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
                del model,loss,clean;torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(checks=records,optimizer_updates=0,benchmark_training_started=False),indent=2)+'\n')
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
