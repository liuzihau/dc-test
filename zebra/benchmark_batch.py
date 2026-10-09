"""Scratch two-GPU training throughput test. Never reads/writes model checkpoints."""
import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'third_party/reasoning_with_latent_tokens'))
from zebra.runtime import install_no_cudagraph_compile
install_no_cudagraph_compile()
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from zebra.entrypoint import make_config
from zebra.model import ZebraMDM


class Objective(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens):
        return self.model._loss(tokens, torch.ones_like(tokens), train_mode=True).loss


def measure(variant, microbatch, device):
    args = SimpleNamespace(run=ROOT/'.cache/runtime/zebra-modular/batch-benchmark',
        recipe=ROOT/f'zebra/configs/{variant}.yaml', seed=1, resume=None, workers=0,
        microbatch=microbatch, devices=2, target_steps=1, checkpoint_interval=1,
        stage='train', eval_batches=1, eval_batch_size=128, smoke=False)
    config, upstream = make_config(args)
    tokenizer = upstream.dataloader.get_tokenizer(config)
    torch.manual_seed(1)
    model = ZebraMDM(config, tokenizer).to(device).train()
    if model.ema:
        model.ema.move_shadow_params_to_device(device)
    wrapped = DDP(Objective(model), device_ids=[device.index], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(model._get_parameters(), lr=config.optim.lr,
        betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
        weight_decay=config.optim.weight_decay)
    # Same dense sequence shapes as training, all non-special tokens so NP is
    # fully active. Synthetic inputs avoid perturbing training RNG/data cursors.
    tokens = torch.randint(4, 10, (256, 384), device=device)
    accumulation = 256 // microbatch
    timings = []
    for update in range(22):
        if update == 6:
            torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for chunk in range(accumulation):
            sync = wrapped.no_sync() if chunk+1 < accumulation else contextlib.nullcontext()
            with sync:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = wrapped(tokens[chunk*microbatch:(chunk+1)*microbatch]) / accumulation
                loss.backward()
        torch.nn.utils.clip_grad_norm_(model._get_parameters(), config.trainer.gradient_clip_val)
        optimizer.step()
        if model.ema:
            model.ema.update(model._get_parameters())
        torch.cuda.synchronize(device)
        elapsed = torch.tensor(time.perf_counter()-start, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if update >= 6:
            timings.append(elapsed.item())
    peak = torch.tensor(torch.cuda.max_memory_reserved(device), device=device, dtype=torch.float64)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    result = dict(variant=variant, microbatch=microbatch,
                  median_update_seconds=statistics.median(timings),
                  peak_reserved_gib=peak.item()/2**30, updates_measured=len(timings))
    del optimizer, wrapped, model, loss, tokens
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    dist.init_process_group('nccl')
    results = []
    for variant in ('mdm', 'mdm_np'):
        for microbatch in (128, 256):
            results.append(measure(variant, microbatch, device))
    speedups = {v: results[2*i]['median_update_seconds']/results[2*i+1]['median_update_seconds']
                for i, v in enumerate(('mdm', 'mdm_np'))}
    capacity = torch.cuda.get_device_properties(device).total_memory/2**30
    safe = all(r['peak_reserved_gib'] < capacity-2 for r in results)
    selected = 256 if safe and min(speedups.values()) > 1.03 else 128
    if rank == 0:
        report = dict(selected_microbatch=selected, measurements=results, speedups=speedups,
                      reason='Require >3% speedup for BOTH models and 2 GiB VRAM headroom.',
                      scope='Synthetic dense training compute + DDP + AdamW + EMA; excludes data loading/validation.')
        temporary = args.report.with_suffix('.partial')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(args.report)
        print(json.dumps(report, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
