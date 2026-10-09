"""CPU-only two-rank actual-model DDP mechanics; no OWT efficacy claim."""
import argparse
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

from owt.test_transformer_np_control import make_model
from owt.test_transformer_np import batch


class Step(nn.Module):
    def __init__(self, model, rank):
        super().__init__()
        self.model, self.rank = model, rank

    def forward(self, x, valid):
        def corrupt(clean, probability, **kwargs):
            if self.rank == 0:
                return torch.full_like(clean, self.model.mask_index)
            noisy = clean.clone()
            noisy[:, 2] = self.model.mask_index
            return noisy
        with patch.object(self.model, 'q_xt', side_effect=corrupt):
            return self.model._loss(x, valid).loss


def worker(rank, directory):
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group('gloo', rank=rank, world_size=2,
        init_method='file://' + str(directory / 'rendezvous'), timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(71)
        model = make_model()
        model._trainer = SimpleNamespace(global_rank=rank)
        step = DistributedDataParallel(Step(model, rank), find_unused_parameters=False)
        optimizer = torch.optim.AdamW(model._get_parameters(), lr=1e-3)
        x, valid = batch()
        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            loss = step(x, valid.clone())
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            losses.append(float(loss))
        total = torch.stack([p.detach().double().sum() for p in model.parameters()]).sum()
        gathered = [torch.zeros_like(total) for _ in range(2)]
        dist.all_gather(gathered, total)
        assert torch.equal(gathered[0], gathered[1])
        selected = sum(float(s[:, 2].sum()) for s in model._last_pair_statistics.values())
        assert selected > 0 if rank == 0 else selected == 0
        (directory / f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,
            steps=3, finite_losses=losses, selected_pairs_last_batch=selected,
            all_parameter_gradients_present=True, synchronized_parameter_checksum=float(total),
            branch_calls=model.branch_calls)) + '\n')
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.device_count() != 0:
        raise RuntimeError('CPU preflight requires CUDA_VISIBLE_DEVICES empty')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    mp.spawn(worker, args=(str(args.output),), nprocs=2, join=True)
    ranks = [json.loads((args.output / f'rank-{rank}.json').read_text()) for rank in range(2)]
    result = dict(preflight=True, device='cpu', backend='gloo', world_size=2,
                  updates_per_rank=3, model='actual small BD3 MatchedTransformerNPMDM',
                  find_unused_parameters=False, ranks=ranks, production_training_launched=False)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
