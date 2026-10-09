"""GPU contract check: original BD3 parity, NP gradient, and full-mask memory."""
import copy
import gc
import json
import os
from types import SimpleNamespace

from owt.entrypoint import ROOT, make_config, verify_cache
import dataloader
from diffusion import Diffusion
from owt.model import OWTMDM
import torch


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2,3'
    args = SimpleNamespace(variant='mdm', data_dir=ROOT/'.cache/huggingface',
        microbatch=8, global_batch=512, workers=1, steps=5000, interval=500,
        resume=None, run=ROOT/'.cache/runtime/owt/contract-check', val_examples=1024)
    config = make_config(args)
    print(json.dumps(verify_cache(config), indent=2))
    tokenizer = dataloader.get_tokenizer(config)
    # Tiny dimensions for a direct numerical loss/gradient equivalence test.
    small = copy.deepcopy(config)
    small.model.hidden_size = 32
    small.model.cond_dim = 16
    small.model.n_blocks = 2
    small.model.n_heads = 4
    small.model.length = small.block_size = 16
    torch.manual_seed(1)
    upstream = Diffusion(small, tokenizer).cuda().train()
    torch.manual_seed(1)
    adapter = OWTMDM(small, tokenizer).cuda().train()
    for name, tensor in upstream.state_dict().items():
        torch.testing.assert_close(tensor, adapter.state_dict()[name], rtol=0, atol=0)
    x = torch.randint(0, 50000, (2,16), device='cuda')
    valid = torch.ones_like(x)
    torch.cuda.manual_seed(731)
    a = upstream._loss(x, valid.clone()).loss
    a.backward()
    torch.cuda.manual_seed(731)
    b = adapter._loss(x, valid.clone()).loss
    b.backward()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for pa, pb in zip(upstream.parameters(), adapter.parameters()):
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)
    # Adding the auxiliary heads must not alter shared initialization or the
    # subsequent CPU/CUDA random stream used for corruption and dropout.
    with_np = copy.deepcopy(small)
    with_np.mechanisms.np.enabled = True
    torch.manual_seed(71)
    control = OWTMDM(small, tokenizer)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    torch.manual_seed(71)
    augmented = OWTMDM(with_np, tokenizer)
    torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
    for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_rng):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, tensor in control.state_dict().items():
        torch.testing.assert_close(tensor, augmented.state_dict()[name], rtol=0, atol=0)
    del upstream, adapter, control, augmented
    gc.collect()
    torch.cuda.empty_cache()
    # Full production geometry and worst-case NP occupancy on one authorized GPU.
    args.variant = 'mdm_np'
    npconfig = make_config(args)
    torch.manual_seed(1)
    model = OWTMDM(npconfig, tokenizer).cuda().train()
    model.ema.move_shadow_params_to_device('cuda')
    optimizer = torch.optim.AdamW(model._get_parameters(), lr=3e-4)
    model._sample_t = lambda dims, device, *a, **kw: torch.ones(dims[0], 1, device=device)
    x = torch.randint(0, 50000, (args.microbatch,1024), device='cuda')
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        result = model._loss(x, torch.ones_like(x))
    result.loss.backward()
    for head in model.backbone.neighbor_heads.heads:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    assert model.backbone.vocab_embed.embedding.grad.abs().sum() > 0
    assert len(model.ema.shadow_params) == len(list(model._get_parameters()))
    optimizer.step()
    model.ema.update(model._get_parameters())
    print(json.dumps(dict(status='PASS', upstream_loss_gradient_parity='exact',
        shared_initialization_and_rng='exact',
        np_full_mask_loss=float(result.loss), np_heads_backbone_gradient='PASS',
        ema_membership='PASS', total_parameters=sum(p.numel() for p in model.parameters()),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30), indent=2))


if __name__ == '__main__':
    main()
