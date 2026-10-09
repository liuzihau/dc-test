"""GPU correctness checks against original author model; no optimizer run."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'third_party/reasoning_with_latent_tokens'))
from zebra.runtime import install_no_cudagraph_compile
install_no_cudagraph_compile()
import torch
from omegaconf import OmegaConf
from zebra.entrypoint import make_config
from zebra.model import ZebraMDM
from difflm import DiffLM


def main():
    args = SimpleNamespace(run=ROOT/'.cache/runtime/zebra-modular/model-check',
        recipe=ROOT/'zebra/configs/mdm.yaml', seed=1, resume=None, workers=0,
        microbatch=128, devices=2, target_steps=1, checkpoint_interval=1,
        stage='train', eval_batches=1, eval_batch_size=4, candidate_window=0,
        generation_layout='author', smoke=False)
    config, upstream = make_config(args)
    config.model.dropout = 0.0
    tokenizer = upstream.dataloader.get_tokenizer(config)
    torch.manual_seed(3)
    original = DiffLM(config, tokenizer).cuda().eval()
    torch.manual_seed(3)
    canonical = ZebraMDM(config, tokenizer).cuda().eval()
    np_cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    np_cfg.mechanisms.np.enabled = True
    torch.manual_seed(3)
    neighbor = ZebraMDM(np_cfg, tokenizer).cuda().eval()
    for key, value in canonical.backbone.state_dict().items():
        torch.testing.assert_close(value, neighbor.backbone.state_dict()[key], rtol=0, atol=0)
    assert len(neighbor.ema.shadow_params) == len(list(neighbor._get_parameters()))
    x0 = torch.randint(4, 10, (2, 384), device='cuda')
    valid = torch.ones_like(x0)
    # All use identical corrupted states; original groups clean/masked positions.
    torch.manual_seed(8)
    ref = original._loss(x0, valid, train_mode=True)
    torch.manual_seed(8)
    actual = canonical._loss(x0, valid, train_mode=True)
    torch.testing.assert_close(ref.loss, actual.loss, atol=0.015, rtol=0.005)
    torch.manual_seed(8)
    extended = neighbor._loss(x0, valid, train_mode=True)
    torch.testing.assert_close(actual.nlls, extended.nlls, rtol=0, atol=0)
    assert extended.loss > actual.loss
    extended.loss.backward()
    assert all(p.grad is not None for p in neighbor.backbone.neighbor_heads.parameters())
    assert neighbor.backbone.vocab_embed.embedding.grad.abs().sum() > 0
    # NP must not alter the objective or head used for validation/generation.
    torch.manual_seed(8)
    va = canonical._loss(x0, valid, train_mode=False)
    torch.manual_seed(8)
    vb = neighbor._loss(x0, valid, train_mode=False)
    torch.testing.assert_close(va.loss, vb.loss, atol=0, rtol=0)
    print('PASS: canonical permutation equivalence, shared initialization, NP gradients, EMA membership, common validation')
    print({'author_loss': ref.loss.item(), 'canonical_loss': actual.loss.item(),
           'np_objective': extended.loss.item()})


if __name__ == '__main__':
    main()
