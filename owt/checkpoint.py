"""Load a finished pilot's frozen EMA on CPU without downloading eval models."""
import gc
import hashlib
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf
import torch
from owt.model import OWTMDM


def checkpoint_provenance(run):
    checkpoint=run/'checkpoints/step-0005000.ckpt'
    stat=checkpoint.stat()
    return dict(checkpoint=str(checkpoint),checkpoint_size=stat.st_size,
        checkpoint_mtime_ns=stat.st_mtime_ns,
        config_sha256=hashlib.sha256((run/'resolved_config.yaml').read_bytes()).hexdigest(),
        parameter_state='EMA',precision='CPU float32',optimizer_step=5000)


def load_ema_model(run):
    provenance=checkpoint_provenance(run)
    cfg=OmegaConf.load(run/'resolved_config.yaml')
    tokenizer=SimpleNamespace(vocab_size=50257,mask_token=None,all_special_ids=[50256])
    with patch('diffusion.metrics.Metrics',return_value=torch.nn.Module()):
        model=OWTMDM(cfg,tokenizer)
    payload=torch.load(provenance['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    assert payload['global_step']==5000
    model.load_state_dict(payload['state_dict'],strict=True)
    model.ema.load_state_dict(payload['ema'])
    model.ema.copy_to(model._get_parameters())
    model.ema=None
    del payload
    gc.collect()
    assert model.mask_index==50257 and model.num_tokens==1024
    model.eval()
    return model,provenance
