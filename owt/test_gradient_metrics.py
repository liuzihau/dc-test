"""Instrumentation must neither perturb gradients nor consume randomness."""
import math
import csv
from types import SimpleNamespace

import torch

from owt.gradient_metrics import gradient_norms
from owt.metrics import PreclipGradientMetrics


def fixture():
    model = torch.nn.Module()
    model.backbone = torch.nn.Module()
    model.backbone.trunk = torch.nn.Linear(2, 1, bias=False)
    model.backbone.output_layer = torch.nn.Module()
    model.backbone.output_layer.linear = torch.nn.Linear(2, 1, bias=False)
    model.backbone.neighbor_heads = torch.nn.Linear(2, 1, bias=False)
    for parameter, values in zip(model.parameters(), [[3., 4.], [0., 12.], [0., 0.]]):
        parameter.grad = torch.tensor([values])
    return model


def test_norms_match_actual_l2_clipping_without_mutation_or_rng(tmp_path):
    model = fixture()
    original = [p.grad.clone() for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    callback = PreclipGradientMetrics(tmp_path)
    trainer = SimpleNamespace(is_global_zero=True, gradient_clip_val=1.,
                              gradient_clip_algorithm='norm', global_step=17)
    callback.on_train_start(trainer, model)
    callback.on_before_optimizer_step(trainer, model, None)
    result = gradient_norms(model, 1.)
    assert result['joint_l2'] == 13.
    assert result['shared_trunk_l2'] == 5.
    assert result['main_readout_l2'] == 12.
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before, p.grad) for before, p in zip(original, model.parameters()))
    assert (tmp_path/'local_metrics/gradient_norms.csv').read_text().splitlines()[1].startswith('18,')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    assert math.isclose(float(model.backbone.trunk.weight.grad[0, 0])/3.,
                        result['estimated_clip_multiplier'], rel_tol=1e-6)


def test_resume_truncates_only_diagnostic_future_rows(tmp_path):
    model = fixture()
    callback = PreclipGradientMetrics(tmp_path)
    trainer = SimpleNamespace(is_global_zero=True, gradient_clip_val=1.,
                              gradient_clip_algorithm='norm', global_step=9)
    callback.on_train_start(trainer, model)
    callback.on_before_optimizer_step(trainer, model, None)
    trainer.global_step = 10
    callback.on_before_optimizer_step(trainer, model, None)
    callback.on_train_start(trainer, model)
    assert len(callback.path.read_text().splitlines()) == 2


def test_installed_lightning_logs_once_per_accumulated_update_before_clipping(tmp_path):
    import lightning as L

    class Toy(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.clipping_observations = []

        def training_step(self, batch, batch_idx):
            return (self.weight*batch[0]).square().mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=.1)

        def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
            before = float(self.weight.grad.norm())
            self.clip_gradients(optimizer, gradient_clip_val, gradient_clip_algorithm)
            self.clipping_observations.append((before, float(self.weight.grad.norm())))

    model = Toy()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.ones(4)), batch_size=1)
    trainer = L.Trainer(accelerator='cpu', devices=1, max_steps=2,
        accumulate_grad_batches=2, gradient_clip_val=1.,
        callbacks=[PreclipGradientMetrics(tmp_path)], logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, default_root_dir=str(tmp_path))
    trainer.fit(model, loader)
    with (tmp_path/'local_metrics/gradient_norms.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row['optimizer_step']) for row in rows] == [1, 2]
    assert math.isclose(float(rows[0]['joint_l2']), 2., rel_tol=1e-6)
    for row, (before, after) in zip(rows, model.clipping_observations):
        assert math.isclose(float(row['joint_l2']), before, rel_tol=1e-6)
        assert math.isclose(after, 1., rel_tol=1e-6)
