"""The first reasoning baseline: five independent states, prev/next CE, no memory.

All fixtures are locally generated pilot tasks, not downloaded benchmarks. These
tests exercise the actual debug model and optimizer on CPU, including resume.
"""

import csv
import json
import math
import socket

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from reasoning import runner
from reasoning.data import ReasoningDataset, prepare_dataset
from reasoning.model import ReasoningModel


STATE_NAMES = ("full", "t0", "t1", "t2", "t3")
STATE_WEIGHTS = (0.05, 0.10, 0.20, 1.00, 0.70)


@pytest.fixture(autouse=True)
def cpu_without_network(monkeypatch):
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("mdm_aux regression tests must not use CUDA or network")

    for name in ("_lazy_init", "set_device", "get_rng_state"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old_threads)


@pytest.fixture(scope="module")
def pilot_datasets(tmp_path_factory):
    """Generate each task once; share immutable fixtures across smoke tests."""
    root = tmp_path_factory.mktemp("mdm-aux-pilots")
    result = {}
    for task in runner.TASKS:
        directory = root / task
        prepare_dataset(directory, task, train_size=4, valid_size=2,
                        test_size=1, seed=33)
        result[task] = (directory, ReasoningDataset(directory))
    return result


def baseline_args(task, directory, run_dir, max_steps=2):
    return runner.parser().parse_args([
        "train", "--task", task, "--variant", "mdm_aux",
        "--data-dir", str(directory), "--run-dir", str(run_dir),
        "--size", "debug", "--device", "cpu", "--precision", "fp32",
        "--global-batch", "2", "--micro-batch", "2",
        "--max-steps", str(max_steps), "--warmup-steps", "2",
        "--val-every", "1", "--validation-examples", "2",
        "--eval-batch-size", "2", "--save-every", "1",
        "--save-seconds", "0", "--log-every", "1",
        "--seed", "14", "--cpu-threads", "1",
    ])


def make_baseline(pilot_datasets, tmp_path, task="countdown"):
    directory, dataset = pilot_datasets[task]
    args = baseline_args(task, directory, tmp_path / "not-launched")
    model = ReasoningModel(runner.build_model_config(args, dataset))
    return model, default_collate([dataset[0], dataset[1]])


def assert_no_memory(model):
    assert model.config["memory_mode"] == "none"
    assert not model.has_dcache and not model.has_final
    assert not model.backbone.step_memory_enabled
    assert not model.backbone.dcachehooping_enabled
    assert model.backbone.dc_final_writer is None
    assert model.backbone.dcachehooping_latent_norm is None
    assert model.backbone.dcachehooping_status_embed is None
    assert model.backbone.dcachehooping_confidence_head is None
    for block in model.backbone.blocks:
        assert block.attention_mode == "merged"
        assert not block.step_memory_enabled
        assert block.step_memory_gate is None
        # Current-only merged attention uses the normal QKV projection.
        assert block.dc_qkv is None and block.dc_norm is None
        assert block.kv_cache is None
    assert not any(any(part in name for part in (
        "dc_final_writer", "dcachehooping", "step_memory_gate", "dc_qkv"
    )) for name, _ in model.named_parameters())


def assert_memory_metrics_zero(metrics):
    for name in ("identity_loss", "identity_final", "identity_forwards",
                 "final_dropout", "adjacent_edges"):
        assert metrics[name].item() == 0, name
    for state in STATE_NAMES:
        for mode in ("cache_only", "current_only"):
            assert metrics[f"{mode}_fraction_{state}"].item() == 0
    assert metrics["trajectory_forwards"].item() == 5
    assert metrics["num_forwards"].item() == 5


@pytest.mark.parametrize("task", runner.TASKS)
def test_task_baseline_five_current_only_forwards_and_gradients(
        task, pilot_datasets, tmp_path, monkeypatch):
    model, batch = make_baseline(pilot_datasets, tmp_path, task)
    assert_no_memory(model)
    assert model.config["trajectory"] == "five"
    assert model.config["neighbors"]
    assert model.config["neighbor_weight"] == 0.5
    assert tuple(model.config["weights"]) == STATE_WEIGHTS
    assert model.config["gate_enabled"]  # A legacy flag must not create a gate.
    assert set(model.backbone.neighbor_heads.heads) == {"prev", "next"}
    original = model.backbone.forward
    calls = []

    def no_memory_forward(*args, **kwargs):
        for key in ("previous_step_kv", "previous_final_hidden",
                    "previous_attention_mask", "step_memory_source_mask"):
            assert kwargs[key] is None, key
        assert kwargs["return_step_kv"] is False
        output = original(*args, **kwargs)
        assert not output.step_kv
        calls.append(args[0].detach().clone())
        return output

    monkeypatch.setattr(model.backbone, "forward", no_memory_forward)
    loss, metrics = model.compute_loss(
        batch, step=2000, generator=torch.Generator().manual_seed(31))
    assert len(calls) == 5
    assert_memory_metrics_zero(metrics)
    assert metrics["neighbor_loss"] > 0
    assert metrics["loss_weight_sum"].item() == pytest.approx(2.05)
    for state in calls:
        torch.testing.assert_close(state[~batch["target_mask"]],
                                   batch["input_ids"][~batch["target_mask"]])
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for direction in ("prev", "next"):
        assert model.backbone.neighbor_heads.heads[direction][1].weight.grad.abs().sum() > 0
    assert model.backbone.blocks[0].attn_qkv.weight.grad.abs().sum() > 0
    assert model.backbone.output_layer.linear.weight.grad.abs().sum() > 0


def test_normalized_five_state_objective_uses_target_masked_neighbor_pairs(
        pilot_datasets, tmp_path, monkeypatch):
    model, batch = make_baseline(pilot_datasets, tmp_path)
    torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=0.05)
    original = model.forward
    states_and_outputs = []

    def record(*args, **kwargs):
        output = original(*args, **kwargs)
        states_and_outputs.append((args[0], output))
        return output

    monkeypatch.setattr(model, "forward", record)
    loss, metrics = model.compute_loss(
        batch, generator=torch.Generator().manual_seed(67))
    clean, valid = batch["input_ids"], batch["attention_mask"].clone()
    for special in model.config["special_ids"]:
        valid &= clean.ne(special)
    base_losses, neighbor_losses = [], []
    clean_source_pairs = 0
    with torch.no_grad():
        for state, output in states_and_outputs:
            masked = state.eq(model.mask_id)
            token_nll = -output["logits"].log_softmax(-1).gather(
                -1, clean.unsqueeze(-1)).squeeze(-1)
            base_losses.append(((token_nll * masked).sum(-1) /
                                masked.sum(-1)).mean())
            directional = []
            for direction, source, target in (
                    ("prev", slice(1, None), slice(None, -1)),
                    ("next", slice(None, -1), slice(1, None))):
                # Only the target must be masked. Both endpoints must be
                # valid content; no prompt corruption, specials, or wraparound.
                pairs = valid[:, source] & valid[:, target] & masked[:, target]
                clean_source_pairs += int((pairs & ~masked[:, source]).sum())
                hidden = output["final_hidden"][:, source][pairs]
                logits = model.backbone.neighbor_heads.heads[direction](hidden)
                logits[:, model.mask_id] = -torch.inf
                directional.append(F.cross_entropy(logits, clean[:, target][pairs])
                                   if pairs.any() else logits.new_zeros(()))
            neighbor_losses.append(sum(directional) / 2)
    assert len(base_losses) == len(neighbor_losses) == 5
    assert clean_source_pairs > 0  # Requiring masked sources changes this CE.
    expected_base = sum(w * value for w, value in zip(STATE_WEIGHTS, base_losses)) / 2.05
    expected_aux = sum(w * value for w, value in zip(STATE_WEIGHTS, neighbor_losses)) / 2.05
    torch.testing.assert_close(metrics["base_loss"], expected_base)
    torch.testing.assert_close(metrics["neighbor_loss"], expected_aux)
    torch.testing.assert_close(loss.detach(), expected_base + 0.5 * expected_aux)


def test_one_state_loss_cannot_backpropagate_to_another_forward(
        pilot_datasets, tmp_path, monkeypatch):
    model, batch = make_baseline(pilot_datasets, tmp_path)
    torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=0.05)
    model.config["weights"] = [0, 0, 0, 0, 1]
    states = []
    original = model.forward

    def retain(*args, **kwargs):
        output = original(*args, **kwargs)
        output["final_hidden"].retain_grad()
        states.append(output["final_hidden"])
        return output

    monkeypatch.setattr(model, "forward", retain)
    loss, _ = model.compute_loss(batch, generator=torch.Generator().manual_seed(71))
    loss.backward()
    assert len(states) == 5
    for index, hidden in enumerate(states):
        nonzero = hidden.grad is not None and bool(hidden.grad.abs().sum() > 0)
        assert nonzero == (index == 4)


def test_legacy_robustness_flags_cannot_activate_memory_or_identity(
        pilot_datasets, tmp_path):
    model, batch = make_baseline(pilot_datasets, tmp_path)
    zero_config = dict(model.config)
    for key in ("cache_only_probability", "current_only_probability",
                "final_dropout", "identity_probability"):
        assert model.config[key] > 0  # Exercise the actual runner defaults.
        zero_config[key] = 0
    clean_control = ReasoningModel(zero_config)
    clean_control.load_state_dict(model.state_dict(), strict=True)
    for training in (True, False):
        model.train(training)
        clean_control.train(training)
        generator = torch.Generator().manual_seed(83)
        control_generator = torch.Generator().manual_seed(83)
        actual, metrics = model.compute_loss(
            batch, step=2000, training=training, generator=generator)
        expected, control_metrics = clean_control.compute_loss(
            batch, step=2000, training=training, generator=control_generator)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name in metrics:
            torch.testing.assert_close(metrics[name], control_metrics[name], rtol=0, atol=0)
        torch.testing.assert_close(generator.get_state(), control_generator.get_state())
        assert_memory_metrics_zero(metrics)
        if not training:
            assert metrics["neighbor_loss"] == 0
            torch.testing.assert_close(actual, metrics["base_loss"])


def assert_tree_equal(actual, expected):
    if torch.is_tensor(actual):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_tree_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("task", runner.TASKS)
def test_mdm_aux_cpu_pilot_training_validation_and_exact_resume(
        task, pilot_datasets, tmp_path):
    directory, dataset = pilot_datasets[task]
    assert dataset.manifest["source"]["kind"] == "synthetic_pilot"
    direct, resumed = tmp_path / "direct", tmp_path / "resumed"
    runner.train(baseline_args(task, directory, direct, max_steps=2))
    runner.train(baseline_args(task, directory, resumed, max_steps=1))
    first = runner.load_checkpoint(resumed / "checkpoints/last.pt")
    assert first["step"] == 1 and first["examples_seen"] == 2
    runner.train(baseline_args(task, directory, resumed, max_steps=2))
    expected = runner.load_checkpoint(direct / "checkpoints/last.pt")
    actual = runner.load_checkpoint(resumed / "checkpoints/last.pt")
    assert actual["step"] == 2 and actual["examples_seen"] == 4
    assert actual["contract"]["variant"] == "mdm_aux"
    assert actual["contract"]["task"] == task
    assert actual["contract"]["device_type"] == "cpu"
    for key in ("model", "optimizer", "rng_by_rank", "contract"):
        assert_tree_equal(actual[key], expected[key])
    for name in ("backbone.blocks.0.attn_qkv.weight",
                 "backbone.neighbor_heads.heads.prev.1.weight",
                 "backbone.neighbor_heads.heads.next.1.weight"):
        assert not torch.equal(first["model"][name], actual["model"][name]), name
    assert actual["rng_by_rank"][0]["cuda"] is None
    restored = ReasoningModel(actual["model_config"])
    restored.load_state_dict(actual["model"], strict=True)
    assert_no_memory(restored)
    for step in (1, 2):
        filename = f"validation/step-{step:09d}.json"
        validation = json.loads((resumed / filename).read_text())
        assert math.isfinite(validation["mean_over_ratios_nll"])
        assert len(validation["ratios"]) == 4
    attempts = sorted((resumed / "logs").glob("attempt-*/resume_attempt.json"))
    assert [json.loads(path.read_text())["resume_step"] for path in attempts] == [0, 1]
    for path in (resumed / "logs").glob("attempt-*/metrics.csv"):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("train/loss"):
                    assert float(row["train/num_forwards"]) == 5
                    assert float(row["train/neighbor_loss"]) > 0
                    for name in ("adjacent_edges", "identity_forwards", "identity_loss", "final_dropout"):
                        assert float(row["train/" + name]) == 0
    assert json.loads((resumed / "status.json").read_text())["status"] == "finished"
