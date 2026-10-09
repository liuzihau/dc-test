"""Real full-task-length CPU fits, recovery and generated-memory evaluation.

These are small correctness tests, not accuracy experiments or an H100 memory
estimate: the backbone is the debug model, but Sudoku/Zebra inputs and decoded
answer slots use their actual task lengths. No downloads or CUDA are allowed.
"""

import csv
import json
import math
import socket

import pytest
import torch

from reasoning import runner
from reasoning.data import prepare_dataset
from reasoning.model import ReasoningModel
from reasoning.tasks import TASK_LENGTHS, task_answer_slots


@pytest.fixture(autouse=True)
def cpu_without_network(monkeypatch):
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU memory integration test attempted CUDA or network access")

    for name in ("_lazy_init", "set_device", "get_rng_state",
                 "reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


def _train_args(task, variant, data_dir, run_dir, max_steps):
    return runner.parser().parse_args([
        "train", "--task", task, "--variant", variant,
        "--data-dir", str(data_dir), "--run-dir", str(run_dir),
        "--size", "debug", "--device", "cpu", "--precision", "fp32",
        "--merged-policy", "current_preserving", "--gradient-mode", "adjacent",
        "--neighbor-weight", "0.5", "--micro-batch", "2", "--global-batch", "2",
        "--max-steps", str(max_steps), "--warmup-steps", "2",
        "--val-every", "1", "--validation-examples", "2", "--eval-batch-size", "2",
        "--save-every", "1", "--save-seconds", "0", "--log-every", "1",
        "--seed", "17", "--cpu-threads", "1",
    ])


def _assert_finite_numbers(value):
    if isinstance(value, dict):
        for child in value.values():
            _assert_finite_numbers(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_finite_numbers(child)
    elif isinstance(value, (int, float)):
        assert math.isfinite(value)


@pytest.mark.parametrize("task", ["sudoku", "zebra"])
@pytest.mark.parametrize("variant", ["both", "both_aux"])
def test_real_task_memory_train_resume_validate_and_generate(
        task, variant, tmp_path, monkeypatch):
    data_dir, run_dir = tmp_path / "dataset", tmp_path / "run"
    prepare_dataset(data_dir, task, train_size=4, valid_size=2, test_size=2, seed=17)

    runner.train(_train_args(task, variant, data_dir, run_dir, max_steps=1))
    first = runner.load_checkpoint(run_dir / "checkpoints/last.pt")
    assert first["step"] == 1 and first["examples_seen"] == 2
    runner.train(_train_args(task, variant, data_dir, run_dir, max_steps=2))
    checkpoint_path = run_dir / "checkpoints/last.pt"
    checkpoint = runner.load_checkpoint(checkpoint_path)
    assert checkpoint["step"] == 2 and checkpoint["examples_seen"] == 4
    contract = checkpoint["contract"]
    assert contract["task"] == task and contract["variant"] == variant
    assert contract["device_type"] == "cpu"
    assert contract["micro_batch"] == contract["global_batch"] == 2
    config = checkpoint["model_config"]
    assert config["max_length"] == TASK_LENGTHS[task]
    assert (config["hidden_size"], config["n_layers"]) == (32, 2)
    assert config["memory_mode"] == "both"
    assert config["attention_mode"] == "merged"
    assert config["merged_policy"] == "current_preserving"
    assert config["gradient_mode"] == "adjacent"
    assert not config["gate_enabled"] and config["cache_only_probability"] == 0
    assert config["neighbors"] == (variant == "both_aux")
    assert config["neighbor_weight"] == 0.5

    attempts = sorted((run_dir / "logs").glob("attempt-*/resume_attempt.json"))
    assert [json.loads(path.read_text())["resume_step"] for path in attempts] == [0, 1]
    for step in (1, 2):
        validation = json.loads((run_dir / "validation" / f"step-{step:09d}.json").read_text())
        assert validation["protocol"] == "cold_independent"
        assert validation["reset_each_ratio"] is True
        assert validation["num_examples"] == 2
        assert set(validation["ratios"]) == {"0.1", "0.3", "0.5", "0.7"}
        _assert_finite_numbers(validation)
    training_rows = []
    for attempt in attempts:
        with (attempt.parent / "metrics.csv").open(newline="") as stream:
            training_rows.extend(row for row in csv.DictReader(stream) if row.get("train/loss"))
    assert [int(row["step"]) for row in training_rows] == [1, 2]
    for row in training_rows:
        assert math.isfinite(float(row["train/loss"]))
        assert float(row["train/adjacent_edges"]) == 4
        assert float(row["train/trajectory_forwards"]) == 5
        assert (float(row["train/neighbor_loss"]) > 0) == (variant == "both_aux")

    # Observe actual inference inputs, without replacing any model computation:
    # generation starts cold, then carries both returned memories every step.
    calls = []
    original_forward = ReasoningModel.forward

    def observe_forward(self, input_ids, *args, **kwargs):
        calls.append((tuple(input_ids.shape), kwargs.get("previous_step_kv") is not None,
                      kwargs.get("previous_final_hidden") is not None))
        return original_forward(self, input_ids, *args, **kwargs)

    monkeypatch.setattr(ReasoningModel, "forward", observe_forward)
    output = tmp_path / "generated-test.json"
    args = runner.parser().parse_args([
        "evaluate", "--checkpoint", str(checkpoint_path), "--data-dir", str(data_dir),
        "--output", str(output), "--split", "test", "--protocol", "generate",
        "--memory-condition", "correct", "--policy", "top_prob",
        "--examples", "1", "--batch-size", "1", "--seed", "2026",
        "--device", "cpu", "--cpu-threads", "1",
    ])
    runner.evaluate(args)
    result = json.loads(output.read_text())
    assert result["step"] == 2 and result["contract"] == contract
    assert result["arguments"]["protocol"] == "generate"
    assert result["arguments"]["split"] == "test"
    assert result["metrics"]["evaluation"] == "closed_loop_generation"
    assert result["metrics"]["memory_condition"] == "correct"
    assert result["metrics"]["num_examples"] == len(result["examples"]) == 1
    assert result["metrics"]["candidate_k"] == 8
    assert result["metrics"]["seed"] == 2026
    assert result["metrics"]["nfe"] == len(calls) == task_answer_slots(task)
    assert result["examples"][0]["all_slots_completed"]
    assert calls[0] == ((1, TASK_LENGTHS[task]), False, False)
    assert all(call == ((1, TASK_LENGTHS[task]), True, True) for call in calls[1:])
    _assert_finite_numbers(result["metrics"])
