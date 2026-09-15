"""Real CPU optimizer-boundary training/recovery tests; no CUDA or downloads."""
import copy
import json
from pathlib import Path

import pytest
import torch

from reasoning.data import ReasoningDataset, prepare_dataset
from reasoning.model import ReasoningModel
from reasoning import runner


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # A CPU job must not initialize a CUDA context even on a shared GPU host.
    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU reasoning test attempted to use CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    monkeypatch.setattr(torch.cuda, "get_rng_state", forbidden)
    old_threads = torch.get_num_threads()
    yield
    torch.set_num_threads(old_threads)


@pytest.fixture
def data_dir(tmp_path):
    output = tmp_path / "dataset"
    prepare_dataset(output, "countdown", train_size=8, valid_size=2, test_size=2, seed=33)
    return output


def train_args(data_dir, run_dir, max_steps=2, variant="both_aux"):
    return runner.parser().parse_args([
        "train", "--task", "countdown", "--variant", variant,
        "--data-dir", str(data_dir), "--run-dir", str(run_dir),
        "--size", "debug", "--device", "cpu", "--precision", "fp32",
        "--global-batch", "2", "--micro-batch", "2",
        "--max-steps", str(max_steps), "--warmup-steps", "2",
        "--val-every", "2", "--validation-examples", "2", "--eval-batch-size", "2",
        "--save-every", "1", "--save-seconds", "0", "--log-every", "1",
        "--seed", "14", "--cpu-threads", "1"])


def assert_tree_equal(actual, expected):
    if torch.is_tensor(actual):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_tree_equal(a, b)
    else:
        assert actual == expected


def test_real_cpu_fit_resume_matches_uninterrupted_optimizer_state(data_dir, tmp_path):
    direct, resumed = tmp_path / "direct", tmp_path / "resumed"
    runner.train(train_args(data_dir, direct, max_steps=3))
    runner.train(train_args(data_dir, resumed, max_steps=1))
    first = runner.load_checkpoint(resumed / "checkpoints/last.pt")
    assert first["step"] == 1 and first["examples_seen"] == 2
    runner.train(train_args(data_dir, resumed, max_steps=3))
    expected = runner.load_checkpoint(direct / "checkpoints/last.pt")
    actual = runner.load_checkpoint(resumed / "checkpoints/last.pt")
    assert actual["step"] == 3 and actual["examples_seen"] == 6
    assert_tree_equal(actual["model"], expected["model"])
    assert_tree_equal(actual["optimizer"], expected["optimizer"])
    assert_tree_equal(actual["rng_by_rank"], expected["rng_by_rank"])
    attempt_metadata = sorted((resumed / "logs").glob("attempt-*/resume_attempt.json"))
    assert [json.loads(p.read_text())["resume_step"] for p in attempt_metadata] == [0, 1]
    assert (resumed / "validation/step-000000002.json").exists()
    assert json.loads((resumed / "status.json").read_text())["status"] == "finished"
    model = ReasoningModel(actual["model_config"])
    model.load_state_dict(actual["model"], strict=True)


def test_global_example_stream_matches_epoch_permutations_and_resume_cursor():
    stream = runner.GlobalExampleStream(5, seed=19)
    expected = []
    for epoch in range(4):
        expected.extend(torch.randperm(5, generator=torch.Generator().manual_seed(19+epoch)).tolist())
    assert stream.indices(0, 20) == expected
    assert stream.indices(7, 11) == expected[7:18]
    assert runner.GlobalExampleStream(5, 19).indices(7, 11) == expected[7:18]
    # World-size two: consecutive local batches cover every logical offset once.
    global_indices = []
    for micro in range(2):
        for rank in range(2):
            global_indices.extend(stream.indices(4+(micro*2+rank)*2, 2))
    assert global_indices == expected[4:12]
    with pytest.raises(ValueError):
        stream.indices(-1, 2)
    with pytest.raises(ValueError):
        runner.GlobalExampleStream(0, 1)


def test_latest_three_checkpoint_retention_and_receipt_verification(data_dir, tmp_path):
    path = tmp_path / "retention"
    runner.train(train_args(data_dir, path, max_steps=5, variant="mdm"))
    receipts = sorted((path / "checkpoints").glob("step-*.pt.json"))
    assert len(receipts) == 3
    assert [json.loads(p.read_text())["step"] for p in receipts] == [3, 4, 5]
    checkpoints = list((path / "checkpoints").glob("step-*.pt"))
    assert len(checkpoints) == 3
    latest = path / "checkpoints/last.pt"
    assert latest.is_symlink() and runner.load_checkpoint(latest)["step"] == 5
    # A tampered receipt is rejected before deserializing the pickle.
    receipt = latest.resolve().with_suffix(".pt.json")
    metadata = json.loads(receipt.read_text())
    metadata["sha256"] = "0" * 64
    receipt.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="corrupted"):
        runner.load_checkpoint(latest)


@pytest.mark.parametrize("field,value", [("lr", .2), ("grad_clip", 7.)])
def test_resume_contract_rejects_changed_training_settings(data_dir, tmp_path, field, value):
    path = tmp_path / "contract"
    args = train_args(data_dir, path, max_steps=1, variant="mdm")
    runner.train(args)
    saved = (path / "checkpoints/last.pt").resolve()
    checksum = runner.digest(saved)
    setattr(args, field, value)
    args.max_steps = 2
    with pytest.raises(ValueError, match="contract differs"):
        runner.train(args)
    assert runner.digest(saved) == checksum
    assert runner.load_checkpoint(saved)["step"] == 1


def test_fresh_training_and_dataset_refuse_to_overwrite(data_dir, tmp_path):
    before = {p.name: runner.digest(p) for p in data_dir.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        prepare_dataset(data_dir, "countdown", train_size=8, valid_size=2, test_size=2)
    assert before == {p.name: runner.digest(p) for p in data_dir.iterdir() if p.is_file()}
    path = tmp_path / "fresh"
    args = train_args(data_dir, path, max_steps=1, variant="vanilla")
    runner.train(args)
    checksum = runner.digest(path / "checkpoints/last.pt")
    args.fresh = True
    with pytest.raises(ValueError, match="NEW run directory"):
        runner.train(args)
    assert checksum == runner.digest(path / "checkpoints/last.pt")


def test_cpu_rng_collection_does_not_touch_available_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    state = runner.rng_state()
    assert state["cuda"] is None
    runner.restore_rng(state)


def test_no_robustness_flag_zeroes_all_source_probabilities(data_dir, tmp_path):
    args = train_args(data_dir, tmp_path / "not-launched")
    args.no_robustness = True
    cfg = runner.build_model_config(args, ReasoningDataset(data_dir))
    for key in ["cache_only_probability", "current_only_probability", "final_dropout", "identity_probability"]:
        assert cfg[key] == 0
    ReasoningModel(cfg)  # Catches misspelled or ignored configuration keys.


def test_invalid_command_geometry_fails_before_writing_run(data_dir, tmp_path):
    path = tmp_path / "invalid"
    args = train_args(data_dir, path)
    args.global_batch, args.micro_batch = 3, 2
    with pytest.raises(ValueError, match="divide evenly"):
        runner.train(args)
    assert not path.exists()
    with pytest.raises(ValueError, match="max_steps must be positive"):
        runner.main(["train", "--task", "countdown", "--variant", "mdm",
                     "--data-dir", str(data_dir), "--run-dir", str(path), "--max-steps", "0"])


def test_plot_discards_abandoned_tail_with_completed_step_convention(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pandas as pd
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'matplotlib'))
    directory = tmp_path / 'trial'
    first = runner.MetricWriter(directory / 'logs/attempt-1', resume_step=0)
    first.log(1, {'train/loss': 4.0})
    first.log(2, {'train/loss': 3.0, 'val/conditional_nll': 2.0})
    first.log(3, {'train/loss': 100.0})
    second = runner.MetricWriter(directory / 'logs/attempt-2', resume_step=2)
    second.log(4, {'train/loss': 2.5, 'val/conditional_nll': 1.5})
    output = tmp_path / 'plot.png'
    runner.plot(SimpleNamespace(runs=[str(directory)], output=str(output), smooth=60,
                                val_metric='val/conditional_nll'))
    frame = pd.read_csv(tmp_path / 'plot-trial.csv')
    assert frame.step.tolist() == [1, 2, 4]
    assert 100.0 not in frame['train/loss'].tolist()
    assert output.exists()
