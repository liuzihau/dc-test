"""CPU-only safety and scheduling checks for immutable Zebra continuations."""

import copy
import json
from pathlib import Path
import subprocess

import pytest
import torch

from reasoning import baseline_queue
from reasoning import zebra_continuation as continuation


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch, tmp_path):
    for key in ("CUDA_VISIBLE_DEVICES", "WORLD_SIZE", "RANK", "LOCAL_RANK",
                "REASONING_MICRO_BATCH", "REASONING_GLOBAL_BATCH", "REASONING_SEED"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(continuation, "ROOT", tmp_path)
    monkeypatch.setattr(baseline_queue, "ROOT", tmp_path)


def write_checkpoint(directory, step, contract=None):
    directory = Path(directory)
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    contract = contract or {"global_batch": 128, "task": "zebra"}
    continuation.atomic_json(directory / "contract.json", contract)
    checkpoint = checkpoints / f"step-{step:09d}-test.pt"
    torch.save({"step": step, "format_version": 1, "contract": contract,
                "model": {"weight": torch.tensor([1., 2.])},
                "optimizer": {"state": {0: {"exp_avg": torch.tensor([.3, .4])}}},
                "rng_by_rank": [{"torch": torch.get_rng_state()}],
                "examples_seen": step * 128}, checkpoint)
    receipt = {"file": checkpoint.name, "step": step,
               "size": checkpoint.stat().st_size,
               "sha256": continuation.digest(checkpoint), "created_ns": step}
    continuation.atomic_json(checkpoint.with_suffix(".pt.json"), receipt)
    last = checkpoints / "last.pt"
    if last.is_symlink():
        last.unlink()
    last.symlink_to(checkpoint.name)
    return checkpoint, receipt


def job(tmp_path, *extra):
    args = continuation.parser().parse_args([
        "run", "--data-root", str(tmp_path / "data"),
        "--output-root", str(tmp_path / "outputs"),
        "--queue-dir", str(tmp_path / "queue"), *extra])
    return continuation.ZebraContinuation(args)


def test_fork_copies_full_state_and_preserves_original(tmp_path):
    source, destination = tmp_path / "original", tmp_path / "continued"
    checkpoint, receipt = write_checkpoint(source, 5000)
    before = {str(path.relative_to(source)): path.read_bytes()
              for path in source.rglob("*") if path.is_file()}
    provenance = continuation.fork_run(source, destination)
    copy_path, copy_receipt = continuation.verified_checkpoint(destination, 5000)
    assert copy_path.read_bytes() == checkpoint.read_bytes()
    assert copy_path.stat().st_ino != checkpoint.stat().st_ino
    assert copy_receipt == receipt
    content = torch.load(copy_path, weights_only=False)
    assert content["step"] == 5000 and content["examples_seen"] == 640000
    torch.testing.assert_close(content["optimizer"]["state"][0]["exp_avg"], torch.tensor([.3, .4]))
    assert content["rng_by_rank"][0]["torch"].numel() > 0
    assert provenance["source_sha256"] == receipt["sha256"]
    assert provenance["full_state"] is True
    # Rotation/deletion in the continuation cannot affect original bytes.
    write_checkpoint(destination, 6000)
    copy_path.unlink()
    copy_path.with_suffix(".pt.json").unlink()
    after = {str(path.relative_to(source)): path.read_bytes()
             for path in source.rglob("*") if path.is_file()}
    assert before == after
    assert continuation.fork_run(source, destination) == provenance


def test_copy_failure_does_not_publish_partial_run(tmp_path, monkeypatch):
    source, destination = tmp_path / "original", tmp_path / "continued"
    checkpoint, _ = write_checkpoint(source, 5000)
    original_copy = continuation.shutil.copy2
    def fail_receipt(src, dst):
        if str(src).endswith(".json"):
            raise OSError("simulated disk full")
        return original_copy(src, dst)
    monkeypatch.setattr(continuation.shutil, "copy2", fail_receipt)
    with pytest.raises(OSError, match="disk full"):
        continuation.fork_run(source, destination)
    assert not destination.exists()
    assert checkpoint.exists()
    assert len(list(tmp_path.glob(".continued.*"))) == 1
    continuation.verified_checkpoint(source, 5000)


def test_existing_provenance_and_contract_are_strict(tmp_path):
    source, destination = tmp_path / "original", tmp_path / "continued"
    write_checkpoint(source, 5000)
    continuation.fork_run(source, destination)
    contract_path = destination / "contract.json"
    contract = json.loads(contract_path.read_text())
    contract["global_batch"] = 64
    continuation.atomic_json(contract_path, contract)
    with pytest.raises(ValueError, match="provenance/contract"):
        continuation.fork_run(source, destination)


def test_fork_refuses_same_directory_and_wrong_source_step(tmp_path):
    source = tmp_path / "original"
    write_checkpoint(source, 4999)
    with pytest.raises(ValueError, match="original run"):
        continuation.fork_run(source, source)
    with pytest.raises(ValueError, match="original run"):
        continuation.fork_run(source, source / "nested-continuation")
    with pytest.raises(ValueError, match="Wrong-step"):
        continuation.fork_run(source, tmp_path / "continued")


def test_receipt_verifies_hash_size_and_checkpoint_name(tmp_path):
    source = tmp_path / "original"
    path, receipt = write_checkpoint(source, 5000)
    broken = copy.deepcopy(receipt)
    broken["file"] = "other.pt"
    continuation.atomic_json(path.with_suffix(".pt.json"), broken)
    with pytest.raises(ValueError, match="corrupt checkpoint"):
        continuation.verified_checkpoint(source)
    continuation.atomic_json(path.with_suffix(".pt.json"), receipt)
    with path.open("ab") as stream:
        stream.write(b"incomplete transfer")
    with pytest.raises(ValueError, match="corrupt checkpoint"):
        continuation.verified_checkpoint(source)


@pytest.mark.parametrize("variant", continuation.VARIANTS)
def test_commands_retain_original_training_recipe(tmp_path, variant):
    active = job(tmp_path)
    command = active.queue(variant).train_command("zebra", variant, active.destination(variant))
    value = lambda flag: command[command.index(flag) + 1]
    assert value("--variant") == variant
    assert value("--micro-batch") == value("--global-batch") == "128"
    assert value("--seed") == "1"
    assert value("--max-steps") == "10000"
    assert value("--data-dir") == str(active.base.data_dir("zebra"))
    assert value("--run-dir") != str(active.source(variant))
    assert "--fresh" not in command and "--stress-memory-routes" not in command
    if variant in ("both", "both_aux"):
        assert value("--gradient-mode") == "adjacent"
        assert value("--merged-policy") == "current_preserving"
        assert value("--neighbor-weight") == ".5" or value("--neighbor-weight") == "0.5"
        assert "--no-robustness" not in command
    else:
        assert value("--gradient-mode") == "detached"
        assert "--no-robustness" in command
        assert "--merged-policy" not in command


def test_recipe_does_not_inherit_unrecorded_batch_or_seed_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("REASONING_MICRO_BATCH", "2")
    monkeypatch.setenv("REASONING_GLOBAL_BATCH", "64")
    monkeypatch.setenv("REASONING_SEED", "27")
    active = job(tmp_path)
    for variant in continuation.VARIANTS:
        command = active.queue(variant).train_command("zebra", variant, active.destination(variant))
        assert command[command.index("--micro-batch") + 1] == "128"
        assert command[command.index("--global-batch") + 1] == "128"
        assert command[command.index("--seed") + 1] == "1"


@pytest.mark.parametrize("hours", ["nan", "inf", "-inf", "0", "-1"])
def test_invalid_time_budget_fails_closed(tmp_path, hours):
    with pytest.raises(ValueError, match="Positive time cap"):
        job(tmp_path, "--hours=" + hours)


def test_round_robin_resumes_completed_rounds_and_preserves_sources(tmp_path, monkeypatch):
    active = job(tmp_path)
    for variant in continuation.VARIANTS:
        write_checkpoint(active.source(variant), 5000)
    monkeypatch.setattr(active.base, "verify_data", lambda task: {"task": task})
    audits, trains, reports, evaluations, audit_reports = [], [], [], [], []
    monkeypatch.setattr(active, "audit", lambda variant, directory, step: audits.append((variant, step)))
    monkeypatch.setattr(active, "report", lambda final=False: reports.append(final))
    monkeypatch.setattr(active, "audit_report", lambda: audit_reports.append(True))
    def execute(command, stage, gpu=False):
        assert gpu
        assert command[2] == "train"
        variant = command[command.index("--variant") + 1]
        directory = Path(command[command.index("--run-dir") + 1])
        target = int(command[command.index("--max-steps") + 1])
        trains.append((variant, target))
        write_checkpoint(directory, target)
    monkeypatch.setattr(active, "execute", execute)
    for queue in active.queues.values():
        monkeypatch.setattr(queue, "evaluate", lambda task, variant, directory: evaluations.append((task, variant)))
    active.run()
    expected = [(variant, step) for step in range(6000, 10001, 1000) for variant in continuation.VARIANTS]
    assert trains == expected
    assert audits == [(v, 5000) for v in continuation.VARIANTS] + [(v, 10000) for v in continuation.VARIANTS]
    assert len(evaluations) == 5
    assert reports == [False] * 5 + [True]
    assert len(audit_reports) == 2
    original_budget = (active.directory / "budget.json").read_bytes()
    for variant in continuation.VARIANTS:
        continuation.verified_checkpoint(active.source(variant), 5000)
    trains.clear()
    active.run()
    assert trains == []
    assert (active.directory / "budget.json").read_bytes() == original_budget
    assert json.loads((active.directory / "status.json").read_text())["status"] == "finished"


def test_expired_budget_prevents_any_child(tmp_path, monkeypatch):
    active = job(tmp_path)
    active.deadline = 10
    monkeypatch.setattr(continuation.time, "time", lambda: 11)
    monkeypatch.setattr(continuation.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not launch"))
    with pytest.raises(TimeoutError, match="budget exhausted"):
        active.execute(["unused"], "expired", gpu=True)


def test_deadline_terminates_only_owned_child(tmp_path, monkeypatch):
    active = job(tmp_path)
    active.directory.mkdir()
    active.deadline = 100
    monkeypatch.setattr(continuation.time, "time", lambda: 90)
    checks = []
    monkeypatch.setattr(active.base, "gpu_check", lambda: checks.append("checked"))
    class Child:
        def __init__(self):
            self.calls = []
            self.stopped = False
        def wait(self, timeout=None):
            self.calls.append(("wait", timeout))
            if not self.stopped:
                raise subprocess.TimeoutExpired("owned", timeout)
            return -15
        def poll(self):
            return None if not self.stopped else -15
        def terminate(self):
            self.calls.append(("terminate",))
            self.stopped = True
        def kill(self):
            pytest.fail("gracefully stopped child should not be killed")
    child = Child()
    monkeypatch.setattr(continuation.subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(subprocess.TimeoutExpired):
        active.execute(["owned"], "timeout", gpu=True)
    assert checks == ["checked"]
    assert child.calls == [("wait", 10), ("terminate",), ("wait", 30)]


def test_nonzero_child_exit_and_changed_queue_spec_fail_closed(tmp_path, monkeypatch):
    active = job(tmp_path)
    active.directory.mkdir()
    active.deadline = 100
    monkeypatch.setattr(continuation.time, "time", lambda: 90)
    class Child:
        def wait(self, timeout=None):
            return 7
    monkeypatch.setattr(continuation.subprocess, "Popen", lambda *args, **kwargs: Child())
    with pytest.raises(RuntimeError, match="exited 7"):
        active.execute(["bad"], "failure")
    changed = {**active.spec, "hours": 24}
    continuation.atomic_json(active.directory / "queue_config.json", changed)
    with pytest.raises(ValueError, match="Changed queue specification"):
        active.run()


def test_audit_command_uses_correct_checkpoint_and_fixed_protocol(tmp_path, monkeypatch):
    active = job(tmp_path, "--audit-examples", "64")
    source = active.source("both")
    checkpoint, _ = write_checkpoint(source, 5000)
    captured = []
    monkeypatch.setattr(active, "execute", lambda command, stage, gpu: captured.append((command, stage, gpu)))
    active.audit("both", source, 5000)
    command, stage, gpu = captured[0]
    value = lambda flag: command[command.index(flag) + 1]
    assert value("--checkpoint") == str(checkpoint)
    assert "--trust-checkpoint" in command
    assert value("--examples") == "64"
    assert value("--batch-size") == "8" and value("--seed") == "2026"
    assert stage == "audit-both-5000" and gpu


def test_final_report_selects_continuations_only_and_step10000(tmp_path, monkeypatch):
    active = job(tmp_path)
    captured = []
    monkeypatch.setattr(active, "execute", lambda command, stage: captured.append((command, stage)))
    active.report(final=True)
    command, stage = captured[-1]
    assert stage == "report-final"
    assert command[command.index("--step") + 1] == "10000"
    assert all(str(active.destination(v)) in command for v in continuation.VARIANTS)
    assert all(str(active.source(v)) not in command for v in continuation.VARIANTS)


def cached_audit(active, variant="both", step=5000):
    source = active.source(variant)
    _, receipt = write_checkpoint(source, step)
    data = active.base.data_dir("zebra")
    continuation.atomic_json(data / "manifest.json", {"task": "zebra"})
    result = {
        "checkpoint_sha256": receipt["sha256"], "step": step,
        "data_sha256": continuation.digest(data / "manifest.json"),
        "arguments": {"examples": active.args.audit_examples, "seed": 2026,
                      "batch_size": 8, "skip_generation": False},
        "protocol": {"splits": ["train", "validation"], "ratios": [1.0, .7, .3]},
        "splits": {name: {"cold": {"permutation_shortcut": {}}}
                   for name in ("train", "validation")},
    }
    output = active.directory / "audits" / f"{variant}-{step}-{receipt['sha256'][:12]}.json"
    continuation.atomic_json(output, result)
    return source, output, result


def test_cached_audit_reuses_exact_protocol_only(tmp_path, monkeypatch):
    active = job(tmp_path)
    source, _, _ = cached_audit(active)
    monkeypatch.setattr(active, "execute", lambda *args, **kwargs: pytest.fail("must reuse verified audit"))
    active.audit("both", source, 5000)


@pytest.mark.parametrize("field,value", [
    (("checkpoint_sha256",), "wrong"), (("data_sha256",), "wrong"),
    (("step",), 4500), (("arguments", "examples"), 3),
    (("arguments", "seed"), 10), (("arguments", "batch_size"), 16),
    (("arguments", "skip_generation"), True),
    (("protocol", "splits"), ["test"]), (("protocol", "ratios"), [.7, .3]),
    (("splits", "train", "cold"), {}),
])
def test_cached_audit_rejects_changed_provenance(tmp_path, monkeypatch, field, value):
    active = job(tmp_path)
    source, output, result = cached_audit(active)
    place = result
    for part in field[:-1]:
        place = place[part]
    place[field[-1]] = value
    continuation.atomic_json(output, result)
    monkeypatch.setattr(active, "execute", lambda *args, **kwargs: pytest.fail("must reject stale audit"))
    with pytest.raises(ValueError, match="Existing diagnostic"):
        active.audit("both", source, 5000)


def official_fixture(tmp_path):
    data = tmp_path / "official-data"
    data.mkdir()
    manifest = {"schema_version": 1, "task": "zebra-official",
                "source": {"kind": "shah_official_zebra_subset", "benchmark_equivalence": False},
                "splits": {}}
    for split, count in (("train", 20000), ("validation", 1000), ("test", 1000)):
        filename = split + ".jsonl"
        (data / filename).write_text("{}\n")
        manifest["splits"][split] = {"filename": filename, "records": count,
                                     "sha256": continuation.digest(data / filename)}
    continuation.atomic_json(data / "manifest.json", manifest)
    active = job(tmp_path, "--official-data-dir", str(data))
    return active, data, manifest


def fake_official_execute(active, monkeypatch, result_change=None):
    calls = []
    def execute(command, stage, gpu=False):
        calls.append((command, stage, gpu))
        value = lambda flag: command[command.index(flag) + 1]
        if command[2] == "train":
            assert gpu
            contract = {"global_batch": 128, "task": "zebra-official", "variant": "vanilla",
                        "data_sha256": continuation.digest(Path(value("--data-dir")) / "manifest.json")}
            write_checkpoint(Path(value("--run-dir")), 5000, contract)
        elif command[2] == "evaluate":
            assert gpu
            checkpoint = Path(value("--checkpoint"))
            contract = json.loads((checkpoint.parent.parent / "contract.json").read_text())
            result = {"step": 5000, "checkpoint": str(checkpoint), "contract": contract,
                      "metrics": {"num_examples": 1000, "valid_solution": 0.},
                      "arguments": {"split": "test", "protocol": "generate", "policy": "top_prob",
                                    "examples": 1000, "seed": 2026, "batch_size": 8,
                                    "memory_condition": "correct"}}
            if result_change is not None:
                result_change(result)
            continuation.atomic_json(Path(value("--output")), result)
        elif command[2] == "plot":
            assert not gpu
        elif stage == "report-official-reference":
            assert not gpu
            assert command[2] == "--evaluation"
        else:
            pytest.fail("Unexpected official stage")
    monkeypatch.setattr(active, "execute", execute)
    return calls


def test_official_reference_fresh_separate_task_directory_and_disclaimer(tmp_path, monkeypatch):
    active, data, _ = official_fixture(tmp_path)
    calls = fake_official_execute(active, monkeypatch)
    active.official_reference()
    assert len(calls) == 4
    training = calls[0][0]
    value = lambda flag: training[training.index(flag) + 1]
    assert value("--task") == "zebra-official" and value("--variant") == "vanilla"
    assert value("--data-dir") == str(data)
    assert value("--max-steps") == "5000"
    assert value("--micro-batch") == value("--global-batch") == "128"
    assert value("--gradient-mode") == "detached" and "--no-robustness" in training
    run = Path(value("--run-dir"))
    assert run.parent.name == "zebra-official"
    assert all(run not in (active.source(v), active.destination(v)) for v in continuation.VARIANTS)
    assert not (run / "continuation_source.json").exists()
    output = json.loads((active.directory / "official-reference/result.json").read_text())
    assert output["task"] == "zebra-official"
    assert "NOT paper96.9 reproduction" in output["limit"]
    assert "do not compare accuracy to synthetic pilot" in output["limit"]
    assert active.spec["official_reference"]["exact_reproduction"] is False


@pytest.mark.parametrize("mutation", ["task", "count", "checksum", "schema", "filename", "source"])
def test_official_manifest_mismatch_prevents_training(tmp_path, monkeypatch, mutation):
    active, data, manifest = official_fixture(tmp_path)
    if mutation == "task":
        manifest["task"] = "zebra"
    elif mutation == "count":
        manifest["splits"]["train"]["records"] = 256
    elif mutation == "schema":
        manifest["schema_version"] = 99
    elif mutation == "filename":
        manifest["splits"]["test"]["filename"] = "../another-test.jsonl"
    elif mutation == "source":
        manifest["source"]["kind"] = "synthetic_pilot"
    else:
        manifest["splits"]["test"]["sha256"] = "bad"
    continuation.atomic_json(data / "manifest.json", manifest)
    monkeypatch.setattr(active, "execute", lambda *args, **kwargs: pytest.fail("must not train wrong data"))
    with pytest.raises(ValueError):
        active.official_reference()


@pytest.mark.parametrize("section,key,value", [
    ("contract", "task", "zebra"), ("contract", "variant", "both"),
    ("arguments", "protocol", "corruption"), ("arguments", "policy", "uniform"),
    ("arguments", "seed", 1), ("arguments", "batch_size", 16),
    ("arguments", "memory_condition", "shuffle_both"),
])
def test_official_evaluation_metadata_fail_closed(tmp_path, monkeypatch, section, key, value):
    active, _, _ = official_fixture(tmp_path)
    fake_official_execute(active, monkeypatch, lambda result: result[section].update({key: value}))
    with pytest.raises(ValueError, match="evaluation provenance"):
        active.official_reference()


def test_no_official_phase_without_explicit_dataset(tmp_path, monkeypatch):
    active = job(tmp_path)
    monkeypatch.setattr(active, "execute", lambda *args, **kwargs: pytest.fail("official data not requested"))
    active.official_reference()
    assert "official_reference" not in active.spec


def test_audit_report_is_cpu_only_and_reads_queue_audits(tmp_path, monkeypatch):
    active = job(tmp_path)
    calls = []
    monkeypatch.setattr(active, "execute", lambda command, stage: calls.append((command, stage)))
    active.audit_report()
    command, stage = calls[0]
    assert stage == "report-audits"
    assert command[command.index("--audit-dir") + 1] == str(active.directory / "audits")
    assert command[command.index("--output-dir") + 1] == str(active.directory / "audit_report")


@pytest.mark.parametrize("extra", [
    ("--official-train-file", "train.pkl"),
    ("--official-test-file", "test.pkl"),
    ("--official-train-file", "train.pkl", "--official-test-file", "test.pkl"),
])
def test_official_source_files_must_be_paired_with_output_dataset(tmp_path, extra):
    with pytest.raises(ValueError, match="both official|separate official-data-dir"):
        job(tmp_path, *extra)


def test_official_missing_manifest_runs_cpu_import_before_training(tmp_path, monkeypatch):
    data = tmp_path / "official-data"
    train_file, test_file = tmp_path / "train.pkl", tmp_path / "test.pkl"
    active = job(tmp_path, "--official-data-dir", str(data),
                 "--official-train-file", str(train_file), "--official-test-file", str(test_file))
    normal_calls = fake_official_execute(active, monkeypatch)
    normal_execute = active.execute
    import_calls = []
    def execute(command, stage, gpu=False):
        if stage == "import-official-source":
            import_calls.append((command, gpu))
            official_fixture(tmp_path)  # Stand in for normalized importer output.
        else:
            normal_execute(command, stage, gpu=gpu)
    monkeypatch.setattr(active, "execute", execute)
    active.official_reference()
    assert len(import_calls) == 1
    command, gpu = import_calls[0]
    assert not gpu
    assert command[command.index("--train-file") + 1] == str(train_file)
    assert command[command.index("--test-file") + 1] == str(test_file)
    assert command[command.index("--train-size") + 1] == "20000"
    assert command[command.index("--valid-size") + 1] == "1000"
    assert command[command.index("--test-size") + 1] == "1000"
    assert command[command.index("--seed") + 1] == "17"
    assert normal_calls[0][1] == "train-official-reference-5000"
    assert active.spec["official_reference"]["train_file"] == str(train_file)
    assert active.spec["official_reference"]["test_file"] == str(test_file)


def test_existing_official_manifest_skips_import_even_with_source_files(tmp_path, monkeypatch):
    _, data, _ = official_fixture(tmp_path)
    active = job(tmp_path, "--official-data-dir", str(data),
                 "--official-train-file", str(tmp_path / "train.pkl"),
                 "--official-test-file", str(tmp_path / "test.pkl"))
    calls = fake_official_execute(active, monkeypatch)
    active.official_reference()
    assert all(stage != "import-official-source" for _, stage, _ in calls)
