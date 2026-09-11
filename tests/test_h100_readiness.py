"""Small, CPU-only trusted-checkpoint/Arrow transfer readiness checks."""

import copy
import json
import shutil

import datasets
from omegaconf import OmegaConf
import pytest
import torch

from scripts.cloud import check_h100_resume as readiness


def fake_config():
    config = {}
    for dotted, value in readiness.REQUIRED_CONFIG.items():
        node = config
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    config["noise"] = {"type": "loglinear", "sigma_min": 0.0001, "sigma_max": 20}
    config["data"]["cache_dir"] = "/old/server/cache"
    return config


def fake_checkpoint(step=1500):
    return {
        "global_step": step,
        "pytorch-lightning_version": "2.5.0.post0",
        "state_dict": {"backbone.weight": torch.ones(2, 2)},
        "hyper_parameters": {"config": OmegaConf.create(fake_config())},
        "optimizer_states": [{
            "state": {0: {"step": torch.tensor(float(step)),
                          "exp_avg": torch.ones(2, 2),
                          "exp_avg_sq": torch.ones(2, 2)}},
            "param_groups": [{"params": [0], "lr": 0.00018}],
        }],
        "lr_schedulers": [{"last_epoch": step, "_step_count": step + 1}],
        "ema": {"decay": 0.9999, "num_updates": step,
                "shadow_params": [torch.ones(2, 2)]},
        "loops": {"fit_loop": {"epoch_loop.state_dict": {"_batches_that_stepped": step}}},
    }


@pytest.fixture
def transfer(tmp_path):
    checkpoint = tmp_path / "source.ckpt"
    torch.save(fake_checkpoint(), checkpoint)
    data_dir = tmp_path / "source_data"
    for offset, dirname in enumerate(readiness.CACHE_NAMES.values()):
        rows = [[100 * offset + index] * 1024 for index in range(20)]
        dataset = datasets.Dataset.from_dict({"input_ids": rows,
                                             "attention_mask": [[1] * 1024] * 20})
        dataset.save_to_disk(str(data_dir / dirname), num_shards=2)
    return checkpoint, data_dir


def test_cpu_transfer_accepts_relocated_checkpoint_and_arrow(transfer, tmp_path, monkeypatch):
    checkpoint, data_dir = transfer
    manifest = readiness.create_manifest(checkpoint, data_dir)
    destination = tmp_path / "destination"
    shutil.copytree(data_dir, destination)
    copied = tmp_path / "copied.ckpt"
    shutil.copyfile(checkpoint, copied)
    monkeypatch.setattr(readiness, "check_gpu", lambda: pytest.fail("CPU mode queried GPU"))
    result = readiness.verify_transfer(copied, destination, manifest, 5000, cpu_only=True)
    assert result["status"] == "PASS"
    assert result["global_step"] == 1500
    assert result["remaining_steps"] == 3500
    assert result["runtime_python"]["executable"] == readiness.sys.executable
    assert result["runtime_python"]["version"] == readiness.sys.version
    assert result["dataset_rows"] == {"train": 20, "validation": 20}
    assert len(manifest["datasets"]["train"]["sample_rows"]) == 17


@pytest.mark.parametrize("section", ["state_dict", "optimizer_states", "ema", "lr_schedulers", "loops"])
def test_rejects_incomplete_training_checkpoint(tmp_path, section):
    payload = fake_checkpoint()
    payload.pop(section)
    checkpoint = tmp_path / "incomplete.ckpt"
    torch.save(payload, checkpoint)
    with pytest.raises(readiness.ReadinessError):
        readiness.checkpoint_snapshot(checkpoint)


@pytest.mark.parametrize("key,value", [
    ("dcachehooping.adjacent_grad.enabled", False),
    ("dcachehooping.two_forward.enabled", True),
    ("step_memory.detach_between_steps", True),
    ("step_memory.pretrain.t2_loss_weight", 0.5),
    ("dcachehooping.latent_dropout_probability", 0.2),
    ("loader.global_batch_size", 256),
    ("model.length", 512),
    ("data.insert_valid_eos", True),
])
def test_rejects_wrong_experiment_config(tmp_path, key, value):
    payload = fake_checkpoint()
    OmegaConf.update(payload["hyper_parameters"]["config"], key, value)
    checkpoint = tmp_path / "wrong.ckpt"
    torch.save(payload, checkpoint)
    with pytest.raises(readiness.ReadinessError, match="Not canonical adjacent"):
        readiness.checkpoint_snapshot(checkpoint)


def test_digest_mismatch_is_rejected_before_deserialization(transfer, monkeypatch):
    checkpoint, data_dir = transfer
    manifest = readiness.create_manifest(checkpoint, data_dir)
    manifest["checkpoint"]["sha256"] = "0" * 64
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Unverified pickle loaded"))
    with pytest.raises(readiness.ReadinessError, match="SHA-256"):
        readiness.verify_transfer(checkpoint, data_dir, manifest, 5000, cpu_only=True)


def test_missing_shard_and_missing_cache_fail_without_download(transfer):
    _, data_dir = transfer
    train = data_dir / readiness.CACHE_NAMES["train"]
    next(train.glob("*.arrow")).unlink()
    with pytest.raises(readiness.ReadinessError, match="Missing or extra Arrow"):
        readiness.dataset_snapshot(data_dir)
    with pytest.raises(readiness.ReadinessError, match="copy it first"):
        readiness.dataset_snapshot(data_dir / "not_downloaded")


def test_sampled_row_and_metadata_mismatch_rejected(transfer):
    checkpoint, data_dir = transfer
    manifest = readiness.create_manifest(checkpoint, data_dir)
    manifest["datasets"]["train"]["sample_rows"][0]["sha256"] = "0" * 64
    with pytest.raises(readiness.ReadinessError, match="train cache differs"):
        readiness.verify_transfer(checkpoint, data_dir, manifest, 5000, cpu_only=True)


def test_truncated_arrow_fails(transfer):
    _, data_dir = transfer
    train = data_dir / readiness.CACHE_NAMES["train"]
    shard = next(train.glob("*.arrow"))
    shard.write_bytes(b"not-an-arrow-file")
    with pytest.raises(readiness.ReadinessError, match="Cannot load/read"):
        readiness.dataset_snapshot(data_dir)


def test_rejects_unsafe_shard_reference(transfer):
    _, data_dir = transfer
    state_file = data_dir / readiness.CACHE_NAMES["train"] / "state.json"
    state = json.loads(state_file.read_text())
    state["_data_files"][0]["filename"] = "../outside.arrow"
    state_file.write_text(json.dumps(state))
    with pytest.raises(readiness.ReadinessError, match="Unsafe"):
        readiness.dataset_snapshot(data_dir)


def test_completed_or_older_max_steps_is_not_a_resume(transfer):
    checkpoint, _ = transfer
    with pytest.raises(readiness.ReadinessError, match="must exceed"):
        readiness.checkpoint_snapshot(checkpoint, max_steps=1500)


def test_cli_manifest_and_cpu_check(transfer, tmp_path):
    checkpoint, data_dir = transfer
    output = tmp_path / "manifest.json"
    common = ["--checkpoint", str(checkpoint), "--data-dir", str(data_dir)]
    assert readiness.main(["manifest", *common, "--output", str(output)]) == 0
    assert readiness.main(["check", *common, "--manifest", str(output), "--cpu-only"]) == 0
    assert readiness.main(["manifest", *common, "--output", str(output)]) == 2


def test_more_scientific_config_changes_are_preserved(transfer):
    checkpoint, data_dir = transfer
    manifest = readiness.create_manifest(checkpoint, data_dir)
    # Comparison includes settings beyond the explicit launch-recipe constants.
    changed = copy.deepcopy(manifest)
    changed["checkpoint"]["scientific_config"]["noise"]["sigma_max"] = 10
    with pytest.raises(readiness.ReadinessError, match="metadata/config"):
        readiness.verify_transfer(checkpoint, data_dir, changed, 5000, cpu_only=True)


def test_newer_local_checkpoint_requires_explicit_descendant_trust(transfer, tmp_path):
    checkpoint, data_dir = transfer
    manifest = readiness.create_manifest(checkpoint, data_dir)
    newer = tmp_path / "newer.ckpt"
    torch.save(fake_checkpoint(step=2000), newer)
    with pytest.raises(readiness.ReadinessError, match="manifest"):
        readiness.verify_transfer(newer, data_dir, manifest, 5000, cpu_only=True)
    result = readiness.verify_transfer(newer, data_dir, manifest, 5000, cpu_only=True,
                                       allow_descendant=True)
    assert result["global_step"] == 2000
    assert "NEWER LOCAL" in result["checkpoint_verification"]


def test_descendant_same_step_still_needs_exact_source_bytes(transfer, tmp_path):
    checkpoint, data_dir = transfer
    manifest = readiness.create_manifest(checkpoint, data_dir)
    changed = fake_checkpoint()
    changed["state_dict"]["backbone.weight"].zero_()
    other = tmp_path / "same-step.ckpt"
    torch.save(changed, other)
    with pytest.raises(readiness.ReadinessError, match="source-step checkpoint requires exact"):
        readiness.verify_transfer(other, data_dir, manifest, 5000, cpu_only=True,
                                  allow_descendant=True)


def test_descendant_cannot_change_unlisted_scientific_ingredients(transfer, tmp_path):
    checkpoint, data_dir = transfer
    manifest = readiness.create_manifest(checkpoint, data_dir)
    changed = fake_checkpoint(step=2000)
    changed["hyper_parameters"]["config"].noise.sigma_max = 10
    other = tmp_path / "other-science.ckpt"
    torch.save(changed, other)
    with pytest.raises(readiness.ReadinessError, match="scientific configuration"):
        readiness.verify_transfer(other, data_dir, manifest, 5000, cpu_only=True,
                                  allow_descendant=True)


def test_gpu_probe_is_never_called_before_data_validation(transfer, monkeypatch):
    checkpoint, data_dir = transfer
    manifest = readiness.create_manifest(checkpoint, data_dir)
    manifest["datasets"]["validation"]["num_rows"] += 1
    monkeypatch.setattr(readiness, "check_gpu", lambda: pytest.fail("GPU checked before data"))
    with pytest.raises(readiness.ReadinessError, match="validation cache differs"):
        readiness.verify_transfer(checkpoint, data_dir, manifest, 5000)


def test_disk_reserve_is_enforced_before_gpu_probe(transfer, monkeypatch):
    from types import SimpleNamespace

    checkpoint, data_dir = transfer
    manifest = readiness.create_manifest(checkpoint, data_dir)
    monkeypatch.setattr(readiness.shutil, "disk_usage", lambda path: SimpleNamespace(free=2**30))
    monkeypatch.setattr(readiness, "check_gpu", lambda: pytest.fail("GPU checked before disk"))
    with pytest.raises(readiness.ReadinessError, match="at least 15 GiB"):
        readiness.verify_transfer(checkpoint, data_dir, manifest, 5000)


def test_mismatched_runtime_package_fails_even_in_cpu_only_mode(transfer, monkeypatch):
    checkpoint, data_dir = transfer
    manifest = readiness.create_manifest(checkpoint, data_dir)
    different = dict(manifest["source_packages"], datasets="4.0.0")
    monkeypatch.setattr(readiness, "runtime_snapshot", lambda: different)
    monkeypatch.setattr(readiness, "check_gpu", lambda: pytest.fail("GPU queried before packages"))
    for cpu_only in (True, False):
        with pytest.raises(readiness.ReadinessError, match="Runtime package versions differ"):
            readiness.verify_transfer(checkpoint, data_dir, manifest, 5000, cpu_only=cpu_only)


@pytest.mark.parametrize("version,expected", [
    ("2.7.1+cu126", "2.7.1"), ("2.7.1", "2.7.1"),
    ("2.7.1+cpu", "2.7.1+cpu"), ("2.7.1+cu128", "2.7.1+cu128"),
])
def test_only_equivalent_torch_cuda126_label_is_normalized(monkeypatch, version, expected):
    original = readiness.importlib.metadata.version
    monkeypatch.setattr(readiness.importlib.metadata, "version",
                        lambda name: version if name == "torch" else original(name))
    assert readiness.runtime_snapshot()["torch"] == expected


def test_output_mount_space_is_checked_without_creating_directories(transfer, tmp_path, monkeypatch):
    from types import SimpleNamespace

    checkpoint, data_dir = transfer
    manifest = readiness.create_manifest(checkpoint, data_dir)
    mount = tmp_path / "other_mount"
    mount.mkdir()
    output = mount / "not_created" / "run"
    checked = []

    def disk_usage(path):
        checked.append(path)
        return SimpleNamespace(free=(1 if path == mount else 100) * 2**30)

    monkeypatch.setattr(readiness.shutil, "disk_usage", disk_usage)
    monkeypatch.setattr(readiness, "check_gpu", lambda: pytest.fail("GPU queried before output disk"))
    for cpu_only in (True, False):
        with pytest.raises(readiness.ReadinessError, match="near output directory"):
            readiness.verify_transfer(checkpoint, data_dir, manifest, 5000,
                                      cpu_only=cpu_only, output_dir=output)
    assert mount in checked
    assert not output.parent.exists()


def test_output_mount_success_is_reported_in_cpu_mode(transfer, tmp_path, monkeypatch):
    from types import SimpleNamespace

    checkpoint, data_dir = transfer
    manifest = readiness.create_manifest(checkpoint, data_dir)
    output = tmp_path / "not_created" / "run"
    monkeypatch.setattr(readiness.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 2**30))
    result = readiness.verify_transfer(checkpoint, data_dir, manifest, 5000,
                                       cpu_only=True, output_dir=output)
    assert result["output_storage"]["free_gib"] == 100
    assert result["output_storage"]["existing_ancestor"] == str(tmp_path)
    assert not output.parent.exists()
