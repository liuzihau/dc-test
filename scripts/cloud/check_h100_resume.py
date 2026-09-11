#!/usr/bin/env python3
"""Verify an explicit, trusted adjacent-five-forward checkpoint and data transfer.

Checkpoint loading uses pickle (weights_only=False) to restore Lightning and
OmegaConf metadata. Supply ONLY your own trusted checkpoint. The ``check``
command verifies its SHA-256 against your trusted transfer manifest before
deserializing it. No checkpoints or datasets are downloaded, and no GPU is
queried until all CPU-only transfer/configuration checks have passed.

Arrow validation checks every shard's name/size, metadata hashes, dataset row
count and deterministic sampled token rows. It is NOT a full dataset checksum.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SCHEMA_VERSION = 1
CACHE_NAMES = {
    "train": "openwebtext-train_train_bs1024_wrapped_specialFalse.dat",
    "validation": "openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat",
}
SAMPLE_ROWS = 17
# Runtime paths, device count and microbatch may change; scientific ingredients
# below must describe the connected five-forward experiment being migrated.
REQUIRED_CONFIG = {
    "seed": 1,
    "diffusion": "absorbing_state",
    "block_size": 1024,
    "model.name": "small",
    "model.hidden_size": 768,
    "model.length": 1024,
    "model.n_blocks": 12,
    "model.n_heads": 12,
    "model.dropout": 0.1,
    "model.adaln": False,
    "model.attn_backend": "sdpa",
    "algo.name": "mdlm",
    "algo.backbone": "dit",
    "algo.parameterization": "subs",
    "algo.time_conditioning": False,
    "algo.ignore_bos": True,
    "loader.global_batch_size": 512,
    "loader.eval_global_batch_size": 512,
    "step_memory.enabled": True,
    "step_memory.use_previous_kv": True,
    "step_memory.detach_between_steps": False,
    "step_memory.spatial_rope_dim": 48,
    "step_memory.temporal_rope_dim": 16,
    "step_memory.gate.enabled": True,
    "step_memory.gate.init": 0.1,
    "step_memory.pretrain.enabled": True,
    "step_memory.pretrain.step_size_min": 0.025,
    "step_memory.pretrain.step_size_max": 0.1,
    "step_memory.pretrain.max_t0_mask_ratio": 0.9975,
    "step_memory.pretrain.full_loss_weight": 0.05,
    "step_memory.pretrain.t0_loss_weight": 0.1,
    "step_memory.pretrain.t1_loss_weight": 0.2,
    "step_memory.pretrain.t2_loss_weight": 1.0,
    "step_memory.pretrain.t3_loss_weight": 0.7,
    "step_memory.pretrain.teacher_token_probability": 1.0,
    "step_memory.pretrain.source_dropout.enabled": True,
    "step_memory.pretrain.source_dropout.cache_only_probability": 0.2,
    "step_memory.pretrain.source_dropout.current_only_probability": 0.05,
    "step_memory.pretrain.source_dropout.warmup_steps": 1000,
    "step_memory.pretrain.identity.enabled": True,
    "step_memory.pretrain.identity.batch_probability": 0.25,
    "step_memory.pretrain.identity.margin": 0.05,
    "step_memory.pretrain.identity.weight": 0.1,
    "step_memory.rollout.enabled": False,
    "dcachehooping.enabled": True,
    "dcachehooping.adjacent_grad.enabled": True,
    "dcachehooping.two_forward.enabled": False,
    "dcachehooping.status_embedding.enabled": False,
    "dcachehooping.latent_dropout_probability": 0.1,
    "dcachehooping.latent_mask_probability": 0.0,
    "dcachehooping.latent_mask_loss_weight": 0.0,
    "dcachehooping.tentative.enabled": False,
    "dcachehooping.tentative.batch_probability": 0.0,
    "dcachehooping.tentative.loss_weight": 0.0,
    "dcachehooping.confidence.enabled": False,
    "dcachehooping.confidence.loss_weight": 0.0,
    "dcachehooping.identity_final_probability": 0.5,
    "training.ema": 0.9999,
    "training.resample": False,
    "training.from_pretrained": None,
    "training.objective_matched_multistate.enabled": False,
    "optim.lr": 0.0003,
    "lr_scheduler._target_": "transformers.get_constant_schedule_with_warmup",
    "lr_scheduler.num_warmup_steps": 2500,
    "trainer.gradient_clip_val": 1.0,
    "trainer.precision": "bf16-mixed",
    "data.train": "openwebtext-train",
    "data.valid": "openwebtext-valid",
    "data.tokenizer_name_or_path": "gpt2",
    "data.wrap": True,
    "data.streaming": False,
    "data.insert_train_eos": True,
    "data.insert_valid_eos": False,
    "data.insert_train_special": False,
    "data.insert_valid_special": False,
}


class ReadinessError(ValueError):
    """Missing, altered or incompatible input; never silently start fresh."""


def require(condition, message):
    if not condition:
        raise ReadinessError(message)


def lookup(config, dotted):
    value = config
    for part in dotted.split("."):
        require(isinstance(value, dict) and part in value,
                "Missing checkpoint configuration: " + dotted)
        value = value[part]
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scientific_config(config):
    for key, expected in REQUIRED_CONFIG.items():
        actual = lookup(config, key)
        equal = actual == expected
        if isinstance(expected, bool):
            equal = type(actual) is bool and equal
        require(equal, f"Not canonical adjacent five-forward: {key}="
                f"{actual!r}, expected {expected!r}")
    snapshot = {
        key: config[key] for key in (
            "seed", "diffusion", "block_size", "model", "algo", "noise",
            "step_memory", "dcachehooping", "training", "optim", "lr_scheduler")
    }
    snapshot["data"] = {key: value for key, value in config["data"].items()
                        if key != "cache_dir"}
    snapshot["loader"] = {key: config["loader"][key] for key in (
        "global_batch_size", "eval_global_batch_size")}
    snapshot["trainer"] = {key: config["trainer"][key] for key in (
        "precision", "gradient_clip_val")}
    return snapshot


def checkpoint_snapshot(path, expected=None, max_steps=None, allow_descendant=False):
    path = Path(path)
    require(path.is_file(), f"Checkpoint is missing: {path}")
    before = path.stat()
    require(before.st_size > 0, f"Checkpoint is empty: {path}")
    # Verify bytes BEFORE unsafe deserialization on the destination server.
    digest = file_sha256(path)
    exact_transfer = expected is None or digest == expected["sha256"]
    if expected is not None:
        if exact_transfer or not allow_descendant:
            require(before.st_size == expected["size_bytes"],
                    "Checkpoint file size differs from transfer manifest")
            require(exact_transfer,
                    "Checkpoint SHA-256 differs from transfer manifest; not loaded")
    import torch
    from omegaconf import OmegaConf

    print("Loading explicitly supplied TRUSTED checkpoint on CPU.", file=sys.stderr)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    require(isinstance(checkpoint, dict), "Checkpoint must be a Lightning mapping")
    step = checkpoint.get("global_step")
    require(type(step) is int and step > 0, "Checkpoint needs positive global_step")
    if max_steps is not None:
        require(max_steps > step,
                f"max_steps={max_steps} must exceed checkpoint global_step={step}")
    state = checkpoint.get("state_dict")
    require(isinstance(state, dict) and bool(state), "Missing model state_dict")
    optimizer = checkpoint.get("optimizer_states")
    require(isinstance(optimizer, list) and bool(optimizer),
            "Missing full optimizer_states; weights-only restart is not a resume")
    for item in optimizer:
        require(bool(item.get("state")) and bool(item.get("param_groups")),
                "Optimizer state/parameter groups are empty")
        states = list(item["state"].values())
        require(any(float(value.get("step", 0)) > 0
                    and torch.is_tensor(value.get("exp_avg"))
                    and value["exp_avg"].numel() > 0
                    and torch.is_tensor(value.get("exp_avg_sq"))
                    and value["exp_avg_sq"].numel() > 0 for value in states),
                "Optimizer has no populated Adam moments/positive update step")
    schedulers = checkpoint.get("lr_schedulers")
    require(isinstance(schedulers, list) and bool(schedulers)
            and all(item.get("last_epoch", 0) > 0 for item in schedulers),
            "Missing/non-progressed learning-rate scheduler state")
    ema = checkpoint.get("ema")
    require(isinstance(ema, dict) and ema.get("num_updates", 0) > 0
            and bool(ema.get("shadow_params"))
            and all(torch.is_tensor(value) and value.numel() > 0
                    for value in ema["shadow_params"]),
            "Missing/non-progressed EMA shadow parameters")
    require(bool(checkpoint.get("loops")), "Missing Lightning loop state")
    config = checkpoint.get("hyper_parameters", {}).get("config")
    require(config is not None, "Missing checkpoint hyper_parameters.config")
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=False)
    require(isinstance(config, dict), "Checkpoint config is not a mapping")
    scientific = scientific_config(config)
    require(ema.get("decay") == lookup(config, "training.ema"),
            "EMA decay disagrees with checkpoint scientific configuration")
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) ==
            (after.st_size, after.st_mtime_ns), "Checkpoint changed during inspection")
    result = {
        "size_bytes": before.st_size, "sha256": digest, "global_step": step,
        "lightning_version": checkpoint.get("pytorch-lightning_version"),
        "model_state_entries": len(state),
        "optimizer_state_entries": sum(len(item["state"]) for item in optimizer),
        "ema_updates": int(ema["num_updates"]),
        "ema_shadow_count": len(ema["shadow_params"]),
        "scheduler_last_epochs": [int(item["last_epoch"]) for item in schedulers],
        "scientific_config": scientific,
        "scientific_config_sha256": json_hash(scientific),
    }
    if expected is not None:
        if exact_transfer:
            require(result == expected, "Checkpoint metadata/config differs from manifest")
        else:
            require(step > expected["global_step"],
                    "A descendant must be newer; source-step checkpoint requires exact SHA-256")
            require(scientific == expected["scientific_config"],
                    "Descendant scientific configuration differs from transfer manifest")
            require(result["scientific_config_sha256"] == expected["scientific_config_sha256"],
                    "Descendant scientific configuration fingerprint differs from manifest")
    return result


def dataset_snapshot(data_dir, compact_expected=None, original_train=None):
    import datasets

    snapshots = {}
    for split, dirname in CACHE_NAMES.items():
        if split == 'train' and (Path(data_dir) / 'compact_train.json').exists():
            require(compact_expected is not None and original_train is not None,
                    'Compact data requires its exported trusted transfer manifest')
            from compact_training import verify_compact
            require(compact_expected['original_num_rows'] == original_train['num_rows'],
                    'Compact logical dataset length differs from source manifest')
            verify_compact(data_dir, compact_expected)
            snapshots[split] = original_train
            continue
        path = Path(data_dir) / dirname
        require(path.is_dir(), f"Prepared {split} cache missing: {path}; copy it first")
        metadata = {}
        for name in ("dataset_info.json", "state.json"):
            item = path / name
            require(item.is_file() and item.stat().st_size > 0,
                    f"Incomplete {split} cache: missing/empty {name}")
            metadata[name] = {"size_bytes": item.stat().st_size,
                              "sha256": file_sha256(item)}
        with (path / "state.json").open() as handle:
            state = json.load(handle)
        require(isinstance(state.get("_fingerprint"), str) and state["_fingerprint"],
                f"Missing {split} dataset fingerprint in state.json")
        data_files = state.get("_data_files", [])
        require(bool(data_files), f"No Arrow shards listed in {split} state.json")
        names = [item.get("filename") for item in data_files]
        require(all(isinstance(name, str) and name.endswith(".arrow")
                    and Path(name).name == name for name in names),
                f"Unsafe or non-Arrow shard filename in {split} state.json")
        require(len(names) == len(set(names)), f"Duplicate {split} shard names")
        actual_names = {item.name for item in path.glob("*.arrow")}
        require(set(names) == actual_names,
                f"Missing or extra Arrow shards in {split} cache")
        shards = []
        for name in names:
            shard = path / name
            require(shard.is_file() and shard.stat().st_size > 0,
                    f"Missing/empty Arrow shard: {shard}")
            shards.append({"filename": name, "size_bytes": shard.stat().st_size})
        try:
            dataset = datasets.load_from_disk(str(path), keep_in_memory=False)
            require(isinstance(dataset, datasets.Dataset),
                    f"{split} must be a Dataset, not DatasetDict")
            rows = len(dataset)
            require(rows > 0, f"{split} dataset has no rows")
            require("input_ids" in dataset.column_names,
                    f"{split} dataset has no input_ids")
            indices = sorted({i * (rows - 1) // (SAMPLE_ROWS - 1)
                              for i in range(SAMPLE_ROWS)})
            samples = []
            dataset = dataset.with_format(None)
            for index in indices:
                row = dataset[index]
                tokens = row["input_ids"]
                require(isinstance(tokens, list) and len(tokens) == 1024
                        and all(type(token) is int for token in tokens),
                        f"{split} row {index}: expected 1024 packed integer tokens")
                samples.append({"row": index, "sha256": json_hash(row)})
            snapshots[split] = {
                "directory": dirname, "num_rows": rows,
                "features": dataset.features.to_dict(),
                "state_fingerprint": state.get("_fingerprint"),
                "loaded_fingerprint": dataset._fingerprint,
                "metadata": metadata, "shards": shards, "sample_rows": samples,
                "size_bytes": sum(item["size_bytes"] for item in shards)
                              + sum(item["size_bytes"] for item in metadata.values()),
            }
        except ReadinessError:
            raise
        except Exception as exc:
            raise ReadinessError(f"Cannot load/read complete {split} Arrow cache: {exc}") from exc
        finally:
            if "dataset" in locals():
                del dataset
    return snapshots


def runtime_snapshot():
    versions = {name: importlib.metadata.version(name) for name in (
        "torch", "lightning", "datasets", "transformers", "numpy", "pyarrow")}
    # The original PyPI torch build reports 2.7.1, whereas the equivalent
    # official cu126-index wheel reports 2.7.1+cu126. Do not normalize CPU or
    # other CUDA-wheel labels: they are not the recorded producer runtime.
    versions["torch"] = versions["torch"].removesuffix("+cu126")
    return versions


def storage_snapshot(path):
    """Inspect the nearest existing directory without creating output paths."""
    requested = Path(path).expanduser().resolve()
    ancestor = requested
    while not ancestor.exists():
        ancestor = ancestor.parent
    require(ancestor.is_dir(), f"Storage path is not a directory: {ancestor}")
    return {"requested_path": str(requested), "existing_ancestor": str(ancestor),
            "free_gib": shutil.disk_usage(ancestor).free / 2**30}


def check_gpu():
    import torch

    require(torch.__version__.split("+")[0] == "2.7.1", "Use torch==2.7.1")
    require(torch.version.cuda == "12.6", "Use the PyTorch CUDA 12.6 wheel")
    require(torch.cuda.is_available(), "CUDA unavailable; check the cloud driver/runtime")
    require(torch.cuda.device_count() == 1,
            "Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    require(torch.cuda.is_bf16_supported(), "Selected GPU does not support BF16")
    properties = torch.cuda.get_device_properties(0)
    require("H100" in properties.name, f"Expected H100, found {properties.name}")
    require(importlib.metadata.version("lightning") == "2.5.0.post0",
            "Use lightning==2.5.0.post0 for checkpoint continuation")
    # Exercise the actual BF16 attention operation/backward, not only metadata.
    tensors = [torch.randn(1, 2, 8, 64, device="cuda:0", dtype=torch.bfloat16,
                           requires_grad=True) for _ in range(3)]
    output = torch.nn.functional.scaled_dot_product_attention(*tensors)
    output.float().square().mean().backward()
    torch.cuda.synchronize()
    require(torch.isfinite(output).all().item()
            and all(value.grad is not None and torch.isfinite(value.grad).all().item()
                    for value in tensors), "BF16 SDPA forward/backward produced nonfinite values")
    return {"name": properties.name, "memory_gib": properties.total_memory / 2**30,
            "compute_capability": [properties.major, properties.minor],
            "bf16_sdpa_forward_backward": "PASS"}


def create_manifest(checkpoint, data_dir):
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verification": "Full checkpoint SHA-256; exact Arrow names/sizes and metadata "
                        "hashes; deterministic sampled packed rows, NOT all-token hashing.",
        "sample_rows_per_split": SAMPLE_ROWS,
        "checkpoint": checkpoint_snapshot(checkpoint),
        "datasets": dataset_snapshot(data_dir),
        "source_packages": runtime_snapshot(),
    }


def verify_transfer(checkpoint, data_dir, manifest, max_steps, cpu_only=False,
                    allow_descendant=False, output_dir=None):
    require(manifest.get("schema_version") == SCHEMA_VERSION,
            "Unsupported transfer-manifest schema")
    require(manifest.get("sample_rows_per_split") == SAMPLE_ROWS,
            "Unexpected token-row sampling specification")
    ckpt = checkpoint_snapshot(checkpoint, manifest["checkpoint"], max_steps=max_steps,
                               allow_descendant=allow_descendant)
    compact = manifest.get('compact_training')
    if compact is not None:
        require((Path(data_dir) / 'compact_train.json').is_file(), 'Missing compact training descriptor')
        require(compact['start_step'] <= ckpt['global_step'] <= compact['end_step'],
                'Checkpoint lies outside compact training coverage')
        require(compact['sampler_seed'] == lookup(manifest['checkpoint']['scientific_config'], 'seed'),
                'Compact sampler seed differs from original Lightning seed')
    data = dataset_snapshot(data_dir, manifest.get('compact_training'), manifest['datasets']['train'])
    for split in CACHE_NAMES:
        require(data[split] == manifest["datasets"][split],
                f"Transferred {split} cache differs from manifest (metadata/shards/rows)")
    packages = runtime_snapshot()
    require(packages == manifest["source_packages"],
            "Runtime package versions differ from source manifest: "
            f"expected {manifest['source_packages']}, found {packages}")
    checkpoint_storage = storage_snapshot(Path(checkpoint).parent)
    free_gib = checkpoint_storage["free_gib"]
    output_storage = storage_snapshot(output_dir) if output_dir is not None else None
    if not cpu_only:
        require(free_gib >= 15,
                f"Only {free_gib:.1f} GiB free near checkpoint; keep at least 15 GiB "
                "available for periodic full-state checkpoints")
    if output_storage is not None:
        require(output_storage["free_gib"] >= 15,
                f"Only {output_storage['free_gib']:.1f} GiB free near output directory "
                f"{output_storage['requested_path']}; keep at least 15 GiB available "
                "for periodic full-state checkpoints")
    logical_gib = sum(item['size_bytes'] for item in data.values()) / 2**30
    stored_gib = ((sum(v['size_bytes'] for v in compact['files'].values())
                   + data['validation']['size_bytes']) / 2**30
                  if compact is not None else logical_gib)
    result = {
        "status": "PASS", "global_step": ckpt["global_step"],
        "max_steps": max_steps, "remaining_steps": max_steps - ckpt["global_step"],
        "dataset_rows": {key: item["num_rows"] for key, item in data.items()},
        "dataset_gib": stored_gib,
        "original_dataset_gib": logical_gib,
        "free_gib_near_checkpoint": free_gib,
        "output_storage": output_storage,
        "runtime_packages": packages,
        "runtime_python": {"version": sys.version, "executable": sys.executable},
        "compact_training": compact,
        "physical_dataset_gib": stored_gib,
        "checkpoint_verification": ("EXACT SOURCE TRANSFER" if ckpt["sha256"] ==
                                    manifest["checkpoint"]["sha256"] else
                                    "TRUSTED NEWER LOCAL CHECKPOINT; scientific config matched"),
        "gpu_check": "SKIPPED (CPU-only; not GPU launch readiness)" if cpu_only else check_gpu(),
        "data_integrity_limit": "Sampled token checks are not a full dataset checksum.",
        "resume_limit": "Full-state continuation, not bitwise replay across GPU/world-size changes.",
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("manifest", help="Snapshot your trusted source checkpoint/data")
    check = commands.add_parser("check", help="Verify your trusted transfer before cloud resume")
    for command in (generate, check):
        command.add_argument("--checkpoint", required=True,
                             help="Your TRUSTED full Lightning checkpoint (pickle is loaded)")
        command.add_argument("--data-dir", required=True)
    generate.add_argument("--output", required=True, help="New manifest path (will not overwrite)")
    check.add_argument("--manifest", required=True, help="Your trusted source-transfer manifest")
    check.add_argument("--max-steps", type=int, default=5000)
    check.add_argument("--cpu-only", action="store_true")
    check.add_argument("--output-dir", help="Check free space on the actual output mount "
                       "without creating directories (also enforced in CPU-only mode)")
    check.add_argument("--allow-descendant", action="store_true",
                       help="Trust a newer checkpoint produced by your local cloud run; "
                            "verify configuration, not source-checkpoint byte identity")
    args = parser.parse_args(argv)
    try:
        if args.command == "manifest":
            output = Path(args.output)
            require(not output.exists(), f"Refusing to overwrite existing manifest: {output}")
            result = create_manifest(args.checkpoint, args.data_dir)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x") as handle:
                json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
            print(json.dumps({"status": "PASS", "manifest": str(output),
                              "global_step": result["checkpoint"]["global_step"],
                              "dataset_rows": {key: item["num_rows"]
                                               for key, item in result["datasets"].items()}}, indent=2))
        else:
            with Path(args.manifest).open() as handle:
                manifest = json.load(handle)
            print(json.dumps(verify_transfer(args.checkpoint, args.data_dir, manifest,
                                            args.max_steps, args.cpu_only,
                                            args.allow_descendant, args.output_dir), indent=2))
    except (ReadinessError, OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
