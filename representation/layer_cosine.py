#!/usr/bin/env python3
"""Matched layer-update cosine probe for the modular Sudoku/Zebra baselines.

For each Transformer block l, this measures

    cos(h^(l-1)_i, h^l_i)

at every eligible solution-token position i.  MDM and MDM+NP receive the
same held-out examples and exactly the same Bernoulli corruption masks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / "third_party/reasoning_with_latent_tokens"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTHOR))

from zebra.runtime import install_no_cudagraph_compile

install_no_cudagraph_compile()

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from zebra.model import ZebraMDM


TASKS = ("sudoku", "zebra")
VARIANTS = ("mdm", "mdm_np")
DISPLAY = {"mdm": "MDM", "mdm_np": "MDM + NP"}
COLORS = {"mdm": "#2878B5", "mdm_np": "#D95319"}
GROUP_STYLES = {"masked": "-", "revealed": "--"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_step(path: Path) -> int:
    # Checkpoints are trusted products of this repository's own training runs.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["global_step"])


def default_checkpoints() -> dict[str, dict[str, Path]]:
    return {
        "sudoku": {
            "mdm": ROOT / "outputs/sudoku/mdm-np-20ep/mdm/checkpoints/25-70500.ckpt",
            "mdm_np": ROOT / "outputs/sudoku/mdm-np-20ep/mdm_np/checkpoints/25-70500.ckpt",
        },
        # Epoch 59 is the latest checkpoint available for BOTH variants.  Using
        # it avoids comparing unequal optimization budgets.
        "zebra": {
            "mdm": ROOT / "outputs/zebra/mdm-np-40ep/mdm/checkpoints/78-172870.ckpt",
            "mdm_np": ROOT / "outputs/zebra/mdm-np-40ep/mdm_np/checkpoints/78-172870.ckpt",
        },
    }


def build_config(task: str, variant: str, checkpoint: Path, output: Path):
    entrypoint = __import__(f"{task}.entrypoint", fromlist=["make_config"])
    args = SimpleNamespace(
        run=(output / "config" / task / variant),
        recipe=ROOT / task / "configs" / f"{variant}.yaml",
        seed=1,
        resume=checkpoint,
        workers=0,
        microbatch=128,
        devices=1,
        target_steps=1,
        checkpoint_interval=1,
        stage="evaluate",
        eval_batches=1,
        eval_batch_size=128,
        candidate_window=0,
        generation_layout="author",
        smoke=False,
    )
    return entrypoint.make_config(args)


def held_out_batch(config, upstream, tokenizer, samples: int):
    _, loader = upstream.dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=None)
    batch = next(iter(loader))
    if batch["input_ids"].shape[0] < samples:
        raise RuntimeError(f"Held-out batch has only {batch['input_ids'].shape[0]} rows")
    return {key: value[:samples].clone() for key, value in batch.items()
            if torch.is_tensor(value)}


def fixed_corruptions(x0: torch.Tensor, solution: torch.Tensor,
                      levels: list[float], seed: int):
    """Create nested, task-level masks shared by MDM and MDM+NP.

    A single U(0,1) draw per token makes masks at different t values nested,
    which reduces irrelevant Monte Carlo variation in comparisons across t.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    uniforms = torch.rand(x0.shape, generator=generator)
    return {t: solution & (uniforms < t) for t in levels}


def capture_block_cosines(model: ZebraMDM, xt: torch.Tensor, t: float):
    captured: list[torch.Tensor] = []
    handles = []

    def hook(_module, inputs, output):
        before = inputs[0].detach().float()
        after = output.detach().float()
        captured.append(F.cosine_similarity(before, after, dim=-1).cpu())

    for block in model.backbone.blocks:
        handles.append(block.register_forward_hook(hook))
    try:
        # t denotes the effective mask probability.  At t=1 we use the
        # schedule's finite endpoint for conditioning while forcing all target
        # positions to MASK.  In these runs time_conditioning is disabled, but
        # retaining the correct convention makes the probe future-proof.
        alpha = max(1.0 - t, float(model.noise.eps))
        sigma = xt.new_full((xt.shape[0], 1), -math.log(alpha), dtype=torch.float32)
        positions = torch.arange(xt.shape[1], device=xt.device).expand_as(xt)
        with torch.inference_mode():
            model.forward(xt, sigma=sigma, sort_idx=positions)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(model.backbone.blocks):
        raise RuntimeError(f"Captured {len(captured)} blocks, expected {len(model.backbone.blocks)}")
    return torch.stack(captured, dim=1)  # [examples, layers, positions]


def summarize(cosines: torch.Tensor, position_mask: torch.Tensor):
    """Average positions within each example, then compute uncertainty over examples."""
    position_mask = position_mask.cpu().bool()
    per_example = []
    token_count = []
    for layer in range(cosines.shape[1]):
        values = cosines[:, layer, :]
        counts = position_mask.sum(dim=1)
        means = (values * position_mask).sum(dim=1) / counts.clamp_min(1)
        means = means[counts > 0]
        n = int(means.numel())
        mean = float(means.mean()) if n else float("nan")
        std = float(means.std(unbiased=True)) if n > 1 else float("nan")
        se = std / math.sqrt(n) if n > 1 else float("nan")
        per_example.append((mean, std, se, n))
        token_count.append(int(position_mask.sum()))
    return per_example, token_count


def load_model(config, tokenizer, checkpoint: Path, device: torch.device):
    model = ZebraMDM.load_from_checkpoint(
        checkpoint, tokenizer=tokenizer, config=config, map_location="cpu")
    model.to(device)
    if model.ema:
        model.ema.move_shadow_params_to_device(device)
    model._eval_mode()
    return model


def run_task(task: str, checkpoints: dict[str, Path], levels: list[float],
             samples: int, seed: int, batch_size: int, device: torch.device,
             output: Path):
    configs = {}
    upstream_modules = {}
    tokenizers = {}
    batch = None
    for variant in VARIANTS:
        config, upstream = build_config(task, variant, checkpoints[variant], output)
        tokenizer = upstream.dataloader.get_tokenizer(config)
        configs[variant], upstream_modules[variant], tokenizers[variant] = config, upstream, tokenizer
        candidate = held_out_batch(config, upstream, tokenizer, samples)
        if batch is None:
            batch = candidate
        elif not torch.equal(batch["input_ids"], candidate["input_ids"]) or not torch.equal(
                batch["loss_mask"], candidate["loss_mask"]):
            raise RuntimeError(f"{task}: MDM and MDM+NP did not load identical held-out rows")

    x0 = batch["input_ids"].long()
    solution = batch["loss_mask"].bool()
    special = torch.zeros_like(solution)
    for token_id in tokenizers["mdm"].all_special_ids:
        special |= x0.eq(int(token_id))
    valid_solution = solution & ~special
    corruption = fixed_corruptions(
        x0, valid_solution, levels,
        seed=seed + (1000 if task == "zebra" else 0))
    rows = []

    for variant in VARIANTS:
        model = load_model(configs[variant], tokenizers[variant], checkpoints[variant], device)
        for t in levels:
            parts = []
            for start in range(0, samples, batch_size):
                stop = min(samples, start + batch_size)
                clean = x0[start:stop].to(device)
                mask = corruption[t][start:stop].to(device)
                xt = torch.where(mask, model.mask_index, clean)
                parts.append(capture_block_cosines(model, xt, t))
            cosines = torch.cat(parts, dim=0)
            groups = {
                "all_solution": valid_solution,
                "masked": valid_solution & corruption[t],
                "revealed": valid_solution & ~corruption[t],
            }
            for group, mask in groups.items():
                statistics, token_counts = summarize(cosines, mask)
                for layer, ((mean, std, se, n), n_tokens) in enumerate(
                        zip(statistics, token_counts), start=1):
                    rows.append({
                        "task": task,
                        "t": t,
                        "variant": variant,
                        "layer": layer,
                        "token_group": group,
                        "mean_cosine": mean,
                        "std_across_examples": std,
                        "standard_error": se,
                        "ci95_low": mean - 1.96 * se if math.isfinite(se) else float("nan"),
                        "ci95_high": mean + 1.96 * se if math.isfinite(se) else float("nan"),
                        "n_examples": n,
                        "n_tokens": n_tokens,
                        "observed_mask_fraction": float(
                            corruption[t][valid_solution].float().mean()),
                    })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows, {
        "n_examples": samples,
        "n_valid_solution_tokens": int(valid_solution.sum()),
        "sequence_length": int(x0.shape[1]),
    }


def _series(rows, task, t, variant, group):
    return sorted((r for r in rows if r["task"] == task and r["t"] == t
                   and r["variant"] == variant and r["token_group"] == group),
                  key=lambda r: r["layer"])


def draw_panel(ax, rows, task: str, t: float, y_limits=None, legend=True):
    for variant in VARIANTS:
        for group in ("masked", "revealed"):
            data = _series(rows, task, t, variant, group)
            if not data or not all(math.isfinite(r["mean_cosine"]) for r in data):
                continue
            x = np.array([r["layer"] for r in data])
            y = np.array([r["mean_cosine"] for r in data])
            low = np.array([r["ci95_low"] for r in data])
            high = np.array([r["ci95_high"] for r in data])
            label = f"{DISPLAY[variant]} — {group}"
            ax.plot(x, y, GROUP_STYLES[group], color=COLORS[variant], marker="o",
                    linewidth=2.0, markersize=4, label=label)
            ax.fill_between(x, low, high, color=COLORS[variant], alpha=.10)
    ax.set_title(f"{task.title()} — t={t:g} ({100*t:.0f}% target masking)")
    ax.set_xlabel("Transformer block ℓ")
    ax.set_ylabel(r"cos($h^{\ell-1}_i$, $h^{\ell}_i$)")
    ax.set_xticks(range(1, 7))
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(alpha=.22)
    if legend:
        ax.legend(fontsize=8, frameon=False)


def make_plots(rows, levels: list[float], output: Path):
    finite = [r["mean_cosine"] for r in rows if r["token_group"] in ("masked", "revealed")
              and math.isfinite(r["mean_cosine"])]
    low, high = min(finite), max(finite)
    margin = max(.003, .08 * (high - low))
    y_limits = (low - margin, min(1.0, high + margin))

    individual = output / "individual"
    individual.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        for t in levels:
            fig, ax = plt.subplots(figsize=(6.6, 4.4))
            draw_panel(ax, rows, task, t, y_limits=y_limits)
            fig.tight_layout()
            fig.savefig(individual / f"{task}_t{t:g}.png", dpi=220)
            fig.savefig(individual / f"{task}_t{t:g}.pdf")
            plt.close(fig)

    for task in TASKS:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        for ax, t in zip(axes.flat, levels):
            draw_panel(ax, rows, task, t, y_limits=y_limits, legend=t == levels[0])
        fig.suptitle(f"{task.title()}: layer-update cosine on 100 held-out examples", y=.995)
        fig.tight_layout()
        fig.savefig(output / f"{task}_four_mask_levels.png", dpi=220)
        fig.savefig(output / f"{task}_four_mask_levels.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(21, 8), sharex=True, sharey=True)
    for row_index, task in enumerate(TASKS):
        for column, t in enumerate(levels):
            draw_panel(axes[row_index, column], rows, task, t,
                       y_limits=y_limits, legend=(row_index == 0 and column == 0))
    fig.suptitle("Layer-update cosine: MDM versus MDM + neighbor prediction", y=.995)
    fig.tight_layout()
    fig.savefig(output / "all_eight_panels.png", dpi=220)
    fig.savefig(output / "all_eight_panels.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/representation/layer-cosine-mdm-vs-np")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--levels", nargs="+", type=float, default=[1.0, .75, .5, .25])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.samples != 100:
        print(f"WARNING: requested {args.samples} rather than the canonical 100 examples")
    if any(not 0 <= t <= 1 for t in args.levels):
        parser.error("Every t must lie in [0,1]")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = default_checkpoints()
    for task in TASKS:
        for variant in VARIANTS:
            if not checkpoints[task][variant].is_file():
                raise FileNotFoundError(checkpoints[task][variant])
    matched_steps = {
        task: {checkpoint_step(path) for path in checkpoints[task].values()}
        for task in TASKS
    }
    if any(len(steps) != 1 for steps in matched_steps.values()):
        raise RuntimeError(f"Unmatched checkpoint budgets: {matched_steps}")

    device = torch.device(args.device)
    all_rows = []
    task_metadata = {}
    for task in TASKS:
        task_rows, metadata = run_task(
            task, checkpoints[task], args.levels, args.samples, args.seed,
            args.batch_size, device, args.output)
        all_rows.extend(task_rows)
        task_metadata[task] = metadata
        print(f"Completed {task}: {len(task_rows)} summary rows", flush=True)

    fieldnames = list(all_rows[0])
    with (args.output / "layer_cosine_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    manifest = {
        "definition": "cosine between each token hidden state entering and leaving Transformer block l",
        "aggregation": "positions within example, then mean and 95% CI across 100 examples",
        "t_convention": "t=1 fully masked, t=0 fully clean; corruption applies to solution positions only",
        "mask_coupling": "same examples and nested Bernoulli masks for both variants",
        "weights": "EMA",
        "seed": args.seed,
        "levels": args.levels,
        "tasks": task_metadata,
        "checkpoints": {
            task: {variant: {"path": str(path), "step": checkpoint_step(path),
                             "sha256": sha256(path)}
                   for variant, path in variants.items()}
            for task, variants in checkpoints.items()
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    make_plots(all_rows, args.levels, args.output)
    print(f"Wrote results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
