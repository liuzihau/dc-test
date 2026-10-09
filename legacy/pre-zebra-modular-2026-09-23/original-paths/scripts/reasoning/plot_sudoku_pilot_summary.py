#!/usr/bin/env python3
"""Plot the completed five-mechanism Sudoku *pilot* without relabelling it official."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CURVES = ROOT / "imports/reasoning-cloud-20260916/final/figures"
DEFAULT_ACCURACY = (ROOT / "imports/reasoning-cloud-20260916/final/overnight-2026-09-16"
                    / "sudoku_accuracy.csv")
RUNS = {
    "vanilla": ("MDM", "sudoku-vanilla-h100-pilot-v1-n20000-v1000-t1000-mb128-gb128-seed1.csv"),
    "mdm": ("TT", "sudoku-mdm-h100-pilot-v1-n20000-v1000-t1000-mb128-gb128-seed1.csv"),
    "mdm_aux": ("TT + NP", "sudoku-mdm_aux-h100-pilot-v1-n20000-v1000-t1000-mb128-gb128-seed1.csv"),
    "both": ("TT + RM", "sudoku-both-h100-current-preserving-pilot-v1-n20000-v1000-t1000-mb128-gb128-seed1.csv"),
    "both_aux": ("TT + RM + NP", "sudoku-both_aux-h100-current-preserving-pilot-v1-n20000-v1000-t1000-mb128-gb128-seed1.csv"),
}
COLORS = {
    "vanilla": "#4C78A8", "mdm": "#F58518", "mdm_aux": "#E45756",
    "both": "#72B7B2", "both_aux": "#B279A2",
}


def load_curves(directory):
    curves = {}
    reference_steps = None
    for variant, (label, filename) in RUNS.items():
        path = directory / filename
        frame = pd.read_csv(path)
        required = {"step", "val/conditional_nll", "val/num_examples", "val/seed"}
        if not required.issubset(frame):
            raise ValueError(f"Missing validation columns in {path}")
        frame = frame.loc[frame["val/conditional_nll"].notna(),
                          ["step", "val/conditional_nll", "val/num_examples", "val/seed"]]
        frame = frame.sort_values("step").drop_duplicates("step", keep="last")
        steps = frame["step"].astype(int).tolist()
        if steps != list(range(500, 5001, 500)):
            raise ValueError(f"Incomplete validation checkpoints in {path}: {steps}")
        if reference_steps is not None and steps != reference_steps:
            raise ValueError("Validation steps differ across mechanisms")
        reference_steps = steps
        if frame["val/num_examples"].nunique() != 1 or frame["val/seed"].nunique() != 1:
            raise ValueError(f"Validation protocol changes within {path}")
        curves[variant] = (label, frame)
    return curves


def load_accuracy(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if row["task"] == "sudoku"]
    by_variant = {row["variant"]: row for row in rows}
    if set(by_variant) != set(RUNS):
        raise ValueError("Accuracy table does not contain exactly the five mechanisms")
    signatures = {(row["step"], row["examples"], row["dataset_sha256"], row["train_seed"])
                  for row in rows}
    if len(signatures) != 1:
        raise ValueError("Accuracy bars are not a matched dataset/step/test-count comparison")
    return by_variant


def plot(curves, accuracy, output):
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), sharex=True)
    for variant, (label, frame) in curves.items():
        for ax in axes:
            ax.plot(frame["step"], frame["val/conditional_nll"], marker="o",
                    markersize=4, linewidth=2, color=COLORS[variant], label=label)
    axes[0].set_title("Full validation-NLL range")
    axes[1].set_title("Same observations on log scale")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Conditional validation NLL")
        ax.grid(alpha=.25)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Sudoku pilot: cold validation NLL (not official paper data)", fontsize=13)
    fig.tight_layout()
    nll_path = output / "sudoku_pilot_validation_nll_full.png"
    fig.savefig(nll_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    variants = list(RUNS)
    labels = [RUNS[v][0] for v in variants]
    values = [float(accuracy[v]["accuracy_pct"]) for v in variants]
    lower = [v - float(accuracy[k]["ci95_low_pct"]) for k, v in zip(variants, values)]
    upper = [float(accuracy[k]["ci95_high_pct"]) - v for k, v in zip(variants, values)]
    fig, ax = plt.subplots(figsize=(9.3, 5.2))
    bars = ax.bar(labels, values, color=[COLORS[v] for v in variants],
                  yerr=[lower, upper], capsize=5)
    ax.set_ylim(80, 100)
    ax.set_ylabel("Whole-puzzle accuracy (%)")
    ax.set_title("Sudoku pilot: step-5000 closed-loop accuracy\n1,000 shared test puzzles; 95% Wilson intervals")
    ax.grid(axis="y", alpha=.25)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=5, fontsize=10)
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    accuracy_path = output / "sudoku_pilot_accuracy_step5000.png"
    fig.savefig(accuracy_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary_path = output / "sudoku_pilot_summary.csv"
    with summary_path.open("w", newline="") as stream:
        fields = ["mechanism", "legacy_variant", "step", "test_examples", "accuracy_pct",
                  "ci95_low_pct", "ci95_high_pct", "final_val_nll", "best_val_nll", "best_val_step"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for variant in variants:
            label, frame = curves[variant]
            best = frame.loc[frame["val/conditional_nll"].idxmin()]
            row = accuracy[variant]
            writer.writerow(dict(mechanism=label, legacy_variant=variant, step=row["step"],
                                 test_examples=row["examples"], accuracy_pct=row["accuracy_pct"],
                                 ci95_low_pct=row["ci95_low_pct"], ci95_high_pct=row["ci95_high_pct"],
                                 final_val_nll=frame.iloc[-1]["val/conditional_nll"],
                                 best_val_nll=best["val/conditional_nll"],
                                 best_val_step=int(best["step"])))
    return nll_path, accuracy_path, summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    parser.add_argument("--accuracy", type=Path, default=DEFAULT_ACCURACY)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/generated/figures/reasoning/sudoku-pilot-five-mechanisms")
    args = parser.parse_args()
    paths = plot(load_curves(args.curves), load_accuracy(args.accuracy), args.output)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
