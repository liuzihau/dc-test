#!/usr/bin/env python3
"""Plot locally recorded author Zebra train and validation losses."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path):
  if not path.exists():
    return []
  with path.open(newline="", encoding="utf-8") as stream:
    return list(csv.DictReader(stream))


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
  if window <= 1 or len(values) < 2:
    return values
  window = min(window, len(values))
  kernel = np.ones(window, dtype=np.float64) / window
  result = np.convolve(values, kernel, mode="valid")
  return np.concatenate((np.full(window - 1, np.nan), result))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--run-dir",
      type=Path,
      default=Path("outputs/reasoning/author-zebra/"
                   "diffu-full-mini-zebra-tat-3ep-2x3090"),
  )
  parser.add_argument("--smooth", type=int, default=128)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()

  metrics_dir = args.run_dir / "local_metrics"
  train = read_csv(metrics_dir / "train.csv")
  valid = read_csv(metrics_dir / "validation.csv")
  if not train and not valid:
    raise SystemExit(f"No local metrics found in {metrics_dir}")

  fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
  if train:
    steps = np.asarray([int(float(row["optimizer_step"])) for row in train])
    losses = np.asarray([float(row["train_loss"]) for row in train])
    axes[0].plot(steps, losses, alpha=0.16, linewidth=0.7, color="#2878B5",
                 label="per update")
    axes[0].plot(steps, rolling_mean(losses, args.smooth), linewidth=2,
                 color="#2878B5", label=f"{args.smooth}-update mean")
    axes[0].legend(frameon=False)
  axes[0].set(title="Training loss", xlabel="Optimizer step", ylabel="ELBO loss")
  axes[0].grid(alpha=0.2)

  if valid:
    steps = np.asarray([int(float(row["optimizer_step"])) for row in valid])
    nll = np.asarray([float(row["val_nll"]) for row in valid])
    axes[1].plot(steps, nll, marker="o", linewidth=2, color="#D95319")
    for x, y in zip(steps, nll):
      axes[1].annotate(f"{y:.4f}", (x, y), xytext=(0, 7),
                       textcoords="offset points", ha="center", fontsize=8)
  axes[1].set(title="Validation NLL", xlabel="Optimizer step", ylabel="NLL")
  axes[1].grid(alpha=0.2)

  output = args.output or metrics_dir / "loss_curves.png"
  output.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output, dpi=180)
  print(output.resolve())


if __name__ == "__main__":
  main()
