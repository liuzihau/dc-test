#!/usr/bin/env python3
"""Plot author-native Zebra generation accuracy against training epoch."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY = (
    REPO_ROOT
    / "outputs/reasoning/author-zebra/"
    "diffu-full-mini-zebra-tat-3ep-2x3090/generation/author_native_history.csv"
)


def number(row, key):
  value = row.get(key, "").strip()
  if not value:
    return None
  try:
    return float(value)
  except ValueError:
    return None


def wilson_interval(successes, total):
  """Return a 95% Wilson interval for a binomial proportion."""
  if total <= 0:
    return (math.nan, math.nan)
  z = 1.959963984540054
  p = successes / total
  denominator = 1.0 + z * z / total
  center = (p + z * z / (2.0 * total)) / denominator
  radius = (
      z
      * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
      / denominator
  )
  return (max(0.0, center - radius), min(1.0, center + radius))


def load_history(path):
  with path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
  rows = [row for row in rows if number(row, "epoch") is not None]
  rows.sort(key=lambda row: float(row["epoch"]))
  if not rows:
    raise ValueError(f"No accuracy rows found in {path}")
  return rows


def series(rows, key):
  values = []
  for row in rows:
    value = number(row, key)
    values.append(math.nan if value is None else 100.0 * value)
  return values


def plot_history(history, output):
  rows = load_history(history)
  epochs = [float(row["epoch"]) for row in rows]
  exact = series(rows, "puzzle_accuracy")
  row_accuracy = series(rows, "mean_row_accuracy")
  cell_accuracy = series(rows, "mean_cell_accuracy")
  chance_exact = series(rows, "chance_puzzle_accuracy")
  chance_row = series(rows, "chance_row_accuracy")
  chance_cell = series(rows, "chance_cell_accuracy")

  exact_lower, exact_upper = [], []
  for row in rows:
    successes = number(row, "n_correct_puzzles")
    total = number(row, "n_total_puzzles")
    if successes is None or total is None:
      exact_lower.append(math.nan)
      exact_upper.append(math.nan)
    else:
      lower, upper = wilson_interval(successes, total)
      exact_lower.append(100.0 * lower)
      exact_upper.append(100.0 * upper)

  plt.style.use("seaborn-v0_8-whitegrid")
  figure, (exact_axis, partial_axis) = plt.subplots(
      2, 1, figsize=(10.5, 8.0), sharex=True,
      gridspec_kw={"height_ratios": [1.1, 1.0]})
  figure.suptitle(
      "Author-native Zebra generation accuracy vs. training epoch",
      fontsize=15, fontweight="bold")

  exact_axis.plot(
      epochs, exact, color="#1665D8", marker="o", linewidth=2.4,
      markersize=5.5, label="Exact puzzle accuracy")
  exact_axis.fill_between(
      epochs, exact_lower, exact_upper, color="#1665D8", alpha=0.16,
      linewidth=0, label="95% Wilson interval")
  if not all(math.isnan(value) for value in chance_exact):
    exact_axis.plot(
        epochs, chance_exact, color="#777777", linestyle=":",
        linewidth=1.5, label="Chance baseline")
  for epoch, value in zip(epochs, exact):
    exact_axis.annotate(
        f"{value:.1f}", (epoch, value), xytext=(0, 7),
        textcoords="offset points", ha="center", fontsize=8,
        color="#104E9D")
  exact_axis.set_ylabel("Exact accuracy (%)")
  exact_axis.set_ylim(0.0, min(100.0, max(5.0, max(exact_upper) * 1.15)))
  exact_axis.legend(loc="upper left", frameon=True)

  partial_axis.plot(
      epochs, row_accuracy, color="#E4572E", marker="s", linewidth=2.0,
      markersize=5, label="Mean row accuracy")
  partial_axis.plot(
      epochs, cell_accuracy, color="#2E8B57", marker="^", linewidth=2.0,
      markersize=5, label="Mean cell accuracy")
  if not all(math.isnan(value) for value in chance_row):
    partial_axis.plot(
        epochs, chance_row, color="#E4572E", linestyle=":",
        linewidth=1.3, alpha=0.8, label="Row chance")
  if not all(math.isnan(value) for value in chance_cell):
    partial_axis.plot(
        epochs, chance_cell, color="#2E8B57", linestyle=":",
        linewidth=1.3, alpha=0.8, label="Cell chance")
  partial_axis.set_xlabel("Training epoch")
  partial_axis.set_ylabel("Partial accuracy (%)")
  partial_axis.set_ylim(0.0, 100.0)
  partial_axis.set_xticks(epochs)
  partial_axis.legend(loc="lower right", ncol=2, frameon=True)

  sample_counts = sorted({row.get("n_total_puzzles", "") for row in rows})
  sample_note = (
      f"Evaluation size: {sample_counts[0]} held-out puzzles/checkpoint"
      if len(sample_counts) == 1 and sample_counts[0]
      else "Held-out evaluation size may vary by checkpoint")
  figure.text(
      0.5, 0.012,
      f"{sample_note}. Points are measured evaluations; no smoothing.",
      ha="center", fontsize=9, color="#555555")
  figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.96))
  output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output, dpi=200, bbox_inches="tight")
  plt.close(figure)
  print(f"Wrote accuracy plot to {output}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("history", nargs="?", type=Path, default=DEFAULT_HISTORY)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  output = args.output or args.history.with_name(
      "author_native_accuracy_vs_epoch.png")
  plot_history(args.history, output)


if __name__ == "__main__":
  main()
