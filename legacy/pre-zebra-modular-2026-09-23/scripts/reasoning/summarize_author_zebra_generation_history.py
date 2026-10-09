#!/usr/bin/env python3
"""Collect author-native Zebra generation checkpoints into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from plot_author_zebra_generation_history import plot_history


STEP_PATTERN = re.compile(r"step-(\d+)-author-native")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("generation_root", type=Path)
  parser.add_argument("--steps-per-epoch", type=int, default=2930)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()

  rows = []
  for directory in args.generation_root.glob("step-*-author-native"):
    match = STEP_PATTERN.fullmatch(directory.name)
    if not match:
      continue
    files = sorted(directory.glob("samples_*.json"), key=lambda path: path.stat().st_mtime)
    if not files:
      continue
    path = files[-1]
    with path.open(encoding="utf-8") as stream:
      payload = json.load(stream)
    metrics = payload.get("eval_metrics", {})
    step = int(match.group(1))
    rows.append({
        "epoch": step / args.steps_per_epoch,
        "optimizer_step": step,
        "n_total_puzzles": metrics.get("n_total_puzzles"),
        "n_correct_puzzles": metrics.get("n_correct_puzzles"),
        "puzzle_accuracy": metrics.get("puzzle_accuracy"),
        "mean_row_accuracy": metrics.get("mean_row_accuracy"),
        "mean_cell_accuracy": metrics.get("mean_cell_accuracy"),
        "chance_puzzle_accuracy": metrics.get("chance_puzzle_accuracy"),
        "chance_row_accuracy": metrics.get("chance_row_accuracy"),
        "chance_cell_accuracy": metrics.get("chance_cell_accuracy"),
        "malformed_rate": metrics.get("malformed_rate"),
        "time_per_batch_seconds": payload.get("time_per_batch"),
        "samples_path": str(path.resolve()),
    })

  rows.sort(key=lambda row: row["optimizer_step"])
  output = args.output or args.generation_root / "author_native_history.csv"
  output.parent.mkdir(parents=True, exist_ok=True)
  fields = [
      "epoch", "optimizer_step", "n_total_puzzles", "n_correct_puzzles",
      "puzzle_accuracy", "mean_row_accuracy", "mean_cell_accuracy",
      "chance_puzzle_accuracy", "chance_row_accuracy", "chance_cell_accuracy",
      "malformed_rate", "time_per_batch_seconds", "samples_path",
  ]
  with output.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
  print(f"Wrote {len(rows)} rows to {output}")
  if rows:
    plot_history(output, output.with_name("author_native_accuracy_vs_epoch.png"))


if __name__ == "__main__":
  main()
