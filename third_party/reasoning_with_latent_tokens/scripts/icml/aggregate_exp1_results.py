#!/usr/bin/env python3
"""
Aggregate exp1 experiment results into a 4x4 accuracy table.

Usage:
    python scripts/icml/aggregate_exp1_results.py

Scans $ESOLM_DATADIR/runs/ for generation outputs matching exp1 experiments
and produces a formatted table of accuracies.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Configuration
METHODS = [
    "ar",
    "ar-mtp-full",
    "diffu-full",
    "diffu-solo-full",
    "diffu-full-topp",
    "diffu-solo-full-topp",
]
DATASETS = ["sudoku-puzzle", "sudoku-small-solver", "zebra", "game-of-24"]
SIZE = "sminy"
RUN_SUFFIX = "-exp1"

# Dataset-specific settings
DATASET_CONFIG = {
    "sudoku-puzzle": {
        "model_length": 192,
        "metric": "puzzle_accuracy",
        "gen_mode": "completions",
    },
    "sudoku-small-solver": {
        "model_length": 128,
        "metric": "sudoku_valid_rate",
        "gen_mode": "sample_eval",
    },
    "zebra": {
        "model_length": 384,
        "metric": "puzzle_accuracy",
        "gen_mode": "completions",
    },
    "game-of-24": {
        "model_length": 64,
        "metric": "puzzle_accuracy",
        "gen_mode": "completions",
    },
}


@dataclass
class Result:
    """Container for evaluation result with metadata."""
    accuracy: Optional[float]
    checkpoint_path: Optional[str]
    samples_path: Optional[str]
    error: Optional[str] = None


def _get_base_method(method: str) -> str:
    """Get the base method name (strips -topp suffix if present)."""
    if method.endswith("-topp"):
        return method[:-5]  # Remove "-topp"
    return method


def _is_topp_method(method: str) -> bool:
    """Check if this is a topp sampling method."""
    return method.endswith("-topp")


def get_checkpoint_dir_name(method: str, dataset: str) -> str:
    """Build the checkpoint directory name."""
    # Format: {method}-{size}-{data}{suffix}
    # Note: topp methods use the same checkpoint as their base method
    base_method = _get_base_method(method)
    return f"{base_method}-{SIZE}-{dataset}{RUN_SUFFIX}"


def get_run_dir_pattern(method: str, dataset: str) -> str:
    """Build the expected run directory name pattern."""
    # Format: {method}-{size}-{data}{suffix}-gen-[steps-{N}-]best[-topp]
    # AR methods don't include steps in the name
    model_length = DATASET_CONFIG[dataset]["model_length"]
    base_method = _get_base_method(method)
    gen_suffix = "-topp" if _is_topp_method(method) else ""

    if base_method == "ar":
        # AR: ar-sminy-{data}{suffix}-gen-best
        return f"ar-{SIZE}-{dataset}{RUN_SUFFIX}-gen-best{gen_suffix}"
    else:
        # Diffusion: {method}-{size}-{data}{suffix}-gen-steps-{N}-best[-topp]
        return f"{base_method}-{SIZE}-{dataset}{RUN_SUFFIX}-gen-steps-{model_length}-best{gen_suffix}"


def find_checkpoint(checkpoints_dir: Path, method: str, dataset: str) -> Optional[Path]:
    """Find the checkpoint file for a given method/dataset combination."""
    ckpt_dir_name = get_checkpoint_dir_name(method, dataset)
    ckpt_dir = checkpoints_dir / ckpt_dir_name / "checkpoints"

    if not ckpt_dir.exists():
        return None

    # Find best*.ckpt (handles best.ckpt, best-v1.ckpt, etc.)
    candidates = sorted(ckpt_dir.glob("best*.ckpt"))
    if candidates:
        return candidates[-1]  # Return the latest version

    return None


def find_samples_json(runs_dir: Path, method: str, dataset: str) -> Optional[Path]:
    """Find the samples.json file for a given method/dataset combination."""
    pattern = get_run_dir_pattern(method, dataset)
    run_dir = runs_dir / pattern

    # Try samples.json first, then samples_*.json (timestamped)
    samples_path = run_dir / "samples.json"
    if samples_path.exists():
        return samples_path

    # Look for timestamped samples files (e.g., samples_20260118_163912.json)
    samples_candidates = sorted(run_dir.glob("samples_*.json"))
    if samples_candidates:
        return samples_candidates[-1]  # Return the latest

    # Also try without explicit "best" in case checkpoint name differs
    # (e.g., best-v1.ckpt produces "best-v1" instead of "best")
    for candidate in runs_dir.glob(f"{pattern.replace('-best', '-best*')}"):
        candidate_samples = candidate / "samples.json"
        if candidate_samples.exists():
            return candidate_samples
        # Try timestamped samples files
        timestamped = sorted(candidate.glob("samples_*.json"))
        if timestamped:
            return timestamped[-1]

    return None


def extract_metric(samples_path: Path, dataset: str) -> Optional[float]:
    """Extract the appropriate metric from samples.json."""
    metric_key = DATASET_CONFIG[dataset]["metric"]

    try:
        with open(samples_path, "r") as f:
            data = json.load(f)

        eval_metrics = data.get("eval_metrics", {})
        value = eval_metrics.get(metric_key)

        if value is not None:
            return float(value)

        return None
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Error reading {samples_path}: {e}")
        return None


def main():
    # Get directories from environment
    datadir = os.environ.get("ESOLM_DATADIR")
    if not datadir:
        print("Error: ESOLM_DATADIR environment variable not set")
        return 1

    runs_dir = Path(datadir) / "runs"
    checkpoints_dir = Path(datadir) / "checkpoints"

    if not runs_dir.exists():
        print(f"Error: Runs directory not found: {runs_dir}")
        return 1

    if not checkpoints_dir.exists():
        print(f"Error: Checkpoints directory not found: {checkpoints_dir}")
        return 1

    # Collect results
    results: dict[str, dict[str, Result]] = {}
    missing = []

    for method in METHODS:
        results[method] = {}
        for dataset in DATASETS:
            # Find checkpoint
            checkpoint_path = find_checkpoint(checkpoints_dir, method, dataset)

            # Find samples.json
            samples_path = find_samples_json(runs_dir, method, dataset)

            if samples_path:
                value = extract_metric(samples_path, dataset)
                if value is not None:
                    results[method][dataset] = Result(
                        accuracy=value,
                        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                        samples_path=str(samples_path),
                    )
                else:
                    metric_key = DATASET_CONFIG[dataset]["metric"]
                    error = f"metric '{metric_key}' not found"
                    missing.append(f"{method}/{dataset}: {error} in {samples_path}")
                    results[method][dataset] = Result(
                        accuracy=None,
                        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                        samples_path=str(samples_path),
                        error=error,
                    )
            else:
                expected_pattern = get_run_dir_pattern(method, dataset)
                error = f"run dir not found (expected: {expected_pattern})"
                missing.append(f"{method}/{dataset}: {error}")
                results[method][dataset] = Result(
                    accuracy=None,
                    checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                    samples_path=None,
                    error=error,
                )

    # Print summary table
    print("=" * 100)
    print("Experiment 1 Results: 4x4 Accuracy Table")
    print("=" * 100)

    # Header
    header = f"{'Method':<20}"
    for dataset in DATASETS:
        header += f"{dataset:>20}"
    print(header)
    print("-" * 100)

    # Rows
    for method in METHODS:
        row = f"{method:<20}"
        for dataset in DATASETS:
            result = results[method][dataset]
            if result.accuracy is not None:
                row += f"{result.accuracy*100:>19.2f}%"
            else:
                row += f"{'N/A':>20}"
        print(row)

    print("=" * 100)

    # Print detailed results with checkpoint paths
    print("\n" + "=" * 100)
    print("Detailed Results (with checkpoint paths)")
    print("=" * 100)

    for method in METHODS:
        for dataset in DATASETS:
            result = results[method][dataset]
            print(f"\n{method} / {dataset}:")

            if result.accuracy is not None:
                print(f"  Accuracy: {result.accuracy*100:.2f}%")
            else:
                print(f"  Accuracy: N/A ({result.error})")

            if result.checkpoint_path:
                print(f"  Checkpoint: {result.checkpoint_path}")
            else:
                expected_ckpt_dir = get_checkpoint_dir_name(method, dataset)
                print(f"  Checkpoint: NOT FOUND (expected in {expected_ckpt_dir}/)")

            if result.samples_path:
                print(f"  Samples: {result.samples_path}")

    print("\n" + "=" * 100)

    # Print any missing results
    if missing:
        print("\nMissing results:")
        for msg in missing:
            print(f"  - {msg}")

    # Summary statistics
    complete_count = sum(
        1 for method in METHODS for dataset in DATASETS
        if results[method][dataset].accuracy is not None
    )
    total_count = len(METHODS) * len(DATASETS)
    print(f"\nResults found: {complete_count}/{total_count}")

    return 0 if complete_count == total_count else 1


if __name__ == "__main__":
    exit(main())
