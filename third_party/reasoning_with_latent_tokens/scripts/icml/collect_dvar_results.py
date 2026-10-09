#!/usr/bin/env python3
"""
Collect DVAR evaluation results into a table.

Usage:
    python scripts/icml/collect_dvar_results.py [--datadir PATH]

Looks for results in ${ESOLM_DATADIR}/runs/dvar-*/samples_*.json
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict


def find_latest_samples_file(run_dir):
    """Find the most recent samples_*.json file in a directory."""
    pattern = os.path.join(run_dir, "samples_*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by modification time, get latest
    return max(files, key=os.path.getmtime)


def parse_run_name(run_name):
    """
    Parse run name to extract method, task, and variant.

    Examples:
        dvar-ar-sudoku-eval-last-v1 -> (ar, sudoku, default)
        dvar-mdm-full-3sat7-eval-steps325-last -> (mdm-full, 3sat7, default)
        dvar-mdm-full-cd4-topp-eval-steps64-last -> (mdm-full, cd4, topp)
    """
    # Pattern: dvar-{method}-{task}[-{variant}]-eval-...
    # Method can be: ar, mdm-full, mdm-causal-output, etc.
    # Task can be: sudoku, cd3, cd4, cd5, 3sat5, 3sat7, 3sat9, path

    methods = ['ar', 'mdm-full', 'mdm-causal-output', 'mdm-solo-full', 'ar-mtp-full']
    tasks = ['sudoku', 'cd3', 'cd4', 'cd5', '3sat5', '3sat7', '3sat9', 'path']

    for method in sorted(methods, key=len, reverse=True):  # Match longer methods first
        if f"dvar-{method}-" in run_name:
            rest = run_name[len(f"dvar-{method}-"):]
            for task in tasks:
                if rest.startswith(task):
                    after_task = rest[len(task):]
                    # Check for variant (e.g., -topp)
                    if after_task.startswith("-eval"):
                        return method, task, "default"
                    elif "-eval" in after_task:
                        variant = after_task.split("-eval")[0].lstrip("-")
                        return method, task, variant if variant else "default"

    return None, None, None


def load_results(datadir):
    """Load all results from run directories."""
    runs_dir = os.path.join(datadir, "runs")
    results = []

    # Find all dvar eval runs
    pattern = os.path.join(runs_dir, "dvar-*-eval-*")
    run_dirs = glob.glob(pattern)

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        method, task, variant = parse_run_name(run_name)

        if method is None:
            continue

        samples_file = find_latest_samples_file(run_dir)
        if samples_file is None:
            print(f"Warning: No samples file found in {run_dir}")
            continue

        try:
            with open(samples_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {samples_file}: {e}")
            continue

        eval_metrics = data.get('eval_metrics', {})
        accuracy = eval_metrics.get('puzzle_accuracy', eval_metrics.get('accuracy'))

        if accuracy is None:
            print(f"Warning: No accuracy metric in {samples_file}")
            continue

        results.append({
            'method': method,
            'task': task,
            'variant': variant,
            'accuracy': accuracy,
            'n_correct': eval_metrics.get('n_correct', 'N/A'),
            'n_total': eval_metrics.get('n_total', 'N/A'),
            'time_per_batch': data.get('time_per_batch', 'N/A'),
            'run_name': run_name,
            'samples_file': samples_file,
        })

    return results


def print_table(results):
    """Print results as a formatted table."""
    if not results:
        print("No results found.")
        return

    # Group by method and variant
    tasks_order = ['sudoku', 'cd3', 'cd4', 'cd5', '3sat5', '3sat7', '3sat9', 'path']
    methods_order = ['ar', 'mdm-full']
    variants_order = ['default', 'topp']

    # Build table data
    table = defaultdict(dict)
    sources = defaultdict(dict)  # Track source files
    for r in results:
        key = (r['method'], r['variant'])
        table[key][r['task']] = r['accuracy']
        sources[key][r['task']] = r['samples_file']

    # Print header
    print("\n" + "=" * 80)
    print("DVAR EVALUATION RESULTS")
    print("=" * 80)

    # Filter to tasks that have data
    available_tasks = [t for t in tasks_order if any(t in table[k] for k in table)]

    # Print table
    header = f"{'Method':<25} | " + " | ".join(f"{t:>8}" for t in available_tasks)
    print(header)
    print("-" * len(header))

    for method in methods_order:
        for variant in variants_order:
            key = (method, variant)
            if key not in table:
                continue

            if variant == 'default':
                row_name = method
            else:
                row_name = f"{method} ({variant})"

            row_values = []
            for task in available_tasks:
                acc = table[key].get(task)
                if acc is not None:
                    row_values.append(f"{acc*100:>7.1f}%")
                else:
                    row_values.append(f"{'--':>8}")

            print(f"{row_name:<25} | " + " | ".join(row_values))

    print("=" * 80)

    # Print source files
    print("\nSource files:")
    for method in methods_order:
        for variant in variants_order:
            key = (method, variant)
            if key not in table:
                continue
            if variant == 'default':
                row_name = method
            else:
                row_name = f"{method} ({variant})"
            for task in available_tasks:
                src = sources[key].get(task)
                if src:
                    print(f"  {row_name} / {task}: {src}")

    # Build CSV content
    csv_lines = []
    csv_lines.append("method,variant," + ",".join(available_tasks))
    for method in methods_order:
        for variant in variants_order:
            key = (method, variant)
            if key not in table:
                continue
            values = [str(table[key].get(t, "")) for t in available_tasks]
            csv_lines.append(f"{method},{variant}," + ",".join(values))

    # Print CSV format
    print("\nCSV format:")
    for line in csv_lines:
        print(line)

    return csv_lines


def main():
    parser = argparse.ArgumentParser(description="Collect DVAR evaluation results")
    parser.add_argument("--datadir", type=str, default=None,
                        help="Data directory (default: $ESOLM_DATADIR)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output CSV file path (default: results/dvar_results.csv)")
    args = parser.parse_args()

    datadir = args.datadir or os.environ.get("ESOLM_DATADIR")
    if not datadir:
        print("Error: ESOLM_DATADIR not set and --datadir not provided")
        return 1

    print(f"Looking for results in: {datadir}/runs/")
    results = load_results(datadir)
    print(f"Found {len(results)} result files")

    csv_lines = print_table(results)

    # Write CSV to file
    if csv_lines:
        output_path = args.output or "results/dvar_results.csv"
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, 'w') as f:
            f.write('\n'.join(csv_lines) + '\n')
        print(f"\nCSV written to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
