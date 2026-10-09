#!/usr/bin/env python3
"""Collect sweep results from .txt output files into a DataFrame."""

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_result_file(filepath: Path) -> Optional[dict]:
    """Extract metrics from a sweep result file."""
    text = filepath.read_text()
    
    result = {"filename": filepath.name}
    
    # Extract dataset from parent directory
    result["dataset"] = filepath.parent.name
    result["method"] = filepath.stem.split("_")[0]

    ckpt_name = filepath.stem.split("_")[1]
    result["ckpt"] = ckpt_name if "steps" not in ckpt_name else "best"
    
    # Extract steps and latent from filename
    # Pattern: method_stepsX_latentY.txt
    fname_match = re.search(r"steps(\d+)_latent(\d+)", filepath.stem)
    if fname_match:
        result["steps"] = int(fname_match.group(1))
        result["latent_tokens"] = int(fname_match.group(2))
    
    # Extract metrics from file content
    # These patterns cover sudoku-small, sudoku-puzzle, zebra, game-of-24, etc.
    patterns = {
        # Sudoku-small metrics
        "total_samples": r"Total samples:\s+(\d+)",
        "format_ok": r"Format OK:\s+(\d+)",
        "format_ok_pct": r"Format OK:\s+\d+\s+\(([\d.]+)%\)",
        "valid_sudoku": r"Valid sudoku:\s+(\d+)",
        "valid_sudoku_pct": r"Valid sudoku:\s+\d+\s+\(([\d.]+)%\)",
        "avg_violations": r"Avg violations:\s+([\d.]+)",
        "avg_loss": r"Avg loss:\s+([\d.]+)",
        "avg_entropy": r"Avg entropy:\s+([\d.]+)",
        "normalized_entropy_pct": r"Normalized entropy:\s+([\d.]+)%",
        # Sudoku-puzzle / zebra metrics
        "puzzles_evaluated": r"Puzzles evaluated:\s+(\d+)",
        "puzzles_correct": r"Puzzles correct:\s+(\d+)",
        "puzzle_accuracy": r"Puzzle accuracy:\s+([\d.]+)",
        "cell_accuracy": r"Cell accuracy:\s+([\d.]+)",
        "row_accuracy": r"Row accuracy:\s+([\d.]+)",
        "column_accuracy": r"Column accuracy:\s+([\d.]+)",
        "box_accuracy": r"Box accuracy:\s+([\d.]+)",
        "valid_sudoku_rate": r"Valid sudoku rate:\s+([\d.]+)",
        # Game-of-24 metrics
        "total_puzzles": r"Total puzzles:\s+(\d+)",
        "correct_solutions": r"Correct solutions:\s+(\d+)",
        "correct_solutions_pct": r"Correct solutions:\s+\d+\s+\(([\d.]+)%\)",
        # Openwebtext metrics
        "gen_ppl": r"Generative perplexity:\s+([\d.]+)",
        "sample_entropy": r"Sample entropy:\s+([\d.]+)",
        # Sudoku conditional
        # lines look like     Valid solutions: 711 (69.4%)
        # we just match the percent
        "valid_solutions_pct": r"Valid solutions:\s+\d+\s+\(([\d.]+)%\)",
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            val = match.group(1)
            result[key] = float(val) if "." in val else int(val)
    
    # Map game-of-24 metrics to standard names
    if "correct_solutions" in result:
        result["puzzles_correct"] = result["correct_solutions"]
    if "correct_solutions_pct" in result:
        result["puzzle_accuracy"] = result["correct_solutions_pct"] / 100.0
    if "total_puzzles" in result:
        result["puzzles_evaluated"] = result["total_puzzles"]
    
    # Return if we found any meaningful metric
    # has_metrics = any(k in result for k in [
    #     "valid_sudoku", "valid_sudoku_pct", "puzzle_accuracy", "puzzles_correct",
    #     "correct_solutions"
    # ])
    # if not has_metrics:
    #     return None
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Collect sweep results into a DataFrame")
    parser.add_argument("output_dir", type=str, default="sweep_results", nargs="?",
                        help="Directory containing .txt result files (searches recursively)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save results to CSV file")
    parser.add_argument("--sort", type=str, default=None,
                        help="Column to sort by (default: auto-detect best metric)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Directory {output_dir} does not exist")
        return
    
    results = []
    # Search recursively for .txt files
    for txt_file in sorted(output_dir.rglob("*.txt")):
        result = parse_result_file(txt_file)
        if result:
            results.append(result)
    
    if not results:
        print("No valid result files found")
        return
    
    df = pd.DataFrame(results)
    
    # Reorder columns - put key columns first
    priority_cols = [
        "dataset", "steps", "latent_tokens",
        # Puzzle metrics (sudoku-puzzle, zebra)
        "puzzle_accuracy", "puzzles_correct", "cell_accuracy", "row_accuracy",
        "column_accuracy", "box_accuracy", "valid_sudoku_rate",
        # Sudoku-small metrics  
        "valid_sudoku", "valid_sudoku_pct", "avg_violations", "avg_loss",
        "normalized_entropy_pct",
        "filename"
    ]
    cols = [c for c in priority_cols if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    df = df[cols]
    
    # # Auto-detect sort column if not specified
    # sort_col = args.sort
    # if sort_col is None:
    #     for candidate in ["puzzle_accuracy", "valid_sudoku_pct", "puzzles_correct", "valid_sudoku"]:
    #         if candidate in df.columns:
    #             sort_col = candidate
    #             break
    
    # if sort_col and sort_col in df.columns:
    #     df = df.sort_values(sort_col, ascending=False)
    
    print(df.to_string(index=False))
    
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved to {args.csv}")


if __name__ == "__main__":
    main()
