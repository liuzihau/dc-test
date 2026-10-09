"""
Sudoku verification for generated samples.

Supports arbitrary base-n Sudoku (base=3 for 9x9, base=6 for 36x36).
"""

import argparse
import json
import math
import re
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd

# ---------------- Constants ----------------

BOS_STR = "[BOS]"
EOS_STR = "[EOS]"
DEFAULT_MASK_INDEX = 11


# ---------------- Parsing ----------------

def parse_sample(sample: str) -> List[str]:
    """
    Tokenize a sequence string into [BOS]/[EOS]/integers/other tokens.
    Commas are ignored as separators.
    """
    return re.findall(r"\[BOS\]|\[EOS\]|-?\d+|[^\s,]+", sample)


# ---------------- Helpers for base-n Sudoku ----------------

def _side(base: int) -> int:
    """Side length n = base*base."""
    if base <= 0:
        raise ValueError(f"`base` must be positive, got {base}.")
    return base * base


def _num_cells(base: int) -> int:
    """Number of cells = n*n = (base*base)^2."""
    n = _side(base)
    return n * n


def _valid_set(base: int) -> set:
    """Valid digits: 0..(n-1) where n=base*base."""
    n = _side(base)
    return set(range(n))


def _rows(board: List[int], *, base: int) -> List[List[int]]:
    n = _side(base)
    return [board[r*n:(r+1)*n] for r in range(n)]


def _cols(board: List[int], *, base: int) -> List[List[int]]:
    n = _side(base)
    return [[board[r*n + c] for r in range(n)] for c in range(n)]


def _blocks(board: List[int], *, base: int) -> List[List[int]]:
    """
    Return all base×base blocks over an n×n board (n=base*base),
    with blocks ordered by block-row then block-col.
    """
    n = _side(base)
    blocks = []
    for br in range(base):
        for bc in range(base):
            r0, c0 = br*base, bc*base
            blk = []
            for dr in range(base):
                for dc in range(base):
                    blk.append(board[(r0+dr)*n + (c0+dc)])
            blocks.append(blk)
    return blocks


# ---------------- Validation ----------------

def check_bos_eos(tokens: List[str], *, base: int) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate [BOS], the exact count of content digits (n^2 cells, where n=base*base),
    and a single [EOS] immediately after the digits.
    """
    info: Dict[str, Any] = {"errors": []}

    if not tokens:
        info["errors"].append("Empty token list.")
        return False, info

    if tokens[0] != BOS_STR:
        info["errors"].append("Missing or misplaced [BOS] at index 0.")

    n = _side(base)
    cells = _num_cells(base)
    middle = tokens[1:1+cells]
    if len(middle) != cells:
        info["errors"].append(f"Expected {cells} digits, got {len(middle)}.")

    valid_max = n - 1
    digits: List[int] = []
    for t in middle:
        if re.fullmatch(r"-?\d+", t) is None:
            info["errors"].append(f"Non-numeric token: {t!r}")
            continue
        v = int(t)
        if not (0 <= v <= valid_max):
            info["errors"].append(f"Digit out of range (0..{valid_max}): {v}")
        digits.append(v)

    eos_slice = tokens[1+cells:1+cells+1]
    if not eos_slice:
        info["errors"].append("Missing [EOS] token after digits.")
    elif eos_slice[0] != EOS_STR:
        info["errors"].append(f"Expected [EOS] after digits, got {eos_slice[0]!r}.")

    ok = (not info["errors"])
    # Always include parsed digits (caller decides whether to use them when ok=False)
    info["content_digits"] = digits
    return ok, info


def check_sudoku(
    board_flat: List[int],
    *,
    base: int,
    puzzle_flat: Optional[List[int]] = None,
    mask_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Validate a base-n Sudoku solution and compute a constraint loss.

    Loss:
      For each unit (row, column, block), let u = number of unique VALID integers in that unit.
      Unit loss = 1 - u / n where n = base^2. Total loss is the sum over all units.
      
      Invalid tokens (outside 0..n-1) don't contribute to reducing loss - they're treated
      at least as bad as duplicates since they take a spot that could have been a valid value.
    """
    n = _side(base)
    if len(board_flat) != n*n:
        raise ValueError(f"Expected flattened board of length {n*n}, got {len(board_flat)}.")

    report = {
        "is_valid": True,
        "violations": 0,
        "bad_rows": [],
        "bad_cols": [],
        "bad_blocks": [],
        "matches_original": True,
        "loss": 0.0,
        "loss_breakdown": {"rows": [], "cols": [], "blocks": []},
    }
    vset = _valid_set(base)
    denom = float(n)

    # Rows
    for i, row in enumerate(_rows(board_flat, base=base)):
        row_set = set(row)
        # Only count unique values that are valid (in 0..n-1)
        u = len(row_set & vset)
        if row_set != vset:
            report["bad_rows"].append(i)
        report["loss_breakdown"]["rows"].append(1.0 - (u / denom))

    # Cols
    for j, col in enumerate(_cols(board_flat, base=base)):
        col_set = set(col)
        u = len(col_set & vset)
        if col_set != vset:
            report["bad_cols"].append(j)
        report["loss_breakdown"]["cols"].append(1.0 - (u / denom))

    # Blocks
    for k, blk in enumerate(_blocks(board_flat, base=base)):
        blk_set = set(blk)
        u = len(blk_set & vset)
        if blk_set != vset:
            report["bad_blocks"].append(k)
        report["loss_breakdown"]["blocks"].append(1.0 - (u / denom))

    # Aggregate
    report["violations"] = len(report["bad_rows"]) + len(report["bad_cols"]) + len(report["bad_blocks"])
    report["loss"] = (
        sum(report["loss_breakdown"]["rows"]) +
        sum(report["loss_breakdown"]["cols"]) +
        sum(report["loss_breakdown"]["blocks"])
    )

    # Check consistency with provided puzzle (if any)
    if puzzle_flat is not None:
        if len(puzzle_flat) != n*n:
            raise ValueError(f"Expected puzzle length {n*n}, got {len(puzzle_flat)}.")
        mi = DEFAULT_MASK_INDEX if mask_index is None else mask_index
        for soln, orig in zip(board_flat, puzzle_flat):
            if orig != mi and soln != orig:
                report["matches_original"] = False
                break

    report["is_valid"] = (report["violations"] == 0) and report["matches_original"]
    return report


def analyze_sample(
    sample: str,
    *,
    base: int = 3,
    puzzle: Optional[List[int]] = None,
    mask_index: Optional[int] = None,
    include_malformed: bool = False,
) -> Dict[str, Any]:
    """
    Entry point for analyzing a single sample.
    Returns validity signals plus constraint loss.
    
    Args:
        include_malformed: If True, compute metrics even when format check fails,
                          by padding/truncating the parsed digits to expected board size.
    """
    tokens = parse_sample(sample)
    bos_ok, bos_info = check_bos_eos(tokens, base=base)
    res = {
        "bos_eos_ok": bos_ok,
        "violations": None,
        "sudoku_valid": None,
        "matches_original": None,
        "loss": None,
    }
    
    should_compute = bos_ok or include_malformed
    board = bos_info.get("content_digits", [])
    
    if should_compute and board:
        # Pad or truncate board to expected size
        n = _side(base)
        num_cells = _num_cells(base)
        if len(board) < num_cells:
            # Pad with -1 (invalid digit) to mark missing cells
            board = board + [-1] * (num_cells - len(board))
        elif len(board) > num_cells:
            board = board[:num_cells]
        
        rep = check_sudoku(board, base=base, puzzle_flat=puzzle, mask_index=mask_index)
        res["sudoku_valid"] = rep["is_valid"]
        res["violations"] = rep["violations"]
        res["matches_original"] = rep["matches_original"]
        res["loss"] = rep["loss"]
    return res


# ---------------- Diversity Metrics ----------------

def compute_diversity_metrics(
    samples: List[str],
    base: int,
    include_malformed: bool = False,
) -> Dict[str, Any]:
    """
    Compute diversity metrics for generated sudoku samples.
    
    For each position in the board, compute the entropy of the digit distribution
    across all samples, then average across positions.
    
    Args:
        samples: List of sample strings
        base: Sudoku base (3 for 9x9, 6 for 36x36)
        include_malformed: If True, include samples that fail format check in diversity
                          computation (using whatever digits were parsed). Default False.
    
    Returns:
        Dictionary with diversity metrics including:
        - avg_entropy: Average entropy across all positions
        - max_entropy: Theoretical maximum (log2(n) for uniform distribution)
        - normalized_entropy: avg_entropy / max_entropy (0 to 1 scale)
        - samples_with_digits: Number of samples used for diversity computation
    """
    n = _side(base)
    num_cells = _num_cells(base)
    
    # Initialize frequency counts: position -> digit -> count
    position_counts: List[Dict[int, int]] = [
        {d: 0 for d in range(n)} for _ in range(num_cells)
    ]
    
    samples_with_digits = 0
    
    for sample in samples:
        tokens = parse_sample(sample)
        bos_ok, bos_info = check_bos_eos(tokens, base=base)
        
        # Skip malformed samples unless include_malformed is True
        if not bos_ok and not include_malformed:
            continue
        
        board = bos_info.get("content_digits", [])
        if not board:
            continue
            
        samples_with_digits += 1
        
        for pos, digit in enumerate(board):
            # Only count if position is within expected board size and digit is valid
            if pos < num_cells and 0 <= digit < n:
                position_counts[pos][digit] += 1
    
    if samples_with_digits == 0:
        return {
            "avg_entropy": float("nan"),
            "max_entropy": math.log2(n) if n > 1 else 0.0,
            "normalized_entropy": float("nan"),
            "samples_with_digits": 0,
        }
    
    # Compute entropy for each position
    entropies = []
    for pos in range(num_cells):
        counts = position_counts[pos]
        total = sum(counts.values())
        if total == 0:
            entropies.append(0.0)
            continue
        
        entropy = 0.0
        for digit, count in counts.items():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        entropies.append(entropy)
    
    avg_entropy = sum(entropies) / len(entropies)
    max_entropy = math.log2(n) if n > 1 else 0.0
    normalized_entropy = avg_entropy / max_entropy if max_entropy > 0 else float("nan")
    
    return {
        "avg_entropy": avg_entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": normalized_entropy,
        "samples_with_digits": samples_with_digits,
    }


# ---------------- Batch Evaluation ----------------

def evaluate_samples_file(
    samples_path: str,
    base: int,
    verbose: bool = False,
    include_malformed: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate all samples in a JSON file.
    
    Args:
        samples_path: Path to samples.json file
        base: Sudoku base (3 for 9x9, 6 for 36x36)
        verbose: If True, print per-sample details
        include_malformed: If True, include malformed samples in diversity computation
    
    Returns:
        Dictionary with evaluation metrics
        
    Note:
        Samples that fail format check (bos_eos_ok=False) are assigned worst-case
        values: max violations, max loss, sudoku_valid=False, matches_original=False.
    """
    with open(samples_path, "r") as f:
        samples_dict = json.load(f)

    samples = samples_dict["generated_seqs"]
    time_per_batch = samples_dict.get("time_per_batch", None)
    
    analyzed = [
        analyze_sample(sample, base=base, include_malformed=include_malformed)
        for sample in samples
    ]
    
    # Compute worst-case values for malformed samples (used when include_malformed=False)
    side = base * base
    max_violations = 3 * side
    max_loss = 3.0 * (side - 1)

    n_samples = len(analyzed)
    
    # For samples with bos_eos_ok=False:
    # - If include_malformed=True: use computed values (if available)
    # - If include_malformed=False: use worst-case values
    sudoku_valids = []
    violations = []
    matches_ori = []
    losses = []
    format_ok_count = 0
    
    for a in analyzed:
        if a["bos_eos_ok"]:
            format_ok_count += 1
        
        # Use computed values if available, otherwise use worst-case
        if a["violations"] is not None:
            sudoku_valids.append(a["sudoku_valid"])
            violations.append(a["violations"])
            matches_ori.append(a["matches_original"])
            losses.append(a["loss"])
        else:
            # Worst case for samples without computed metrics
            sudoku_valids.append(False)
            violations.append(max_violations)
            matches_ori.append(False)
            losses.append(max_loss)

    format_ok_rate = format_ok_count / n_samples if n_samples > 0 else float("nan")
    sudoku_valid_rate = sum(sudoku_valids) / n_samples if n_samples > 0 else float("nan")
    avg_violations = sum(violations) / n_samples if n_samples > 0 else float("nan")
    matches_original_rate = sum(matches_ori) / n_samples if n_samples > 0 else float("nan")
    avg_loss = sum(losses) / n_samples if n_samples > 0 else float("nan")
    
    # Compute diversity metrics
    diversity = compute_diversity_metrics(samples, base=base, include_malformed=include_malformed)
    
    if verbose:
        for i, a in enumerate(analyzed):
            if a["bos_eos_ok"]:
                status = "✓" if a["sudoku_valid"] else "✗"
                print(f"  Sample {i}: {status} (violations={a['violations']}, loss={a['loss']:.4f})")
            else:
                print(f"  Sample {i}: ✗ (malformed - failed format check)")

    return {
        "samples_path": samples_path,
        "base": base,
        "n_samples": n_samples,
        "n_format_ok": format_ok_count,
        "format_ok_rate": format_ok_rate,
        "n_valid": sum(sudoku_valids),
        "sudoku_valid_rate": sudoku_valid_rate,
        "avg_violations": avg_violations,
        "matches_original_rate": matches_original_rate,
        "avg_loss": avg_loss,
        "time_per_batch": time_per_batch,
        # Diversity metrics
        "avg_entropy": diversity["avg_entropy"],
        "max_entropy": diversity["max_entropy"],
        "normalized_entropy": diversity["normalized_entropy"],
        "samples_with_digits": diversity["samples_with_digits"],
    }


# ---------------- CLI ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated Sudoku samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate sudoku-small (base 3, 9x9 grid)
  python verify.py /path/to/samples.json --data sudoku-small

  # Evaluate sudoku-large (base 6, 36x36 grid) 
  python verify.py /path/to/samples.json --data sudoku-large

  # Or specify base directly
  python verify.py /path/to/samples.json --base 3

  # Verbose output (show per-sample results)
  python verify.py /path/to/samples.json --data sudoku-small -v
        """
    )
    
    parser.add_argument(
        "samples_path",
        type=str,
        help="Path to samples.json file"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data",
        type=str,
        choices=["sudoku-small", "sudoku-large"],
        help="Dataset name (sudoku-small=base3, sudoku-large=base6)"
    )
    group.add_argument(
        "--base",
        type=int,
        help="Sudoku base directly (3 for 9x9, 6 for 36x36)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print per-sample evaluation details"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: save results to CSV file"
    )
    
    parser.add_argument(
        "--include-malformed",
        action="store_true",
        help="Include malformed samples in diversity computation (default: skip them)"
    )
    
    args = parser.parse_args()
    
    # Determine base from data name or direct argument
    if args.data:
        base = {"sudoku-small": 3, "sudoku-large": 6}[args.data]
    else:
        base = args.base
    
    n = base * base  # side length
    print(f"Evaluating: {args.samples_path}")
    print(f"Sudoku type: base={base} ({n}x{n} grid)")
    print()
    
    results = evaluate_samples_file(
        args.samples_path,
        base=base,
        verbose=args.verbose,
        include_malformed=args.include_malformed,
    )
    
    # Compute maximums for context:
    # - Max violations: 3*n (n rows + n cols + n blocks, each can be violated)
    # - Max loss: 3*(n-1) (each of 3n units has max loss (n-1)/n when all digits are the same)
    max_violations = 3 * n
    max_loss = 3 * (n - 1)
    
    print()
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Total samples:      {results['n_samples']}")
    print(f"Format OK:          {results['n_format_ok']} ({results['format_ok_rate']*100:.1f}%)")
    print(f"Valid sudoku:       {results['n_valid']} ({results['sudoku_valid_rate']*100:.1f}%)")
    print(f"Avg violations:     {results['avg_violations']:.2f} / {max_violations}")
    print(f"Avg loss:           {results['avg_loss']:.4f} / {max_loss}")
    if results['time_per_batch'] is not None:
        print(f"Time per batch:     {results['time_per_batch']:.2f}s")
    print("-" * 50)
    print("DIVERSITY")
    print("-" * 50)
    print(f"Avg entropy:        {results['avg_entropy']:.4f} / {results['max_entropy']:.4f} bits")
    print(f"Normalized entropy: {results['normalized_entropy']*100:.1f}%")
    print(f"  (100% = uniform distribution, 0% = all identical)")
    print("=" * 50)
    
    if args.output:
        df = pd.DataFrame([results])
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()
