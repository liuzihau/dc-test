#!/usr/bin/env python3
"""
sudoku9_gen_cli.py

Generate 9x9 Sudoku puzzles (base-3) and serialize to your legacy sequence format:
[BOS] + 81 tokens (digits 1..9 -> 0..8; blanks -> MASK=vocab_size) + [EOS] + optional padding.

Output: a single .npz with arrays:
  - puzzles   : (num_samples, seq_len) int
  - solutions : (num_samples, seq_len) int
  - meta      : (num_samples,) object (JSON strings)

Usage:
  python sudoku9_gen_cli.py \
      --num-samples 1000 \
      --seq-len 96 \
      --vocab-size 11 \
      --target-givens 30 \
      --symmetry rotational \
      --seed 1337 \
      --out /path/to/sudoku_puzzles.npz
"""

from __future__ import annotations
import argparse, json, random, time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable
import numpy as np

# -------------------------
# Board helpers & bit utils
# -------------------------

Board = List[List[int]]  # 0..9 (0 = blank)
ALL_DIG_MASK = 0b1111111110
DIG_TO_BIT = [0] + [1 << d for d in range(1, 10)]
BIT_TO_DIG = {1 << d: d for d in range(1, 10)}

def popcount(x: int) -> int:
    cnt = 0
    while x:
        x &= x - 1
        cnt += 1
    return cnt

def box_index(r: int, c: int) -> int:
    return (r // 3) * 3 + (c // 3)

def empty_board() -> Board:
    return [[0 for _ in range(9)] for _ in range(9)]

def copy_board(b: Board) -> Board:
    return [row[:] for row in b]

# -------------------------
# Solver (MRV + bit masks)
# -------------------------

@dataclass
class SolveStats:
    node_expansions: int
    backtracks: int
    max_depth: int
    time_ms: float

@dataclass
class SolveResult:
    solved: bool
    solution: Optional[Board]
    stats: SolveStats

class SudokuSolver9:
    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def _init_masks(self, b: Board):
        row_mask = [0]*9; col_mask = [0]*9; box_mask = [0]*9
        for r in range(9):
            for c in range(9):
                v = b[r][c]
                if v:
                    bit = DIG_TO_BIT[v]
                    if (row_mask[r] & bit) or (col_mask[c] & bit) or (box_mask[box_index(r,c)] & bit):
                        raise ValueError("Invalid board: duplicate digit in row/col/box.")
                    row_mask[r] |= bit; col_mask[c] |= bit; box_mask[box_index(r,c)] |= bit
        return row_mask, col_mask, box_mask

    def _find_mrv_cell(self, b: Board, row_mask, col_mask, box_mask):
        best_r = best_c = -1
        best_cands = 0
        best_count = 10
        for r in range(9):
            for c in range(9):
                if b[r][c] == 0:
                    used = row_mask[r] | col_mask[c] | box_mask[box_index(r,c)]
                    cand_bits = (~used) & ALL_DIG_MASK
                    cnt = popcount(cand_bits)
                    if cnt == 0:
                        return r, c, 0
                    if cnt < best_count:
                        best_count = cnt
                        best_r, best_c, best_cands = r, c, cand_bits
                        if cnt == 1:
                            return best_r, best_c, best_cands
        return best_r, best_c, best_cands

    def _iter_bits_randomized(self, bits: int):
        if bits == 0: 
            return
            yield
        arr = []
        while bits:
            lsb = bits & -bits
            arr.append(lsb)
            bits ^= lsb
        self.rng.shuffle(arr)
        for b in arr:
            yield b

    def solve(self, b: Board) -> SolveResult:
        start = time.time()
        board = copy_board(b)
        row_mask, col_mask, box_mask = self._init_masks(board)
        node_exp, backs, max_depth = 0, 0, 0

        def dfs(depth: int) -> bool:
            nonlocal node_exp, backs, max_depth
            max_depth = max(max_depth, depth)
            r, c, cand_bits = self._find_mrv_cell(board, row_mask, col_mask, box_mask)
            if r == -1:  # solved
                return True
            if cand_bits == 0:
                backs += 1
                return False
            node_exp += 1
            for bit in self._iter_bits_randomized(cand_bits):
                d = BIT_TO_DIG[bit]
                board[r][c] = d
                row_mask[r] |= bit; col_mask[c] |= bit; box_mask[box_index(r,c)] |= bit
                if dfs(depth+1): 
                    return True
                board[r][c] = 0
                row_mask[r] ^= bit; col_mask[c] ^= bit; box_mask[box_index(r,c)] ^= bit
            backs += 1
            return False

        solved = dfs(0)
        end = time.time()
        stats = SolveStats(node_expansions=node_exp, backtracks=backs, max_depth=max_depth, time_ms=(end-start)*1000)
        return SolveResult(solved=solved, solution=board if solved else None, stats=stats)

    def count_solutions(self, b: Board, limit: int = 2):
        board = copy_board(b)
        row_mask, col_mask, box_mask = self._init_masks(board)
        count = 0
        first: Optional[Board] = None

        def dfs() -> bool:
            nonlocal count, first
            r, c, cand_bits = self._find_mrv_cell(board, row_mask, col_mask, box_mask)
            if r == -1:
                count += 1
                if first is None:
                    first = copy_board(board)
                return count >= limit
            if cand_bits == 0:
                return False
            for bit in self._iter_bits_randomized(cand_bits):
                d = BIT_TO_DIG[bit]
                board[r][c] = d
                row_mask[r] |= bit; col_mask[c] |= bit; box_mask[box_index(r,c)] |= bit
                if dfs(): 
                    return True
                board[r][c] = 0
                row_mask[r] ^= bit; col_mask[c] ^= bit; box_mask[box_index(r,c)] ^= bit
            return False

        dfs()
        return count, first

# -------------------------
# Full solution + puzzle gen
# -------------------------

def generate_full_solution(rng: Optional[random.Random] = None) -> Board:
    rng = rng or random.Random()
    solver = SudokuSolver9(rng=rng)
    b = empty_board()
    # seed first row with a few digits to diversify
    digits = list(range(1,10)); rng.shuffle(digits)
    cols = list(range(9)); rng.shuffle(cols)
    for c in cols[:3]:
        b[0][c] = digits.pop()
    res = solver.solve(b)
    if not res.solved or res.solution is None:
        res = solver.solve(empty_board())
        if not res.solved or res.solution is None:
            # Retry generation instead of failing
            for _ in range(5):
                res = solver.solve(empty_board())
                if res.solved and res.solution is not None:
                    return res.solution
            raise RuntimeError("Failed to generate a full solution after multiple retries.")
    return res.solution


@dataclass
class Puzzle:
    puzzle: Board
    solution: Board
    unique: bool
    num_givens: int
    symmetry: str
    attempts: int

def rotational_partner(index: int) -> int:
    r, c = divmod(index, 9)
    rr, cc = 8 - r, 8 - c
    return rr*9 + cc

def make_puzzle(
    target_givens: int = 30,
    symmetry: str = "none",   # "none" | "rotational"
    seed: Optional[int] = None,
    max_fail_streak: int = 40,
    initial_solution: Optional[Board] = None,
    num_solutions: int = 1,
) -> Puzzle:
    rng = random.Random(seed)
    
    solution = initial_solution if initial_solution is not None else generate_full_solution(rng=rng)
    
    work = copy_board(solution)
    solver = SudokuSolver9(rng=rng)
    limit = num_solutions + 1

    indices = list(range(81))
    rng.shuffle(indices)
    if symmetry == "rotational":
        seen = set(); units = []
        for i in indices:
            j = rotational_partner(i)
            a, b = (i, j) if i <= j else (j, i)
            if (a, b) not in seen:
                seen.add((a, b))
                units.append((a, b) if a != b else (a,))
    elif symmetry == "none":
        units = [(i,) for i in indices]
    else:
        raise ValueError("symmetry must be 'none' or 'rotational'")

    attempts = 0
    fail_streak = 0
    def current_givens() -> int:
        return sum(work[r][c] != 0 for r in range(9) for c in range(9))

    for unit in units:
        if current_givens() <= target_givens:
            break
        removed = []
        for idx in unit:
            r, c = divmod(idx, 9)
            if work[r][c] != 0:
                removed.append((r, c, work[r][c]))
                work[r][c] = 0
        attempts += 1
        count, _ = solver.count_solutions(work, limit=limit)
        if count == 1:
            fail_streak = 0
        else:
            for r, c, v in removed:
                work[r][c] = v
            fail_streak += 1
            if fail_streak >= max_fail_streak:
                break

    final_count, _ = solver.count_solutions(work, limit=limit)
    return Puzzle(
        puzzle=work,
        solution=solution,
        unique=(final_count == 1),
        num_givens=sum(work[r][c] != 0 for r in range(9) for c in range(9)),
        symmetry=symmetry,
        attempts=attempts,
    )

# -------------------------
# Serialization to legacy format
# -------------------------

def board_to_tokens_flat(b: Board, mask_token: int) -> np.ndarray:
    """
    Map 9x9 board to length-81 flat tokens:
      - digits 1..9 -> 0..8
      - 0 (blank)   -> mask_token
    """
    arr = np.empty(81, dtype=np.int64)
    k = 0
    for r in range(9):
        for c in range(9):
            v = b[r][c]
            arr[k] = (v - 1) if v > 0 else mask_token
            k += 1
    return arr

def flat_tokens_to_board(arr: np.ndarray) -> Board:
    return (arr.reshape(9, 9) + 1).tolist()

def board_to_tokens_flat_solution(b: Board) -> np.ndarray:
    """
    Full solution mapping:
      digits 1..9 -> 0..8 (no masks).
    """
    arr = np.empty(81, dtype=np.int64)
    k = 0
    for r in range(9):
        for c in range(9):
            v = b[r][c]
            arr[k] = (v - 1)  # v in 1..9
            k += 1
    return arr

def make_sequences(puzzles: List[Puzzle], seq_len: int, vocab_size: int):
    """
    Build:
      puzzles_seq  : masked where blank == vocab_size
      solutions_seq: full digits (no masks)
    Layout: [BOS] + 81 + [EOS] + padding (EOS)
    """
    min_seq_len = 83
    if seq_len < min_seq_len:
        raise ValueError(f"seq_len must be >= {min_seq_len} (got {seq_len}).")
    if vocab_size < 11:
        raise ValueError(f"vocab_size must be >= 11 (got {vocab_size}).")

    BOS = vocab_size - 2
    EOS = vocab_size - 1
    MASK = vocab_size  # (as requested)

    puzzles_seq = np.full((len(puzzles), seq_len), fill_value=EOS, dtype=np.int64)
    solutions_seq = np.full((len(puzzles), seq_len), fill_value=EOS, dtype=np.int64)

    puzzles_seq[:, 0] = BOS
    puzzles_seq[:, min_seq_len - 1] = EOS

    solutions_seq[:, 0] = BOS
    solutions_seq[:, min_seq_len - 1] = EOS

    for i, pz in enumerate(puzzles):
        puzzles_seq[i, 1:82]  = board_to_tokens_flat(pz.puzzle, mask_token=MASK)
        solutions_seq[i, 1:82] = board_to_tokens_flat_solution(pz.solution)

    return puzzles_seq, solutions_seq

# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate 9x9 Sudoku puzzles in legacy sequence format.")
    ap.add_argument("--num-samples", type=int, required=True)
    ap.add_argument("--seq-len", type=int, required=True, help=">= 83")
    ap.add_argument("--vocab-size", type=int, required=True, help=">= 11; MASK will be vocab_size")
    ap.add_argument("--target-givens", type=int, default=30, help="minimum givens (stop removing when reached)")
    ap.add_argument("--symmetry", type=str, default="none", choices=["none", "rotational"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=str, required=True, help="output .npz path")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    puzzles: List[Puzzle] = []
    for i in range(args.num_samples):
        # Vary seed per sample deterministically so runs are reproducible and distinct
        seed_i = None if args.seed is None else (args.seed + 1000003*i)
        pz = make_puzzle(
            target_givens=args.target_givens,
            symmetry=args.symmetry,
            seed=seed_i,
        )
        if not pz.unique:
            # extremely rare; regenerate until unique (safeguard)
            # keep a bounded retry to avoid infinite loops
            for _ in range(5):
                seed_i = None if args.seed is None else (seed_i + 1)
                pz = make_puzzle(
                    target_givens=args.target_givens,
                    symmetry=args.symmetry,
                    seed=seed_i,
                )
                if pz.unique:
                    break
        puzzles.append(pz)

    puzzles_seq, solutions_seq = make_sequences(puzzles, seq_len=args.seq_len, vocab_size=args.vocab_size)

    # Minimal per-sample metadata (JSON strings so you can load as an array of objects)
    meta = np.array([
        json.dumps({
            "unique": pz.unique,
            "num_givens": pz.num_givens,
            "symmetry": pz.symmetry,
            "attempts": pz.attempts,
        })
        for pz in puzzles
    ], dtype=object)

    np.savez_compressed(
        args.out,
        puzzles=puzzles_seq,
        solutions=solutions_seq,
        meta=meta,
        seq_len=np.array(args.seq_len, dtype=np.int64),
        vocab_size=np.array(args.vocab_size, dtype=np.int64),
        target_givens=np.array(args.target_givens, dtype=np.int64),
        symmetry=np.array(args.symmetry),
        seed=np.array(-1 if args.seed is None else args.seed, dtype=np.int64),
    )

if __name__ == "__main__":
    main()