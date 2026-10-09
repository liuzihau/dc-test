"""
Sudoku Conditional dataset - procedurally generated puzzle-solution pairs.

This dataset generates sudoku puzzles without requiring unique solutions,
making it more suitable for training compared to datasets that only keep
unique-solution puzzles.

Key features:
- Generates puzzles procedurally using sudoku9.py solver
- Does NOT require unique solutions (configurable)
- Profiles solution count for each puzzle
- Format: [BOS] puzzle(81) [SEP] solution(81) [EOS] = 165 tokens
"""

import os
import random
import time
import typing
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from tqdm import tqdm
import transformers

from synthetic_data.sudoku.sudoku9 import (
    SudokuSolver9,
    generate_full_solution,
    copy_board,
    Board,
)


# ============================================================================
# Vocabulary - same as sudoku_puzzle for compatibility
# ============================================================================

# Token indices:
#   0: PAD (padding token)
#   1-9: digits 1-9 (digit value)
#   10: MASK (empty cell in puzzle)
#   11: SEP (separator between puzzle and solution)
#   12: BOS (beginning of sequence)
#   13: EOS (end of sequence)
SUDOKU_CONDITIONAL_VOCAB_TOKENS = [
    '<PAD>',   # 0: padding
    '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 1-9: digits
    '<M>',     # 10: empty cell in puzzle
    '<SEP>',   # 11: separator
    '<BOS>',   # 12: beginning of sequence
    '<EOS>',   # 13: end of sequence
]


def build_vocab():
    """Build vocabulary mappings."""
    vocab_map = {token: i for i, token in enumerate(SUDOKU_CONDITIONAL_VOCAB_TOKENS)}
    id_to_token = {i: token for i, token in enumerate(SUDOKU_CONDITIONAL_VOCAB_TOKENS)}
    return vocab_map, id_to_token


# ============================================================================
# Tokenizer
# ============================================================================

class SudokuConditionalTokenizer(transformers.PreTrainedTokenizer):
    """Tokenizer for sudoku conditional dataset.
    
    Uses the same vocabulary as SudokuPuzzleTokenizer for compatibility.
    """
    
    def __init__(
        self,
        bos_token='<BOS>',
        eos_token='<EOS>',
        sep_token='<SEP>',
        pad_token='<PAD>',
        **kwargs):
        
        self._vocab_str_to_int, self._vocab_int_to_str = build_vocab()
        
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            pad_token=pad_token,
            **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return text.strip().split()

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int['<PAD>'])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str.get(index, '<PAD>')

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int
    
    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        """Decode a sequence of token IDs to a string."""
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        tokens = []
        for tid in token_ids:
            token = self._convert_id_to_token(tid)
            if skip_special_tokens and token in ['<PAD>', '<BOS>', '<EOS>', '<SEP>', '<M>']:
                continue
            tokens.append(token)
        return ' '.join(tokens)
    
    def batch_decode(self, sequences, skip_special_tokens=False, **kwargs):
        """Decode a batch of token ID sequences."""
        return [self.decode(seq, skip_special_tokens=skip_special_tokens, **kwargs) 
                for seq in sequences]


# ============================================================================
# Puzzle Generation - allows multiple solutions
# ============================================================================

@dataclass
class ConditionalPuzzle:
    """A sudoku puzzle with its solution and metadata."""
    puzzle: Board
    solution: Board
    num_givens: int
    num_solutions: int  # Actual count (capped at max_count)
    num_solutions_capped: bool  # True if count >= max_count


def rotational_partner(index: int) -> int:
    """Get the rotationally symmetric partner of a cell index."""
    r, c = divmod(index, 9)
    rr, cc = 8 - r, 8 - c
    return rr * 9 + cc


def make_puzzle_multi_solution(
    target_givens: int = 30,
    max_count_solutions: int = 10,
    symmetry: str = "none",
    seed: Optional[int] = None,
    initial_solution: Optional[Board] = None,
    require_unique: bool = False,
    skip_solution_count: bool = True,
) -> ConditionalPuzzle:
    """
    Generate a sudoku puzzle, optionally allowing multiple solutions.
    
    Unlike make_puzzle() in sudoku9.py which enforces uniqueness, this function
    allows puzzles with multiple solutions (unless require_unique=True).
    
    Args:
        target_givens: Stop removing cells when this many remain (0 = no puzzle, just solution)
        max_count_solutions: Maximum number of solutions to count
        symmetry: "none" or "rotational"
        seed: Random seed for reproducibility
        initial_solution: Pre-generated solution board (optional)
        require_unique: If True, only remove cells that maintain uniqueness
        skip_solution_count: If True and require_unique=False, skip counting solutions (faster)
        
    Returns:
        ConditionalPuzzle with puzzle, solution, and solution count metadata
    """
    rng = random.Random(seed)
    
    # Generate or use provided solution
    solution = initial_solution if initial_solution is not None else generate_full_solution(rng=rng)
    work = copy_board(solution)
    
    def current_givens() -> int:
        return sum(work[r][c] != 0 for r in range(9) for c in range(9))
    
    # Fast path: if target_givens=0 and no constraints, just clear the board
    if target_givens == 0 and not require_unique:
        for r in range(9):
            for c in range(9):
                work[r][c] = 0
        return ConditionalPuzzle(
            puzzle=work,
            solution=solution,
            num_givens=0,
            num_solutions=-1,  # Unknown (not computed)
            num_solutions_capped=False,
        )
    
    # If we can skip solution counting (no uniqueness requirement), use fast path
    can_skip_counting = skip_solution_count and not require_unique
    
    if can_skip_counting:
        # Fast path: just remove cells until target_givens, no solution checking
        # Since we start from valid solution, puzzle is always solvable
        indices = list(range(81))
        rng.shuffle(indices)
        
        if symmetry == "rotational":
            seen = set()
            units = []
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
        
        for unit in units:
            if current_givens() <= target_givens:
                break
            for idx in unit:
                r, c = divmod(idx, 9)
                if work[r][c] != 0:
                    work[r][c] = 0
        
        return ConditionalPuzzle(
            puzzle=work,
            solution=solution,
            num_givens=current_givens(),
            num_solutions=-1,  # Unknown (not computed)
            num_solutions_capped=False,
        )
    
    # Slow path: need to count solutions (for require_unique or statistics)
    solver = SudokuSolver9(rng=rng)
    
    # Prepare cell removal order
    indices = list(range(81))
    rng.shuffle(indices)
    
    if symmetry == "rotational":
        seen = set()
        units = []
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
    
    # Remove cells until we reach target_givens
    for unit in units:
        if current_givens() <= target_givens:
            break
        
        # Try removing cells in this unit
        removed = []
        for idx in unit:
            r, c = divmod(idx, 9)
            if work[r][c] != 0:
                removed.append((r, c, work[r][c]))
                work[r][c] = 0
        
        if not removed:
            continue
        
        # Check solution count
        count, _ = solver.count_solutions(work, limit=max_count_solutions)
        
        if count == 0:
            # Puzzle became unsolvable - restore
            for r, c, v in removed:
                work[r][c] = v
        elif require_unique and count > 1:
            # Multiple solutions but we require unique - restore
            for r, c, v in removed:
                work[r][c] = v
        # Otherwise keep the removal (even if multiple solutions)
    
    # Final solution count
    final_count, _ = solver.count_solutions(work, limit=max_count_solutions)
    
    return ConditionalPuzzle(
        puzzle=work,
        solution=solution,
        num_givens=current_givens(),
        num_solutions=final_count,
        num_solutions_capped=(final_count >= max_count_solutions),
    )


# ============================================================================
# Serialization to token sequences
# ============================================================================

def board_to_tokens(board: Board, mask_token: int) -> np.ndarray:
    """
    Convert 9x9 board to 81 tokens.
    Digits 1-9 -> token IDs 1-9
    Empty (0) -> mask_token
    """
    arr = np.empty(81, dtype=np.int64)
    k = 0
    for r in range(9):
        for c in range(9):
            v = board[r][c]
            arr[k] = v if v > 0 else mask_token
            k += 1
    return arr


def solution_to_tokens(board: Board) -> np.ndarray:
    """
    Convert solved 9x9 board to 81 tokens.
    Digits 1-9 -> token IDs 1-9 (no masks)
    """
    arr = np.empty(81, dtype=np.int64)
    k = 0
    for r in range(9):
        for c in range(9):
            arr[k] = board[r][c]  # Should be 1-9
            k += 1
    return arr


def puzzle_to_sequence(puzzle: ConditionalPuzzle, vocab_map: dict) -> dict:
    """
    Convert a puzzle to token sequence.
    
    Format: [BOS] puzzle(81) [SEP] solution(81) [EOS]
    
    Returns:
        dict with 'input_ids' and 'loss_mask'
    """
    BOS = vocab_map['<BOS>']
    EOS = vocab_map['<EOS>']
    SEP = vocab_map['<SEP>']
    MASK = vocab_map['<M>']
    
    puzzle_tokens = board_to_tokens(puzzle.puzzle, MASK)
    solution_tokens = solution_to_tokens(puzzle.solution)
    
    # Format: [BOS] puzzle [SEP] solution [EOS]
    sequence = np.concatenate([
        [BOS],
        puzzle_tokens,
        [SEP],
        solution_tokens,
        [EOS]
    ]).astype(np.int64)
    
    # Loss mask: 0 for problem tokens, 1 for solution tokens
    # Problem: [BOS] + 81 puzzle + [SEP] = 83 tokens
    # Solution starts at index 83
    loss_mask = np.zeros(len(sequence), dtype=np.int64)
    loss_mask[83:] = 1
    
    return {
        'input_ids': sequence,
        'loss_mask': loss_mask,
    }


# ============================================================================
# Main data generation
# ============================================================================

def generate_synthetic_data(dataset_size, seq_len, config=None):
    """
    Generate sudoku conditional dataset.
    
    Args:
        dataset_size: Number of puzzles to generate
        seq_len: Sequence length (must be >= 165)
        config: dict with configuration:
            - vocab_size: int (must be 14)
            - target_givens: int, target number of given cells (default 30)
            - max_count_solutions: int, cap for solution counting (default 10)
            - symmetry: str, "none" or "rotational" (default "none")
            - require_unique: bool, if True only generate unique-solution puzzles (default False)
            - seed: int, random seed (optional)
            - cache_dir: str, directory for caching (optional)
            
    Returns:
        dict with:
            - 'input_ids': numpy array (dataset_size, seq_len)
            - 'loss_mask': numpy array (dataset_size, seq_len)
    """
    if config is None:
        config = {}
    
    vocab_map, id_to_token = build_vocab()
    built_vocab_size = len(vocab_map)
    
    # Validate vocab_size
    config_vocab_size = config.get("vocab_size")
    if config_vocab_size is not None and config_vocab_size != built_vocab_size:
        raise ValueError(
            f"Config vocab_size ({config_vocab_size}) does not match "
            f"hardcoded vocab_size ({built_vocab_size}). Please use vocab_size: {built_vocab_size}"
        )
    
    # Sequence length check
    # Format: [BOS] puzzle(81) [SEP] solution(81) [EOS] = 165 tokens
    min_seq_len = 165
    if seq_len < min_seq_len:
        raise ValueError(f"seq_len must be >= {min_seq_len} (got {seq_len})")
    
    # Config parameters
    # target_givens can be a single int or a range [min, max] for uniform sampling
    target_givens_config = config.get("target_givens", 30)
    # Handle OmegaConf ListConfig, regular list/tuple, or single value
    try:
        # Try to treat as a sequence (list, tuple, ListConfig)
        if hasattr(target_givens_config, '__len__') and not isinstance(target_givens_config, str) and len(target_givens_config) == 2:
            target_givens_min, target_givens_max = int(target_givens_config[0]), int(target_givens_config[1])
        else:
            target_givens_min = target_givens_max = int(target_givens_config)
    except (TypeError, ValueError):
        # Fallback for single value
        target_givens_min = target_givens_max = int(target_givens_config)
    
    max_count_solutions = config.get("max_count_solutions", 10)
    symmetry = config.get("symmetry", "none")
    require_unique = config.get("require_unique", False)
    skip_solution_count = config.get("skip_solution_count", True)  # Skip counting solutions by default
    base_seed = config.get("seed", None)
    
    # ---- caching setup ----
    cache_dir = config.get("cache_dir", None)
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        fname = f"sudoku_conditional_ds{dataset_size}_sl{seq_len}"
        if target_givens_min == target_givens_max:
            fname += f"_givens{target_givens_min}"
        else:
            fname += f"_givens{target_givens_min}-{target_givens_max}"
        if not skip_solution_count:
            fname += f"_maxsol{max_count_solutions}"
        fname += f"_sym{symmetry}_unique{require_unique}"
        if base_seed is not None:
            fname += f"_seed{base_seed}"
        fname += ".npz"
        cache_path = os.path.join(cache_dir, fname)
        
        # Try to load from cache
        if os.path.exists(cache_path):
            try:
                cached = np.load(cache_path, allow_pickle=True)
                cached_input_ids = cached["input_ids"]
                cached_loss_mask = cached["loss_mask"]
                if cached_input_ids.shape == (dataset_size, seq_len):
                    print(f"[SUDOKU-CONDITIONAL] Loading cached data from: {cache_path}")
                    return {
                        'input_ids': cached_input_ids,
                        'loss_mask': cached_loss_mask,
                    }
            except Exception:
                pass
    
    print(f"[SUDOKU-CONDITIONAL] Generating {dataset_size} puzzles...")
    if target_givens_min == target_givens_max:
        print(f"  target_givens={target_givens_min}")
    else:
        print(f"  target_givens=[{target_givens_min}, {target_givens_max}] (uniform)")
    if not skip_solution_count:
        print(f"  max_count_solutions={max_count_solutions}")
    print(f"  symmetry={symmetry}, require_unique={require_unique}, skip_solution_count={skip_solution_count}")
    
    PAD = vocab_map['<PAD>']
    
    # Initialize arrays
    # loss_mask starts as all 1s so padding tokens are included in loss
    # (loss_mask should be contiguous 0s from left, then 1s to end)
    input_ids = np.full((dataset_size, seq_len), fill_value=PAD, dtype=np.int64)
    loss_mask = np.ones((dataset_size, seq_len), dtype=np.int64)
    
    # Track solution count distribution for profiling
    solution_counts = []
    givens_counts = []
    
    # Create RNG for target_givens sampling (separate from puzzle generation)
    givens_rng = random.Random(base_seed if base_seed is not None else 42)
    
    # Generate puzzles
    for i in tqdm(range(dataset_size), desc="Generating puzzles"):
        # Deterministic seed per sample for reproducibility
        seed_i = None if base_seed is None else (base_seed + 1000003 * i)
        
        # Sample target_givens uniformly from range
        target_givens = givens_rng.randint(target_givens_min, target_givens_max)
        
        puzzle = make_puzzle_multi_solution(
            target_givens=target_givens,
            max_count_solutions=max_count_solutions,
            symmetry=symmetry,
            seed=seed_i,
            require_unique=require_unique,
            skip_solution_count=skip_solution_count,
        )
        
        # Convert to sequence
        result = puzzle_to_sequence(puzzle, vocab_map)
        seq = result['input_ids']
        seq_loss_mask = result['loss_mask']
        
        # Store (truncate if needed, shouldn't happen normally)
        seq_len_actual = min(len(seq), seq_len)
        input_ids[i, :seq_len_actual] = seq[:seq_len_actual]
        loss_mask[i, :seq_len_actual] = seq_loss_mask[:seq_len_actual]
        
        # Track statistics
        solution_counts.append(puzzle.num_solutions)
        givens_counts.append(puzzle.num_givens)
    
    # Print profiling statistics
    solution_counts = np.array(solution_counts)
    givens_counts = np.array(givens_counts)
    
    print("\n" + "=" * 60)
    print("SUDOKU-CONDITIONAL Dataset Statistics")
    print("=" * 60)
    print(f"Total puzzles: {dataset_size}")
    print("-" * 40)
    
    # Only show solution count distribution if we computed it
    if not skip_solution_count:
        print("Solution count distribution:")
        for count in range(1, max_count_solutions + 1):
            n = np.sum(solution_counts == count)
            pct = n / dataset_size * 100
            label = f"{count}" if count < max_count_solutions else f">={count}"
            if n > 0:
                print(f"  {label} solutions: {n} ({pct:.1f}%)")
        print("-" * 40)
        print(f"Unique solutions: {np.sum(solution_counts == 1)} ({np.sum(solution_counts == 1) / dataset_size * 100:.1f}%)")
        print(f"Multiple solutions: {np.sum(solution_counts > 1)} ({np.sum(solution_counts > 1) / dataset_size * 100:.1f}%)")
    else:
        print("(Solution counting skipped)")
    
    print(f"Givens: mean={givens_counts.mean():.1f}, min={givens_counts.min()}, max={givens_counts.max()}")
    print("=" * 60 + "\n")
    
    # Save to cache
    if cache_path:
        try:
            np.savez_compressed(
                cache_path,
                input_ids=input_ids,
                loss_mask=loss_mask,
                solution_counts=solution_counts,
                givens_counts=givens_counts,
                dataset_size=np.array(dataset_size, dtype=np.int64),
                seq_len=np.array(seq_len, dtype=np.int64),
                created_ts=np.array(time.time()),
            )
            print(f"[SUDOKU-CONDITIONAL] Saved cached data to: {cache_path}")
        except Exception:
            pass
    
    return {
        'input_ids': input_ids,
        'loss_mask': loss_mask,
    }


# ============================================================================
# Evaluation utilities (reused from sudoku_puzzle with minor adaptations)
# ============================================================================

def extract_solution_tokens(token_ids, vocab_map=None):
    """Extract solution tokens from a generated sequence."""
    if vocab_map is None:
        vocab_map, _ = build_vocab()
    
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()
    
    SEP = vocab_map['<SEP>']
    EOS = vocab_map['<EOS>']
    PAD = vocab_map['<PAD>']
    
    try:
        sep_idx = token_ids.index(SEP)
    except ValueError:
        # No SEP found - try BOS format
        BOS = vocab_map['<BOS>']
        try:
            bos_idx = token_ids.index(BOS)
            solution_tokens = []
            for tid in token_ids[bos_idx + 1:]:
                if tid == EOS or tid == PAD:
                    break
                solution_tokens.append(tid)
            return solution_tokens
        except ValueError:
            return []
    
    # Extract tokens after SEP until EOS or PAD
    solution_tokens = []
    for tid in token_ids[sep_idx + 1:]:
        if tid == EOS or tid == PAD:
            break
        solution_tokens.append(tid)
    
    return solution_tokens


def decode_solution_tokens(solution_tokens, id_to_token=None):
    """Convert solution token IDs to integer values (1-9)."""
    if id_to_token is None:
        _, id_to_token = build_vocab()
    
    values = []
    for tid in solution_tokens:
        token_str = id_to_token.get(tid, '')
        if token_str.isdigit():
            values.append(int(token_str))
        else:
            values.append(-1)
    
    return values


def solution_to_grid(solution_values):
    """Reshape flat solution values into a 9x9 grid."""
    if len(solution_values) != 81:
        return None
    return np.array(solution_values, dtype=np.int32).reshape(9, 9)


def check_sudoku_valid(grid):
    """Check if a sudoku grid is valid (no repeated digits in rows/cols/boxes)."""
    if grid is None:
        return False
    
    # Check rows
    for row in grid:
        if len(set(row)) != 9 or not all(1 <= v <= 9 for v in row):
            return False
    
    # Check columns
    for col in grid.T:
        if len(set(col)) != 9:
            return False
    
    # Check 3x3 boxes
    for box_row in range(3):
        for box_col in range(3):
            box = grid[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3].flatten()
            if len(set(box)) != 9:
                return False
    
    return True


def extract_puzzle_tokens(token_ids, vocab_map=None):
    """
    Extract puzzle tokens from sequence (before SEP).
    
    Format: [BOS] puzzle(81) [SEP] solution(81) [EOS]
    Returns the 81 puzzle tokens.
    """
    if vocab_map is None:
        vocab_map, _ = build_vocab()
    
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()
    
    BOS = vocab_map['<BOS>']
    SEP = vocab_map['<SEP>']
    
    # Find BOS and SEP positions
    try:
        bos_idx = token_ids.index(BOS)
        sep_idx = token_ids.index(SEP)
    except ValueError:
        return []
    
    # Puzzle is between BOS and SEP
    puzzle_tokens = token_ids[bos_idx + 1:sep_idx]
    return puzzle_tokens


def decode_puzzle_tokens(puzzle_tokens, id_to_token=None):
    """
    Convert puzzle token IDs to grid values.
    Digits 1-9 stay as is, MASK (<M>) becomes 0.
    """
    if id_to_token is None:
        _, id_to_token = build_vocab()
    
    values = []
    for tid in puzzle_tokens:
        token_str = id_to_token.get(tid, '')
        if token_str.isdigit():
            values.append(int(token_str))
        elif token_str == '<M>':
            values.append(0)  # Empty cell
        else:
            values.append(-1)  # Invalid
    
    return values


def puzzle_to_grid(puzzle_values):
    """Reshape flat puzzle values into a 9x9 grid."""
    if len(puzzle_values) != 81:
        return None
    return np.array(puzzle_values, dtype=np.int32).reshape(9, 9)


def check_solution_consistent_with_puzzle(solution_grid, puzzle_grid):
    """
    Check if a solution is consistent with the puzzle's given cells.
    
    Args:
        solution_grid: 9x9 grid with solution (values 1-9)
        puzzle_grid: 9x9 grid with puzzle (0 for empty, 1-9 for givens)
    
    Returns:
        True if all given cells in puzzle match the solution
    """
    if solution_grid is None or puzzle_grid is None:
        return False
    
    # Check that given cells (non-zero in puzzle) match the solution
    given_mask = puzzle_grid > 0
    return np.all(solution_grid[given_mask] == puzzle_grid[given_mask])


def check_solution(predicted_grid, ground_truth_grid, puzzle_grid=None):
    """
    Evaluate a predicted solution.
    
    For puzzles with multiple solutions, a solution is considered correct if:
    1. It's a valid sudoku (no constraint violations)
    2. It's consistent with the puzzle (given cells match)
    
    Args:
        predicted_grid: 9x9 predicted solution
        ground_truth_grid: 9x9 ground truth solution (one possible solution)
        puzzle_grid: 9x9 puzzle grid (0 for empty, 1-9 for givens), optional
    """
    if predicted_grid is None:
        return {
            'cell_accuracy': 0.0,
            'row_accuracy': 0.0,
            'column_accuracy': 0.0,
            'box_accuracy': 0.0,
            'matches_ground_truth': False,
            'is_valid_solution': False,
            'n_correct_cells': 0,
            'n_total_cells': 81,
            'is_valid_sudoku': False,
            'is_consistent_with_puzzle': False,
            'error_type': 'malformed',
        }
    
    # Check if valid sudoku
    is_valid_sudoku = check_sudoku_valid(predicted_grid)
    
    # Check if consistent with puzzle (if puzzle provided)
    if puzzle_grid is not None:
        is_consistent = check_solution_consistent_with_puzzle(predicted_grid, puzzle_grid)
    else:
        is_consistent = True  # Assume consistent if no puzzle provided
    
    # A valid solution = valid sudoku AND consistent with puzzle
    is_valid_solution = is_valid_sudoku and is_consistent
    
    # Compare with ground truth
    cell_correct = (predicted_grid == ground_truth_grid)
    n_correct_cells = np.sum(cell_correct)
    matches_ground_truth = np.all(cell_correct)
    
    row_correct = np.all(cell_correct, axis=1)
    n_correct_rows = np.sum(row_correct)
    
    col_correct = np.all(cell_correct, axis=0)
    n_correct_cols = np.sum(col_correct)
    
    n_correct_boxes = 0
    for box_row in range(3):
        for box_col in range(3):
            box = cell_correct[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3]
            if np.all(box):
                n_correct_boxes += 1
    
    # Determine error type
    if is_valid_solution:
        error_type = 'none'  # Valid solution (may or may not match ground truth)
    elif not is_valid_sudoku:
        error_type = 'invalid_sudoku'
    elif not is_consistent:
        error_type = 'inconsistent_with_puzzle'
    else:
        error_type = 'unknown'
    
    return {
        'cell_accuracy': n_correct_cells / 81,
        'row_accuracy': n_correct_rows / 9,
        'column_accuracy': n_correct_cols / 9,
        'box_accuracy': n_correct_boxes / 9,
        'matches_ground_truth': bool(matches_ground_truth),
        'is_valid_solution': is_valid_solution,
        'n_correct_cells': int(n_correct_cells),
        'n_total_cells': 81,
        'is_valid_sudoku': is_valid_sudoku,
        'is_consistent_with_puzzle': is_consistent,
        'error_type': error_type,
    }


def evaluate_completions(predicted_ids, ground_truth_ids):
    """
    Evaluate puzzle completions.
    
    For puzzles with multiple solutions, a solution is considered correct if:
    1. It's a valid sudoku (no constraint violations)
    2. It's consistent with the puzzle (given cells match)
    
    We also report how often the prediction matches the stored ground truth,
    but the primary metric is whether it's a valid solution to the puzzle.
    
    Args:
        predicted_ids: numpy array (n_samples, seq_len) - model outputs
        ground_truth_ids: numpy array (n_samples, seq_len) - ground truth
        
    Returns:
        dict with evaluation metrics
    """
    vocab_map, id_to_token = build_vocab()
    
    results = []
    
    for pred_seq, gt_seq in zip(predicted_ids, ground_truth_ids):
        # Extract puzzle (given cells)
        puzzle_tokens = extract_puzzle_tokens(gt_seq, vocab_map)
        puzzle_values = decode_puzzle_tokens(puzzle_tokens, id_to_token)
        puzzle_grid = puzzle_to_grid(puzzle_values)
        
        # Extract solutions
        gt_solution = extract_solution_tokens(gt_seq, vocab_map)
        pred_solution = extract_solution_tokens(pred_seq, vocab_map)
        
        pred_values = decode_solution_tokens(pred_solution, id_to_token)
        gt_values = decode_solution_tokens(gt_solution, id_to_token)
        
        pred_grid = solution_to_grid(pred_values)
        gt_grid = solution_to_grid(gt_values)

        if pred_grid is None:
            print(f"Predicted grid is None")
            print(f"Predicted sequence: {pred_seq}")
            print(f"Predicted values: {pred_values}")
        # Evaluate with puzzle context
        result = check_solution(pred_grid, gt_grid, puzzle_grid)
        results.append(result)
    
    # Aggregate metrics
    n = len(results)
    total_cell_acc = sum(r['cell_accuracy'] for r in results)
    total_row_acc = sum(r['row_accuracy'] for r in results)
    total_col_acc = sum(r['column_accuracy'] for r in results)
    total_box_acc = sum(r['box_accuracy'] for r in results)
    
    # Primary metric: valid solutions (valid sudoku + consistent with puzzle)
    total_valid_solutions = sum(1 for r in results if r['is_valid_solution'])
    
    # Secondary metric: matches ground truth exactly
    total_matches_gt = sum(1 for r in results if r['matches_ground_truth'])
    
    # Breakdown
    total_valid_sudoku = sum(1 for r in results if r['is_valid_sudoku'])
    total_consistent = sum(1 for r in results if r['is_consistent_with_puzzle'])
    
    n_malformed = sum(1 for r in results if r['error_type'] == 'malformed')
    n_invalid_sudoku = sum(1 for r in results if r['error_type'] == 'invalid_sudoku')
    n_inconsistent = sum(1 for r in results if r['error_type'] == 'inconsistent_with_puzzle')
    
    metrics = {
        # Primary metrics
        'puzzle_accuracy': total_valid_solutions / n if n > 0 else 0.0,  # Valid solutions
        'n_valid_solutions': total_valid_solutions,
        'valid_solution_rate': total_valid_solutions / n if n > 0 else 0.0,
        
        # Ground truth comparison (secondary)
        'ground_truth_match_rate': total_matches_gt / n if n > 0 else 0.0,
        'n_matches_ground_truth': total_matches_gt,
        
        # Cell-level accuracy (vs ground truth)
        'mean_cell_accuracy': total_cell_acc / n if n > 0 else 0.0,
        'mean_row_accuracy': total_row_acc / n if n > 0 else 0.0,
        'mean_column_accuracy': total_col_acc / n if n > 0 else 0.0,
        'mean_box_accuracy': total_box_acc / n if n > 0 else 0.0,
        
        # Validity breakdown
        'valid_sudoku_rate': total_valid_sudoku / n if n > 0 else 0.0,
        'consistent_with_puzzle_rate': total_consistent / n if n > 0 else 0.0,
        
        # Error counts
        'n_total_puzzles': n,
        'n_malformed': n_malformed,
        'n_invalid_sudoku': n_invalid_sudoku,
        'n_inconsistent_with_puzzle': n_inconsistent,
    }
    
    print("\n" + "=" * 60)
    print("Sudoku Conditional Evaluation Results")
    print("=" * 60)
    print(f"  Puzzles evaluated: {metrics['n_total_puzzles']}")
    print("-" * 40)
    print("  PRIMARY METRIC (valid solution to puzzle):")
    print(f"    Valid solutions: {metrics['n_valid_solutions']} ({metrics['puzzle_accuracy']*100:.1f}%)")
    print("-" * 40)
    print("  Validity breakdown:")
    print(f"    Valid sudoku (no constraint violations): {metrics['valid_sudoku_rate']*100:.1f}%")
    print(f"    Consistent with puzzle (givens match):   {metrics['consistent_with_puzzle_rate']*100:.1f}%")
    print("-" * 40)
    print("  Ground truth comparison (secondary):")
    print(f"    Exact match with stored solution: {metrics['n_matches_ground_truth']} ({metrics['ground_truth_match_rate']*100:.1f}%)")
    print(f"    Cell accuracy vs ground truth: {metrics['mean_cell_accuracy']*100:.1f}%")
    print("-" * 40)
    print("  Error breakdown:")
    print(f"    Malformed (wrong length): {metrics['n_malformed']}")
    print(f"    Invalid sudoku: {metrics['n_invalid_sudoku']}")
    print(f"    Inconsistent with puzzle: {metrics['n_inconsistent_with_puzzle']}")
    print("=" * 60 + "\n")
    
    return metrics
