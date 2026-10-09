"""
Sudoku dataset loader for diffusion-vs-ar data.

Loads 9x9 sudoku puzzle-solution pairs from CSV files.

Data format:
    CSV with columns: quizzes, solutions
    Each puzzle/solution is an 81-character string (row-major order)
    '0' in puzzle means empty cell, digits 1-9 are given cells

Sequence format:
    [BOS] puzzle(81) [SEP] solution(81) [EOS] [PAD...]

Vocab (14 tokens):
    <PAD>=0, <BOS>=1, <EOS>=2, <SEP>=3, <M>=4 (empty/masked cell), 1-9=5-13
"""

import numpy as np
import os
import typing
import transformers
from tqdm import tqdm

from .base import load_csv, pad_sequence, create_loss_mask, BaseTokenizer, save_to_cache, load_from_cache, resolve_data_path


# ============================================================================
# Vocabulary Definition
# ============================================================================

DVAR_SUDOKU_VOCAB_TOKENS = [
    '<PAD>',   # 0: padding token
    '<BOS>',   # 1: beginning of sequence
    '<EOS>',   # 2: end of sequence
    '<SEP>',   # 3: separator
    '<M>',     # 4: masked/empty cell (0 in CSV)
    '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 5-13: digits 1-9
]

# Token ID constants
PAD = 0
BOS = 1
EOS = 2
SEP = 3
MASK = 4  # Empty cell marker
DIGIT_OFFSET = 4  # digit d (1-9) has token ID d + 4

VOCAB_SIZE = len(DVAR_SUDOKU_VOCAB_TOKENS)


class DvarSudokuTokenizer(transformers.PreTrainedTokenizer):
    """Tokenizer for diffusion-vs-ar Sudoku dataset."""

    def __init__(
        self,
        bos_token='<BOS>',
        eos_token='<EOS>',
        sep_token='<SEP>',
        pad_token='<PAD>',
        **kwargs
    ):
        self._vocab_str_to_int = {token: i for i, token in enumerate(DVAR_SUDOKU_VOCAB_TOKENS)}
        self._vocab_int_to_str = {i: token for i, token in enumerate(DVAR_SUDOKU_VOCAB_TOKENS)}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            pad_token=pad_token,
            **kwargs
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return text.strip().split()

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, PAD)

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str.get(index, '<PAD>')

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int.copy()

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        """Decode a sequence of token IDs to a string."""
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()

        tokens = []
        for tid in token_ids:
            token = self._convert_id_to_token(tid)
            if skip_special_tokens and token in ['<PAD>', '<BOS>', '<EOS>', '<SEP>']:
                continue
            tokens.append(token)
        return ' '.join(tokens)

    def batch_decode(self, sequences, skip_special_tokens=False, **kwargs):
        """Decode a batch of token ID sequences."""
        return [self.decode(seq, skip_special_tokens=skip_special_tokens, **kwargs)
                for seq in sequences]


def char_to_token(c: str) -> int:
    """Convert a sudoku character to a token ID.

    '0' -> MASK (4)
    '1'-'9' -> 5-13
    """
    if c == '0':
        return MASK
    d = int(c)
    if 1 <= d <= 9:
        return DIGIT_OFFSET + d
    raise ValueError(f"Invalid sudoku character: {c}")


def token_to_char(t: int) -> str:
    """Convert a token ID back to a sudoku character."""
    if t == MASK:
        return '0'
    if DIGIT_OFFSET + 1 <= t <= DIGIT_OFFSET + 9:
        return str(t - DIGIT_OFFSET)
    return '?'


def encode_sudoku_pair(puzzle: str, solution: str) -> dict:
    """Encode a puzzle-solution pair as tokens.

    Args:
        puzzle: 81-char string with '0' for empty cells
        solution: 81-char string with all digits filled

    Returns:
        dict with:
            - 'tokens': list of token IDs (without BOS/EOS)
            - 'solution_start': index where solution begins
    """
    tokens = []

    # Encode puzzle (81 characters)
    for c in puzzle:
        tokens.append(char_to_token(c))

    tokens.append(SEP)
    solution_start = len(tokens)

    # Encode solution (81 characters)
    for c in solution:
        tokens.append(char_to_token(c))

    return {
        'tokens': tokens,
        'solution_start': solution_start,
    }


def get_cache_filename(dataset_size: int, seq_len: int, split: str) -> str:
    """Generate a cache filename."""
    return f"dvar_sudoku_{split}_ds{dataset_size}_sl{seq_len}.npz"


def generate_synthetic_data(dataset_size, seq_len, config=None):
    """
    Load and tokenize Sudoku data from CSV files.

    Args:
        dataset_size: number of samples to use (None for all)
        seq_len: sequence length (will pad if shorter)
        config: dict with configuration. Key fields:
            - data_path: str, path to CSV file
            - cache_dir: str, optional cache directory
            - split: str, 'train' or 'test' for cache naming

    Returns:
        dict with:
            - 'input_ids': numpy array of shape (dataset_size, seq_len)
            - 'loss_mask': numpy array of shape (dataset_size, seq_len)
    """
    if config is None:
        config = {}

    data_path = config.get('data_path')
    if data_path is None:
        raise ValueError("data_path must be provided for dvar-sudoku")

    cache_dir = config.get('cache_dir')
    split = config.get('split', 'train')

    # Check cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, get_cache_filename(
            dataset_size if dataset_size else 0, seq_len, split))
        cached = load_from_cache(cache_path, dataset_size, seq_len)
        if cached is not None:
            print(f"[DVAR-SUDOKU] Loaded from cache: {cache_path}")
            return cached

    # Load data
    data_path = resolve_data_path(data_path)
    print(f"[DVAR-SUDOKU] Loading from: {data_path}")
    data = load_csv(data_path, has_header=True)

    if dataset_size is None:
        dataset_size = len(data)
    else:
        dataset_size = min(dataset_size, len(data))

    print(f"[DVAR-SUDOKU] Processing {dataset_size} examples")

    # Pre-allocate arrays
    input_ids = np.full((dataset_size, seq_len), fill_value=PAD, dtype=np.int64)
    loss_mask = np.zeros((dataset_size, seq_len), dtype=np.int64)

    max_tokens_seen = 0

    for i in tqdm(range(dataset_size)):
        row = data[i]
        puzzle = row['quizzes']
        solution = row['solutions']

        # Encode
        encoded = encode_sudoku_pair(puzzle, solution)
        tokens = encoded['tokens']
        solution_start_in_tokens = encoded['solution_start']

        # Build full sequence: [BOS] + tokens + [EOS]
        full_seq = [BOS] + tokens + [EOS]
        total_len = len(full_seq)

        if total_len > seq_len:
            raise ValueError(
                f"Example {i} requires {total_len} tokens but seq_len is {seq_len}. "
                f"Increase seq_len."
            )

        max_tokens_seen = max(max_tokens_seen, total_len)

        # Fill input_ids
        input_ids[i, :total_len] = full_seq

        # Create loss_mask: 0 for puzzle, 1 for solution (including EOS and padding)
        solution_start = solution_start_in_tokens + 1  # +1 for BOS
        loss_mask[i, solution_start:] = 1

    print(f"[DVAR-SUDOKU] Done! Max tokens: {max_tokens_seen}/{seq_len}")

    # Save to cache
    if cache_dir:
        save_to_cache(cache_path, input_ids, loss_mask, {
            'dataset_size': dataset_size,
            'seq_len': seq_len,
            'vocab_size': VOCAB_SIZE,
        })
        print(f"[DVAR-SUDOKU] Saved to cache: {cache_path}")

    return {
        'input_ids': input_ids,
        'loss_mask': loss_mask,
    }


def is_valid_sudoku(grid: np.ndarray) -> bool:
    """Check if a 9x9 grid is a valid sudoku solution.

    Args:
        grid: 9x9 numpy array with values 1-9

    Returns:
        True if valid, False otherwise
    """
    for i in range(9):
        # Check row
        if len(set(grid[i, :])) != 9 or not all(1 <= v <= 9 for v in grid[i, :]):
            return False
        # Check column
        if len(set(grid[:, i])) != 9:
            return False

    # Check 3x3 boxes
    for box_r in range(3):
        for box_c in range(3):
            box = grid[box_r*3:(box_r+1)*3, box_c*3:(box_c+1)*3].flatten()
            if len(set(box)) != 9:
                return False

    return True


def extract_solution_from_sequence(seq) -> np.ndarray:
    """Extract the 9x9 solution grid from a token sequence.

    Args:
        seq: numpy array or list of token IDs

    Returns:
        9x9 numpy array with values 0-9 (0 for invalid/missing)
    """
    if hasattr(seq, 'tolist'):
        seq = seq.tolist()

    # Find SEP position
    if SEP not in seq:
        return np.zeros((9, 9), dtype=np.int64)

    sep_pos = seq.index(SEP)

    # Solution starts after SEP, ends at EOS or end
    solution_start = sep_pos + 1
    if EOS in seq[solution_start:]:
        solution_end = seq.index(EOS, solution_start)
    else:
        solution_end = len(seq)

    solution_tokens = seq[solution_start:solution_end]

    # Convert to grid
    grid = np.zeros((9, 9), dtype=np.int64)
    for idx, t in enumerate(solution_tokens[:81]):
        if DIGIT_OFFSET + 1 <= t <= DIGIT_OFFSET + 9:
            row, col = idx // 9, idx % 9
            grid[row, col] = t - DIGIT_OFFSET

    return grid


def evaluate_completions(predicted_ids, ground_truth_ids):
    """
    Evaluate sudoku completions.

    Args:
        predicted_ids: numpy array of shape (n_samples, seq_len)
        ground_truth_ids: numpy array of shape (n_samples, seq_len)

    Returns:
        dict with evaluation metrics
    """
    n_total = len(predicted_ids)
    n_valid = 0
    n_exact_match = 0

    for pred_seq, gt_seq in zip(predicted_ids, ground_truth_ids):
        pred_grid = extract_solution_from_sequence(pred_seq)
        gt_grid = extract_solution_from_sequence(gt_seq)

        # Check validity
        if is_valid_sudoku(pred_grid):
            n_valid += 1

        # Check exact match
        if np.array_equal(pred_grid, gt_grid):
            n_exact_match += 1

    results = {
        'puzzle_accuracy': n_valid / n_total if n_total > 0 else 0.0,
        'exact_match_accuracy': n_exact_match / n_total if n_total > 0 else 0.0,
        'n_valid': n_valid,
        'n_exact_match': n_exact_match,
        'n_total': n_total,
    }

    print()
    print("=" * 50)
    print("DVAR-SUDOKU EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total puzzles:      {n_total}")
    print(f"Valid solutions:    {n_valid} ({results['puzzle_accuracy']*100:.1f}%)")
    print(f"Exact matches:      {n_exact_match} ({results['exact_match_accuracy']*100:.1f}%)")
    print("=" * 50)

    return results
