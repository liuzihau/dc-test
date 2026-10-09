"""
Sudoku Puzzle dataset loader.

Loads pre-generated sudoku puzzles from .npy files.

Data format (per example, 325 entries):
  - First value: Number of filled (given) cells
  - Rest 324 values: 4 values per cell (81 cells x 4)
    Format: (row, column, correct_value, strategy_id)
    First n_given cells are the puzzle givens, rest are cells to solve.

Strategy IDs:
  0: given (cell is provided in puzzle)
  2: Lone single
  3: Hidden single
  4: Naked pair
  5: Naked triplet
  6: Locked candidate
  7: XY Wing
  8: Unique rectangle
"""

import os
import typing

import numpy as np
from tqdm import tqdm
import time
import transformers


# Hardcoded vocabulary list - single source of truth for sudoku puzzles
# Token indices:
#   0: PAD (padding token)
#   1-9: digits 1-9 (digit value)
#   10: MASK (empty cell in puzzle)
#   11: SEP (separator between puzzle and solution)
#   12: BOS (beginning of sequence)
#   13: EOS (end of sequence)
SUDOKU_PUZZLE_VOCAB_TOKENS = [
    '<PAD>',   # 0: padding
    '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 1-9: digits
    '<M>',     # 10: empty cell in puzzle
    '<SEP>',   # 11: separator
    '<BOS>',   # 12: beginning of sequence
    '<EOS>',   # 13: end of sequence
]


def build_vocab():
    """
    Build vocabulary mappings for sudoku puzzles.
    
    Returns:
        (vocab_map, id_to_token) tuple
    """
    vocab_map = {token: i for i, token in enumerate(SUDOKU_PUZZLE_VOCAB_TOKENS)}
    id_to_token = {i: token for i, token in enumerate(SUDOKU_PUZZLE_VOCAB_TOKENS)}
    return vocab_map, id_to_token


class SudokuPuzzleTokenizer(transformers.PreTrainedTokenizer):
    """Tokenizer for sudoku puzzle dataset.
    
    Uses the vocabulary defined in SUDOKU_PUZZLE_VOCAB_TOKENS.
    """
    
    def __init__(
        self,
        bos_token='<BOS>',
        eos_token='<EOS>',
        sep_token='<SEP>',
        pad_token='<PAD>',
        **kwargs):
        
        # Use build_vocab() as single source of truth
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
        # Split on whitespace, treating each token as a separate unit
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


def get_vocab_maps():
    """
    Get the vocabulary mappings for sudoku puzzles.
    
    Returns:
        (vocab_map, id_to_token) tuple
    """
    return build_vocab()


def parse_sudoku_example(example):
    """
    Parse a single sudoku example from the raw format.
    
    Args:
        example: numpy array of 325 values
        
    Returns:
        dict with:
            - 'puzzle': 9x9 grid with 0 for empty cells, 1-9 for givens
            - 'solution': 9x9 grid with complete solution
            - 'n_given': number of given cells
            - 'strategies': 9x9 grid with strategy IDs for each cell
            - 'solve_order': list of (row, col) in solving order
    """
    n_given = int(example[0])
    
    # Parse the 324 values (4 per cell)
    cell_data = example[1:].reshape(81, 4)
    
    puzzle = np.zeros((9, 9), dtype=np.int32)
    solution = np.zeros((9, 9), dtype=np.int32)
    strategies = np.zeros((9, 9), dtype=np.int32)
    solve_order = []
    
    for i in range(81):
        row, col, value, strategy = cell_data[i]
        row, col, value, strategy = int(row), int(col), int(value), int(strategy)
        
        solution[row, col] = value
        strategies[row, col] = strategy
        
        if i < n_given:
            # This is a given cell
            puzzle[row, col] = value
        else:
            # This is a cell to solve
            solve_order.append((row, col))
    
    return {
        'puzzle': puzzle,
        'solution': solution,
        'n_given': n_given,
        'strategies': strategies,
        'solve_order': solve_order,
    }


def sudoku_to_sequence(example, vocab_map=None, include_puzzle=True):
    """
    Convert a parsed sudoku example to a token sequence and loss mask.
    
    Format: [BOS] puzzle(81 tokens) [SEP] solution(81 tokens) [EOS]
    
    If include_puzzle=False: [BOS] solution(81 tokens) [EOS]
    
    Args:
        example: raw sudoku example array
        vocab_map: pre-built vocab map (will build if not provided)
        include_puzzle: whether to include the puzzle before the solution
        
    Returns:
        dict with:
            - 'input_ids': token sequence
            - 'loss_mask': 1 for solution tokens (train on), 0 for problem tokens (ignore)
            - 'solution_start': index where solution tokens begin (after SEP)
    """
    if vocab_map is None:
        vocab_map, _ = build_vocab()
    
    parsed = parse_sudoku_example(example)
    
    # Get special token IDs from vocab_map
    BOS = vocab_map['<BOS>']
    EOS = vocab_map['<EOS>']
    SEP = vocab_map['<SEP>']
    MASK = vocab_map['<M>']
    
    # Convert solution to tokens (1-9 -> token IDs 1-9)
    # Solution values are 1-9, token IDs for digits are also 1-9
    solution_flat = parsed['solution'].flatten()
    
    if include_puzzle:
        # Convert puzzle to tokens (0 -> MASK, 1-9 -> 1-9)
        puzzle_flat = np.where(
            parsed['puzzle'] == 0,
            MASK,
            parsed['puzzle']
        ).flatten()
        # Format: [BOS] puzzle [SEP] solution [EOS]
        sequence = np.concatenate([
            [BOS],
            puzzle_flat,
            [SEP],
            solution_flat,
            [EOS]
        ])
        # Problem tokens: [BOS] + 81 puzzle + [SEP] = 83 tokens
        # Solution starts at index 83
        solution_start = 83
    else:
        # Format: [BOS] solution [EOS]
        sequence = np.concatenate([
            [BOS],
            solution_flat,
            [EOS]
        ])
        # Solution starts at index 1 (after BOS)
        solution_start = 1
    
    sequence = sequence.astype(np.int64)
    
    # Create loss_mask: 0 for problem tokens, 1 for solution tokens
    loss_mask = np.zeros(len(sequence), dtype=np.int64)
    loss_mask[solution_start:] = 1
    
    return {
        'input_ids': sequence,
        'loss_mask': loss_mask,
        'solution_start': solution_start,
    }


def decode_sudoku_sequence(sequence, id_to_token=None):
    """
    Decode a token sequence back to readable format.
    
    Args:
        sequence: numpy array or list of token indices
        id_to_token: optional id_to_token map (will build if not provided)
        
    Returns:
        list of decoded token strings
    """
    if id_to_token is None:
        _, id_to_token = build_vocab()
    
    decoded = []
    for token in sequence:
        token = int(token)
        if token in id_to_token:
            token_str = id_to_token[token]
            # Use '.' for MASK for readability
            if token_str == '<M>':
                decoded.append('.')
            else:
                decoded.append(token_str)
        else:
            decoded.append(f"?{token}")
    
    return decoded


def generate_synthetic_data(dataset_size, seq_len, config=None):
    """
    Load sudoku puzzle data from pre-generated .npy files.
    
    Args:
        dataset_size: number of samples to load (None to use all available data)
        seq_len: sequence length (will pad/truncate as needed)
        config: dict with configuration. Key fields:
            - data_path: str, path to the .npy data file (required)
            - include_puzzle: bool, whether to include puzzle before solution
                              (default: True)
            - seed: optional int for shuffling
            - cache_dir: optional str, directory to cache processed data
            
    Returns:
        dict with:
            - 'input_ids': numpy array of shape (dataset_size, seq_len)
            - 'loss_mask': numpy array of shape (dataset_size, seq_len) with 1 for
                           solution tokens (train on) and 0 for problem tokens (ignore)
    
    Note:
        vocab_size is determined automatically from the hardcoded vocabulary.
        If cache_dir is provided, processed data will be cached to disk for faster
        subsequent loads.
    """
    if config is None:
        config = {}
    
    # Build vocabulary using single source of truth
    vocab_map, id_to_token = build_vocab()
    built_vocab_size = len(vocab_map)
    
    # Check if vocab_size is provided in config and assert it matches
    config_vocab_size = config.get("vocab_size")
    if config_vocab_size is not None:
        assert config_vocab_size == built_vocab_size, (
            f"Config vocab_size ({config_vocab_size}) does not match "
            f"hardcoded vocab_size ({built_vocab_size}). Please update the config."
        )
    vocab_size = built_vocab_size
    
    data_path = config.get("data_path")
    if data_path is None:
        raise ValueError("data_path must be provided in config")
    
    include_puzzle = config.get("include_puzzle", True)
    
    # Calculate expected sequence length
    if include_puzzle:
        # [BOS] + 81 puzzle + [SEP] + 81 solution + [EOS] = 165
        min_seq_len = 165
    else:
        # [BOS] + 81 solution + [EOS] = 83
        min_seq_len = 83
    
    if seq_len < min_seq_len:
        raise ValueError(f"seq_len must be >= {min_seq_len} (got {seq_len})")
    
    # Load raw data to determine dataset_size if needed
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    raw_data = np.load(data_path)
    
    # Use all available data if dataset_size is None
    if dataset_size is None:
        dataset_size = len(raw_data)
    
    # Limit dataset_size to available data
    if dataset_size > len(raw_data):
        dataset_size = len(raw_data)
    
    # ---- caching setup ----
    cache_dir = config.get("cache_dir", None)
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # Build cache filename based on relevant parameters
        # v2: uses separate PAD token (vs v1 which used EOS for padding)
        data_basename = os.path.basename(data_path).replace('.npy', '')
        fname = f"sudoku_puzzle_v2_{data_basename}_ds{dataset_size}_sl{seq_len}_vs{vocab_size}"
        fname += f"_puzzle{include_puzzle}"
        if "seed" in config:
            fname += f"_seed{config['seed']}"
        fname += ".npz"
        cache_path = os.path.join(cache_dir, fname)
        
        # Try to load a matching cache
        if os.path.exists(cache_path):
            try:
                cached = np.load(cache_path, allow_pickle=True)
                cached_input_ids = cached["input_ids"]
                cached_loss_mask = cached["loss_mask"]
                # Validate shape matches
                if cached_input_ids.shape == (dataset_size, seq_len) and cached_loss_mask.shape == (dataset_size, seq_len):
                    print(f"[SUDOKU-PUZZLE] Loading cached data from: {cache_path}")
                    return {
                        'input_ids': cached_input_ids,
                        'loss_mask': cached_loss_mask,
                    }
            except Exception:
                # any load/validation failure falls through to regeneration
                pass
    
    print(f"[SUDOKU-PUZZLE] Processing data from: {data_path}")
    print(f"[SUDOKU-PUZZLE] Loaded {len(raw_data)} examples, using {dataset_size}")
    
    # Optionally shuffle
    if "seed" in config:
        np.random.seed(int(config["seed"]))
        indices = np.random.permutation(len(raw_data))
    else:
        indices = np.arange(len(raw_data))
    
    indices = indices[:dataset_size]
    
    PAD = vocab_map['<PAD>']
    
    # Create dataset with PAD tokens for input_ids, 1 for loss_mask initially
    # (we set loss_mask to 0 for padding and problem tokens below)
    input_ids = np.full((dataset_size, seq_len), fill_value=PAD, dtype=np.int64)
    loss_mask = np.ones((dataset_size, seq_len), dtype=np.int64)
    
    # Track statistics
    solution_token_count = 0
    
    # Convert each example
    for i, idx in enumerate(tqdm(indices, desc="Processing sudoku puzzles")):
        result = sudoku_to_sequence(raw_data[idx], vocab_map, include_puzzle)
        seq = result['input_ids']
        seq_loss_mask = result['loss_mask']
        
        # Truncate if necessary (shouldn't happen normally)
        seq = seq[:seq_len]
        seq_loss_mask = seq_loss_mask[:seq_len]
        
        input_ids[i, :len(seq)] = seq
        loss_mask[i, :len(seq_loss_mask)] = seq_loss_mask
    
    # Save to cache if cache_dir is provided
    if cache_path:
        try:
            np.savez_compressed(
                cache_path,
                input_ids=input_ids,
                loss_mask=loss_mask,
                # metadata for sanity/debug (not required to load)
                dataset_size=np.array(dataset_size, dtype=np.int64),
                seq_len=np.array(seq_len, dtype=np.int64),
                vocab_size=np.array(vocab_size, dtype=np.int64),
                created_ts=np.array(time.time()),
            )
            print(f"[SUDOKU-PUZZLE] Saved cached data to: {cache_path}")
        except Exception:
            # If caching fails, just return the data; generation already succeeded.
            pass
    
    return {
        'input_ids': input_ids,
        'loss_mask': loss_mask,
    }


# ============================================================================
# Solution Parsing and Verification Utilities
# ============================================================================

def extract_solution_tokens(token_ids, vocab_map=None):
    """
    Extract solution tokens from a generated sequence.
    
    The sequence format is: [BOS] puzzle [SEP] solution [EOS] [PAD...]
    This function extracts the solution tokens (after SEP, before EOS/PAD).
    
    Args:
        token_ids: numpy array or list of token IDs
        vocab_map: optional vocab map (will build if not provided)
        
    Returns:
        list of solution token IDs (just the digit tokens)
    """
    if vocab_map is None:
        vocab_map, _ = build_vocab()
    
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()
    
    SEP = vocab_map['<SEP>']
    EOS = vocab_map['<EOS>']
    PAD = vocab_map['<PAD>']
    
    # Find SEP position
    try:
        sep_idx = token_ids.index(SEP)
    except ValueError:
        # No SEP found - might be solution-only format
        # Try to find BOS and extract from there
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
    """
    Convert solution token IDs to integer values (1-9).
    
    Args:
        solution_tokens: list of token IDs (should be digit tokens: 1-9)
        id_to_token: optional id_to_token map
        
    Returns:
        list of integer values (1-9), or -1 for invalid tokens
    """
    if id_to_token is None:
        _, id_to_token = build_vocab()
    
    values = []
    for tid in solution_tokens:
        token_str = id_to_token.get(tid, '')
        # Check if it's a digit (1-9)
        if token_str.isdigit():
            values.append(int(token_str))
        else:
            # Non-digit token in solution (shouldn't happen for valid outputs)
            values.append(-1)
    
    return values


def solution_to_grid(solution_values):
    """
    Reshape flat solution values into a 9x9 grid.
    
    Args:
        solution_values: list of integer values (length should be 81)
        
    Returns:
        numpy array of shape (9, 9) or None if wrong length
    """
    if len(solution_values) != 81:
        return None
    
    return np.array(solution_values, dtype=np.int32).reshape(9, 9)


def check_sudoku_valid(grid):
    """
    Check if a sudoku grid is valid (no repeated digits in rows/cols/boxes).
    
    Args:
        grid: numpy array of shape (9, 9) with values 1-9
        
    Returns:
        bool indicating if the grid is valid
    """
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


def check_solution(predicted_grid, ground_truth_grid):
    """
    Compare predicted solution grid against ground truth.
    
    Args:
        predicted_grid: numpy array of shape (9, 9), or None if malformed
        ground_truth_grid: numpy array of shape (9, 9)
        
    Returns:
        dict with:
            - 'cell_accuracy': fraction of cells correct
            - 'row_accuracy': fraction of rows fully correct
            - 'column_accuracy': fraction of columns fully correct
            - 'box_accuracy': fraction of 3x3 boxes fully correct
            - 'puzzle_correct': bool, whether entire puzzle is correct
            - 'n_correct_cells': number of correct cells
            - 'n_total_cells': total number of cells (81)
            - 'is_valid': whether the predicted grid is a valid sudoku
            - 'error_type': one of 'none', 'malformed', 'invalid', 'incorrect'
    """
    if predicted_grid is None:
        return {
            'cell_accuracy': 0.0,
            'row_accuracy': 0.0,
            'column_accuracy': 0.0,
            'box_accuracy': 0.0,
            'puzzle_correct': False,
            'n_correct_cells': 0,
            'n_total_cells': 81,
            'is_valid': False,
            'error_type': 'malformed',
        }
    
    cell_correct = (predicted_grid == ground_truth_grid)
    n_correct_cells = np.sum(cell_correct)
    puzzle_correct = np.all(cell_correct)
    is_valid = check_sudoku_valid(predicted_grid)
    
    # Row accuracy: each row must be entirely correct
    row_correct = np.all(cell_correct, axis=1)
    n_correct_rows = np.sum(row_correct)
    
    # Column accuracy: each column must be entirely correct
    col_correct = np.all(cell_correct, axis=0)
    n_correct_cols = np.sum(col_correct)
    
    # Box accuracy: each 3x3 box must be entirely correct
    n_correct_boxes = 0
    for box_row in range(3):
        for box_col in range(3):
            box = cell_correct[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3]
            if np.all(box):
                n_correct_boxes += 1
    
    if puzzle_correct:
        error_type = 'none'
    elif not is_valid:
        error_type = 'invalid'
    else:
        error_type = 'incorrect'
    
    return {
        'cell_accuracy': n_correct_cells / 81,
        'row_accuracy': n_correct_rows / 9,
        'column_accuracy': n_correct_cols / 9,
        'box_accuracy': n_correct_boxes / 9,
        'puzzle_correct': bool(puzzle_correct),
        'n_correct_cells': int(n_correct_cells),
        'n_total_cells': 81,
        'is_valid': is_valid,
        'error_type': error_type,
    }


def _compute_chance_performance(gt_grids, n_trials=10):
    """
    Estimate chance performance by randomly shuffling each row of ground truth grids.
    
    For each puzzle, shuffle each row independently and compare to original.
    This simulates a random guess that produces a valid row structure but random assignments.
    
    Args:
        gt_grids: list of ground truth grids (numpy arrays of shape (9, 9))
        n_trials: number of random shuffles per puzzle to average over
        
    Returns:
        dict with chance_cell_accuracy, chance_row_accuracy, chance_column_accuracy,
        chance_box_accuracy, chance_puzzle_accuracy
    """
    if not gt_grids:
        return {
            'chance_cell_accuracy': 0.0,
            'chance_row_accuracy': 0.0,
            'chance_column_accuracy': 0.0,
            'chance_box_accuracy': 0.0,
            'chance_puzzle_accuracy': 0.0,
        }
    
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    total_cell_acc = 0.0
    total_row_acc = 0.0
    total_col_acc = 0.0
    total_box_acc = 0.0
    total_puzzle_correct = 0
    total_trials = 0
    
    for gt_grid in gt_grids:
        if gt_grid is None:
            continue
        
        for _ in range(n_trials):
            # Create shuffled version: shuffle each row independently
            shuffled = np.empty_like(gt_grid)
            for row_idx in range(9):
                shuffled[row_idx] = rng.permutation(gt_grid[row_idx])
            
            # Compare shuffled to original
            cell_correct = (shuffled == gt_grid)
            cell_acc = np.mean(cell_correct)
            
            # Row accuracy
            row_acc = np.mean(np.all(cell_correct, axis=1))
            
            # Column accuracy
            col_acc = np.mean(np.all(cell_correct, axis=0))
            
            # Box accuracy
            box_correct = 0
            for box_row in range(3):
                for box_col in range(3):
                    box = cell_correct[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3]
                    if np.all(box):
                        box_correct += 1
            box_acc = box_correct / 9
            
            puzzle_correct = np.all(cell_correct)
            
            total_cell_acc += cell_acc
            total_row_acc += row_acc
            total_col_acc += col_acc
            total_box_acc += box_acc
            total_puzzle_correct += int(puzzle_correct)
            total_trials += 1
    
    return {
        'chance_cell_accuracy': total_cell_acc / total_trials if total_trials > 0 else 0.0,
        'chance_row_accuracy': total_row_acc / total_trials if total_trials > 0 else 0.0,
        'chance_column_accuracy': total_col_acc / total_trials if total_trials > 0 else 0.0,
        'chance_box_accuracy': total_box_acc / total_trials if total_trials > 0 else 0.0,
        'chance_puzzle_accuracy': total_puzzle_correct / total_trials if total_trials > 0 else 0.0,
    }


def evaluate_completions(predicted_ids, ground_truth_ids):
    """
    Evaluate puzzle completions by comparing predicted vs ground truth token sequences.
    
    This is the main entry point for evaluating model outputs on sudoku puzzles.
    It compares the solution portion of predicted sequences against ground truth.
    
    Args:
        predicted_ids: numpy array of shape (n_samples, seq_len) - model outputs
        ground_truth_ids: numpy array of shape (n_samples, seq_len) - ground truth sequences
        
    Returns:
        dict with:
            - 'mean_cell_accuracy': average cell accuracy across samples
            - 'mean_row_accuracy': average row accuracy across samples
            - 'mean_column_accuracy': average column accuracy across samples
            - 'mean_box_accuracy': average 3x3 box accuracy across samples
            - 'puzzle_accuracy': fraction of puzzles fully correct
            - 'n_correct_puzzles': number of fully correct puzzles
            - 'n_total_puzzles': total number of puzzles evaluated
            - 'n_malformed': count of malformed solutions (wrong token count)
            - 'n_invalid': count of invalid sudoku grids
            - 'n_incorrect': count of valid but incorrect solutions
            - 'valid_rate': fraction of valid sudoku solutions
            - 'chance_cell_accuracy': expected cell accuracy from random guessing
            - 'chance_row_accuracy': expected row accuracy from random guessing
            - 'chance_column_accuracy': expected column accuracy from random guessing
            - 'chance_box_accuracy': expected box accuracy from random guessing
            - 'chance_puzzle_accuracy': expected puzzle accuracy from random guessing
    """
    vocab_map, id_to_token = build_vocab()
    
    results = []
    gt_grids = []  # Collect ground truth grids for chance computation
    
    for pred_seq, gt_seq in zip(predicted_ids, ground_truth_ids):
        # Extract solution tokens (after SEP, before EOS)
        gt_solution = extract_solution_tokens(gt_seq, vocab_map)
        pred_solution = extract_solution_tokens(pred_seq, vocab_map)
        
        # Decode to integer values (1-9)
        pred_values = decode_solution_tokens(pred_solution, id_to_token)
        gt_values = decode_solution_tokens(gt_solution, id_to_token)
        
        # Convert to grids
        pred_grid = solution_to_grid(pred_values)
        gt_grid = solution_to_grid(gt_values)
        
        # Store gt_grid for chance computation
        if gt_grid is not None:
            gt_grids.append(gt_grid)
        
        # Compare
        result = check_solution(pred_grid, gt_grid)
        results.append(result)
    
    # Aggregate metrics
    n = len(results)
    total_cell_acc = sum(r['cell_accuracy'] for r in results)
    total_row_acc = sum(r['row_accuracy'] for r in results)
    total_col_acc = sum(r['column_accuracy'] for r in results)
    total_box_acc = sum(r['box_accuracy'] for r in results)
    total_puzzle_correct = sum(1 for r in results if r['puzzle_correct'])
    total_valid = sum(1 for r in results if r['is_valid'])
    
    # Count error types
    n_malformed = sum(1 for r in results if r['error_type'] == 'malformed')
    n_invalid = sum(1 for r in results if r['error_type'] == 'invalid')
    n_incorrect = sum(1 for r in results if r['error_type'] == 'incorrect')
    
    # Compute chance performance
    chance_metrics = _compute_chance_performance(gt_grids)
    
    metrics = {
        'mean_cell_accuracy': total_cell_acc / n if n > 0 else 0.0,
        'mean_row_accuracy': total_row_acc / n if n > 0 else 0.0,
        'mean_column_accuracy': total_col_acc / n if n > 0 else 0.0,
        'mean_box_accuracy': total_box_acc / n if n > 0 else 0.0,
        'puzzle_accuracy': total_puzzle_correct / n if n > 0 else 0.0,
        'n_correct_puzzles': total_puzzle_correct,
        'n_total_puzzles': n,
        'n_malformed': n_malformed,
        'n_invalid': n_invalid,
        'n_incorrect': n_incorrect,
        'valid_rate': total_valid / n if n > 0 else 0.0,
        'malformed_rate': n_malformed / n if n > 0 else 0.0,
        'invalid_rate': n_invalid / n if n > 0 else 0.0,
        'incorrect_rate': n_incorrect / n if n > 0 else 0.0,
        **chance_metrics,
    }
    
    # Print evaluation results
    print("\n" + "="*60)
    print("Sudoku Puzzle Evaluation Results")
    print("="*60)
    print(f"  Puzzles evaluated: {metrics['n_total_puzzles']}")
    print(f"  Puzzles correct: {metrics['n_correct_puzzles']}")
    print(f"  Puzzle accuracy: {metrics['puzzle_accuracy']:.4f}")
    print(f"  Cell accuracy: {metrics['mean_cell_accuracy']:.4f}")
    print(f"  Row accuracy: {metrics['mean_row_accuracy']:.4f}")
    print(f"  Column accuracy: {metrics['mean_column_accuracy']:.4f}")
    print(f"  Box accuracy: {metrics['mean_box_accuracy']:.4f}")
    print(f"  Valid sudoku rate: {metrics['valid_rate']:.4f}")
    print("-" * 40)
    print("  Error breakdown:")
    print(f"    Malformed (wrong token count): {metrics['n_malformed']} ({metrics['malformed_rate']:.4f})")
    print(f"    Invalid (breaks sudoku rules): {metrics['n_invalid']} ({metrics['invalid_rate']:.4f})")
    print(f"    Incorrect (valid but wrong): {metrics['n_incorrect']} ({metrics['incorrect_rate']:.4f})")
    print("-" * 40)
    print("  Chance baseline (random row shuffle):")
    print(f"    Cell accuracy: {metrics['chance_cell_accuracy']:.4f}")
    print(f"    Row accuracy: {metrics['chance_row_accuracy']:.4f}")
    print(f"    Column accuracy: {metrics['chance_column_accuracy']:.4f}")
    print(f"    Box accuracy: {metrics['chance_box_accuracy']:.4f}")
    print(f"    Puzzle accuracy: {metrics['chance_puzzle_accuracy']:.4f}")
    print("="*60 + "\n")
    
    return metrics
