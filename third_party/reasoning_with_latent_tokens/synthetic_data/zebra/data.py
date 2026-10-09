"""
Zebra Puzzle dataset loader.

Loads pre-generated zebra puzzles from .pkl files.

Data format (per example, list of 3 elements):
  1. Clues: list of strings in symbolic format
     Format: "[Clue type] LHS [c/n] [attr/house] [val] RHS [c/n] [attr/house] [val] CLUE_END"
     - 'c' means attribute comparison, 'n' means house number
     
  2. Solution box: 2D list
     - First row: house numbers [0, 1, 2, ...]
     - Subsequent rows: attribute values for each house
     
  3. Solve order: order in which solver fills cells

Clue types include various logical relationships between houses/attributes.
"""

import os
import pickle
import typing

import numpy as np
from tqdm import tqdm
import time
import transformers


def parse_zebra_example(example):
    """
    Parse a single zebra puzzle example.
    
    Args:
        example: list of [clues, solution_box, solve_order]
        
    Returns:
        dict with:
            - 'clues': list of clue strings
            - 'solution': 2D numpy array (attributes x houses)
            - 'n_houses': number of houses
            - 'n_attrs': number of attributes
            - 'solve_order': solving order
    """
    clues, solution_box, solve_order = example

    # remove ANSWER token and everything after
    ans_idx = clues.index("ANSWER")
    if ans_idx != -1:
        clues = clues[:ans_idx]
    
    # Solution box: first row is house numbers, rest are attributes
    solution = np.array(solution_box, dtype=np.int32)
    
    n_houses = len(solution_box[0])
    n_attrs = len(solution_box) - 1  # Exclude house number row
    
    return {
        'clues': clues,
        'solution': solution,
        'n_houses': n_houses,
        'n_attrs': n_attrs,
        'solve_order': solve_order,
    }


# Hardcoded vocabulary list - single source of truth for zebra puzzles
ZEBRA_VOCAB_TOKENS = [
    '<PAD>', '<BOS>', '<EOS>', '<SEP>', '0', '1', '2', '3', '4', '5',
    '!=', '=', 'ANSWER', 'CLUE_END', 'LHS', 'RHS', 'c', 'ends',
    'immediate-left', 'inbetween', 'left-of', 'n', 'nbr'
]


def build_vocab():
    """
    Build vocabulary mappings for zebra puzzles.
    
    Returns:
        (vocab_map, id_to_token) tuple
    """
    vocab_map = {token: i for i, token in enumerate(ZEBRA_VOCAB_TOKENS)}
    id_to_token = {i: token for i, token in enumerate(ZEBRA_VOCAB_TOKENS)}    
    return vocab_map, id_to_token


class ZebraTokenizer(transformers.PreTrainedTokenizer):
    """Tokenizer for zebra puzzle dataset.
    
    Uses the vocabulary defined in ZEBRA_VOCAB_TOKENS.
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
        return text.strip().split()

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, 0)  # Return PAD for unknown

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
            if skip_special_tokens and token in ['<PAD>', '<BOS>', '<EOS>', '<SEP>']:
                continue
            tokens.append(token)
        return ' '.join(tokens)
    
    def batch_decode(self, sequences, skip_special_tokens=False, **kwargs):
        """Decode a batch of token ID sequences."""
        return [self.decode(seq, skip_special_tokens=skip_special_tokens, **kwargs) 
                for seq in sequences]


def tokenize_clues(clues, vocab_map):
    """
    Convert clue strings/tokens to token IDs
    
    Args:
        clues: list of clue strings or flat list of tokens
        vocab_map: dict mapping token strings to IDs
        
    Returns:
        list of token IDs
    """
    tokens = []
    # Flatten clues into a single list of tokens
    all_tokens = []
    for clue in clues:
        if isinstance(clue, str):
            # If it's a string, split it
            parts = clue.strip().split()
            all_tokens.extend(parts)
        else:
            # If it's already a token, add it directly
            all_tokens.append(clue)
    
    # Convert to token IDs
    for token in all_tokens:
        if token in vocab_map:
            tokens.append(vocab_map[token])
        else:
            raise ValueError(f"Unknown token '{token}' not in vocabulary. "
                             f"This should not happen if vocab was built from data.")
    return tokens


def zebra_to_sequence(example, vocab_map, max_clue_tokens=None):
    """
    Convert a parsed zebra example to a token sequence and loss mask.
    
    Format: [BOS] clue_tokens house_indices [SEP] solution_tokens [EOS]
    
    The house_indices (first row of solution grid) are included before SEP to
    indicate the number of houses, but are not part of the solution (loss_mask=0).
    
    Args:
        example: raw example [clues, solution, solve_order]
        vocab_map: pre-built vocab map (required)
        max_clue_tokens: optional max tokens for clues
        
    Returns:
        dict with:
            - 'input_ids': numpy array of tokens
            - 'loss_mask': 1 for solution tokens (train on), 0 for problem tokens (ignore)
            - 'solution_start': index where solution tokens begin (after SEP)
    """
    parsed = parse_zebra_example(example)
    
    # Token IDs for special tokens
    BOS = vocab_map['<BOS>']
    EOS = vocab_map['<EOS>']
    SEP = vocab_map['<SEP>']
    
    # Tokenize clues
    clue_tokens = tokenize_clues(parsed['clues'], vocab_map)
    
    if max_clue_tokens and len(clue_tokens) > max_clue_tokens:
        clue_tokens = clue_tokens[:max_clue_tokens]
    
    # Extract house indices (first row of solution) - part of problem context, not solution
    house_indices = parsed['solution'][0].flatten().tolist()
    house_indices_tokens = []
    for val in house_indices:
        str_val = str(int(val))
        if str_val in vocab_map:
            house_indices_tokens.append(vocab_map[str_val])
        else:
            raise ValueError(f"House index value '{val}' not in vocabulary.")
    
    # Flatten solution (skip first row which is house indices)
    # Solution values are attribute values (0-indexed)
    solution_flat = parsed['solution'][1:].flatten().tolist()
    
    # Map solution values to token IDs (use number tokens)
    solution_tokens = []
    for val in solution_flat:
        str_val = str(int(val))
        if str_val in vocab_map:
            solution_tokens.append(vocab_map[str_val])
        else:
            raise ValueError(f"Solution value '{val}' not in vocabulary. "
                             f"This should not happen if vocab was built from data.")
    
    # Build sequence: [BOS] clues house_indices [SEP] solution [EOS]
    sequence = [BOS] + clue_tokens + house_indices_tokens + [SEP] + solution_tokens + [EOS]
    
    # Problem tokens: [BOS] + clue_tokens + house_indices + [SEP]
    # Solution starts after SEP
    solution_start = 1 + len(clue_tokens) + len(house_indices_tokens) + 1
    
    sequence = np.array(sequence, dtype=np.int64)
    
    # Create loss_mask: 0 for problem tokens (including house indices), 1 for solution tokens
    loss_mask = np.zeros(len(sequence), dtype=np.int64)
    loss_mask[solution_start:] = 1
    
    return {
        'input_ids': sequence,
        'loss_mask': loss_mask,
        'solution_start': solution_start,
    }


def generate_synthetic_data(dataset_size, seq_len, config=None):
    """
    Load zebra puzzle data from pre-generated .pkl files.
    
    Args:
        dataset_size: number of samples to load (None to use all available data)
        seq_len: sequence length (will pad/truncate as needed)
        config: dict with configuration. Key fields:
            - data_path: str, path to the .pkl data file (required)
            - seed: optional int for shuffling
            - max_clue_tokens: optional max tokens for clues section
            - cache_dir: optional str, directory to cache processed data
            
    Returns:
        dict with:
            - 'input_ids': numpy array of shape (dataset_size, seq_len)
            - 'loss_mask': numpy array of shape (dataset_size, seq_len) with 1 for
                           solution tokens (train on) and 0 for problem tokens (ignore)
        
    Note:
        vocab_size is determined automatically by scanning the dataset.
        The vocabulary is built deterministically from all unique tokens
        in the data, sorted to ensure consistent tokenization.
        If cache_dir is provided, processed data will be cached to disk for faster
        subsequent loads.
    """
    if config is None:
        config = {}
    
    data_path = config.get("data_path")
    if data_path is None:
        raise ValueError("data_path must be provided in config")
    
    max_clue_tokens = config.get("max_clue_tokens", None)
    
    # Build hardcoded vocabulary (needed for cache filename)
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
    
    # Load raw data to determine dataset_size if needed
    if not os.path.exists(data_path):
        # Fall back to a previously processed cache when the raw .pkl is
        # unavailable (raw puzzle files may have been cleaned up; the cache is
        # a deterministic transform of them).
        cache_dir = config.get("cache_dir", None)
        if cache_dir:
            import glob
            data_basename = os.path.basename(data_path).replace('.pkl', '')
            suffix = ""
            if max_clue_tokens is not None:
                suffix += f"_mct{max_clue_tokens}"
            if "seed" in config:
                suffix += f"_seed{config['seed']}"
            ds_pat = str(dataset_size) if dataset_size is not None else "*"
            pattern = os.path.join(
                cache_dir, f"zebra_{data_basename}_ds{ds_pat}_sl{seq_len}_vs{vocab_size}{suffix}.npz")
            matches = sorted(glob.glob(pattern))
            if matches:
                cached = np.load(matches[0], allow_pickle=True)
                print(f"[ZEBRA] Raw data missing; loading cached data from: {matches[0]}")
                return {
                    'input_ids': cached["input_ids"],
                    'loss_mask': cached["loss_mask"],
                }
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)
    
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
        data_basename = os.path.basename(data_path).replace('.pkl', '')
        fname = f"zebra_{data_basename}_ds{dataset_size}_sl{seq_len}_vs{vocab_size}"
        if max_clue_tokens is not None:
            fname += f"_mct{max_clue_tokens}"
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
                    print(f"[ZEBRA] Loading cached data from: {cache_path}")
                    return {
                        'input_ids': cached_input_ids,
                        'loss_mask': cached_loss_mask,
                    }
            except Exception:
                # any load/validation failure falls through to regeneration
                pass
    
    print(f"[ZEBRA] Processing data from: {data_path}")
    print(f"[ZEBRA] Loaded {len(raw_data)} examples, using {dataset_size}")
    
    # Optionally shuffle
    if "seed" in config:
        np.random.seed(int(config["seed"]))
        indices = np.random.permutation(len(raw_data))
    else:
        indices = np.arange(len(raw_data))
    
    indices = indices[:dataset_size]
    
    PAD = vocab_map['<PAD>']
    
    # Create dataset with PAD tokens for input_ids, 0 for loss_mask (don't train on padding)
    input_ids = np.full((dataset_size, seq_len), fill_value=PAD, dtype=np.int64)
    loss_mask = np.ones((dataset_size, seq_len), dtype=np.int64)
    
    # Track statistics
    truncated = 0
    max_observed_len = 0
    
    # Convert each example
    for i, idx in enumerate(tqdm(indices, desc="Processing zebra puzzles")):
        result = zebra_to_sequence(
            raw_data[idx], 
            vocab_map,
            max_clue_tokens
        )
        seq = result['input_ids']
        seq_loss_mask = result['loss_mask']
        
        max_observed_len = max(max_observed_len, len(seq))
        
        if len(seq) > seq_len:
            truncated += 1
            seq = seq[:seq_len]
            seq_loss_mask = seq_loss_mask[:seq_len]
            # Make sure last token is EOS
            seq[-1] = vocab_map['<EOS>']
            seq_loss_mask[-1] = 1  # Train on EOS
        
        input_ids[i, :len(seq)] = seq
        loss_mask[i, :len(seq_loss_mask)] = seq_loss_mask
    
    # Count solution tokens for reporting
    solution_tokens = np.sum(loss_mask)
    total_tokens = dataset_size * seq_len
    
    print(f"\n{'='*60}")
    print(f"[ZEBRA] Dataset created")
    print(f"{'='*60}")
    print(f"  Dataset size: {dataset_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Max observed sequence length: {max_observed_len}")
    print(f"  Truncated examples: {truncated} ({100*truncated/dataset_size:.1f}%)")
    print(f"  PAD token: {PAD}")
    print(f"  Solution tokens: {solution_tokens} / {total_tokens} ({100*solution_tokens/total_tokens:.1f}%)")
    print(f"  Sample input_ids (first 40): {input_ids[0, :40].tolist()}")
    print(f"  Sample loss_mask (first 40): {loss_mask[0, :40].tolist()}")
    print(f"{'='*60}\n")
    
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
            print(f"[ZEBRA] Saved cached data to: {cache_path}")
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

def get_vocab_maps():
    """
    Get the vocabulary mappings for zebra puzzles.
    
    Returns:
        (vocab_map, id_to_token) tuple
    """
    return build_vocab()


def extract_solution_tokens(token_ids, vocab_map=None):
    """
    Extract solution tokens from a generated sequence.
    
    The sequence format is: [BOS] clues [SEP] solution [EOS] [PAD...]
    This function extracts the solution tokens (after SEP, before EOS/PAD).
    
    Args:
        token_ids: numpy array or list of token IDs
        vocab_map: optional vocab map (will build if not provided)
        
    Returns:
        list of solution token IDs (just the numbers, not special tokens)
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
        return []  # No SEP found
    
    # Extract tokens after SEP until EOS or PAD
    solution_tokens = []
    for tid in token_ids[sep_idx + 1:]:
        if tid == EOS or tid == PAD:
            break
        solution_tokens.append(tid)
    
    return solution_tokens


def decode_solution_tokens(solution_tokens, vocab_map=None, id_to_token=None):
    """
    Convert solution token IDs to integer values.
    
    Args:
        solution_tokens: list of token IDs (should be number tokens: 0-5)
        vocab_map: optional vocab map
        id_to_token: optional id_to_token map
        
    Returns:
        list of integer values
    """
    if id_to_token is None:
        _, id_to_token = build_vocab()
    
    values = []
    for tid in solution_tokens:
        token_str = id_to_token.get(tid, '')
        # Check if it's a digit
        if token_str.isdigit():
            values.append(int(token_str))
        else:
            # Non-digit token in solution (shouldn't happen for valid outputs)
            values.append(-1)
    
    return values


def solution_to_grid(solution_values, n_houses, n_attrs):
    """
    Reshape flat solution values into a 2D grid.
    
    Args:
        solution_values: list of integer values (length should be n_houses * n_attrs)
        n_houses: number of houses
        n_attrs: number of attributes
        
    Returns:
        numpy array of shape (n_attrs, n_houses) or None if wrong length
    """
    expected_len = n_houses * n_attrs
    if len(solution_values) != expected_len:
        return None
    
    # Solution is stored as [attr0_house0, attr0_house1, ..., attr1_house0, ...]
    return np.array(solution_values, dtype=np.int32).reshape(n_attrs, n_houses)


def check_solution(predicted_grid, ground_truth_grid):
    """
    Compare predicted solution grid against ground truth.
    
    Args:
        predicted_grid: numpy array of shape (n_attrs, n_houses), or None if malformed
        ground_truth_grid: numpy array of shape (n_attrs, n_houses)
        
    Returns:
        dict with:
            - 'cell_accuracy': fraction of cells correct
            - 'row_accuracy': fraction of rows (attributes) fully correct
            - 'puzzle_correct': bool, whether entire puzzle is correct
            - 'n_correct_cells': number of correct cells
            - 'n_total_cells': total number of cells
            - 'error_type': one of 'none', 'malformed', 'wrong_shape', 'incorrect'
    """
    if predicted_grid is None or ground_truth_grid is None:
        return {
            'cell_accuracy': 0.0,
            'row_accuracy': 0.0,
            'puzzle_correct': False,
            'n_correct_cells': 0,
            'n_total_cells': ground_truth_grid.size if ground_truth_grid is not None else 0,
            'error_type': 'malformed',
        }
    
    if predicted_grid.shape != ground_truth_grid.shape:
        return {
            'cell_accuracy': 0.0,
            'row_accuracy': 0.0,
            'puzzle_correct': False,
            'n_correct_cells': 0,
            'n_total_cells': ground_truth_grid.size,
            'error_type': 'wrong_shape',
        }
    
    cell_correct = (predicted_grid == ground_truth_grid)
    n_correct_cells = np.sum(cell_correct)
    n_total_cells = ground_truth_grid.size
    
    # Row accuracy: each row must be entirely correct
    row_correct = np.all(cell_correct, axis=1)
    n_correct_rows = np.sum(row_correct)
    n_total_rows = ground_truth_grid.shape[0]
    
    puzzle_correct = np.all(cell_correct)
    
    return {
        'cell_accuracy': n_correct_cells / n_total_cells,
        'row_accuracy': n_correct_rows / n_total_rows,
        'puzzle_correct': bool(puzzle_correct),
        'n_correct_cells': int(n_correct_cells),
        'n_total_cells': int(n_total_cells),
        'error_type': 'none' if puzzle_correct else 'incorrect',
    }


def extract_house_indices(token_ids, vocab_map=None):
    """
    Extract house indices from a sequence to determine n_houses.
    
    The sequence format is: [BOS] clues house_indices [SEP] solution [EOS]
    House indices are the digit tokens immediately before SEP (e.g., "0 1 2 3" for 4 houses).
    
    Args:
        token_ids: numpy array or list of token IDs
        vocab_map: optional vocab map (will build if not provided)
        
    Returns:
        list of house index integers, or empty list if cannot be determined
    """
    if vocab_map is None:
        vocab_map, _ = build_vocab()
    
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()
    
    SEP = vocab_map['<SEP>']
    CLUE_END = vocab_map['CLUE_END']
    
    # Find SEP position
    try:
        sep_idx = token_ids.index(SEP)
    except ValueError:
        return []
    
    # Find the last CLUE_END before SEP
    last_clue_end_idx = -1
    for i in range(sep_idx - 1, -1, -1):
        if token_ids[i] == CLUE_END:
            last_clue_end_idx = i
            break
    
    if last_clue_end_idx == -1:
        return []
    
    # Extract tokens between last CLUE_END and SEP - these are house indices
    house_tokens = token_ids[last_clue_end_idx + 1:sep_idx]
    
    # Decode to integers
    _, id_to_token = build_vocab()
    house_indices = []
    for tid in house_tokens:
        token_str = id_to_token.get(tid, '')
        if token_str.isdigit():
            house_indices.append(int(token_str))
    
    return house_indices


def _compute_chance_performance(gt_grids, n_trials=10):
    """
    Estimate chance performance by randomly shuffling each row of ground truth grids.
    
    For each puzzle, shuffle each attribute row independently and compare to original.
    This simulates a random guess that produces a valid grid structure but random assignments.
    
    Args:
        gt_grids: list of ground truth grids (numpy arrays of shape (n_attrs, n_houses))
        n_trials: number of random shuffles per puzzle to average over
        
    Returns:
        dict with chance_cell_accuracy, chance_row_accuracy, chance_puzzle_accuracy
    """
    if not gt_grids:
        return {'chance_cell_accuracy': 0.0, 'chance_row_accuracy': 0.0, 'chance_puzzle_accuracy': 0.0}
    
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    total_cell_acc = 0.0
    total_row_acc = 0.0
    total_puzzle_correct = 0
    total_trials = 0
    
    for gt_grid in gt_grids:
        if gt_grid is None:
            continue
        n_attrs, n_houses = gt_grid.shape
        
        for _ in range(n_trials):
            # Create shuffled version: shuffle each row independently
            shuffled = np.empty_like(gt_grid)
            for row_idx in range(n_attrs):
                shuffled[row_idx] = rng.permutation(gt_grid[row_idx])
            
            # Compare shuffled to original
            cell_correct = (shuffled == gt_grid)
            cell_acc = np.mean(cell_correct)
            row_acc = np.mean(np.all(cell_correct, axis=1))
            puzzle_correct = np.all(cell_correct)
            
            total_cell_acc += cell_acc
            total_row_acc += row_acc
            total_puzzle_correct += int(puzzle_correct)
            total_trials += 1
    
    return {
        'chance_cell_accuracy': total_cell_acc / total_trials if total_trials > 0 else 0.0,
        'chance_row_accuracy': total_row_acc / total_trials if total_trials > 0 else 0.0,
        'chance_puzzle_accuracy': total_puzzle_correct / total_trials if total_trials > 0 else 0.0,
    }


def evaluate_completions(predicted_ids, ground_truth_ids):
    """
    Evaluate puzzle completions by comparing predicted vs ground truth token sequences.
    
    This is the main entry point for evaluating model outputs on zebra puzzles.
    It compares the solution portion of predicted sequences against ground truth.
    Each puzzle can have a different number of houses - dimensions are inferred per-puzzle.
    
    Args:
        predicted_ids: numpy array of shape (n_samples, seq_len) - model outputs
        ground_truth_ids: numpy array of shape (n_samples, seq_len) - ground truth sequences
        
    Returns:
        dict with:
            - 'mean_cell_accuracy': average cell accuracy across samples
            - 'mean_row_accuracy': average row accuracy across samples  
            - 'puzzle_accuracy': fraction of puzzles fully correct
            - 'n_correct_puzzles': number of fully correct puzzles
            - 'n_total_puzzles': total number of puzzles evaluated
            - 'n_malformed': count of malformed solutions (wrong token count)
            - 'n_wrong_shape': count of wrong shape predictions
            - 'n_incorrect': count of valid but incorrect solutions
            - 'malformed_rate': fraction of malformed solutions
            - 'wrong_shape_rate': fraction of wrong shape predictions
            - 'incorrect_rate': fraction of incorrect solutions
            - 'chance_cell_accuracy': expected cell accuracy from random guessing
            - 'chance_row_accuracy': expected row accuracy from random guessing
            - 'chance_puzzle_accuracy': expected puzzle accuracy from random guessing
    """
    vocab_map, id_to_token = build_vocab()
    
    results = []
    gt_grids = []  # Collect ground truth grids for chance computation
    
    for pred_seq, gt_seq in zip(predicted_ids, ground_truth_ids):
        # Infer n_houses from house indices in ground truth (tokens before SEP)
        house_indices = extract_house_indices(gt_seq, vocab_map)
        n_houses = len(house_indices) if house_indices else None
        
        # Extract solution tokens (after SEP, before EOS/PAD)
        gt_solution = extract_solution_tokens(gt_seq, vocab_map)
        pred_solution = extract_solution_tokens(pred_seq, vocab_map)
        
        # Infer n_attrs from solution length and n_houses
        if n_houses and len(gt_solution) % n_houses == 0:
            n_attrs = len(gt_solution) // n_houses
        else:
            # Fallback: try to infer from solution length alone
            n_houses = n_houses or 4
            n_attrs = len(gt_solution) // n_houses if n_houses else 4
        
        # Decode to integer values
        pred_values = decode_solution_tokens(pred_solution, vocab_map, id_to_token)
        gt_values = decode_solution_tokens(gt_solution, vocab_map, id_to_token)
        
        # Convert to grids
        pred_grid = solution_to_grid(pred_values, n_houses, n_attrs)
        gt_grid = solution_to_grid(gt_values, n_houses, n_attrs)
        
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
    total_puzzle_correct = sum(1 for r in results if r['puzzle_correct'])
    
    # Count error types
    n_malformed = sum(1 for r in results if r['error_type'] == 'malformed')
    n_wrong_shape = sum(1 for r in results if r['error_type'] == 'wrong_shape')
    n_incorrect = sum(1 for r in results if r['error_type'] == 'incorrect')
    
    # Compute chance performance
    chance_metrics = _compute_chance_performance(gt_grids)
    
    metrics = {
        'mean_cell_accuracy': total_cell_acc / n if n > 0 else 0.0,
        'mean_row_accuracy': total_row_acc / n if n > 0 else 0.0,
        'puzzle_accuracy': total_puzzle_correct / n if n > 0 else 0.0,
        'n_correct_puzzles': total_puzzle_correct,
        'n_total_puzzles': n,
        'n_malformed': n_malformed,
        'n_wrong_shape': n_wrong_shape,
        'n_incorrect': n_incorrect,
        'malformed_rate': n_malformed / n if n > 0 else 0.0,
        'wrong_shape_rate': n_wrong_shape / n if n > 0 else 0.0,
        'incorrect_rate': n_incorrect / n if n > 0 else 0.0,
        **chance_metrics,
    }
    
    # Print evaluation results
    print("\n" + "="*60)
    print("Puzzle Evaluation Results")
    print("="*60)
    print(f"  Puzzles evaluated: {metrics['n_total_puzzles']}")
    print(f"  Puzzles correct: {metrics['n_correct_puzzles']}")
    print(f"  Puzzle accuracy: {metrics['puzzle_accuracy']:.4f}")
    print(f"  Cell accuracy: {metrics['mean_cell_accuracy']:.4f}")
    print(f"  Row accuracy: {metrics['mean_row_accuracy']:.4f}")
    print("-" * 40)
    print("  Error breakdown:")
    print(f"    Malformed (wrong token count): {metrics['n_malformed']} ({metrics['malformed_rate']:.4f})")
    print(f"    Wrong shape: {metrics['n_wrong_shape']} ({metrics['wrong_shape_rate']:.4f})")
    print(f"    Incorrect (valid but wrong): {metrics['n_incorrect']} ({metrics['incorrect_rate']:.4f})")
    print("-" * 40)
    print("  Chance baseline (random row shuffle):")
    print(f"    Cell accuracy: {metrics['chance_cell_accuracy']:.4f}")
    print(f"    Row accuracy: {metrics['chance_row_accuracy']:.4f}")
    print(f"    Puzzle accuracy: {metrics['chance_puzzle_accuracy']:.4f}")
    print("="*60 + "\n")
    
    return metrics
