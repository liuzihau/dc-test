import numpy as np
import random
from tqdm import tqdm
import os
import time

from synthetic_data.sudoku.sudoku9 import make_puzzle, generate_full_solution, board_to_tokens_flat, flat_tokens_to_board


def generate_sudoku_boards(dataset_size, seq_len, config=None):
    """
    Generates sequences representing valid 9x9 Sudoku boards.
    Each sequence is: [BOS] + 81 tokens (digits 1-9 mapped to 0-8) + [EOS] + padding.

    Args:
        dataset_size: number of Sudoku boards.
        seq_len: must be >= 83 (81 cells + BOS + EOS), will be padded if longer.
        config: dict with configuration. Must include:
            - vocab_size: int, must be >= 11 (0..8 for digits, BOS=vocab_size-2, EOS=vocab_size-1).
            - seed: optional int
    """
    if config is None:
        config = {}
    
    vocab_size = config.get("vocab_size")
    if vocab_size is None:
        raise ValueError("vocab_size must be provided in config")
    
    # --- checks ---
    min_seq_len = 81 + 2
    if seq_len < min_seq_len:
        raise ValueError(f"seq_len must be >= {min_seq_len} (got {seq_len}).")

    if vocab_size < 11:
        raise ValueError(f"vocab_size must be >= 11 (got {vocab_size}).")
    if config and "seed" in config:
        np.random.seed(int(config["seed"]))

    use_solver = config.get("use_solver", False)
    random_mask_ratio = config.get("random_mask_ratio", 0.0)
    search_mask = config.get("search_mask", False)
    search_mask_target_givens = config.get("search_mask_target_givens", 30)
    num_solutions = config.get("search_num_solutions", 1)

    BOS = vocab_size - 2
    EOS = vocab_size - 1
    MASK = vocab_size  # <-- blanks use this

    # ---- caching setup ----
    cache_dir = config.get("cache_dir", None)
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        fname = f"sudoku_boards_ds{dataset_size}_sl{seq_len}_vs{vocab_size}"
        fname += f"_solver{use_solver}"
        if random_mask_ratio > 0.0:
            fname += f"_random_mask{random_mask_ratio}"
        if search_mask:
            fname += f"_search_mask{search_mask}_target_givens{search_mask_target_givens}_num_solutions{num_solutions}"
        fname += ".npz"
        cache_path = os.path.join(cache_dir, fname)

        # Try to load a matching cache
        if os.path.exists(cache_path):
            try:
                cached = np.load(cache_path, allow_pickle=True)
                # be flexible about key name: support either 'dataset' or 'puzzles'
                arr = cached["dataset"] 
                # Validate shape & content assumptions
                if arr.shape == (dataset_size, seq_len):
                    return arr
            except Exception:
                # any load/validation failure falls through to regeneration
                pass

    # Create dataset with padding (zeros by default)
    dataset = np.full((dataset_size, seq_len), fill_value=EOS, dtype=int)
    dataset[:, 0] = BOS
    dataset[:, min_seq_len - 1] = EOS  # Position 82 (0-indexed)

    # --- helpers for valid Sudoku generation ---
    base = 3  # size of subgrid
    side = base * base  # 9

    def pattern(r, c):
        # base valid Latin/Sudoku pattern
        return (base * (r % base) + r // base + c) % side

    def random_board():
        # randomize numbers 0..8 (representing digits 1..9)
        nums = np.random.permutation(side)
        # shuffle rows: within bands, and bands themselves
        rows = np.r_[np.random.permutation(base)[:, None] * base + np.random.permutation(base)].reshape(-1)
        # shuffle cols: within stacks, and stacks themselves
        cols = np.r_[np.random.permutation(base)[:, None] * base + np.random.permutation(base)].reshape(-1)
        board = nums[[pattern(r, c) for r in rows for c in cols]]
        # optional transpose for extra variety
        if np.random.rand() < 0.5:
            board = board.reshape(side, side).T.reshape(-1)
        return board  # values in 0..8

    # --- fill dataset ---
    for i in tqdm(range(dataset_size)):

        if use_solver:
            # shift digits to 0-8
            flat_board = board_to_tokens_flat(generate_full_solution(), MASK)
        else:
            flat_board = random_board()

        if random_mask_ratio > 0.0:
            # mask out some of the board
            mask = np.random.rand(81) < random_mask_ratio
            flat_board[mask] = MASK

        if search_mask:
            initial_solution = flat_tokens_to_board(flat_board)
            pz = make_puzzle(
                target_givens=search_mask_target_givens,
                initial_solution=initial_solution,
                num_solutions=num_solutions,
            )
            flat_board = board_to_tokens_flat(pz.puzzle, MASK)  # length 81 core

        dataset[i, 1:82] = flat_board  # Fill positions 1-81

    if cache_path:
        try:
            np.savez_compressed(
                cache_path,
                dataset=dataset,
                # metadata for sanity/debug (not required to load)
                dataset_size=np.array(dataset_size, dtype=np.int64),
                seq_len=np.array(seq_len, dtype=np.int64),
                vocab_size=np.array(vocab_size, dtype=np.int64),
                created_ts=np.array(time.time()),
            )
        except Exception:
            # If caching fails, just return the data; generation already succeeded.
            pass

    return dataset


def generate_sudoku_any_base(dataset_size,
                              base=3,
                              seq_len=None,
                              config=None,
                              rectangular_blocks=None):
    if config is None:
        config = {}
    
    if "seed" in config:
        np.random.seed(int(config["seed"]))

    side = base * base
    min_seq_len = side * side + 2
    if seq_len is None:
        seq_len = min_seq_len
    if seq_len < min_seq_len:
        raise ValueError(f"seq_len must be >= {min_seq_len} (got {seq_len}).")

    vocab_size = config.get("vocab_size")
    if vocab_size is None:
        vocab_size = side + 2  # Default if not provided
    if vocab_size < side + 2:
        raise ValueError(f"vocab_size must be >= side+2={side+2} (got {vocab_size}).")

    BOS = vocab_size - 2
    EOS = vocab_size - 1

    cache_dir = config.get("cache_dir", None)
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        fname = f"sudoku_boards_base{base}_ds{dataset_size}_sl{seq_len}_vs{vocab_size}"
        fname += ".npz"
        cache_path = os.path.join(cache_dir, fname)

        # Try to load a matching cache
        if os.path.exists(cache_path):
            try:
                cached = np.load(cache_path, allow_pickle=True)
                # be flexible about key name: support either 'dataset' or 'puzzles'
                arr = cached["dataset"] 
                # Validate shape & content assumptions
                if arr.shape == (dataset_size, seq_len):
                    return arr
            except Exception:
                # any load/validation failure falls through to regeneration
                pass

    dataset = np.full((dataset_size, seq_len), fill_value=EOS, dtype=np.int64)
    dataset[:, 0] = BOS
    dataset[:, min_seq_len - 1] = EOS

    if rectangular_blocks is None:
        a = b = base
    else:
        a, b = rectangular_blocks
        if a * b != side:
            raise ValueError(f"a*b must equal side ({side}), got {a*b}.")

    def pattern(r, c):
        return (a * (r % a) + r // a + c) % side

    def random_board():
        nums = np.random.permutation(side)
        band_order = np.random.permutation(side // a)
        within_band = [np.random.permutation(a) + k * a for k in band_order]
        rows = np.concatenate(within_band)

        stack_order = np.random.permutation(side // b)
        within_stack = [np.random.permutation(b) + k * b for k in stack_order]
        cols = np.concatenate(within_stack)

        board = nums[[pattern(r, c) for r in rows for c in cols]]
        if np.random.rand() < 0.5:
            board = board.reshape(side, side).T.reshape(-1)
        return board

    for i in tqdm(range(dataset_size)):
        flat = random_board()
        dataset[i, 1 : 1 + side * side] = flat

    if cache_path:
        try:
            np.savez_compressed(
                cache_path,
                dataset=dataset,
                # metadata for sanity/debug (not required to load)
                dataset_size=np.array(dataset_size, dtype=np.int64),
                seq_len=np.array(seq_len, dtype=np.int64),
                vocab_size=np.array(vocab_size, dtype=np.int64),
                base=np.array(base, dtype=np.int64),
                created_ts=np.array(time.time()),
            )
        except Exception:
            # If caching fails, just return the data; generation already succeeded.
            pass

    return dataset


def generate_synthetic_data(dataset_size, seq_len, config=None):
    """
    Main entry point for sudoku data generation.
    
    Args:
        dataset_size: number of samples to generate
        seq_len: sequence length
        config: dict with configuration. Key fields:
            - vocab_size: int, vocabulary size (required)
            - base: int, sudoku base (default 3 for 9x9)
            - seed: optional int
            - use_solver: bool
            - random_mask_ratio: float
            - search_mask: bool
            - search_mask_target_givens: int
            - search_num_solutions: int
    """
    if config is None:
        config = {}
    
    if "vocab_size" not in config:
        raise ValueError("vocab_size must be provided in config")
    
    base = config.get("base", 3)
    if base == 3:
        return generate_sudoku_boards(dataset_size, seq_len, config)
    else:
        print(f"Generating Sudoku boards for base {base}...")
        print("Ignoring all other data_config parameters.")
        return generate_sudoku_any_base(dataset_size, base, seq_len, config)

