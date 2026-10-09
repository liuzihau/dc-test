"""
Diffusion-vs-AR dataset loaders.

This module provides loaders for datasets from the diffusion-vs-ar repository:
- dvar-sudoku: 9x9 Sudoku puzzles
- dvar-cd3, dvar-cd4, dvar-cd5: Countdown arithmetic puzzles
- dvar-3sat5, dvar-3sat7, dvar-3sat9: 3-SAT satisfiability problems
- dvar-path: Graph path finding

All datasets follow the same interface:
- generate_synthetic_data(dataset_size, seq_len, config) -> dict with input_ids and loss_mask
- evaluate_completions(predicted_ids, ground_truth_ids) -> dict with metrics
"""

from .sudoku import (
    DvarSudokuTokenizer,
    VOCAB_SIZE as SUDOKU_VOCAB_SIZE,
)
from .countdown import (
    CountdownTokenizer,
    VOCAB_SIZE as COUNTDOWN_VOCAB_SIZE,
)
from .sat3 import (
    SAT3Tokenizer,
    VOCAB_SIZE as SAT3_VOCAB_SIZE,
)
from .path import (
    PathTokenizer,
    VOCAB_SIZE as PATH_VOCAB_SIZE,
)


def generate_synthetic_data(dataset_size, seq_len, config=None):
    """
    Dispatcher for diffusion-vs-ar dataset generation.

    Routes to the appropriate dataset loader based on config['name'].

    Args:
        dataset_size: number of samples to generate/load
        seq_len: sequence length
        config: dict with configuration. Must include 'name' field.

    Returns:
        dict with 'input_ids' and 'loss_mask' numpy arrays

    Supported dataset names:
        - 'dvar-sudoku': 9x9 Sudoku puzzles
        - 'dvar-cd3', 'dvar-cd4', 'dvar-cd5': Countdown puzzles
        - 'dvar-3sat5', 'dvar-3sat7', 'dvar-3sat9': 3-SAT problems
        - 'dvar-path': Path finding problems
    """
    if config is None:
        config = {}

    name = config.get('name', '')

    if name == 'dvar-sudoku':
        from .sudoku import generate_synthetic_data as gen
        return gen(dataset_size, seq_len, config)

    elif name.startswith('dvar-cd'):
        from .countdown import generate_synthetic_data as gen
        # Extract variant (cd3, cd4, cd5)
        variant = name.replace('dvar-', '')  # e.g., 'cd4'
        config_with_variant = {**config, 'variant': variant}
        return gen(dataset_size, seq_len, config_with_variant)

    elif name.startswith('dvar-3sat'):
        from .sat3 import generate_synthetic_data as gen
        # Extract variant (3sat5, 3sat7, 3sat9)
        variant = name.replace('dvar-', '')  # e.g., '3sat7'
        config_with_variant = {**config, 'variant': variant}
        return gen(dataset_size, seq_len, config_with_variant)

    elif name == 'dvar-path':
        from .path import generate_synthetic_data as gen
        return gen(dataset_size, seq_len, config)

    else:
        raise ValueError(
            f"Unknown dvar dataset: {name}. "
            f"Supported: dvar-sudoku, dvar-cd[3,4,5], dvar-3sat[5,7,9], dvar-path"
        )


def evaluate_completions(predicted_ids, ground_truth_ids, config):
    """
    Dispatcher for diffusion-vs-ar dataset evaluation.

    Routes to the appropriate evaluator based on config['name'].

    Args:
        predicted_ids: numpy array of shape (n_samples, seq_len)
        ground_truth_ids: numpy array of shape (n_samples, seq_len)
        config: dict with configuration. Must include 'name' field.

    Returns:
        dict with evaluation metrics
    """
    if config is None:
        return None

    config_dict = config if isinstance(config, dict) else dict(config)
    name = config_dict.get('name', '')

    if name == 'dvar-sudoku':
        from .sudoku import evaluate_completions as eval_fn
        return eval_fn(predicted_ids, ground_truth_ids)

    elif name.startswith('dvar-cd'):
        from .countdown import evaluate_completions as eval_fn
        return eval_fn(predicted_ids, ground_truth_ids)

    elif name.startswith('dvar-3sat'):
        from .sat3 import evaluate_completions as eval_fn
        return eval_fn(predicted_ids, ground_truth_ids)

    elif name == 'dvar-path':
        from .path import evaluate_completions as eval_fn
        return eval_fn(predicted_ids, ground_truth_ids)

    else:
        return None


def get_tokenizer(tokenizer_name: str):
    """
    Get a tokenizer for a dvar dataset.

    Args:
        tokenizer_name: Name like 'dvar-sudoku', 'dvar-cd', 'dvar-3sat', 'dvar-path'

    Returns:
        Tokenizer instance
    """
    if tokenizer_name == 'dvar-sudoku':
        return DvarSudokuTokenizer()

    elif tokenizer_name in ('dvar-cd', 'dvar-cd3', 'dvar-cd4', 'dvar-cd5'):
        return CountdownTokenizer()

    elif tokenizer_name in ('dvar-3sat', 'dvar-3sat5', 'dvar-3sat7', 'dvar-3sat9'):
        return SAT3Tokenizer()

    elif tokenizer_name == 'dvar-path':
        return PathTokenizer()

    else:
        raise ValueError(
            f"Unknown dvar tokenizer: {tokenizer_name}. "
            f"Supported: dvar-sudoku, dvar-cd, dvar-3sat, dvar-path"
        )


def get_vocab_size(name: str) -> int:
    """Get vocabulary size for a dvar dataset.

    Args:
        name: Dataset name like 'dvar-sudoku', 'dvar-cd4', etc.

    Returns:
        Vocabulary size
    """
    if name == 'dvar-sudoku':
        return SUDOKU_VOCAB_SIZE
    elif name.startswith('dvar-cd'):
        return COUNTDOWN_VOCAB_SIZE
    elif name.startswith('dvar-3sat'):
        return SAT3_VOCAB_SIZE
    elif name == 'dvar-path':
        return PATH_VOCAB_SIZE
    else:
        raise ValueError(f"Unknown dvar dataset: {name}")
