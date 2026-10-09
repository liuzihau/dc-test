#!/usr/bin/env python3
"""
Standalone script to evaluate sudoku infilling on sudoku-puzzle data.

This script:
1. Loads sudoku-puzzle data from .npy files (external puzzle database)
2. Extracts the puzzle portion (partially filled board) from each example
3. Converts puzzles from sudoku-puzzle tokenization to sudoku-small format
4. Uses generate_completions to infill the empty cells
5. Evaluates whether the completed boards match the ground truth solutions

Token mapping differences:
    sudoku-puzzle (vocab_size=14):
        - Digits 1-9 → token IDs 1-9
        - MASK (empty cell) → token ID 10
        - SEP → token ID 11, BOS → token ID 12, EOS → token ID 13, PAD → token ID 0

    sudoku-small (vocab_size=11):
        - Digits 1-9 → token IDs 0-8 (digit - 1)
        - BOS → token ID 9 (vocab_size - 2)
        - EOS → token ID 10 (vocab_size - 1)
        - MASK → token ID 11 (vocab_size, used for blanks during infilling)

Usage:
    # With explicit paths
    python scripts/eval_sudoku_puzzle_infilling.py \
        --checkpoint /path/to/best.ckpt \
        --puzzle_data /path/to/sudoku_puzzles.npy \
        --num_samples 1000 \
        --num_steps 81

    # With auto-detection (requires ESOLM_DATADIR and ESOLM_PUZZLE_DIR env vars)
    python scripts/eval_sudoku_puzzle_infilling.py \
        --auto_checkpoint \
        --auto_puzzle_data \
        --num_samples 1000

    # Sweep over different step counts
    for steps in 32 64 128 256; do
        python scripts/eval_sudoku_puzzle_infilling.py \
            --auto_checkpoint --auto_puzzle_data \
            --num_steps $steps --num_samples 1000
    done
"""

import argparse
import os
import sys
import random
from typing import Optional, List, Tuple

import numpy as np
import torch
import omegaconf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data.sudoku_puzzle.data import (
    parse_sudoku_example,
    build_vocab as build_puzzle_vocab,
    check_sudoku_valid,
)
from synthetic_data.sudoku.verify import check_sudoku
from dataloader import SyntheticTokenizer
from difflm import DiffLM


# ==============================================================================
# Constants for sudoku-small tokenization (target format for model)
# ==============================================================================
SUDOKU_SMALL_VOCAB_SIZE = 11
SMALL_BOS_TOKEN = SUDOKU_SMALL_VOCAB_SIZE - 2  # 9
SMALL_EOS_TOKEN = SUDOKU_SMALL_VOCAB_SIZE - 1  # 10
SMALL_MASK_TOKEN = SUDOKU_SMALL_VOCAB_SIZE     # 11 (vocab_size, used for blanks)
SMALL_SEQ_LEN = 128  # Model length for sudoku-small

# Constants for sudoku-puzzle tokenization (source format from data)
PUZZLE_BOS = 12
PUZZLE_EOS = 13
PUZZLE_SEP = 11
PUZZLE_MASK = 10
PUZZLE_PAD = 0


# ==============================================================================
# Data loading and conversion
# ==============================================================================

def load_puzzle_data(data_path: str, num_samples: Optional[int] = None, seed: int = 42):
    """
    Load sudoku puzzle data from .npy file.

    Args:
        data_path: Path to .npy file containing puzzle data
        num_samples: Number of samples to load (None for all)
        seed: Random seed for shuffling

    Returns:
        List of parsed puzzle dicts with 'puzzle' and 'solution' keys
    """
    print(f"Loading puzzle data from: {data_path}")
    raw_data = np.load(data_path)

    n_available = len(raw_data)
    if num_samples is None:
        num_samples = n_available
    num_samples = min(num_samples, n_available)

    # Shuffle and select
    np.random.seed(seed)
    indices = np.random.permutation(n_available)[:num_samples]

    parsed_examples = []
    for idx in indices:
        parsed = parse_sudoku_example(raw_data[idx])
        parsed_examples.append(parsed)

    print(f"Loaded {len(parsed_examples)} puzzles from {n_available} available")
    return parsed_examples


def puzzle_to_sudoku_small_tokens(
    puzzle: np.ndarray,
    seq_len: int = SMALL_SEQ_LEN,
    lock_special_tokens: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 9x9 puzzle board (from sudoku-puzzle format) to sudoku-small token format.

    sudoku-small tokenization:
    - Digits 1-9 → tokens 0-8 (digit - 1)
    - Empty cells (0) → MASK token (11)
    - Format: [BOS] + 81 tokens + [EOS] + padding(EOS)

    Args:
        puzzle: 9x9 board with 0 for empty cells, 1-9 for given digits
        seq_len: sequence length to pad to
        lock_special_tokens: if True, set loss_mask=0 for BOS, EOS, and padding

    Returns:
        (input_ids, loss_mask): numpy arrays of shape (seq_len,)
            - input_ids: tokens including BOS, puzzle, EOS, and padding
            - loss_mask: 0 for given tokens, 1 for tokens to predict (empty cells)
    """
    # Initialize arrays
    # Default: EOS for padding, loss_mask=1 (predict everything by default)
    input_ids = np.full(seq_len, fill_value=SMALL_EOS_TOKEN, dtype=np.int64)
    loss_mask = np.ones(seq_len, dtype=np.int64)

    # Set BOS (position 0)
    input_ids[0] = SMALL_BOS_TOKEN
    if lock_special_tokens:
        loss_mask[0] = 0  # BOS is given

    # Flatten puzzle and set tokens (positions 1-81)
    puzzle_flat = puzzle.flatten()
    for k, v in enumerate(puzzle_flat):
        pos = k + 1  # Position in sequence (after BOS)
        if v > 0:
            # Given cell: digit 1-9 → token 0-8
            input_ids[pos] = v - 1
            loss_mask[pos] = 0  # Given, don't predict
        else:
            # Empty cell: use MASK token
            input_ids[pos] = SMALL_MASK_TOKEN
            loss_mask[pos] = 1  # Need to predict

    # EOS at position 82
    input_ids[82] = SMALL_EOS_TOKEN
    if lock_special_tokens:
        loss_mask[82] = 0  # EOS is given
        # Padding (positions 83+) - set loss_mask=0
        loss_mask[83:] = 0

    return input_ids, loss_mask


def extract_board_from_tokens(tokens: np.ndarray) -> List[int]:
    """
    Extract the 81-cell board from a sudoku-small token sequence.

    Args:
        tokens: Token sequence of shape (seq_len,) in sudoku-small format

    Returns:
        Flat list of 81 values in range 0-8 (sudoku-small format)
    """
    # Board tokens are at positions 1-81
    board_tokens = tokens[1:82]
    return board_tokens.tolist()


def tokens_to_solution_grid(board_tokens: List[int]) -> np.ndarray:
    """
    Convert sudoku-small board tokens (0-8) to a solution grid (1-9).

    Args:
        board_tokens: List of 81 token values in range 0-8

    Returns:
        numpy array of shape (9, 9) with values 1-9
    """
    # Convert 0-8 tokens to 1-9 values
    values = [t + 1 if 0 <= t <= 8 else -1 for t in board_tokens]
    return np.array(values, dtype=np.int32).reshape(9, 9)


def check_board_validity(board_flat: List[int], puzzle_flat: List[int]) -> dict:
    """
    Check if a completed board is valid.

    Args:
        board_flat: 81 values in range 0-8 (sudoku-small format)
        puzzle_flat: 81 values - original puzzle tokens (0-8 for given, MASK_TOKEN for empty)

    Returns:
        Dictionary with validation results
    """
    return check_sudoku(
        board_flat,
        base=3,
        puzzle_flat=puzzle_flat,
        mask_index=SMALL_MASK_TOKEN,
    )


# ==============================================================================
# Model loading (same as eval_sudoku_infilling.py)
# ==============================================================================

def create_config(
    checkpoint_path: str,
    n_latent_tokens: int = 0,
    unmask_policy: str = 'uniform',
    topk_candidate_max: int = 0,
    topk_candidate_min: int = 0,
    model_type: str = 'diffu-causal-output',
):
    """Create a config for the specified model type.

    Supported model types:
        - diffu-causal-output: causal_output attention, shuffle both clean and masked tokens
        - diffu-solo-full: solo_full attention, shuffle both clean and masked tokens
        - diffu-full: full (bidirectional) attention, no shuffling
    """
    # Set attention mode and shuffle settings based on model type
    if model_type == 'diffu-causal-output':
        diffusion_attn_mode = 'causal_output'
        shuffle_clean_tokens = True
        shuffle_masked_tokens = True
    elif model_type == 'diffu-solo-full':
        diffusion_attn_mode = 'solo_full'
        shuffle_clean_tokens = True
        shuffle_masked_tokens = True
    elif model_type == 'diffu-full':
        diffusion_attn_mode = 'full'
        shuffle_clean_tokens = True
        shuffle_masked_tokens = True
    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                        f"Supported: diffu-causal-output, diffu-solo-full, diffu-full")

    config = omegaconf.OmegaConf.create({
        'data': {
            'vocab_size': SUDOKU_SMALL_VOCAB_SIZE,
            'tokenizer_name_or_path': 'synthetic',
        },
        'model': {
            'name': 'small',  # sminy uses same structure
            'type': 'ddit',
            'hidden_size': 768,
            'cond_dim': 128,
            'length': SMALL_SEQ_LEN,
            'n_blocks': 6,
            'n_heads': 12,
            'scale_by_sigma': True,
            'dropout': 0.1,
            'tie_word_embeddings': False,
            'absolute_pos_embed': True,
            'disable_adaln': True,
            'decode_head': False,
        },
        'algo': {
            'name': 'difflm',
            'backbone': 'esolm_dit',
            'parameterization': 'subs',
            'time_conditioning': False,
            'T': 0,
            'subs_masking': False,
            'causal_attention': False,
            'alpha_0': 1.0,
            'batch_split': 1.0,
            'diffusion_shuffle': True,
            'diffusion_attn_mode': diffusion_attn_mode,
            'sequential_shuffle': False,
            'sequential_attn_mode': 'causal',
            'loss_type': 'elbo',
            'ar_noise': False,
            'next_token_prediction': False,
            'mtp_window_size': -1,
            'mtp_loss_ratio': None,
            'mtp_mode': 'contiguous',
            'mtp_rebalance_loss': False,
            'shuffle_clean_tokens': shuffle_clean_tokens,
            'shuffle_masked_tokens': shuffle_masked_tokens,
        },
        'training': {
            'ema': 0.9999,
            'antithetic_sampling': True,
            'importance_sampling': False,
            'sampling_eps': 1e-3,
            'change_of_variables': False,
        },
        'sampling': {
            'predictor': 'ddpm_cache',
            'steps': 128,
            'noise_removal': 'none',
            'noise_scale': 1.0,
            'p_nucleus': 1.0,
            'use_float64': False,
            'stride_length': 1,
            'num_strides': 1,
            'semi_ar': False,
            'kv_cache': None,
            'trim_masked_tokens': False,
            'unmask_policy': unmask_policy,
            'n_latent_tokens': n_latent_tokens,
            'topk_candidate_max': topk_candidate_max,
            'topk_candidate_min': topk_candidate_min,
            'fixed_tokens_per_step': True,  # Match gen_master.sh / synthetic_base.yaml
            'reshuffle_masks': False,
        },
        'loader': {
            'global_batch_size': 512,
            'eval_batch_size': 128,
        },
        'eval': {
            'checkpoint_path': checkpoint_path,
            'disable_ema': False,
            'compute_generative_perplexity': False,
            'gen_ppl_eval_model_name_or_path': 'gpt2-large',  # Required for Metrics init even if not used
            'perplexity_batch_size': 128,
        },
        'optim': {
            'lr': 3e-4,
        }
    })
    return config


def load_model(
    checkpoint_path: str,
    device: str = 'cuda',
    n_latent_tokens: int = 0,
    unmask_policy: str = 'uniform',
    topk_candidate_max: int = 0,
    topk_candidate_min: int = 0,
    model_type: str = 'diffu-causal-output',
):
    """Load the trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    print(f"Model type: {model_type}")

    config = create_config(
        checkpoint_path,
        n_latent_tokens=n_latent_tokens,
        unmask_policy=unmask_policy,
        topk_candidate_max=topk_candidate_max,
        topk_candidate_min=topk_candidate_min,
        model_type=model_type,
    )
    tokenizer = SyntheticTokenizer(vocab_size=SUDOKU_SMALL_VOCAB_SIZE)

    model = DiffLM.load_from_checkpoint(
        checkpoint_path,
        config=config,
        tokenizer=tokenizer,
    )
    model = model.to(device)
    model.eval()

    print(f"Model loaded. mask_index={model.mask_index}, vocab_size={model.vocab_size}")
    print(f"  diffusion_attn_mode={config.algo.diffusion_attn_mode}")
    print(f"  shuffle_clean_tokens={config.algo.shuffle_clean_tokens}, shuffle_masked_tokens={config.algo.shuffle_masked_tokens}")
    print(f"  n_latent_tokens={n_latent_tokens}, unmask_policy={unmask_policy}")
    if unmask_policy != 'uniform':
        print(f"  topk_candidate_max={topk_candidate_max}, topk_candidate_min={topk_candidate_min}")

    return model


# ==============================================================================
# Main evaluation
# ==============================================================================

def display_board(grid: np.ndarray, title: str = ""):
    """Display a 9x9 sudoku board."""
    print(f"\n{title}")
    print("+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+")
    for i in range(9):
        row_str = ""
        for j in range(9):
            if j % 3 == 0:
                row_str += "| "
            v = grid[i, j]
            if 1 <= v <= 9:
                row_str += f"{v} "
            else:
                row_str += ". "
        row_str += "|"
        print(row_str)
        if (i + 1) % 3 == 0:
            print("+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+")


def evaluate_infilling(
    model,
    parsed_examples: List[dict],
    num_steps: int = 128,
    batch_size: int = 128,
    seed: int = 42,
    verbose: bool = False,
    show_examples: int = 0,
    lock_special_tokens: bool = False,
):
    """
    Evaluate the model on sudoku infilling task using sudoku-puzzle data.

    Args:
        model: Loaded DiffLM model
        parsed_examples: List of parsed puzzle dicts with 'puzzle' and 'solution' keys
        num_steps: Number of diffusion steps
        batch_size: Batch size for inference
        seed: Random seed for reproducibility
        lock_special_tokens: If True, set loss_mask=0 for BOS, EOS, and padding
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_samples = len(parsed_examples)
    n_latent_tokens = model.config.sampling.get('n_latent_tokens', 0)
    unmask_policy = model.config.sampling.get('unmask_policy', 'uniform')
    topk_candidate_max = model.config.sampling.get('topk_candidate_max', 0)
    topk_candidate_min = model.config.sampling.get('topk_candidate_min', 0)

    # Compute statistics about the puzzles
    n_givens_list = [ex['n_given'] for ex in parsed_examples]

    print(f"\n{'='*60}")
    print(f"Sudoku Puzzle Infilling Evaluation")
    print(f"{'='*60}")
    print(f"  num_samples: {num_samples}")
    print(f"  givens: mean={np.mean(n_givens_list):.1f}, min={np.min(n_givens_list)}, max={np.max(n_givens_list)}")
    print(f"  num_steps: {num_steps}")
    print(f"  n_latent_tokens: {n_latent_tokens}")
    print(f"  unmask_policy: {unmask_policy}")
    if unmask_policy != 'uniform':
        print(f"  topk_candidate_max: {topk_candidate_max}")
        print(f"  topk_candidate_min: {topk_candidate_min}")
    print(f"  batch_size: {batch_size}")
    print(f"{'='*60}\n")

    # Convert puzzles to sudoku-small tokens
    print("Converting puzzles to sudoku-small format...")
    all_input_ids = []
    all_loss_masks = []
    all_solutions = []

    for ex in parsed_examples:
        puzzle = ex['puzzle']  # 9x9 numpy array
        solution = ex['solution']  # 9x9 numpy array

        input_ids, loss_mask = puzzle_to_sudoku_small_tokens(
            puzzle, seq_len=SMALL_SEQ_LEN, lock_special_tokens=lock_special_tokens
        )
        all_input_ids.append(input_ids)
        all_loss_masks.append(loss_mask)
        all_solutions.append(solution)

    all_input_ids = np.array(all_input_ids, dtype=np.int64)
    all_loss_masks = np.array(all_loss_masks, dtype=np.int64)

    # Run inference in batches
    print(f"\nRunning inference...")
    all_predictions = []

    device = next(model.parameters()).device

    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        print(f"  Processing samples {batch_start}-{batch_end}...")

        batch_input_ids = torch.tensor(all_input_ids[batch_start:batch_end], device=device)
        batch_loss_masks = torch.tensor(all_loss_masks[batch_start:batch_end], device=device)

        batch = {
            'input_ids': batch_input_ids,
            'loss_mask': batch_loss_masks,
        }

        with torch.no_grad():
            predictions = model.restore_model_and_complete(
                batch=batch,
                num_steps=num_steps,
                return_stats=False,
            )

        all_predictions.append(predictions.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)

    # Evaluate results
    print(f"\nEvaluating results...")
    n_valid = 0
    n_correct = 0
    n_matches_givens = 0
    total_violations = 0
    total_loss = 0.0
    total_cell_accuracy = 0.0

    # Collect grids for saving
    all_pred_grids = []
    all_puzzle_grids = []
    all_gt_grids = []
    all_results = []  # Per-sample results

    examples_shown = 0

    for i in range(num_samples):
        # Extract predicted board (in sudoku-small format: 0-8)
        pred_board_tokens = extract_board_from_tokens(all_predictions[i])
        orig_input_tokens = extract_board_from_tokens(all_input_ids[i])

        # Convert to solution grids (1-9)
        pred_grid = tokens_to_solution_grid(pred_board_tokens)
        gt_grid = all_solutions[i]
        puzzle_grid = parsed_examples[i]['puzzle']

        # Collect grids
        all_pred_grids.append(pred_grid)
        all_puzzle_grids.append(puzzle_grid)
        all_gt_grids.append(gt_grid)

        # Check validity using check_sudoku
        result = check_board_validity(pred_board_tokens, orig_input_tokens)

        if result['is_valid']:
            n_valid += 1
        if result['matches_original']:
            n_matches_givens += 1
        total_violations += result['violations']
        total_loss += result['loss']

        # Check if solution matches ground truth
        cell_correct = (pred_grid == gt_grid)
        cell_accuracy = np.mean(cell_correct)
        total_cell_accuracy += cell_accuracy

        is_correct = np.all(cell_correct)
        if is_correct:
            n_correct += 1

        # Store per-sample result
        all_results.append({
            'index': i,
            'n_given': parsed_examples[i]['n_given'],
            'is_valid': result['is_valid'],
            'is_correct': is_correct,
            'cell_accuracy': cell_accuracy,
            'violations': result['violations'],
        })

        # Verbose output
        if verbose:
            status = "✓" if np.all(cell_correct) else "✗"
            valid_str = "valid" if result['is_valid'] else "invalid"
            print(f"  Sample {i}: {status} cell_acc={cell_accuracy:.2f}, {valid_str}, violations={result['violations']}")

        # Show example boards
        if examples_shown < show_examples:
            display_board(puzzle_grid, f"Example {i+1} - Puzzle ({parsed_examples[i]['n_given']} givens)")
            display_board(gt_grid, f"Example {i+1} - Ground Truth Solution")
            display_board(pred_grid, f"Example {i+1} - Predicted (valid={result['is_valid']}, correct={np.all(cell_correct)})")
            examples_shown += 1

    # Print results
    max_violations = 3 * 9  # 9 rows + 9 cols + 9 blocks

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total samples:        {num_samples}")
    print(f"Valid sudoku:         {n_valid} ({100*n_valid/num_samples:.1f}%)")
    print(f"Correct solution:     {n_correct} ({100*n_correct/num_samples:.1f}%)")
    print(f"Matches givens:       {n_matches_givens} ({100*n_matches_givens/num_samples:.1f}%)")
    print(f"Avg cell accuracy:    {total_cell_accuracy/num_samples:.4f}")
    print(f"Avg violations:       {total_violations/num_samples:.2f} / {max_violations}")
    print(f"Avg loss:             {total_loss/num_samples:.4f}")
    print(f"{'='*60}\n")

    return {
        'num_samples': num_samples,
        'n_givens_mean': np.mean(n_givens_list),
        'n_givens_min': np.min(n_givens_list),
        'n_givens_max': np.max(n_givens_list),
        'n_valid': n_valid,
        'valid_rate': n_valid / num_samples,
        'n_correct': n_correct,
        'correct_rate': n_correct / num_samples,
        'n_matches_givens': n_matches_givens,
        'matches_rate': n_matches_givens / num_samples,
        'avg_cell_accuracy': total_cell_accuracy / num_samples,
        'avg_violations': total_violations / num_samples,
        'avg_loss': total_loss / num_samples,
        # Return raw data for saving
        'predictions': all_predictions,
        'input_ids': all_input_ids,
        'loss_masks': all_loss_masks,
        'solutions': np.array([ex['solution'] for ex in parsed_examples]),
        # Grids for inspection
        'pred_grids': np.array(all_pred_grids),
        'puzzle_grids': np.array(all_puzzle_grids),
        'gt_grids': np.array(all_gt_grids),
        'per_sample_results': all_results,
    }


def find_checkpoint(base_dir: str, run_name: str = "diffu-causal-output-sminy-sudoku-small-solver") -> str:
    """Find the best checkpoint in the expected directory."""
    import glob

    checkpoint_dir = os.path.join(base_dir, "checkpoints", run_name, "checkpoints")

    # Try best*.ckpt first (sorted to get latest version)
    best_ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "best*.ckpt")))
    if best_ckpts:
        return best_ckpts[-1]

    # Fall back to last.ckpt
    last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        return last_ckpt

    raise FileNotFoundError(
        f"No checkpoint found in {checkpoint_dir}\n"
        f"Expected best*.ckpt or last.ckpt"
    )


def find_puzzle_data(puzzle_dir: str, split: str = "test") -> str:
    """Find puzzle data file in the puzzle directory."""
    import glob

    # Common patterns for puzzle data files
    patterns = [
        os.path.join(puzzle_dir, f"sudoku_{split}.npy"),
        os.path.join(puzzle_dir, f"{split}.npy"),
        os.path.join(puzzle_dir, f"sudoku_puzzles_{split}.npy"),
    ]

    for pattern in patterns:
        if os.path.exists(pattern):
            return pattern

    # Try glob for any .npy file with 'test' or 'valid' in name
    test_files = glob.glob(os.path.join(puzzle_dir, f"*{split}*.npy"))
    if test_files:
        return sorted(test_files)[0]

    # List available files for error message
    available = glob.glob(os.path.join(puzzle_dir, "*.npy"))
    raise FileNotFoundError(
        f"No puzzle data file found in {puzzle_dir}\n"
        f"Available .npy files: {available}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sudoku infilling on sudoku-puzzle data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with auto-detected paths
  python scripts/eval_sudoku_puzzle_infilling.py --auto_checkpoint --auto_puzzle_data

  # Evaluate with explicit paths
  python scripts/eval_sudoku_puzzle_infilling.py \
    --checkpoint /path/to/best.ckpt \
    --puzzle_data /path/to/sudoku_test.npy

  # Sweep over latent tokens and steps
  for latent in 0 8 16 32 64 128; do
    for steps in 32 64 128; do
      python scripts/eval_sudoku_puzzle_infilling.py \
        --auto_checkpoint --auto_puzzle_data \
        --n_latent_tokens $latent --num_steps $steps \
        --output_dir results/puzzle_infilling_sweep
    done
  done
"""
    )

    # Checkpoint options
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument('--checkpoint', type=str,
                           help='Path to model checkpoint')
    ckpt_group.add_argument('--auto_checkpoint', action='store_true',
                           help='Auto-detect checkpoint from ESOLM_DATADIR')

    # Puzzle data options
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--puzzle_data', type=str,
                           help='Path to puzzle data .npy file')
    data_group.add_argument('--auto_puzzle_data', action='store_true',
                           help='Auto-detect puzzle data from ESOLM_PUZZLE_DIR')

    parser.add_argument('--run_name', type=str,
                        default='diffu-causal-output-sminy-sudoku-small-solver',
                        help='Run name for auto checkpoint detection')
    parser.add_argument('--split', type=str, default='test',
                        help='Data split to use (test, train, valid)')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of puzzles to evaluate')
    parser.add_argument('--num_steps', type=int, default=128,
                        help='Number of diffusion steps')
    parser.add_argument('--n_latent_tokens', type=int, default=0,
                        help='Number of latent tokens for inference')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for inference')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed per-sample results')
    parser.add_argument('--show_examples', type=int, default=0,
                        help='Number of example boards to display')
    parser.add_argument('--lock_special_tokens', action='store_true',
                        help='If set, lock BOS, EOS, and padding tokens (loss_mask=0)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results and samples')

    # Sampling policy options
    parser.add_argument('--unmask_policy', type=str, default='uniform',
                        choices=['uniform', 'topp', 'entropy', 'topp_margin', 'random'],
                        help='Policy for selecting which tokens to unmask (default: uniform)')
    parser.add_argument('--topk_candidate_max', type=int, default=8,
                        help='Max candidates for topp/entropy policies (default: 8)')
    parser.add_argument('--topk_candidate_min', type=int, default=8,
                        help='Min candidates to keep for topp/entropy policies (default: 8)')

    # Model type options
    parser.add_argument('--model_type', type=str, default='diffu-causal-output',
                        choices=['diffu-causal-output', 'diffu-solo-full', 'diffu-full'],
                        help='Model type determining attention mode and shuffle settings')

    args = parser.parse_args()

    # Resolve checkpoint path
    if args.auto_checkpoint:
        datadir = os.environ.get('ESOLM_DATADIR')
        if not datadir:
            print("Error: ESOLM_DATADIR environment variable not set")
            print("Set it to your data directory (e.g., $HOME/)")
            sys.exit(1)
        checkpoint_path = find_checkpoint(datadir, args.run_name)
        print(f"Auto-detected checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)

    # Resolve puzzle data path
    if args.auto_puzzle_data:
        puzzle_dir = os.environ.get('ESOLM_PUZZLE_DIR')
        if not puzzle_dir:
            print("Error: ESOLM_PUZZLE_DIR environment variable not set")
            print("Set it to your puzzle data directory")
            sys.exit(1)
        puzzle_data_path = find_puzzle_data(puzzle_dir, args.split)
        print(f"Auto-detected puzzle data: {puzzle_data_path}")
    else:
        puzzle_data_path = args.puzzle_data
        if not os.path.exists(puzzle_data_path):
            print(f"Error: Puzzle data not found: {puzzle_data_path}")
            sys.exit(1)

    # Load puzzle data
    parsed_examples = load_puzzle_data(
        puzzle_data_path,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    # Load model
    model = load_model(
        checkpoint_path,
        device=args.device,
        n_latent_tokens=args.n_latent_tokens,
        unmask_policy=args.unmask_policy,
        topk_candidate_max=args.topk_candidate_max,
        topk_candidate_min=args.topk_candidate_min,
        model_type=args.model_type,
    )

    # Run evaluation
    results = evaluate_infilling(
        model=model,
        parsed_examples=parsed_examples,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=args.verbose,
        show_examples=args.show_examples,
        lock_special_tokens=args.lock_special_tokens,
    )

    # Save results if output_dir specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # Create filename with parameters
        fname_base = f"{args.model_type}_latent{args.n_latent_tokens}_steps{args.num_steps}"
        if args.unmask_policy != 'uniform':
            fname_base += f"_{args.unmask_policy}_max{args.topk_candidate_max}_min{args.topk_candidate_min}"

        # Save results to .txt
        results_path = os.path.join(args.output_dir, f"{fname_base}_results.txt")
        with open(results_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Sudoku Puzzle Infilling Evaluation Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"model_type: {args.model_type}\n")
            f.write(f"run_name: {args.run_name}\n")
            f.write(f"checkpoint: {checkpoint_path}\n")
            f.write(f"puzzle_data: {puzzle_data_path}\n")
            f.write(f"num_samples: {results['num_samples']}\n")
            f.write(f"n_givens: mean={results['n_givens_mean']:.1f}, min={results['n_givens_min']}, max={results['n_givens_max']}\n")
            f.write(f"num_steps: {args.num_steps}\n")
            f.write(f"n_latent_tokens: {args.n_latent_tokens}\n")
            f.write(f"unmask_policy: {args.unmask_policy}\n")
            if args.unmask_policy != 'uniform':
                f.write(f"topk_candidate_max: {args.topk_candidate_max}\n")
                f.write(f"topk_candidate_min: {args.topk_candidate_min}\n")
            f.write(f"seed: {args.seed}\n")
            f.write("-" * 60 + "\n")
            f.write(f"n_valid: {results['n_valid']}\n")
            f.write(f"valid_rate: {results['valid_rate']*100:.1f}%\n")
            f.write(f"n_correct: {results['n_correct']}\n")
            f.write(f"correct_rate: {results['correct_rate']*100:.1f}%\n")
            f.write(f"n_matches_givens: {results['n_matches_givens']}\n")
            f.write(f"matches_rate: {results['matches_rate']*100:.1f}%\n")
            f.write(f"avg_cell_accuracy: {results['avg_cell_accuracy']:.4f}\n")
            f.write(f"avg_violations: {results['avg_violations']:.2f}\n")
            f.write(f"avg_loss: {results['avg_loss']:.4f}\n")
            f.write("=" * 60 + "\n")
        print(f"Results saved to: {results_path}")

        # Save samples to .npz
        samples_path = os.path.join(args.output_dir, f"{fname_base}_samples.npz")
        np.savez_compressed(
            samples_path,
            predictions=results['predictions'],
            input_ids=results['input_ids'],
            loss_masks=results['loss_masks'],
            solutions=results['solutions'],
            pred_grids=results['pred_grids'],
            puzzle_grids=results['puzzle_grids'],
            gt_grids=results['gt_grids'],
            n_latent_tokens=args.n_latent_tokens,
            num_steps=args.num_steps,
            seed=args.seed,
            valid_rate=results['valid_rate'],
            correct_rate=results['correct_rate'],
        )
        print(f"Samples saved to: {samples_path}")

        # Save human-readable solutions file
        solutions_path = os.path.join(args.output_dir, f"{fname_base}_solutions.txt")
        with open(solutions_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("Generated Sudoku Solutions (for inspection)\n")
            f.write(f"latent_tokens={args.n_latent_tokens}, steps={args.num_steps}\n")
            f.write("=" * 70 + "\n\n")

            for i, res in enumerate(results['per_sample_results']):
                puzzle = results['puzzle_grids'][i]
                pred = results['pred_grids'][i]
                gt = results['gt_grids'][i]

                status = "CORRECT" if res['is_correct'] else ("VALID" if res['is_valid'] else "INVALID")
                f.write(f"Sample {i}: {status} | givens={res['n_given']} | cell_acc={res['cell_accuracy']:.2f} | violations={res['violations']}\n")
                f.write("-" * 70 + "\n")

                # Write three boards side by side: Puzzle | Predicted | Ground Truth
                f.write(f"{'PUZZLE':<23} | {'PREDICTED':<23} | {'GROUND TRUTH':<23}\n")
                for row in range(9):
                    puzzle_row = " ".join(str(v) if v > 0 else "." for v in puzzle[row])
                    pred_row = " ".join(str(v) if 1 <= v <= 9 else "?" for v in pred[row])
                    gt_row = " ".join(str(v) for v in gt[row])
                    f.write(f"{puzzle_row:<23} | {pred_row:<23} | {gt_row:<23}\n")
                    if (row + 1) % 3 == 0 and row < 8:
                        f.write(f"{'-'*23} | {'-'*23} | {'-'*23}\n")
                f.write("\n")

        print(f"Solutions saved to: {solutions_path}")

        # Also save a CSV summary for easy analysis
        csv_path = os.path.join(args.output_dir, f"{fname_base}_summary.csv")
        with open(csv_path, 'w') as f:
            f.write("index,n_given,is_valid,is_correct,cell_accuracy,violations\n")
            for res in results['per_sample_results']:
                f.write(f"{res['index']},{res['n_given']},{res['is_valid']},{res['is_correct']},{res['cell_accuracy']:.4f},{res['violations']}\n")
        print(f"CSV summary saved to: {csv_path}")

    print("Done!")
    return results


if __name__ == '__main__':
    main()
