#!/usr/bin/env python3
"""
Standalone script to evaluate sudoku infilling on a model trained on sudoku-small.

This script:
1. Generates sudoku puzzles (partially filled boards) using the sudoku_conditional generator
2. Converts puzzles to sudoku-small tokenization format
3. Uses generate_completions to infill the empty cells
4. Evaluates whether the completed boards are valid sudoku solutions

Expected checkpoint path for diffu-causal-output-sminy trained on sudoku-small-solver:
    ${ESOLM_DATADIR}/checkpoints/diffu-causal-output-sminy-sudoku-small-solver/checkpoints/best.ckpt
    
    Where ESOLM_DATADIR is typically set to something like $HOME/

Usage:
    # With explicit checkpoint path
    python scripts/eval_sudoku_infilling.py \
        --checkpoint /path/to/checkpoints/diffu-causal-output-sminy-sudoku-small-solver/checkpoints/best.ckpt \
        --num_samples 1000 \
        --target_givens 30 \
        --num_steps 81
        
    # With auto-detection (requires ESOLM_DATADIR env var)
    python scripts/eval_sudoku_infilling.py \
        --auto_checkpoint \
        --num_samples 1000 \
        --target_givens 30

Sweep over different numbers of givens:
    for g in 10 20 30 40 50 60 70; do
        python scripts/eval_sudoku_infilling.py --auto_checkpoint --target_givens $g --num_samples 1000
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

from synthetic_data.sudoku_conditional.data import make_puzzle_multi_solution
from synthetic_data.sudoku.verify import check_sudoku
from dataloader import SyntheticTokenizer
from difflm import DiffLM


# ==============================================================================
# Constants for sudoku-small tokenization
# ==============================================================================
SUDOKU_SMALL_VOCAB_SIZE = 11
BOS_TOKEN = SUDOKU_SMALL_VOCAB_SIZE - 2  # 9
EOS_TOKEN = SUDOKU_SMALL_VOCAB_SIZE - 1  # 10
MASK_TOKEN = SUDOKU_SMALL_VOCAB_SIZE     # 11 (vocab_size, used for blanks)
SEQ_LEN = 128  # Model length for sudoku-small


# ==============================================================================
# Puzzle generation and conversion
# ==============================================================================

def generate_puzzle(target_givens: int, seed: Optional[int] = None) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Generate a sudoku puzzle with the specified number of given cells.
    
    Returns:
        (puzzle, solution): 9x9 boards where puzzle has 0s for empty cells
    """
    result = make_puzzle_multi_solution(
        target_givens=target_givens,
        max_count_solutions=10,
        symmetry="none",
        seed=seed,
        require_unique=False,
        skip_solution_count=True,
    )
    return result.puzzle, result.solution


def puzzle_to_sudoku_small_tokens(
    puzzle: List[List[int]], 
    seq_len: int = SEQ_LEN,
    lock_special_tokens: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 9x9 puzzle board to sudoku-small token format.
    
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
    input_ids = np.full(seq_len, fill_value=EOS_TOKEN, dtype=np.int64)
    loss_mask = np.ones(seq_len, dtype=np.int64)
    
    # Set BOS (position 0)
    input_ids[0] = BOS_TOKEN
    if lock_special_tokens:
        loss_mask[0] = 0  # BOS is given
    
    # Set puzzle tokens (positions 1-81)
    k = 1
    for r in range(9):
        for c in range(9):
            v = puzzle[r][c]
            if v > 0:
                # Given cell: digit 1-9 → token 0-8
                input_ids[k] = v - 1
                loss_mask[k] = 0  # Given, don't predict
            else:
                # Empty cell: use MASK token
                input_ids[k] = MASK_TOKEN
                loss_mask[k] = 1  # Need to predict
            k += 1
    
    # EOS at position 82
    input_ids[82] = EOS_TOKEN
    if lock_special_tokens:
        loss_mask[82] = 0  # EOS is given
        # Padding (positions 83+) - set loss_mask=0
        loss_mask[83:] = 0
    
    return input_ids, loss_mask


def extract_board_from_tokens(tokens: np.ndarray) -> List[int]:
    """
    Extract the 81-cell board from a token sequence.
    
    Args:
        tokens: Token sequence of shape (seq_len,)
        
    Returns:
        Flat list of 81 values in range 0-8 (sudoku-small format)
    """
    # Board tokens are at positions 1-81
    board_tokens = tokens[1:82]
    return board_tokens.tolist()


def check_board_validity(board_flat: List[int], puzzle_flat: List[int]) -> dict:
    """
    Check if a completed board is valid.
    
    Args:
        board_flat: 81 values in range 0-8 (sudoku-small format)
        puzzle_flat: 81 values - original puzzle tokens (0-8 for given, MASK_TOKEN for empty)
        
    Returns:
        Dictionary with validation results
    """
    # Convert puzzle to mask format expected by check_sudoku
    # In sudoku-small, MASK_TOKEN (11) indicates empty cells
    puzzle_for_check = []
    for v in puzzle_flat:
        if v == MASK_TOKEN:
            puzzle_for_check.append(MASK_TOKEN)  # Empty cell
        else:
            puzzle_for_check.append(v)  # Given cell (0-8)
    
    return check_sudoku(
        board_flat,
        base=3,
        puzzle_flat=puzzle_for_check,
        mask_index=MASK_TOKEN,
    )


# ==============================================================================
# Model loading
# ==============================================================================

def create_config(checkpoint_path: str, n_latent_tokens: int = 0):
    """Create a config matching the diffu-causal-output-sminy-sudoku-small-solver training config."""
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
            'length': SEQ_LEN,
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
            'diffusion_attn_mode': 'causal_output',  # Key config for diffu-causal-output
            'sequential_shuffle': False,
            'sequential_attn_mode': 'causal',
            'loss_type': 'elbo',
            'ar_noise': False,
            'next_token_prediction': False,
            'mtp_window_size': -1,
            'mtp_loss_ratio': None,
            'mtp_mode': 'contiguous',
            'mtp_rebalance_loss': False,
            'shuffle_clean_tokens': True,  # Key config for diffu-causal-output
            'shuffle_masked_tokens': True,  # Key config for diffu-causal-output
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
            'trim_masked_tokens': None,
            'unmask_policy': 'uniform',
            'n_latent_tokens': n_latent_tokens,
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


def load_model(checkpoint_path: str, device: str = 'cuda', n_latent_tokens: int = 0):
    """Load the trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    config = create_config(checkpoint_path, n_latent_tokens=n_latent_tokens)
    tokenizer = SyntheticTokenizer(vocab_size=SUDOKU_SMALL_VOCAB_SIZE)
    
    model = DiffLM.load_from_checkpoint(
        checkpoint_path,
        config=config,
        tokenizer=tokenizer,
    )
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded. mask_index={model.mask_index}, vocab_size={model.vocab_size}")
    print(f"num_tokens={model.num_tokens}, n_latent_tokens={n_latent_tokens}")
    
    return model


# ==============================================================================
# Main evaluation
# ==============================================================================

def display_board(tokens: List[int], title: str = ""):
    """Display a sudoku board from flat tokens (0-8 format)."""
    print(f"\n{title}")
    print("+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+")
    for i in range(9):
        row = tokens[i*9:(i+1)*9]
        row_str = ""
        for j, v in enumerate(row):
            if j % 3 == 0:
                row_str += "| "
            # Convert from 0-8 to 1-9 for display
            if 0 <= v <= 8:
                row_str += f"{v+1} "
            else:
                row_str += ". "  # Invalid or mask token
        row_str += "|"
        print(row_str)
        if (i + 1) % 3 == 0:
            print("+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+")


def evaluate_infilling(
    model,
    num_samples: int = 1000,
    target_givens: int = 30,
    num_steps: int = 128,
    batch_size: int = 128,
    seed: int = 42,
    verbose: bool = False,
    show_examples: int = 0,
    lock_special_tokens: bool = False,
):
    """
    Evaluate the model on sudoku infilling task.
    
    Args:
        model: Loaded DiffLM model
        num_samples: Number of puzzles to generate and evaluate
        target_givens: Target number of given cells per puzzle
        num_steps: Number of diffusion steps
        batch_size: Batch size for inference
        seed: Random seed for reproducibility
        lock_special_tokens: If True, set loss_mask=0 for BOS, EOS, and padding
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    n_latent_tokens = model.config.sampling.get('n_latent_tokens', 0)
    
    print(f"\n{'='*60}")
    print(f"Sudoku Infilling Evaluation")
    print(f"{'='*60}")
    print(f"  num_samples: {num_samples}")
    print(f"  target_givens: {target_givens}")
    print(f"  num_steps: {num_steps}")
    print(f"  n_latent_tokens: {n_latent_tokens}")
    print(f"  batch_size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Generate puzzles
    print("Generating puzzles...")
    puzzles = []
    solutions = []
    for i in range(num_samples):
        puzzle, solution = generate_puzzle(target_givens, seed=seed + i * 1000)
        puzzles.append(puzzle)
        solutions.append(solution)
    
    # Convert to tokens
    all_input_ids = []
    all_loss_masks = []
    for puzzle in puzzles:
        input_ids, loss_mask = puzzle_to_sudoku_small_tokens(
            puzzle, seq_len=SEQ_LEN, lock_special_tokens=lock_special_tokens
        )
        all_input_ids.append(input_ids)
        all_loss_masks.append(loss_mask)
    
    all_input_ids = np.array(all_input_ids, dtype=np.int64)
    all_loss_masks = np.array(all_loss_masks, dtype=np.int64)
    
    # Count actual givens
    actual_givens = [(81 - loss_mask[1:82].sum()) for loss_mask in all_loss_masks]
    print(f"Actual givens: mean={np.mean(actual_givens):.1f}, min={np.min(actual_givens)}, max={np.max(actual_givens)}")
    
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
    n_matches_original = 0
    total_violations = 0
    total_loss = 0.0
    
    examples_shown = 0
    
    for i in range(num_samples):
        # Extract predicted board
        pred_board = extract_board_from_tokens(all_predictions[i])
        orig_input = extract_board_from_tokens(all_input_ids[i])
        
        # Check validity
        result = check_board_validity(pred_board, orig_input)
        
        if result['is_valid']:
            n_valid += 1
        if result['matches_original']:
            n_matches_original += 1
        total_violations += result['violations']
        total_loss += result['loss']
        
        # Verbose output
        if verbose:
            status = "✓" if result['is_valid'] else "✗"
            print(f"  Sample {i}: {status} violations={result['violations']}, loss={result['loss']:.4f}")
        
        # Show example boards
        if examples_shown < show_examples:
            # Show puzzle (given cells)
            puzzle_display = []
            for j, v in enumerate(orig_input):
                if v == MASK_TOKEN:
                    puzzle_display.append(-1)  # Will display as "."
                else:
                    puzzle_display.append(v)
            display_board(puzzle_display, f"Example {i+1} - Puzzle (given cells)")
            
            # Show predicted solution
            display_board(pred_board, f"Example {i+1} - Predicted (valid={result['is_valid']})")
            examples_shown += 1
    
    # Print results
    max_violations = 3 * 9  # 9 rows + 9 cols + 9 blocks
    max_loss = 3 * 8  # Each unit can have max loss of (n-1)/n = 8/9
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total samples:        {num_samples}")
    print(f"Valid sudoku:         {n_valid} ({100*n_valid/num_samples:.1f}%)")
    print(f"Matches givens:       {n_matches_original} ({100*n_matches_original/num_samples:.1f}%)")
    print(f"Avg violations:       {total_violations/num_samples:.2f} / {max_violations}")
    print(f"Avg loss:             {total_loss/num_samples:.4f} / {max_loss:.4f}")
    print(f"{'='*60}\n")
    
    return {
        'num_samples': num_samples,
        'target_givens': target_givens,
        'actual_givens_mean': np.mean(actual_givens),
        'n_valid': n_valid,
        'valid_rate': n_valid / num_samples,
        'n_matches_original': n_matches_original,
        'matches_rate': n_matches_original / num_samples,
        'avg_violations': total_violations / num_samples,
        'avg_loss': total_loss / num_samples,
        # Return raw data for saving
        'predictions': all_predictions,
        'input_ids': all_input_ids,
        'loss_masks': all_loss_masks,
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sudoku infilling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with auto-detected checkpoint
  python scripts/eval_sudoku_infilling.py --auto_checkpoint --target_givens 30
  
  # Evaluate with explicit checkpoint path
  python scripts/eval_sudoku_infilling.py --checkpoint /path/to/best.ckpt --target_givens 30
  
  # Sweep over latent tokens and givens
  for latent in 0 8 16 32 64 128; do
    for givens in 0 10 20 30; do
      python scripts/eval_sudoku_infilling.py --auto_checkpoint \\
        --target_givens $givens --n_latent_tokens $latent \\
        --output_dir results/infilling_sweep
    done
  done
"""
    )
    
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument('--checkpoint', type=str,
                           help='Path to model checkpoint')
    ckpt_group.add_argument('--auto_checkpoint', action='store_true',
                           help='Auto-detect checkpoint from ESOLM_DATADIR')
    
    parser.add_argument('--run_name', type=str, 
                        default='diffu-causal-output-sminy-sudoku-small-solver',
                        help='Run name for auto checkpoint detection')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of puzzles to evaluate')
    parser.add_argument('--target_givens', type=int, default=30,
                        help='Target number of given cells (0-81)')
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
    
    # Load model
    model = load_model(checkpoint_path, device=args.device, n_latent_tokens=args.n_latent_tokens)
    
    # Run evaluation
    results = evaluate_infilling(
        model=model,
        num_samples=args.num_samples,
        target_givens=args.target_givens,
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
        fname_base = f"givens{args.target_givens}_latent{args.n_latent_tokens}_steps{args.num_steps}"
        
        # Save results to .txt
        results_path = os.path.join(args.output_dir, f"{fname_base}_results.txt")
        with open(results_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Sudoku Infilling Evaluation Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"run_name: {args.run_name}\n")
            f.write(f"checkpoint: {checkpoint_path}\n")
            f.write(f"num_samples: {results['num_samples']}\n")
            f.write(f"target_givens: {results['target_givens']}\n")
            f.write(f"actual_givens_mean: {results['actual_givens_mean']:.1f}\n")
            f.write(f"num_steps: {args.num_steps}\n")
            f.write(f"n_latent_tokens: {args.n_latent_tokens}\n")
            f.write(f"seed: {args.seed}\n")
            f.write("-" * 60 + "\n")
            f.write(f"n_valid: {results['n_valid']}\n")
            f.write(f"valid_rate: {results['valid_rate']*100:.1f}%\n")
            f.write(f"n_matches_original: {results['n_matches_original']}\n")
            f.write(f"matches_rate: {results['matches_rate']*100:.1f}%\n")
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
            target_givens=args.target_givens,
            n_latent_tokens=args.n_latent_tokens,
            num_steps=args.num_steps,
            seed=args.seed,
            valid_rate=results['valid_rate'],
        )
        print(f"Samples saved to: {samples_path}")
    
    print("Done!")
    return results


if __name__ == '__main__':
    main()

