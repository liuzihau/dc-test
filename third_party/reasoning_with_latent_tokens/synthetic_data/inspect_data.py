#!/usr/bin/env python3
"""
Unified entry point for inspecting synthetic puzzle datasets.

Usage:
    python -m synthetic_data.inspect_data sudoku-puzzle /path/to/sudoku-train-data.npy [--num 5]
    python -m synthetic_data.inspect_data zebra /path/to/zebra-train-data.pkl [--num 5]
    
This script allows you to load and inspect the processed data format
to verify that the conversion is working correctly before training.
"""

import argparse
import pickle
import sys
import numpy as np


def inspect_sudoku_puzzle(data_path, num_examples=5, vocab_size=13):
    """
    Load and inspect sudoku puzzle data for debugging/verification.
    
    Args:
        data_path: path to the .npy file
        num_examples: number of examples to display
        vocab_size: vocabulary size for sequence conversion
        
    Returns:
        List of parsed examples
    """
    from synthetic_data.sudoku_puzzle.data import parse_sudoku_example, sudoku_to_sequence, decode_sudoku_sequence
    
    raw_data = np.load(data_path)
    print(f"\n{'='*70}")
    print(f"SUDOKU PUZZLE DATA INSPECTION")
    print(f"{'='*70}")
    print(f"Data file: {data_path}")
    print(f"Total examples: {len(raw_data)}")
    print(f"Example shape: {raw_data[0].shape}")
    print()
    
    STRATEGY_NAMES = {
        0: "Given",
        2: "Lone single",
        3: "Hidden single",
        4: "Naked pair",
        5: "Naked triplet",
        6: "Locked candidate",
        7: "XY Wing",
        8: "Unique rectangle",
    }
    
    examples = []
    for i in range(min(num_examples, len(raw_data))):
        parsed = parse_sudoku_example(raw_data[i])
        examples.append(parsed)
        
        print(f"{'─'*70}")
        print(f"Example {i+1}:")
        print(f"  Number of given cells: {parsed['n_given']}")
        print(f"  Number of cells to solve: {81 - parsed['n_given']}")
        
        # Strategy distribution
        # unique, counts = np.unique(parsed['strategies'], return_counts=True)
        # print(f"  Strategy distribution:")
        # for strat, count in zip(unique, counts):
        #     name = STRATEGY_NAMES.get(strat, f"Unknown({strat})")
        #     print(f"    {name}: {count}")
        
        # print(f"\n  Puzzle:")
        # for row in range(9):
        #     row_str = ""
        #     for col in range(9):
        #         val = parsed['puzzle'][row, col]
        #         row_str += f"{val if val > 0 else '.'} "
        #         if col in [2, 5]:
        #             row_str += "| "
        #     print(f"    {row_str}")
        #     if row in [2, 5]:
        #         print(f"    {'─'*21}")
        
        # print(f"\n  Solution:")
        # for row in range(9):
        #     row_str = ""
        #     for col in range(9):
        #         val = parsed['solution'][row, col]
        #         row_str += f"{val} "
        #         if col in [2, 5]:
        #             row_str += "| "
        #     print(f"    {row_str}")
        #     if row in [2, 5]:
        #         print(f"    {'─'*21}")
        
        # Show sequence representation
        result = sudoku_to_sequence(raw_data[i], vocab_size, include_puzzle=True)
        seq = result['input_ids']
        loss_mask = result['loss_mask']
        solution_start = result['solution_start']
        
        print(f"\n  Sequence (length {len(seq)}):")
        print(f"    Solution starts at index: {solution_start}")
        print(f"    Encoded tokens (full): {seq.tolist()}")
        print(f"    Loss mask (full): {loss_mask.tolist()}")
        # Decode full sequence
        decoded = decode_sudoku_sequence(seq, vocab_size)
        print(f"    Decoded tokens (full): {decoded}")
        # Show which tokens are trained on
        train_mask = ["T" if m == 1 else "_" for m in loss_mask]
        print(f"    Training mask: {''.join(train_mask)}")
    
    print(f"{'='*70}\n")
    return examples


def inspect_zebra(data_path, num_examples=5):
    """
    Load and inspect zebra puzzle data for debugging/verification.
    
    Args:
        data_path: path to the .pkl file
        num_examples: number of examples to display
        
    Returns:
        List of parsed examples
    """
    from synthetic_data.zebra.data import (
        parse_zebra_example, 
        zebra_to_sequence, 
        build_vocab
    )
    
    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    # Build hardcoded vocabulary
    vocab_map, id_to_token = build_vocab()
    
    print(f"\n{'='*70}")
    print(f"ZEBRA PUZZLE DATA INSPECTION")
    print(f"{'='*70}")
    print(f"Data file: {data_path}")
    print(f"Total examples: {len(raw_data)}")
    print(f"Built vocabulary size: {len(vocab_map)}")
    print(f"Vocabulary tokens: {list(vocab_map.keys())}")
    print()

    # compute max clues length and max solution length
    clues_lengths = []
    max_clues_length = 0
    max_solution_length = 0
    for example in raw_data:
        parsed = parse_zebra_example(example)
        clues_lengths.append(len(parsed['clues']))
        max_clues_length = max(max_clues_length, len(parsed['clues']))
        max_solution_length = max(max_solution_length, len(parsed['solution'].flatten()))
    print(f"Max clues length: {max_clues_length}")
    # get 99 percentile of clues length
    print(f"99 percentile of clues length: {np.percentile(clues_lengths, 99)}")
    print(f"Max solution length: {max_solution_length}")
    
    examples = []
    for i in range(min(num_examples, len(raw_data))):
        break
        parsed = parse_zebra_example(raw_data[i])
        examples.append(parsed)
        
        # print(f"{'─'*70}")
        # print(f"Example {i+1}:")
        # print(f"  Number of houses: {parsed['n_houses']}")
        # print(f"  Number of attributes: {parsed['n_attrs']}")
        # print(f"  Number of clues: {len(parsed['clues'])}")
        
        # print(f"\n  Clues:")
        # for j, clue in enumerate(parsed['clues'][:10]):  # Show first 10 clues
        #     print(f"    {j+1}. {clue}")
        # if len(parsed['clues']) > 10:
        #     print(f"    ... and {len(parsed['clues'])-10} more clues")
        
        # print(f"\n  Solution box:")
        # print(f"    Houses: {parsed['solution'][0].tolist()}")
        # for attr_idx in range(1, len(parsed['solution'])):
        #     print(f"    Attr {attr_idx-1}: {parsed['solution'][attr_idx].tolist()}")
        
        # print(f"\n  Solve order (first 10):")
        # print(f"    {parsed['solve_order'][:10]}")
        
        # Show sequence representation
        result = zebra_to_sequence(raw_data[i], vocab_map)
        seq = result['input_ids']
        loss_mask = result['loss_mask']
        solution_start = result['solution_start']
        
        print(f"\n  Sequence (length {len(seq)}):")
        print(f"    Solution starts at index: {solution_start}")
        print(f"    Encoded tokens (full): {seq.tolist()}")
        print(f"    Loss mask (full): {loss_mask.tolist()}")
        
        # Decode full sequence
        decoded = [id_to_token.get(int(t), f"?{t}") for t in seq]
        print(f"    Decoded tokens (full): {decoded}")
        # Show which tokens are trained on
        train_mask = ["T" if m == 1 else "_" for m in loss_mask]
        print(f"    Training mask: {''.join(train_mask)}")
        print()
    
    print(f"{'='*70}\n")
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Inspect synthetic puzzle datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect sudoku puzzle data
  python -m synthetic_data.inspect_data sudoku-puzzle /path/to/sudoku-train-data.npy --num 3
  
  # Inspect zebra puzzle data  
  python -m synthetic_data.inspect_data zebra /path/to/zebra-train-data.pkl --num 5
  
  # Test loading as training sequences (with sequence length)
  python -m synthetic_data.inspect_data sudoku-puzzle /path/to/sudoku-train-data.npy --test-load --seq-len 200
"""
    )
    
    parser.add_argument(
        "dataset",
        choices=["sudoku-puzzle", "zebra"],
        help="Type of dataset to inspect"
    )
    parser.add_argument(
        "data_path",
        help="Path to the data file (.npy for sudoku-puzzle, .pkl for zebra)"
    )
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=5,
        help="Number of examples to display (default: 5)"
    )
    parser.add_argument(
        "--test-load",
        action="store_true",
        help="Test loading data as training sequences"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Sequence length for test load (default: 256)"
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=100,
        help="Dataset size for test load (default: 100)"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "sudoku-puzzle":
        from synthetic_data.sudoku_puzzle import generate_synthetic_data
        
        # print(f"\nInspecting Sudoku Puzzle data from: {args.data_path}")
        # inspect_sudoku_puzzle(args.data_path, num_examples=args.num)
        
        if args.test_load:
            print("\n" + "="*70)
            print("Testing generate_synthetic_data()...")
            print("="*70)
            
            config = {
                "vocab_size": 13,
                "data_path": args.data_path,
                "include_puzzle": True,
            }
            
            result = generate_synthetic_data(
                args.dataset_size, 
                args.seq_len, 
                config
            )
            
            input_ids = result['input_ids']
            loss_mask = result['loss_mask']
            
            print(f"\nInput IDs shape: {input_ids.shape}")
            print(f"Loss mask shape: {loss_mask.shape}")
            print(f"Dataset dtype: {input_ids.dtype}")
            print(f"\nFirst 3 sequences:")
            for i in range(min(3, len(input_ids))):
                print(f"  [{i}] input_ids: {input_ids[i, :].tolist()}...")
                print(f"  [{i}] loss_mask: {loss_mask[i, :].tolist()}...")
    
    elif args.dataset == "zebra":
        from synthetic_data.zebra import generate_synthetic_data
        
        # print(f"\nInspecting Zebra Puzzle data from: {args.data_path}")
        # inspect_zebra(args.data_path, num_examples=args.num)
        
        if args.test_load:
            print("\n" + "="*70)
            print("Testing generate_synthetic_data()...")
            print("="*70)
            
            # vocab_size is automatically determined by scanning the dataset
            config = {
                "data_path": args.data_path,
            }
            
            result = generate_synthetic_data(
                args.dataset_size,
                args.seq_len,
                config
            )
            
            input_ids = result['input_ids']
            loss_mask = result['loss_mask']
            
            print(f"\nInput IDs shape: {input_ids.shape}")
            print(f"Loss mask shape: {loss_mask.shape}")
            print(f"Dataset dtype: {input_ids.dtype}")
            print(f"\nFirst 3 sequences:")
            for i in range(min(3, len(input_ids))):
                print(f"  [{i}] input_ids: {input_ids[i, :].tolist()}...")
                print(f"  [{i}] loss_mask: {loss_mask[i, :].tolist()}...")
    
    else:
        print(f"Unknown dataset: {args.dataset}")
        sys.exit(1)


if __name__ == "__main__":
    main()

