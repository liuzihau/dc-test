#!/usr/bin/env python
"""
Inspect diffusion-vs-ar (dvar) datasets.

Usage:
    python scripts/inspect_dvar.py                    # Show all datasets
    python scripts/inspect_dvar.py --dataset sudoku  # Show specific dataset
    python scripts/inspect_dvar.py --dataset cd4 --n 10  # Show 10 examples

Available datasets: sudoku, cd3, cd4, cd5, 3sat5, 3sat7, 3sat9, path
"""

import argparse
import json
import csv
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_path(filename):
    """Get path to data file."""
    return os.path.join(PROJECT_ROOT, 'diffusion-vs-ar', 'data', filename)


def load_jsonl(path, n=5):
    """Load first n lines from a JSONL file."""
    data = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            data.append(json.loads(line.strip()))
    return data


def load_csv_rows(path, n=5):
    """Load first n rows from a CSV file."""
    data = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            data.append(row)
    return data


def print_separator(char='=', width=80):
    print(char * width)


def print_header(text):
    print()
    print_separator('=')
    print(f"  {text}")
    print_separator('=')
    print()


# =============================================================================
# Sudoku helpers
# =============================================================================

SUDOKU_VOCAB = ['<PAD>', '<BOS>', '<EOS>', '<SEP>', '<M>', '1', '2', '3', '4', '5', '6', '7', '8', '9']
SUDOKU_PAD, SUDOKU_BOS, SUDOKU_EOS, SUDOKU_SEP, SUDOKU_MASK = 0, 1, 2, 3, 4
SUDOKU_DIGIT_OFFSET = 4  # digit d (1-9) -> d + 4


def sudoku_char_to_token(c):
    if c == '0':
        return SUDOKU_MASK
    return SUDOKU_DIGIT_OFFSET + int(c)


def sudoku_decode(tokens):
    result = []
    for t in tokens:
        if t < len(SUDOKU_VOCAB):
            result.append(SUDOKU_VOCAB[t])
        else:
            result.append(f'?{t}')
    return ' '.join(result)


def encode_sudoku(puzzle, solution):
    tokens = []
    for c in puzzle:
        tokens.append(sudoku_char_to_token(c))
    tokens.append(SUDOKU_SEP)
    solution_start = len(tokens)
    for c in solution:
        tokens.append(sudoku_char_to_token(c))
    return tokens, solution_start


# =============================================================================
# Countdown helpers
# =============================================================================

COUNTDOWN_VOCAB = ['<PAD>', '<BOS>', '<EOS>', '<SEP>', ',', '=', '+', '-', '*', '/',
                   '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
CD_PAD, CD_BOS, CD_EOS, CD_SEP = 0, 1, 2, 3
CD_COMMA, CD_EQUALS = 4, 5
CD_ADD, CD_SUB, CD_MUL, CD_DIV = 6, 7, 8, 9
CD_DIGIT_OFFSET = 10

CD_CHAR_MAP = {',': CD_COMMA, '=': CD_EQUALS, '+': CD_ADD, '-': CD_SUB, '*': CD_MUL, '/': CD_DIV}


def countdown_tokenize(s):
    tokens = []
    for c in s:
        if c.isdigit():
            tokens.append(CD_DIGIT_OFFSET + int(c))
        elif c in CD_CHAR_MAP:
            tokens.append(CD_CHAR_MAP[c])
    return tokens


def countdown_decode(tokens):
    result = []
    for t in tokens:
        if t == CD_PAD:
            break
        elif t < len(COUNTDOWN_VOCAB):
            result.append(COUNTDOWN_VOCAB[t])
        else:
            result.append(f'?{t}')
    return ''.join(result)


def encode_countdown(input_str, output_str):
    tokens = countdown_tokenize(input_str)
    tokens.append(CD_SEP)
    solution_start = len(tokens)
    tokens.extend(countdown_tokenize(output_str))
    return tokens, solution_start


# =============================================================================
# 3-SAT helpers
# =============================================================================

SAT3_VOCAB = ['<PAD>', '<BOS>', '<EOS>', '<SEP>', '/', ',',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
SAT3_PAD, SAT3_BOS, SAT3_EOS, SAT3_SEP = 0, 1, 2, 3
SAT3_SLASH, SAT3_COMMA = 4, 5
SAT3_DIGIT_OFFSET = 6
SAT3_LETTER_OFFSET = 16

SAT3_CHAR_MAP = {'/': SAT3_SLASH, ',': SAT3_COMMA}
for d in range(10):
    SAT3_CHAR_MAP[str(d)] = SAT3_DIGIT_OFFSET + d
for i, c in enumerate(['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']):
    SAT3_CHAR_MAP[c] = SAT3_LETTER_OFFSET + i


def sat3_tokenize(s):
    tokens = []
    for c in s:
        if c in SAT3_CHAR_MAP:
            tokens.append(SAT3_CHAR_MAP[c])
    return tokens


def sat3_decode(tokens):
    result = []
    for t in tokens:
        if t == SAT3_PAD:
            break
        elif t < len(SAT3_VOCAB):
            result.append(SAT3_VOCAB[t])
        else:
            result.append(f'?{t}')
    return ''.join(result)


def encode_sat3(input_str, output_str):
    tokens = sat3_tokenize(input_str)
    tokens.append(SAT3_SEP)
    solution_start = len(tokens)
    tokens.extend(sat3_tokenize(output_str))
    return tokens, solution_start


# =============================================================================
# Path helpers
# =============================================================================

PATH_VOCAB = ['<PAD>', '<BOS>', '<EOS>', '<SEP>', '/', ',', '-',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
PATH_PAD, PATH_BOS, PATH_EOS, PATH_SEP = 0, 1, 2, 3
PATH_SLASH, PATH_COMMA, PATH_DASH = 4, 5, 6
PATH_DIGIT_OFFSET = 7
PATH_NODE_10, PATH_NODE_11 = 17, 18


def path_tokenize(s):
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == '/':
            tokens.append(PATH_SLASH)
            i += 1
        elif c == ',':
            tokens.append(PATH_COMMA)
            i += 1
        elif c == '-':
            tokens.append(PATH_DASH)
            i += 1
        elif c.isdigit():
            if c == '1' and i + 1 < len(s) and s[i + 1] in '01':
                if s[i:i+2] == '10':
                    tokens.append(PATH_NODE_10)
                else:
                    tokens.append(PATH_NODE_11)
                i += 2
            else:
                tokens.append(PATH_DIGIT_OFFSET + int(c))
                i += 1
        else:
            i += 1
    return tokens


def path_decode(tokens):
    result = []
    for t in tokens:
        if t == PATH_PAD:
            break
        elif t < len(PATH_VOCAB):
            result.append(PATH_VOCAB[t])
        else:
            result.append(f'?{t}')
    return ''.join(result)


def encode_path(input_str, output_str):
    tokens = path_tokenize(input_str)
    tokens.append(PATH_SEP)
    solution_start = len(tokens)
    tokens.extend(path_tokenize(output_str))
    return tokens, solution_start


# =============================================================================
# Inspection functions
# =============================================================================

def inspect_sudoku(n=3):
    """Inspect the sudoku dataset."""
    print_header("SUDOKU DATASET (dvar-sudoku)")

    print("Format: 9x9 Sudoku puzzle-solution pairs")
    print("Vocab:  14 tokens - PAD, BOS, EOS, SEP, <M> (empty), 1-9")
    print("Sequence: [BOS] puzzle(81) [SEP] solution(81) [EOS]")
    print()

    data_path = get_data_path('sudoku_test.csv')
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    data = load_csv_rows(data_path, n)

    for i, row in enumerate(data):
        puzzle = row['quizzes']
        solution = row['solutions']

        print(f"--- Example {i+1} ---")
        print(f"Puzzle:   {puzzle[:27]}...")
        print(f"          (81 chars, '0' = empty cell)")
        print(f"Solution: {solution[:27]}...")
        print()

        # Show as grid
        print("Puzzle grid:")
        for r in range(9):
            row_str = puzzle[r*9:(r+1)*9]
            print(f"  {' '.join(c if c != '0' else '.' for c in row_str)}")
        print()

        # Tokenize
        tokens, solution_start = encode_sudoku(puzzle, solution)
        full_tokens = [SUDOKU_BOS] + tokens + [SUDOKU_EOS]

        print(f"Token count: {len(full_tokens)} (BOS + 81 + SEP + 81 + EOS = 165)")
        print(f"Solution starts at position: {solution_start + 1} (after BOS)")

        # Decode back
        decoded = sudoku_decode(full_tokens)
        print(f"Decoded (first 60 tokens):")
        print(f"  {' '.join(decoded.split()[:60])}...")
        print()


def inspect_countdown(variant='cd4', n=3):
    """Inspect a countdown dataset."""
    print_header(f"COUNTDOWN DATASET (dvar-{variant})")

    n_inputs = {'cd3': 3, 'cd4': 4, 'cd5': 5}[variant]
    n_ops = n_inputs - 1

    print(f"Format: {n_inputs} numbers + target, {n_ops} arithmetic operations")
    print("Vocab:  20 tokens - PAD, BOS, EOS, SEP, comma, equals, +-*/, 0-9")
    print("Sequence: [BOS] nums,target [SEP] arithmetic_chain [EOS]")
    print()

    data_path = get_data_path(f'{variant}_test.jsonl')
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    data = load_jsonl(data_path, n)

    for i, row in enumerate(data):
        input_str = row['input']
        output_str = row['output']

        print(f"--- Example {i+1} ---")
        print(f"Input:  {input_str}")
        print(f"        (numbers separated by comma, last is target)")
        print(f"Output: {output_str}")
        print(f"        (arithmetic chain to reach target)")
        print()

        # Parse to show clearly
        nums = input_str.split(',')
        print(f"Given numbers: {', '.join(nums[:-1])}")
        print(f"Target: {nums[-1]}")
        print()

        # Tokenize
        tokens, solution_start = encode_countdown(input_str, output_str)
        full_tokens = [CD_BOS] + tokens + [CD_EOS]

        print(f"Token count: {len(full_tokens)}")
        print(f"Solution starts at position: {solution_start + 1}")

        # Decode back
        decoded = countdown_decode(full_tokens)
        print(f"Detokenized: {decoded}")
        print()


def inspect_sat3(variant='3sat7', n=3):
    """Inspect a 3-SAT dataset."""
    print_header(f"3-SAT DATASET (dvar-{variant})")

    n_vars = int(variant.replace('3sat', ''))

    print(f"Format: 3-SAT clauses, {n_vars} true variables in solution")
    print("Vocab:  25 tokens - PAD, BOS, EOS, SEP, /, comma, 0-9, b-j")
    print("Sequence: [BOS] clauses [SEP] true_vars [EOS]")
    print()

    data_path = get_data_path(f'{variant}_test.jsonl')
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    data = load_jsonl(data_path, n)

    for i, row in enumerate(data):
        input_str = row['input']
        output_str = row['output']

        print(f"--- Example {i+1} ---")
        print(f"Input:  {input_str[:60]}{'...' if len(input_str) > 60 else ''}")
        print(f"        (clauses separated by '/', literals by ',')")
        print(f"Output: {output_str}")
        print(f"        (variables set to true)")
        print()

        # Parse clauses
        clauses = input_str.split('/')
        print(f"Number of clauses: {len(clauses)}")
        print(f"First 3 clauses: {clauses[:3]}")
        print(f"True variables: {output_str.split(',')}")
        print()

        # Tokenize
        tokens, solution_start = encode_sat3(input_str, output_str)
        full_tokens = [SAT3_BOS] + tokens + [SAT3_EOS]

        print(f"Token count: {len(full_tokens)}")
        print(f"Solution starts at position: {solution_start + 1}")

        # Decode back
        decoded = sat3_decode(full_tokens)
        print(f"Detokenized: {decoded[:70]}{'...' if len(decoded) > 70 else ''}")
        print()


def inspect_path(n=3):
    """Inspect the path finding dataset."""
    print_header("PATH FINDING DATASET (dvar-path)")

    print("Format: Graph edges + start/end nodes, path as edge sequence")
    print("Vocab:  19 tokens - PAD, BOS, EOS, SEP, /, comma, -, 0-9, 10, 11")
    print("Sequence: [BOS] edges-start,end [SEP] path_edges [EOS]")
    print()

    data_path = get_data_path('path_test.jsonl')
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    data = load_jsonl(data_path, n)

    for i, row in enumerate(data):
        input_str = row['input']
        output_str = row['output']

        print(f"--- Example {i+1} ---")
        print(f"Input:  {input_str[:60]}{'...' if len(input_str) > 60 else ''}")
        print(f"        (edges separated by '/', '-' before start,end)")
        print(f"Output: {output_str}")
        print(f"        (path as edge sequence)")
        print()

        # Parse
        parts = input_str.split('-')
        edges_part = parts[0]
        endpoints = parts[1] if len(parts) > 1 else ''

        edges = edges_part.split('/')
        print(f"Number of edges: {len(edges)}")
        print(f"Start,End: {endpoints}")
        print(f"Path edges: {output_str.split('/')}")
        print()

        # Tokenize
        tokens, solution_start = encode_path(input_str, output_str)
        full_tokens = [PATH_BOS] + tokens + [PATH_EOS]

        print(f"Token count: {len(full_tokens)}")
        print(f"Solution starts at position: {solution_start + 1}")

        # Decode back
        decoded = path_decode(full_tokens)
        print(f"Detokenized: {decoded[:70]}{'...' if len(decoded) > 70 else ''}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Inspect dvar datasets')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        choices=['sudoku', 'cd3', 'cd4', 'cd5', '3sat5', '3sat7', '3sat9', 'path', 'all'],
                        help='Dataset to inspect (default: show summary of all)')
    parser.add_argument('--n', '-n', type=int, default=3,
                        help='Number of examples to show (default: 3)')

    args = parser.parse_args()

    if args.dataset is None or args.dataset == 'all':
        # Show all datasets with fewer examples
        n = min(args.n, 2) if args.dataset is None else args.n
        inspect_sudoku(n)
        inspect_countdown('cd4', n)
        inspect_sat3('3sat7', n)
        inspect_path(n)
    elif args.dataset == 'sudoku':
        inspect_sudoku(args.n)
    elif args.dataset.startswith('cd'):
        inspect_countdown(args.dataset, args.n)
    elif args.dataset.startswith('3sat'):
        inspect_sat3(args.dataset, args.n)
    elif args.dataset == 'path':
        inspect_path(args.n)


if __name__ == '__main__':
    main()
