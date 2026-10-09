"""
Countdown dataset loader for diffusion-vs-ar data.

Loads arithmetic chain puzzles from JSONL files (cd3, cd4, cd5).

Data format:
    JSONL with fields: input, output
    input: comma-separated numbers (last number is the target)
    output: arithmetic chain like "90+95=185,11*185=2035,2035/37=55"

    cd3: 4 numbers (3 input + target), 2 operations
    cd4: 5 numbers (4 input + target), 3 operations
    cd5: 6 numbers (5 input + target), 4 operations

Sequence format:
    [BOS] 9 0 , 1 1 , 3 7 , 9 5 , 5 5 [SEP] 9 0 + 9 5 = 1 8 5 , ... [EOS] [PAD...]

Character-level tokenization (each digit is a token).

Vocab (20 tokens):
    <PAD>=0, <BOS>=1, <EOS>=2, <SEP>=3, ','=4, '='=5,
    +,-,*,/=6-9, 0-9=10-19
"""

import numpy as np
import os
import typing
import transformers
from tqdm import tqdm

from .base import load_jsonl, save_to_cache, load_from_cache, resolve_data_path


# ============================================================================
# Vocabulary Definition
# ============================================================================

COUNTDOWN_VOCAB_TOKENS = [
    '<PAD>',   # 0: padding token
    '<BOS>',   # 1: beginning of sequence
    '<EOS>',   # 2: end of sequence
    '<SEP>',   # 3: separator between input and output
    ',',       # 4: comma (separates numbers and steps)
    '=',       # 5: equals sign
    '+',       # 6: addition
    '-',       # 7: subtraction
    '*',       # 8: multiplication
    '/',       # 9: division
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 10-19: digits
]

# Token ID constants
PAD = 0
BOS = 1
EOS = 2
SEP = 3
COMMA = 4
EQUALS = 5
ADD = 6
SUB = 7
MUL = 8
DIV = 9
DIGIT_OFFSET = 10  # digit d has token ID d + 10

VOCAB_SIZE = len(COUNTDOWN_VOCAB_TOKENS)

# Operator mappings
CHAR_TO_TOKEN = {
    ',': COMMA,
    '=': EQUALS,
    '+': ADD,
    '-': SUB,
    '*': MUL,
    '/': DIV,
}

OP_TO_FUNC = {
    ADD: lambda a, b: a + b,
    SUB: lambda a, b: a - b,
    MUL: lambda a, b: a * b,
    DIV: lambda a, b: a / b if b != 0 else None,
}


class CountdownTokenizer(transformers.PreTrainedTokenizer):
    """Tokenizer for Countdown dataset."""

    def __init__(
        self,
        bos_token='<BOS>',
        eos_token='<EOS>',
        sep_token='<SEP>',
        pad_token='<PAD>',
        **kwargs
    ):
        self._vocab_str_to_int = {token: i for i, token in enumerate(COUNTDOWN_VOCAB_TOKENS)}
        self._vocab_int_to_str = {i: token for i, token in enumerate(COUNTDOWN_VOCAB_TOKENS)}

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


def tokenize_string(s: str) -> typing.List[int]:
    """Tokenize a string character-by-character.

    Args:
        s: String like "90,11,37,95,55" or "90+95=185,11*185=2035"

    Returns:
        List of token IDs
    """
    tokens = []
    for c in s:
        if c.isdigit():
            tokens.append(DIGIT_OFFSET + int(c))
        elif c in CHAR_TO_TOKEN:
            tokens.append(CHAR_TO_TOKEN[c])
        # Skip other characters (shouldn't be any)
    return tokens


def detokenize(tokens: typing.List[int]) -> str:
    """Convert token IDs back to a string."""
    result = []
    for t in tokens:
        if t == PAD:
            break
        elif t == BOS:
            result.append('[BOS]')
        elif t == EOS:
            result.append('[EOS]')
        elif t == SEP:
            result.append('|')
        elif t == COMMA:
            result.append(',')
        elif t == EQUALS:
            result.append('=')
        elif t == ADD:
            result.append('+')
        elif t == SUB:
            result.append('-')
        elif t == MUL:
            result.append('*')
        elif t == DIV:
            result.append('/')
        elif DIGIT_OFFSET <= t <= DIGIT_OFFSET + 9:
            result.append(str(t - DIGIT_OFFSET))
        else:
            result.append(f'?{t}')
    return ''.join(result)


def encode_countdown(input_str: str, output_str: str) -> dict:
    """Encode an input-output pair as tokens.

    Args:
        input_str: Comma-separated numbers like "90,11,37,95,55"
        output_str: Arithmetic chain like "90+95=185,11*185=2035,2035/37=55"

    Returns:
        dict with:
            - 'tokens': list of token IDs (without BOS/EOS)
            - 'solution_start': index where solution begins
    """
    tokens = []

    # Encode input
    tokens.extend(tokenize_string(input_str))
    tokens.append(SEP)

    solution_start = len(tokens)

    # Encode output
    tokens.extend(tokenize_string(output_str))

    return {
        'tokens': tokens,
        'solution_start': solution_start,
    }


def get_cache_filename(dataset_size: int, seq_len: int, variant: str, split: str) -> str:
    """Generate a cache filename."""
    return f"dvar_countdown_{variant}_{split}_ds{dataset_size}_sl{seq_len}.npz"


def generate_synthetic_data(dataset_size, seq_len, config=None):
    """
    Load and tokenize Countdown data from JSONL files.

    Args:
        dataset_size: number of samples to use (None for all)
        seq_len: sequence length (will pad if shorter)
        config: dict with configuration. Key fields:
            - data_path: str, path to JSONL file
            - variant: str, 'cd3', 'cd4', or 'cd5'
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
        raise ValueError("data_path must be provided for countdown dataset")

    variant = config.get('variant', 'cd4')
    cache_dir = config.get('cache_dir')
    split = config.get('split', 'train')

    # Check cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, get_cache_filename(
            dataset_size if dataset_size else 0, seq_len, variant, split))
        cached = load_from_cache(cache_path, dataset_size, seq_len)
        if cached is not None:
            print(f"[COUNTDOWN] Loaded from cache: {cache_path}")
            return cached

    # Load data
    data_path = resolve_data_path(data_path)
    print(f"[COUNTDOWN] Loading from: {data_path}")
    data = load_jsonl(data_path)

    if dataset_size is None:
        dataset_size = len(data)
    else:
        dataset_size = min(dataset_size, len(data))

    print(f"[COUNTDOWN] Processing {dataset_size} examples ({variant})")

    # Pre-allocate arrays
    input_ids = np.full((dataset_size, seq_len), fill_value=PAD, dtype=np.int64)
    loss_mask = np.zeros((dataset_size, seq_len), dtype=np.int64)

    max_tokens_seen = 0

    for i in tqdm(range(dataset_size)):
        row = data[i]
        input_str = row['input']
        output_str = row['output']

        # Encode
        encoded = encode_countdown(input_str, output_str)
        tokens = encoded['tokens']
        solution_start_in_tokens = encoded['solution_start']

        # Build full sequence: [BOS] + tokens + [EOS]
        full_seq = [BOS] + tokens + [EOS]
        total_len = len(full_seq)

        if total_len > seq_len:
            raise ValueError(
                f"Example {i} requires {total_len} tokens but seq_len is {seq_len}. "
                f"Increase seq_len. Input: {input_str}, Output: {output_str}"
            )

        max_tokens_seen = max(max_tokens_seen, total_len)

        # Fill input_ids
        input_ids[i, :total_len] = full_seq

        # Create loss_mask: 0 for input, 1 for output (including EOS and padding)
        solution_start = solution_start_in_tokens + 1  # +1 for BOS
        loss_mask[i, solution_start:] = 1

    print(f"[COUNTDOWN] Done! Max tokens: {max_tokens_seen}/{seq_len}")

    # Save to cache
    if cache_dir:
        save_to_cache(cache_path, input_ids, loss_mask, {
            'dataset_size': dataset_size,
            'seq_len': seq_len,
            'vocab_size': VOCAB_SIZE,
            'variant': variant,
        })
        print(f"[COUNTDOWN] Saved to cache: {cache_path}")

    return {
        'input_ids': input_ids,
        'loss_mask': loss_mask,
    }


def parse_number_from_tokens(tokens: typing.List[int], start: int) -> typing.Tuple[int, int]:
    """Parse a number from tokens starting at given index.

    Returns:
        (value, end_index) or (None, start) if no number found
    """
    digits = []
    i = start
    while i < len(tokens):
        t = tokens[i]
        if DIGIT_OFFSET <= t <= DIGIT_OFFSET + 9:
            digits.append(str(t - DIGIT_OFFSET))
            i += 1
        else:
            break

    if not digits:
        return None, start

    return int(''.join(digits)), i


def verify_solution(input_numbers: typing.List[int], target: int, solution_tokens: typing.List[int]) -> typing.Tuple[bool, str]:
    """Verify that a solution is correct.

    Args:
        input_numbers: List of input numbers (not including target)
        target: Target number
        solution_tokens: Tokens representing the solution

    Returns:
        (is_valid, error_message or None)
    """
    # Parse the solution into steps
    # Each step: num1 [OP] num2 = result
    steps = []
    i = 0

    while i < len(solution_tokens):
        # Parse first operand
        num1, i = parse_number_from_tokens(solution_tokens, i)
        if num1 is None:
            break

        # Parse operator
        if i >= len(solution_tokens):
            return False, "Missing operator"
        op = solution_tokens[i]
        if op not in (ADD, SUB, MUL, DIV):
            return False, f"Invalid operator at position {i}"
        i += 1

        # Parse second operand
        num2, i = parse_number_from_tokens(solution_tokens, i)
        if num2 is None:
            return False, "Missing second operand"

        # Parse equals
        if i >= len(solution_tokens) or solution_tokens[i] != EQUALS:
            return False, "Missing equals sign"
        i += 1

        # Parse result
        result, i = parse_number_from_tokens(solution_tokens, i)
        if result is None:
            return False, "Missing result"

        steps.append((num1, op, num2, result))

        # Skip comma if present
        if i < len(solution_tokens) and solution_tokens[i] == COMMA:
            i += 1

    if not steps:
        return False, "No steps found"

    # Verify arithmetic for each step
    for step_idx, (num1, op, num2, claimed_result) in enumerate(steps):
        func = OP_TO_FUNC.get(op)
        if func is None:
            return False, f"Invalid operator in step {step_idx}"

        actual_result = func(num1, num2)
        if actual_result is None:
            return False, f"Invalid operation in step {step_idx}"

        # Check if result is integer
        if actual_result != int(actual_result):
            return False, f"Non-integer result in step {step_idx}"

        if int(actual_result) != claimed_result:
            return False, f"Wrong result in step {step_idx}: {num1} op {num2} = {claimed_result}, expected {int(actual_result)}"

    # Check final result matches target
    final_result = steps[-1][3]
    if final_result != target:
        return False, f"Final result {final_result} != target {target}"

    return True, None


def extract_solution_from_sequence(seq) -> typing.List[int]:
    """Extract solution tokens from a sequence.

    Args:
        seq: numpy array or list of token IDs

    Returns:
        List of solution tokens (between SEP and EOS)
    """
    if hasattr(seq, 'tolist'):
        seq = seq.tolist()

    # Find SEP position
    if SEP not in seq:
        return []

    sep_pos = seq.index(SEP)

    # Find EOS position
    solution_start = sep_pos + 1
    if EOS in seq[solution_start:]:
        eos_pos = seq.index(EOS, solution_start)
    elif PAD in seq[solution_start:]:
        eos_pos = seq.index(PAD, solution_start)
    else:
        eos_pos = len(seq)

    return seq[solution_start:eos_pos]


def extract_input_from_sequence(seq) -> typing.Tuple[typing.List[int], int]:
    """Extract input numbers and target from a sequence.

    Args:
        seq: numpy array or list of token IDs

    Returns:
        (input_numbers, target) or ([], None) if parsing fails
    """
    if hasattr(seq, 'tolist'):
        seq = seq.tolist()

    # Find BOS and SEP positions
    if BOS not in seq or SEP not in seq:
        return [], None

    bos_pos = seq.index(BOS)
    sep_pos = seq.index(SEP)

    # Parse numbers between BOS and SEP
    input_tokens = seq[bos_pos + 1:sep_pos]
    numbers = []
    i = 0

    while i < len(input_tokens):
        num, i = parse_number_from_tokens(input_tokens, i)
        if num is not None:
            numbers.append(num)
        # Skip comma
        if i < len(input_tokens) and input_tokens[i] == COMMA:
            i += 1
        elif num is None:
            i += 1

    if len(numbers) < 2:
        return [], None

    # Last number is target, rest are inputs
    return numbers[:-1], numbers[-1]


def evaluate_completions(predicted_ids, ground_truth_ids):
    """
    Evaluate countdown completions.

    Args:
        predicted_ids: numpy array of shape (n_samples, seq_len)
        ground_truth_ids: numpy array of shape (n_samples, seq_len)

    Returns:
        dict with evaluation metrics
    """
    n_total = len(predicted_ids)
    n_correct = 0
    n_parse_failed = 0

    for pred_seq, gt_seq in zip(predicted_ids, ground_truth_ids):
        # Extract input numbers and target from ground truth
        input_numbers, target = extract_input_from_sequence(gt_seq)
        if not input_numbers or target is None:
            n_parse_failed += 1
            continue

        # Extract solution from prediction
        solution_tokens = extract_solution_from_sequence(pred_seq)
        if not solution_tokens:
            continue

        # Verify
        is_valid, _ = verify_solution(input_numbers, target, solution_tokens)
        if is_valid:
            n_correct += 1

    results = {
        'puzzle_accuracy': n_correct / n_total if n_total > 0 else 0.0,
        'n_correct': n_correct,
        'n_parse_failed': n_parse_failed,
        'n_total': n_total,
    }

    print()
    print("=" * 50)
    print("COUNTDOWN EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total puzzles:      {n_total}")
    print(f"Correct solutions:  {n_correct} ({results['puzzle_accuracy']*100:.1f}%)")
    print(f"Parse failed:       {n_parse_failed}")
    print("=" * 50)

    return results
