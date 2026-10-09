"""
3-SAT dataset loader for diffusion-vs-ar data.

Loads satisfiability problems from JSONL files (3sat5, 3sat7, 3sat9).

Data format:
    JSONL with fields: input, output
    input: clauses separated by '/', each clause has 3 literals separated by ','
           literals are digits 0-9 and letters b-j representing variables
    output: comma-separated variables that are set to true

    3sat5: 5 true variables in output
    3sat7: 7 true variables in output
    3sat9: 9 true variables in output

Sequence format:
    [BOS] c , d , g / 3 , 5 , 6 / ... [SEP] b , c , 3 , e , f , g , h [EOS] [PAD...]

Character-level tokenization.

Vocab (26 tokens):
    <PAD>=0, <BOS>=1, <EOS>=2, <SEP>=3, '/'=4, ','=5,
    0-9=6-15, b-k=16-25
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

# Variables: 0-9 (digits) and b-k (letters, where 'a' is skipped or '1' represents 'a')
# Actually looking at the data, it uses: 0-9 and b-j (or similar letters)
# Let's build a comprehensive vocab

SAT3_VOCAB_TOKENS = [
    '<PAD>',   # 0: padding token
    '<BOS>',   # 1: beginning of sequence
    '<EOS>',   # 2: end of sequence
    '<SEP>',   # 3: separator between input and output
    '/',       # 4: clause separator
    ',',       # 5: literal/variable separator
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 6-15: digits
    'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',       # 16-24: letters (skip 'a' as it might be '1')
]

# Token ID constants
PAD = 0
BOS = 1
EOS = 2
SEP = 3
SLASH = 4
COMMA = 5
DIGIT_OFFSET = 6  # digit d has token ID d + 6
LETTER_OFFSET = 16  # letter 'b' has token ID 16, etc.

VOCAB_SIZE = len(SAT3_VOCAB_TOKENS)

# Build character to token mapping
CHAR_TO_TOKEN = {
    '/': SLASH,
    ',': COMMA,
}
for d in range(10):
    CHAR_TO_TOKEN[str(d)] = DIGIT_OFFSET + d
for i, c in enumerate(['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']):
    CHAR_TO_TOKEN[c] = LETTER_OFFSET + i


class SAT3Tokenizer(transformers.PreTrainedTokenizer):
    """Tokenizer for 3-SAT dataset."""

    def __init__(
        self,
        bos_token='<BOS>',
        eos_token='<EOS>',
        sep_token='<SEP>',
        pad_token='<PAD>',
        **kwargs
    ):
        self._vocab_str_to_int = {token: i for i, token in enumerate(SAT3_VOCAB_TOKENS)}
        self._vocab_int_to_str = {i: token for i, token in enumerate(SAT3_VOCAB_TOKENS)}

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
        s: String like "c,d,g/3,5,6" or "b,c,3,e,f,g,h"

    Returns:
        List of token IDs
    """
    tokens = []
    for c in s:
        if c in CHAR_TO_TOKEN:
            tokens.append(CHAR_TO_TOKEN[c])
        # Skip other characters (spaces, etc.)
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
        elif t == SLASH:
            result.append('/')
        elif t == COMMA:
            result.append(',')
        elif DIGIT_OFFSET <= t <= DIGIT_OFFSET + 9:
            result.append(str(t - DIGIT_OFFSET))
        elif LETTER_OFFSET <= t < VOCAB_SIZE:
            idx = t - LETTER_OFFSET
            letters = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
            if idx < len(letters):
                result.append(letters[idx])
            else:
                result.append(f'?{t}')
        else:
            result.append(f'?{t}')
    return ''.join(result)


def encode_sat3(input_str: str, output_str: str) -> dict:
    """Encode an input-output pair as tokens.

    Args:
        input_str: Clauses like "c,d,g/3,5,6/..."
        output_str: Variables like "b,c,3,e,f,g,h"

    Returns:
        dict with:
            - 'tokens': list of token IDs (without BOS/EOS)
            - 'solution_start': index where solution begins
    """
    tokens = []

    # Encode input (clauses)
    tokens.extend(tokenize_string(input_str))
    tokens.append(SEP)

    solution_start = len(tokens)

    # Encode output (satisfying assignment)
    tokens.extend(tokenize_string(output_str))

    return {
        'tokens': tokens,
        'solution_start': solution_start,
    }


def get_cache_filename(dataset_size: int, seq_len: int, variant: str, split: str) -> str:
    """Generate a cache filename."""
    return f"dvar_sat3_{variant}_{split}_ds{dataset_size}_sl{seq_len}.npz"


def generate_synthetic_data(dataset_size, seq_len, config=None):
    """
    Load and tokenize 3-SAT data from JSONL files.

    Args:
        dataset_size: number of samples to use (None for all)
        seq_len: sequence length (will pad if shorter)
        config: dict with configuration. Key fields:
            - data_path: str, path to JSONL file
            - variant: str, '3sat5', '3sat7', or '3sat9'
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
        raise ValueError("data_path must be provided for 3-SAT dataset")

    variant = config.get('variant', '3sat7')
    cache_dir = config.get('cache_dir')
    split = config.get('split', 'train')

    # Check cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, get_cache_filename(
            dataset_size if dataset_size else 0, seq_len, variant, split))
        cached = load_from_cache(cache_path, dataset_size, seq_len)
        if cached is not None:
            print(f"[3SAT] Loaded from cache: {cache_path}")
            return cached

    # Load data
    data_path = resolve_data_path(data_path)
    print(f"[3SAT] Loading from: {data_path}")
    data = load_jsonl(data_path)

    if dataset_size is None:
        dataset_size = len(data)
    else:
        dataset_size = min(dataset_size, len(data))

    print(f"[3SAT] Processing {dataset_size} examples ({variant})")

    # Pre-allocate arrays
    input_ids = np.full((dataset_size, seq_len), fill_value=PAD, dtype=np.int64)
    loss_mask = np.zeros((dataset_size, seq_len), dtype=np.int64)

    max_tokens_seen = 0

    for i in tqdm(range(dataset_size)):
        row = data[i]
        input_str = row['input']
        output_str = row['output']

        # Encode
        encoded = encode_sat3(input_str, output_str)
        tokens = encoded['tokens']
        solution_start_in_tokens = encoded['solution_start']

        # Build full sequence: [BOS] + tokens + [EOS]
        full_seq = [BOS] + tokens + [EOS]
        total_len = len(full_seq)

        if total_len > seq_len:
            raise ValueError(
                f"Example {i} requires {total_len} tokens but seq_len is {seq_len}. "
                f"Increase seq_len. Input length: {len(input_str)}, Output: {output_str}"
            )

        max_tokens_seen = max(max_tokens_seen, total_len)

        # Fill input_ids
        input_ids[i, :total_len] = full_seq

        # Create loss_mask: 0 for input, 1 for output (including EOS and padding)
        solution_start = solution_start_in_tokens + 1  # +1 for BOS
        loss_mask[i, solution_start:] = 1

    print(f"[3SAT] Done! Max tokens: {max_tokens_seen}/{seq_len}")

    # Save to cache
    if cache_dir:
        save_to_cache(cache_path, input_ids, loss_mask, {
            'dataset_size': dataset_size,
            'seq_len': seq_len,
            'vocab_size': VOCAB_SIZE,
            'variant': variant,
        })
        print(f"[3SAT] Saved to cache: {cache_path}")

    return {
        'input_ids': input_ids,
        'loss_mask': loss_mask,
    }


def token_to_var(t: int) -> typing.Optional[str]:
    """Convert a token ID to a variable name."""
    if DIGIT_OFFSET <= t <= DIGIT_OFFSET + 9:
        return str(t - DIGIT_OFFSET)
    if LETTER_OFFSET <= t < VOCAB_SIZE:
        idx = t - LETTER_OFFSET
        letters = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        if idx < len(letters):
            return letters[idx]
    return None


def parse_clauses(input_str: str) -> typing.List[typing.Set[str]]:
    """Parse input string into list of clause sets.

    Each clause is a set of variable names (positive literals).
    """
    clauses = []
    for clause_str in input_str.split('/'):
        literals = set(clause_str.split(','))
        clauses.append(literals)
    return clauses


def check_satisfiability(clauses: typing.List[typing.Set[str]], true_vars: typing.Set[str]) -> bool:
    """Check if the assignment satisfies all clauses.

    In this 3-SAT format, each clause is satisfied if at least one of its
    literals is in the true_vars set.

    Args:
        clauses: List of clause sets (each clause is a set of variable names)
        true_vars: Set of variables assigned to true

    Returns:
        True if all clauses are satisfied, False otherwise
    """
    for clause in clauses:
        # A clause is satisfied if at least one literal is true
        if not clause.intersection(true_vars):
            return False
    return True


def extract_solution_from_sequence(seq) -> typing.Set[str]:
    """Extract the satisfying assignment from a token sequence.

    Args:
        seq: numpy array or list of token IDs

    Returns:
        Set of variable names assigned to true
    """
    if hasattr(seq, 'tolist'):
        seq = seq.tolist()

    # Find SEP position
    if SEP not in seq:
        return set()

    sep_pos = seq.index(SEP)

    # Find EOS position
    solution_start = sep_pos + 1
    if EOS in seq[solution_start:]:
        eos_pos = seq.index(EOS, solution_start)
    elif PAD in seq[solution_start:]:
        eos_pos = seq.index(PAD, solution_start)
    else:
        eos_pos = len(seq)

    # Parse variables
    true_vars = set()
    for t in seq[solution_start:eos_pos]:
        var = token_to_var(t)
        if var is not None:
            true_vars.add(var)

    return true_vars


def extract_input_from_sequence(seq) -> str:
    """Extract the input clauses string from a token sequence.

    Args:
        seq: numpy array or list of token IDs

    Returns:
        Input string reconstructed from tokens
    """
    if hasattr(seq, 'tolist'):
        seq = seq.tolist()

    # Find BOS and SEP positions
    if BOS not in seq or SEP not in seq:
        return ""

    bos_pos = seq.index(BOS)
    sep_pos = seq.index(SEP)

    # Reconstruct input string
    input_tokens = seq[bos_pos + 1:sep_pos]
    return detokenize(input_tokens).replace('[BOS]', '').replace('[EOS]', '').replace('|', '')


def evaluate_completions(predicted_ids, ground_truth_ids):
    """
    Evaluate 3-SAT completions.

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
        # Extract input (clauses) from ground truth
        input_str = extract_input_from_sequence(gt_seq)
        if not input_str:
            n_parse_failed += 1
            continue

        # Parse clauses
        clauses = parse_clauses(input_str)

        # Extract predicted assignment
        pred_vars = extract_solution_from_sequence(pred_seq)
        if not pred_vars:
            continue

        # Check if assignment satisfies all clauses
        if check_satisfiability(clauses, pred_vars):
            n_correct += 1

    results = {
        'puzzle_accuracy': n_correct / n_total if n_total > 0 else 0.0,
        'n_correct': n_correct,
        'n_parse_failed': n_parse_failed,
        'n_total': n_total,
    }

    print()
    print("=" * 50)
    print("3-SAT EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total problems:     {n_total}")
    print(f"Satisfied:          {n_correct} ({results['puzzle_accuracy']*100:.1f}%)")
    print(f"Parse failed:       {n_parse_failed}")
    print("=" * 50)

    return results
