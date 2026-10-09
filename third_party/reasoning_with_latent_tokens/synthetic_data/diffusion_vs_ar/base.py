"""
Base utilities for diffusion-vs-ar dataset loaders.

Provides common functionality for loading and tokenizing data from the
diffusion-vs-ar repository.
"""

import csv
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional


def resolve_data_path(path: str) -> str:
    """Resolve a data path relative to the project root.

    Handles the case where Hydra changes the working directory.

    Args:
        path: Path that may be relative to project root

    Returns:
        Absolute path
    """
    if os.path.isabs(path):
        return path

    # If the path exists from current directory, use it
    if os.path.exists(path):
        return os.path.abspath(path)

    # Try to get original working directory from Hydra
    try:
        import hydra
        original_cwd = hydra.utils.get_original_cwd()
        resolved = os.path.join(original_cwd, path)
        if os.path.exists(resolved):
            return resolved
    except (ImportError, AttributeError, ValueError):
        pass

    # Fallback: find project root by looking for known markers
    current = os.getcwd()
    for _ in range(10):  # Limit search depth
        candidate = os.path.join(current, path)
        if os.path.exists(candidate):
            return candidate
        # Check for project markers
        if os.path.exists(os.path.join(current, 'main.py')) or \
           os.path.exists(os.path.join(current, 'diffusion-vs-ar')):
            candidate = os.path.join(current, path)
            if os.path.exists(candidate):
                return candidate
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    # Last resort: return as-is
    return path


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file and return a list of dictionaries.

    Args:
        path: Path to the JSONL file

    Returns:
        List of dictionaries, one per line
    """
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_csv(path: str, has_header: bool = True) -> List[Dict]:
    """Load a CSV file and return a list of dictionaries.

    Args:
        path: Path to the CSV file
        has_header: Whether the CSV has a header row

    Returns:
        List of dictionaries with column names as keys
    """
    data = []
    with open(path, 'r') as f:
        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        else:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
    return data


def pad_sequence(tokens: List[int], seq_len: int, pad_token: int = 0) -> np.ndarray:
    """Pad a token sequence to fixed length.

    Args:
        tokens: List of token IDs
        seq_len: Target sequence length
        pad_token: Token ID to use for padding

    Returns:
        Numpy array of shape (seq_len,)

    Raises:
        ValueError: If tokens exceed seq_len
    """
    if len(tokens) > seq_len:
        raise ValueError(f"Token sequence length {len(tokens)} exceeds seq_len {seq_len}")

    result = np.full(seq_len, pad_token, dtype=np.int64)
    result[:len(tokens)] = tokens
    return result


def create_loss_mask(seq_len: int, solution_start: int, contiguous: bool = True) -> np.ndarray:
    """Create loss mask for a sequence.

    Args:
        seq_len: Total sequence length
        solution_start: Index where solution tokens begin
        contiguous: If True, mask is 1 from solution_start to end (including padding)
                   If False, only mark actual solution tokens (would need end index)

    Returns:
        Numpy array of shape (seq_len,) with 0 for problem tokens, 1 for solution tokens
    """
    mask = np.zeros(seq_len, dtype=np.int64)
    if contiguous:
        mask[solution_start:] = 1
    return mask


class BaseTokenizer:
    """Base class for diffusion-vs-ar tokenizers.

    Subclasses should define:
        - VOCAB_TOKENS: List of token strings
        - Special token constants (PAD, BOS, EOS, SEP)
    """

    VOCAB_TOKENS: List[str] = []
    PAD = 0
    BOS = 1
    EOS = 2
    SEP = 3

    def __init__(self):
        self._vocab_str_to_int = {token: i for i, token in enumerate(self.VOCAB_TOKENS)}
        self._vocab_int_to_str = {i: token for i, token in enumerate(self.VOCAB_TOKENS)}

    @property
    def vocab_size(self) -> int:
        return len(self.VOCAB_TOKENS)

    @property
    def pad_token(self) -> str:
        return self.VOCAB_TOKENS[self.PAD]

    @property
    def pad_token_id(self) -> int:
        return self.PAD

    @property
    def bos_token(self) -> str:
        return self.VOCAB_TOKENS[self.BOS]

    @property
    def bos_token_id(self) -> int:
        return self.BOS

    @property
    def eos_token(self) -> str:
        return self.VOCAB_TOKENS[self.EOS]

    @property
    def eos_token_id(self) -> int:
        return self.EOS

    @property
    def sep_token(self) -> str:
        return self.VOCAB_TOKENS[self.SEP]

    @property
    def sep_token_id(self) -> int:
        return self.SEP

    def encode_token(self, token: str) -> int:
        """Encode a single token string to an ID."""
        return self._vocab_str_to_int.get(token, self.PAD)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to a string."""
        return self._vocab_int_to_str.get(token_id, '<UNK>')

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        """Decode a sequence of token IDs to a string."""
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()

        tokens = []
        for tid in token_ids:
            token = self.decode_token(tid)
            if skip_special_tokens and token in ['<PAD>', '<BOS>', '<EOS>', '<SEP>']:
                continue
            tokens.append(token)
        return ' '.join(tokens)

    def batch_decode(self, sequences, skip_special_tokens: bool = False) -> List[str]:
        """Decode a batch of token ID sequences."""
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]

    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary mapping."""
        return self._vocab_str_to_int.copy()


def number_to_digit_tokens(n: int, digit_offset: int, neg_token: Optional[int] = None) -> List[int]:
    """Convert an integer to a list of digit tokens.

    Args:
        n: Integer to convert
        digit_offset: Token ID offset for digit '0'
        neg_token: Optional token ID for negative sign (if None, abs value used)

    Returns:
        List of token IDs representing the number
    """
    if n < 0:
        if neg_token is not None:
            return [neg_token] + [digit_offset + int(d) for d in str(abs(n))]
        else:
            return [digit_offset + int(d) for d in str(abs(n))]
    return [digit_offset + int(d) for d in str(n)]


def digit_tokens_to_number(tokens: List[int], digit_offset: int, neg_token: Optional[int] = None) -> Optional[int]:
    """Convert digit tokens back to an integer.

    Args:
        tokens: List of token IDs
        digit_offset: Token ID offset for digit '0'
        neg_token: Optional token ID for negative sign

    Returns:
        Integer value, or None if parsing fails
    """
    if not tokens:
        return None

    negative = False
    start = 0
    if neg_token is not None and tokens[0] == neg_token:
        negative = True
        start = 1

    if start >= len(tokens):
        return None

    digits = []
    for t in tokens[start:]:
        d = t - digit_offset
        if not (0 <= d <= 9):
            return None
        digits.append(str(d))

    if not digits:
        return None

    try:
        val = int(''.join(digits))
        return -val if negative else val
    except ValueError:
        return None


def save_to_cache(cache_path: str, input_ids: np.ndarray, loss_mask: np.ndarray,
                  metadata: Optional[Dict] = None):
    """Save tokenized data to a cache file.

    Args:
        cache_path: Path to save the cache file
        input_ids: Array of shape (dataset_size, seq_len)
        loss_mask: Array of shape (dataset_size, seq_len)
        metadata: Optional metadata dict to save
    """
    import time
    save_dict = {
        'input_ids': input_ids,
        'loss_mask': loss_mask,
        'created_ts': np.array(time.time()),
    }
    if metadata:
        for k, v in metadata.items():
            if isinstance(v, (int, float)):
                save_dict[k] = np.array(v)
            else:
                save_dict[k] = v
    np.savez_compressed(cache_path, **save_dict)


def load_from_cache(cache_path: str, dataset_size: Optional[int] = None,
                    seq_len: Optional[int] = None) -> Optional[Dict[str, np.ndarray]]:
    """Load tokenized data from a cache file.

    Args:
        cache_path: Path to the cache file
        dataset_size: Expected dataset size (optional validation)
        seq_len: Expected sequence length (optional validation)

    Returns:
        Dict with 'input_ids' and 'loss_mask', or None if cache doesn't exist or is invalid
    """
    if not os.path.exists(cache_path):
        return None

    try:
        data = np.load(cache_path, allow_pickle=True)
        input_ids = data['input_ids']
        loss_mask = data['loss_mask']

        # Validate if requested
        if seq_len is not None and input_ids.shape[1] != seq_len:
            print(f"[CACHE] seq_len mismatch: cache has {input_ids.shape[1]}, requested {seq_len}")
            return None

        # Limit to requested size
        if dataset_size is not None and dataset_size < len(input_ids):
            input_ids = input_ids[:dataset_size]
            loss_mask = loss_mask[:dataset_size]

        return {
            'input_ids': input_ids,
            'loss_mask': loss_mask,
        }
    except Exception as e:
        print(f"[CACHE] Load failed: {e}")
        return None
