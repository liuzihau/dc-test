"""
Path finding dataset loader for diffusion-vs-ar data.

Loads graph path finding problems from JSONL files.

Data format:
    JSONL with fields: input, output, reversed
    input: edges separated by '/' followed by '-' and start,end nodes
           Format: "e1/e2/.../eN-start,end" where each edge is "node1,node2"
    output: path as edges like "5,6/6,11/11,7/7,2/2,1"
    reversed: same path but in reverse direction

    Nodes are numbered 0-11 (12 nodes total).

Sequence format:
    [BOS] 5 , 9 / 1 1 , 7 / ... - 5 , 1 [SEP] 5 , 6 / 6 , 1 1 / ... [EOS] [PAD...]

Character-level tokenization for digits, special tokens for separators.

Vocab (19 tokens):
    <PAD>=0, <BOS>=1, <EOS>=2, <SEP>=3, '/'=4, ','=5, '-'=6,
    0-9=7-16, 10=17, 11=18
"""

import numpy as np
import os
import typing
import transformers
from tqdm import tqdm
from collections import defaultdict

from .base import load_jsonl, save_to_cache, load_from_cache, resolve_data_path


# ============================================================================
# Vocabulary Definition
# ============================================================================

# Node numbers go from 0 to 11, so we need tokens for multi-digit numbers
# Approach: character-level for single digits, special tokens for 10 and 11

PATH_VOCAB_TOKENS = [
    '<PAD>',   # 0: padding token
    '<BOS>',   # 1: beginning of sequence
    '<EOS>',   # 2: end of sequence
    '<SEP>',   # 3: separator between input and output
    '/',       # 4: edge/path separator
    ',',       # 5: node separator within edge
    '-',       # 6: separates edges from start,end in input
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 7-16: digits
    '10',      # 17: special token for node 10
    '11',      # 18: special token for node 11
]

# Token ID constants
PAD = 0
BOS = 1
EOS = 2
SEP = 3
SLASH = 4
COMMA = 5
DASH = 6
DIGIT_OFFSET = 7  # digit d has token ID d + 7
NODE_10 = 17
NODE_11 = 18

VOCAB_SIZE = len(PATH_VOCAB_TOKENS)


class PathTokenizer(transformers.PreTrainedTokenizer):
    """Tokenizer for Path finding dataset."""

    def __init__(
        self,
        bos_token='<BOS>',
        eos_token='<EOS>',
        sep_token='<SEP>',
        pad_token='<PAD>',
        **kwargs
    ):
        self._vocab_str_to_int = {token: i for i, token in enumerate(PATH_VOCAB_TOKENS)}
        self._vocab_int_to_str = {i: token for i, token in enumerate(PATH_VOCAB_TOKENS)}

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


def tokenize_node(node_str: str) -> typing.List[int]:
    """Tokenize a node number.

    Args:
        node_str: Node number as string (e.g., "5", "10", "11")

    Returns:
        List of token IDs (single token for 0-11)
    """
    node = int(node_str)
    if node == 10:
        return [NODE_10]
    elif node == 11:
        return [NODE_11]
    elif 0 <= node <= 9:
        return [DIGIT_OFFSET + node]
    else:
        raise ValueError(f"Invalid node number: {node}")


def tokenize_string(s: str) -> typing.List[int]:
    """Tokenize the input or output string.

    Handles special cases for multi-digit numbers (10, 11).

    Args:
        s: String like "5,9/11,7/2,1-5,1" or "5,6/6,11/11,7"

    Returns:
        List of token IDs
    """
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == '/':
            tokens.append(SLASH)
            i += 1
        elif c == ',':
            tokens.append(COMMA)
            i += 1
        elif c == '-':
            tokens.append(DASH)
            i += 1
        elif c.isdigit():
            # Check for multi-digit numbers (10, 11)
            if c == '1' and i + 1 < len(s) and s[i + 1] in '01':
                num_str = s[i:i + 2]
                if num_str == '10':
                    tokens.append(NODE_10)
                elif num_str == '11':
                    tokens.append(NODE_11)
                i += 2
            else:
                tokens.append(DIGIT_OFFSET + int(c))
                i += 1
        else:
            i += 1  # Skip unknown characters
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
        elif t == DASH:
            result.append('-')
        elif t == NODE_10:
            result.append('10')
        elif t == NODE_11:
            result.append('11')
        elif DIGIT_OFFSET <= t <= DIGIT_OFFSET + 9:
            result.append(str(t - DIGIT_OFFSET))
        else:
            result.append(f'?{t}')
    return ''.join(result)


def encode_path(input_str: str, output_str: str) -> dict:
    """Encode an input-output pair as tokens.

    Args:
        input_str: Graph edges and endpoints like "5,9/11,7/...-5,1"
        output_str: Path like "5,6/6,11/11,7/7,2/2,1"

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


def get_cache_filename(dataset_size: int, seq_len: int, split: str) -> str:
    """Generate a cache filename."""
    return f"dvar_path_{split}_ds{dataset_size}_sl{seq_len}.npz"


def generate_synthetic_data(dataset_size, seq_len, config=None):
    """
    Load and tokenize Path finding data from JSONL files.

    Args:
        dataset_size: number of samples to use (None for all)
        seq_len: sequence length (will pad if shorter)
        config: dict with configuration. Key fields:
            - data_path: str, path to JSONL file
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
        raise ValueError("data_path must be provided for path dataset")

    cache_dir = config.get('cache_dir')
    split = config.get('split', 'train')

    # Check cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, get_cache_filename(
            dataset_size if dataset_size else 0, seq_len, split))
        cached = load_from_cache(cache_path, dataset_size, seq_len)
        if cached is not None:
            print(f"[PATH] Loaded from cache: {cache_path}")
            return cached

    # Load data
    data_path = resolve_data_path(data_path)
    print(f"[PATH] Loading from: {data_path}")
    data = load_jsonl(data_path)

    if dataset_size is None:
        dataset_size = len(data)
    else:
        dataset_size = min(dataset_size, len(data))

    print(f"[PATH] Processing {dataset_size} examples")

    # Pre-allocate arrays
    input_ids = np.full((dataset_size, seq_len), fill_value=PAD, dtype=np.int64)
    loss_mask = np.zeros((dataset_size, seq_len), dtype=np.int64)

    max_tokens_seen = 0

    for i in tqdm(range(dataset_size)):
        row = data[i]
        input_str = row['input']
        output_str = row['output']

        # Encode
        encoded = encode_path(input_str, output_str)
        tokens = encoded['tokens']
        solution_start_in_tokens = encoded['solution_start']

        # Build full sequence: [BOS] + tokens + [EOS]
        full_seq = [BOS] + tokens + [EOS]
        total_len = len(full_seq)

        if total_len > seq_len:
            raise ValueError(
                f"Example {i} requires {total_len} tokens but seq_len is {seq_len}. "
                f"Increase seq_len. Input: {input_str}"
            )

        max_tokens_seen = max(max_tokens_seen, total_len)

        # Fill input_ids
        input_ids[i, :total_len] = full_seq

        # Create loss_mask: 0 for input, 1 for output (including EOS and padding)
        solution_start = solution_start_in_tokens + 1  # +1 for BOS
        loss_mask[i, solution_start:] = 1

    print(f"[PATH] Done! Max tokens: {max_tokens_seen}/{seq_len}")

    # Save to cache
    if cache_dir:
        save_to_cache(cache_path, input_ids, loss_mask, {
            'dataset_size': dataset_size,
            'seq_len': seq_len,
            'vocab_size': VOCAB_SIZE,
        })
        print(f"[PATH] Saved to cache: {cache_path}")

    return {
        'input_ids': input_ids,
        'loss_mask': loss_mask,
    }


def token_to_node(t: int) -> typing.Optional[int]:
    """Convert a token ID to a node number."""
    if t == NODE_10:
        return 10
    elif t == NODE_11:
        return 11
    elif DIGIT_OFFSET <= t <= DIGIT_OFFSET + 9:
        return t - DIGIT_OFFSET
    return None


def parse_edges_from_input(input_str: str) -> typing.Tuple[typing.Set[typing.Tuple[int, int]], int, int]:
    """Parse input string into graph edges and start/end nodes.

    Args:
        input_str: String like "5,9/11,7/...-5,1"

    Returns:
        (edges, start, end) where edges is a set of (node1, node2) tuples
    """
    # Split on '-' to separate edges from start,end
    parts = input_str.split('-')
    if len(parts) != 2:
        return set(), -1, -1

    edges_str, endpoints_str = parts

    # Parse edges
    edges = set()
    for edge_str in edges_str.split('/'):
        if ',' in edge_str:
            nodes = edge_str.split(',')
            if len(nodes) == 2:
                try:
                    n1, n2 = int(nodes[0]), int(nodes[1])
                    edges.add((min(n1, n2), max(n1, n2)))  # Normalize edge direction
                except ValueError:
                    pass

    # Parse start,end
    endpoints = endpoints_str.split(',')
    if len(endpoints) != 2:
        return edges, -1, -1

    try:
        start, end = int(endpoints[0]), int(endpoints[1])
    except ValueError:
        return edges, -1, -1

    return edges, start, end


def parse_path_from_tokens(tokens: typing.List[int]) -> typing.List[int]:
    """Parse a path from solution tokens.

    Args:
        tokens: Solution tokens representing path edges

    Returns:
        List of nodes in path order
    """
    path = []
    current_edge = []

    for t in tokens:
        node = token_to_node(t)
        if node is not None:
            current_edge.append(node)
        elif t == SLASH:
            if len(current_edge) == 2:
                if not path:
                    path.append(current_edge[0])
                path.append(current_edge[1])
            current_edge = []
        elif t == COMMA:
            continue  # Skip commas
        elif t in (EOS, PAD):
            break

    # Handle last edge
    if len(current_edge) == 2:
        if not path:
            path.append(current_edge[0])
        path.append(current_edge[1])

    return path


def verify_path(edges: typing.Set[typing.Tuple[int, int]], start: int, end: int,
                path: typing.List[int]) -> typing.Tuple[bool, str]:
    """Verify that a path is valid.

    Args:
        edges: Set of (node1, node2) tuples representing graph edges
        start: Start node
        end: End node
        path: List of nodes in order

    Returns:
        (is_valid, error_message or None)
    """
    if not path:
        return False, "Empty path"

    # Check start and end
    if path[0] != start:
        return False, f"Path starts at {path[0]}, expected {start}"
    if path[-1] != end:
        return False, f"Path ends at {path[-1]}, expected {end}"

    # Check each edge exists
    for i in range(len(path) - 1):
        n1, n2 = path[i], path[i + 1]
        edge = (min(n1, n2), max(n1, n2))
        if edge not in edges:
            return False, f"Edge {n1}-{n2} not in graph"

    return True, None


def extract_solution_from_sequence(seq) -> typing.List[int]:
    """Extract the path from a token sequence.

    Args:
        seq: numpy array or list of token IDs

    Returns:
        List of nodes in path order
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

    # Parse path
    solution_tokens = seq[solution_start:eos_pos]
    return parse_path_from_tokens(solution_tokens)


def extract_input_from_sequence(seq) -> str:
    """Extract the input string from a token sequence.

    Args:
        seq: numpy array or list of token IDs

    Returns:
        Reconstructed input string
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
    Evaluate path finding completions.

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
        # Extract input from ground truth
        input_str = extract_input_from_sequence(gt_seq)
        if not input_str:
            n_parse_failed += 1
            continue

        # Parse graph and endpoints
        edges, start, end = parse_edges_from_input(input_str)
        if start < 0 or end < 0:
            n_parse_failed += 1
            continue

        # Extract predicted path
        pred_path = extract_solution_from_sequence(pred_seq)
        if not pred_path:
            continue

        # Verify path
        is_valid, _ = verify_path(edges, start, end, pred_path)
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
    print("PATH FINDING EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total problems:     {n_total}")
    print(f"Valid paths:        {n_correct} ({results['puzzle_accuracy']*100:.1f}%)")
    print(f"Parse failed:       {n_parse_failed}")
    print("=" * 50)

    return results
