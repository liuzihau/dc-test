"""
Game of 24 dataset: generates arithmetic puzzles with step-by-step solutions.

Each example consists of:
- Problem: target number followed by N input numbers
- Solution: step-by-step left-to-right computation

The data is fully symbolic (no natural language) for training from scratch.

Sequence format:
    [BOS] target [SEP] n1 [SEP] n2 [SEP] ... [SEP] nN [SEP] 
    step1 [SEP] step2 [SEP] ... [SEP] stepN-1 [EOS] [PAD...]

Each step: operand1 [OP] operand2 [=] result

Loss mask: 0 for problem tokens (BOS through last SEP before solution), 
           1 for solution tokens (steps + EOS)
"""

import numpy as np
import itertools
import typing
from tqdm import tqdm
import transformers


# ============================================================================
# Vocabulary Definition - Single Source of Truth
# ============================================================================

# Hardcoded vocabulary list - defines all tokens and their IDs
GAME24_VOCAB_TOKENS = [
    '<PAD>',   # 0: padding token
    '<BOS>',   # 1: beginning of sequence
    '<EOS>',   # 2: end of sequence
    '<SEP>',   # 3: separator
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 4-13: digits
    '+',       # 14: addition
    '-',       # 15: subtraction  
    '*',       # 16: multiplication
    '/',       # 17: division
    '=',       # 18: equals
    'NEG',     # 19: negative sign (for negative intermediate results)
]

# Token ID constants (derived from vocabulary position)
PAD = 0
BOS = 1
EOS = 2
SEP = 3
# Digits are 4-13 (digit d has token ID d + 4)
DIGIT_OFFSET = 4
ADD = 14
SUB = 15
MUL = 16
DIV = 17
EQUALS = 18
NEG = 19

VOCAB_SIZE = len(GAME24_VOCAB_TOKENS)

# Operator mappings
OPERATORS = [ADD, SUB, MUL, DIV]
OP_TO_STR = {ADD: '+', SUB: '-', MUL: '*', DIV: '/'}
OP_TO_FUNC = {
    ADD: lambda a, b: a + b,
    SUB: lambda a, b: a - b,
    MUL: lambda a, b: a * b,
    DIV: lambda a, b: a / b if b != 0 else None,
}


def build_vocab():
    """
    Build vocabulary mappings for Game of 24.
    
    Returns:
        (vocab_map, id_to_token) tuple
    """
    vocab_map = {token: i for i, token in enumerate(GAME24_VOCAB_TOKENS)}
    id_to_token = {i: token for i, token in enumerate(GAME24_VOCAB_TOKENS)}
    return vocab_map, id_to_token


class Game24Tokenizer(transformers.PreTrainedTokenizer):
    """Tokenizer for Game of 24 dataset.
    
    Uses the vocabulary defined in GAME24_VOCAB_TOKENS.
    """
    
    def __init__(
        self,
        bos_token='<BOS>',
        eos_token='<EOS>',
        sep_token='<SEP>',
        pad_token='<PAD>',
        **kwargs):
        
        # Use build_vocab() as single source of truth
        self._vocab_str_to_int, self._vocab_int_to_str = build_vocab()
        
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            pad_token=pad_token,
            **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return text.strip().split()

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, PAD)  # Return PAD for unknown

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


# ============================================================================
# Token Conversion Utilities
# ============================================================================


def digit_to_token(d):
    """Convert a digit (0-9) to its token ID."""
    return DIGIT_OFFSET + d


def token_to_digit(t):
    """Convert a token ID back to a digit (0-9), or None if not a digit token."""
    if DIGIT_OFFSET <= t < DIGIT_OFFSET + 10:
        return t - DIGIT_OFFSET
    return None


def number_to_tokens(n):
    """Convert an integer to a list of digit tokens.
    
    Handles negative numbers by prepending NEG token.
    
    Args:
        n: integer to convert
        
    Returns:
        list of tokens representing the number
    """
    if n < 0:
        return [NEG] + [digit_to_token(int(d)) for d in str(abs(n))]
    else:
        return [digit_to_token(int(d)) for d in str(n)]


def tokens_to_number(tokens):
    """Convert a list of digit tokens back to an integer.
    
    Args:
        tokens: list of tokens (may start with NEG for negatives)
        
    Returns:
        integer value
    """
    if not tokens:
        return None
    
    negative = False
    start = 0
    if tokens[0] == NEG:
        negative = True
        start = 1
    
    if start >= len(tokens):
        return None
    
    digits = []
    for t in tokens[start:]:
        d = token_to_digit(t)
        if d is None:
            return None
        digits.append(str(d))
    
    if not digits:
        return None
    
    num_str = ''.join(digits)
    try:
        val = int(num_str)
        return -val if negative else val
    except ValueError:
        return None


def evaluate_left_to_right(numbers, operators):
    """Evaluate expression left-to-right, returning intermediate steps.
    
    Args:
        numbers: list of N numbers
        operators: list of N-1 operators
        
    Returns:
        list of (a, op, b, result) tuples for each step, or None if invalid
        (e.g., division by zero or non-integer result)
    """
    if len(operators) != len(numbers) - 1:
        return None
    
    steps = []
    current = numbers[0]
    
    for i, op in enumerate(operators):
        next_num = numbers[i + 1]
        func = OP_TO_FUNC.get(op)
        if func is None:
            return None
        
        result = func(current, next_num)
        
        # Check for invalid operations
        if result is None:
            return None
        
        # For integer-only mode, check if result is integer
        if not isinstance(result, int):
            if result != int(result):
                return None
            result = int(result)
        
        steps.append((current, op, next_num, result))
        current = result
    
    return steps


def find_all_solutions(numbers, target, operators=None, integer_only=True):
    """Find all valid solutions for given numbers and target.
    
    Tries all permutations of numbers and combinations of operators,
    evaluating left-to-right.
    
    Args:
        numbers: list of input numbers
        target: target value to achieve
        operators: list of allowed operators (default: [+, -, *, /])
        integer_only: if True, reject solutions with non-integer intermediates
        
    Returns:
        list of (number_permutation, operator_sequence, steps) tuples
    """
    if operators is None:
        operators = OPERATORS
    
    solutions = []
    n = len(numbers)
    
    for num_perm in itertools.permutations(numbers):
        for ops in itertools.product(operators, repeat=n - 1):
            steps = evaluate_left_to_right(list(num_perm), list(ops))
            
            if steps is None:
                continue
            
            # Check if final result matches target
            final_result = steps[-1][3]
            if abs(final_result - target) < 1e-10:
                solutions.append((list(num_perm), list(ops), steps))
    
    return solutions


def encode_problem_and_solution(numbers, target, steps):
    """Encode a problem and its solution as a token sequence.
    
    Format:
        [BOS] target [SEP] n1 [SEP] n2 [SEP] ... [SEP] nN [SEP]
        step1 [SEP] step2 [SEP] ... [SEP] stepN-1 [EOS]
    
    Each step: operand1 [OP] operand2 [=] result
    
    The problem portion is: [BOS] target [SEP] n1 [SEP] ... [SEP] nN [SEP]
    The solution portion is: step1 [SEP] step2 [SEP] ... [SEP] stepN-1 [EOS]
    
    Args:
        numbers: list of input numbers (shuffled order for the problem)
        target: target value
        steps: list of (a, op, b, result) tuples
        
    Returns:
        dict with:
            - 'tokens': list of tokens (without BOS/EOS - those are added later)
            - 'solution_start': index in the returned tokens where solution begins
    """
    tokens = []
    
    # Encode target FIRST
    tokens.extend(number_to_tokens(target))
    tokens.append(SEP)
    
    # Encode input numbers
    for i, num in enumerate(numbers):
        tokens.extend(number_to_tokens(num))
        tokens.append(SEP)
    
    # Mark where solution starts (after BOS is added, this will be offset by 1)
    solution_start = len(tokens)
    
    # Encode solution steps
    for i, (a, op, b, result) in enumerate(steps):
        # operand1
        tokens.extend(number_to_tokens(a))
        # operator
        tokens.append(op)
        # operand2
        tokens.extend(number_to_tokens(b))
        # equals sign
        tokens.append(EQUALS)
        # result
        tokens.extend(number_to_tokens(result))
        
        # Separator after each step (except last, where EOS goes)
        if i < len(steps) - 1:
            tokens.append(SEP)
    
    return {
        'tokens': tokens,
        'solution_start': solution_start,
    }


def generate_game_of_24_example(
    num_numbers=4,
    number_range=(1, 13),
    target_range=(0, 100),
    operators=None,
    integer_only=True,
    max_attempts=1000,
    rng=None,
):
    """Generate a single solvable Game of 24 example.
    
    Args:
        num_numbers: how many input numbers
        number_range: (min, max) for input numbers (inclusive)
        target_range: (min, max) for target (inclusive)
        operators: allowed operators (default: [+, -, *, /])
        integer_only: require integer intermediate results
        max_attempts: max random attempts before giving up
        rng: numpy random generator
        
    Returns:
        tuple of (numbers, target, steps) or None if no solution found
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if operators is None:
        operators = OPERATORS
    
    for _ in range(max_attempts):
        # Generate random input numbers
        numbers = rng.integers(
            number_range[0], 
            number_range[1] + 1, 
            size=num_numbers
        ).tolist()
        
        # Generate random target
        target = rng.integers(target_range[0], target_range[1] + 1)
        
        # Find solutions
        solutions = find_all_solutions(numbers, target, operators, integer_only)
        
        if solutions:
            # Pick a random solution
            perm, ops, steps = solutions[rng.integers(len(solutions))]
            return perm, target, steps
    
    return None


def get_cache_filename(dataset_size, seq_len, num_numbers, number_range, target_range, 
                       op_names, seed=None, split=None):
    """Generate a cache filename based on config parameters.
    
    This ensures the main script and generate_synthetic_data use the same naming.
    """
    if isinstance(num_numbers, int):
        num_numbers = [num_numbers]
    num_str = "_".join(str(n) for n in sorted(num_numbers))
    ops_str = "".join(sorted(op_names)).replace("+", "add").replace("-", "sub").replace("*", "mul").replace("/", "div")
    fname = f"game24_lossmask_contiguous"
    if split:
        fname += f"_{split}"
    fname += f"_ds{dataset_size}_sl{seq_len}_vs{VOCAB_SIZE}"
    fname += f"_nums{num_str}_nr{number_range[0]}_{number_range[1]}"
    fname += f"_tr{target_range[0]}_{target_range[1]}_ops{ops_str}"
    if seed is not None:
        fname += f"_seed{seed}"
    fname += ".npz"
    return fname


def _load_from_npz(data_path, dataset_size, seq_len):
    """Load pre-generated Game of 24 data from a .npz file.
    
    Args:
        data_path: path to the .npz file
        dataset_size: number of samples to use (None to use all)
        seq_len: sequence length (for validation)
        
    Returns:
        dict with 'input_ids' and 'loss_mask' numpy arrays
    """
    import os
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"[GAME OF 24] Loading from: {data_path}")
    
    data = np.load(data_path, allow_pickle=True)
    input_ids = data['input_ids']
    loss_mask = data['loss_mask']
    
    # Use all data if dataset_size is None
    if dataset_size is None:
        dataset_size = len(input_ids)
    
    # Limit to available data
    if dataset_size > len(input_ids):
        print(f"[GAME OF 24] Warning: requested {dataset_size} samples but only {len(input_ids)} available")
        dataset_size = len(input_ids)
    
    # Validate seq_len matches
    if input_ids.shape[1] != seq_len:
        raise ValueError(
            f"Sequence length mismatch: data has seq_len={input_ids.shape[1]}, "
            f"but requested seq_len={seq_len}"
        )
    
    print(f"[GAME OF 24] Loaded {dataset_size} samples, seq_len={seq_len}")
    
    return {
        'input_ids': input_ids[:dataset_size],
        'loss_mask': loss_mask[:dataset_size],
    }


def generate_synthetic_data(dataset_size, seq_len, config=None):
    """
    Generate a dataset of Game of 24 style arithmetic puzzles.
    
    Args:
        dataset_size: number of samples to generate (or load from cache), 
                      or None to use all available data from data_path
        seq_len: sequence length (will pad if shorter)
        config: dict with configuration. Key fields:
            - data_path: str, path to pre-generated .npz file (if provided, loads from file)
            - num_numbers: int or list of ints, number of input numbers per puzzle
                           If list, samples uniformly from the list for each example
                           (default: [3, 4, 5])
            - number_range: [min, max] for input numbers (default: [1, 13])
            - target_range: [min, max] for target (default: [0, 100])
            - operators: list of operator names to allow (default: ["+", "-", "*", "/"])
            - integer_only: bool, require integer intermediates (default: True)
            - seed: optional int for reproducibility
            - cache_dir: optional str, directory to cache generated data
            - split: optional str, split name (e.g., "train" or "test") for simple cache filenames
            
    Returns:
        dict with:
            - 'input_ids': numpy array of shape (dataset_size, seq_len)
            - 'loss_mask': numpy array of shape (dataset_size, seq_len) with 0 for
                           problem tokens (don't train) and 1 for solution tokens (train)
    
    Note:
        vocab_size is determined by the hardcoded GAME24_VOCAB_TOKENS (currently 20).
        The vocabulary includes PAD, BOS, EOS, SEP, digits 0-9, operators, and NEG.
    """
    import os
    import time
    
    if config is None:
        config = {}
    
    # Check if config specifies vocab_size and validate it matches
    config_vocab_size = config.get("vocab_size")
    if config_vocab_size is not None and config_vocab_size != VOCAB_SIZE:
        raise ValueError(
            f"Config vocab_size ({config_vocab_size}) does not match "
            f"hardcoded VOCAB_SIZE ({VOCAB_SIZE}). Remove vocab_size from config."
        )
    
    # ---- Check for data_path (pre-generated .npz file) ----
    data_path = config.get("data_path")
    if data_path is not None:
        return _load_from_npz(data_path, dataset_size, seq_len)
    
    # ---- Generate data on the fly ----
    # Configuration
    num_numbers_config = config.get("num_numbers", [3, 4, 5])
    # Support both single int and list of ints
    if isinstance(num_numbers_config, int):
        num_numbers_choices = [num_numbers_config]
    else:
        num_numbers_choices = list(num_numbers_config)
    
    number_range = tuple(config.get("number_range", [1, 13]))
    target_range = tuple(config.get("target_range", [0, 100]))
    integer_only = config.get("integer_only", True)
    
    # Parse operators (default excludes division to avoid non-integer results)
    op_name_to_token = {"+": ADD, "-": SUB, "*": MUL, "/": DIV}
    op_names = config.get("operators", ["+", "-", "*", "/"])
    operators = [op_name_to_token[name] for name in op_names]
    
    # Random seed
    seed = config.get("seed", None)
    rng = np.random.default_rng(seed)
    
    # Split name (for cache filename)
    split = config.get("split", None)  # e.g., "train" or "test"
    
    # ---- Caching setup ----
    cache_dir = config.get("cache_dir", None)
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        fname = get_cache_filename(
            dataset_size=dataset_size,
            seq_len=seq_len,
            num_numbers=num_numbers_choices,
            number_range=number_range,
            target_range=target_range,
            op_names=op_names,
            seed=seed,
            split=split,
        )
        cache_path = os.path.join(cache_dir, fname)
        
        # Try to load from cache
        if os.path.exists(cache_path):
            try:
                cached = np.load(cache_path, allow_pickle=True)
                cached_input_ids = cached["input_ids"]
                cached_loss_mask = cached["loss_mask"]
                
                # Validate shapes match
                if (cached_input_ids.shape[0] >= dataset_size and 
                    cached_input_ids.shape[1] == seq_len and
                    cached_loss_mask.shape[0] >= dataset_size and
                    cached_loss_mask.shape[1] == seq_len):
                    # Truncate to requested size if cache has more
                    print(f"[GAME OF 24] Loaded from cache: {cache_path}")
                    return {
                        'input_ids': cached_input_ids[:dataset_size],
                        'loss_mask': cached_loss_mask[:dataset_size],
                    }
            except Exception as e:
                print(f"[GAME OF 24] Cache load failed: {e}, regenerating...")
    
    # Pre-allocate arrays
    # input_ids: padded with PAD token
    # loss_mask: 0 for problem tokens and padding, 1 for solution tokens
    input_ids = np.full((dataset_size, seq_len), fill_value=PAD, dtype=np.int64)
    loss_mask = np.zeros((dataset_size, seq_len), dtype=np.int64)
    
    print(f"\n{'='*60}")
    print(f"[GAME OF 24] Generating {dataset_size} puzzles")
    print(f"{'='*60}")
    print(f"  Vocab size: {VOCAB_SIZE}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Num numbers: {num_numbers_choices}")
    print(f"  Number range: {number_range}")
    print(f"  Target range: {target_range}")
    print(f"  Operators: {op_names}")
    print(f"  Integer only: {integer_only}")
    print(f"  PAD token: {PAD}")
    print(f"{'='*60}")
    
    # Generate examples
    examples_generated = 0
    max_tokens_seen = 0
    
    for i in tqdm(range(dataset_size)):
        # Sample num_numbers for this example
        num_numbers = rng.choice(num_numbers_choices)
        
        result = generate_game_of_24_example(
            num_numbers=num_numbers,
            number_range=number_range,
            target_range=target_range,
            operators=operators,
            integer_only=integer_only,
            rng=rng,
        )
        
        if result is None:
            raise RuntimeError(
                f"Failed to generate example {i} after max attempts. "
                f"Try relaxing constraints (larger number/target range, more operators)."
            )
        
        solution_numbers, target, steps = result
        
        # Shuffle the input numbers so they don't match the solution order
        # This makes the problem harder - model must figure out correct ordering
        shuffled_numbers = rng.permutation(solution_numbers).tolist()
        
        # Encode to tokens (shuffled inputs, but solution still uses correct order)
        encoded = encode_problem_and_solution(shuffled_numbers, target, steps)
        tokens = encoded['tokens']
        solution_start_in_tokens = encoded['solution_start']
        
        # Build full sequence: [BOS] + tokens + [EOS]
        full_seq = [BOS] + tokens + [EOS]
        total_len = len(full_seq)
        
        if total_len > seq_len:
            raise ValueError(
                f"Example {i} requires {total_len} tokens but seq_len is {seq_len}. "
                f"Increase seq_len or reduce num_numbers."
            )
        
        max_tokens_seen = max(max_tokens_seen, total_len)
        
        # Fill in input_ids
        input_ids[i, :total_len] = full_seq
        
        # Create loss_mask: 0 for problem (BOS + problem tokens), 1 for solution (solution tokens + EOS)
        # solution_start_in_tokens is the index in 'tokens' where solution begins
        # In full_seq, this becomes solution_start_in_tokens + 1 (due to BOS)
        solution_start = solution_start_in_tokens + 1  # +1 for BOS

        # NOTE this MUST be contiguous, or AR training will break
        # so loss_mask should be 1 on PADDING tokens as well
        loss_mask[i, solution_start:] = 1 
        
        examples_generated += 1
    
    # Count solution tokens for reporting
    solution_tokens = np.sum(loss_mask)
    total_tokens = dataset_size * seq_len
    
    print(f"\n[GAME OF 24] Generation complete!")
    print(f"  Examples generated: {examples_generated}")
    print(f"  Max tokens used: {max_tokens_seen} / {seq_len}")
    print(f"  Solution tokens: {solution_tokens} / {total_tokens} ({100*solution_tokens/total_tokens:.1f}%)")
    
    # Show a sample
    if examples_generated > 0:
        sample_idx = 0
        sample = input_ids[sample_idx]
        sample_mask = loss_mask[sample_idx]
        # Find where padding starts
        non_pad = np.where(sample != PAD)[0]
        if len(non_pad) > 0:
            end_pos = non_pad[-1] + 1
        else:
            end_pos = 0
        sample_tokens = sample[:end_pos].tolist()
        sample_mask_vals = sample_mask[:end_pos].tolist()
        print(f"\n  Sample sequence (first example):")
        print(f"    Input IDs: {sample_tokens}")
        print(f"    Loss mask: {sample_mask_vals}")
        print(f"    Decoded: {decode_sequence(sample_tokens)}")
    
    print(f"{'='*60}\n")
    
    # Save to cache if configured
    if cache_path:
        try:
            np.savez_compressed(
                cache_path,
                input_ids=input_ids,
                loss_mask=loss_mask,
                dataset_size=np.array(dataset_size, dtype=np.int64),
                seq_len=np.array(seq_len, dtype=np.int64),
                vocab_size=np.array(VOCAB_SIZE, dtype=np.int64),
                created_ts=np.array(time.time()),
            )
            print(f"[GAME OF 24] Saved to cache: {cache_path}")
        except Exception as e:
            print(f"[GAME OF 24] Cache save failed: {e}")
    
    return {
        'input_ids': input_ids,
        'loss_mask': loss_mask,
    }


def decode_sequence(tokens):
    """Decode a token sequence to a human-readable string.
    
    For debugging/display purposes.
    """
    parts = []
    for t in tokens:
        if t == PAD:
            parts.append('[PAD]')
        elif t == BOS:
            parts.append('[BOS]')
        elif t == EOS:
            parts.append('[EOS]')
        elif t == SEP:
            parts.append('|')
        elif t == EQUALS:
            parts.append('=')
        elif t == NEG:
            parts.append('-')
        elif t == ADD:
            parts.append('+')
        elif t == SUB:
            parts.append('-')
        elif t == MUL:
            parts.append('*')
        elif t == DIV:
            parts.append('/')
        else:
            # Check if it's a digit token
            d = token_to_digit(t)
            if d is not None:
                parts.append(str(d))
            else:
                parts.append(f'[?{t}]')
    
    return ''.join(parts)


def verify_solution(numbers, target, solution_tokens):
    """Verify that a solution token sequence is correct.
    
    This is the corrected verifier (the original had a bug comparing result == result).
    
    Args:
        numbers: list of input numbers
        target: target value
        solution_tokens: tokens representing the solution steps
        
    Returns:
        (is_valid, error_message or None)
    """
    # Parse the solution tokens into steps
    # Each step: operand1 [OP] operand2 [=] result
    
    steps = []
    current_tokens = []
    
    for t in solution_tokens:
        if t == SEP or t == EOS:  # SEP or EOS
            if current_tokens:
                # Parse this step
                step = parse_step(current_tokens)
                if step is None:
                    return False, "Failed to parse step"
                steps.append(step)
                current_tokens = []
        else:
            current_tokens.append(t)
    
    # Handle any remaining tokens
    if current_tokens:
        step = parse_step(current_tokens)
        if step is None:
            return False, "Failed to parse final step"
        steps.append(step)
    
    if not steps:
        return False, "No steps found"
    
    # Verify each step's arithmetic
    for i, (a, op, b, claimed_result) in enumerate(steps):
        func = OP_TO_FUNC.get(op)
        if func is None:
            return False, f"Invalid operator in step {i}"
        
        actual_result = func(a, b)
        if actual_result is None:
            return False, f"Invalid operation in step {i} (e.g., division by zero)"
        
        # FIX: Compare actual_result to claimed_result (not result to result)
        if abs(actual_result - claimed_result) > 1e-10:
            return False, f"Step {i}: {a} {OP_TO_STR[op]} {b} = {claimed_result}, but actual is {actual_result}"
    
    # Verify final result matches target
    final_result = steps[-1][3]
    if abs(final_result - target) > 1e-10:
        return False, f"Final result {final_result} does not match target {target}"
    
    # Verify all input numbers are used exactly once
    used_numbers = []
    
    # First operand of first step
    used_numbers.append(steps[0][0])
    
    # Second operands of all steps
    for _, _, b, _ in steps:
        used_numbers.append(b)
    
    # Check counts
    input_counts = {}
    for n in numbers:
        input_counts[n] = input_counts.get(n, 0) + 1
    
    used_counts = {}
    for n in used_numbers:
        used_counts[n] = used_counts.get(n, 0) + 1
    
    if input_counts != used_counts:
        return False, f"Numbers used {used_counts} don't match input {input_counts}"
    
    return True, None


def parse_step(tokens):
    """Parse a step's tokens into (operand1, operator, operand2, result).
    
    Expected format: num1_tokens [OP] num2_tokens [=] result_tokens
    
    Returns:
        (a, op, b, result) tuple or None if parsing fails
    """
    # Find operator position
    op_pos = None
    op = None
    for i, t in enumerate(tokens):
        if t in (ADD, SUB, MUL, DIV):
            op_pos = i
            op = t
            break
    
    if op_pos is None:
        return None
    
    # Find equals position
    eq_pos = None
    for i, t in enumerate(tokens):
        if t == EQUALS:
            eq_pos = i
            break
    
    if eq_pos is None:
        return None
    
    # Parse operands and result
    a_tokens = tokens[:op_pos]
    b_tokens = tokens[op_pos + 1:eq_pos]
    result_tokens = tokens[eq_pos + 1:]
    
    a = tokens_to_number(a_tokens)
    b = tokens_to_number(b_tokens)
    result = tokens_to_number(result_tokens)
    
    if a is None or b is None or result is None:
        return None
    
    return (a, op, b, result)


def extract_problem_signature(seq):
    """Extract (target, sorted_input_numbers) from a sequence for duplicate detection.
    
    Returns:
        tuple of (target, tuple of sorted input numbers) or None if parsing fails
    """
    # Find EOS position
    eos_positions = np.where(seq == EOS)[0]
    if len(eos_positions) == 0:
        return None
    eos_pos = eos_positions[0]
    
    # Parse tokens between BOS and first solution step
    # Format: [BOS] target [SEP] n1 [SEP] n2 [SEP] ... [SEP] nN [SEP] step1 ...
    tokens = seq[1:eos_pos].tolist()  # Skip BOS
    
    # Split by SEP to get parts
    parts = []
    current = []
    for t in tokens:
        if t == SEP:
            if current:
                parts.append(current)
            current = []
        elif t in (ADD, SUB, MUL, DIV, EQUALS):
            # We've hit solution tokens, stop
            break
        else:
            current.append(t)
    if current:
        parts.append(current)
    
    if len(parts) < 2:
        return None
    
    # First part is target
    target = tokens_to_number(parts[0])
    if target is None:
        return None
    
    # Remaining parts are input numbers
    input_numbers = []
    for part in parts[1:]:
        num = tokens_to_number(part)
        if num is not None:
            input_numbers.append(num)
    
    # Return signature: (target, sorted tuple of inputs)
    return (target, tuple(sorted(input_numbers)))


def analyze_duplicates(input_ids):
    """Analyze dataset for duplicate problems (same target + same input numbers).
    
    Args:
        input_ids: numpy array of shape (dataset_size, seq_len)
    
    Returns:
        dict with analysis results
    """
    from collections import Counter
    
    signatures = []
    for i in range(len(input_ids)):
        sig = extract_problem_signature(input_ids[i])
        if sig is not None:
            signatures.append(sig)
    
    sig_counts = Counter(signatures)
    
    total = len(signatures)
    unique = len(sig_counts)
    duplicates = total - unique
    
    # Count how many examples are part of duplicate groups
    examples_with_duplicates = sum(count for count in sig_counts.values() if count > 1)
    
    # Find most common duplicates
    most_common = sig_counts.most_common(10)
    
    return {
        "total": total,
        "unique": unique,
        "duplicates": duplicates,
        "examples_with_duplicates": examples_with_duplicates,
        "duplicate_rate": duplicates / total if total > 0 else 0,
        "most_common": most_common,
    }


def remove_test_duplicates_from_train(train_input_ids, train_loss_mask, test_input_ids):
    """Remove examples from training set that have the same problem signature as test examples.
    
    Args:
        train_input_ids: numpy array of shape (train_size, seq_len)
        train_loss_mask: numpy array of shape (train_size, seq_len)
        test_input_ids: numpy array of shape (test_size, seq_len)
        
    Returns:
        (filtered_input_ids, filtered_loss_mask) with test duplicates removed
    """
    # Get all test problem signatures
    test_signatures = set()
    for i in range(len(test_input_ids)):
        sig = extract_problem_signature(test_input_ids[i])
        if sig is not None:
            test_signatures.add(sig)
    
    print(f"  Found {len(test_signatures)} unique test problem signatures")
    
    # Filter training examples
    keep_indices = []
    removed_count = 0
    for i in range(len(train_input_ids)):
        sig = extract_problem_signature(train_input_ids[i])
        if sig is None or sig not in test_signatures:
            keep_indices.append(i)
        else:
            removed_count += 1
    
    print(f"  Removed {removed_count} training examples that overlap with test set")
    
    keep_indices = np.array(keep_indices)
    return train_input_ids[keep_indices], train_loss_mask[keep_indices]


def extract_problem_from_sequence(seq):
    """Extract target and input numbers from a sequence.
    
    Args:
        seq: numpy array or list of token IDs
        
    Returns:
        (target, input_numbers) tuple or (None, None) if parsing fails
    """
    # Convert to list if numpy
    if hasattr(seq, 'tolist'):
        seq = seq.tolist()
    
    # Find BOS position
    if BOS not in seq:
        return None, None
    bos_pos = seq.index(BOS)
    
    # Find EOS position
    if EOS not in seq:
        return None, None
    eos_pos = seq.index(EOS)
    
    # Get tokens between BOS and EOS (excluding BOS and EOS)
    tokens = seq[bos_pos + 1:eos_pos]
    
    # Split by SEP to get parts
    # Format: target [SEP] n1 [SEP] n2 [SEP] ... [SEP] nN [SEP] solution_steps
    parts = []
    current = []
    in_solution = False
    
    for t in tokens:
        if t == SEP:
            if current:
                parts.append(current)
            current = []
        elif t in (ADD, SUB, MUL, DIV, EQUALS):
            # We've hit solution tokens - stop parsing problem
            in_solution = True
            break
        else:
            current.append(t)
    
    # If we didn't hit solution tokens, add the last part if non-empty
    if not in_solution and current:
        parts.append(current)
    
    if len(parts) < 2:
        return None, None
    
    # First part is target
    target = tokens_to_number(parts[0])
    if target is None:
        return None, None
    
    # Remaining parts (up to where solution starts) are input numbers
    input_numbers = []
    for part in parts[1:]:
        num = tokens_to_number(part)
        if num is not None:
            input_numbers.append(num)
    
    if len(input_numbers) == 0:
        return None, None
    
    return target, input_numbers


def extract_solution_from_sequence(seq):
    """Extract solution tokens from a sequence.
    
    The solution starts after the problem (target + input numbers + their SEPs).
    We detect the solution by finding the first operator token.
    
    Args:
        seq: numpy array or list of token IDs
        
    Returns:
        list of solution tokens (between first operator and EOS), or None if no solution found
    """
    # Convert to list if numpy
    if hasattr(seq, 'tolist'):
        seq = seq.tolist()
    
    # Find EOS position
    if EOS not in seq:
        # Try to find end by looking for first PAD
        if PAD in seq:
            eos_pos = seq.index(PAD)
        else:
            eos_pos = len(seq)
    else:
        eos_pos = seq.index(EOS)
    
    # Find where solution starts (first operator token after BOS)
    solution_start = None
    for i, t in enumerate(seq[:eos_pos]):
        if t in (ADD, SUB, MUL, DIV):
            # Found an operator - go back to find the start of this step
            # The step format is: num_tokens [OP] num_tokens [=] result_tokens
            # We need to find where this step's first operand starts
            # Search backward to find SEP or BOS
            j = i - 1
            while j >= 0 and seq[j] not in (SEP, BOS):
                j -= 1
            solution_start = j + 1
            break
    
    if solution_start is None:
        return None
    
    # Extract solution tokens (from solution_start to EOS, including EOS for parsing)
    solution_tokens = seq[solution_start:eos_pos]
    
    return solution_tokens


def evaluate_completions(predicted_ids, ground_truth_ids):
    """
    Evaluate Game of 24 puzzle completions by verifying predicted solutions.
    
    This is the main entry point for evaluating model outputs on Game of 24 puzzles.
    It extracts the problem from ground truth, extracts the solution from predictions,
    and verifies that the solution is mathematically correct and uses all input numbers.
    
    Args:
        predicted_ids: numpy array of shape (n_samples, seq_len) - model outputs
        ground_truth_ids: numpy array of shape (n_samples, seq_len) - ground truth sequences
        
    Returns:
        dict with:
            - 'puzzle_accuracy': fraction of puzzles with correct solutions
            - 'n_correct_puzzles': number of fully correct puzzles
            - 'n_total_puzzles': total number of puzzles evaluated
            - 'n_valid_solutions': number of mathematically valid solutions
            - 'n_parse_failed': count of solutions that couldn't be parsed
            - 'n_malformed': count of malformed solutions
            - 'arithmetic_accuracy': fraction of solutions with correct arithmetic
            - 'error_breakdown': dict with counts of each error type
    """
    n_total = len(predicted_ids)
    n_correct = 0
    n_valid_arithmetic = 0
    n_parse_failed = 0
    n_malformed = 0
    
    error_counts = {
        'problem_parse_failed': 0,
        'no_solution_found': 0,
        'step_parse_failed': 0,
        'arithmetic_error': 0,
        'wrong_target': 0,
        'wrong_numbers_used': 0,
    }
    
    for pred_seq, gt_seq in zip(predicted_ids, ground_truth_ids):
        # Extract problem from ground truth
        target, input_numbers = extract_problem_from_sequence(gt_seq)
        
        if target is None or input_numbers is None:
            n_parse_failed += 1
            error_counts['problem_parse_failed'] += 1
            continue
        
        # Extract solution from prediction
        solution_tokens = extract_solution_from_sequence(pred_seq)
        
        if solution_tokens is None or len(solution_tokens) == 0:
            n_malformed += 1
            error_counts['no_solution_found'] += 1
            continue
        
        # Verify the solution
        is_valid, error_msg = verify_solution(input_numbers, target, solution_tokens)
        
        if is_valid:
            n_correct += 1
            n_valid_arithmetic += 1
        else:
            # Categorize the error
            if error_msg and 'parse' in error_msg.lower():
                error_counts['step_parse_failed'] += 1
            elif error_msg and ('actual is' in error_msg or 'division by zero' in error_msg):
                error_counts['arithmetic_error'] += 1
            elif error_msg and 'target' in error_msg.lower():
                error_counts['wrong_target'] += 1
            elif error_msg and 'numbers' in error_msg.lower():
                error_counts['wrong_numbers_used'] += 1
            else:
                n_malformed += 1
    
    puzzle_accuracy = n_correct / n_total if n_total > 0 else 0.0
    
    results = {
        'puzzle_accuracy': puzzle_accuracy,
        'n_correct_puzzles': n_correct,
        'n_total_puzzles': n_total,
        'n_valid_solutions': n_valid_arithmetic,
        'n_parse_failed': n_parse_failed,
        'n_malformed': n_malformed,
        'arithmetic_accuracy': n_valid_arithmetic / n_total if n_total > 0 else 0.0,
        'error_breakdown': error_counts,
    }
    
    # Print summary
    print()
    print("=" * 50)
    print("GAME OF 24 EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total puzzles:      {n_total}")
    print(f"Correct solutions:  {n_correct} ({puzzle_accuracy*100:.1f}%)")
    print(f"Parse failed:       {n_parse_failed}")
    print(f"Malformed:          {n_malformed}")
    print("-" * 50)
    print("ERROR BREAKDOWN")
    print("-" * 50)
    for error_type, count in error_counts.items():
        if count > 0:
            print(f"  {error_type}: {count}")
    print("=" * 50)
    
    return results


if __name__ == "__main__":
    import argparse
    import time as time_module
    import os
    
    parser = argparse.ArgumentParser(description="Generate Game of 24 train/test datasets")
    parser.add_argument("--train_size", type=int, default=500000, help="Number of training examples")
    parser.add_argument("--test_size", type=int, default=10000, help="Number of test examples")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument("--train_seed", type=int, default=42, help="Random seed for training set")
    parser.add_argument("--test_seed", type=int, default=12345, help="Random seed for test set")
    parser.add_argument("--cache_dir", type=str, default="cache/", help="Cache directory")
    parser.add_argument("--force_regenerate", action="store_true", help="Force regeneration even if cache exists")
    args = parser.parse_args()
    
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Base config shared by train and test
    base_config = {
        "num_numbers": [3, 4, 5],  # Variable number of inputs
        "number_range": [1, 13],
        "target_range": [0, 100],
        "operators": ["+", "-", "*", "/"],
        "integer_only": True,
    }
    
    print(f"{'='*60}")
    print("Game of 24 Dataset Generation")
    print(f"{'='*60}")
    print(f"  Train size: {args.train_size:,}")
    print(f"  Test size: {args.test_size:,}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Vocab size: {VOCAB_SIZE}")
    print(f"  Cache directory: {args.cache_dir}")
    print(f"{'='*60}\n")
    
    # Build expected cache filenames
    test_fname = get_cache_filename(
        dataset_size=args.test_size,
        seq_len=args.seq_len,
        num_numbers=base_config["num_numbers"],
        number_range=base_config["number_range"],
        target_range=base_config["target_range"],
        op_names=base_config["operators"],
        seed=args.test_seed,
        split="test",
    )
    test_path = os.path.join(args.cache_dir, test_fname)
    
    # Note: train filename will be determined after dedup (size may change)
    
    # Check if test already exists
    test_exists = os.path.exists(test_path)
    
    if not args.force_regenerate and test_exists:
        print(f"Loading existing test set from: {test_path}")
        test_cached = np.load(test_path)
        test_input_ids = test_cached['input_ids']
        test_loss_mask = test_cached['loss_mask']
        print(f"  Loaded test: {test_input_ids.shape}")
    else:
        # ========== Generate Test Set ==========
        print("Step 1: Generating test set...")
        start_time = time_module.time()
        
        test_config = {**base_config, "seed": args.test_seed}
        test_result = generate_synthetic_data(
            dataset_size=args.test_size,
            seq_len=args.seq_len,
            config=test_config,
        )
        test_input_ids = test_result['input_ids']
        test_loss_mask = test_result['loss_mask']
        
        test_elapsed = time_module.time() - start_time
        print(f"Test set generation took {test_elapsed:.2f} seconds")
        
        # Save test set
        np.savez_compressed(
            test_path,
            input_ids=test_input_ids,
            loss_mask=test_loss_mask,
        )
        print(f"  Saved test set to: {test_path}")
    
    # ========== Generate Training Set ==========
    print("\nStep 2: Generating training set...")
    start_time = time_module.time()
    
    train_config = {**base_config, "seed": args.train_seed}
    train_result = generate_synthetic_data(
        dataset_size=args.train_size,
        seq_len=args.seq_len,
        config=train_config,
    )
    train_input_ids = train_result['input_ids']
    train_loss_mask = train_result['loss_mask']
    
    train_elapsed = time_module.time() - start_time
    print(f"Training set generation took {train_elapsed:.2f} seconds")
    
    # ========== Remove Test Duplicates from Training Set ==========
    print("\nStep 3: Removing test duplicates from training set...")
    start_time = time_module.time()
    
    train_input_ids, train_loss_mask = remove_test_duplicates_from_train(
        train_input_ids, train_loss_mask, test_input_ids
    )
    
    dedup_elapsed = time_module.time() - start_time
    print(f"Deduplication took {dedup_elapsed:.2f} seconds")
    
    # ========== Save Training Set ==========
    # Use actual size after dedup for the filename
    train_fname = get_cache_filename(
        dataset_size=len(train_input_ids),
        seq_len=args.seq_len,
        num_numbers=base_config["num_numbers"],
        number_range=base_config["number_range"],
        target_range=base_config["target_range"],
        op_names=base_config["operators"],
        seed=args.train_seed,
        split="train",
    )
    train_path = os.path.join(args.cache_dir, train_fname)
    
    print(f"\nStep 4: Saving training set to cache...")
    np.savez_compressed(
        train_path,
        input_ids=train_input_ids,
        loss_mask=train_loss_mask,
    )
    print(f"  Saved training set to: {train_path}")
    
    # ========== Summary ==========
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Training set: {len(train_input_ids):,} examples")
    print(f"  Test set: {len(test_input_ids):,} examples")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Vocab size: {VOCAB_SIZE}")
    print(f"  Cache directory: {args.cache_dir}")
    
    # Show sample examples
    print(f"\n  Sample training examples:")
    for i in range(min(5, len(train_input_ids))):
        seq = train_input_ids[i]
        non_pad = np.where(seq != PAD)[0]
        end_pos = non_pad[-1] + 1 if len(non_pad) > 0 else 0
        decoded = decode_sequence(seq[:end_pos].tolist())
        print(f"    {i}: {decoded}")
    
    print(f"\n  Sample test examples:")
    for i in range(min(5, len(test_input_ids))):
        seq = test_input_ids[i]
        non_pad = np.where(seq != PAD)[0]
        end_pos = non_pad[-1] + 1 if len(non_pad) > 0 else 0
        decoded = decode_sequence(seq[:end_pos].tolist())
        print(f"    {i}: {decoded}")
    
    # Test tokenizer
    print(f"\n  Tokenizer info:")
    tokenizer = Game24Tokenizer()
    print(f"    Vocab size: {tokenizer.vocab_size}")
    print(f"    PAD token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    print(f"    BOS token: '{tokenizer.bos_token}' (id={tokenizer.bos_token_id})")
    print(f"    EOS token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
    
    print(f"\n{'='*60}")
    print("Usage: To load these datasets in training code, use:")
    print(f"\n  # Load training set")
    print(f"  generate_synthetic_data(dataset_size={len(train_input_ids)}, seq_len={args.seq_len},")
    print(f"                          config={{'cache_dir': '{args.cache_dir}', 'split': 'train', 'seed': {args.train_seed}}})")
    print(f"\n  # Load test set")
    print(f"  generate_synthetic_data(dataset_size={len(test_input_ids)}, seq_len={args.seq_len},")
    print(f"                          config={{'cache_dir': '{args.cache_dir}', 'split': 'test', 'seed': {args.test_seed}}})")
    print(f"{'='*60}")
