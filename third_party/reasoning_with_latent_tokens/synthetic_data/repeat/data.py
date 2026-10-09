"""
Repeat dataset: generates copies of the same fixed sequence.

This is useful for sanity checking that a model can overfit to a single sequence.
If the model can't learn to perfectly predict this, something is fundamentally broken.
"""

import numpy as np


def generate_synthetic_data(dataset_size, seq_len, config=None):
    """
    Generate a dataset where every sample is the same fixed sequence.
    
    The sequence is: [BOS] + repeating pattern of digits + [EOS] + padding
    
    Args:
        dataset_size: number of samples (all identical)
        seq_len: sequence length
        config: dict with configuration. Key fields:
            - vocab_size: int, vocabulary size (required)
            - seed: optional int for reproducibility of the pattern
            - pattern: optional list of ints to use as the repeating pattern
                       If not provided, uses [0, 1, 2, ..., vocab_size-3] cycling
    
    Returns:
        numpy array of shape (dataset_size, seq_len)
    """
    if config is None:
        config = {}
    
    vocab_size = config.get("vocab_size")
    if vocab_size is None:
        raise ValueError("vocab_size must be provided in config")
    
    if vocab_size < 3:
        raise ValueError(f"vocab_size must be >= 3 (got {vocab_size})")
    
    # Special tokens
    BOS = vocab_size - 2
    EOS = vocab_size - 1
    
    # Number of content tokens (excluding BOS/EOS)
    num_content_tokens = vocab_size - 2
    
    # Get or generate the pattern
    if "pattern" in config:
        pattern = np.array(config["pattern"], dtype=np.int64)
        # Validate pattern values
        if pattern.max() >= num_content_tokens or pattern.min() < 0:
            raise ValueError(
                f"Pattern values must be in [0, {num_content_tokens-1}], "
                f"got range [{pattern.min()}, {pattern.max()}]"
            )
    else:
        # Default: cycle through all content tokens
        if "seed" in config:
            np.random.seed(int(config["seed"]))
            # Random permutation for variety
            pattern = np.random.permutation(num_content_tokens)
        else:
            # Simple sequential pattern: 0, 1, 2, ..., num_content_tokens-1
            pattern = np.arange(num_content_tokens)
    
    # Content length (excluding BOS and EOS)
    content_len = seq_len - 2
    if content_len < 1:
        raise ValueError(f"seq_len must be >= 3 (got {seq_len})")
    
    # Create the fixed sequence by tiling the pattern
    num_tiles = (content_len + len(pattern) - 1) // len(pattern)
    content = np.tile(pattern, num_tiles)[:content_len]
    
    # Build the full sequence: [BOS] + content + [EOS]
    fixed_seq = np.concatenate([
        [BOS],
        content,
        [EOS]
    ]).astype(np.int64)
    
    assert len(fixed_seq) == seq_len, f"Expected {seq_len}, got {len(fixed_seq)}"
    
    # Tile to create dataset_size copies
    dataset = np.tile(fixed_seq, (dataset_size, 1))
    
    # Print info about the dataset (useful for debugging)
    print(f"\n{'='*60}")
    print(f"[REPEAT DATASET] Created repeat dataset for overfitting test")
    print(f"{'='*60}")
    print(f"  Dataset size: {dataset_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  BOS token: {BOS}, EOS token: {EOS}")
    print(f"  Pattern length: {len(pattern)}")
    print(f"  Pattern: {pattern.tolist()[:20]}{'...' if len(pattern) > 20 else ''}")
    print(f"  First 30 tokens of sequence: {fixed_seq[:30].tolist()}")
    print(f"  Last 10 tokens of sequence: {fixed_seq[-10:].tolist()}")
    print(f"{'='*60}\n")
    
    return dataset
