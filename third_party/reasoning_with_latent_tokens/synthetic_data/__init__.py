"""
High-level dispatcher for synthetic dataset generation.

This module routes dataset generation requests to the appropriate
synthetic dataset module based on the configuration.
"""


def generate_synthetic_data(dataset_size, seq_len, config=None):
    """
    High-level dispatcher for synthetic data generation.
    
    Routes to the appropriate synthetic dataset module based on config.
    
    Args:
        dataset_size: number of samples to generate
        seq_len: sequence length
        config: dict with configuration. Key fields:
            - vocab_size: int, vocabulary size (required for most datasets)
            - name: str, type of synthetic dataset (default: "sudoku")
            - base: int, for sudoku datasets (default: 3)
            - data_path: str, path to data file (for sudoku-puzzle and zebra)
            - Other fields are passed to the specific dataset generator
    
    Returns:
        numpy array of shape (dataset_size, seq_len)
    
    Supported dataset types:
        - 'sudoku': procedurally generated sudoku boards (default)
        - 'sudoku-puzzle': pre-generated sudoku puzzles from .npy files
        - 'sudoku-conditional': procedurally generated puzzle-solution pairs (allows non-unique)
        - 'zebra': pre-generated zebra/logic puzzles from .pkl files
        - 'game-of-24': arithmetic puzzles with step-by-step solutions
        - 'repeat': repeated fixed sequence for overfitting tests
    """
    if config is None:
        config = {}
    
    # Determine which synthetic dataset to use
    # Default to "sudoku" for backward compatibility
    name = config.get("name", "sudoku")
    
    # If "base" is present but "name" is not explicitly set, assume sudoku (for backward compatibility)
    if "base" in config and "name" not in config:
        name = "sudoku"
    
    # Route to appropriate dataset generator
    if name == "sudoku":
        # Ensure vocab_size is in config for sudoku
        if "vocab_size" not in config:
            raise ValueError("vocab_size must be provided in config for sudoku")
        from synthetic_data.sudoku import generate_synthetic_data as sudoku_generate
        return sudoku_generate(dataset_size, seq_len, config)
    
    elif name == "sudoku-puzzle":
        # Ensure vocab_size is in config
        if "vocab_size" not in config:
            raise ValueError("vocab_size must be provided in config for sudoku-puzzle")
        from synthetic_data.sudoku_puzzle import generate_synthetic_data as sudoku_puzzle_generate
        return sudoku_puzzle_generate(dataset_size, seq_len, config)
    
    elif name == "zebra":
        from synthetic_data.zebra import generate_synthetic_data as zebra_generate
        return zebra_generate(dataset_size, seq_len, config)
    
    elif name == "game-of-24":
        from synthetic_data.game_of_24 import generate_synthetic_data as game24_generate
        return game24_generate(dataset_size, seq_len, config)
    
    elif name == "sudoku-conditional":
        # Ensure vocab_size is in config
        if "vocab_size" not in config:
            raise ValueError("vocab_size must be provided in config for sudoku-conditional")
        from synthetic_data.sudoku_conditional import generate_synthetic_data as sudoku_conditional_generate
        return sudoku_conditional_generate(dataset_size, seq_len, config)
    
    elif name == "repeat":
        # Ensure vocab_size is in config for repeat
        if "vocab_size" not in config:
            raise ValueError("vocab_size must be provided in config for repeat")
        from synthetic_data.repeat import generate_synthetic_data as repeat_generate
        return repeat_generate(dataset_size, seq_len, config)

    elif name.startswith("dvar-"):
        # Diffusion-vs-AR datasets: sudoku, countdown (cd), 3-SAT, path
        from synthetic_data.diffusion_vs_ar import generate_synthetic_data as dvar_generate
        return dvar_generate(dataset_size, seq_len, config)

    else:
        raise ValueError(
            f"Unknown synthetic dataset type: {name}. "
            f"Supported types: 'sudoku', 'sudoku-puzzle', 'zebra', 'game-of-24', 'repeat', "
            f"'dvar-sudoku', 'dvar-cd[3,4,5]', 'dvar-3sat[5,7,9]', 'dvar-path'"
        )


def evaluate_completions(predicted_ids, ground_truth_ids, config):
    """
    Evaluate puzzle completions by comparing predicted vs ground truth.
    
    Routes to the appropriate dataset-specific evaluation function.
    
    Args:
        predicted_ids: numpy array of shape (n_samples, seq_len) - model outputs
        ground_truth_ids: numpy array of shape (n_samples, seq_len) - ground truth
        config: dict or OmegaConf with configuration. Must contain 'name' field.
        
    Returns:
        dict with evaluation metrics, or None if no evaluation is available
        for the dataset type.
        
    Supported dataset types with evaluation:
        - 'zebra': logic puzzle evaluation (cell, row, puzzle accuracy)
        - 'sudoku-puzzle': sudoku puzzle evaluation (cell, row, column, box, puzzle accuracy)
        - 'sudoku-conditional': sudoku puzzle evaluation (cell, row, column, box, puzzle accuracy)
        - 'game-of-24': arithmetic puzzle evaluation (solution validity, puzzle accuracy)
    """
    if config is None:
        return None
    
    # Convert config to dict if needed
    config_dict = config if isinstance(config, dict) else dict(config)
    name = config_dict.get("name")
    
    if name == "zebra":
        from synthetic_data.zebra import evaluate_completions as zebra_evaluate
        return zebra_evaluate(predicted_ids, ground_truth_ids)
    
    if name == "sudoku-puzzle":
        from synthetic_data.sudoku_puzzle import evaluate_completions as sudoku_puzzle_evaluate
        return sudoku_puzzle_evaluate(predicted_ids, ground_truth_ids)
    
    if name == "sudoku-conditional":
        from synthetic_data.sudoku_conditional import evaluate_completions as sudoku_conditional_evaluate
        return sudoku_conditional_evaluate(predicted_ids, ground_truth_ids)
    
    if name == "game-of-24":
        from synthetic_data.game_of_24 import evaluate_completions as game24_evaluate
        return game24_evaluate(predicted_ids, ground_truth_ids)

    if name.startswith("dvar-"):
        from synthetic_data.diffusion_vs_ar import evaluate_completions as dvar_evaluate
        return dvar_evaluate(predicted_ids, ground_truth_ids, config)

    # No evaluation available for this dataset type
    return None


def evaluate_samples(samples_path, config, verbose=False):
    """
    Evaluate generated samples from a samples.json file.
    
    Routes to the appropriate dataset-specific evaluation function.
    This is used for sample_eval mode (unconditional generation).
    
    Args:
        samples_path: str, path to the samples.json file
        config: dict or OmegaConf with configuration. Must contain 'name' field.
        verbose: bool, if True print per-sample details
        
    Returns:
        dict with evaluation metrics, or None if no evaluation is available
        for the dataset type.
        
    Supported dataset types with evaluation:
        - 'sudoku': sudoku board evaluation (validity, violations, loss, diversity)
    """
    if config is None:
        return None
    
    # Convert config to dict if needed
    config_dict = config if isinstance(config, dict) else dict(config)
    name = config_dict.get("name")
    
    if name == "sudoku":
        from synthetic_data.sudoku.verify import evaluate_samples_file
        base = config_dict.get("base", 3)
        results = evaluate_samples_file(
            samples_path,
            base=base,
            verbose=verbose,
            include_malformed=False,
        )
        
        # Print summary
        n = base * base
        max_violations = 3 * n
        max_loss = 3 * (n - 1)
        
        print()
        print("=" * 50)
        print("SUDOKU EVALUATION RESULTS")
        print("=" * 50)
        print(f"Sudoku type:        base={base} ({n}x{n} grid)")
        print(f"Total samples:      {results['n_samples']}")
        print(f"Format OK:          {results['n_format_ok']} ({results['format_ok_rate']*100:.1f}%)")
        print(f"Valid sudoku:       {results['n_valid']} ({results['sudoku_valid_rate']*100:.1f}%)")
        print(f"Avg violations:     {results['avg_violations']:.2f} / {max_violations}")
        print(f"Avg loss:           {results['avg_loss']:.4f} / {max_loss:.1f}")
        print("-" * 50)
        print("DIVERSITY")
        print("-" * 50)
        print(f"Avg entropy:        {results['avg_entropy']:.4f} / {results['max_entropy']:.4f} bits")
        print(f"Normalized entropy: {results['normalized_entropy']*100:.1f}%")
        print("=" * 50)
        
        return results
    
    # No evaluation available for this dataset type
    return None
