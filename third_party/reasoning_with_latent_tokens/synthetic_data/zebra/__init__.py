"""
Zebra Puzzle dataset: loads pre-generated logic puzzles (Einstein's riddle style).

This dataset contains grid-based logic puzzles where clues constrain
relationships between entities (houses) and their attributes (properties).
"""
from synthetic_data.zebra.data import generate_synthetic_data, evaluate_completions, ZebraTokenizer

__all__ = ['generate_synthetic_data', 'evaluate_completions', 'ZebraTokenizer']

