"""
Game of 24 synthetic dataset module.

This module provides functions for generating Game of 24 style puzzles
with step-by-step solutions.
"""

from synthetic_data.game_of_24.data import generate_synthetic_data, evaluate_completions, Game24Tokenizer

__all__ = ['generate_synthetic_data', 'evaluate_completions', 'Game24Tokenizer']
