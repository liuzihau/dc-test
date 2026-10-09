"""
Sudoku Puzzle dataset: loads pre-generated sudoku puzzles with solution strategies.

This dataset contains puzzles with unique solutions where each cell is annotated
with the solving strategy needed to determine its value.
"""
from synthetic_data.sudoku_puzzle.data import generate_synthetic_data, evaluate_completions

__all__ = ['generate_synthetic_data', 'evaluate_completions']

