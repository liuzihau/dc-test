"""
Sudoku Conditional dataset - generates puzzle-solution pairs procedurally.

Unlike sudoku-puzzle which loads from pre-generated files with unique solutions,
this dataset generates puzzles on-the-fly and allows multiple solutions.
"""

from synthetic_data.sudoku_conditional.data import (
    generate_synthetic_data,
    evaluate_completions,
    SudokuConditionalTokenizer,
    SUDOKU_CONDITIONAL_VOCAB_TOKENS,
)
