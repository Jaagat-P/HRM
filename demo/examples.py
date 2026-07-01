"""Built-in example Sudoku puzzles for the HRM demo.

Each puzzle is an 81-char string, row-major, with '0' for blanks. These are used by the CLI
(`python -m demo.cli --example <name>`) so the demo runs with zero setup.
"""
from typing import Dict

EXAMPLES: Dict[str, str] = {
    # A classic newspaper-style puzzle.
    "easy": (
        "530070000"
        "600195000"
        "098000060"
        "800060003"
        "400803001"
        "700020006"
        "060000280"
        "000419005"
        "000080079"
    ),
    # A hard puzzle with only 17 givens (the minimum for a unique solution).
    "hard": (
        "000000010"
        "400000000"
        "020000000"
        "000050407"
        "008000300"
        "001090000"
        "300400200"
        "050100000"
        "000806000"
    ),
    # Fully blank board: no constraints, model must produce *a* valid grid.
    "blank": "0" * 81,
}

DEFAULT_EXAMPLE = "hard"


def get_example(name: str) -> str:
    if name not in EXAMPLES:
        raise KeyError(f"Unknown example {name!r}. Choices: {', '.join(sorted(EXAMPLES))}")
    return EXAMPLES[name]
