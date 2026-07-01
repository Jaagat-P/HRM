"""Sudoku board encoding/decoding for HRM inference.

The model consumes a 9x9 board flattened to a length-81 sequence, with each cell encoded
as ``digit + 1`` (matching dataset/build_sudoku_dataset.py):

    blank (0) -> token 1,  digit 1 -> token 2,  ...  digit 9 -> token 10

so vocab_size is 11 (token 0 is reserved as PAD and never appears in a full board).

A puzzle is written as an 81-character string of digits 1-9, using '0' or '.' for blanks,
read left-to-right, top-to-bottom.
"""
from typing import List

import torch

SEQ_LEN = 81
VOCAB_SIZE = 11  # PAD + digits 0..9
BLANK_CHARS = ".0"


def parse_puzzle_string(puzzle: str) -> List[int]:
    """Parse an 81-char Sudoku string into a list of 81 digits (blanks -> 0)."""
    cleaned = "".join(ch for ch in puzzle if not ch.isspace())
    if len(cleaned) != SEQ_LEN:
        raise ValueError(f"Sudoku must have {SEQ_LEN} cells, got {len(cleaned)}")

    digits = []
    for ch in cleaned:
        if ch in BLANK_CHARS:
            digits.append(0)
        elif ch.isdigit() and ch != "0":
            digits.append(int(ch))
        else:
            raise ValueError(f"Invalid Sudoku character: {ch!r}")
    return digits


def encode_puzzle(puzzle: str, device: str = "cpu") -> torch.Tensor:
    """Encode a puzzle string into a model input tensor of shape [1, 81] (int32)."""
    digits = parse_puzzle_string(puzzle)
    tokens = [d + 1 for d in digits]  # digit -> token
    return torch.tensor(tokens, dtype=torch.int32, device=device).unsqueeze(0)


def decode_board(tokens: torch.Tensor) -> str:
    """Decode a length-81 token tensor back into an 81-char digit string (blanks -> '0')."""
    tokens = tokens.flatten().tolist()
    if len(tokens) != SEQ_LEN:
        raise ValueError(f"Expected {SEQ_LEN} tokens, got {len(tokens)}")

    chars = []
    for tok in tokens:
        digit = int(tok) - 1  # token -> digit
        chars.append(str(digit) if 0 <= digit <= 9 else "?")
    return "".join(chars)


def format_board(board: str) -> str:
    """Render an 81-char board string as a human-readable 9x9 grid with box separators."""
    if len(board) != SEQ_LEN:
        raise ValueError(f"Expected {SEQ_LEN} chars, got {len(board)}")

    lines = []
    for r in range(9):
        row = board[r * 9:(r + 1) * 9]
        cells = [(ch if ch != "0" else ".") for ch in row]
        # Group into 3-cell boxes separated by " | ".
        groups = [" ".join(cells[c:c + 3]) for c in range(0, 9, 3)]
        lines.append(" | ".join(groups))
        if r in (2, 5):
            lines.append("------+-------+------")
    return "\n".join(lines)
