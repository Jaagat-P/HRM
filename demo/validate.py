"""Validity checks for Sudoku boards and HRM solutions.

Used to verify solver output: a correct solution must (a) be a valid completed Sudoku and
(b) keep every non-blank cell of the original puzzle unchanged.
"""
from dataclasses import dataclass
from typing import List

from demo.sudoku import SEQ_LEN, parse_puzzle_string

_FULL = set("123456789")


def _rows(board: str) -> List[str]:
    return [board[r * 9:(r + 1) * 9] for r in range(9)]


def _cols(board: str) -> List[str]:
    return ["".join(board[r * 9 + c] for r in range(9)) for c in range(9)]


def _boxes(board: str) -> List[str]:
    out = []
    for br in range(3):
        for bc in range(3):
            out.append("".join(
                board[(br * 3 + i) * 9 + (bc * 3 + j)] for i in range(3) for j in range(3)
            ))
    return out


def is_complete_valid(board: str) -> bool:
    """True if board is a fully-filled, rule-respecting 9x9 Sudoku."""
    if len(board) != SEQ_LEN:
        return False
    units = _rows(board) + _cols(board) + _boxes(board)
    return all(set(unit) == _FULL for unit in units)


def respects_givens(puzzle: str, solution: str) -> bool:
    """True if every non-blank cell of the original puzzle is unchanged in the solution."""
    givens = parse_puzzle_string(puzzle)
    if len(solution) != SEQ_LEN:
        return False
    for i, g in enumerate(givens):
        if g != 0 and str(g) != solution[i]:
            return False
    return True


@dataclass
class SolutionReport:
    complete_valid: bool
    respects_givens: bool

    @property
    def correct(self) -> bool:
        return self.complete_valid and self.respects_givens


def check_solution(puzzle: str, solution: str) -> SolutionReport:
    """Full check that a solution solves the given puzzle correctly."""
    return SolutionReport(
        complete_valid=is_complete_valid(solution),
        respects_givens=respects_givens(puzzle, solution),
    )
