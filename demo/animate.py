"""Animate an HRM reasoning trajectory in the terminal.

Replays the per-step board predictions in place, so you can watch the solution emerge across
reasoning steps. Cells that changed since the previous step are highlighted.
"""
import sys
import time
from typing import List

from demo.sudoku import format_board

_CLEAR = "\033[2J\033[H"   # clear screen + move cursor home
_HL = "\033[93m"           # bright yellow
_RESET = "\033[0m"


def _highlight_changes(board: str, prev: str) -> str:
    """Render a board grid, highlighting cells that differ from the previous board."""
    # Walk the formatted grid and colorize digit cells that changed since the previous step.
    plain = format_board(board)
    out, idx = [], 0
    for ch in plain:
        if ch.isdigit() or ch == ".":
            orig = board[idx]
            changed = prev and orig != prev[idx]
            display = "." if orig == "0" else orig
            out.append(f"{_HL}{display}{_RESET}" if changed else display)
            idx += 1
        else:
            out.append(ch)
    return "".join(out)


def animate(trajectory: List[str], delay: float = 0.4, stream=sys.stdout) -> None:
    prev = ""
    for i, board in enumerate(trajectory, 1):
        stream.write(_CLEAR)
        stream.write(f"HRM reasoning - step {i}/{len(trajectory)}\n\n")
        stream.write(_highlight_changes(board, prev))
        stream.write("\n")
        stream.flush()
        prev = board
        time.sleep(delay)
