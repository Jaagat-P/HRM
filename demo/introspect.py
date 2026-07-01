"""Introspection helpers over an HRM reasoning trajectory.

HRM refines its answer across reasoning steps. These helpers summarize that process:
how many steps until the board stopped changing, and until it first became a valid solution.
This is the hook for studying whether the model spends more steps on harder puzzles.
"""
from dataclasses import dataclass
from typing import List, Optional

from demo.solver import SolveResult
from demo.validate import is_complete_valid


@dataclass
class TrajectoryStats:
    num_steps: int
    stabilized_at: Optional[int]   # 1-indexed step after which the board never changed
    first_valid_at: Optional[int]  # 1-indexed first step whose board is a valid solution
    changes_per_step: List[int]    # number of changed cells vs the previous step


def _cell_diff(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def analyze(result: SolveResult) -> TrajectoryStats:
    traj = result.trajectory

    changes = [0]
    for prev, cur in zip(traj, traj[1:]):
        changes.append(_cell_diff(prev, cur))

    stabilized_at = None
    for i in range(len(traj) - 1, 0, -1):
        if changes[i] != 0:
            stabilized_at = i + 1  # step after last change (1-indexed)
            break
    if stabilized_at is None:
        stabilized_at = 1

    first_valid_at = None
    for i, board in enumerate(traj):
        if is_complete_valid(board):
            first_valid_at = i + 1
            break

    return TrajectoryStats(
        num_steps=result.num_steps,
        stabilized_at=stabilized_at,
        first_valid_at=first_valid_at,
        changes_per_step=changes,
    )
