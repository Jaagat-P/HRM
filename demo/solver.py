"""Run HRM inference on a single puzzle and capture the step-by-step reasoning trajectory.

HRM uses Adaptive Computation Time (ACT): each call to the model performs one "reasoning
step" (internally H_cycles x L_cycles recurrent passes) and decides whether to halt. During
evaluation the model runs a fixed number of steps (halt_max_steps), so by calling it in a
loop and decoding the logits after each step we get the full trajectory of intermediate
board predictions - i.e. we can watch the solution emerge.
"""
import dataclasses
from dataclasses import dataclass
from typing import List, Optional

import torch

from demo.checkpoint import LoadedHRM
from demo.sudoku import decode_board, encode_puzzle


def _carry_to_device(carry, device: str):
    inner = carry.inner_carry
    return dataclasses.replace(
        carry,
        inner_carry=type(inner)(inner.z_H.to(device), inner.z_L.to(device)),
        steps=carry.steps.to(device),
        halted=carry.halted.to(device),
        current_data={k: v.to(device) for k, v in carry.current_data.items()},
    )


@dataclass
class SolveResult:
    final_board: str            # 81-char solved board
    trajectory: List[str]       # one 81-char board per reasoning step
    num_steps: int              # reasoning steps taken before halting


@torch.no_grad()
def solve_tokens(loaded: LoadedHRM, input_tokens: torch.Tensor, max_steps: Optional[int] = None) -> SolveResult:
    """Solve a puzzle given a [1, seq_len] token tensor, returning the per-step trajectory."""
    model, device = loaded.model, loaded.device

    batch = {
        "inputs": input_tokens.to(device),
        "puzzle_identifiers": torch.zeros(input_tokens.shape[0], dtype=torch.int32, device=device),
    }
    carry = _carry_to_device(model.initial_carry(batch), device)

    trajectory: List[str] = []
    while True:
        carry, outputs = model(carry, batch)
        prediction = outputs["logits"].argmax(dim=-1)  # [1, seq_len]
        trajectory.append(decode_board(prediction[0]))

        if bool(carry.halted.all()) or (max_steps is not None and len(trajectory) >= max_steps):
            break

    return SolveResult(final_board=trajectory[-1], trajectory=trajectory, num_steps=len(trajectory))


def solve_puzzle(loaded: LoadedHRM, puzzle: str, max_steps: Optional[int] = None) -> SolveResult:
    """Solve a Sudoku given an 81-char puzzle string."""
    tokens = encode_puzzle(puzzle, device=loaded.device)
    return solve_tokens(loaded, tokens, max_steps=max_steps)
