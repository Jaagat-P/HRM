"""Smoke test: HRM forward pass without FlashAttention (CPU / Apple Silicon MPS).

FlashAttention is CUDA-only, so the upstream code cannot even be imported on a Mac.
models/layers.py now falls back to torch.scaled_dot_product_attention when FlashAttention
is unavailable. This test verifies the model builds and runs a forward pass end-to-end on
whatever device is available (mps > cpu), so contributors without an NVIDIA GPU can iterate.

Run: python -m tests.test_forward_cpu
"""
import dataclasses

import torch

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tiny_config() -> dict:
    # Small enough to run instantly on CPU; seq_len=81 mirrors a 9x9 Sudoku board.
    return dict(
        batch_size=2, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
        puzzle_emb_ndim=0, H_cycles=2, L_cycles=2, H_layers=2, L_layers=2,
        hidden_size=128, expansion=4, num_heads=4, pos_encodings="rope",
        halt_max_steps=4, halt_exploration_prob=0.1,
    )


def move_carry_to(carry, device: str):
    inner = carry.inner_carry
    return dataclasses.replace(
        carry,
        inner_carry=type(inner)(inner.z_H.to(device), inner.z_L.to(device)),
        steps=carry.steps.to(device),
        halted=carry.halted.to(device),
        current_data={k: v.to(device) for k, v in carry.current_data.items()},
    )


def test_forward():
    device = pick_device()
    model = HierarchicalReasoningModel_ACTV1(tiny_config()).to(device)
    model.eval()

    batch = {
        "inputs": torch.randint(0, 11, (2, 81), device=device),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.int32, device=device),
    }
    carry = move_carry_to(model.initial_carry(batch), device)

    with torch.no_grad():
        _, outputs = model(carry, batch)

    assert outputs["logits"].shape == (2, 81, 11), outputs["logits"].shape
    print(f"OK: HRM forward pass runs on {device} without FlashAttention "
          f"(logits {tuple(outputs['logits'].shape)})")


if __name__ == "__main__":
    test_forward()
