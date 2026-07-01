"""Load a pretrained HRM checkpoint for inference on any device (CPU / MPS / CUDA).

The official checkpoints (e.g. ``sapientinc/HRM-checkpoint-sudoku-extreme``) ship two files:
  * ``all_config.yaml`` - the full training config, including the ``arch`` block
  * ``checkpoint``      - a torch state_dict saved from the compiled + loss-wrapped model,
                          so every key is prefixed ``_orig_mod.model.``

This module rebuilds the bare ``HierarchicalReasoningModel_ACTV1``, strips that prefix, and
loads the weights. ``vocab_size`` and ``num_puzzle_identifiers`` are read directly from the
weight shapes so we don't need the dataset metadata to reconstruct the model.
"""
import os
from dataclasses import dataclass
from typing import Optional

import torch
import yaml

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1

_CKPT_PREFIX = "_orig_mod.model."


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class LoadedHRM:
    model: HierarchicalReasoningModel_ACTV1
    device: str
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int
    arch: dict


def _resolve_files(source: str):
    """Return (config_path, checkpoint_path) for a local dir or a HF repo id."""
    if os.path.isdir(source):
        return os.path.join(source, "all_config.yaml"), os.path.join(source, "checkpoint")

    from huggingface_hub import hf_hub_download  # imported lazily so tests don't need network
    return (
        hf_hub_download(source, "all_config.yaml"),
        hf_hub_download(source, "checkpoint"),
    )


def load_hrm(source: str, device: Optional[str] = None, seq_len: int = 81) -> LoadedHRM:
    """Load a pretrained HRM.

    Args:
        source: a Hugging Face repo id (e.g. "sapientinc/HRM-checkpoint-sudoku-extreme")
                or a local directory containing all_config.yaml and checkpoint.
        device: torch device string; auto-detected (cuda > mps > cpu) when None.
        seq_len: sequence length the model was trained on (81 for 9x9 Sudoku). Only affects
                 the size of positional-encoding buffers; must be >= the actual input length.
    """
    device = device or pick_device()
    config_path, checkpoint_path = _resolve_files(source)

    with open(config_path, "r") as f:
        arch = yaml.safe_load(f)["arch"]

    # Load weights first so we can read dimensions straight off the tensors.
    raw_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = {k.removeprefix(_CKPT_PREFIX): v for k, v in raw_state.items()}

    vocab_size = state["inner.embed_tokens.embedding_weight"].shape[0]
    num_puzzle_identifiers = state["inner.puzzle_emb.weights"].shape[0]

    model_cfg = dict(
        {k: v for k, v in arch.items() if k not in ("name", "loss")},
        batch_size=1,  # inference-only; the sparse puzzle-embedding buffer is unused in eval
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_puzzle_identifiers=num_puzzle_identifiers,
    )

    model = HierarchicalReasoningModel_ACTV1(model_cfg)
    model.load_state_dict(state, assign=True)
    model = model.to(device)
    model.eval()

    return LoadedHRM(
        model=model,
        device=device,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_puzzle_identifiers=num_puzzle_identifiers,
        arch=arch,
    )
