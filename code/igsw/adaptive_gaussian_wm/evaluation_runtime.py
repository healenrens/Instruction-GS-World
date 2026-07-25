"""Runtime state helpers for deterministic counterfactual evaluation."""
from __future__ import annotations

import torch


def rng_state(device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    return torch.random.get_rng_state()


def restore_rng_state(state: torch.Tensor, device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    else:
        torch.random.set_rng_state(state)
