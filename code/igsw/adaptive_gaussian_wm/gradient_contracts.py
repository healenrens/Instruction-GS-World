"""Reusable fail-fast gradient contracts for server verification."""

from __future__ import annotations

import torch


def nonzero_gradient_parameter_names(model, prefix: str) -> list[str]:
    names = []
    for name, parameter in model.named_parameters():
        if not name.startswith(prefix) or parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError(f"non-finite gradient in {name}")
        if float(gradient.norm()) > 0.0:
            names.append(name)
    return names
