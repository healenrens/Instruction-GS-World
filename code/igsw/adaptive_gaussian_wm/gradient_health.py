"""Fail-fast gradient auditing with parameter-level diagnostics."""
from __future__ import annotations

import torch


@torch.no_grad()
def clip_finite_grad_norm_(
    named_parameters,
    max_norm: float,
) -> torch.Tensor:
    entries = [
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.grad is not None
    ]
    if not entries:
        raise RuntimeError("no gradients were produced")
    finite = torch.stack(
        [torch.isfinite(parameter.grad).all() for _, parameter in entries]
    )
    if not bool(finite.all()):
        flags = finite.cpu().tolist()
        offenders = [
            name
            for (name, _), is_finite in zip(entries, flags, strict=True)
            if not is_finite
        ]
        raise RuntimeError(
            "non-finite gradients in parameters: " + ", ".join(offenders)
        )
    return torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in entries],
        max_norm,
        error_if_nonfinite=True,
    )
