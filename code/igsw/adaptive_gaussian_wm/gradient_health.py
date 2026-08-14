"""Fail-fast gradient auditing with parameter-level diagnostics."""
from __future__ import annotations

import torch


def _stable_total_norm(gradients: list[torch.Tensor]) -> torch.Tensor:
    if not gradients:
        raise RuntimeError("no gradients were produced")
    maximum = torch.stack(
        [gradient.detach().abs().max().float() for gradient in gradients]
    ).max()
    if float(maximum) == 0.0:
        return maximum.double()
    scaled_squares = torch.stack(
        [
            (gradient.detach().float() / maximum).square().sum()
            for gradient in gradients
        ]
    )
    return maximum.double() * scaled_squares.double().sum().sqrt()


@torch.no_grad()
def optimizer_group_grad_norms(
    optimizer: torch.optim.Optimizer,
) -> dict[str, torch.Tensor]:
    """Report pre-clipping gradient norms for each named optimizer group."""
    result = {}
    for index, group in enumerate(optimizer.param_groups):
        gradients = [
            parameter.grad
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if not gradients:
            continue
        norm = _stable_total_norm(gradients)
        name = group.get("group_name", str(index))
        result[f"grad_norm_{name}"] = norm
    return result


@torch.no_grad()
def parameter_prefix_grad_norms(
    named_parameters,
    groups: dict[str, tuple[str, ...]],
) -> dict[str, torch.Tensor]:
    entries = [
        (name, parameter.grad)
        for name, parameter in named_parameters
        if parameter.grad is not None
    ]
    result = {}
    for group_name, prefixes in groups.items():
        gradients = [
            gradient
            for name, gradient in entries
            if name.startswith(prefixes)
        ]
        if gradients:
            result[f"grad_norm_{group_name}"] = _stable_total_norm(gradients)
    return result


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
    gradients = [parameter.grad for _, parameter in entries]
    total_norm = _stable_total_norm(gradients)
    if not bool(torch.isfinite(total_norm)):
        raise RuntimeError("finite gradients produced a non-finite stable total norm")
    coefficient = (max_norm / total_norm.clamp_min(1e-12)).clamp(max=1.0)
    for gradient in gradients:
        gradient.mul_(coefficient.to(device=gradient.device, dtype=gradient.dtype))
    return total_norm
