"""Exact matching helpers for small unordered hypothesis sets."""
from __future__ import annotations

import itertools

import torch


def exact_target_component_assignment(cost: torch.Tensor) -> torch.Tensor:
    """Return the minimum-cost target-to-component permutation."""
    if cost.ndim != 3:
        raise ValueError("cost must have shape [G,T,C]")
    targets, components = cost.shape[1:]
    if targets != components:
        raise ValueError("exact set matching requires a square cost matrix")
    permutations = torch.tensor(
        tuple(itertools.permutations(range(components))),
        device=cost.device,
        dtype=torch.long,
    )
    expanded = cost[:, None].expand(-1, len(permutations), -1, -1)
    index = permutations[None, :, :, None].expand(
        cost.shape[0],
        -1,
        -1,
        1,
    )
    permutation_cost = torch.gather(
        expanded,
        dim=3,
        index=index,
    ).squeeze(-1).mean(dim=-1)
    return permutations[permutation_cost.argmin(dim=-1)]


def select_components(
    value: torch.Tensor,
    assignment: torch.Tensor,
) -> torch.Tensor:
    """Gather component-axis values in target order."""
    if value.ndim < 2 or assignment.ndim != 2:
        raise ValueError("value and assignment must start with [G,C] and [G,T]")
    if value.shape[0] != assignment.shape[0]:
        raise ValueError("value and assignment must share the group axis")
    index = assignment.reshape(
        *assignment.shape,
        *((1,) * (value.ndim - 2)),
    ).expand(
        *assignment.shape,
        *value.shape[2:],
    )
    return torch.gather(value, dim=1, index=index)
