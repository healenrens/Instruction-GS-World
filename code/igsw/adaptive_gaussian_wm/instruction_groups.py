"""Task-group identities for instruction positives and counterfactuals."""
from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F


def stable_task_index(task: str) -> int:
    if not isinstance(task, str) or not task.strip():
        raise ValueError("causal pair task must be a non-empty string")
    digest = hashlib.sha256(task.strip().encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def build_condition_task_bank(
    paths: list[str],
    condition_store,
) -> torch.Tensor:
    bank = torch.full(
        (len(condition_store.features),),
        -1,
        dtype=torch.long,
    )
    for path in paths:
        pair = torch.load(path, map_location="cpu", weights_only=False)
        condition_index = condition_store.lookup_index(
            pair.get("instruction", "")
        )
        task_index = stable_task_index(pair.get("task", ""))
        previous = int(bank[condition_index])
        if previous >= 0 and previous != task_index:
            raise ValueError("one instruction maps to multiple task groups")
        bank[condition_index] = task_index
    if not bool((bank >= 0).any()):
        raise ValueError("condition task bank contains no observed instruction")
    return bank


def build_paraphrase_index_bank(
    condition_task_bank: torch.Tensor,
) -> torch.Tensor:
    if condition_task_bank.ndim != 1:
        raise ValueError("condition task bank must have shape [U]")
    result = torch.full_like(condition_task_bank, -1)
    positions = torch.arange(len(condition_task_bank))
    for index, task_index in enumerate(condition_task_bank.tolist()):
        if task_index < 0:
            continue
        candidate = (
            (condition_task_bank == task_index)
            & (positions != index)
        )
        if bool(candidate.any()):
            result[index] = int(candidate.nonzero()[0])
    return result


def select_different_task_condition(
    condition_index: torch.Tensor,
    task_index: torch.Tensor,
    condition_task_bank: torch.Tensor,
    condition_features: torch.Tensor,
    hard_rank: int = 0,
) -> torch.Tensor:
    if condition_index.ndim != 1 or task_index.shape != condition_index.shape:
        raise ValueError("condition and task indices must have shape [B]")
    if condition_task_bank.ndim != 1:
        raise ValueError("condition task bank must have shape [U]")
    if (
        condition_features.ndim != 2
        or condition_features.shape[0] != len(condition_task_bank)
    ):
        raise ValueError("condition features must have shape [U,D]")
    observed = condition_task_bank >= 0
    different = (
        observed[None]
        & (condition_task_bank[None] != task_index[:, None])
    )
    if not bool(different.any(dim=1).all()):
        raise ValueError("wrong instruction requires a different observed task")
    if hard_rank < 0 or bool((different.sum(dim=1) <= hard_rank).any()):
        raise ValueError("wrong task rank exceeds available counterfactuals")
    normalized = F.normalize(condition_features.float(), dim=-1)
    similarity = normalized[condition_index] @ normalized.transpose(0, 1)
    ranking = similarity.masked_fill(
        ~different,
        -torch.inf,
    ).argsort(dim=1, descending=True)
    return ranking[:, hard_rank]
