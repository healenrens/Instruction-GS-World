"""Global held-split image-goal bank for deterministic hard negatives."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .goal_conditioning import encode_explicit_goal


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def _weighted_pool(
    slots: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    weight = activity.float().clamp_min(0.0)
    return (slots.float() * weight[..., None]).sum(dim=1) / (
        weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
    )


@torch.no_grad()
def build_goal_bank(model, loader, device: torch.device) -> dict[str, torch.Tensor]:
    fields: dict[str, list[torch.Tensor]] = {
        "slots": [],
        "center": [],
        "activity": [],
        "rgb": [],
        "sequence_index": [],
        "future_times": [],
        "goal_time": [],
    }
    for cpu_batch in loader:
        batch = _move_batch(cpu_batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            history = model.encode_history(batch)
            goal = encode_explicit_goal(model, batch, history)
        for name in ("slots", "center", "activity", "rgb"):
            if name not in goal:
                raise ValueError(f"global goal bank requires {name}")
            fields[name].append(goal[name].detach())
        fields["sequence_index"].append(batch["sequence_index"].detach())
        fields["future_times"].append(batch["future_times"].detach())
        fields["goal_time"].append(batch["goal_time"].detach())
    if not fields["slots"]:
        raise ValueError("global goal bank requires at least one batch")
    bank = {name: torch.cat(chunks) for name, chunks in fields.items()}
    bank["summary"] = _weighted_pool(bank["slots"], bank["activity"])
    return bank


def select_goal_from_bank(
    current_slots: torch.Tensor,
    current_activity: torch.Tensor,
    current_goal: dict[str, torch.Tensor],
    sequence_index: torch.Tensor,
    bank: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    current_summary = _weighted_pool(current_slots, current_activity)
    similarity = (
        F.normalize(current_summary, dim=-1)
        @ F.normalize(bank["summary"], dim=-1).transpose(0, 1)
    )
    different = sequence_index[:, None] != bank["sequence_index"][None]
    if not bool(different.any(dim=1).all()):
        raise ValueError("global goal bank needs another source clip")
    position = similarity.masked_fill(
        ~different,
        torch.finfo(similarity.dtype).min,
    ).argmax(dim=-1)
    wrong = {
        name: bank[name][position]
        for name in ("slots", "center", "activity", "rgb")
    }
    wrong["current_rgb"] = current_goal["current_rgb"]
    selected_similarity = similarity[
        torch.arange(len(position), device=position.device),
        position,
    ]
    return wrong, {
        "cosine_similarity": selected_similarity,
        "different_source": (
            sequence_index != bank["sequence_index"][position]
        ).float(),
        "eligible_candidates": different.sum(dim=1),
    }
