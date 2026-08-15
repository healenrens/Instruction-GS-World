"""Annotation-free cross-clip consistency proxies for v48 slot sets."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


PAIR_NAMES = (
    "same_episode",
    "same_group_different_episode",
    "different_group",
)


def _set_similarity(
    left: torch.Tensor,
    right: torch.Tensor,
    left_active: torch.Tensor,
    right_active: torch.Tensor,
) -> torch.Tensor:
    if not bool(left_active.any()) or not bool(right_active.any()):
        raise ValueError("v48 cross-episode comparison has no active slot")
    left = F.normalize(left[left_active].float(), dim=-1, eps=1e-6)
    right = F.normalize(right[right_active].float(), dim=-1, eps=1e-6)
    similarity = left @ right.transpose(0, 1)
    return 0.5 * (
        similarity.max(dim=1).values.mean() + similarity.max(dim=0).values.mean()
    )


def _pair_categories(
    episode_ids: torch.Tensor,
    group_ids: torch.Tensor,
) -> dict[str, list[tuple[int, int]]]:
    categories = {name: [] for name in PAIR_NAMES}
    for left in range(len(episode_ids)):
        for right in range(left + 1, len(episode_ids)):
            same_episode = bool(episode_ids[left] == episode_ids[right])
            same_group = bool(group_ids[left] == group_ids[right])
            if same_episode:
                name = "same_episode"
            elif same_group:
                name = "same_group_different_episode"
            else:
                name = "different_group"
            categories[name].append((left, right))
    return categories


def _even_subset(
    pairs: list[tuple[int, int]],
    maximum: int,
) -> list[tuple[int, int]]:
    if len(pairs) <= maximum:
        return pairs
    indices = torch.linspace(0, len(pairs) - 1, maximum).round().long().tolist()
    return [pairs[index] for index in indices]


def _summary(values: list[torch.Tensor]) -> dict[str, float | int | bool]:
    if not values:
        return {"available": False, "pairs": 0}
    tensor = torch.stack(values).float()
    standard_error = float(tensor.std(unbiased=False) / math.sqrt(max(len(tensor), 1)))
    return {
        "available": True,
        "pairs": len(tensor),
        "mean_set_similarity": float(tensor.mean()),
        "standard_error": standard_error,
    }


def cross_episode_consistency(
    slot_features: torch.Tensor,
    active_slots: torch.Tensor,
    episode_ids: torch.Tensor,
    group_ids: torch.Tensor,
    maximum_pairs_per_category: int,
) -> dict:
    if slot_features.ndim != 3 or active_slots.shape != slot_features.shape[:2]:
        raise ValueError("v48 cross-episode slot tensors differ")
    if (
        episode_ids.shape != slot_features.shape[:1]
        or group_ids.shape != episode_ids.shape
    ):
        raise ValueError("v48 cross-episode identifiers differ")
    if maximum_pairs_per_category < 1:
        raise ValueError("v48 cross-episode pair budget must be positive")
    if not bool(active_slots.any(dim=1).all()):
        raise ValueError("v48 cross-episode sample has no active slot")
    categories = _pair_categories(episode_ids, group_ids)
    summaries = {}
    for name, candidates in categories.items():
        values = []
        for left, right in _even_subset(candidates, maximum_pairs_per_category):
            values.append(
                _set_similarity(
                    slot_features[left],
                    slot_features[right],
                    active_slots[left],
                    active_slots[right],
                )
            )
        summaries[name] = _summary(values)
    same_group = summaries["same_group_different_episode"]
    different = summaries["different_group"]
    margin_available = bool(same_group["available"] and different["available"])
    return {
        "contract": (
            "slot-set similarity proxy; sampling_group is a task-level label, "
            "not an object identity annotation"
        ),
        "categories": summaries,
        "same_group_margin_over_different": (
            same_group["mean_set_similarity"] - different["mean_set_similarity"]
            if margin_available
            else None
        ),
        "semantic_object_correspondence_verified": False,
    }
