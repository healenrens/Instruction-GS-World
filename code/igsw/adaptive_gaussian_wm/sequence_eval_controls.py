"""Causal controls and grouped summaries for visual-sequence evaluation."""
from __future__ import annotations

import torch


def ablate_history(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove visual motion while preserving history length and timestamps."""
    result = dict(batch)
    for name in (
        "history_features",
        "history_coordinates",
        "history_valid",
        "history_rgb",
        "history_rgb_valid",
    ):
        value = result.get(name)
        if value is not None:
            result[name] = value[:, -1:].expand_as(value)
    return result


def ablate_future_time(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Give every future query the same mean horizon."""
    result = dict(batch)
    times = batch["future_times"]
    result["future_times"] = times.mean(dim=1, keepdim=True).expand_as(times)
    return result


def empty_history_mask(
    batch: dict[str, torch.Tensor],
    slots: int,
) -> torch.Tensor:
    return torch.zeros(
        batch["history_features"].shape[0],
        batch["history_features"].shape[1],
        slots,
        device=batch["history_features"].device,
        dtype=torch.bool,
    )


def grouped_metric_means(
    tensors: dict[str, dict[str, torch.Tensor]],
    group_ids: torch.Tensor,
) -> list[dict]:
    """Aggregate every prediction metric by an integer sample group."""
    if group_ids.ndim != 1:
        raise ValueError("group ids must have shape [N]")
    results = []
    for group in torch.unique(group_ids, sorted=True):
        mask = group_ids == group
        metrics = {}
        for metric, predictions in tensors.items():
            metrics[metric] = {}
            for name, values in predictions.items():
                if values.shape[0] != len(group_ids):
                    raise ValueError("metric samples and group ids differ")
                metrics[metric][name] = float(values[mask].mean())
        results.append(
            {
                "group": int(group),
                "samples": int(mask.sum()),
                "mean": metrics,
            }
        )
    return results
