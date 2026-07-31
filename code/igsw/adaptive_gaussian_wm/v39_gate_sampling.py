"""Deterministic, data-diverse batch selection for the v39 CUDA gate."""

from __future__ import annotations

import json

import torch
from torch.utils.data import default_collate


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def _scan_summary(
    indices: list[int],
    sequence_indices: list[int],
    valid: torch.Tensor,
    content_valid: torch.Tensor,
    stability: torch.Tensor,
) -> dict:
    quantiles = torch.quantile(
        stability.float(), torch.tensor([0.0, 0.5, 0.9, 1.0])
    )
    return {
        "goal_gate_candidate_indices": indices,
        "goal_gate_candidate_count": len(indices),
        "goal_gate_episode_count": len(set(sequence_indices)),
        "goal_gate_valid_count": int(valid.sum()),
        "goal_gate_valid_fraction": float(valid.float().mean()),
        "goal_gate_content_valid_fraction": float(content_valid.float().mean()),
        "goal_gate_stability_min": float(quantiles[0]),
        "goal_gate_stability_median": float(quantiles[1]),
        "goal_gate_stability_p90": float(quantiles[2]),
        "goal_gate_stability_max": float(quantiles[3]),
    }


def select_v39_gate_batch(
    dataset,
    runtime,
    device: torch.device,
    candidate_count: int,
    batch_size: int = 4,
) -> tuple[dict, dict]:
    """Scan evenly selected dataset items and retain the strongest gate batch."""
    if batch_size < 2:
        raise ValueError("v39 gate batch must contain at least two samples")
    if candidate_count < batch_size:
        raise ValueError("v39 goal scan is smaller than one gate batch")
    candidate_count = min(candidate_count, len(dataset))
    candidate_count -= candidate_count % batch_size
    if candidate_count < batch_size:
        raise RuntimeError("v39 dataset is smaller than one complete gate batch")

    scanned_indices: list[int] = []
    sequence_indices: list[int] = []
    validity = []
    content_validity = []
    stability_errors = []
    best_batch = None
    best_indices: list[int] = []
    best_sequences: list[int] = []
    best_score: tuple[int, float] | None = None

    for start in range(0, candidate_count, batch_size):
        indices = list(range(start, start + batch_size))
        raw = default_collate([dataset[(index, 4)] for index in indices])
        batch = runtime(_to_device(raw, device))
        valid = batch["future_horizon_valid"][:, 1].detach().cpu()
        content = batch["goal_content_valid"].detach().cpu()
        stability = batch["goal_stability_error"].detach().float().cpu()
        sequences = batch["sequence_index"].detach().cpu().tolist()

        scanned_indices.extend(indices)
        sequence_indices.extend(int(value) for value in sequences)
        validity.append(valid)
        content_validity.append(content)
        stability_errors.append(stability)
        content_stability = stability[content]
        mean_stability = float(
            content_stability.mean() if len(content_stability) else stability.mean()
        )
        score = (int(valid.sum()), -mean_stability)
        if best_score is None or score > best_score:
            if best_batch is not None:
                del best_batch
            best_batch = batch
            best_indices = indices
            best_sequences = [int(value) for value in sequences]
            best_score = score
        else:
            del batch

    valid = torch.cat(validity)
    content_valid = torch.cat(content_validity)
    stability = torch.cat(stability_errors)
    summary = _scan_summary(
        scanned_indices,
        sequence_indices,
        valid,
        content_valid,
        stability,
    )
    summary.update(
        goal_gate_selected_indices=best_indices,
        goal_gate_selected_sequence_indices=best_sequences,
        goal_gate_selected_valid_count=best_score[0] if best_score else 0,
    )
    if summary["goal_gate_episode_count"] < batch_size:
        raise RuntimeError(
            "v39 goal scan did not cover four episodes: "
            + json.dumps(summary, sort_keys=True)
        )
    if best_batch is None or not bool(valid.any()):
        raise RuntimeError(
            "v39 goal scan found no stable terminal target: "
            + json.dumps(summary, sort_keys=True)
        )
    return best_batch, summary
