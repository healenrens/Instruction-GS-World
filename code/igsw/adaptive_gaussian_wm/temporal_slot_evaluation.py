"""Cross-frame identity and spatial stability metrics for object slots."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .train_runtime import move_to_device


METRICS = (
    "tracking_same_cosine",
    "tracking_best_wrong_cosine",
    "tracking_identity_margin",
    "tracking_retrieval_accuracy",
    "feature_same_cosine",
    "feature_best_wrong_cosine",
    "feature_identity_margin",
    "feature_retrieval_accuracy",
    "center_same_distance",
    "center_nearest_wrong_distance",
    "center_identity_margin",
    "center_retrieval_accuracy",
    "activity_same_difference",
    "activity_nearest_wrong_difference",
    "activity_identity_margin",
    "support_same_cosine",
    "support_best_wrong_cosine",
    "support_identity_margin",
    "support_retrieval_accuracy",
)


def slot_support(token_state, slot_state) -> torch.Tensor:
    if token_state.assignment.shape[:2] != slot_state.assignment.shape[:2]:
        raise ValueError("token and slot assignments do not align")
    micro_object = slot_state.assignment.float() * token_state.activation.float()
    support = torch.einsum(
        "bmk,bmn->bkn",
        micro_object,
        token_state.assignment.float(),
    )
    return support / support.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _weighted(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if values.shape != weight.shape:
        raise ValueError("slot metric and activity weight must align")
    return (values * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1e-6)


def _cosine_correspondence(
    current: torch.Tensor,
    following: torch.Tensor,
    weight: torch.Tensor,
    prefix: str,
) -> dict[str, torch.Tensor]:
    similarity = torch.einsum(
        "btkd,btjd->btkj",
        F.normalize(current.float(), dim=-1),
        F.normalize(following.float(), dim=-1),
    )
    count = similarity.shape[-1]
    identity = torch.arange(count, device=similarity.device)
    same = similarity[..., identity, identity]
    wrong = similarity.masked_fill(
        torch.eye(count, device=similarity.device, dtype=torch.bool)[None, None],
        -torch.inf,
    ).max(dim=-1).values
    retrieval = similarity.argmax(dim=-1) == identity
    return {
        f"{prefix}_same_cosine": _weighted(same, weight),
        f"{prefix}_best_wrong_cosine": _weighted(wrong, weight),
        f"{prefix}_identity_margin": _weighted(same - wrong, weight),
        f"{prefix}_retrieval_accuracy": _weighted(retrieval.float(), weight),
    }


def _distance_correspondence(
    current: torch.Tensor,
    following: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, torch.Tensor]:
    distance = (
        current[:, :, :, None].float() - following[:, :, None].float()
    ).square().sum(dim=-1).sqrt()
    count = distance.shape[-1]
    identity = torch.arange(count, device=distance.device)
    same = distance[..., identity, identity]
    wrong = distance.masked_fill(
        torch.eye(count, device=distance.device, dtype=torch.bool)[None, None],
        torch.inf,
    ).min(dim=-1).values
    retrieval = distance.argmin(dim=-1) == identity
    return {
        "center_same_distance": _weighted(same, weight),
        "center_nearest_wrong_distance": _weighted(wrong, weight),
        "center_identity_margin": _weighted(wrong - same, weight),
        "center_retrieval_accuracy": _weighted(retrieval.float(), weight),
    }


def _activity_correspondence(
    current: torch.Tensor,
    following: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, torch.Tensor]:
    difference = (current[:, :, :, None] - following[:, :, None]).abs().float()
    count = difference.shape[-1]
    identity = torch.arange(count, device=difference.device)
    same = difference[..., identity, identity]
    wrong = difference.masked_fill(
        torch.eye(count, device=difference.device, dtype=torch.bool)[None, None],
        torch.inf,
    ).min(dim=-1).values
    return {
        "activity_same_difference": _weighted(same, weight),
        "activity_nearest_wrong_difference": _weighted(wrong, weight),
        "activity_identity_margin": _weighted(wrong - same, weight),
    }


def sequence_metrics(
    tracking: torch.Tensor,
    features: torch.Tensor,
    centers: torch.Tensor,
    activity: torch.Tensor,
    support: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sequence_shape = tracking.shape[:3]
    if features.shape[:3] != sequence_shape or centers.shape[:3] != sequence_shape:
        raise ValueError("slot state sequences do not align")
    if activity.shape != sequence_shape or support.shape[:3] != sequence_shape:
        raise ValueError("slot activity or support sequence does not align")
    if tracking.shape[1] < 2:
        raise ValueError("temporal slot evaluation requires two frames")
    weight = (
        activity[:, :-1].float().clamp(0.0, 1.0)
        * activity[:, 1:].float().clamp(0.0, 1.0)
    ).sqrt()
    result = {}
    result.update(
        _cosine_correspondence(
            tracking[:, :-1], tracking[:, 1:], weight, "tracking"
        )
    )
    result.update(
        _cosine_correspondence(
            features[:, :-1], features[:, 1:], weight, "feature"
        )
    )
    result.update(
        _distance_correspondence(centers[:, :-1], centers[:, 1:], weight)
    )
    result.update(
        _activity_correspondence(activity[:, :-1], activity[:, 1:], weight)
    )
    result.update(
        _cosine_correspondence(
            support[:, :-1], support[:, 1:], weight, "support"
        )
    )
    if set(result) != set(METRICS):
        raise RuntimeError("temporal slot metrics differ from the declared contract")
    return result


def _stack_states(token_states, slot_states) -> dict[str, torch.Tensor]:
    if len(token_states) != len(slot_states):
        raise ValueError("token and slot state sequence lengths differ")
    return {
        "tracking": torch.stack([state.tracking_slots for state in slot_states], dim=1),
        "features": torch.stack([state.decoded_feature for state in slot_states], dim=1),
        "centers": torch.stack([state.center for state in slot_states], dim=1),
        "activity": torch.stack([state.activity for state in slot_states], dim=1),
        "support": torch.stack(
            [slot_support(token, slot) for token, slot in zip(token_states, slot_states)],
            dim=1,
        ),
    }


def _task_balanced(values: torch.Tensor, clusters: torch.Tensor, layout) -> dict:
    episode_values = []
    episode_tasks = []
    for episode in torch.unique(clusters, sorted=True):
        selected = clusters == episode
        tasks = torch.unique(layout.ids[selected])
        if len(tasks) != 1:
            raise ValueError("one episode maps to multiple temporal-slot tasks")
        episode_values.append(values[selected].mean())
        episode_tasks.append(tasks[0])
    episode_values = torch.stack(episode_values)
    episode_tasks = torch.stack(episode_tasks)
    per_task = {}
    for task_id, name in enumerate(layout.names):
        selected = episode_tasks == task_id
        if not bool(selected.any()):
            raise ValueError(f"temporal slots have no evidence for task {name}")
        per_task[name] = {
            "episodes": int(selected.sum()),
            "mean": float(episode_values[selected].mean()),
        }
    task_values = torch.tensor([entry["mean"] for entry in per_task.values()])
    return {
        "task_count": len(per_task),
        "mean_over_tasks": float(task_values.mean()),
        "worst_task": float(task_values.min()),
        "per_task": per_task,
    }


def _summarize(metrics: dict[str, torch.Tensor], times, clusters, layout) -> dict:
    return {
        "samples": len(clusters),
        "clusters": len(torch.unique(clusters)),
        "transitions": next(iter(metrics.values())).shape[1],
        "chance_retrieval_accuracy": 1.0 / 16.0,
        "mean": {name: float(value.mean()) for name, value in metrics.items()},
        "by_transition": {
            str(index): {
                "from_time_seconds": float(times[:, index].mean()),
                "to_time_seconds": float(times[:, index + 1].mean()),
                "mean": {
                    name: float(value[:, index].mean())
                    for name, value in metrics.items()
                },
            }
            for index in range(next(iter(metrics.values())).shape[1])
        },
        "by_task": {
            name: _task_balanced(value.mean(dim=1), clusters, layout)
            for name, value in metrics.items()
        },
    }


@torch.no_grad()
def evaluate_temporal_slots(model, loader, device, amp, layout) -> dict:
    online_chunks = {name: [] for name in METRICS}
    target_chunks = {name: [] for name in METRICS}
    cluster_chunks = []
    online_time_chunks = []
    target_time_chunks = []
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and amp == "bf16"
            else torch.no_grad()
        )
        with context:
            online = model.encode_history(batch)
            target_history, target_future = model.encode_targets(batch)
        online_state = _stack_states(online["token_states"], online["slot_states"])
        target_state = _stack_states(
            target_history["token_states"] + target_future["token_states"],
            target_history["slot_states"] + target_future["slot_states"],
        )
        online_metrics = sequence_metrics(**online_state)
        target_metrics = sequence_metrics(**target_state)
        for name in METRICS:
            online_chunks[name].append(online_metrics[name].cpu())
            target_chunks[name].append(target_metrics[name].cpu())
        cluster_chunks.append(batch["sequence_index"].long().cpu())
        online_time_chunks.append(batch["history_times"].float().cpu())
        target_time_chunks.append(
            torch.cat((batch["history_times"], batch["future_times"]), dim=1)
            .float().cpu()
        )
    clusters = torch.cat(cluster_chunks)
    online_metrics = {name: torch.cat(values) for name, values in online_chunks.items()}
    target_metrics = {name: torch.cat(values) for name, values in target_chunks.items()}
    chance = 1.0 / model.config.object_slots
    report = {
        "identity_contract": "slot_index_anchored_to_first_history_frame",
        "online_history": _summarize(
            online_metrics, torch.cat(online_time_chunks), clusters, layout
        ),
        "ema_target_history_and_future": _summarize(
            target_metrics, torch.cat(target_time_chunks), clusters, layout
        ),
    }
    for section in report.values():
        if isinstance(section, dict) and "chance_retrieval_accuracy" in section:
            section["chance_retrieval_accuracy"] = chance
    return report
