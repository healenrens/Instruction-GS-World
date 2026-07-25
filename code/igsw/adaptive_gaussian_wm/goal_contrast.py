"""Counterfactual image-goal supervision for the deployable action Prior."""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F


ACTION_ERROR_NORMALIZER_FLOOR = 1e-2


def _gather_without_grad(value: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return value.detach()
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value.detach())
    return torch.cat(gathered, dim=0)


def _weighted_pool(
    slots: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    weight = activity.float().clamp_min(0.0)
    return (slots.float() * weight[..., None]).sum(dim=1) / (
        weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
    )


def select_hard_wrong_goal(
    current_slots: torch.Tensor,
    current_activity: torch.Tensor,
    goal: dict[str, torch.Tensor],
    sequence_index: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Select the closest goal from a different source clip across all ranks."""
    if sequence_index.shape != (current_slots.shape[0],):
        raise ValueError("sequence_index must have shape [B]")
    global_sequence = _gather_without_grad(sequence_index)
    goal_fields = ["slots", "center", "activity"]
    if "rgb" in goal:
        goal_fields.append("rgb")
    global_goal = {
        name: _gather_without_grad(goal[name]) for name in goal_fields
    }
    current_summary = _weighted_pool(current_slots.detach(), current_activity)
    goal_summary = _weighted_pool(
        global_goal["slots"],
        global_goal["activity"],
    )
    similarity = (
        F.normalize(current_summary, dim=-1)
        @ F.normalize(goal_summary, dim=-1).transpose(0, 1)
    )
    different = sequence_index[:, None] != global_sequence[None]
    if not bool(different.any(dim=1).all()):
        raise ValueError("goal contrast requires another source clip per sample")
    position = similarity.masked_fill(
        ~different,
        torch.finfo(similarity.dtype).min,
    ).argmax(dim=-1)
    wrong = {name: value[position] for name, value in global_goal.items()}
    if "current_rgb" in goal:
        wrong["current_rgb"] = goal["current_rgb"]
    metrics = {
        "wrong_goal_cosine_similarity": similarity[
            torch.arange(len(position), device=position.device),
            position,
        ].mean(),
        "wrong_goal_unique_fraction": (
            sequence_index != global_sequence[position]
        ).float().mean(),
    }
    return wrong, metrics


def _flow_endpoint_prediction(
    prior,
    latent: torch.Tensor,
    flow_time: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    if latent.shape[:2] != flow_time.shape:
        raise ValueError("flow state and time must share [B,Q]")
    endpoint_output = bool(prior.endpoint_prediction)
    prediction = prior(latent, flow_time, context, endpoint_output)
    if endpoint_output:
        return prediction
    return latent + (1.0 - flow_time)[..., None, None] * prediction


def deterministic_goal_objective(
    prior,
    target: torch.Tensor,
    correct_context: torch.Tensor,
    wrong_context: torch.Tensor,
    relative_margin: float,
    action_weight: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Anchor the zero-source deployment path and rank the image goal first."""
    if relative_margin < 0.0:
        raise ValueError("goal relative margin must be non-negative")
    if correct_context.shape != wrong_context.shape:
        raise ValueError("correct and wrong goal contexts must align")
    if action_weight.shape != target.shape[:-1]:
        raise ValueError("goal action weight must have shape [B,Q,A]")
    def weighted_sample_error(value: torch.Tensor) -> torch.Tensor:
        channel_error = value.square().mean(dim=-1)
        return (channel_error * action_weight).flatten(1).sum(dim=1) / (
            action_weight.flatten(1).sum(dim=1).clamp_min(1e-6)
        )

    zero_error = weighted_sample_error(target).detach()
    requested_margin = relative_margin * zero_error
    normalizer = zero_error.clamp_min(ACTION_ERROR_NORMALIZER_FLOOR)
    fractions = (0.0, 0.5, 0.9)
    correct_predictions = []
    wrong_predictions = []
    correct_errors = []
    wrong_errors = []
    for fraction in fractions:
        flow_time = target.new_full(target.shape[:2], fraction)
        latent = fraction * target
        correct = _flow_endpoint_prediction(
            prior,
            latent,
            flow_time,
            correct_context,
        )
        wrong = _flow_endpoint_prediction(
            prior,
            latent,
            flow_time,
            wrong_context,
        )
        correct_predictions.append(correct)
        wrong_predictions.append(wrong)
        correct_errors.append(weighted_sample_error(correct - target))
        wrong_errors.append(weighted_sample_error(wrong - target))
    correct_error = torch.stack(correct_errors)
    wrong_error = torch.stack(wrong_errors)
    rank = F.relu(
        (correct_error + requested_margin[None] - wrong_error)
        / normalizer[None]
    )
    correct = correct_predictions[0]
    wrong = wrong_predictions[0]
    path_difference = torch.stack(correct_predictions) - torch.stack(
        wrong_predictions
    )
    return (
        correct_error.mean(),
        rank.mean(),
        {
            "wrong_goal_endpoint_mse": wrong_error.mean(),
            "goal_margin_satisfaction": (
                wrong_error >= correct_error + requested_margin[None]
            ).float().mean(),
            "goal_relative_advantage": (
                (wrong_error - correct_error)
                / normalizer[None]
            ).mean(),
            "goal_rank_normalizer_mean": normalizer.mean(),
            "correct_wrong_action_rms": path_difference.square().mean().sqrt(),
            **{
                f"goal_path_endpoint_mse_t{int(10 * fraction):02d}": (
                    correct_error[index].mean()
                )
                for index, fraction in enumerate(fractions)
            },
        },
        correct,
        wrong,
    )
