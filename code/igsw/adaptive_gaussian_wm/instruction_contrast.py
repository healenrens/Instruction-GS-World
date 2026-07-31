"""Counterfactual instruction supervision for the deployable action Prior."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .distributed_statistics import (
    gather_batch_with_grad,
    gather_batch_without_grad,
)


def _gather_without_grad(value: torch.Tensor) -> torch.Tensor:
    return gather_batch_without_grad(value)


def hard_wrong_condition(
    condition: torch.Tensor,
    raw_condition: torch.Tensor,
    condition_index: torch.Tensor,
    task_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose a different instruction with the closest frozen Qwen feature."""
    if condition.ndim != 2 or raw_condition.ndim != 2:
        raise ValueError("condition tensors must have shape [B,D]")
    if condition.shape[0] != raw_condition.shape[0]:
        raise ValueError("projected and raw condition batches must align")
    if condition_index.shape != (condition.shape[0],):
        raise ValueError("condition_index must have shape [B]")
    if task_index.shape != condition_index.shape:
        raise ValueError("task_index must have shape [B]")

    global_raw = _gather_without_grad(raw_condition.detach())
    global_index = _gather_without_grad(condition_index.detach())
    global_task = _gather_without_grad(task_index.detach())
    similarity = (
        F.normalize(raw_condition.detach().float(), dim=-1)
        @ F.normalize(global_raw.float(), dim=-1).transpose(0, 1)
    )
    different = (
        (condition_index[:, None] != global_index[None])
        & (task_index[:, None] != global_task[None])
    )
    if not bool(different.any(dim=1).all()):
        raise ValueError(
            "instruction contrast requires a different task per sample"
        )
    negative_infinity = torch.finfo(similarity.dtype).min
    wrong_position = similarity.masked_fill(
        ~different,
        negative_infinity,
    ).argmax(dim=-1)
    global_condition = gather_batch_with_grad(condition)
    return (
        global_condition[wrong_position],
        global_index[wrong_position],
        wrong_position,
    )


def gathered_wrong_tokens(
    token_features: torch.Tensor,
    token_valid: torch.Tensor,
    wrong_position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if token_features.ndim != 3:
        raise ValueError("instruction tokens must have shape [B,L,D]")
    if token_valid.shape != token_features.shape[:2]:
        raise ValueError("instruction token mask must have shape [B,L]")
    global_tokens = _gather_without_grad(token_features.detach())
    global_valid = _gather_without_grad(token_valid.detach())
    return global_tokens[wrong_position], global_valid[wrong_position]


def flow_endpoint_prediction(
    prior,
    latent: torch.Tensor,
    flow_time: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    """Convert the Prior output at a known flow state into an endpoint."""
    if latent.shape[:2] != flow_time.shape:
        raise ValueError("flow state and time must share [B,Q]")
    endpoint_output = bool(prior.endpoint_prediction)
    prediction = prior(latent, flow_time, context, endpoint_output)
    if endpoint_output:
        return prediction
    remaining = (1.0 - flow_time)[..., None, None]
    return latent + remaining * prediction


def deterministic_instruction_objective(
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
    """Anchor zero-source inference and rank its correct instruction first."""
    if relative_margin < 0.0:
        raise ValueError("instruction relative margin must be non-negative")
    if correct_context.shape != wrong_context.shape:
        raise ValueError("correct and wrong Prior contexts must align")
    if action_weight.shape != target.shape[:-1]:
        raise ValueError("instruction action weight must have shape [B,Q,A]")
    latent = torch.zeros_like(target)
    flow_time = target.new_zeros(target.shape[:2])
    correct = flow_endpoint_prediction(
        prior,
        latent,
        flow_time,
        correct_context,
    )
    wrong = flow_endpoint_prediction(
        prior,
        latent,
        flow_time,
        wrong_context,
    )
    def weighted_sample_error(value: torch.Tensor) -> torch.Tensor:
        channel_error = value.square().mean(dim=-1)
        return (channel_error * action_weight).flatten(1).sum(dim=1) / (
            action_weight.flatten(1).sum(dim=1).clamp_min(1e-6)
        )

    correct_error = weighted_sample_error(correct - target)
    wrong_error = weighted_sample_error(wrong - target)
    zero_error = weighted_sample_error(target).detach()
    requested_margin = relative_margin * zero_error
    rank = F.relu(correct_error + requested_margin - wrong_error)
    relative_advantage = (wrong_error - correct_error) / zero_error.clamp_min(
        1e-8
    )
    return (
        correct_error.mean(),
        rank.mean(),
        {
            "wrong_instruction_endpoint_mse": wrong_error.mean(),
            "instruction_margin_satisfaction": (
                wrong_error >= correct_error + requested_margin
            ).float().mean(),
            "instruction_relative_advantage": relative_advantage.mean(),
            "correct_wrong_action_rms": (
                correct - wrong
            ).square().mean().sqrt(),
        },
        correct,
        wrong,
    )
