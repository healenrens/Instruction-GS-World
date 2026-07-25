"""Training-aligned diagnostics for posterior object effects."""
from __future__ import annotations

import torch

from .action_embedding import effect_supervision_actions


def posterior_effect_batch(
    model,
    output: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target, prediction, and weights using the training convention."""
    activity = output["target_future_activity"].float()
    effect_actions = effect_supervision_actions(
        output["posterior_actions"],
        model.config.canonical_action_dim,
    )
    if model.config.object_aligned_actions:
        target = (
            output["target_future_slots"]
            - output["online_history_slots"][:, -1, None]
        ).float()
        prediction = model.latent_actions.predict_object_effect(
            effect_actions
        ).float()
        return target, prediction, activity

    object_effect = (
        output["target_future_slots"]
        - output["target_history_slots"][:, -1, None]
    ).float()
    denominator = activity.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    target = (
        object_effect * activity[..., None]
    ).sum(dim=-2) / denominator
    prediction = model.latent_actions.predict_effect(
        effect_actions
    ).float()
    return target, prediction, torch.ones_like(activity[..., 0])


def summarize_posterior_effects(
    target_chunks: list[torch.Tensor],
    prediction_chunks: list[torch.Tensor],
    weight_chunks: list[torch.Tensor],
    relation_items: int = 512,
) -> dict[str, float]:
    """Summarize weighted effect quality without collapsing object identity."""
    target = torch.cat(target_chunks).float()
    prediction = torch.cat(prediction_chunks).float()
    weight = torch.cat(weight_chunks).float()
    if target.shape != prediction.shape or target.shape[:-1] != weight.shape:
        raise ValueError("posterior effect diagnostics received incompatible shapes")

    target = target.reshape(-1, target.shape[-1])
    prediction = prediction.reshape_as(target)
    weight = weight.reshape(-1).clamp_min(0.0)
    weight_sum = weight.sum().clamp_min(1e-6)
    expanded_weight = weight[:, None]
    residual = prediction - target
    mse = (
        residual.square() * expanded_weight
    ).sum() / (weight_sum * target.shape[-1])

    target_mean = (
        target * expanded_weight
    ).sum(dim=0, keepdim=True) / weight_sum
    residual_energy = (residual.square() * expanded_weight).sum()
    target_energy = (
        (target - target_mean).square() * expanded_weight
    ).sum().clamp_min(1e-8)
    cosine = torch.nn.functional.cosine_similarity(
        prediction,
        target,
        dim=-1,
    )
    direction = (cosine * weight).sum() / weight_sum

    active = torch.nonzero(weight > 0.0, as_tuple=False).flatten()
    if not len(active):
        relation_mse = target.new_zeros(())
    else:
        if len(active) > relation_items:
            positions = torch.linspace(
                0,
                len(active) - 1,
                relation_items,
            ).round().long()
            active = active[positions]
        predicted_code = torch.nn.functional.normalize(
            prediction[active],
            dim=-1,
        )
        target_code = torch.nn.functional.normalize(
            target[active],
            dim=-1,
        )
        predicted_relation = predicted_code @ predicted_code.transpose(0, 1)
        target_relation = target_code @ target_code.transpose(0, 1)
        relation_mse = (
            predicted_relation - target_relation
        ).square().mean()

    return {
        "effect_prediction_mse": float(mse),
        "effect_prediction_r2": float(1.0 - residual_energy / target_energy),
        "effect_direction_cosine": float(direction),
        "action_effect_relation_mse": float(relation_mse),
        "effect_active_weight": float(weight.sum()),
        "effect_items": float(len(weight)),
    }
