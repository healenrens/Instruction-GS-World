"""Information constraints for continuous latent actions."""
from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .action_embedding import effect_supervision_actions
from .distributed_statistics import (
    gather_batch_with_grad,
    statistical_batch_size,
)
from .jepa_losses import weighted_mean
from .scale import signed_gap_scale


def sparse_action_regularization(
    actions: torch.Tensor,
    l1_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Promote sparse actions without allowing scale or dimension collapse."""
    if actions.ndim != 4:
        raise ValueError("actions must have shape [B,Q,A,D]")
    if l1_weight <= 0.0:
        raise ValueError("l1_weight must be positive")

    codes = actions.flatten(-2).flatten(0, 1)
    dimension = codes.shape[-1]
    norm_floor = F.relu(
        math.sqrt(dimension) - codes.norm(dim=-1)
    ).mean()
    sparsity = codes.abs().mean()

    centered = codes - codes.mean(dim=0, keepdim=True)
    variance = centered.square().mean(dim=0)
    variance_floor = F.relu(
        1.0 - torch.sqrt(variance + 1e-4)
    ).mean()
    covariance = (
        centered.transpose(0, 1) @ centered
        / max(centered.shape[0] - 1, 1)
    )
    off_diagonal = ~torch.eye(
        dimension,
        device=codes.device,
        dtype=torch.bool,
    )
    covariance_loss = covariance.square().masked_select(
        off_diagonal
    ).mean()
    mean_loss = codes.mean(dim=0).square().mean()

    total = (
        norm_floor
        + l1_weight * sparsity
        + 0.1 * variance_floor
        + 0.001 * covariance_loss
        + 0.1 * mean_loss
    )
    return total, {
        "total": total,
        "norm_floor": norm_floor,
        "sparsity": sparsity,
        "variance_floor": variance_floor,
        "covariance": covariance_loss,
        "mean": mean_loss,
    }


def effect_aligned_action_loss(
    model,
    output: dict,
    future_loss: torch.Tensor,
    zero_action_loss: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Align posterior codes to effects using the full distributed microbatch."""
    action_use = F.relu(future_loss - zero_action_loss + 1e-3)
    posterior_actions = output["posterior_actions"]
    canonical_dim = model.config.canonical_action_dim
    effect_actions = effect_supervision_actions(
        posterior_actions,
        canonical_dim,
    )
    current_slots = (
        output["online_history_slots"]
        if model.config.object_aligned_actions
        else output["target_history_slots"]
    )
    object_effect = (
        output["target_future_slots"]
        - current_slots[:, -1, None]
    ).detach()
    effect_weight = output["target_future_activity"].detach()
    if model.config.object_aligned_actions:
        predicted_effect = model.latent_actions.predict_object_effect(
            effect_actions
        )
        target_effect = object_effect
        effect_direction = weighted_mean(
            (
                F.normalize(predicted_effect, dim=-1)
                - F.normalize(target_effect, dim=-1)
            ).square(),
            effect_weight,
        )
        effect_magnitude = weighted_mean(
            F.smooth_l1_loss(
                predicted_effect.norm(dim=-1),
                target_effect.norm(dim=-1),
                beta=0.05,
                reduction="none",
            ),
            effect_weight,
        )
        local_action_codes = effect_actions.flatten(0, 2)
        local_effect_codes = (
            object_effect * effect_weight[..., None]
        ).flatten(0, 2)
    else:
        effect_denominator = effect_weight.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        target_effect = (
            object_effect * effect_weight[..., None]
        ).sum(dim=-2) / effect_denominator
        predicted_effect = model.latent_actions.predict_effect(
            effect_actions
        )
        effect_direction = F.mse_loss(
            F.normalize(predicted_effect, dim=-1),
            F.normalize(target_effect, dim=-1),
        )
        effect_magnitude = F.smooth_l1_loss(
            predicted_effect.norm(dim=-1),
            target_effect.norm(dim=-1),
            beta=0.05,
        )
        local_action_codes = effect_actions.flatten(-2).flatten(0, 1)
        local_effect_codes = (
            object_effect * effect_weight[..., None]
        ).flatten(-2).flatten(0, 1)
    action_codes = gather_batch_with_grad(
        local_action_codes
    )
    effect_codes = gather_batch_with_grad(
        local_effect_codes
    )
    action_codes = F.normalize(
        action_codes - action_codes.mean(dim=0, keepdim=True),
        dim=-1,
    )
    effect_codes = F.normalize(
        effect_codes - effect_codes.mean(dim=0, keepdim=True),
        dim=-1,
    )
    action_relations = action_codes @ action_codes.transpose(0, 1)
    effect_relations = effect_codes @ effect_codes.transpose(0, 1)
    effect_structure = F.mse_loss(
        action_relations,
        effect_relations.detach(),
    )
    semantic_basis_reconstruction = future_loss * 0.0
    semantic_basis_orthogonality = future_loss * 0.0
    if model.config.learned_semantic_action_basis:
        basis = model.latent_actions.posterior.semantic_basis()
        projected = object_effect @ basis
        reconstructed = projected @ basis.transpose(0, 1)
        relative_error = (
            (reconstructed - object_effect).square().sum(dim=-1)
            / object_effect.square().sum(dim=-1).clamp_min(1e-6)
        )
        semantic_basis_reconstruction = weighted_mean(
            relative_error,
            effect_weight,
        )
        gram = basis.transpose(0, 1) @ basis
        semantic_basis_orthogonality = (
            gram - torch.eye(gram.shape[0], device=gram.device)
        ).square().mean()
    semantic_basis_loss = (
        semantic_basis_reconstruction + semantic_basis_orthogonality
    )
    structure_weight = 0.0 if canonical_dim else 0.5
    effect_alignment = (
        effect_direction + 0.1 * effect_magnitude
        + structure_weight * effect_structure
    )

    if model.latent_actions.center_effect_head is not None:
        predicted_center_effect = model.latent_actions.predict_center_effect(
            effect_actions
        )
        target_center_effect = (
            output["target_future_centers"]
            - (
                output["online_history_centers"]
                if model.config.object_aligned_actions
                else output["target_history_centers"]
            )[:, -1, None]
        )
        center_effect = weighted_mean(
            F.smooth_l1_loss(
                predicted_center_effect,
                target_center_effect.detach(),
                reduction="none",
                beta=0.05,
            ),
            output["target_future_activity"].detach(),
        )
    else:
        center_effect = posterior_actions.sum() * 0.0

    regularized_actions = (
        posterior_actions[..., canonical_dim:]
        if canonical_dim < posterior_actions.shape[-1]
        else posterior_actions
    )
    local_action_flat = regularized_actions.flatten(0, 2)
    action_flat = gather_batch_with_grad(local_action_flat)
    action_std = action_flat.std(dim=0, unbiased=False)
    action_variance = F.relu(0.5 - action_std).mean()
    centered_action = action_flat - action_flat.mean(dim=0, keepdim=True)
    action_covariance = (
        centered_action.transpose(0, 1) @ centered_action
        / max(centered_action.shape[0] - 1, 1)
    )
    covariance_mask = ~torch.eye(
        action_covariance.shape[0],
        device=action_covariance.device,
        dtype=torch.bool,
    )
    action_covariance_loss = action_covariance.square().masked_select(
        covariance_mask
    ).mean()
    anti_collapse = (
        0.1 * action_variance + 0.01 * action_covariance_loss
        if model.config.normalize_posterior
        else action_variance * 0.0
    )
    total = (
        action_use
        + effect_alignment
        + 2.0 * center_effect
        + anti_collapse
        + model.config.semantic_action_basis_weight * semantic_basis_loss
    )
    statistical_count = statistical_batch_size(
        local_action_flat.shape[0],
        local_action_flat.device,
    )
    effect_statistical_count = torch.tensor(
        float(effect_codes.shape[0]),
        device=local_action_flat.device,
    )
    dynamics_residual = output["dynamics_actions"][..., canonical_dim:]
    residual_keep_fraction = (
        (dynamics_residual.abs().sum(dim=-1) > 0.0).float().mean()
        if dynamics_residual.shape[-1]
        else dynamics_residual.new_zeros(())
    )
    return total, {
        "action_use": action_use,
        "effect_alignment": effect_alignment,
        "effect_direction": effect_direction,
        "effect_magnitude": effect_magnitude,
        "effect_structure": effect_structure,
        "semantic_basis_reconstruction": semantic_basis_reconstruction,
        "semantic_basis_orthogonality": semantic_basis_orthogonality,
        "center_effect": center_effect,
        "action_variance": action_variance,
        "action_covariance": action_covariance_loss,
        "action_anti_collapse": anti_collapse,
        "action_statistical_samples": statistical_count,
        "effect_statistical_samples": effect_statistical_count,
        "action_residual_keep_fraction": residual_keep_fraction,
    }


def _weighted_per_sample(
    error: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return (error * weight).flatten(1).sum(dim=1) / weight.flatten(1).sum(
        dim=1
    ).clamp_min(1e-6)


def _cosine_error_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    activity: torch.Tensor,
) -> torch.Tensor:
    error = 1.0 - F.cosine_similarity(prediction, target, dim=-1)
    return _weighted_per_sample(error, activity)


def _cross_rank_shuffled_actions(actions: torch.Tensor) -> torch.Tensor:
    action_bank = gather_batch_with_grad(actions.detach())
    if action_bank.shape[0] < 2:
        raise ValueError("action specificity requires at least two samples")
    local_count = actions.shape[0]
    start = (
        dist.get_rank() * local_count
        if dist.is_available() and dist.is_initialized()
        else 0
    )
    shuffled = action_bank.roll(action_bank.shape[0] // 2, dims=0)
    return shuffled[start : start + local_count]


def action_specificity_loss(
    model,
    batch: dict[str, torch.Tensor],
    output: dict,
    margin: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Require matched posterior actions to beat fixed cross-rank negatives."""
    if margin <= 0.0:
        raise ValueError("action specificity margin must be positive")
    shuffled_actions = (
        output["dynamics_actions"].roll(1, dims=-2)
        if model.config.object_aligned_actions
        else _cross_rank_shuffled_actions(output["dynamics_actions"])
    )
    history_activity = torch.stack(
        [state.activity for state in output["history_slot_states"]],
        dim=1,
    )
    shuffled = model.dynamics(
        output["online_history_slots"],
        history_activity,
        signed_gap_scale(batch["history_times"], model.config.gap_reference),
        signed_gap_scale(batch["future_times"], model.config.gap_reference),
        shuffled_actions,
        output["history_mask"],
        output["online_history_centers"],
        output.get("language_condition"),
    )
    shuffled_centers = (
        shuffled.future_centers
        if shuffled.future_centers is not None
        else model.object_aggregator.decode_center(shuffled.future_slots)
    )
    shuffled_features = model.object_aggregator.decode_feature(
        shuffled.future_slots
    )
    activity = output["target_future_activity"].detach()
    target_slots = output["target_future_slots"].detach()
    target_features = output["target_future_object_features"].detach()
    target_centers = output["target_future_centers"].detach()
    matched_error = _cosine_error_per_sample(
        output["predicted_future_slots"],
        target_slots,
        activity,
    )
    shuffled_error = _cosine_error_per_sample(
        shuffled.future_slots,
        target_slots,
        activity,
    )
    matched_error = matched_error + 2.0 * _cosine_error_per_sample(
        output["predicted_future_object_features"],
        target_features,
        activity,
    )
    shuffled_error = shuffled_error + 2.0 * _cosine_error_per_sample(
        shuffled_features,
        target_features,
        activity,
    )
    if model.config.slot_auxiliary:
        matched_center = F.smooth_l1_loss(
            output["predicted_future_centers"],
            target_centers,
            beta=0.05,
            reduction="none",
        ).mean(dim=-1)
        shuffled_center = F.smooth_l1_loss(
            shuffled_centers,
            target_centers,
            beta=0.05,
            reduction="none",
        ).mean(dim=-1)
        matched_error = matched_error + _weighted_per_sample(
            matched_center,
            activity,
        )
        shuffled_error = shuffled_error + _weighted_per_sample(
            shuffled_center,
            activity,
        )
    ranking = F.relu(matched_error - shuffled_error + margin).mean()
    sample_codes = gather_batch_with_grad(
        output["dynamics_actions"].flatten(-2).flatten(0, 1)
    )
    sample_std = sample_codes.std(dim=0, unbiased=False).mean()
    sample_variance_floor = F.relu(0.1 - sample_std)
    return ranking + sample_variance_floor, {
        "action_specificity_ranking": ranking,
        "action_specificity_matched": matched_error.mean(),
        "action_specificity_shuffled": shuffled_error.mean(),
        "action_specificity_margin": (
            shuffled_error - matched_error
        ).mean(),
        "action_specificity_action_rms": (
            output["dynamics_actions"] - shuffled_actions
        ).square().mean().sqrt(),
        "action_specificity_slot_rms": (
            output["predicted_future_slots"] - shuffled.future_slots
        ).square().mean().sqrt(),
        "action_specificity_sample_std": sample_std,
        "action_specificity_variance_floor": sample_variance_floor,
    }
