"""Relative change supervision for background-separated full-DINO prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .change_objectives import dense_feature_loss


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    expanded = weight
    while expanded.ndim < value.ndim:
        expanded = expanded[..., None]
    return (value * expanded).sum() / expanded.expand_as(value).sum().clamp_min(1.0)


@torch.no_grad()
def relative_change_targets(
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-sample relative targets without a fixed feature-distance threshold."""
    future = batch["future_features"].float()
    current = batch["history_features"][:, -1:].float()
    if current.shape[2:] != future.shape[2:]:
        raise ValueError("current and future DINO grids do not align")
    current = current.expand_as(future)
    current_valid = batch["history_valid"][:, -1:].expand_as(batch["future_valid"])
    valid = batch["future_valid"] & current_valid
    score = 1.0 - F.cosine_similarity(
        F.normalize(future, dim=-1),
        F.normalize(current, dim=-1),
        dim=-1,
    )
    score = score.clamp_min(0.0)
    rows = score.flatten(0, 1)
    row_valid = valid.flatten(0, 1)
    targets = torch.zeros_like(rows)
    for index in range(rows.shape[0]):
        values = rows[index, row_valid[index]]
        if values.numel() == 0:
            raise ValueError("future change target has no valid patches")
        lower = torch.quantile(values, 0.5)
        upper = torch.quantile(values, 0.9)
        spread = upper - lower
        if float(spread) > 1e-6:
            targets[index] = ((rows[index] - lower) / spread).clamp(0.0, 1.0)
    active = targets.unflatten(0, score.shape[:2]) * valid.float()
    potential = active.max(dim=1).values
    potential_valid = valid.any(dim=1)
    return active, potential, potential_valid


def _pool_to_tokens(
    assignment: torch.Tensor,
    patch_target: torch.Tensor,
) -> torch.Tensor:
    if assignment.ndim != 3 or patch_target.ndim != 3:
        raise ValueError("token pooling expects [B,M,N] and [B,Q,N]")
    if assignment.shape[0] != patch_target.shape[0] or assignment.shape[2] != (
        patch_target.shape[2]
    ):
        raise ValueError("token assignment and patch target do not align")
    mass = assignment.sum(dim=-1).clamp_min(1e-6)
    return torch.einsum("bmn,bqn->bqm", assignment.float(), patch_target.float()) / (
        mass[:, None]
    )


def _balanced_binary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    error = F.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction="none"
    )
    positive = target.float() * weight.float()
    negative = (1.0 - target.float()) * weight.float()
    positive_loss = (error * positive).sum() / positive.sum().clamp_min(1.0)
    negative_loss = (error * negative).sum() / negative.sum().clamp_min(1.0)
    positive_present = (positive.sum() > 0).to(error.dtype)
    negative_present = (negative.sum() > 0).to(error.dtype)
    return (
        positive_loss * positive_present + negative_loss * negative_present
    ) / (positive_present + negative_present).clamp_min(1.0)


def _feature_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    mse = (prediction.float() - target.float()).square().mean(dim=-1)
    cosine = 1.0 - F.cosine_similarity(
        prediction.float(), target.float(), dim=-1
    )
    return mse + 0.1 * cosine


def change_aware_world_model_loss(
    batch: dict[str, torch.Tensor],
    output: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train exact-field persistence and a separately gated local residual."""
    state = output.get("change_future_readout")
    reference = output.get("change_reference_readout")
    if state is None or reference is None:
        raise ValueError("v37 objective requires change residual readout states")
    prediction = output["rendered_future_features"]
    target = batch["future_features"]
    if prediction.shape != target.shape:
        raise ValueError("predicted and target DINO fields do not align")
    active_target, potential_target, potential_valid = relative_change_targets(batch)
    valid = batch["future_valid"].float()
    change_weight = active_target * valid
    static_weight = (1.0 - active_target) * valid
    error = _feature_error(prediction, target)
    change_feature = _weighted_mean(error, change_weight)
    static_feature = _weighted_mean(error, static_weight)
    change_present = (change_weight.sum() > 0).to(error.dtype)
    static_present = (static_weight.sum() > 0).to(error.dtype)
    full_dino = (
        change_feature * change_present + static_feature * static_present
    ) / (change_present + static_present).clamp_min(1.0)

    tokens = output["history_token_states"][-1]
    slots = output["history_slot_states"][-1]
    potential_token_target = _pool_to_tokens(
        tokens.assignment.detach(), potential_target[:, None]
    )[:, 0]
    token_weight = tokens.activation.squeeze(-1).detach().float()
    potential_partition = _balanced_binary_loss(
        slots.potential_change_logits,
        potential_token_target.detach(),
        token_weight,
    )
    predicted_mass = (
        slots.potential_change.float() * tokens.activation.squeeze(-1).detach().float()
    ).sum(dim=1) / tokens.activation.squeeze(-1).detach().float().sum(
        dim=1
    ).clamp_min(1.0)
    target_mass = (
        potential_target * potential_valid.float()
    ).sum(dim=1) / potential_valid.float().sum(dim=1).clamp_min(1.0)
    potential_mass = F.smooth_l1_loss(predicted_mass, target_mass.detach())
    potential_loss = potential_partition + potential_mass

    active_token_target = _pool_to_tokens(
        tokens.assignment.detach(), active_target
    )
    token_weight = token_weight[:, None]
    active_token_loss = _balanced_binary_loss(
        state.active_change_logits,
        active_token_target.detach(),
        token_weight,
    )
    active_map_loss = F.smooth_l1_loss(
        state.active_change_map.float(), active_target.detach(), reduction="none"
    )
    active_map_loss = _weighted_mean(active_map_loss, valid)
    active_loss = active_token_loss + active_map_loss

    current = batch["history_features"][:, -1:].float().expand_as(prediction)
    residual = prediction.float() - current
    static_leakage = _weighted_mean(
        residual.square().mean(dim=-1), static_weight
    )
    persistence_error = _feature_error(current, target)
    persistence_change = _weighted_mean(persistence_error, change_weight)
    persistence_static = _weighted_mean(persistence_error, static_weight)
    total = full_dino + 0.2 * potential_loss + 0.2 * active_loss + 0.1 * static_leakage
    return total, {
        "feature_full_dino": full_dino,
        "feature_change_region": change_feature,
        "feature_static_region": static_feature,
        "feature_change_gain_over_persistence": persistence_change - change_feature,
        "feature_static_gain_over_persistence": persistence_static - static_feature,
        "change_potential": potential_loss,
        "change_potential_partition": potential_partition,
        "change_potential_mass": potential_mass,
        "change_potential_predicted_fraction": predicted_mass.mean().detach(),
        "change_potential_target_fraction": target_mass.mean().detach(),
        "change_active": active_loss,
        "change_active_token": active_token_loss,
        "change_active_map": active_map_loss,
        "change_active_predicted_fraction": state.active_change_map.mean().detach(),
        "change_active_target_fraction": active_target.mean().detach(),
        "change_static_leakage": static_leakage,
        "change_residual_rms": residual.square().mean().sqrt().detach(),
        "change_reference_residual_rms": reference.residual.square().mean().sqrt().detach(),
    }


def world_model_feature_loss(
    model,
    batch: dict[str, torch.Tensor],
    output: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if model.config.change_residual_readout:
        return change_aware_world_model_loss(batch, output)
    loss = dense_feature_loss(
        output["rendered_future_features"],
        batch["future_features"],
        batch["future_valid"],
        output["feature_loss_coverage"],
    )
    return loss, {}
