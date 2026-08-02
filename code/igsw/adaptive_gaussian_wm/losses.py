"""JEPA-first losses and lightweight allocation/readout regularizers."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .action_regularization import (
    action_specificity_loss,
    effect_aligned_action_loss,
)
from .change_objectives import (
    scale_invariant_object_change_loss,
)
from .change_readout_objective import world_model_feature_loss
from .dual_horizon_objective import prior_flow_loss
from .gpstoken import GPSTokenState
from .jepa_losses import (
    masked_history_loss,
    object_change_loss,
    object_latent_loss,
    weighted_mean as _weighted_mean,
)
from .loss_weights import AdaptiveGaussianLossWeights
from .object_memory_objectives import object_memory_geometry_loss
from .object_slots import ObjectSlotState
from .rgb_objective import rgb_loss_bundle
from .carrier_objectives import carrier_loss_bundle
from .training_diagnostics import object_memory_training_diagnostics
from .zero_action_margin import observed_zero_action_margin_loss


def allocator_loss(
    state: GPSTokenState,
    target_features: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reconstruction_error = (
        state.reconstructed_features - target_features
    ).square().mean(dim=-1)
    valid_weight = valid_mask.to(target_features.dtype)[..., None]
    feature_mean = (target_features * valid_weight).sum(dim=1, keepdim=True)
    feature_mean = feature_mean / valid_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    novelty = (target_features - feature_mean).square().mean(dim=-1).sqrt()
    novelty = novelty / (
        (novelty * valid_mask).sum(dim=1, keepdim=True)
        / valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
    ).clamp_min(1e-6)
    reconstruction_weight = valid_mask.to(novelty.dtype) * (1.0 + novelty)
    reconstruction = _weighted_mean(reconstruction_error, reconstruction_weight)

    assignment = state.assignment.clamp_min(1e-8)
    entropy = -(assignment * assignment.log()).sum(dim=1)
    partition_entropy = _weighted_mean(entropy, valid_mask)
    activation = state.activation.squeeze(-1)
    sparsity = activation.mean()
    binary = (activation * (1.0 - activation)).mean()
    coverage = (state.assignment * activation[..., None]).sum(dim=1)
    if state.density_mode == "legacy":
        coverage_target = coverage.new_full(coverage.shape, 0.5)
    else:
        scaled_novelty = novelty / (1.0 + novelty)
        coverage_target = 0.1 + 0.7 * scaled_novelty
    coverage_penalty = _weighted_mean(
        F.relu(coverage_target - coverage).square(),
        valid_mask,
    )

    difference = state.center[:, :, None] - state.center[:, None, :]
    distance_square = difference.square().sum(dim=-1)
    count = distance_square.shape[-1]
    off_diagonal = ~torch.eye(
        count,
        device=distance_square.device,
        dtype=torch.bool,
    )[None]
    pair_activity = activation[:, :, None] * activation[:, None, :]
    repulsion = torch.exp(-distance_square / 0.04) * pair_activity
    diversity = repulsion.masked_select(off_diagonal.expand_as(repulsion)).mean()

    token_rate = activation.mean()
    budget = (token_rate - state.fixed_token_fraction).square()
    sample_variance = (
        (target_features - feature_mean).square().mean(dim=-1)
        * valid_mask
    ).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)
    relative_complexity = sample_variance.sqrt()
    relative_complexity = relative_complexity / relative_complexity.mean().clamp_min(
        1e-6
    )
    target_fraction = (
        state.fixed_token_fraction * relative_complexity.sqrt()
    ).clamp(0.2, 0.8)
    target_fraction = (
        target_fraction
        * state.fixed_token_fraction
        / target_fraction.mean().clamp_min(1e-6)
    ).clamp(0.1, 0.9)
    if state.hard_token_gate:
        budget = F.mse_loss(
            state.budget_fraction,
            target_fraction.detach(),
        )
    density_alignment = F.mse_loss(
        activation.mean(dim=1),
        target_fraction.detach(),
    )
    scene_feature = feature_mean
    token_importance = (
        state.decoded_features - scene_feature
    ).square().mean(dim=-1).add(1e-6).sqrt()
    activation_score = activation - activation.mean(dim=1, keepdim=True)
    activation_score = activation_score / activation_score.std(
        dim=1, keepdim=True, unbiased=False
    ).clamp_min(1e-4)
    importance_score = token_importance - token_importance.mean(
        dim=1, keepdim=True
    )
    importance_score = importance_score / importance_score.std(
        dim=1, keepdim=True, unbiased=False
    ).clamp_min(1e-4)
    importance_alignment = F.mse_loss(
        activation_score,
        importance_score.detach(),
    )
    if state.density_mode == "legacy":
        total = (
            reconstruction
            + 0.02 * partition_entropy
            + 0.02 * sparsity
            + 0.02 * binary
            + 0.2 * coverage_penalty
            + 0.02 * diversity
        )
    else:
        total = (
            reconstruction
            + 0.02 * partition_entropy
            + 0.5 * budget
            + (0.05 * token_rate if state.hard_token_gate else token_rate * 0.0)
            + (
                0.5 * density_alignment
                if state.density_mode == "adaptive"
                else density_alignment * 0.0
            )
            + 0.5 * importance_alignment
            + 0.01 * binary
            + 0.1 * coverage_penalty
            + 0.02 * diversity
        )
    return total, {
        "allocator_reconstruction": reconstruction,
        "allocator_partition_entropy": partition_entropy,
        "allocator_sparsity": sparsity,
        "allocator_binary": binary,
        "allocator_coverage": coverage_penalty,
        "allocator_diversity": diversity,
        "allocator_budget": budget,
        "allocator_token_rate": token_rate,
        "allocator_count_mean": state.active_count.float().mean(),
        "allocator_count_std": state.active_count.float().std(unbiased=False),
        "allocator_density_alignment": density_alignment,
        "allocator_importance_alignment": importance_alignment,
        "allocator_target_fraction_std": target_fraction.std(unbiased=False),
    }


def slot_regularization(
    state: ObjectSlotState,
    tokens: GPSTokenState,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assignment = state.assignment.clamp_min(1e-8)
    full_assignment = torch.cat(
        (assignment, state.background_assignment[..., None].clamp_min(1e-8)), dim=-1
    )
    entropy = -(full_assignment * full_assignment.log()).sum(dim=-1).mean()
    normalized = F.normalize(state.tracking_slots, dim=-1)
    similarity = torch.einsum("bkd,bjd->bkj", normalized, normalized)
    count = similarity.shape[-1]
    off_diagonal = ~torch.eye(
        count,
        device=similarity.device,
        dtype=torch.bool,
    )[None]
    diversity = similarity.square().masked_select(
        off_diagonal.expand_as(similarity)
    ).mean()
    activity_binary = (
        state.activity * (1.0 - state.activity)
    ).mean()
    if state.center_auxiliary_enabled:
        center_decode = F.smooth_l1_loss(
            state.decoded_center,
            state.center.detach(),
            beta=0.05,
        )
    else:
        center_decode = state.slots.sum() * 0.0
    if state.auxiliary_enabled:
        feature_decode = F.mse_loss(
            F.normalize(state.decoded_feature, dim=-1),
            F.normalize(state.feature.detach(), dim=-1),
        )
    else:
        feature_decode = state.slots.sum() * 0.0
    token_weight = state.assignment * tokens.activation
    center_difference = (
        tokens.center[:, :, None] - state.center[:, None]
    ).square().sum(dim=-1)
    compactness = (
        center_difference * token_weight
    ).sum() / token_weight.sum().clamp_min(1e-6)
    conditional_assignment = assignment / state.potential_change[..., None].clamp_min(
        1e-6
    )
    reconstructed_token_feature = torch.einsum(
        "bmk,bkc->bmc",
        conditional_assignment,
        state.decoded_feature,
    )
    token_feature_reconstruction = _weighted_mean(
        (
            F.normalize(reconstructed_token_feature, dim=-1)
            - F.normalize(tokens.decoded_features.detach(), dim=-1)
        ).square(),
        tokens.activation.squeeze(-1) * state.potential_change.detach(),
    )
    total = (
        0.5 * center_decode
        + 0.5 * feature_decode
        + token_feature_reconstruction
        + 0.5 * compactness
        + 0.02 * entropy
        + 0.2 * diversity
        + 0.01 * activity_binary
    )
    return total, {
        "slot_entropy": entropy,
        "slot_diversity": diversity,
        "slot_activity_binary": activity_binary,
        "slot_center_decode": center_decode,
        "slot_feature_decode": feature_decode,
        "slot_token_feature_reconstruction": token_feature_reconstruction,
        "slot_compactness": compactness,
    }


def adaptive_world_model_loss(
    model,
    batch: dict[str, torch.Tensor],
    output: dict,
    weights: AdaptiveGaussianLossWeights,
    collect_diagnostics: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    future_alignment = object_latent_loss(
        output["predicted_future_slots"],
        output["target_future_slots"],
        output["target_future_activity"],
    )
    future_change = object_change_loss(
        output["predicted_future_slots"],
        output["target_future_slots"],
        output["online_history_slots"][:, -1],
        output["target_history_slots"][:, -1],
        output["target_future_activity"],
    )
    future_feature_alignment = object_latent_loss(
        output["predicted_future_object_features"],
        output["target_future_object_features"],
        output["target_future_activity"],
    )
    future_feature_change = scale_invariant_object_change_loss(
        output["predicted_future_object_features"],
        output["target_future_object_features"],
        output["online_history_object_features"][:, -1],
        output["target_history_object_features"][:, -1],
        output["target_future_activity"],
    )
    if model.config.slot_auxiliary:
        future_center = _weighted_mean(
            F.smooth_l1_loss(
                output["predicted_future_centers"],
                output["target_future_centers"].detach(),
                reduction="none",
                beta=0.05,
            ),
            output["target_future_activity"].detach(),
        )
    else:
        future_center = output["predicted_future_slots"].sum() * 0.0
    future_latent = future_alignment + 2.0 * future_change
    future_object_feature = (
        future_feature_alignment + 2.0 * future_feature_change
    )
    center_weight = (
        0.0
        if model.config.relative_transport_dynamics
        else 5.0 if model.config.learned_velocity_baseline else 1.0
    )
    future = (
        0.5 * future_latent
        + 2.0 * future_object_feature
        + center_weight * future_center
    )
    zero_alignment = object_latent_loss(
        output["zero_action_future_slots"],
        output["target_future_slots"],
        output["target_future_activity"],
    )
    zero_change = object_change_loss(
        output["zero_action_future_slots"],
        output["target_future_slots"],
        output["online_history_slots"][:, -1],
        output["target_history_slots"][:, -1],
        output["target_future_activity"],
    )
    zero_features = model.object_aggregator.decode_feature(
        output["zero_action_future_slots"]
    )
    zero_feature_alignment = object_latent_loss(
        zero_features,
        output["target_future_object_features"],
        output["target_future_activity"],
    )
    zero_feature_change = scale_invariant_object_change_loss(
        zero_features,
        output["target_future_object_features"],
        output["online_history_object_features"][:, -1],
        output["target_history_object_features"][:, -1],
        output["target_future_activity"],
    )
    zero_future = (
        0.5 * (zero_alignment + 2.0 * zero_change)
        + 2.0 * (zero_feature_alignment + 2.0 * zero_feature_change)
    )
    action = future * 0.0
    action_parts = {}
    if weights.action > 0.0:
        action, action_parts = effect_aligned_action_loss(
            model,
            output,
            future,
            zero_future,
        )
    action_specificity = future * 0.0
    action_specificity_parts = {}
    if weights.action_specificity > 0.0:
        action_specificity, action_specificity_parts = action_specificity_loss(
            model,
            batch,
            output,
        )
    history = masked_history_loss(
        output["predicted_history_slots"],
        output["target_history_slots"],
        output["history_mask"],
        output["target_history_activity"],
    )
    flow = (
        prior_flow_loss(model, batch, output)
        if weights.flow > 0.0
        else future * 0.0
    )
    feature, change_parts = world_model_feature_loss(model, batch, output)
    rgb, rgb_delta, rgb_object, rgb_parts = rgb_loss_bundle(
        model, batch, output, feature
    )
    geometry, geometry_parts = object_memory_geometry_loss(output, batch)
    (
        current_readout,
        readout_regularization,
        carrier_support,
        carrier_compact,
        readout_parts,
    ) = carrier_loss_bundle(model, batch, output, feature, weights)
    zero_margin, zero_margin_parts = observed_zero_action_margin_loss(model, batch, output)
    language_effect = feature * 0.0
    language_effect_parts = {}
    if model.language_effect_alignment is not None:
        if output["language_condition"] is None or "condition_index" not in batch:
            raise ValueError("language-effect alignment requires condition ids")
        language_effect, language_effect_parts = model.language_effect_alignment(
            output["language_condition"],
            model.latent_actions.predict_effect(output["posterior_actions"]),
            batch["condition_index"],
        )

    allocator_terms = []
    parts: dict[str, torch.Tensor] = {}
    for index, state in enumerate(output["history_token_states"]):
        term, detail = allocator_loss(
            state,
            batch["history_features"][:, index],
            batch["history_valid"][:, index],
        )
        allocator_terms.append(term)
        for name, value in detail.items():
            parts.setdefault(name, value.new_zeros(()))
            parts[name] = parts[name] + value
    allocator = torch.stack(allocator_terms).mean()
    for name in tuple(parts):
        parts[name] = parts[name] / len(allocator_terms)

    slot_terms = []
    for state, tokens in zip(
        output["history_slot_states"],
        output["history_token_states"],
        strict=True,
    ):
        term, detail = slot_regularization(state, tokens)
        slot_terms.append(term)
        for name, value in detail.items():
            parts.setdefault(name, value.new_zeros(()))
            parts[name] = parts[name] + value
    slot = torch.stack(slot_terms).mean()
    for name in (
        "slot_entropy",
        "slot_diversity",
        "slot_activity_binary",
        "slot_center_decode",
        "slot_feature_decode",
        "slot_token_feature_reconstruction",
        "slot_compactness",
    ):
        parts[name] = parts[name] / len(slot_terms)

    total = (
        weights.future * future
        + weights.history * history
        + weights.flow * flow
        + weights.feature * feature
        + weights.allocator * allocator
        + weights.slot * slot
        + weights.action * action
        + weights.action_specificity * action_specificity
        + weights.geometry * geometry
        + weights.current_readout * current_readout
        + weights.readout_regularization * readout_regularization
        + weights.carrier_support * carrier_support
        + weights.carrier_compact * carrier_compact
        + weights.rgb * model.config.rgb_loss_weight * rgb
        + model.config.zero_action_margin_weight * zero_margin
        + model.config.language_effect_weight * language_effect
    )
    parts.update(
        {
            "total": total,
            "future": future,
            "future_alignment": future_alignment,
            "future_change": future_change,
            "future_latent": future_latent,
            "future_object_feature": future_object_feature,
            "future_feature_alignment": future_feature_alignment,
            "future_feature_change": future_feature_change,
            "future_center": future_center,
            "zero_future": zero_future,
            "action": action,
            "action_specificity": action_specificity,
            "history": history,
            "flow": flow,
            "feature": feature,
            "allocator": allocator,
            "slot": slot,
            "geometry": geometry,
            "current_readout": current_readout,
            "readout_regularization": readout_regularization,
            "carrier_support": carrier_support,
            "carrier_compact": carrier_compact,
            "rgb_future": rgb,
            "rgb_change_future": rgb_delta,
            "rgb_object_future": rgb_object,
            "zero_action_margin": zero_margin,
            "language_effect": language_effect,
        }
    )
    parts.update(action_parts)
    parts.update(action_specificity_parts)
    parts.update(geometry_parts)
    parts.update(readout_parts)
    parts.update(change_parts)
    if collect_diagnostics:
        parts.update(
            object_memory_training_diagnostics(
                model,
                batch,
                output,
                {
                    "future": future,
                    "future_latent": future_latent,
                    "future_object_feature": future_object_feature,
                    "future_center": future_center,
                    "feature": feature,
                },
                geometry_parts,
            )
        )
    parts.update({f"rgb_future_{name}": value for name, value in rgb_parts.items()})
    parts.update(zero_margin_parts)
    parts.update(language_effect_parts)
    return total, parts
