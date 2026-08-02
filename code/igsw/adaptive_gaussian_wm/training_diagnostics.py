"""Detached long-run diagnostics for Object Memory JEPA training."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .change_objectives import (
    dense_feature_loss,
    scale_invariant_object_change_loss,
)
from .change_readout_diagnostics import change_residual_readout_diagnostics
from .diagnostic_statistics import correlation_moments, ratio_moments
from .dense_readout_diagnostics import dense_object_readout_diagnostics
from .dual_horizon_diagnostics import dual_horizon_effect_diagnostics
from .jepa_losses import object_change_loss, object_latent_loss, weighted_mean
from .lifecycle_transport_diagnostics import lifecycle_transport_diagnostics
from .object_correspondence_diagnostics import object_correspondence_diagnostics
from .readout_diagnostics import gaussian_readout_diagnostics


def _object_prediction_terms(
    model,
    slots: torch.Tensor,
    features: torch.Tensor,
    centers: torch.Tensor,
    output: dict,
) -> dict[str, torch.Tensor]:
    activity = output["target_future_activity"].detach().float()
    target_slots = output["target_future_slots"].detach().float()
    current_slots = output["online_history_slots"][:, -1].detach().float()
    target_current_slots = output["target_history_slots"][:, -1].detach().float()
    alignment = object_latent_loss(slots.float(), target_slots, activity)
    change = object_change_loss(
        slots.float(),
        target_slots,
        current_slots,
        target_current_slots,
        activity,
    )
    latent = alignment + 2.0 * change

    target_features = output["target_future_object_features"].detach().float()
    current_features = output["online_history_object_features"][:, -1].detach().float()
    target_current_features = output["target_history_object_features"][
        :, -1
    ].detach().float()
    feature_alignment = object_latent_loss(
        features.float(), target_features, activity
    )
    feature_change = scale_invariant_object_change_loss(
        features.float(),
        target_features,
        current_features,
        target_current_features,
        activity,
    )
    object_feature = feature_alignment + 2.0 * feature_change

    if model.config.slot_auxiliary:
        center = weighted_mean(
            F.smooth_l1_loss(
                centers.float(),
                output["target_future_centers"].detach().float(),
                reduction="none",
                beta=0.05,
            ),
            activity,
        )
    else:
        center = slots.new_zeros((), dtype=torch.float32)
    center_weight = (
        0.0
        if model.config.relative_transport_dynamics
        else 5.0 if model.config.learned_velocity_baseline else 1.0
    )
    return {
        "latent": latent,
        "object_feature": object_feature,
        "center": center,
        "total": 0.5 * latent + 2.0 * object_feature + center_weight * center,
    }


def _persistence_tensors(output: dict) -> tuple[torch.Tensor, ...]:
    target_slots = output["target_future_slots"]
    target_features = output["target_future_object_features"]
    target_centers = output["target_future_centers"]
    slots = output["online_history_slots"][:, -1, None].expand_as(target_slots)
    features = output["online_history_object_features"][:, -1, None].expand_as(
        target_features
    )
    centers = output["online_history_centers"][:, -1, None].expand_as(target_centers)
    return slots.detach(), features.detach(), centers.detach()


def _per_sample_feature_complexity(
    features: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    features = features.detach().float()
    weight = valid.detach().float()
    denominator = weight.sum(dim=1).clamp_min(1.0)
    mean = (features * weight[..., None]).sum(dim=1) / denominator[:, None]
    variance = (features - mean[:, None]).square().mean(dim=-1)
    return torch.sqrt(
        (variance * weight).sum(dim=1) / denominator
    ).clamp_min(0.0)


def _per_sample_future_change(batch: dict) -> torch.Tensor:
    future = batch["future_features"].detach().float()
    current = batch["history_features"][:, -1:, :, :].detach().float()
    current = current.expand_as(future)
    valid = batch["future_valid"] & batch["history_valid"][:, -1:, :]
    error = (future - current).square().mean(dim=-1)
    weight = valid.float()
    denominator = weight.sum(dim=(1, 2)).clamp_min(1.0)
    return torch.sqrt((error * weight).sum(dim=(1, 2)) / denominator)


def _per_sample_reconstruction_error(output: dict, batch: dict) -> torch.Tensor:
    tokens = output["history_token_states"][-1]
    target = batch["history_features"][:, -1].detach().float()
    valid = batch["history_valid"][:, -1].detach().float()
    error = (tokens.reconstructed_features.detach().float() - target).square().mean(
        dim=-1
    )
    return (error * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def _geometry_persistence_diagnostics(
    output: dict,
    geometry_parts: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    activity = output["target_future_activity"].detach().float()
    target_scale = output["target_future_relative_scale"].detach().float()
    persistence_scale = output["online_history_relative_scale"][
        :, -1, None
    ].expand_as(target_scale).detach().float()
    scale = weighted_mean(
        F.smooth_l1_loss(
            persistence_scale.clamp_min(1e-6).log(),
            target_scale.clamp_min(1e-6).log(),
            beta=0.1,
            reduction="none",
        ),
        activity,
    )
    target_relations = output["target_future_relations"].detach().float()
    persistence_relations = output["online_history_relations"][
        :, -1, None
    ].expand_as(target_relations).detach().float()
    relation_error = F.smooth_l1_loss(
        persistence_relations[..., :3],
        target_relations[..., :3],
        beta=0.1,
        reduction="none",
    ).mean(dim=-1)
    pair_weight = activity.unsqueeze(-1) * activity.unsqueeze(-2)
    relations = weighted_mean(relation_error, pair_weight)
    return {
        "baseline_persistence_relative_scale": scale,
        "baseline_persistence_image_plane_relations": relations,
        "geometry_relative_scale_gain_over_persistence": (
            scale - geometry_parts["geometry_relative_scale"].detach().float()
        ),
        "geometry_relations_gain_over_persistence": (
            relations
            - geometry_parts["geometry_image_plane_relations"].detach().float()
        ),
    }


def _lifecycle_diagnostics(output: dict) -> dict[str, torch.Tensor]:
    prediction = output.get(
        "predicted_future_track_presence",
        output["predicted_future_existence"],
    ).detach().float()
    visibility_prediction = output["predicted_future_visibility"].detach().float()
    target = output.get(
        "target_future_track_presence",
        output["target_future_existence"],
    ).detach().float()
    visibility_target = output["target_future_visibility"].detach().float()
    in_frame_target = output["target_future_in_frame"].detach().float()
    current = output.get(
        "target_history_track_presence",
        output["target_history_existence"],
    )[:, -1, None].detach().float()
    current = current.expand_as(target)
    target_binary = target >= 0.5
    prediction_binary = prediction >= 0.5
    current_binary = current >= 0.5
    total = target.new_tensor(float(target.numel()))
    positive = target_binary.float().sum()
    negative = (~target_binary).float().sum()
    result = {
        "memory_track_presence_target_mean": target.mean(),
        "memory_track_presence_prediction_mean": prediction.mean(),
        "memory_track_presence_brier": (prediction - target).square().mean(),
        "memory_track_presence_accuracy": (
            prediction_binary == target_binary
        ).float().mean(),
        "memory_visibility_target_mean": visibility_target.mean(),
        "memory_visibility_prediction_mean": visibility_prediction.mean(),
        "memory_visibility_brier": (
            visibility_prediction - visibility_target
        ).square().mean(),
        "memory_in_frame_target_mean": in_frame_target.mean(),
        "memory_track_presence_change_target_mean": (target - current).abs().mean(),
        "memory_track_presence_change_prediction_mean": (
            prediction - current
        ).abs().mean(),
    }
    result.update(
        ratio_moments(
            "memory_track_presence_target_positive_rate", positive, total
        )
    )
    result.update(
        ratio_moments(
            "memory_track_presence_prediction_positive_rate",
            prediction_binary.float().sum(),
            total,
        )
    )
    result.update(
        ratio_moments(
            "memory_track_presence_positive_recall",
            (prediction_binary & target_binary).float().sum(),
            positive,
        )
    )
    result.update(
        ratio_moments(
            "memory_track_presence_negative_recall",
            ((~prediction_binary) & (~target_binary)).float().sum(),
            negative,
        )
    )
    result.update(
        ratio_moments(
            "memory_target_appearance_rate",
            ((~current_binary) & target_binary).float().sum(),
            total,
        )
    )
    result.update(
        ratio_moments(
            "memory_target_disappearance_rate",
            (current_binary & (~target_binary)).float().sum(),
            total,
        )
    )
    result.update(
        ratio_moments(
            "memory_target_occlusion_rate_among_existent",
            (target * (1.0 - visibility_target)).sum(),
            target.sum(),
        )
    )
    result.update(
        ratio_moments(
            "memory_predicted_occlusion_rate_among_existent",
            (prediction * (1.0 - visibility_prediction)).sum(),
            prediction.sum(),
        )
    )
    result.update(
        ratio_moments(
            "memory_target_out_of_frame_rate_among_existent",
            (target * (1.0 - in_frame_target)).sum(),
            target.sum(),
        )
    )
    future_states = output["target_future_slot_states"]
    target_update = torch.stack(
        [state.update_gate.detach().float() for state in future_states], dim=1
    )
    current_update = output["history_slot_states"][-1].update_gate.detach().float()
    result.update(
        {
            "memory_current_update_gate_mean": current_update.mean(),
            "memory_target_future_update_gate_mean": target_update.mean(),
            "memory_target_future_low_update_rate": (
                target_update < 0.1
            ).float().mean(),
        }
    )
    return result


def _token_diagnostics(output: dict, batch: dict) -> dict[str, torch.Tensor]:
    tokens = output["history_token_states"][-1]
    count = tokens.active_count.detach().float()
    current_features = batch["history_features"][:, -1]
    current_valid = batch["history_valid"][:, -1]
    complexity = _per_sample_feature_complexity(current_features, current_valid)
    future_change = _per_sample_future_change(batch)
    reconstruction = _per_sample_reconstruction_error(output, batch)
    visible_mass = output["online_history_visibility"][:, -1].detach().float().sum(
        dim=-1
    )
    result = {
        "token_current_count_mean": count.mean(),
        "token_current_count_std": count.std(unbiased=False),
        "token_current_budget_fraction_mean": (
            tokens.budget_fraction.detach().float().mean()
        ),
        "token_spatial_complexity_mean": complexity.mean(),
        "token_future_change_mean": future_change.mean(),
        "token_current_reconstruction_error_mean": reconstruction.mean(),
        "token_visible_object_mass_mean": visible_mass.mean(),
    }
    for name, value in (
        ("token_count_vs_spatial_complexity_correlation", complexity),
        ("token_count_vs_future_change_correlation", future_change),
        ("token_count_vs_reconstruction_error_correlation", reconstruction),
        ("token_count_vs_visible_object_mass_correlation", visible_mass),
    ):
        result.update(correlation_moments(name, count, value))
    return result


def _horizon_diagnostics(
    model,
    batch: dict,
    output: dict,
    persistence: tuple[torch.Tensor, ...],
) -> dict[str, torch.Tensor]:
    loss_coverage = output["feature_loss_coverage"]
    result: dict[str, torch.Tensor] = {}
    slots, features, centers = persistence
    for index in range(output["target_future_slots"].shape[1]):
        sliced = dict(output)
        for name in (
            "target_future_slots",
            "target_future_activity",
            "target_future_centers",
            "target_future_object_features",
        ):
            sliced[name] = output[name][:, index : index + 1]
        predicted = _object_prediction_terms(
            model,
            output["predicted_future_slots"][:, index : index + 1],
            output["predicted_future_object_features"][:, index : index + 1],
            output["predicted_future_centers"][:, index : index + 1],
            sliced,
        )
        baseline = _object_prediction_terms(
            model,
            slots[:, index : index + 1],
            features[:, index : index + 1],
            centers[:, index : index + 1],
            sliced,
        )
        predicted_dense = dense_feature_loss(
            output["rendered_future_features"][:, index : index + 1].detach(),
            batch["future_features"][:, index : index + 1],
            batch["future_valid"][:, index : index + 1],
            loss_coverage[:, index : index + 1],
        )
        copy_dense = dense_feature_loss(
            batch["history_features"][:, -1:].expand_as(
                batch["future_features"][:, index : index + 1]
            ),
            batch["future_features"][:, index : index + 1],
            batch["future_valid"][:, index : index + 1],
            loss_coverage[:, index : index + 1],
        )
        prefix = f"horizon_{index}"
        result.update(
            {
                f"{prefix}_seconds_mean": batch["future_times"][
                    :, index
                ].detach().float().mean(),
                f"{prefix}_object_prediction": predicted["total"],
                f"{prefix}_persistence_object": baseline["total"],
                f"{prefix}_object_gain_over_persistence": (
                    baseline["total"] - predicted["total"]
                ),
                f"{prefix}_dense_feature": predicted_dense,
                f"{prefix}_persistence_dense_feature": copy_dense,
                f"{prefix}_dense_gain_over_persistence": (
                    copy_dense - predicted_dense
                ),
                f"{prefix}_existence_brier": (
                    output["predicted_future_existence"][:, index].detach().float()
                    - output["target_future_existence"][:, index].detach().float()
                ).square().mean(),
            }
        )
    return result


@torch.no_grad()
def object_memory_training_diagnostics(
    model,
    batch: dict,
    output: dict,
    future_parts: dict[str, torch.Tensor],
    geometry_parts: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Measure predictive gains and memory behavior without changing gradients."""
    if model.config.architecture not in (
        "object_memory_v1",
        "object_memory_v2",
        "object_memory_v3",
    ):
        return {}
    persistence = _persistence_tensors(output)
    loss_coverage = output["feature_loss_coverage"]
    baseline = _object_prediction_terms(model, *persistence, output)
    dense_baseline = dense_feature_loss(
        batch["history_features"][:, -1:].expand_as(batch["future_features"]),
        batch["future_features"],
        batch["future_valid"],
        loss_coverage,
    )
    predicted_core = future_parts["future"] + 0.5 * future_parts["feature"]
    baseline_core = baseline["total"] + 0.5 * dense_baseline
    result = {
        "baseline_persistence_future": baseline["total"],
        "baseline_persistence_latent": baseline["latent"],
        "baseline_persistence_object_feature": baseline["object_feature"],
        "baseline_persistence_center": baseline["center"],
        "baseline_persistence_dense_feature": dense_baseline,
        "dynamics_gain_over_persistence": (
            baseline["total"] - future_parts["future"].detach().float()
        ),
        "dynamics_latent_gain_over_persistence": (
            baseline["latent"] - future_parts["future_latent"].detach().float()
        ),
        "dynamics_object_feature_gain_over_persistence": (
            baseline["object_feature"]
            - future_parts["future_object_feature"].detach().float()
        ),
        "dynamics_center_gain_over_persistence": (
            baseline["center"] - future_parts["future_center"].detach().float()
        ),
        "dense_feature_gain_over_persistence": (
            dense_baseline - future_parts["feature"].detach().float()
        ),
        "predictive_core": predicted_core.detach().float(),
        "baseline_persistence_predictive_core": baseline_core,
        "predictive_gain_over_persistence": (
            baseline_core - predicted_core.detach().float()
        ),
    }
    result.update(
        ratio_moments(
            "dynamics_relative_gain_over_persistence",
            baseline["total"] - future_parts["future"].detach().float(),
            baseline["total"],
        )
    )
    result.update(
        ratio_moments(
            "dense_feature_relative_gain_over_persistence",
            dense_baseline - future_parts["feature"].detach().float(),
            dense_baseline,
        )
    )
    result.update(
        ratio_moments(
            "predictive_relative_gain_over_persistence",
            baseline_core - predicted_core.detach().float(),
            baseline_core,
        )
    )
    result.update(_geometry_persistence_diagnostics(output, geometry_parts))
    result.update(_lifecycle_diagnostics(output))
    if model.config.architecture in ("object_memory_v2", "object_memory_v3"):
        result.update(lifecycle_transport_diagnostics(output))
    if model.config.architecture == "object_memory_v3":
        result.update(object_correspondence_diagnostics(output))
    result.update(_token_diagnostics(output, batch))
    result.update(_horizon_diagnostics(model, batch, output, persistence))
    result.update(dual_horizon_effect_diagnostics(batch, output))
    if model.config.change_residual_readout:
        result.update(change_residual_readout_diagnostics(batch, output))
    elif model.config.dense_object_readout:
        result.update(dense_object_readout_diagnostics(model, batch, output))
    else:
        result.update(gaussian_readout_diagnostics(model, batch, output))
    return result
