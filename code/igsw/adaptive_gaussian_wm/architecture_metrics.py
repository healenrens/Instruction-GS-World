"""Integrated architecture metrics that require grouped future targets."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .scale import signed_gap_scale


def normalized_slot_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (
        F.normalize(prediction, dim=-1) - F.normalize(target, dim=-1)
    ).square().mean(dim=(-1, -2, -3))


def swapped_density_reconstruction(
    output: dict,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    state = output["history_token_states"][-1]
    target = batch["history_features"][:, -1]
    baseline = (state.reconstructed_features - target).square().mean(dim=(1, 2))
    rank = torch.argsort(state.activation.sum(dim=1).squeeze(-1))
    source = torch.empty_like(rank)
    source[rank] = rank.flip(0)
    active_assignment = state.assignment * state.activation[source]
    coverage = active_assignment.sum(dim=1).clamp(0.0, 1.0)
    active = torch.einsum(
        "bmn,bmc->bnc",
        active_assignment,
        state.decoded_features,
    )
    background = target.mean(dim=1, keepdim=True)
    reconstruction = active + (1.0 - coverage)[..., None] * background
    swapped = (reconstruction - target).square().mean(dim=(1, 2))
    return baseline, swapped


def temporal_identity_error(
    model,
    output: dict,
    batch: dict[str, torch.Tensor],
) -> float:
    history_slots = output["online_history_slots"]
    history_centers = output["online_history_centers"]
    if history_slots.shape[1] == 1:
        return 0.0
    history_activity = torch.stack(
        [state.activity for state in output["history_slot_states"]],
        dim=1,
    )
    shuffled_slots = history_slots.clone()
    shuffled_activity = history_activity.clone()
    shuffled_centers = history_centers.clone()
    for frame in range(1, history_slots.shape[1]):
        order = torch.roll(
            torch.arange(model.config.object_slots, device=history_slots.device),
            shifts=frame,
        )
        shuffled_slots[:, frame] = shuffled_slots[:, frame, order]
        shuffled_activity[:, frame] = shuffled_activity[:, frame, order]
        shuffled_centers[:, frame] = shuffled_centers[:, frame, order]
    history_scale = signed_gap_scale(
        batch["history_times"],
        model.config.gap_reference,
    )
    future_scale = signed_gap_scale(
        batch["future_times"],
        model.config.gap_reference,
    )
    prediction = model.dynamics(
        shuffled_slots,
        shuffled_activity,
        history_scale,
        future_scale,
        output["dynamics_actions"],
        torch.zeros(
            shuffled_slots.shape[:3],
            device=history_slots.device,
            dtype=torch.bool,
        ),
        shuffled_centers,
        output.get("language_condition"),
    ).future_slots
    return float(
        normalized_slot_error(
            prediction,
            output["target_future_slots"],
        ).mean()
    )


def _pairwise_distance(value: torch.Tensor, normalize: bool) -> torch.Tensor:
    representation = F.normalize(value, dim=-1) if normalize else value
    return (
        representation[:, :, None] - representation[:, None, :]
    ).square().mean(dim=(-1, -2, -3))


def _mode_geometry(distance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    off_diagonal = ~torch.eye(
        distance.shape[-1],
        device=distance.device,
        dtype=torch.bool,
    )[None]
    separation = distance.masked_fill(
        ~off_diagonal,
        torch.inf,
    ).amin(dim=(1, 2))
    return separation, off_diagonal


def _group_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = torch.where(mask, value, torch.zeros_like(value))
    return float(selected.sum() / mask.sum().clamp_min(1))


def _coverage(
    sample_error: torch.Tensor,
    deterministic_error: torch.Tensor,
    separation: torch.Tensor,
    ambiguity: torch.Tensor,
) -> tuple[dict[str, float | int], torch.Tensor]:
    radius_square = 0.25 * separation
    separated = ambiguity & torch.isfinite(radius_square) & (
        radius_square > 1e-8
    )
    hit = sample_error <= radius_square[None, :, None]
    recall = hit.any(dim=0).float().mean(dim=-1)
    precision = hit.any(dim=-1).float().mean(dim=0)
    best_one = sample_error[0].mean(dim=-1)
    best_n = sample_error.amin(dim=0).mean(dim=-1)
    deterministic = deterministic_error.mean(dim=-1)
    return {
        "separated_ambiguous_groups": int(separated.sum()),
        "mode_recall_at_n": _group_mean(recall, separated),
        "sample_precision_at_n": _group_mean(precision, separated),
        "best_of_1_mse": _group_mean(best_one, separated),
        "best_of_n_mse": _group_mean(best_n, separated),
        "deterministic_prediction_mse": _group_mean(
            deterministic,
            separated,
        ),
    }, separated


def mode_metrics(
    model,
    output: dict,
    batch: dict[str, torch.Tensor],
    prior_samples: int,
) -> dict[str, float | int]:
    group_count = batch["history_features"].shape[0] // 3
    representatives = torch.arange(
        0,
        group_count * 3,
        3,
        device=batch["history_features"].device,
    )
    representative_batch = {
        key: value[representatives] if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    target_slots = output["target_future_slots"][: group_count * 3].reshape(
        group_count,
        3,
        *output["target_future_slots"].shape[1:],
    )
    target_centers = output["target_future_centers"][: group_count * 3].reshape(
        group_count,
        3,
        *output["target_future_centers"].shape[1:],
    )
    target_features = output["target_future_object_features"][
        : group_count * 3
    ].reshape(
        group_count,
        3,
        *output["target_future_object_features"].shape[1:],
    )
    ambiguity = batch["ambiguity"][representatives].bool()
    samples, sample_centers = model.predict_prior_states(
        representative_batch,
        prior_samples,
        stochastic=True,
    )
    deterministic, deterministic_centers = model.predict_prior_states(
        representative_batch,
        1,
        stochastic=False,
    )
    sample_error = (
        F.normalize(samples[:, :, None], dim=-1)
        - F.normalize(target_slots[None], dim=-1)
    ).square().mean(dim=(-1, -2, -3))
    deterministic_error = (
        F.normalize(deterministic[:, :, None], dim=-1)
        - F.normalize(target_slots[None], dim=-1)
    ).square().mean(dim=(-1, -2, -3))[0]
    target_distance = _pairwise_distance(target_slots, normalize=True)
    target_separation, off_diagonal = _mode_geometry(target_distance)
    latent_coverage, separated = _coverage(
        sample_error,
        deterministic_error,
        target_separation,
        ambiguity,
    )
    posterior_slot_prediction = output["predicted_future_slots"][
        : group_count * 3
    ].reshape_as(target_slots)
    posterior_slot_error = (
        F.normalize(posterior_slot_prediction, dim=-1)
        - F.normalize(target_slots, dim=-1)
    ).square().mean(dim=(-1, -2, -3))
    posterior_slot_choice = (
        F.normalize(posterior_slot_prediction[:, :, None], dim=-1)
        - F.normalize(target_slots[:, None], dim=-1)
    ).square().mean(dim=(-1, -2, -3)).argmin(dim=-1)
    posterior_slot_hit = (
        posterior_slot_error <= 0.25 * target_separation[:, None]
    )

    center_error = (
        sample_centers[:, :, None] - target_centers[None]
    ).square().mean(dim=(-1, -2, -3))
    deterministic_center_error = (
        deterministic_centers[:, :, None] - target_centers[None]
    ).square().mean(dim=(-1, -2, -3))[0]
    center_distance = _pairwise_distance(target_centers, normalize=False)
    center_separation, _ = _mode_geometry(center_distance)
    center_coverage, center_separated = _coverage(
        center_error,
        deterministic_center_error,
        center_separation,
        ambiguity,
    )
    posterior_center_prediction = output["predicted_future_centers"][
        : group_count * 3
    ].reshape_as(target_centers)
    posterior_center_error = (
        posterior_center_prediction - target_centers
    ).square().mean(dim=(-1, -2, -3))
    posterior_center_choice = (
        posterior_center_prediction[:, :, None] - target_centers[:, None]
    ).square().mean(dim=(-1, -2, -3)).argmin(dim=-1)
    posterior_center_hit = (
        posterior_center_error <= 0.25 * center_separation[:, None]
    )

    sample_features = model.object_aggregator.decode_feature(samples)
    deterministic_features = model.object_aggregator.decode_feature(
        deterministic
    )
    feature_error = (
        F.normalize(sample_features[:, :, None], dim=-1)
        - F.normalize(target_features[None], dim=-1)
    ).square().mean(dim=(-1, -2, -3))
    deterministic_feature_error = (
        F.normalize(deterministic_features[:, :, None], dim=-1)
        - F.normalize(target_features[None], dim=-1)
    ).square().mean(dim=(-1, -2, -3))[0]
    feature_distance = _pairwise_distance(target_features, normalize=True)
    feature_separation, _ = _mode_geometry(feature_distance)
    feature_coverage, feature_separated = _coverage(
        feature_error,
        deterministic_feature_error,
        feature_separation,
        ambiguity,
    )
    posterior_features = output["predicted_future_object_features"][
        : group_count * 3
    ].reshape_as(target_features)
    posterior_feature_error = (
        F.normalize(posterior_features, dim=-1)
        - F.normalize(target_features, dim=-1)
    ).square().mean(dim=(-1, -2, -3))
    posterior_feature_choice = (
        F.normalize(posterior_features[:, :, None], dim=-1)
        - F.normalize(target_features[:, None], dim=-1)
    ).square().mean(dim=(-1, -2, -3)).argmin(dim=-1)
    posterior_feature_hit = (
        posterior_feature_error <= 0.25 * feature_separation[:, None]
    )

    posterior = output["posterior_actions"][: group_count * 3].reshape(
        group_count,
        3,
        *output["posterior_actions"].shape[1:],
    )
    posterior_distance = _pairwise_distance(posterior, normalize=True)
    posterior_separation = posterior_distance.masked_select(
        off_diagonal.expand_as(posterior_distance)
    ).reshape(group_count, -1).mean(dim=-1)
    action_separation, _ = _mode_geometry(posterior_distance)
    context = output["prior_context"][: group_count * 3].reshape(
        group_count,
        3,
        *output["prior_context"].shape[1:],
    )
    action_samples = model.latent_actions.prior.sample(
        context[:, 0],
        sample_count=prior_samples,
        stochastic=True,
    )
    deterministic_actions = model.latent_actions.prior.sample(
        context[:, 0],
        sample_count=1,
        stochastic=False,
    )
    action_error = (
        F.normalize(action_samples[:, :, None], dim=-1)
        - F.normalize(posterior[None], dim=-1)
    ).square().mean(dim=(-1, -2, -3))
    deterministic_action_error = (
        F.normalize(deterministic_actions[:, :, None], dim=-1)
        - F.normalize(posterior[None], dim=-1)
    ).square().mean(dim=(-1, -2, -3))[0]
    action_coverage, action_separated = _coverage(
        action_error,
        deterministic_action_error,
        action_separation,
        ambiguity,
    )
    prior_diversity = samples.std(dim=0).mean(dim=(-1, -2, -3))
    center_diversity = sample_centers.std(dim=0).mean(dim=(-1, -2, -3))
    deterministic_group = ~ambiguity
    history_centers = output["target_history_centers"]
    history_delta = (
        batch["history_times"][:, -1] - batch["history_times"][:, -2]
    ).clamp_min(1e-6)
    center_velocity = (
        history_centers[:, -1] - history_centers[:, -2]
    ) / history_delta[:, None, None]
    future_horizon = (
        batch["future_times"] - batch["history_times"][:, -1, None]
    )
    extrapolated_centers = (
        history_centers[:, -1, None]
        + future_horizon[:, :, None, None] * center_velocity[:, None]
    )
    extrapolation_error = (
        extrapolated_centers - output["target_future_centers"]
    ).square().mean(dim=(-1, -2, -3))
    center_copy_error = (
        history_centers[:, -1, None] - output["target_future_centers"]
    ).square().mean(dim=(-1, -2, -3))
    deterministic_samples = ~batch["ambiguity"].bool()
    extrapolation_mse = _group_mean(
        extrapolation_error,
        deterministic_samples,
    )
    center_copy_mse = _group_mean(center_copy_error, deterministic_samples)
    result: dict[str, float | int] = {
        "ambiguous_groups": int(ambiguity.sum()),
        "deterministic_groups": int(deterministic_group.sum()),
        "deterministic_center_extrapolation_mse": extrapolation_mse,
        "deterministic_center_copy_mse": center_copy_mse,
        "deterministic_center_extrapolation_improvement": (
            (center_copy_mse - extrapolation_mse)
            / max(center_copy_mse, 1e-8)
        ),
        "prior_context_mode_max_difference": float(
            (context - context[:, :1]).abs().max()
        ),
        "target_mode_separation": _group_mean(
            target_separation,
            separated,
        ),
        "target_center_mode_separation": _group_mean(
            center_separation,
            center_separated,
        ),
        "target_object_feature_mode_separation": _group_mean(
            feature_separation,
            feature_separated,
        ),
        "posterior_action_mode_separation": _group_mean(
            posterior_separation,
            separated,
        ),
        "posterior_oracle_mse": _group_mean(
            posterior_slot_error.mean(dim=-1),
            separated,
        ),
        "posterior_oracle_mode_recall": _group_mean(
            posterior_slot_hit.float().mean(dim=-1),
            separated,
        ),
        "posterior_oracle_mode_nearest_accuracy": _group_mean(
            (
                posterior_slot_choice
                == torch.arange(3, device=posterior_slot_choice.device)[None]
            ).float().mean(dim=-1),
            separated,
        ),
        "posterior_oracle_center_mse": _group_mean(
            posterior_center_error.mean(dim=-1),
            center_separated,
        ),
        "posterior_oracle_center_mode_recall": _group_mean(
            posterior_center_hit.float().mean(dim=-1),
            center_separated,
        ),
        "posterior_oracle_center_mode_nearest_accuracy": _group_mean(
            (
                posterior_center_choice
                == torch.arange(3, device=posterior_center_choice.device)[None]
            ).float().mean(dim=-1),
            center_separated,
        ),
        "posterior_oracle_object_feature_mse": _group_mean(
            posterior_feature_error.mean(dim=-1),
            feature_separated,
        ),
        "posterior_oracle_object_feature_mode_recall": _group_mean(
            posterior_feature_hit.float().mean(dim=-1),
            feature_separated,
        ),
        "posterior_oracle_object_feature_mode_nearest_accuracy": _group_mean(
            (
                posterior_feature_choice
                == torch.arange(3, device=posterior_feature_choice.device)[None]
            ).float().mean(dim=-1),
            feature_separated,
        ),
        "action_target_mode_separation": _group_mean(
            action_separation,
            action_separated,
        ),
        "ambiguous_prior_diversity": _group_mean(
            prior_diversity,
            ambiguity,
        ),
        "deterministic_prior_diversity": _group_mean(
            prior_diversity,
            deterministic_group,
        ),
        "ambiguous_center_diversity": _group_mean(
            center_diversity,
            ambiguity,
        ),
        "deterministic_center_diversity": _group_mean(
            center_diversity,
            deterministic_group,
        ),
        **latent_coverage,
    }
    for index, name in enumerate(("negative", "neutral", "positive")):
        result[f"posterior_oracle_mode_recall_{name}"] = _group_mean(
            posterior_slot_hit[:, index].float(),
            separated,
        )
        result[f"posterior_oracle_object_feature_mode_recall_{name}"] = (
            _group_mean(
                posterior_feature_hit[:, index].float(),
                feature_separated,
            )
        )
        result[f"posterior_oracle_center_mode_recall_{name}"] = _group_mean(
            posterior_center_hit[:, index].float(),
            center_separated,
        )
    result.update(
        {
            f"center_{name}": value
            for name, value in center_coverage.items()
        }
    )
    result.update(
        {
            f"object_feature_{name}": value
            for name, value in feature_coverage.items()
        }
    )
    result.update(
        {
            f"action_{name}": value
            for name, value in action_coverage.items()
        }
    )
    return result
