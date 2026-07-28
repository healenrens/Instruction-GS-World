"""Detached diagnostics for the v30 dense object-conditioned readout."""

from __future__ import annotations

import math

import torch

from .change_objectives import dense_feature_loss
from .diagnostic_statistics import ratio_moments
from .readout_repair import direct_current_state


DENSE_READOUT_REQUIRED_DIAGNOSTICS = frozenset(
    {
        "readout_current_dense_feature",
        "readout_current_token_feature",
        "readout_current_gaussian_feature",
        "readout_current_scene_mean_feature",
        "readout_current_dense_gap_to_token",
        "readout_current_dense_gain_over_gaussian",
        "readout_future_dense_feature",
        "readout_future_persistence_feature",
        "readout_future_dense_gain_over_persistence",
        "readout_dynamic_dense_feature",
        "readout_dynamic_persistence_feature",
        "readout_dynamic_dense_gain_over_persistence",
        "readout_static_dense_feature",
        "readout_static_persistence_feature",
        "readout_static_dense_gain_over_persistence",
        "readout_dense_effective_tokens",
        "readout_dense_coverage_fraction",
        "readout_dense_future_assignment_js",
        "readout_dense_future_feature_residual_rms",
        "readout_dense_future_assignment_residual_rms",
        "readout_dense_future_background_residual_rms",
    }
)


def validate_dense_readout_diagnostic_contract(metrics: dict[str, float]) -> None:
    """Reject incomplete or numerically invalid dense-readout diagnostics."""
    missing = DENSE_READOUT_REQUIRED_DIAGNOSTICS.difference(metrics)
    if missing:
        raise ValueError(f"missing dense readout diagnostics: {sorted(missing)}")
    nonfinite = {
        name
        for name in DENSE_READOUT_REQUIRED_DIAGNOSTICS
        if not math.isfinite(metrics[name])
    }
    if nonfinite:
        raise ValueError(f"non-finite dense readout diagnostics: {sorted(nonfinite)}")
    coverage = metrics["readout_dense_coverage_fraction"]
    if not 0.0 <= coverage <= 1.0:
        raise ValueError(f"dense readout coverage is outside [0, 1]: {coverage}")
    if metrics["readout_dense_effective_tokens"] <= 0.0:
        raise ValueError("dense readout effective-token count must be positive")
    nonnegative = {
        "readout_dense_future_assignment_js",
        "readout_dense_future_feature_residual_rms",
        "readout_dense_future_assignment_residual_rms",
        "readout_dense_future_background_residual_rms",
    }
    invalid = {name for name in nonnegative if metrics[name] < -1e-6}
    if invalid:
        raise ValueError(f"negative dense readout diagnostics: {sorted(invalid)}")


def _feature_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    return dense_feature_loss(
        prediction,
        target,
        valid,
        torch.ones_like(valid, dtype=prediction.dtype),
    )


def _scene_mean(batch: dict, target: torch.Tensor) -> torch.Tensor:
    valid = batch["history_valid"][:, -1:].float()
    mean = (target.float() * valid[..., None]).sum(dim=2, keepdim=True)
    mean = mean / valid.sum(dim=2, keepdim=True)[..., None].clamp_min(1.0)
    return mean.expand_as(target)


def _change_masks(batch: dict, fraction: float = 0.2) -> tuple[torch.Tensor, ...]:
    future = batch["future_features"].detach().float()
    current = batch["history_features"][:, -1:].detach().float().expand_as(future)
    current_valid = batch["history_valid"][:, -1:].expand_as(batch["future_valid"])
    valid = batch["future_valid"] & current_valid
    score = (future - current).square().mean(dim=-1).sqrt()
    flattened_valid = valid.flatten(0, 1)
    flattened_score = score.flatten(0, 1)
    flattened_dynamic = torch.zeros_like(flattened_valid)
    for row in range(flattened_valid.shape[0]):
        indices = torch.nonzero(flattened_valid[row], as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError("dense diagnostic has no valid future patches")
        count = max(1, math.ceil(float(indices.numel()) * fraction))
        selected = torch.topk(
            flattened_score[row, indices], count, sorted=False
        ).indices
        flattened_dynamic[row, indices[selected]] = True
    dynamic = flattened_dynamic.unflatten(0, valid.shape[:2])
    return dynamic, valid & ~dynamic, valid


def _assignment_js(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    prediction = prediction.float().clamp_min(1e-7)
    reference = reference.float().clamp_min(1e-7)
    midpoint = 0.5 * (prediction + reference)
    divergence = 0.5 * (
        prediction * (prediction.log() - midpoint.log())
        + reference * (reference.log() - midpoint.log())
    ).sum(dim=2)
    weight = valid.float()
    return (divergence * weight).sum() / weight.sum().clamp_min(1.0)


def _rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(value.detach().float().square().mean())


@torch.no_grad()
def dense_object_readout_diagnostics(
    model,
    batch: dict,
    output: dict,
) -> dict[str, torch.Tensor]:
    """Separate carrier quality, future transport, and Dynamics quality."""
    current = output.get("current_dense_readout")
    future = output.get("dense_future_readout")
    reference = output.get("dense_reference_readout")
    if current is None or future is None or reference is None:
        raise ValueError("dense diagnostics require all dense readout states")
    tokens = output["history_token_states"][-1]
    current_target = batch["history_features"][:, -1:].detach().float()
    current_valid = batch["history_valid"][:, -1:]
    token_prediction = tokens.reconstructed_features.detach().float()[:, None]
    gaussian_prediction, _ = model.gaussian_readout.splat_features(
        direct_current_state(tokens),
        batch["history_coordinates"][:, -1:],
    )
    scene = _scene_mean(batch, current_target)
    current_dense = _feature_loss(current.feature, current_target, current_valid)
    current_token = _feature_loss(token_prediction, current_target, current_valid)
    current_gaussian = _feature_loss(gaussian_prediction, current_target, current_valid)
    current_scene = _feature_loss(scene, current_target, current_valid)

    future_target = batch["future_features"].detach().float()
    future_valid = batch["future_valid"]
    persistence = current_target.expand_as(future_target)
    future_dense = _feature_loss(
        output["rendered_future_features"].detach().float(),
        future_target,
        future_valid,
    )
    future_persistence = _feature_loss(persistence, future_target, future_valid)
    dynamic, static, _ = _change_masks(batch)
    dynamic_dense = _feature_loss(
        output["rendered_future_features"].detach().float(), future_target, dynamic
    )
    dynamic_persistence = _feature_loss(persistence, future_target, dynamic)
    static_dense = _feature_loss(
        output["rendered_future_features"].detach().float(), future_target, static
    )
    static_persistence = _feature_loss(persistence, future_target, static)

    effective = (
        current.assignment[:, 0]
        .float()
        .square()
        .sum(dim=1)
        .clamp_min(1e-8)
        .reciprocal()
    )
    valid_weight = current_valid[:, 0].float()
    denominator = valid_weight.sum().clamp_min(1.0)
    effective_mean = (effective * valid_weight).sum() / denominator
    coverage_fraction = (
        (current.coverage[:, 0] > 1e-4) & current_valid[:, 0]
    ).float().sum() / denominator
    result = {
        "readout_current_dense_feature": current_dense,
        "readout_current_token_feature": current_token,
        "readout_current_gaussian_feature": current_gaussian,
        "readout_current_scene_mean_feature": current_scene,
        "readout_current_dense_gap_to_token": current_dense - current_token,
        "readout_current_dense_gain_over_gaussian": current_gaussian - current_dense,
        "readout_future_dense_feature": future_dense,
        "readout_future_persistence_feature": future_persistence,
        "readout_future_dense_gain_over_persistence": (
            future_persistence - future_dense
        ),
        "readout_dynamic_dense_feature": dynamic_dense,
        "readout_dynamic_persistence_feature": dynamic_persistence,
        "readout_dynamic_dense_gain_over_persistence": (
            dynamic_persistence - dynamic_dense
        ),
        "readout_static_dense_feature": static_dense,
        "readout_static_persistence_feature": static_persistence,
        "readout_static_dense_gain_over_persistence": static_persistence - static_dense,
        "readout_dense_effective_tokens": effective_mean,
        "readout_dense_coverage_fraction": coverage_fraction,
        "readout_dense_future_assignment_js": _assignment_js(
            future.assignment, reference.assignment, future_valid
        ),
        "readout_dense_future_feature_residual_rms": _rms(
            future.feature_residual - reference.feature_residual
        ),
        "readout_dense_future_assignment_residual_rms": _rms(
            future.assignment - reference.assignment
        ),
        "readout_dense_future_background_residual_rms": _rms(
            future.background_residual - reference.background_residual
        ),
    }
    result.update(
        ratio_moments(
            "readout_current_dense_relative_gain_over_gaussian",
            current_gaussian - current_dense,
            current_gaussian,
        )
    )
    result.update(
        ratio_moments(
            "readout_future_dense_relative_gain_over_persistence",
            future_persistence - future_dense,
            future_persistence,
        )
    )
    return result
