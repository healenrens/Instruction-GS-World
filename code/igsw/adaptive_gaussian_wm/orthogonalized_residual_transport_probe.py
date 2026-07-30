"""Transport counterfactuals for orthogonalized object residual fields."""

from __future__ import annotations

import torch

from .orthogonalized_object_residual import (
    orthogonalized_residual_design,
    render_orthogonalized_object_residual,
)


def render_orthogonalized_variants(
    fit,
    coordinates: torch.Tensor,
    budget: int,
    predicted_centers: torch.Tensor,
    predicted_scale_ratio: torch.Tensor,
    predicted_presence: torch.Tensor,
    predicted_feature_delta: torch.Tensor,
    target_centers: torch.Tensor,
    target_scale_ratio: torch.Tensor,
    target_presence: torch.Tensor,
    target_feature_delta: torch.Tensor,
) -> dict[str, torch.Tensor]:
    def render(
        centers=None,
        scale=None,
        presence=None,
        feature=None,
        *,
        rigid: bool,
    ):
        return render_orthogonalized_object_residual(
            fit,
            coordinates,
            local_budget=budget,
            target_object_centers=centers,
            object_scale_ratio=scale,
            object_presence=presence,
            object_feature_delta=feature,
            transport_local_residual=rigid,
        )

    predicted_state = (predicted_centers, predicted_scale_ratio, predicted_presence)
    target_state = (target_centers, target_scale_ratio, target_presence)
    return {
        "persistence": render(rigid=False),
        "feature_only": render(feature=predicted_feature_delta, rigid=False),
        "target_feature_only": render(feature=target_feature_delta, rigid=False),
        "predicted_root": render(*predicted_state, rigid=False),
        "predicted_root_feature": render(
            *predicted_state, feature=predicted_feature_delta, rigid=False
        ),
        "predicted_rigid": render(*predicted_state, rigid=True),
        "predicted_rigid_feature": render(
            *predicted_state, feature=predicted_feature_delta, rigid=True
        ),
        "target_root": render(*target_state, rigid=False),
        "target_root_predicted_feature": render(
            *target_state, feature=predicted_feature_delta, rigid=False
        ),
        "target_root_target_feature": render(
            *target_state, feature=target_feature_delta, rigid=False
        ),
        "target_rigid": render(*target_state, rigid=True),
        "target_rigid_predicted_feature": render(
            *target_state, feature=predicted_feature_delta, rigid=True
        ),
        "target_rigid_target_feature": render(
            *target_state, feature=target_feature_delta, rigid=True
        ),
    }


def transported_column_amplification(
    fit,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    *,
    budget: int,
    centers: torch.Tensor,
    scale_ratio: torch.Tensor,
    presence: torch.Tensor,
    transport_local_residual: bool,
) -> dict[str, torch.Tensor]:
    local_arguments = (
        {
            "local_object_centers": centers,
            "local_scale_ratio": scale_ratio,
            "local_presence": presence,
        }
        if transport_local_residual
        else {}
    )
    design, _ = orthogonalized_residual_design(
        fit,
        coordinates,
        local_budget=budget,
        root_object_centers=centers,
        root_scale_ratio=scale_ratio,
        root_presence=presence,
        **local_arguments,
    )
    solution = fit.solutions[budget]
    weighted = design.float() * valid.float().sqrt()[:, None]
    target_norm = weighted.square().sum(dim=0).sqrt()
    ratio = target_norm / solution.column_scale.clamp_min(1e-8)
    ratio = ratio[solution.active_columns]
    prefix = "rigid" if transport_local_residual else "root"
    return {
        f"{prefix}_column_amplification_max": ratio.max(),
        f"{prefix}_column_amplification_rms": ratio.square().mean().sqrt(),
    }
