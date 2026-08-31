"""E0 objective: object-only continuous reconstruction from compressed carriers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .object_transition_metrics_v62 import (
    compact_object_baseline_errors_v62,
    decoded_field_errors_v62,
    lifecycle_error_v62,
    weighted_mean_v62,
)


def teacher_object_codec_objective_v62(state, decoded, frame, config):
    errors = decoded_field_errors_v62(decoded, frame)
    lifecycle = lifecycle_error_v62(state, frame)
    baseline = compact_object_baseline_errors_v62(frame, config.covariance_floor)
    identity_weight = frame["support"].float() * frame["visibility"].float()
    target_identity = (
        (frame["dino"].float() + frame["siglip"].float()) * identity_weight[..., None]
    ).sum(dim=1)
    target_identity = F.normalize(
        target_identity / identity_weight.sum(dim=1, keepdim=True).clamp_min(1e-6),
        dim=-1,
        eps=1e-6,
    )
    identity = weighted_mean_v62(
        1.0 - F.cosine_similarity(state.identity.float(), target_identity, dim=-1),
        frame["object_valid"].float(),
    )
    object_valid = frame["object_valid"].float()
    responsibility = state.assignment.float()
    responsibility = responsibility / responsibility.sum(dim=1, keepdim=True).clamp_min(
        1e-6
    )
    mass = (responsibility * object_valid[:, None, None]).sum(dim=(0, 2))
    mass = mass / mass.sum().clamp_min(1e-6)
    carrier_entropy = -(mass * mass.clamp_min(1e-6).log()).sum()
    has_valid = (object_valid.sum() > 0).float()
    effective_carriers = carrier_entropy.exp() * has_valid
    normalized_assignment = F.normalize(state.assignment.float(), dim=-1, eps=1e-6)
    overlap = torch.einsum("bkp,bjp->bkj", normalized_assignment, normalized_assignment)
    diagonal = torch.eye(config.carrier_count, device=overlap.device, dtype=torch.bool)
    overlap = overlap.masked_fill(diagonal[None], 0.0).mean(dim=(1, 2))
    overlap = weighted_mean_v62(overlap, object_valid)
    capacity = overlap + has_valid * (
        torch.log(torch.tensor(float(config.carrier_count), device=mass.device))
        - carrier_entropy
    )
    covariance_eigenvalues = torch.linalg.eigvalsh(state.covariance.float())
    covariance_weight = object_valid[:, None].expand_as(covariance_eigenvalues[..., 0])
    covariance_min_eigenvalue = weighted_mean_v62(
        covariance_eigenvalues[..., 0], covariance_weight
    )
    covariance_condition_number = weighted_mean_v62(
        covariance_eigenvalues[..., 1]
        / covariance_eigenvalues[..., 0].clamp_min(config.covariance_floor),
        covariance_weight,
    )
    total = (
        config.support_weight * errors["support_bce"]
        + config.semantic_weight
        * (errors["dino_cosine_error"] + errors["siglip_cosine_error"] + 0.5 * identity)
        + config.visibility_weight * errors["visibility_bce"]
        + config.lifecycle_weight * lifecycle
        + config.capacity_weight * capacity
    )
    parts = {
        "loss": total.detach(),
        **{name: value.detach() for name, value in errors.items()},
        "lifecycle_cross_entropy": lifecycle.detach(),
        "identity_cosine_error": identity.detach(),
        "carrier_capacity_penalty": capacity.detach(),
        "carrier_support_overlap": overlap.detach(),
        "carrier_effective_count": effective_carriers.detach(),
        "covariance_min_eigenvalue": covariance_min_eigenvalue.detach(),
        "covariance_condition_number": covariance_condition_number.detach(),
        "object_valid_fraction": frame["object_valid"].float().mean(),
        "positive_point_fraction": frame["support"].float().mean(),
        "support_gap_recovery": (
            1.0 - errors["support_bce"] / baseline["support_bce"].clamp_min(1e-6)
        ).detach(),
        "semantic_gap_recovery": (
            1.0
            - (errors["dino_cosine_error"] + errors["siglip_cosine_error"])
            / (
                baseline["dino_cosine_error"] + baseline["siglip_cosine_error"]
            ).clamp_min(1e-6)
        ).detach(),
        "lifecycle_gap_recovery": (
            1.0 - lifecycle / baseline["lifecycle_cross_entropy"].clamp_min(1e-6)
        ).detach(),
    }
    parts.update(
        {f"compact_baseline_{name}": value.detach() for name, value in baseline.items()}
    )
    return total, parts
