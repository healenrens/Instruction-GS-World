"""Absolute and intervention metrics for v62 object transitions."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .covariance_geometry_v62 import (
    mahalanobis_squared_v62,
    object_spatial_moments_v62,
)


def weighted_mean_v62(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value.float() * weight.float()).sum() / weight.float().sum().clamp_min(1.0)


def decoded_field_errors_v62(decoded, frame):
    valid = frame["object_valid"][:, None].float()
    support = frame["support"].float()
    positive = support * valid
    point_weight = valid.expand_as(support)
    visibility_weight = frame["membership"].float() * valid
    support_bce = weighted_mean_v62(
        F.binary_cross_entropy_with_logits(
            decoded.support_logits.float(), support, reduction="none"
        ),
        point_weight,
    )
    dino_error = weighted_mean_v62(
        1.0 - F.cosine_similarity(decoded.dino.float(), frame["dino"].float(), dim=-1),
        positive,
    )
    siglip_error = weighted_mean_v62(
        1.0
        - F.cosine_similarity(decoded.siglip.float(), frame["siglip"].float(), dim=-1),
        positive,
    )
    visibility_bce = weighted_mean_v62(
        F.binary_cross_entropy_with_logits(
            decoded.visibility_logits.float(),
            frame["visibility"].float(),
            reduction="none",
        ),
        visibility_weight,
    )
    probability = torch.sigmoid(decoded.support_logits.float())
    intersection = (probability * support * point_weight).sum()
    union = ((probability + support - probability * support) * point_weight).sum()
    support_iou = intersection / union.clamp_min(1e-6)
    return {
        "support_bce": support_bce,
        "dino_cosine_error": dino_error,
        "siglip_cosine_error": siglip_error,
        "visibility_bce": visibility_bce,
        "support_soft_iou": support_iou,
    }


def compact_object_baseline_errors_v62(frame, covariance_floor: float):
    valid = frame["object_valid"][:, None].float()
    support = frame["support"].float()
    positive = support * valid
    coordinates = frame["coordinates"].float()
    center, covariance = object_spatial_moments_v62(
        coordinates, positive, covariance_floor
    )
    offset = coordinates - center[:, None]
    squared = mahalanobis_squared_v62(offset[:, :, None], covariance[:, None])[:, :, 0]
    support_probability = torch.exp(-0.5 * squared).clamp(1e-4, 1.0 - 1e-4)
    support_bce = weighted_mean_v62(
        F.binary_cross_entropy_with_logits(
            torch.logit(support_probability), support, reduction="none"
        ),
        valid.expand_as(support),
    )

    def pooled_error(name):
        target = frame[name].float()
        pooled = (target * positive[..., None]).sum(dim=1)
        pooled = F.normalize(
            pooled / positive.sum(dim=1, keepdim=True).clamp_min(1e-6), dim=-1
        )
        error = 1.0 - F.cosine_similarity(pooled[:, None], target, dim=-1)
        return weighted_mean_v62(error, positive)

    visible_probability = torch.full_like(support, 0.95)
    visibility_bce = weighted_mean_v62(
        F.binary_cross_entropy_with_logits(
            torch.logit(visible_probability),
            frame["visibility"].float(),
            reduction="none",
        ),
        frame["membership"].float() * valid,
    )
    lifecycle_probability = frame["lifecycle"].new_tensor((0.98, 0.01, 0.01))
    lifecycle = -(frame["lifecycle"].float() * lifecycle_probability.log()).sum(-1)
    lifecycle = weighted_mean_v62(lifecycle, frame["object_valid"].float())
    return {
        "support_bce": support_bce,
        "dino_cosine_error": pooled_error("dino"),
        "siglip_cosine_error": pooled_error("siglip"),
        "visibility_bce": visibility_bce,
        "lifecycle_cross_entropy": lifecycle,
    }


def lifecycle_error_v62(state, frame):
    target = frame["lifecycle"].float()
    cross_entropy = -(target * state.lifecycle_logits.float().log_softmax(dim=-1)).sum(
        -1
    )
    return weighted_mean_v62(cross_entropy, frame["object_valid"].float())


def state_geometry_error_v62(predicted, target, valid):
    predicted_weight = predicted.presence.float()
    target_weight = target.presence.float()
    predicted_center = (predicted.center.float() * predicted_weight[..., None]).sum(1)
    predicted_center = predicted_center / predicted_weight.sum(
        1, keepdim=True
    ).clamp_min(1e-6)
    target_center = (target.center.float() * target_weight[..., None]).sum(1)
    target_center = target_center / target_weight.sum(1, keepdim=True).clamp_min(1e-6)
    center = weighted_mean_v62((predicted_center - target_center).norm(dim=-1), valid)
    predicted_scale = predicted.covariance.float().diagonal(dim1=-2, dim2=-1).sum(-1)
    target_scale = target.covariance.float().diagonal(dim1=-2, dim2=-1).sum(-1)
    predicted_scale = (predicted_scale * predicted_weight).sum(1)
    predicted_scale = predicted_scale / predicted_weight.sum(1).clamp_min(1e-6)
    target_scale = (target_scale * target_weight).sum(1)
    target_scale = target_scale / target_weight.sum(1).clamp_min(1e-6)
    scale = weighted_mean_v62(
        (
            predicted_scale.clamp_min(1e-6).log() - target_scale.clamp_min(1e-6).log()
        ).abs(),
        valid,
    )
    return center + 0.5 * scale, center, scale


def transition_error_sum_v62(field_errors, lifecycle, geometry):
    return (
        field_errors["support_bce"]
        + field_errors["dino_cosine_error"]
        + field_errors["siglip_cosine_error"]
        + 0.25 * field_errors["visibility_bce"]
        + 0.25 * lifecycle
        + 0.25 * geometry
    )
