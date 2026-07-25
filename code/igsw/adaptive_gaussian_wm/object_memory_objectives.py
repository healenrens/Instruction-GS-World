"""Scale-free geometry objectives for persistent object memory."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .jepa_losses import weighted_mean


def object_memory_geometry_loss(
    output: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise observable support geometry without metric-depth claims."""
    reference = output["predicted_future_slots"].sum() * 0.0
    predicted_scale = output.get("predicted_future_relative_scale")
    target_scale = output.get("target_future_relative_scale")
    predicted_relations = output.get("predicted_future_relations")
    target_relations = output.get("target_future_relations")
    if predicted_scale is None:
        return reference, {
            "geometry_relative_scale": reference,
            "geometry_image_plane_relations": reference,
        }
    if target_scale is None or predicted_relations is None or target_relations is None:
        raise ValueError("factorized Dynamics requires target memory geometry")
    activity = output["target_future_activity"].detach()
    scale_error = F.smooth_l1_loss(
        predicted_scale.clamp_min(1e-6).log(),
        target_scale.detach().clamp_min(1e-6).log(),
        beta=0.1,
        reduction="none",
    )
    scale = weighted_mean(scale_error, activity)

    # Only image-plane displacement and scale ratio are observable here.
    relation_error = F.smooth_l1_loss(
        predicted_relations[..., :3],
        target_relations.detach()[..., :3],
        beta=0.1,
        reduction="none",
    ).mean(dim=-1)
    pair_weight = (
        activity.unsqueeze(-1) * activity.unsqueeze(-2)
    )
    relations = weighted_mean(relation_error, pair_weight)
    total = scale + relations
    return total, {
        "geometry_relative_scale": scale,
        "geometry_image_plane_relations": relations,
    }
