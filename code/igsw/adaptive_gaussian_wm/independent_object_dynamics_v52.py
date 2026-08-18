"""External-mask motion probe for v52 independent Object State evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .independent_object_truth_v52 import patch_truth_masks


def _canonical_slots(assignment, masks, object_ids, object_slots):
    distributions, visibility = [], []
    for object_id in object_ids:
        support = masks == object_id
        mass = torch.einsum(
            "tnk,tn->tk", assignment[..., :object_slots], support.float()
        )
        mass = mass / support.sum(dim=1, keepdim=True).clamp_min(1.0)
        mass = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        distributions.append(mass)
        visibility.append(support.any(dim=1))
    distribution = torch.stack(distributions, dim=1)
    visible = torch.stack(visibility, dim=1)
    canonical = distribution.sum(dim=0)
    canonical = canonical / visible.sum(dim=0).clamp_min(1.0)[:, None]
    return canonical.argmax(dim=-1), visible


@torch.no_grad()
def independent_motion_metrics(model, features, state, truth) -> dict:
    masks = patch_truth_masks(truth["instance_masks"], features.grid_hw)
    assignment = state["assignment"][0].float()
    object_ids = truth["object_ids"]
    slots, visible = _canonical_slots(
        assignment, masks, object_ids, model.config.object_slots
    )
    if not bool((visible.sum(dim=0) >= 2).all()):
        raise ValueError("independent object disappears at the DINO patch grid")
    coordinates = features.coordinates[0].float()
    centers = assignment.new_zeros(len(masks), len(object_ids), 2)
    for frame in range(len(masks)):
        for object_index, object_id in enumerate(object_ids):
            support = masks[frame] == object_id
            if bool(support.any()):
                centers[frame, object_index] = coordinates[frame, support].mean(dim=0)

    root_motion = model.motion_readout(state["dynamic"].float()).reshape(
        *state["dynamic"].shape[:3], len(model.config.dynamic_horizons), 2
    )[0]
    prediction_sum = zero_sum = active_count = 0.0
    for horizon_index, horizon in enumerate(model.config.dynamic_horizons):
        if horizon >= len(masks):
            continue
        pair_valid = visible[:-horizon] & visible[horizon:]
        dt = truth["frame_times"][horizon:] - truth["frame_times"][:-horizon]
        displacement = centers[horizon:] - centers[:-horizon]
        velocity = displacement / dt[:, None, None].clamp_min(1e-4)
        global_velocity = (velocity * pair_valid[..., None]).sum(dim=1, keepdim=True)
        global_velocity = global_velocity / pair_valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1)[..., None]
        target = velocity - global_velocity
        predicted = torch.stack(
            [root_motion[horizon:, int(slot), horizon_index] for slot in slots],
            dim=1,
        )
        active = pair_valid & (target.norm(dim=-1) >= model.config.object_motion_floor)
        prediction_error = F.smooth_l1_loss(
            predicted, target, reduction="none"
        ).mean(dim=-1)
        zero_error = F.smooth_l1_loss(
            torch.zeros_like(target), target, reduction="none"
        ).mean(dim=-1)
        prediction_sum += float((prediction_error * active).sum())
        zero_sum += float((zero_error * active).sum())
        active_count += float(active.sum())
    return {
        "dynamic_motion_error": (prediction_sum, active_count),
        "dynamic_zero_motion_error": (zero_sum, active_count),
        "motion_active_cases": (active_count, 1.0),
    }
