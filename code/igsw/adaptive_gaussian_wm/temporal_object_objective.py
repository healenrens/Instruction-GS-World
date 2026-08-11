"""Three compact objectives for the v44 Temporal Object Set model."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .distributed_statistics import gather_batch_with_grad
from .v44_curriculum import V44Curriculum


def _normalize_distribution(value: torch.Tensor) -> torch.Tensor:
    return value.float() / value.float().sum(dim=-1, keepdim=True).clamp_min(1e-6)


def _js_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = _normalize_distribution(left).clamp_min(1e-7)
    right = _normalize_distribution(right).clamp_min(1e-7)
    middle = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.log() - middle.log())).sum(dim=-1)
        + (right * (right.log() - middle.log())).sum(dim=-1)
    )


def _pairwise_center(center: torch.Tensor) -> torch.Tensor:
    return center[:, :, None] - center[:, None]


def object_state_distance(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weight = target["presence"].float().clamp_min(0.05)
    denominator = weight.sum().clamp_min(1.0)
    dynamic = 1.0 - F.cosine_similarity(
        prediction["dynamic"].float(), target["dynamic"].float(), dim=-1, eps=1e-6
    )
    dynamic = (dynamic * weight).sum() / denominator
    relative_center = F.smooth_l1_loss(
        _pairwise_center(prediction["center"].float()),
        _pairwise_center(target["center"].float()),
        reduction="none",
    ).mean(dim=-1)
    pair_weight = weight[:, :, None] * weight[:, None]
    relative_center = (
        relative_center * pair_weight
    ).sum() / pair_weight.sum().clamp_min(1.0)
    relative_scale = F.smooth_l1_loss(
        prediction["log_scale"].float(), target["log_scale"].float(), reduction="none"
    ).mean(dim=-1)
    relative_scale = (relative_scale * weight).sum() / denominator
    lifecycle = F.smooth_l1_loss(
        torch.stack((prediction["presence"], prediction["visibility"]), dim=-1).float(),
        torch.stack((target["presence"], target["visibility"]), dim=-1).float(),
    )
    total = dynamic + 0.5 * relative_center + 0.25 * relative_scale + 0.25 * lifecycle
    return total, {
        "dynamic": dynamic,
        "relative_center": relative_center,
        "relative_scale": relative_scale,
        "lifecycle": lifecycle,
    }


def _object_loss(model, output: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    config = model.config
    full, masked = output["full_state"], output["masked_state"]
    correspondence = output["correspondence"]
    assignment = full["assignment"].float()
    transported = torch.einsum(
        "btsn,btnm->btsm", assignment[:, :-1], correspondence.forward
    )
    transport = _js_distance(transported, assignment[:, 1:]).mean()
    cycle = correspondence.cycle_error.mean()
    reconstructed = torch.einsum(
        "btsn,btsd->btnd", assignment, full["decoded_slots"].float()
    )
    reconstruction = (
        1.0
        - F.cosine_similarity(
            reconstructed,
            full["patch_embedding"].float(),
            dim=-1,
            eps=1e-6,
        )
    ).mean()
    mask = (~output["observation_mask"])[..., None].float()
    masked_count = mask.sum().clamp_min(1.0)
    masked_semantic = 1.0 - F.cosine_similarity(
        masked["semantic"].float(), full["semantic"].detach().float(), dim=-1, eps=1e-6
    )
    masked_dynamic = F.smooth_l1_loss(
        masked["dynamic"].float(), full["dynamic"].detach().float(), reduction="none"
    ).mean(dim=-1)
    masked_state = ((masked_semantic + masked_dynamic) * mask).sum() / (
        masked_count * config.total_slots
    )
    observed_frames = output["observation_mask"][..., None].float()
    lifecycle_count = (observed_frames.sum() * config.total_slots).clamp_min(1.0)
    lifecycle_target = full["observed_presence"].detach().float()
    lifecycle_prediction = (
        (
            (full["predicted_presence"].float() - lifecycle_target).abs()
            + (full["predicted_visibility"].float() - lifecycle_target).abs()
        )
        * observed_frames
    ).sum() / lifecycle_count
    motion = correspondence.residual_motion.float()
    uncertainty = correspondence.cycle_error.float()
    scene_target = torch.exp(-4.0 * motion) * (1.0 - uncertainty)
    transient_target = uncertainty
    object_target = (1.0 - scene_target - transient_target).clamp_min(0.0)
    role_target = torch.stack((object_target, scene_target, transient_target), dim=2)
    role_target = role_target / role_target.sum(dim=2, keepdim=True).clamp_min(1e-6)
    next_assignment = assignment[:, 1:]
    role_prediction = torch.stack(
        (
            next_assignment[:, :, : config.object_slots].sum(dim=2),
            next_assignment[:, :, config.object_slots],
            next_assignment[:, :, config.object_slots + 1],
        ),
        dim=2,
    ).clamp_min(1e-7)
    role = -(role_target * role_prediction.log()).sum(dim=2).mean()
    object_semantic = F.normalize(
        full["semantic"][:, :, : config.object_slots].float(), dim=-1, eps=1e-6
    )
    gram = torch.einsum("btkd,btjd->btkj", object_semantic, object_semantic)
    identity = torch.eye(config.object_slots, device=gram.device)[None, None]
    diversity = ((gram - identity).square() * (1.0 - identity)).mean()
    scene_fraction = assignment[:, :, config.object_slots].mean()
    transient_fraction = assignment[:, :, config.object_slots + 1].mean()
    role_capacity = (scene_fraction - 0.60).clamp_min(0.0) + (
        transient_fraction - 0.25
    ).clamp_min(0.0)
    total = (
        transport
        + 0.25 * cycle
        + 0.50 * reconstruction
        + 0.50 * masked_state
        + 0.10 * lifecycle_prediction
        + 0.20 * role
        + 0.05 * diversity
        + role_capacity
    )
    mass = assignment[:, :, : config.object_slots].mean(dim=-1)
    effective_objects = (mass > 0.01).float().sum(dim=-1).mean()
    return total, {
        "loss_object_transport": transport,
        "loss_object_cycle": cycle,
        "loss_object_reconstruction": reconstruction,
        "loss_object_masked_state": masked_state,
        "loss_object_lifecycle_prediction": lifecycle_prediction,
        "loss_object_role": role,
        "loss_object_diversity": diversity,
        "object_scene_fraction": scene_fraction,
        "object_transient_fraction": transient_fraction,
        "object_effective_count": effective_objects,
        "object_assignment_entropy": -(
            assignment.clamp_min(1e-7) * assignment.clamp_min(1e-7).log()
        )
        .sum(dim=2)
        .mean(),
    }


def _intervention_loss(
    correct: torch.Tensor,
    zero: torch.Tensor,
    shuffled: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    return F.relu(margin + correct - zero) + F.relu(margin + correct - shuffled)


def temporal_object_set_loss(
    model,
    output: dict,
    curriculum: V44Curriculum,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    object_loss, parts = _object_loss(model, output)
    short_correct, short_parts = object_state_distance(
        output["short_prediction"], output["short_target"]
    )
    short_zero, _ = object_state_distance(output["short_zero"], output["short_target"])
    short_shuffled, _ = object_state_distance(
        output["short_shuffled"], output["short_target"]
    )
    effect_intervention = _intervention_loss(
        short_correct, short_zero, short_shuffled, model.config.intervention_margin
    )
    gathered_effect = gather_batch_with_grad(output["short_effect"].float()).flatten(1)
    effect_std = gathered_effect.std(dim=0, unbiased=False)
    effect_variance = F.relu(0.05 - effect_std).mean()
    effect_loss = short_correct + effect_intervention + 0.05 * effect_variance

    goal_correct, goal_parts = object_state_distance(
        output["goal_prediction"], output["goal_target"]
    )
    goal_zero, _ = object_state_distance(output["goal_zero"], output["goal_target"])
    goal_shuffled, _ = object_state_distance(
        output["goal_shuffled"], output["goal_target"]
    )
    goal_intervention = _intervention_loss(
        goal_correct, goal_zero, goal_shuffled, model.config.intervention_margin
    )
    goal_effect_alignment = (
        1.0
        - F.cosine_similarity(
            output["goal_effect"].float().flatten(1),
            output["trajectory_effect"].detach().float().flatten(1),
            dim=-1,
            eps=1e-6,
        ).mean()
    )
    goal_loss = goal_correct + goal_intervention + 0.5 * goal_effect_alignment
    total = (
        curriculum.object_weight * object_loss
        + curriculum.effect_weight * effect_loss
        + curriculum.goal_weight * goal_loss
    )
    parts.update(
        {
            "loss": total,
            "loss_object": object_loss,
            "loss_effect": effect_loss,
            "loss_goal": goal_loss,
            "effect_correct_distance": short_correct,
            "effect_zero_distance": short_zero,
            "effect_shuffled_distance": short_shuffled,
            "effect_relative_gain_over_zero": (short_zero - short_correct)
            / short_zero.detach().clamp_min(1e-6),
            "effect_relative_gain_over_shuffled": (short_shuffled - short_correct)
            / short_shuffled.detach().clamp_min(1e-6),
            "effect_action_std": effect_std.mean(),
            "effect_action_norm": output["short_effect"].float().norm(dim=-1).mean(),
            "goal_correct_distance": goal_correct,
            "goal_zero_distance": goal_zero,
            "goal_shuffled_distance": goal_shuffled,
            "goal_relative_gain_over_zero": (goal_zero - goal_correct)
            / goal_zero.detach().clamp_min(1e-6),
            "goal_effect_alignment": goal_effect_alignment,
            "curriculum_object_weight": torch.tensor(
                curriculum.object_weight, device=total.device
            ),
            "curriculum_effect_weight": torch.tensor(
                curriculum.effect_weight, device=total.device
            ),
            "curriculum_goal_weight": torch.tensor(
                curriculum.goal_weight, device=total.device
            ),
        }
    )
    parts.update({f"effect_state_{name}": value for name, value in short_parts.items()})
    parts.update({f"goal_state_{name}": value for name, value in goal_parts.items()})
    if not bool(torch.isfinite(total)):
        raise RuntimeError("v44 objective produced a non-finite loss")
    return total, parts
