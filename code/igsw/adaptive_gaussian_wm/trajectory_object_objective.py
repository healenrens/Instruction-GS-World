"""Pure-video losses for trajectory-anchored object state and latent effects."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .distributed_statistics import gather_batch_with_grad
from .trajectory_state_alignment import SlotTrajectory
from .trajectory_teacher import TrajectoryEvidence, transported_assignment


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _manual_binary_cross_entropy(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    prediction = prediction.float().clamp(1e-6, 1.0 - 1e-6)
    target = target.float().clamp(0.0, 1.0)
    return -(target * prediction.log() + (1.0 - target) * (1.0 - prediction).log())


def _trajectory_assignment_loss(
    assignment: torch.Tensor,
    evidence: TrajectoryEvidence,
    object_slots: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    persistent = assignment[..., : object_slots + 1].float()
    source = F.normalize(persistent[:, :-1].clamp_min(0.0), p=1, dim=-1, eps=1e-6)
    aligned_target = transported_assignment(evidence.forward, persistent[:, 1:])
    distance = 1.0 - F.cosine_similarity(source, aligned_target, dim=-1, eps=1e-6)
    source_mass = persistent[:, :-1].sum(dim=-1)
    target_mass = aligned_target.sum(dim=-1)
    weight = evidence.confidence.float() * source_mass * target_mass
    forward_loss = _weighted_mean(distance, weight)

    backward_source = F.normalize(
        persistent[:, 1:].clamp_min(0.0), p=1, dim=-1, eps=1e-6
    )
    aligned_backward = transported_assignment(evidence.backward, persistent[:, :-1])
    backward_distance = 1.0 - F.cosine_similarity(
        backward_source, aligned_backward, dim=-1, eps=1e-6
    )
    backward_weight = evidence.confidence.float() * backward_source.sum(-1)
    backward_loss = _weighted_mean(backward_distance, backward_weight)
    transient = assignment[..., object_slots + 1].float()
    transient_penalty = _weighted_mean(
        transient[:, :-1] + torch.einsum(
            "btij,btj->bti", evidence.forward.float(), transient[:, 1:]
        ),
        evidence.confidence.float(),
    )
    loss = 0.5 * (forward_loss + backward_loss) + 0.25 * transient_penalty
    return loss, {
        "loss_trajectory_assignment": loss,
        "trajectory_forward_distance": forward_loss,
        "trajectory_backward_distance": backward_loss,
        "trajectory_transient_penalty": transient_penalty,
        "trajectory_confidence": evidence.confidence.float().mean(),
        "trajectory_confident_fraction": (evidence.confidence > 0).float().mean(),
    }


def _common_fate_loss(
    patches: torch.Tensor,
    coordinates: torch.Tensor,
    assignment: torch.Tensor,
    evidence: TrajectoryEvidence,
    config,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    patch_count = patches.shape[2]
    count = min(config.common_fate_queries, patch_count)
    indices = torch.linspace(
        0, patch_count - 1, count, device=patches.device
    ).round().long().unique()
    feature = F.normalize(patches[:, :-1, indices].float(), dim=-1, eps=1e-6)
    appearance = torch.einsum("btiq,btjq->btij", feature, feature)
    appearance = ((appearance - 0.20) / 0.80).clamp(0.0, 1.0)
    flow = evidence.residual_flow[:, :, indices].float()
    motion_distance = (
        flow[:, :, :, None] - flow[:, :, None]
    ).square().sum(dim=-1)
    motion = torch.exp(
        -motion_distance / (2.0 * config.common_fate_motion_sigma**2)
    )
    position = coordinates[:, :-1, indices].float()
    spatial_distance = (
        position[:, :, :, None] - position[:, :, None]
    ).square().sum(dim=-1)
    spatial = torch.exp(
        -spatial_distance / (2.0 * config.common_fate_spatial_sigma**2)
    )
    target = appearance * (0.5 + 0.5 * motion) * spatial
    persistent = assignment[:, :-1, indices, : config.object_slots + 1].float()
    prediction = torch.einsum("btio,btjo->btij", persistent, persistent).clamp(0.0, 1.0)
    pair_weight = evidence.confidence[:, :, indices].float()
    pair_weight = torch.sqrt(pair_weight[:, :, :, None] * pair_weight[:, :, None])
    diagonal = torch.eye(len(indices), device=patches.device, dtype=torch.bool)
    pair_weight = pair_weight.masked_fill(diagonal[None, None], 0.0)
    loss = _weighted_mean(_manual_binary_cross_entropy(prediction, target), pair_weight)
    positive = (target >= 0.5).float() * pair_weight
    negative = (target <= 0.1).float() * pair_weight
    positive_score = _weighted_mean(prediction, positive)
    negative_score = _weighted_mean(prediction, negative)
    return loss, {
        "loss_common_fate": loss,
        "common_fate_positive_coassignment": positive_score,
        "common_fate_negative_coassignment": negative_score,
        "common_fate_margin": positive_score - negative_score,
        "common_fate_query_count": patches.new_tensor(float(len(indices))),
    }


def _identity_loss(
    identity: torch.Tensor,
    trajectory: SlotTrajectory,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, transitions, slots = trajectory.adjacent.shape[:3]
    source = F.normalize(identity[:, :-1].float(), dim=-1, eps=1e-6)
    target = F.normalize(identity[:, 1:].detach().float(), dim=-1, eps=1e-6)
    candidates = target.permute(1, 0, 2, 3).reshape(
        transitions, batch * slots, identity.shape[-1]
    )
    logits = torch.einsum("btkd,tqd->btkq", source, candidates) / temperature
    matched = trajectory.adjacent.argmax(dim=-1)
    sequence_offset = torch.arange(batch, device=identity.device)[:, None, None] * slots
    labels = matched + sequence_offset
    item_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).reshape_as(labels)
    weight = trajectory.confidence.float()
    loss = _weighted_mean(item_loss, weight)
    accuracy = _weighted_mean((logits.argmax(-1) == labels).float(), weight)
    return loss, accuracy


def _masked_state_loss(
    full: dict[str, torch.Tensor],
    masked: dict[str, torch.Tensor],
    observation_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    hidden = (~observation_mask)[..., None].float()
    weight = hidden * full["visibility"].detach().float().clamp_min(0.05)
    identity = 1.0 - F.cosine_similarity(
        masked["identity"].float(), full["identity"].detach().float(), dim=-1, eps=1e-6
    )
    dynamic = 1.0 - F.cosine_similarity(
        masked["dynamic"].float(), full["dynamic"].detach().float(), dim=-1, eps=1e-6
    )
    center = F.smooth_l1_loss(
        masked["center"].float(), full["center"].detach().float(), reduction="none"
    ).mean(dim=-1)
    presence = (masked["presence"].float() - full["presence"].detach().float()).abs()
    values = {
        "masked_identity_distance": _weighted_mean(identity, weight),
        "masked_dynamic_distance": _weighted_mean(dynamic, weight),
        "masked_center_distance": _weighted_mean(center, weight),
        "masked_presence_distance": _weighted_mean(presence, weight),
    }
    total = (
        values["masked_identity_distance"]
        + 0.10 * values["masked_dynamic_distance"]
        + 0.25 * values["masked_center_distance"]
        + 0.25 * values["masked_presence_distance"]
    )
    return total, {"loss_masked_state": total, **values}


def object_state_distance(
    prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weight = target["presence"].detach().float().clamp_min(0.05)
    identity = 1.0 - F.cosine_similarity(
        prediction["identity"].float(), target["identity"].detach().float(), dim=-1, eps=1e-6
    )
    dynamic = 1.0 - F.cosine_similarity(
        prediction["dynamic"].float(), target["dynamic"].detach().float(), dim=-1, eps=1e-6
    )
    predicted_relative = prediction["center"][:, :, None] - prediction["center"][:, None]
    target_relative = target["center"][:, :, None] - target["center"][:, None]
    geometry = F.smooth_l1_loss(
        predicted_relative.float(), target_relative.detach().float(), reduction="none"
    ).mean(dim=-1)
    pair_weight = weight[:, :, None] * weight[:, None]
    scale = F.smooth_l1_loss(
        prediction["log_scale"].float(), target["log_scale"].detach().float(), reduction="none"
    )
    lifecycle = (
        (prediction["presence"].float() - target["presence"].detach().float()).abs()
        + (prediction["visibility"].float() - target["visibility"].detach().float()).abs()
    )
    parts = {
        "identity": _weighted_mean(identity, weight),
        "dynamic": _weighted_mean(dynamic, weight),
        "relative_geometry": _weighted_mean(geometry, pair_weight),
        "relative_scale": _weighted_mean(scale, weight),
        "lifecycle": _weighted_mean(lifecycle, weight),
    }
    total = (
        0.25 * parts["identity"]
        + parts["dynamic"]
        + 0.50 * parts["relative_geometry"]
        + 0.25 * parts["relative_scale"]
        + 0.25 * parts["lifecycle"]
    )
    return total, parts


def _effect_regularization(effect: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gathered = gather_batch_with_grad(effect.float()).flatten(1)
    deviation = gathered.std(dim=0, unbiased=False)
    variance = F.relu(0.05 - deviation).mean()
    centered = gathered - gathered.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(len(centered), 1)
    off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
    return variance, off_diagonal.square().mean()


def trajectory_object_state_loss(
    model,
    patches: torch.Tensor,
    valid: torch.Tensor,
    observation_mask: torch.Tensor,
    evidence: TrajectoryEvidence,
    trajectory: SlotTrajectory,
    output: dict,
    effect_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = F.normalize(patches.float(), dim=-1, eps=1e-6)
    reconstruction_error = 1.0 - (
        output["reconstruction"].float() * target
    ).sum(dim=-1)
    reconstruction = _weighted_mean(reconstruction_error, valid.float())
    frame_mean = F.normalize(
        (target * valid[..., None].float()).sum(dim=2, keepdim=True)
        / valid.float().sum(dim=2, keepdim=True).clamp_min(1.0)[..., None],
        dim=-1,
        eps=1e-6,
    )
    frame_mean_error = _weighted_mean(
        1.0 - (frame_mean * target).sum(dim=-1), valid.float()
    )
    assignment = output["full_state"]["assignment"]
    trajectory_loss, trajectory_parts = _trajectory_assignment_loss(
        assignment, evidence, model.config.object_slots
    )
    common_fate, common_fate_parts = _common_fate_loss(
        patches, output["coordinates"], assignment, evidence, model.config
    )
    identity, identity_top1 = _identity_loss(
        output["full_state"]["identity"], trajectory, model.config.identity_temperature
    )
    masked, masked_parts = _masked_state_loss(
        output["full_state"], output["masked_state"], observation_mask
    )
    full = output["full_state"]
    assignment_object = assignment[..., : model.config.object_slots].sum(dim=-1)
    assignment_transient = assignment[..., model.config.object_slots + 1]
    motion = evidence.motion_salience.float()
    motion_coverage = _weighted_mean(
        F.relu(0.5 - assignment_object[:, :-1]), motion
    )
    lifecycle = F.relu(full["visibility"].float() - full["presence"].float()).mean()
    transient_fraction = _weighted_mean(assignment_transient, valid.float())
    lifecycle = lifecycle + F.relu(
        transient_fraction - model.config.transient_budget
    )
    identity_normalized = F.normalize(full["identity"].float(), dim=-1, eps=1e-6)
    pair_similarity = torch.einsum(
        "btkd,btjd->btkj", identity_normalized, identity_normalized
    )
    slots = model.config.object_slots
    diversity = F.relu(pair_similarity - 0.20)
    diversity = (
        diversity.sum(dim=(-1, -2)) - 0.80 * slots
    ).clamp_min(0.0).mean() / (slots * (slots - 1))
    state_loss = (
        model.config.reconstruction_weight * reconstruction
        + model.config.trajectory_weight * trajectory_loss
        + model.config.common_fate_weight * common_fate
        + model.config.identity_weight * identity
        + model.config.masked_state_weight * masked
        + model.config.lifecycle_weight * lifecycle
        + model.config.diversity_weight * diversity
        + model.config.motion_coverage_weight * motion_coverage
    )

    correct, effect_state_parts = object_state_distance(
        output["effect_prediction"], output["effect_target"]
    )
    zero, _ = object_state_distance(output["zero_prediction"], output["effect_target"])
    shuffled, _ = object_state_distance(
        output["shuffled_prediction"], output["effect_target"]
    )
    intervention = F.relu(model.config.intervention_margin + correct - zero)
    intervention = intervention + F.relu(
        model.config.intervention_margin + correct - shuffled
    )
    variance, covariance = _effect_regularization(output["effect"])
    effect_loss = correct + intervention + 0.05 * variance + 0.01 * covariance
    total = state_loss + float(effect_weight) * effect_loss
    owner_mass = assignment.float().sum(dim=2)
    valid_count = valid.float().sum(dim=2, keepdim=True).clamp_min(1.0)
    active = (
        owner_mass[..., : model.config.object_slots] / valid_count >= 0.01
    ).float().sum(dim=-1).mean()
    parts = {
        "loss": total,
        "loss_object_state": state_loss,
        "loss_reconstruction": reconstruction,
        "diagnostic_frame_mean_error": frame_mean_error,
        "loss_identity": identity,
        "trajectory_identity_top1": identity_top1,
        "loss_lifecycle": lifecycle,
        "loss_identity_diversity": diversity,
        "loss_motion_object_coverage": motion_coverage,
        "object_active_count": active,
        "object_owner_fraction": _weighted_mean(assignment_object, valid.float()),
        "scene_owner_fraction": _weighted_mean(
            assignment[..., model.config.object_slots], valid.float()
        ),
        "transient_owner_fraction": transient_fraction,
        "presence_mean": full["presence"].float().mean(),
        "visibility_mean": full["visibility"].float().mean(),
        "loss_effect": effect_loss,
        "effect_correct_distance": correct,
        "effect_zero_distance": zero,
        "effect_shuffled_distance": shuffled,
        "effect_gain_over_zero": (zero - correct) / zero.detach().clamp_min(1e-6),
        "effect_gain_over_shuffled": (
            shuffled - correct
        ) / shuffled.detach().clamp_min(1e-6),
        "effect_intervention": intervention,
        "effect_variance_penalty": variance,
        "effect_covariance_penalty": covariance,
        "effect_std": gather_batch_with_grad(output["effect"].float()).std(
            dim=0, unbiased=False
        ).mean(),
        "effect_norm": output["effect"].float().norm(dim=-1).mean(),
        "curriculum_effect_weight": total.new_tensor(float(effect_weight)),
    }
    parts.update(trajectory_parts)
    parts.update(common_fate_parts)
    parts.update(masked_parts)
    parts.update(
        {f"effect_state_{name}": value for name, value in effect_state_parts.items()}
    )
    if not bool(torch.isfinite(total)):
        raise RuntimeError("v49 objective produced a non-finite loss")
    return total, parts

