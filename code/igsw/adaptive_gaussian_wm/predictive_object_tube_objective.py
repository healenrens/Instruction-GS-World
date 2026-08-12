"""Predictive object-tube, continuous-effect, and image-goal objectives."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .distributed_statistics import gather_batch_with_grad, gather_batch_without_grad
from .v45_curriculum import V45Curriculum


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
    prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
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
        prediction["log_scale"].float(),
        target["log_scale"].float(),
        reduction="none",
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


def _tube_loss(
    model, output: dict
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    full = output["full_state"]
    assignment = full["assignment"].float()
    correspondence = output["correspondence"]
    transported = torch.einsum(
        "btsn,btnm->btsm",
        assignment[:, :-1].detach(),
        correspondence.forward.float(),
    )
    target_patches = full["tube_target_patches"][:, 1:].float()
    target_valid = assignment[:, 1:].detach().sum(dim=2).clamp(0.0, 1.0)
    target_weight = transported * target_valid[:, :, None]
    target_mass = target_weight.sum(dim=-1)
    denominator = target_mass + model.config.observation_mass_tau
    target = torch.einsum(
        "btsn,btnd->btsd", target_weight, target_patches
    ) / denominator[..., None]
    prediction = full["tube_prediction"].float()
    distance = 1.0 - F.cosine_similarity(
        prediction, target.detach(), dim=-1, eps=1e-6
    )
    support = target_mass / denominator
    loss = (distance * support).sum() / support.sum().clamp_min(1.0)
    return (
        loss,
        {
            "tube_target_support": support.mean(),
            "tube_correct_distance": loss,
        },
        target.detach(),
        support.detach(),
    )


def _identity_loss(
    model,
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = F.normalize(prediction.float(), dim=-1, eps=1e-6)
    target = F.normalize(target.float(), dim=-1, eps=1e-6)
    gathered_target = gather_batch_without_grad(target)
    batch, steps, count = source.shape[:3]
    candidate = gathered_target.permute(1, 0, 2, 3).reshape(steps, -1, source.shape[-1])
    logits = torch.einsum("btsd,tqd->btsq", source, candidate)
    logits = logits / model.config.identity_temperature
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    labels = rank * batch * count + torch.arange(
        batch * count, device=logits.device
    ).reshape(batch, count)
    labels = labels[:, None].expand(logits.shape[:-1])
    item_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).reshape(labels.shape)
    loss = (item_loss * weight).sum() / weight.sum().clamp_min(1.0)
    accuracy = ((logits.argmax(dim=-1) == labels).float() * weight).sum()
    accuracy = accuracy / weight.sum().clamp_min(1.0)
    return loss, accuracy


def _object_loss(model, output: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    full, masked = output["full_state"], output["masked_state"]
    correspondence = output["correspondence"]
    assignment = full["assignment"].float()
    transported = torch.einsum(
        "btsn,btnm->btsm", assignment[:, :-1], correspondence.forward.float()
    )
    transport = _js_distance(transported, assignment[:, 1:]).mean()
    tube, tube_parts, tube_target, tube_support = _tube_loss(model, output)
    identity, identity_top1 = _identity_loss(
        model,
        full["tube_prediction"],
        tube_target,
        tube_support,
    )
    reconstructed = torch.einsum(
        "btsn,btsd->btnd", assignment, full["decoded_slots"].float()
    )
    reconstruction = (
        1.0
        - F.cosine_similarity(
            reconstructed, full["patch_embedding"].float(), dim=-1, eps=1e-6
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
    masked_state = ((masked_semantic + masked_dynamic) * mask).sum()
    masked_state = masked_state / (masked_count * model.config.object_slots)
    observed_frames = output["observation_mask"][..., None].float()
    lifecycle_count = (
        observed_frames.sum() * model.config.object_slots
    ).clamp_min(1.0)
    lifecycle_target = full["observed_presence"].detach().float()
    lifecycle_prediction = (
        (
            (full["predicted_presence"].float() - lifecycle_target).abs()
            + (full["predicted_visibility"].float() - lifecycle_target).abs()
        )
        * observed_frames
    ).sum() / lifecycle_count
    semantic = F.normalize(full["semantic"].float(), dim=-1, eps=1e-6)
    gram = torch.einsum("btkd,btjd->btkj", semantic, semantic)
    identity_matrix = torch.eye(model.config.object_slots, device=gram.device)[None, None]
    diversity = ((gram - identity_matrix).square() * (1.0 - identity_matrix)).mean()
    total = (
        tube
        + 0.5 * transport
        + 0.25 * identity
        + 0.5 * masked_state
        + 0.1 * lifecycle_prediction
        + 0.05 * diversity
    )
    valid_count = assignment.sum(dim=2).sum(dim=-1, keepdim=True).clamp_min(1.0)
    supported_mass = full["slot_mass"].float() / valid_count
    effective = (supported_mass > 0.01).float().sum(dim=-1).mean()
    result = {
        "loss_object_transport": transport,
        "diagnostic_correspondence_cycle": correspondence.cycle_error.float().mean(),
        "diagnostic_current_reconstruction": reconstruction,
        "loss_object_tube": tube,
        "loss_object_identity": identity,
        "object_identity_top1": identity_top1,
        "loss_object_masked_state": masked_state,
        "loss_object_lifecycle_prediction": lifecycle_prediction,
        "loss_object_diversity": diversity,
        "object_effective_count": effective,
        "object_correction_gate": full["correction_gate"].float().mean(),
        "object_min_correction_gate": full["correction_gate"].float().amin(),
        "object_assignment_entropy": -(
            assignment.clamp_min(1e-7) * assignment.clamp_min(1e-7).log()
        ).sum(dim=2).mean(),
    }
    result.update(tube_parts)
    return total, result


def _intervention_loss(
    correct: torch.Tensor,
    zero: torch.Tensor,
    shuffled: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    return F.relu(margin + correct - zero) + F.relu(margin + correct - shuffled)


def _effect_statistics(effect: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gathered = gather_batch_with_grad(effect.float()).flatten(1)
    standard_deviation = gathered.std(dim=0, unbiased=False)
    variance = F.relu(0.05 - standard_deviation).mean()
    centered = gathered - gathered.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(len(centered), 1)
    off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
    return variance, off_diagonal.square().mean()


def predictive_object_tube_loss(
    model, output: dict, curriculum: V45Curriculum
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
    effect_variance, effect_covariance = _effect_statistics(output["short_effect"])
    effect_loss = (
        short_correct
        + effect_intervention
        + 0.05 * effect_variance
        + 0.01 * effect_covariance
    )
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
    goal_alignment = 1.0 - F.cosine_similarity(
        output["goal_effect"].float().flatten(1),
        output["trajectory_effect"].detach().float().flatten(1),
        dim=-1,
        eps=1e-6,
    ).mean()
    goal_loss = goal_correct + goal_intervention + 0.5 * goal_alignment
    total = (
        curriculum.object_weight * object_loss
        + curriculum.effect_weight * effect_loss
        + curriculum.goal_weight * goal_loss
    )
    effect = output["short_effect"].float()
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
            "effect_variance_penalty": effect_variance,
            "effect_covariance_penalty": effect_covariance,
            "effect_action_std": gather_batch_with_grad(effect).flatten(1).std(
                dim=0, unbiased=False
            ).mean(),
            "effect_action_norm": effect.norm(dim=-1).mean(),
            "effect_action_rms": effect.square().mean().sqrt(),
            "effect_action_max_abs": effect.abs().amax(),
            "goal_correct_distance": goal_correct,
            "goal_zero_distance": goal_zero,
            "goal_shuffled_distance": goal_shuffled,
            "goal_relative_gain_over_zero": (goal_zero - goal_correct)
            / goal_zero.detach().clamp_min(1e-6),
            "goal_effect_alignment": goal_alignment,
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
        raise RuntimeError("v45 objective produced a non-finite loss")
    return total, parts
