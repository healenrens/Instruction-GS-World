"""Joint geometry, rendering, and action-usage objectives."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .action_field import CorrelatedActionField
from .rendering import render_dino, render_rgb


@dataclass
class ActionLossWeights:
    move: float = 4.0
    deterministic: float = 0.25
    appearance: float = 0.2
    visibility: float = 0.1
    rgb: float = 1.0
    dino: float = 0.2
    effect: float = 0.1
    action_alignment: float = 1.0
    usage: float = 0.1
    usage_margin: float = 0.01
    local: float = 0.1
    alpha: float = 0.01
    slot: float = 0.01


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.float()
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _oriented_effect_target(target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    motion = target[..., :3]
    weight = valid.float()[..., None]
    count = weight.sum(dim=1).clamp_min(1.0)
    mean = (motion * weight).sum(dim=1) / count
    centered = (motion - mean[:, None]) * weight
    std = torch.sqrt(centered.square().sum(dim=1) / count + 1e-6)
    speed = motion[..., :2].norm(dim=-1)
    mean_speed = (speed * valid.float()).sum(dim=1) / valid.float().sum(dim=1).clamp_min(1)
    moving_fraction = ((speed > 0.01) & valid).float().sum(dim=1)
    moving_fraction = moving_fraction / valid.float().sum(dim=1).clamp_min(1)
    effect = torch.cat((mean, std, mean_speed[:, None], moving_fraction[:, None]), dim=-1)
    scale = effect.new_tensor([0.05, 0.05, 0.10, 0.05, 0.05, 0.10, 0.05, 1.0])
    return (effect / scale).clamp(-4.0, 4.0)


def _multiscale_rgb_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.permute(0, 3, 1, 2)
    target = target.permute(0, 3, 1, 2)
    losses = [F.l1_loss(prediction, target)]
    for size in (2, 4):
        losses.append(
            F.l1_loss(
                F.avg_pool2d(prediction, size),
                F.avg_pool2d(target, size),
            )
        )
    return torch.stack(losses).mean()


def _local_field_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    prediction = prediction[..., :3].reshape(len(prediction), rows, cols, 3)
    target = target[..., :3].reshape(len(target), rows, cols, 3)
    valid = valid.reshape(len(valid), rows, cols)
    horizontal = valid[:, :, 1:] & valid[:, :, :-1]
    vertical = valid[:, 1:] & valid[:, :-1]
    pred_dx = prediction[:, :, 1:] - prediction[:, :, :-1]
    target_dx = target[:, :, 1:] - target[:, :, :-1]
    pred_dy = prediction[:, 1:] - prediction[:, :-1]
    target_dy = target[:, 1:] - target[:, :-1]
    horizontal_loss = (pred_dx - target_dx).abs().mean(dim=-1)
    vertical_loss = (pred_dy - target_dy).abs().mean(dim=-1)
    return 0.5 * (
        _masked_mean(horizontal_loss, horizontal)
        + _masked_mean(vertical_loss, vertical)
    )


def _slot_separation(actions: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    distance = torch.cdist(actions, actions)
    count = actions.shape[1]
    off_diagonal = ~torch.eye(count, dtype=torch.bool, device=actions.device)[None]
    return F.relu(margin - distance)[off_diagonal.expand_as(distance)].mean()


def effect_aligned_action_target(
    target: torch.Tensor,
    valid: torch.Tensor,
    visible: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Four canonical spatial action tokens with normalized motion statistics."""
    if rows < 2 or cols < 2:
        raise ValueError("effect-aligned action tokens require at least a 2x2 control grid")
    target = target[..., :3].reshape(len(target), rows, cols, 3)
    valid = valid.reshape(len(valid), rows, cols)
    visible = visible.reshape(len(visible), rows, cols)
    row_slices = (slice(0, rows // 2), slice(rows // 2, rows))
    col_slices = (slice(0, cols // 2), slice(cols // 2, cols))
    scale = target.new_tensor([0.05, 0.05, 0.10])
    tokens = []
    for row_slice in row_slices:
        for col_slice in col_slices:
            region = target[:, row_slice, col_slice]
            region_valid = valid[:, row_slice, col_slice]
            weight = region_valid.float()[..., None]
            count = weight.sum(dim=(1, 2)).clamp_min(1.0)
            mean = (region * weight).sum(dim=(1, 2)) / count
            centered = (region - mean[:, None, None]) * weight
            std = torch.sqrt(centered.square().sum(dim=(1, 2)) / count + 1e-6)
            moving = (region[..., :2].norm(dim=-1) > 0.01) & region_valid
            moving_fraction = moving.float().sum(dim=(1, 2))
            moving_fraction = moving_fraction / region_valid.float().sum(dim=(1, 2)).clamp_min(1)
            region_visible = visible[:, row_slice, col_slice]
            visible_fraction = region_visible.float().sum(dim=(1, 2))
            visible_fraction = visible_fraction / region_visible[0].numel()
            token = torch.cat(
                (
                    mean / scale,
                    std / scale,
                    moving_fraction[:, None] * 2.0 - 1.0,
                    visible_fraction[:, None] * 2.0 - 1.0,
                ),
                dim=-1,
            ).clamp(-1.0, 1.0)
            tokens.append(token)
    return torch.stack(tokens, dim=1)


def posterior_joint_loss(
    model: CorrelatedActionField,
    batch: dict,
    output: dict,
    weights: ActionLossWeights,
    render_height: int,
    render_width: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    field = output["control_field"]
    valid = batch["motion_valid"]
    target = batch["target"]
    movers = (target[..., :2].norm(dim=-1) > 0.01) & valid
    geometry_per = F.smooth_l1_loss(
        field[..., :3],
        target[..., :3],
        beta=0.01,
        reduction="none",
    ).mean(dim=-1)
    geometry_weight = 1.0 + weights.move * movers.float()
    geometry = _masked_mean(geometry_per * geometry_weight, valid)
    deterministic_per = F.smooth_l1_loss(
        output["deterministic_control_field"][..., :3],
        target[..., :3],
        beta=0.01,
        reduction="none",
    ).mean(dim=-1)
    deterministic = _masked_mean(
        deterministic_per * geometry_weight,
        valid,
    )

    control_active = batch["state"][..., -1] > 0.5
    appearance_prediction = 0.5 * torch.tanh(field[..., 3:6])
    appearance_per = F.smooth_l1_loss(
        appearance_prediction,
        target[..., 3:6],
        beta=0.05,
        reduction="none",
    ).mean(dim=-1)
    appearance = _masked_mean(
        appearance_per,
        batch["visible"] & control_active,
    )
    visibility = _masked_mean(
        F.binary_cross_entropy_with_logits(
            output["control_visibility_logits"],
            batch["visible"].float(),
            reduction="none",
        ),
        batch["matched"],
    )

    rendered_rgb, rgb_alpha = render_rgb(
        batch,
        output["dense_field"],
        render_height,
        render_width,
    )
    target_rgb = F.interpolate(
        batch["rgb1"],
        size=(render_height, render_width),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    rgb = _multiscale_rgb_loss(rendered_rgb, target_rgb)
    alpha = F.relu(0.95 - rgb_alpha).mean()

    dino = field.sum() * 0.0
    if model.config.dino_dim:
        rendered_dino, _ = render_dino(batch, output["dense_field"], model.config.dino_dim)
        rendered_dino = F.normalize(rendered_dino, dim=-1)
        target_dino = F.normalize(batch["dino1"], dim=-1)
        dino = (1.0 - (rendered_dino * target_dino).sum(dim=-1)).mean()

    effect = F.smooth_l1_loss(
        output["effect_prediction"],
        _oriented_effect_target(target, valid),
        beta=0.1,
    )
    action_target = effect_aligned_action_target(
        target,
        valid,
        batch["visible"],
        model.config.control_rows,
        model.config.control_cols,
    )
    if output["actions"].shape[1] != 4 or output["actions"].shape[2] < 8:
        raise ValueError("effect-aligned actions require four tokens with at least 8D")
    action_alignment = F.smooth_l1_loss(
        output["actions"][..., :8],
        action_target,
        beta=0.1,
    )
    usage = field.sum() * 0.0
    if len(field) > 1:
        order = torch.roll(torch.arange(len(field), device=field.device), shifts=1)
        shuffled = model.decode_actions(output, output["actions"][order])["control_field"]
        correct_error = (field[..., :3] - target[..., :3]).norm(dim=-1)
        shuffled_error = (shuffled[..., :3] - target[..., :3]).norm(dim=-1)
        usage = _masked_mean(
            F.relu(weights.usage_margin + correct_error - shuffled_error),
            movers,
        )
    local = _local_field_loss(
        field,
        target,
        valid,
        model.config.control_rows,
        model.config.control_cols,
    )
    slot = _slot_separation(output["actions"])
    total = (
        geometry
        + weights.deterministic * deterministic
        + weights.appearance * appearance
        + weights.visibility * visibility
        + weights.rgb * rgb
        + weights.dino * dino
        + weights.effect * effect
        + weights.action_alignment * action_alignment
        + weights.usage * usage
        + weights.local * local
        + weights.alpha * alpha
        + weights.slot * slot
    )
    parts = {
        "loss": total.detach(),
        "geometry": geometry.detach(),
        "deterministic": deterministic.detach(),
        "appearance": appearance.detach(),
        "visibility": visibility.detach(),
        "rgb": rgb.detach(),
        "dino": dino.detach(),
        "effect": effect.detach(),
        "action_alignment": action_alignment.detach(),
        "usage": usage.detach(),
        "local": local.detach(),
        "alpha": alpha.detach(),
        "slot": slot.detach(),
        "mover_fraction": movers.float().mean().detach(),
        "valid_fraction": valid.float().mean().detach(),
        "jump_reject_fraction": (
            ((batch["tracker_image_jump"] >= 0.12) | (batch["tracker_depth_jump"] >= 0.35))
            & batch["matched"]
        ).float().mean().detach(),
    }
    return total, parts
