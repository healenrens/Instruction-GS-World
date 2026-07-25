"""Differentiable RGB and DINO rendering for the correlated action field."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from igsw.causal_geometry import unproject_uv
from igsw.gaussians import GaussianSet, render_gaussianset


def _scaled_intrinsics(
    intrinsics: torch.Tensor,
    source_hw: torch.Tensor,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    source_height, source_width = source_hw.tolist()
    scaled = intrinsics.clone()
    scaled[0] *= target_width / source_width
    scaled[1] *= target_height / source_height
    return scaled


def _sample_feature_grid(
    feature_grid: torch.Tensor,
    uv: torch.Tensor,
    image_hw: torch.Tensor,
) -> torch.Tensor:
    height, width = image_hw.tolist()
    sample_grid = torch.stack(
        (
            (uv[:, 0] + 0.5) / width * 2.0 - 1.0,
            (uv[:, 1] + 0.5) / height * 2.0 - 1.0,
        ),
        dim=-1,
    )[None, None]
    feature_chw = feature_grid.permute(2, 0, 1)[None]
    sampled = F.grid_sample(
        feature_chw,
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled[0, :, 0].T.contiguous()


def deformed_gaussian_attributes(
    batch: dict,
    batch_index: int,
    dense_field: torch.Tensor,
) -> dict[str, torch.Tensor]:
    means = batch["dense_means"][batch_index]
    uv = batch["dense_uv"][batch_index]
    intrinsics = batch["intrinsics"][batch_index]
    image_hw = batch["image_hw"][batch_index]
    height, width = image_hw.tolist()
    active = batch["dense_active"][batch_index].float()[:, None]

    target_uv = uv + dense_field[:, :2] * dense_field.new_tensor([width, height])
    target_depth = means[:, 2].clamp_min(1e-5) * dense_field[:, 2].clamp(-1.5, 1.5).exp()
    target_means = unproject_uv(target_uv, target_depth, intrinsics)

    color_delta = 0.5 * torch.tanh(dense_field[:, 3:6])
    colors = (batch["dense_color"][batch_index] + active * color_delta).clamp(0.0, 1.0)
    scale_xy = (active * 0.35 * torch.tanh(dense_field[:, 6:8])).exp()
    base_scales = batch["dense_scales"][batch_index]
    scales = torch.cat(
        (
            base_scales[:, :2] * scale_xy,
            base_scales[:, 2:3] * scale_xy.mean(dim=-1, keepdim=True),
        ),
        dim=-1,
    )
    base_logit = dense_field.new_tensor(4.59511985013459)
    opacities = torch.sigmoid(base_logit + active[:, 0] * dense_field[:, 8])
    angle = active[:, 0] * 0.5 * torch.tanh(dense_field[:, 9])
    zeros = torch.zeros_like(angle)
    quaternions = torch.stack(
        (torch.cos(angle * 0.5), zeros, zeros, torch.sin(angle * 0.5)),
        dim=-1,
    )
    return {
        "means": target_means,
        "quats": quaternions,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
    }


def _render_values(
    attributes: dict[str, torch.Tensor],
    values: torch.Tensor,
    viewmat: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gaussians = GaussianSet(
        attributes["means"],
        attributes["quats"],
        attributes["scales"],
        attributes["opacities"],
        values,
    )
    colors, alphas, _ = render_gaussianset(
        gaussians,
        viewmat,
        intrinsics,
        width,
        height,
    )
    return colors[0], alphas[0]


def render_rgb(
    batch: dict,
    dense_field: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = []
    alphas = []
    for index in range(len(dense_field)):
        attributes = deformed_gaussian_attributes(batch, index, dense_field[index])
        intrinsics = _scaled_intrinsics(
            batch["intrinsics"][index],
            batch["image_hw"][index],
            height,
            width,
        )
        color, alpha = _render_values(
            attributes,
            attributes["colors"],
            batch["viewmat"][index],
            intrinsics,
            width,
            height,
        )
        rendered.append(color)
        alphas.append(alpha)
    return torch.stack(rendered), torch.stack(alphas)


def render_dino(
    batch: dict,
    dense_field: torch.Tensor,
    dino_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "dino0" not in batch or "dino1" not in batch:
        raise ValueError("DINO rendering requires cached dino0 and dino1 grids")
    if dense_field.shape[-1] != 10 + dino_dim:
        raise ValueError("dense field and DINO dimensions disagree")
    target_height, target_width = batch["dino1"].shape[1:3]
    rendered = []
    alphas = []
    for index in range(len(dense_field)):
        attributes = deformed_gaussian_attributes(batch, index, dense_field[index])
        current_features = _sample_feature_grid(
            batch["dino0"][index],
            batch["dense_uv"][index],
            batch["image_hw"][index],
        )
        active = batch["dense_active"][index].float()[:, None]
        feature_delta = active * 0.5 * torch.tanh(dense_field[index, :, 10:])
        values = F.normalize(current_features + feature_delta, dim=-1)
        intrinsics = _scaled_intrinsics(
            batch["intrinsics"][index],
            batch["image_hw"][index],
            target_height,
            target_width,
        )
        feature, alpha = _render_values(
            attributes,
            values,
            batch["viewmat"][index],
            intrinsics,
            target_width,
            target_height,
        )
        rendered.append(feature)
        alphas.append(alpha)
    return torch.stack(rendered), torch.stack(alphas)
