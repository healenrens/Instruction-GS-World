"""RGB preprocessing, micro-Gaussian color aggregation, rendering, and loss."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .decoder import GaussianReadoutState
from .gaussian_math import mahalanobis_squared_from_precision, precision_2d
from .gpstoken import GPSTokenState


def resize_and_pad_rgb(
    frames: torch.Tensor,
    short_side: int,
    pad_multiple: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize [T,H,W,3] uint8 without distortion and return [T,3,Hr,Wr]."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("rgb_path must have shape [T,H,W,3]")
    if short_side < 16 or pad_multiple < 1:
        raise ValueError("invalid RGB resize configuration")
    height, width = frames.shape[1:3]
    scale = short_side / min(height, width)
    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))
    padded_height = math.ceil(resized_height / pad_multiple) * pad_multiple
    padded_width = math.ceil(resized_width / pad_multiple) * pad_multiple
    channels_first = frames.permute(0, 3, 1, 2).float()
    resized = F.interpolate(
        channels_first,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    output = resized.new_zeros(
        frames.shape[0],
        3,
        padded_height,
        padded_width,
    )
    output[:, :, :resized_height, :resized_width] = resized
    valid = torch.zeros(
        frames.shape[0],
        padded_height,
        padded_width,
        dtype=torch.bool,
    )
    valid[:, :resized_height, :resized_width] = True
    return output.round().clamp(0, 255).to(torch.uint8), valid


def micro_rgb_from_assignment(
    state: GPSTokenState,
    rgb: torch.Tensor,
    valid: torch.Tensor,
    grid_height: int,
    grid_width: int,
) -> torch.Tensor:
    """Aggregate current RGB with the learned GPSToken assignment."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("current RGB must have shape [B,3,H,W]")
    if valid.shape != (rgb.shape[0], *rgb.shape[-2:]):
        raise ValueError("current RGB valid mask has the wrong shape")
    content_height = valid.any(dim=-1).sum(dim=-1)
    content_width = valid.any(dim=-2).sum(dim=-1)
    if not bool(((content_height > 0) & (content_width > 0)).all()):
        raise ValueError("current RGB valid mask is empty")
    x = torch.linspace(0.0, 1.0, grid_width, device=rgb.device)
    y = torch.linspace(0.0, 1.0, grid_height, device=rgb.device)
    pixel_x = x[None, None] * (content_width[:, None, None] - 1)
    pixel_y = y[None, :, None] * (content_height[:, None, None] - 1)
    pixel_x = pixel_x.expand(-1, grid_height, -1)
    pixel_y = pixel_y.expand(-1, -1, grid_width)
    sample_grid = torch.stack(
        (
            2.0 * pixel_x / max(rgb.shape[-1] - 1, 1) - 1.0,
            2.0 * pixel_y / max(rgb.shape[-2] - 1, 1) - 1.0,
        ),
        dim=-1,
    )
    pixels = F.grid_sample(
        rgb.float() / 255.0,
        mode="bilinear",
        align_corners=True,
        grid=sample_grid,
    ).permute(0, 2, 3, 1).reshape(rgb.shape[0], -1, 3)
    if pixels.shape[1] != state.assignment.shape[-1]:
        raise ValueError("RGB feature grid and token assignment do not align")
    mass = state.assignment.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    normalized = state.assignment / mass
    return torch.einsum("bmn,bnc->bmc", normalized, pixels)


def current_micro_rgb(
    state: GPSTokenState,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    if "history_rgb" not in batch or "feature_grid_hw" not in batch:
        raise ValueError("RGB supervision requires history_rgb and feature_grid_hw")
    grid_hw = batch["feature_grid_hw"]
    if grid_hw.ndim != 2 or grid_hw.shape[1] != 2:
        raise ValueError("feature_grid_hw must have shape [B,2]")
    if not bool((grid_hw == grid_hw[:1]).all()):
        raise ValueError("feature grid dimensions differ within the batch")
    return micro_rgb_from_assignment(
        state,
        batch["history_rgb"][:, -1],
        batch["history_rgb_valid"][:, -1],
        int(grid_hw[0, 0]),
        int(grid_hw[0, 1]),
    )


def masked_rgb_mean(rgb: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 5 or valid.shape != (rgb.shape[0], rgb.shape[1], *rgb.shape[-2:]):
        raise ValueError("RGB and valid mask shapes do not align")
    value = rgb.float() / 255.0
    weight = valid.to(value.dtype)[:, :, None]
    return (value * weight).sum(dim=(-2, -1)) / weight.sum(
        dim=(-2, -1)
    ).clamp_min(1.0)


def _coordinate_chunk(
    start: int,
    end: int,
    height: int,
    width: int,
    device: torch.device,
    valid: torch.Tensor,
) -> torch.Tensor:
    index = torch.arange(start, end, device=device)
    y = index.div(width, rounding_mode="floor")
    x = index.remainder(width)
    content_height = valid.any(dim=-1).sum(dim=-1).clamp_min(1)
    content_width = valid.any(dim=-2).sum(dim=-1).clamp_min(1)
    x = 2.0 * x.float()[None, None] / (content_width[..., None] - 1).clamp_min(1) - 1.0
    y = 2.0 * y.float()[None, None] / (content_height[..., None] - 1).clamp_min(1) - 1.0
    return torch.stack((x.expand_as(y), y.expand_as(x)), dim=-1)


def render_gaussian_rgb(
    readout: GaussianReadoutState,
    height: int,
    width: int,
    background: torch.Tensor,
    chunk_size: int,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render coverage-aware RGB without materializing all M x H x W weights."""
    if readout.rgb is None:
        raise ValueError("Gaussian readout does not contain RGB")
    if background.shape != (*readout.center.shape[:2], 3):
        raise ValueError("background must have shape [B,Q,3]")
    if valid.shape != (*readout.center.shape[:2], height, width):
        raise ValueError("render valid mask must have shape [B,Q,H,W]")
    opacity = readout.opacity.squeeze(-1).float()
    activation = readout.activation.squeeze(-1).float()
    depth = readout.depth_order.squeeze(-1).float()
    precision = precision_2d(readout.covariance)
    rgb_chunks = []
    coverage_chunks = []
    for start in range(0, height * width, chunk_size):
        end = min(start + chunk_size, height * width)
        coordinates = _coordinate_chunk(
            start,
            end,
            height,
            width,
            readout.center.device,
            valid,
        )
        difference = coordinates[:, :, None] - readout.center[..., None, :].float()
        distance = mahalanobis_squared_from_precision(
            precision,
            difference,
        )
        density = torch.exp(-0.5 * distance)
        density = density * opacity[..., None] * activation[..., None]
        coverage = 1.0 - torch.exp(-density.sum(dim=2))
        mixture = torch.softmax(
            torch.log(density.clamp_min(1e-8)) - depth[..., None],
            dim=2,
        )
        foreground = torch.einsum(
            "bqmn,bqmc->bqnc",
            mixture,
            readout.rgb.float(),
        )
        rendered = (
            coverage[..., None] * foreground
            + (1.0 - coverage[..., None]) * background[:, :, None].float()
        )
        rgb_chunks.append(rendered)
        coverage_chunks.append(coverage)
    rgb = torch.cat(rgb_chunks, dim=2).reshape(
        readout.center.shape[0],
        readout.center.shape[1],
        height,
        width,
        3,
    ).permute(0, 1, 4, 2, 3)
    coverage = torch.cat(coverage_chunks, dim=2).reshape(
        readout.center.shape[0],
        readout.center.shape[1],
        height,
        width,
    )
    return rgb, coverage


def render_future_rgb(
    readout: GaussianReadoutState,
    batch: dict[str, torch.Tensor],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    required = ("history_rgb", "history_rgb_valid")
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"RGB supervision batch is missing {missing}")
    query_count = readout.center.shape[1]
    background = masked_rgb_mean(
        batch["history_rgb"][:, -1:],
        batch["history_rgb_valid"][:, -1:],
    ).expand(-1, query_count, -1)
    valid = batch["history_rgb_valid"][:, -1:].expand(
        -1,
        query_count,
        -1,
        -1,
    )
    return render_gaussian_rgb(
        readout,
        batch["history_rgb"].shape[-2],
        batch["history_rgb"].shape[-1],
        background,
        chunk_size,
        valid,
    )


def render_current_rgb(
    readout: GaussianReadoutState,
    batch: dict[str, torch.Tensor],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    required = ("history_rgb", "history_rgb_valid")
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"RGB supervision batch is missing {missing}")
    query_count = readout.center.shape[1]
    current_rgb = batch["history_rgb"][:, -1:].expand(
        -1, query_count, -1, -1, -1
    )
    current_valid = batch["history_rgb_valid"][:, -1:].expand(
        -1, query_count, -1, -1
    )
    background = masked_rgb_mean(current_rgb, current_valid)
    return render_gaussian_rgb(
        readout,
        current_rgb.shape[-2],
        current_rgb.shape[-1],
        background,
        chunk_size,
        current_valid,
    )


def residual_future_rgb(
    future_render: torch.Tensor,
    current_render: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Use the observed frame as a dense base and Gaussian change as residual."""
    if future_render.shape != current_render.shape:
        raise ValueError("future and current RGB renders must align")
    current = batch["history_rgb"][:, -1:].float() / 255.0
    if current.shape[2:] != future_render.shape[2:]:
        raise ValueError("current RGB and future render must align")
    return (
        current.expand_as(future_render) + future_render - current_render
    ).clamp(0.0, 1.0)


def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def rgb_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    ssim_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction.shape != target.shape:
        raise ValueError("RGB prediction and target shapes differ")
    if valid.shape != (prediction.shape[0], prediction.shape[1], *prediction.shape[-2:]):
        raise ValueError("RGB valid mask shape mismatch")
    target_float = target.float() / 255.0
    error = prediction.float() - target_float
    charbonnier = _masked_mean(
        torch.sqrt(error.square() + 1e-6).mean(dim=2),
        valid,
    )
    flat_prediction = prediction.flatten(0, 1).float()
    flat_target = target_float.flatten(0, 1)
    mu_prediction = F.avg_pool2d(flat_prediction, 3, 1, 1)
    mu_target = F.avg_pool2d(flat_target, 3, 1, 1)
    variance_prediction = F.avg_pool2d(flat_prediction.square(), 3, 1, 1)
    variance_prediction = variance_prediction - mu_prediction.square()
    variance_target = F.avg_pool2d(flat_target.square(), 3, 1, 1)
    variance_target = variance_target - mu_target.square()
    covariance = F.avg_pool2d(flat_prediction * flat_target, 3, 1, 1)
    covariance = covariance - mu_prediction * mu_target
    ssim = (
        (2.0 * mu_prediction * mu_target + 0.01**2)
        * (2.0 * covariance + 0.03**2)
        / (
            (mu_prediction.square() + mu_target.square() + 0.01**2)
            * (variance_prediction + variance_target + 0.03**2)
        ).clamp_min(1e-6)
    ).mean(dim=1).clamp(-1.0, 1.0).reshape_as(valid)
    flat_valid = valid.flatten(0, 1).float()[:, None]
    ssim_valid = F.avg_pool2d(flat_valid, 3, 1, 1)
    ssim_valid = (ssim_valid[:, 0] >= 1.0 - 1e-6).reshape_as(valid)
    ssim_loss = _masked_mean(1.0 - ssim, ssim_valid)
    mse = _masked_mean(error.square().mean(dim=2), valid)
    total = charbonnier + ssim_weight * ssim_loss
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-8))
    return total, {
        "charbonnier": charbonnier,
        "ssim_loss": ssim_loss,
        "mse": mse,
        "psnr_db": psnr,
    }
