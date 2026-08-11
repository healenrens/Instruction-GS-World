"""Frozen standard DINOv2 patch extraction for v44 raw-video training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .v44_config import TemporalObjectSetConfig


@dataclass(frozen=True)
class FrozenVideoFeatures:
    patches: torch.Tensor
    coordinates: torch.Tensor
    valid: torch.Tensor
    grid_hw: tuple[int, int]


class FrozenDinoVideoRuntime:
    """Own the immutable pretrained encoder outside model/checkpoint state."""

    def __init__(
        self,
        config: TemporalObjectSetConfig,
        device: torch.device,
        amp: str,
        frame_batch: int,
    ):
        if device.type != "cuda":
            raise ValueError("v44 DINO extraction requires CUDA")
        if frame_batch < 1:
            raise ValueError("DINO frame batch must be positive")
        import timm

        self.device = device
        self.dtype = torch.bfloat16 if amp == "bf16" else torch.float32
        self.frame_batch = int(frame_batch)
        self.backbone = (
            timm.create_model(
                config.dino_model_name,
                pretrained=True,
                num_classes=0,
                img_size=config.dino_image_size,
            )
            .to(device=device, dtype=self.dtype)
            .eval()
        )
        self.backbone.requires_grad_(False)
        self.patch_size = int(self.backbone.patch_embed.patch_size[0])
        self.prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        self.feature_dim = int(self.backbone.embed_dim)
        if self.feature_dim != config.patch_dim:
            raise ValueError("standard DINO feature dimension differs from v44 config")
        cfg = self.backbone.default_cfg
        self.mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
        grid = config.dino_image_size // self.patch_size
        axis = torch.linspace(-1.0, 1.0, grid, device=device)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.coordinates = torch.stack((x, y), dim=-1).reshape(-1, 2)
        self.grid_hw = (grid, grid)
        self.image_size = config.dino_image_size

    @torch.no_grad()
    def __call__(self, batch: dict[str, torch.Tensor]) -> FrozenVideoFeatures:
        rgb = batch["video_rgb"]
        pixel_valid = batch["video_pixel_valid"]
        if rgb.ndim != 5 or rgb.shape[2] != 3 or rgb.dtype != torch.uint8:
            raise ValueError("v44 video RGB must have shape [B,T,3,H,W] uint8")
        if pixel_valid.shape != (rgb.shape[0], rgb.shape[1], *rgb.shape[-2:]):
            raise ValueError("v44 pixel validity shape differs from RGB")
        batch_size, frames = rgb.shape[:2]
        flat = rgb.reshape(batch_size * frames, *rgb.shape[2:])
        flat_valid = pixel_valid.reshape(batch_size * frames, *pixel_valid.shape[2:])
        outputs, valid_outputs = [], []
        for start in range(0, len(flat), self.frame_batch):
            images = F.interpolate(
                flat[start : start + self.frame_batch].float() / 255.0,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            validity = F.interpolate(
                flat_valid[start : start + self.frame_batch, None].float(),
                size=(self.image_size, self.image_size),
                mode="nearest",
            )
            images = ((images - self.mean) / self.std) * validity
            encoded = self.backbone.forward_features(images.to(self.dtype))
            patches = encoded[:, self.prefix_tokens :].float()
            patches = F.normalize(patches, dim=-1, eps=1e-6)
            outputs.append(patches.to(self.dtype).clone())
            valid_outputs.append(
                F.avg_pool2d(validity, self.patch_size, self.patch_size).flatten(1)
                >= 0.5
            )
        patches = torch.cat(outputs).reshape(batch_size, frames, -1, self.feature_dim)
        valid = torch.cat(valid_outputs).reshape(batch_size, frames, -1)
        if patches.shape[2] != len(self.coordinates):
            raise RuntimeError("DINO patch count differs from v44 coordinate grid")
        return FrozenVideoFeatures(
            patches=patches,
            coordinates=self.coordinates[None, None].expand(batch_size, frames, -1, -1),
            valid=valid,
            grid_hw=self.grid_hw,
        )
