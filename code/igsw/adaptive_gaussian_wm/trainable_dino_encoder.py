"""Model-owned DINOv2 encoder with a strict online/EMA target boundary."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AdaptiveGaussianWMConfig


@dataclass
class DinoRegionFeatures:
    native: torch.Tensor
    projected: torch.Tensor
    coordinates: torch.Tensor
    valid: torch.Tensor
    grid_hw: tuple[int, int]


class TrainableDinoRegionEncoder(nn.Module):
    """DINOv2-L with frozen lower blocks and trainable upper blocks/projector."""

    def __init__(self, config: AdaptiveGaussianWMConfig, frame_batch: int = 16):
        super().__init__()
        if frame_batch < 1:
            raise ValueError("DINO frame batch must be positive")
        import timm

        self.backbone = timm.create_model(
            config.dino_model_name,
            pretrained=True,
            num_classes=0,
            img_size=config.dino_image_size,
        )
        self.image_size = config.dino_image_size
        self.frame_batch = int(frame_batch)
        self.trainable_blocks = config.dino_trainable_blocks
        self.patch_size = int(self.backbone.patch_embed.patch_size[0])
        self.prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        self.native_dim = int(self.backbone.embed_dim)
        if self.native_dim != config.feature_dim:
            raise ValueError(
                f"DINO native dimension {self.native_dim} != {config.feature_dim}"
            )
        if len(self.backbone.blocks) != 24:
            raise ValueError("v43 expects a 24-block DINOv2-L backbone")
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for block in self.backbone.blocks[-config.dino_trainable_blocks :]:
            block.requires_grad_(True)
        self.backbone.norm.requires_grad_(True)
        self.projector = nn.Sequential(
            nn.LayerNorm(self.native_dim),
            nn.Linear(self.native_dim, config.dino_projector_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.dino_projector_dim, config.region_dim),
        )
        cfg = self.backbone.default_cfg
        self.register_buffer(
            "pixel_mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.tensor(cfg["std"]).view(1, 3, 1, 1)
        )
        grid = self.image_size // self.patch_size
        axis = torch.linspace(-1.0, 1.0, grid)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer(
            "patch_coordinates",
            torch.stack((x, y), dim=-1).reshape(-1, 2),
            persistent=False,
        )

    def freeze_as_target(self) -> None:
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.backbone.eval()
            for block in self.backbone.blocks[-self.trainable_blocks :]:
                block.train()
            self.backbone.norm.train()
        return self

    def _preprocess(
        self,
        rgb: torch.Tensor,
        pixel_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3 or rgb.dtype != torch.uint8:
            raise ValueError("DINO RGB must have shape [F,3,H,W] uint8")
        if pixel_valid.shape != (rgb.shape[0], rgb.shape[2], rgb.shape[3]):
            raise ValueError("DINO pixel validity must have shape [F,H,W]")
        if pixel_valid.dtype != torch.bool:
            raise ValueError("DINO pixel validity must be boolean")
        images = F.interpolate(
            rgb.float() / 255.0,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        resized_valid = F.interpolate(
            pixel_valid[:, None].float(),
            size=(self.image_size, self.image_size),
            mode="nearest",
        )
        normalized = (images - self.pixel_mean) / self.pixel_std
        return normalized * resized_valid, resized_valid

    def _encode_frames(
        self,
        rgb: torch.Tensor,
        pixel_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        native_chunks = []
        projected_chunks = []
        valid_chunks = []
        for start in range(0, rgb.shape[0], self.frame_batch):
            stop = start + self.frame_batch
            images, resized_valid = self._preprocess(
                rgb[start:stop], pixel_valid[start:stop]
            )
            features = self.backbone.forward_features(images)
            if features.ndim != 3:
                raise RuntimeError("DINO forward_features must return [F,N,C]")
            patches = features[:, self.prefix_tokens :]
            native_chunks.append(patches)
            projected_chunks.append(self.projector(patches))
            valid_chunks.append(
                F.avg_pool2d(
                    resized_valid,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                ).flatten(1)
                >= 0.5
            )
        return (
            torch.cat(native_chunks),
            torch.cat(projected_chunks),
            torch.cat(valid_chunks),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        pixel_valid: torch.Tensor,
    ) -> DinoRegionFeatures:
        if rgb.ndim != 5 or rgb.shape[2] != 3:
            raise ValueError("DINO sequence RGB must have shape [B,T,3,H,W]")
        if pixel_valid.shape != (rgb.shape[0], rgb.shape[1], *rgb.shape[-2:]):
            raise ValueError("DINO sequence validity must have shape [B,T,H,W]")
        batch, frames = rgb.shape[:2]
        flat = rgb.reshape(batch * frames, *rgb.shape[2:])
        flat_valid = pixel_valid.reshape(batch * frames, *pixel_valid.shape[2:])
        native, projected, valid = self._encode_frames(flat, flat_valid)
        token_count = native.shape[1]
        expected = len(self.patch_coordinates)
        if token_count != expected:
            raise RuntimeError(f"DINO patch count {token_count} != {expected}")
        if valid.shape != (batch * frames, token_count):
            raise RuntimeError("DINO patch validity differs from feature grid")
        if not bool(valid.any(dim=1).all()):
            raise ValueError("each DINO frame must contain a valid image patch")
        coordinates = self.patch_coordinates.to(native.dtype)[None, None].expand(
            batch, frames, -1, -1
        )
        grid = self.image_size // self.patch_size
        return DinoRegionFeatures(
            native=native.reshape(batch, frames, token_count, -1),
            projected=projected.reshape(batch, frames, token_count, -1),
            coordinates=coordinates,
            valid=valid.reshape(batch, frames, token_count),
            grid_hw=(grid, grid),
        )
