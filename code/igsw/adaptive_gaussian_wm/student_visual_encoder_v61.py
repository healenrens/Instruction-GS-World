"""One deployable visual encoder for DINO/SigLIP2 carrier ablations."""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class StudentTokenField:
    features: torch.Tensor
    coordinates: torch.Tensor
    valid: torch.Tensor
    pooled: torch.Tensor
    grid_hw: tuple[int, int]


def _grid_coordinates(height: int, width: int, device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device)
    x = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


class StudentVisualEncoderV61(nn.Module):
    """The only visual encoder retained at deployment time."""

    def __init__(
        self,
        config,
        dino_checkpoint: str,
        siglip2_checkpoint: str,
        frame_batch: int,
    ):
        super().__init__()
        if frame_batch < 1:
            raise ValueError("v61 student frame batch must be positive")
        self.config = config
        self.kind = config.student_encoder
        self.frame_batch = int(frame_batch)
        if self.kind == "dino":
            self._build_dino(dino_checkpoint)
        else:
            self._build_siglip2(siglip2_checkpoint)
        self.projector = nn.Sequential(
            nn.Linear(self.native_dim, config.student_dim),
            nn.LayerNorm(config.student_dim),
        )

    def _build_dino(self, checkpoint: str) -> None:
        if not os.path.isfile(checkpoint):
            raise ValueError(f"v61 DINO checkpoint is missing: {checkpoint}")
        import timm

        self.backbone = timm.create_model(
            self.config.dino_model_name,
            pretrained=True,
            pretrained_cfg_overlay={"file": checkpoint},
            num_classes=0,
            img_size=self.config.dino_image_size,
        )
        self.native_dim = int(self.backbone.embed_dim)
        self.patch_size = int(self.backbone.patch_embed.patch_size[0])
        self.prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        self.image_size = self.config.dino_image_size
        cfg = self.backbone.default_cfg
        self.register_buffer("pixel_mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        self._freeze_lower_blocks(self.backbone.blocks, self.backbone.norm)

    def _build_siglip2(self, checkpoint: str) -> None:
        from transformers import Siglip2VisionModel

        self.backbone = Siglip2VisionModel.from_pretrained(
            checkpoint,
            local_files_only=True,
        )
        self.native_dim = int(self.backbone.config.hidden_size)
        self.patch_size = int(self.backbone.config.patch_size)
        self.image_size = self.config.siglip2_image_size
        self.max_patches = int(self.backbone.config.num_patches)
        self.register_buffer("pixel_mean", torch.full((1, 3, 1, 1), 0.5))
        self.register_buffer("pixel_std", torch.full((1, 3, 1, 1), 0.5))
        self._freeze_lower_blocks(
            self.backbone.encoder.layers,
            self.backbone.post_layernorm,
        )
        self.backbone.gradient_checkpointing_enable()

    def _freeze_lower_blocks(self, blocks, output_norm) -> None:
        self.backbone.requires_grad_(False)
        count = min(self.config.student_trainable_blocks, len(blocks))
        for block in blocks[-count:]:
            block.requires_grad_(True)
        output_norm.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        blocks = (
            self.backbone.blocks
            if self.kind == "dino"
            else self.backbone.encoder.layers
        )
        if mode:
            self.backbone.eval()
            for block in blocks[-self.config.student_trainable_blocks :]:
                block.train()
            if self.kind == "dino":
                self.backbone.norm.train()
            else:
                self.backbone.post_layernorm.train()
        return self

    def _resize(self, rgb: torch.Tensor, valid: torch.Tensor):
        image = F.interpolate(
            rgb.float() / 255.0,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mask = F.interpolate(
            valid[:, None].float(),
            size=(self.image_size, self.image_size),
            mode="nearest",
        )
        return ((image - self.pixel_mean) / self.pixel_std) * mask, mask

    def _dino_frames(self, rgb: torch.Tensor, valid: torch.Tensor):
        image, mask = self._resize(rgb, valid)
        encoded = self.backbone.forward_features(image)
        native = encoded[:, self.prefix_tokens :]
        grid = self.image_size // self.patch_size
        patch_valid = (
            F.avg_pool2d(mask, self.patch_size, self.patch_size).flatten(1) >= 0.5
        )
        coordinates = _grid_coordinates(grid, grid, rgb.device)
        return native, patch_valid, coordinates, (grid, grid)

    def _siglip2_frames(self, rgb: torch.Tensor, valid: torch.Tensor):
        image, mask = self._resize(rgb, valid)
        grid = self.image_size // self.patch_size
        patches = image.reshape(
            len(image), 3, grid, self.patch_size, grid, self.patch_size
        )
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(len(image), grid * grid, -1)
        patch_valid = (
            F.avg_pool2d(mask, self.patch_size, self.patch_size).flatten(1) >= 0.5
        )
        actual = patches.shape[1]
        if actual > self.max_patches:
            raise RuntimeError(
                "SigLIP2 image produces more patches than its checkpoint"
            )
        if actual < self.max_patches:
            patches = F.pad(patches, (0, 0, 0, self.max_patches - actual))
            patch_valid = F.pad(
                patch_valid, (0, self.max_patches - actual), value=False
            )
        spatial_shapes = torch.tensor(
            (grid, grid), device=rgb.device, dtype=torch.long
        )[None].expand(len(rgb), -1)
        output = self.backbone(
            pixel_values=patches,
            pixel_attention_mask=patch_valid,
            spatial_shapes=spatial_shapes,
        )
        coordinates = _grid_coordinates(grid, grid, rgb.device)
        return (
            output.last_hidden_state[:, :actual],
            patch_valid[:, :actual],
            coordinates,
            (grid, grid),
        )

    def _encode_frames(self, rgb: torch.Tensor, valid: torch.Tensor):
        outputs, masks = [], []
        coordinates, grid_hw = None, None
        for start in range(0, len(rgb), self.frame_batch):
            stop = min(start + self.frame_batch, len(rgb))
            if self.kind == "dino":
                native, mask, coordinates, grid_hw = self._dino_frames(
                    rgb[start:stop], valid[start:stop]
                )
            else:
                native, mask, coordinates, grid_hw = self._siglip2_frames(
                    rgb[start:stop], valid[start:stop]
                )
            outputs.append(native)
            masks.append(mask)
        return torch.cat(outputs), torch.cat(masks), coordinates, grid_hw

    def forward(
        self, rgb: torch.Tensor, pixel_valid: torch.Tensor
    ) -> StudentTokenField:
        if rgb.ndim != 5 or rgb.shape[2] != 3 or rgb.dtype != torch.uint8:
            raise ValueError("v61 Student expects [B,T,3,H,W] uint8 RGB")
        if pixel_valid.shape != (rgb.shape[0], rgb.shape[1], *rgb.shape[-2:]):
            raise ValueError("v61 Student pixel-valid shape differs from RGB")
        batch, frames = rgb.shape[:2]
        native, valid, coordinates, grid_hw = self._encode_frames(
            rgb.flatten(0, 1), pixel_valid.flatten(0, 1)
        )
        projected = F.normalize(self.projector(native), dim=-1, eps=1e-6)
        weight = valid.float()
        pooled = (projected * weight[..., None]).sum(dim=1)
        pooled = F.normalize(
            pooled / weight.sum(dim=1, keepdim=True).clamp_min(1.0), dim=-1
        )
        token_count = projected.shape[1]
        return StudentTokenField(
            features=projected.reshape(batch, frames, token_count, -1),
            coordinates=coordinates[None, None].expand(batch, frames, -1, -1),
            valid=valid.reshape(batch, frames, token_count),
            pooled=pooled.reshape(batch, frames, -1),
            grid_hw=grid_hw,
        )
