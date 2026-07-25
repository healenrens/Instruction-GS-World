"""Frozen DINOv2 dense patch features (PLAN §7, direction A) — replaces the frozen Qwen image-patch
grid as the TOKEN feature source. DINOv2's self-supervised dense features are far stronger spatially
than the Qwen VLM patches sampled at token uv, which is the suspected lever for the moderate ~0.6 dcos.
Qwen STILL does the language conditioning (encode_cond); only the per-token VISUAL feature changes."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoFeatures(nn.Module):
    def __init__(self, model_name: str = "vit_large_patch14_dinov2.lvd142m", img_size: int = 518):
        super().__init__()
        import timm
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=img_size).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_dim = int(self.model.embed_dim)
        self.img_size = img_size
        self.patch = int(self.model.patch_embed.patch_size[0])
        self.n_prefix = int(getattr(self.model, "num_prefix_tokens", 1))
        cfg = self.model.default_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))

    @torch.no_grad()
    def grid_batch(self, rgb_uint8: torch.Tensor):
        """[B,H,W,3] uint8 RGB -> frozen dense grids [B,gh,gw,C]."""
        if rgb_uint8.ndim != 4 or rgb_uint8.shape[-1] != 3:
            raise ValueError("RGB batch must have shape [B,H,W,3]")
        x = rgb_uint8.float().permute(0, 3, 1, 2) / 255.0
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        x = (x.to(self.mean.device, self.mean.dtype) - self.mean) / self.std
        feats = self.model.forward_features(x)
        patches = feats[:, self.n_prefix:, :]
        g = self.img_size // self.patch
        return patches.reshape(len(x), g, g, -1).float(), (g, g)

    @torch.no_grad()
    def grid(self, rgb_uint8: np.ndarray):
        """[H,W,3] uint8 RGB -> dense patch-feature grid [gh,gw,C] + (gh,gw). Frozen, no grad."""
        batch = torch.from_numpy(rgb_uint8)[None]
        grid, shape = self.grid_batch(batch)
        return grid[0], shape
