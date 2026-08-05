"""Frozen per-rank DINO extraction for RGB-only episode batches."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from igsw.gpstoken_wm.dino_features import DinoFeatures

from .rgb_episode_cache_contract import (
    JIT_DINO_FEATURE_CONTRACT,
    JIT_DINO_FEATURE_DIM,
    JIT_DINO_IMAGE_SIZE,
    JIT_DINO_MODEL,
)


class JitDinoFeatureRuntime:
    """Inject frozen dense DINO features without joining model state or optimizer."""

    def __init__(
        self,
        device: torch.device,
        amp: str,
        frame_batch: int,
        goal_stability_threshold: float = 0.05,
    ):
        if device.type != "cuda":
            raise ValueError("JIT DINO training requires CUDA")
        if frame_batch < 1:
            raise ValueError("JIT DINO frame batch must be positive")
        if goal_stability_threshold <= 0.0:
            raise ValueError("goal stability threshold must be positive")
        self.device = device
        self.dtype = torch.bfloat16 if amp == "bf16" else torch.float32
        self.frame_batch = frame_batch
        self.goal_stability_threshold = float(goal_stability_threshold)
        self.extractor = DinoFeatures(JIT_DINO_MODEL, JIT_DINO_IMAGE_SIZE).to(device)
        self.extractor = self.extractor.to(dtype=self.dtype).eval()
        if any(parameter.requires_grad for parameter in self.extractor.parameters()):
            raise RuntimeError("JIT DINO extractor contains trainable parameters")
        self.grid_height = JIT_DINO_IMAGE_SIZE // self.extractor.patch
        self.grid_width = self.grid_height
        if self.extractor.embed_dim != JIT_DINO_FEATURE_DIM:
            raise ValueError(
                "JIT DINO feature dimension differs: "
                f"{self.extractor.embed_dim} != {JIT_DINO_FEATURE_DIM}"
            )
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.grid_height, device=device),
            torch.linspace(-1.0, 1.0, self.grid_width, device=device),
            indexing="ij",
        )
        self.coordinates = torch.stack((x, y), dim=-1).reshape(-1, 2)

    @torch.no_grad()
    def _encode(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 5 or rgb.shape[2] != 3 or rgb.dtype != torch.uint8:
            raise ValueError("JIT DINO RGB must have shape [B,T,3,H,W] uint8")
        batch, frames = rgb.shape[:2]
        images = rgb.permute(0, 1, 3, 4, 2).reshape(
            batch * frames, rgb.shape[-2], rgb.shape[-1], 3
        )
        grids = []
        for start in range(0, len(images), self.frame_batch):
            grid, shape = self.extractor.grid_batch(
                images[start : start + self.frame_batch]
            )
            if shape != (self.grid_height, self.grid_width):
                raise RuntimeError(f"JIT DINO grid differs: {shape}")
            grids.append(F.normalize(grid.float(), dim=-1).to(self.dtype))
        normalized = torch.cat(grids)
        normalized = F.layer_norm(normalized.float(), (normalized.shape[-1],)).to(
            self.dtype
        )
        return normalized.reshape(batch, frames, -1, normalized.shape[-1])

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = {"history_jit_rgb", "future_jit_rgb"}
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"JIT DINO batch is missing: {sorted(missing)}")
        history_rgb = batch.pop("history_jit_rgb")
        future_rgb = batch.pop("future_jit_rgb")
        history = self._encode(history_rgb)
        future = self._encode(future_rgb)
        batch_size, history_count, token_count = history.shape[:3]
        future_count = future.shape[1]
        valid = torch.ones(
            batch_size, token_count, dtype=torch.bool, device=self.device
        )
        batch.update(
            history_features=history,
            future_features=future,
            history_coordinates=self.coordinates[None, None].expand(
                batch_size, history_count, -1, -1
            ),
            future_coordinates=self.coordinates[None, None].expand(
                batch_size, future_count, -1, -1
            ),
            history_valid=valid[:, None].expand(-1, history_count, -1),
            future_valid=valid[:, None].expand(-1, future_count, -1),
            feature_grid_hw=torch.tensor(
                [self.grid_height, self.grid_width],
                device=self.device,
                dtype=torch.long,
            )[None].expand(batch_size, -1),
        )
        if "goal_probe_jit_rgb" in batch:
            if future_count != 2:
                raise ValueError("goal stability probes require short and goal targets")
            probe = self._encode(batch.pop("goal_probe_jit_rgb"))
            trajectory = torch.cat((probe, future[:, 1:2]), dim=1).float()
            stability = (
                1.0
                - F.cosine_similarity(
                    trajectory[:, 1:],
                    trajectory[:, :-1],
                    dim=-1,
                )
            ).mean(dim=(-1, -2))
            goal_pixels = future_rgb[:, 1].float()
            goal_rgb_mean = goal_pixels.mean(dim=(1, 2, 3))
            goal_rgb_std = goal_pixels.std(dim=(1, 2, 3), unbiased=False)
            goal_content_valid = (
                (goal_rgb_std > 2.0)
                & (goal_rgb_mean > 2.0)
                & (goal_rgb_mean < 253.0)
            )
            goal_valid = (
                stability <= self.goal_stability_threshold
            ) & goal_content_valid
            horizon_valid = batch.get("future_horizon_valid")
            if horizon_valid is None or horizon_valid.shape != (batch_size, 2):
                raise ValueError("dynamic targets require [B,2] horizon validity")
            horizon_valid = horizon_valid.clone()
            horizon_valid[:, 1] &= goal_valid
            batch.update(
                future_horizon_valid=horizon_valid,
                goal_stability_error=stability,
                goal_rgb_mean=goal_rgb_mean,
                goal_rgb_std=goal_rgb_std,
                goal_content_valid=goal_content_valid,
                goal_valid=goal_valid,
            )
        return batch


def build_feature_runtime(args, dataset, device: torch.device):
    if getattr(args, "architecture", "") == "object_region_memory_v1":
        return None
    source = getattr(args, "feature_source", "cached")
    if source == "cached":
        return None
    if source != "jit":
        raise ValueError(f"unknown feature source: {source}")
    if dataset.feature_contract != JIT_DINO_FEATURE_CONTRACT:
        raise ValueError("JIT feature source and dataset contract differ")
    return JitDinoFeatureRuntime(
        device,
        args.amp,
        args.jit_dino_batch,
        args.goal_stability_threshold,
    )


def prepare_feature_batch(runtime, batch: dict[str, torch.Tensor]):
    return runtime(batch) if runtime is not None else batch
