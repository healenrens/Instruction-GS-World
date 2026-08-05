"""Frozen Wan2.2 causal video VAE features for object-region detail."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .video_vae_contract import (
    VIDEO_VAE_DIFFUSERS_VERSION,
    validate_video_vae_artifact,
)


@dataclass(frozen=True)
class WanVideoFeatures:
    feature: torch.Tensor
    appearance_energy: torch.Tensor
    motion_energy: torch.Tensor


def _content_extent(valid: torch.Tensor) -> tuple[int, int]:
    if valid.ndim != 3 or valid.dtype != torch.bool:
        raise ValueError("video validity must have shape [T,H,W] and be boolean")
    support = valid.any(dim=0)
    rows = support.any(dim=1).nonzero().flatten()
    columns = support.any(dim=0).nonzero().flatten()
    if len(rows) == 0 or len(columns) == 0:
        raise ValueError("video clip contains no valid pixels")
    height = int(rows[-1].item()) + 1
    width = int(columns[-1].item()) + 1
    rectangle = torch.zeros_like(support)
    rectangle[:height, :width] = True
    if not bool((support == rectangle).all()):
        raise ValueError("video validity must describe top-left rectangular content")
    return height, width


def _resize_clip(
    rgb: torch.Tensor,
    valid: torch.Tensor,
    short_side: int,
) -> torch.Tensor:
    height, width = _content_extent(valid)
    scale = short_side / float(min(height, width))
    resized_height = max(16, round(height * scale))
    resized_width = max(16, round(width * scale))
    padded_height = math.ceil(resized_height / 16) * 16
    padded_width = math.ceil(resized_width / 16) * 16
    frames = F.interpolate(
        rgb[:, :, :height, :width].float() / 127.5 - 1.0,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    output = frames.new_zeros((len(frames), 3, padded_height, padded_width))
    output[:, :, :resized_height, :resized_width] = frames
    return output


def _pad_spatial_batch(clips: list[torch.Tensor]) -> torch.Tensor:
    height = max(clip.shape[-2] for clip in clips)
    width = max(clip.shape[-1] for clip in clips)
    batch = clips[0].new_zeros((len(clips), 3, clips[0].shape[0], height, width))
    for index, clip in enumerate(clips):
        batch[index, :, :, : clip.shape[-2], : clip.shape[-1]] = clip.permute(
            1, 0, 2, 3
        )
    return batch


class FrozenWanVideoVAE:
    """External frozen module; weights are deliberately absent from checkpoints."""

    def __init__(
        self,
        model_root: str,
        contract_path: str,
        short_side: int,
        clip_frames: int,
        clip_batch: int,
        output_grid: tuple[int, int],
    ):
        self.contract = validate_video_vae_artifact(
            model_root, contract_path, verify_hashes=False
        )
        if short_side % 16 or clip_frames != 5 or clip_batch < 1:
            raise ValueError("invalid frozen Wan VAE runtime configuration")
        self.model_root = model_root
        self.short_side = short_side
        self.clip_frames = clip_frames
        self.clip_batch = clip_batch
        self.output_grid = output_grid
        self._vae = None
        self._device = None

    def _load(self, device: torch.device):
        if self._vae is not None:
            if self._device != device:
                raise RuntimeError("frozen video VAE cannot migrate between devices")
            return self._vae
        from diffusers import AutoencoderKLWan
        import diffusers

        if diffusers.__version__ != VIDEO_VAE_DIFFUSERS_VERSION:
            raise RuntimeError(
                "diffusers version differs from the pinned v44 runtime: "
                f"{diffusers.__version__}"
            )

        vae = AutoencoderKLWan.from_pretrained(
            self.model_root,
            subfolder="vae",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        vae.requires_grad_(False).eval().to(device)
        vae.enable_tiling()
        config = vae.config
        observed = (
            int(config.z_dim),
            int(config.scale_factor_temporal),
            int(config.scale_factor_spatial),
            int(config.patch_size),
        )
        if observed != (48, 4, 16, 2):
            raise ValueError(f"loaded Wan VAE contract differs: {observed}")
        self._vae = vae
        self._device = device
        return vae

    def _encode(self, video: torch.Tensor) -> torch.Tensor:
        vae = self._load(video.device)
        outputs = []
        for start in range(0, len(video), self.clip_batch):
            clip = video[start : start + self.clip_batch]
            posterior = vae.encode(clip.to(torch.bfloat16)).latent_dist
            outputs.append(posterior.mode())
        latent = torch.cat(outputs).float()
        if latent.shape[1] != 48 or latent.shape[2] != 2:
            raise RuntimeError(f"five-frame Wan VAE output differs: {latent.shape}")
        mean = latent.new_tensor(vae.config.latents_mean).view(1, 48, 1, 1, 1)
        std = latent.new_tensor(vae.config.latents_std).view(1, 48, 1, 1, 1)
        return (latent - mean) / std

    @torch.no_grad()
    def encode_clips(
        self,
        clips: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, WanVideoFeatures]:
        if not clips:
            raise ValueError("video VAE received no clips")
        names = list(clips)
        batch = None
        prepared = []
        for name in names:
            rgb, valid = clips[name]
            if rgb.ndim != 5 or rgb.shape[1:3] != (self.clip_frames, 3):
                raise ValueError(f"{name} RGB must have shape [B,5,3,H,W]")
            if valid.shape != (rgb.shape[0], rgb.shape[1], *rgb.shape[-2:]):
                raise ValueError(f"{name} video validity shape differs")
            batch = rgb.shape[0] if batch is None else batch
            if rgb.shape[0] != batch:
                raise ValueError("video clip batches differ")
            prepared.extend(
                _resize_clip(rgb[index], valid[index], self.short_side)
                for index in range(batch)
            )
        video = _pad_spatial_batch(prepared)
        latent = self._encode(video)
        final = latent[:, :, -1]
        motion = final - latent[:, :, 0]
        feature_map = torch.cat((final, motion), dim=1)
        feature_map = F.interpolate(
            feature_map,
            size=self.output_grid,
            mode="bilinear",
            align_corners=False,
        )
        flat = feature_map.flatten(2).transpose(1, 2).to(torch.bfloat16)
        result = {}
        for offset, name in enumerate(names):
            start = offset * batch
            stop = start + batch
            result[name] = WanVideoFeatures(
                feature=flat[start:stop].clone(),
                appearance_energy=final[start:stop].square().mean().sqrt(),
                motion_energy=motion[start:stop].square().mean().sqrt(),
            )
        return result
