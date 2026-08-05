"""Construction boundary for frozen external v44 components."""
from __future__ import annotations

from .wan_video_vae_encoder import FrozenWanVideoVAE


def build_frozen_video_vae(config) -> FrozenWanVideoVAE:
    grid = config.dino_image_size // 14
    return FrozenWanVideoVAE(
        model_root=config.video_vae_model,
        contract_path=config.video_vae_contract,
        short_side=config.video_vae_short_side,
        clip_frames=config.video_vae_clip_frames,
        clip_batch=config.video_vae_batch,
        output_grid=(grid, grid),
    )
