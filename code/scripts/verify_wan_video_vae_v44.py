#!/usr/bin/env python3
"""Create the pinned Wan2.2 VAE artifact contract after a real GPU probe."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.dual_encoder_temporal_dataset import (  # noqa: E402
    DualEncoderDynamicEpisodeDataset,
)
from igsw.adaptive_gaussian_wm.video_vae_contract import (  # noqa: E402
    VIDEO_VAE_CONTRACT,
    VIDEO_VAE_DIFFUSERS_VERSION,
    VIDEO_VAE_MODEL_ID,
    file_sha256,
)
from igsw.adaptive_gaussian_wm.wan_video_vae_encoder import (  # noqa: E402
    _pad_spatial_batch,
    _resize_clip,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    for name in ("model", "data", "output", "manifest"):
        require(os.path.isabs(getattr(args, name)), f"--{name} must be absolute")
    require(bool(re.fullmatch(r"[0-9a-f]{40}", args.revision)),
            "--revision must be a pinned 40-char commit")
    return args


def model_files(root: str) -> dict[str, str]:
    vae_root = os.path.join(root, "vae")
    require(os.path.isfile(os.path.join(vae_root, "config.json")),
            "Wan VAE config is missing")
    paths = []
    for directory, _, filenames in os.walk(vae_root):
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            paths.append(os.path.join(directory, filename))
    require(paths, "Wan VAE directory is empty")
    return {
        os.path.relpath(path, root): file_sha256(path)
        for path in sorted(paths)
    }


def real_clip(data: str) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = DualEncoderDynamicEpisodeDataset(
        data,
        "train",
        max_items=16,
        feature_source="jit",
        video_clip_frames=5,
    )
    sample = dataset[(0, 4)]
    return sample["short_video_rgb"], sample["short_video_valid"]


@torch.no_grad()
def gpu_probe(
    root: str,
    rgb: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, object]:
    require(torch.cuda.is_available(), "Wan VAE probe requires CUDA")
    import diffusers
    from diffusers import AutoencoderKLWan

    require(
        diffusers.__version__ == VIDEO_VAE_DIFFUSERS_VERSION,
        "Wan VAE probe requires the pinned diffusers runtime",
    )

    device = torch.device("cuda:0")
    vae = AutoencoderKLWan.from_pretrained(
        root,
        subfolder="vae",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).requires_grad_(False).eval().to(device)
    vae.enable_tiling()
    observed = (
        int(vae.config.z_dim),
        int(vae.config.scale_factor_temporal),
        int(vae.config.scale_factor_spatial),
        int(vae.config.patch_size),
    )
    require(observed == (48, 4, 16, 2), f"Wan VAE config differs: {observed}")
    clip = _pad_spatial_batch([_resize_clip(rgb, valid, 256)]).to(device)
    perturbed_rgb = rgb.clone()
    perturbed_rgb[-1] = perturbed_rgb[-1].flip(-1)
    perturbed = _pad_spatial_batch(
        [_resize_clip(perturbed_rgb, valid, 256)]
    ).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    latent = vae.encode(clip.to(torch.bfloat16)).latent_dist.mode()
    changed = vae.encode(perturbed.to(torch.bfloat16)).latent_dist.mode()
    decoded = vae.decode(latent).sample
    torch.cuda.synchronize(device)
    require(latent.shape[:3] == (1, 48, 2),
            f"Wan VAE five-frame latent shape differs: {latent.shape}")
    require(decoded.shape[:3] == (1, 3, 5),
            f"Wan VAE decoded clip shape differs: {decoded.shape}")
    require(bool(torch.isfinite(latent).all()) and bool(torch.isfinite(decoded).all()),
            "Wan VAE probe produced non-finite values")
    sensitivity = float((latent.float() - changed.float()).abs().mean())
    require(sensitivity > 1e-6, "Wan VAE latent is insensitive to video content")
    return {
        "probe_status": "passed",
        "probe_clip_frames": 5,
        "probe_input_shape": list(clip.shape),
        "probe_latent_shape": list(latent.shape),
        "probe_decoded_shape": list(decoded.shape),
        "probe_input_sensitivity_l1": sensitivity,
        "probe_peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "diffusers_version": diffusers.__version__,
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(device),
    }


def atomic_json(path: str, value: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    require(os.path.isdir(args.model), "Wan VAE model root is missing")
    files = model_files(args.model)
    rgb, valid = real_clip(args.data)
    probe = gpu_probe(args.model, rgb, valid)
    contract = {
        "contract": VIDEO_VAE_CONTRACT,
        "model_id": VIDEO_VAE_MODEL_ID,
        "revision": args.revision,
        "subfolder": "vae",
        "class_name": "AutoencoderKLWan",
        "latent_dim": 48,
        "temporal_compression": 4,
        "spatial_compression": 16,
        "patch_size": 2,
        "files": files,
        **probe,
    }
    atomic_json(args.output, contract)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as handle:
        for relative, digest in files.items():
            handle.write(f"{digest}  {relative}\n")
    print(json.dumps(contract, sort_keys=True))


if __name__ == "__main__":
    main()
