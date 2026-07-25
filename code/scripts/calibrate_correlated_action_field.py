"""Create an inference-calibrated posterior checkpoint without changing weights."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.latent_particle_wm.action_field import ActionFieldConfig  # noqa: E402


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--posterior_xy_scale", type=float, default=0.25)
    parser.add_argument("--posterior_depth_scale", type=float, default=0.0)
    args = parser.parse_args()
    if not 0.0 <= args.posterior_xy_scale <= 1.0:
        raise ValueError("posterior_xy_scale must be in [0, 1]")
    if not 0.0 <= args.posterior_depth_scale <= 1.0:
        raise ValueError("posterior_depth_scale must be in [0, 1]")

    source = os.path.abspath(args.checkpoint)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    config = ActionFieldConfig(**checkpoint["config"])
    if config.posterior_xy_scale != 1.0 or config.posterior_depth_scale != 1.0:
        raise ValueError("source checkpoint is already posterior-calibrated")
    config.posterior_xy_scale = args.posterior_xy_scale
    config.posterior_depth_scale = args.posterior_depth_scale
    checkpoint["config"] = config.to_dict()
    checkpoint["inference_calibration"] = {
        "kind": "posterior_residual_scale",
        "posterior_xy_scale": args.posterior_xy_scale,
        "posterior_depth_scale": args.posterior_depth_scale,
        "prior_xy_scale": 1.0,
        "prior_depth_scale": 1.0,
        "selected_on": ["train", "heldseed"],
        "source_checkpoint": source,
        "source_sha256": file_sha256(source),
    }

    output = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = f"{output}.tmp.{os.getpid()}"
    torch.save(checkpoint, temporary)
    os.replace(temporary, output)
    print(output)


if __name__ == "__main__":
    main()
