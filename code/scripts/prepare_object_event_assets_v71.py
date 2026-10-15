#!/usr/bin/env python3
"""Download the matched public RAE assets; run only during preparation."""

import argparse
import json
from pathlib import Path
import subprocess

from huggingface_hub import hf_hub_download, snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", default="/mnt/pfs/xuhaoming/wm-align")
    parser.add_argument("--rae-repository", required=True)
    parser.add_argument("--image-size", type=int, choices=(256, 512), default=512)
    args = parser.parse_args()
    root, repository = Path(args.work_root).resolve(), Path(args.rae_repository).resolve()
    model_root = root / "models"
    encoder = model_root / "dinov2-with-registers-base"
    snapshot_download("facebook/dinov2-with-registers-base", local_dir=encoder,
                      allow_patterns=["*.json", "*.safetensors"])
    variant = "ViTXL_n08" if args.image_size == 256 else "ViTXL_n08_i512"
    statistics = "imagenet1k" if args.image_size == 256 else "imagenet1k_512"
    decoder = hf_hub_download("nyu-visionx/RAE-collections",
                              f"decoders/dinov2/wReg_base/{variant}/model.pt", local_dir=model_root / "rae")
    stats = hf_hub_download("nyu-visionx/RAE-collections",
                            f"stats/dinov2/wReg_base/{statistics}/stat.pt", local_dir=model_root / "rae")
    payload = {"rae_repository": str(repository),
               "rae_revision": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip(),
               "image_size": args.image_size, "encoder": str(encoder),
               "decoder_config": str(repository / "configs/decoder/ViTXL"),
               "decoder": decoder, "statistics": stats}
    path = root / "metadata" / f"rae_assets_{args.image_size}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"assets": str(path), **payload}), flush=True)


if __name__ == "__main__":
    main()
