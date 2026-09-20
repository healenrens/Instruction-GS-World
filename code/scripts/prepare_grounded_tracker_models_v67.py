#!/usr/bin/env python3
"""Download only the two frozen segmentation models; no environment reinstall."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models_root", required=True)
    args = parser.parse_args()
    # This explicitly online preparation command is separate from the offline GPU job.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from huggingface_hub import snapshot_download

    for repo, name in (
        ("IDEA-Research/grounding-dino-base", "grounding-dino-base"),
        ("facebook/sam2.1-hiera-large", "sam2.1-hiera-large"),
    ):
        path = Path(args.models_root) / name
        print(f"[grounded-tracker-prepare] repo={repo} destination={path}", flush=True)
        snapshot_download(
            repo_id=repo,
            local_dir=str(path),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        )
    print("[grounded-tracker-prepare] complete", flush=True)


if __name__ == "__main__":
    main()
