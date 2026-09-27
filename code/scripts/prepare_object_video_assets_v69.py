#!/usr/bin/env python3
"""Download-only preparation on a network-enabled host. Never called by training or tests."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request


def download(url, destination):
    destination = Path(destination)
    if destination.is_file():
        print(f"[v69-assets] reuse={destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as output:
        shutil.copyfileobj(source, output, length=8*1024*1024)
    temporary.replace(destination)
    print(f"[v69-assets] saved={destination}", flush=True)


def repository(archive_url, destination, cache):
    destination = Path(destination)
    if (destination / "hubconf.py").is_file():
        print(f"[v69-assets] reuse_repository={destination}", flush=True)
        return
    archive = cache / f"{destination.name}.tar.gz"
    download(archive_url, archive)
    temporary = destination.with_name(destination.name + ".extracting")
    temporary.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(archive), "--strip-components=1", "-C", str(temporary)], check=True)
    temporary.rename(destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime_root", required=True)
    p.add_argument("--dinov3_weights_url", required=True, help="Official download URL after accepting the DINOv3 model license.")
    p.add_argument("--dinov3_archive", default="https://github.com/facebookresearch/dinov3/archive/refs/heads/main.tar.gz")
    p.add_argument("--vjepa_archive", default="https://github.com/facebookresearch/vjepa2/archive/refs/heads/main.tar.gz")
    p.add_argument("--vjepa_weights_url", default="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt")
    p.add_argument("--include_vjepa", action="store_true")
    args = p.parse_args()
    root = Path(args.runtime_root)
    cache = root / "models/object_video_v69_downloads"
    repository(args.dinov3_archive, root / "third_party/dinov3", cache)
    download(args.dinov3_weights_url, root / "models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    if args.include_vjepa:
        repository(args.vjepa_archive, root / "third_party/vjepa2", cache)
        download(args.vjepa_weights_url, root / "models/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt")
    print(json.dumps({"runtime_root": str(root), "download_stage_only": True, "training_fetches_code_or_weights": False}), flush=True)


if __name__ == "__main__":
    main()
