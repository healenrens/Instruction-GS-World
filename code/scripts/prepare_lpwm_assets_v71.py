#!/usr/bin/env python3
"""Fetch upstream source and local T5/VGG/LPIPS assets during preparation only."""

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igsw.object_event_baselines import LPWM_REVISION


LPIPS_REVISION = "082bb24f84c091ea94de2867d34c4544f68e0963"
LPIPS_URL = (f"https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/{LPIPS_REVISION}"
             "/lpips/weights/v0.1/vgg.pth")


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        partial = path.with_suffix(path.suffix+".partial")
        urlretrieve(url, partial)
        partial.replace(path)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", default="/mnt/pfs/public/xuhaoming/instruct_gs_world")
    args = parser.parse_args()
    root = Path(args.work_root).resolve()
    cache, temporary = root/"cache/lpwm", root/"tmp/lpwm-v71"
    cache.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HOME=str(cache/"huggingface"), HF_HUB_CACHE=str(cache/"huggingface/hub"),
                      HF_XET_CACHE=str(cache/"huggingface/xet"), TORCH_HOME=str(cache/"torch"),
                      XDG_CACHE_HOME=str(cache), TMPDIR=str(temporary))
    from huggingface_hub import HfApi, snapshot_download
    from torchvision.models import VGG16_Weights

    repository = root/"third_party"/f"lpwm-{LPWM_REVISION[:7]}"
    if not repository.is_dir():
        archive = root/"tmp/lpwm-v71"/f"lpwm-{LPWM_REVISION}.tar.gz"
        download(f"https://codeload.github.com/taldatech/lpwm/tar.gz/{LPWM_REVISION}", archive)
        stage = archive.parent/f"lpwm-{LPWM_REVISION}"
        stage.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xzf", str(archive), "--strip-components=1", "-C", str(stage)], check=True)
        repository.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(repository)
    sys.path.insert(0, str(repository))
    loss_module = importlib.import_module("utils.loss_functions")
    model_root = root/"models/lpwm-v71"
    vgg_url, lpips_url = VGG16_Weights.IMAGENET1K_V1.url, LPIPS_URL
    vgg = download(vgg_url, model_root/"vgg16.pth")
    lpips = download(lpips_url, model_root/"vgg_lpips.pth")
    api, model_id = HfApi(), "google-t5/t5-large"
    revision = api.model_info(model_id).sha
    files = api.list_repo_files(model_id, revision=revision)
    weights = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
    t5 = model_root/"t5-large"
    snapshot_download(model_id, revision=revision, local_dir=t5,
                      allow_patterns=["*.json", "spiece.model", weights])
    report = {"repository": str(repository), "upstream_revision": LPWM_REVISION,
              "native_config": str(repository/"configs/langtable.json"),
              "vgg16_weights": vgg, "vgg16_source": vgg_url,
              "lpips_weights": lpips, "lpips_source": lpips_url,
              "lpips_version": "0.1", "lpips_revision": LPIPS_REVISION,
              "lpips_upstream_mirror": loss_module.URL_MAP["vgg_lpips"],
              "t5_encoder": str(t5), "t5_model_id": model_id, "t5_revision": revision,
              "training_downloads": False}
    path = root/"metadata/lpwm_assets_v71.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
