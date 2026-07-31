#!/usr/bin/env python3
"""Server gate for RGB-only, per-rank JIT-DINO Object Memory JEPA v38."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
import json
import os
import subprocess
import sys

import torch
from torch.utils.data import default_collate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.checkpointing import CHECKPOINT_VERSION  # noqa: E402
from igsw.adaptive_gaussian_wm.episode_sequence_dataset import (  # noqa: E402
    CausalVisualEpisodeDataset,
)
from igsw.adaptive_gaussian_wm.jit_dino_runtime import (  # noqa: E402
    JitDinoFeatureRuntime,
)
from igsw.adaptive_gaussian_wm.rgb_episode_cache_contract import (  # noqa: E402
    JIT_DINO_IMAGE_SIZE,
    JIT_DINO_MODEL,
    RGB_EPISODE_CACHE_VERSION,
    file_sha256,
    validate_manifest,
)
from igsw.adaptive_gaussian_wm.robotwin_lerobot_source import (  # noqa: E402
    LEROBOT_DEFAULT_ROOT,
    LEROBOT_DEFAULT_VARIANTS,
    LEROBOT_EXPECTED_FPS,
    LEROBOT_SOURCE_KIND,
)
from igsw.adaptive_gaussian_wm.sequence_contract import (  # noqa: E402
    EPISODE_MANIFEST_NAME,
    EPISODE_VERIFIED_NAME,
)


FEATURE_CONTRACT = "jit_backbone_native_dinov2_l_1024"
READOUT_BACKEND = "change_only_object_residual"


def load_verification_core():
    path = os.path.join(os.path.dirname(__file__), "verify_object_memory_jepa_v37.py")
    spec = importlib.util.spec_from_file_location("object_memory_v37_verify_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load verification core: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher_sidecar", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--expected_local_gpus", default="auto")
    parser.add_argument("--jit_dino_batch", type=int, default=4)
    args = parser.parse_args()
    for name in ("data", "output"):
        require(os.path.isabs(getattr(args, name)), f"--{name} must be absolute")
    for name in ("teacher_sidecar", "checkpoint"):
        value = getattr(args, name)
        require(not value or os.path.isabs(value), f"--{name} must be absolute")
    require(args.jit_dino_batch > 0, "--jit_dino_batch must be positive")
    return args


def verify_manifest(data: str) -> tuple[str, dict]:
    path = os.path.join(data, EPISODE_MANIFEST_NAME)
    checksum_path = os.path.join(data, EPISODE_VERIFIED_NAME)
    require(os.path.isfile(path), "RGB episode manifest is missing")
    require(os.path.isfile(checksum_path), "RGB manifest checksum is missing")
    checksum = subprocess.run(
        ["sha256sum", "-c", "--status", EPISODE_VERIFIED_NAME],
        cwd=data,
        check=False,
    )
    require(checksum.returncode == 0, "RGB manifest checksum failed")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest, path)
    source = manifest["source"]
    require(
        manifest["episode_cache_version"] == RGB_EPISODE_CACHE_VERSION,
        "data is not the v38 RGB-only cache",
    )
    require(source.get("kind") == LEROBOT_SOURCE_KIND, "source is not LeRobot-v3")
    require(
        os.path.realpath(source.get("path", ""))
        == os.path.realpath(LEROBOT_DEFAULT_ROOT),
        "RoboTwin source root is not authoritative",
    )
    require(
        tuple(source.get("variants", ())) == LEROBOT_DEFAULT_VARIANTS,
        "source variants differ",
    )
    require(
        float(source.get("expected_source_fps", 0.0)) == LEROBOT_EXPECTED_FPS
        and int(source.get("source_frame_stride", 0)) == 1,
        "RoboTwin source is not native stride-1 30 Hz",
    )
    return file_sha256(path), manifest


def main() -> None:
    args = parse_args()
    require(torch.cuda.is_available(), "v38 verifier requires CUDA")
    local_gpus = torch.cuda.device_count()
    require(local_gpus > 0, "no visible CUDA GPU")
    if args.expected_local_gpus != "auto":
        require(
            args.expected_local_gpus.isdigit()
            and int(args.expected_local_gpus) == local_gpus,
            f"visible GPU count is {local_gpus}, expected {args.expected_local_gpus}",
        )
    core = load_verification_core()
    commit = core.verify_repository()
    manifest_sha256, _ = verify_manifest(args.data)
    dataset = CausalVisualEpisodeDataset(
        args.data,
        "train",
        history_frames=4,
        future_frames=4,
        anchors="3,5,8",
        max_items=1,
        teacher_sidecar=args.teacher_sidecar,
        feature_source="jit",
    )
    if dataset.teacher_sidecar is not None:
        dataset.teacher_sidecar.verify_hashes()
    device = torch.device("cuda:0")
    raw_batch = core.to_device(default_collate([dataset[0]]), device)
    dino_amp = "bf16" if torch.cuda.is_bf16_supported() else "fp32"
    feature_runtime = JitDinoFeatureRuntime(device, dino_amp, args.jit_dino_batch)
    batch = feature_runtime(raw_batch)
    del feature_runtime
    torch.cuda.empty_cache()

    config = AdaptiveGaussianWMConfig.object_memory_full(dataset.feature_dim)
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    require(model.config.change_residual_readout, "v38 change readout is disabled")
    require(model.gaussian_readout is None, "legacy Gaussian readout remains")
    checkpoint = core.verify_checkpoint(args.checkpoint, model)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if torch.cuda.is_bf16_supported()
        else nullcontext
    )
    model.eval()
    with torch.no_grad(), amp_context():
        causal = core.verify_causal_paths(model, batch)
    model.train()
    model.zero_grad(set_to_none=True)
    with amp_context():
        _, training = core.verify_forward_backward(model, batch)
    report = {
        "status": "passed",
        "architecture": "object_memory_v1",
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_contract": "rolling_recovery_v1",
        "readout_backend": READOUT_BACKEND,
        "feature_source": "jit",
        "feature_contract": FEATURE_CONTRACT,
        "feature_dim": dataset.feature_dim,
        "jit_dino_model": JIT_DINO_MODEL,
        "jit_dino_image_size": JIT_DINO_IMAGE_SIZE,
        "jit_dino_frame_batch": args.jit_dino_batch,
        "control_hz": float(dataset.control_hz),
        "git_commit": commit,
        "data": os.path.abspath(args.data),
        "data_manifest_sha256": manifest_sha256,
        "teacher_sidecar": "enabled" if args.teacher_sidecar else "disabled",
        "teacher_sidecar_sha256": dataset.teacher_sidecar_sha256,
        "local_gpu_count": local_gpus,
        "gpu_policy": args.expected_local_gpus,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "relative_geometry_max_difference": core.verify_relative_geometry(device),
        "ddp_parameter_contract": core.verify_parameters(model),
        **checkpoint,
        **causal,
        **training,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
