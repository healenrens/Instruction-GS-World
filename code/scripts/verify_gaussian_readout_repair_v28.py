#!/usr/bin/env python3
"""Server-only preflight for causal, isolated Gaussian readout repair."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import subprocess
import sys

import torch
from torch.utils.data._utils.collate import default_collate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.checkpointing import (  # noqa: E402
    CHECKPOINT_VERSION,
    warm_start_model,
)
from igsw.adaptive_gaussian_wm.loss_weights import (  # noqa: E402
    AdaptiveGaussianLossWeights,
)
from igsw.adaptive_gaussian_wm.readout_repair import (  # noqa: E402
    current_readout_objective,
)
from igsw.adaptive_gaussian_wm.sequence_contract import (  # noqa: E402
    EPISODE_MANIFEST_NAME,
)
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v28_training import ARCHITECTURE  # noqa: E402


PREFLIGHT_CONTRACT = "gaussian_readout_preflight_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--init_from", required=True)
    parser.add_argument("--base_gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher_sidecar", default="")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def swap_future(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    changed = dict(batch)
    batch_size = batch["future_features"].shape[0]
    order = torch.arange(batch_size - 1, -1, -1, device=batch["future_features"].device)
    for name in tuple(batch):
        if name.startswith("future_") and torch.is_tensor(batch[name]):
            changed[name] = batch[name][order]
    return changed


def current_output(model, batch: dict, weights) -> dict:
    history_mask = torch.zeros(
        batch["history_features"].shape[0],
        batch["history_features"].shape[1],
        model.config.object_slots,
        dtype=torch.bool,
        device=batch["history_features"].device,
    )
    return model(
        batch,
        history_mask=history_mask,
        phase="object_memory_representation_loss",
        loss_weights=weights,
    )


def state_max_difference(left, right) -> float:
    fields = (
        "feature",
        "center",
        "covariance",
        "depth_order",
        "opacity",
        "activation",
    )
    return max(
        float((getattr(left, name).float() - getattr(right, name).float()).abs().max())
        for name in fields
    )


def verify_base_gate(path: str, commit: str, data_sha256: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    expected = {
        "status": "passed",
        "architecture": ARCHITECTURE,
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_contract": "rolling_recovery_v1",
        "git_commit": commit,
        "data_manifest_sha256": data_sha256,
    }
    mismatch = {
        name: {"base_gate": report.get(name), "current": value}
        for name, value in expected.items()
        if report.get(name) != value
    }
    require(not mismatch, f"base verifier gate differs: {mismatch}")
    return report


def main() -> None:
    args = parse_args()
    for name in ("data", "init_from", "base_gate", "output"):
        require(os.path.isabs(getattr(args, name)), f"--{name} must be absolute")
    require(torch.cuda.is_available(), "readout repair verifier requires CUDA")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
    )
    require(not status.strip(), "repository must be clean before verification")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    manifest_path = os.path.join(args.data, EPISODE_MANIFEST_NAME)
    require(os.path.isfile(manifest_path), "episode manifest is missing")
    data_sha256 = file_sha256(manifest_path)
    base_gate = verify_base_gate(args.base_gate, commit, data_sha256)
    checkpoint = torch.load(
        args.init_from, map_location="cpu", weights_only=False, mmap=True
    )
    require(
        checkpoint.get("checkpoint_version") == CHECKPOINT_VERSION,
        "readout repair requires a v28 initialization checkpoint",
    )
    require(
        checkpoint.get("config", {}).get("architecture") == ARCHITECTURE,
        "initialization architecture differs",
    )
    require(
        checkpoint.get("phase") == "representation"
        and checkpoint.get("args", {}).get("training_stage") == "representation",
        "isolated repair must start from representation",
    )
    dataset = CausalVisualSequenceDataset(
        args.data,
        "train",
        max_items=2,
        teacher_sidecar=args.teacher_sidecar,
    )
    require(
        base_gate.get("teacher_sidecar_sha256", "")
        == getattr(dataset, "teacher_sidecar_sha256", ""),
        "base gate and readout preflight use different teacher sidecars",
    )
    cpu_batch = default_collate([dataset[0], dataset[1]])
    device = torch.device("cuda:0")
    config = AdaptiveGaussianWMConfig.object_memory_full(dataset.feature_dim)
    require(config.gaussian_feature_residual, "feature residual is not enabled")
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    warm_start = warm_start_model(model, checkpoint)
    expected_missing = {
        "gaussian_readout.feature_residual_head.weight",
        "gaussian_readout.feature_residual_head.bias",
    }
    require(
        set(warm_start["missing"]) == expected_missing,
        f"unexpected warm-start missing parameters: {warm_start['missing']}",
    )
    require(not warm_start["unexpected"], "warm-start has unexpected parameters")
    require(not warm_start["shape_mismatch"], "warm-start has shape mismatches")
    batch = move_to_device(cpu_batch, device)
    weights = AdaptiveGaussianLossWeights(
        future=0.0,
        history=0.0,
        flow=0.0,
        feature=0.0,
        allocator=0.0,
        slot=0.0,
        action=0.0,
        action_specificity=0.0,
        geometry=0.0,
        rgb=0.0,
        current_readout=1.0,
        readout_regularization=0.01,
    )
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_bf16_supported()
        else nullcontext()
    )
    model.eval()
    with torch.no_grad(), amp:
        original = current_output(model, batch, weights)
        changed = current_output(model, swap_future(batch), weights)
    future_swap_difference = state_max_difference(
        original["current_gaussian_readout"],
        changed["current_gaussian_readout"],
    )
    require(
        future_swap_difference < 1e-6,
        "future values leaked into the current readout anchor",
    )
    model.requires_grad_(False)
    model.gaussian_readout.requires_grad_(True)
    model.train()
    model.zero_grad(set_to_none=True)
    with amp:
        output = current_output(model, batch, weights)
        anchor, regularization, _ = current_readout_objective(model, batch, output)
        loss = anchor + 0.01 * regularization
    loss.backward()
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    require(gradients, "isolated readout objective produced no gradients")
    require(
        all(name.startswith("gaussian_readout.") for name in gradients),
        "isolated readout gradients escaped the readout module",
    )
    require(
        all(bool(torch.isfinite(value).all()) for value in gradients.values()),
        "isolated readout gradients are non-finite",
    )
    require(
        any(name.startswith("gaussian_readout.feature_residual_head.") for name in gradients),
        "feature residual head received no gradient",
    )
    report = {
        "status": "passed",
        "contract": PREFLIGHT_CONTRACT,
        "architecture": ARCHITECTURE,
        "checkpoint_version": CHECKPOINT_VERSION,
        "git_commit": commit,
        "data_manifest_sha256": data_sha256,
        "teacher_sidecar_sha256": base_gate.get("teacher_sidecar_sha256", ""),
        "init_checkpoint": os.path.abspath(args.init_from),
        "init_checkpoint_sha256": file_sha256(args.init_from),
        "future_swap_max_difference": future_swap_difference,
        "anchor_loss": float(anchor.detach()),
        "regularization_loss": float(regularization.detach()),
        "gradient_parameter_count": len(gradients),
        "gradient_parameters": sorted(gradients),
        "warm_start": warm_start,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
