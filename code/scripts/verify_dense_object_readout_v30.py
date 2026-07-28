#!/usr/bin/env python3
"""Fail-closed server preflight for the v30 dense object readout."""

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

from igsw.adaptive_gaussian_wm import AdaptiveGaussianObjectWorldModel  # noqa: E402
from igsw.adaptive_gaussian_wm.checkpointing import (  # noqa: E402
    CHECKPOINT_VERSION,
    warm_start_model,
)
from igsw.adaptive_gaussian_wm.config import AdaptiveGaussianWMConfig  # noqa: E402
from igsw.adaptive_gaussian_wm.dense_readout_contracts import (  # noqa: E402
    DENSE_PREFLIGHT_CONTRACT,
)
from igsw.adaptive_gaussian_wm.feature_readout_runtime import (  # noqa: E402
    decode_current_dense_readout,
)
from igsw.adaptive_gaussian_wm.loss_weights import (  # noqa: E402
    AdaptiveGaussianLossWeights,
)
from igsw.adaptive_gaussian_wm.readout_runtime import (  # noqa: E402
    current_background_feature,
)
from igsw.adaptive_gaussian_wm.sequence_contract import (  # noqa: E402
    EPISODE_MANIFEST_NAME,
    EPISODE_VERIFIED_NAME,
)
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v28_training import ARCHITECTURE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--init_from", required=True)
    parser.add_argument("--gate_report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher_sidecar", default="")
    parser.add_argument("--history_frames", type=int, default=4)
    parser.add_argument("--future_frames", type=int, default=4)
    parser.add_argument("--sequence_anchors", default="3,5,8")
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


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def verify_manifest(data: str) -> str:
    manifest = os.path.join(data, EPISODE_MANIFEST_NAME)
    verified = os.path.join(data, EPISODE_VERIFIED_NAME)
    require(os.path.isfile(manifest), f"missing episode manifest: {manifest}")
    require(os.path.isfile(verified), f"missing manifest digest: {verified}")
    with open(verified, encoding="utf-8") as handle:
        expected = handle.read().strip().split()[0]
    actual = file_sha256(manifest)
    require(actual == expected, "episode manifest digest differs")
    return actual


def max_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def rms_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.sqrt((left.float() - right.float()).square().mean().clamp_min(1e-12))
    )


def direct_dense_state(model, batch: dict, output: dict, slot_offset: float = 0.0):
    tokens = output["history_token_states"][-1]
    slots = output["history_slot_states"][-1]
    predicted_slots = slots.slots[:, None] + slot_offset
    predicted_centers = slots.center[:, None] + slot_offset * 0.5
    predicted_scale = (
        slots.relative_scale[:, None] if hasattr(slots, "relative_scale") else None
    )
    return model.dense_readout(
        tokens,
        slots.assignment,
        slots.slots,
        predicted_slots,
        batch["history_coordinates"][:, -1:],
        batch["history_valid"][:, -1:],
        current_background_feature(batch),
        current_centers=slots.center,
        predicted_centers=predicted_centers,
        current_activity=slots.activity,
        predicted_visibility=slots.activity[:, None],
        current_scale=getattr(slots, "relative_scale", None),
        predicted_scale=predicted_scale,
    )


def main() -> None:
    args = parse_args()
    for name in ("data", "init_from", "gate_report", "output"):
        require(os.path.isabs(getattr(args, name)), f"--{name} must be absolute")
    require(torch.cuda.is_available(), "dense readout preflight requires CUDA")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
    )
    require(not status.strip(), "dense readout preflight requires a clean worktree")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    data_sha256 = verify_manifest(args.data)
    source_sha256 = file_sha256(args.init_from)
    dataset = CausalVisualSequenceDataset(
        args.data,
        "train",
        history_frames=args.history_frames,
        future_frames=args.future_frames,
        anchors=args.sequence_anchors,
        max_items=2,
        teacher_sidecar=args.teacher_sidecar,
    )
    require(dataset.data_sha256 == data_sha256, "dataset manifest differs")
    base_gate = load_json(args.gate_report)
    expected_gate = {
        "status": "passed",
        "architecture": ARCHITECTURE,
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_contract": "rolling_recovery_v1",
        "readout_backend": "dense_object_assignment",
        "git_commit": commit,
        "data_manifest_sha256": data_sha256,
        "teacher_sidecar_sha256": dataset.teacher_sidecar_sha256,
    }
    require(
        all(base_gate.get(key) == value for key, value in expected_gate.items()),
        "base verifier report differs from the v30 contract",
    )

    source = torch.load(
        args.init_from, map_location="cpu", weights_only=False, mmap=True
    )
    source_version = source.get("checkpoint_version")
    require(
        source_version in (28, CHECKPOINT_VERSION),
        "dense readout source must be a v28 or v30 checkpoint",
    )
    require(source.get("phase") == "representation", "source is not representation")
    config = AdaptiveGaussianWMConfig.object_memory_full(dataset.feature_dim)
    require(config.dense_object_readout, "v30 config did not enable dense readout")
    require(config.gaussian_children == 1, "v30 config enabled Gaussian children")
    device = torch.device("cuda:0")
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    warm_start = warm_start_model(model, source)
    require(not warm_start["unexpected"], "warm start has unexpected parameters")
    require(not warm_start["shape_mismatch"], "warm start has shape mismatches")
    require(
        all(name.startswith("dense_readout.") for name in warm_start["missing"]),
        "warm start has missing parameters outside the new dense readout",
    )
    if source_version < CHECKPOINT_VERSION:
        require(
            bool(warm_start["missing"]),
            "older source unexpectedly contains every v30 dense parameter",
        )
    del source
    model.requires_grad_(False)
    model.dense_readout.requires_grad_(True)
    model.train()
    cpu_batch = default_collate([dataset[0]])
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
    with amp:
        output = model(
            batch,
            phase="object_memory_representation_loss",
            loss_weights=weights,
        )
    loss = output["loss"]
    require(bool(torch.isfinite(loss)), "dense preflight loss is not finite")
    baseline_difference = max_difference(
        output["current_dense_readout"].feature[:, 0],
        output["history_token_states"][-1].reconstructed_features,
    )
    baseline_rms = rms_difference(
        output["current_dense_readout"].feature[:, 0],
        output["history_token_states"][-1].reconstructed_features,
    )
    require(
        baseline_difference < 3e-2 and baseline_rms < 2e-3,
        "dense readout initialization does not preserve GPSToken reconstruction",
    )
    loss.backward()
    gradients = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    require(gradients, "dense preflight produced no gradients")
    require(
        all(name.startswith("dense_readout.") for name, _ in gradients),
        "dense preflight reached a frozen module",
    )
    require(
        all(bool(torch.isfinite(gradient).all()) for _, gradient in gradients),
        "dense preflight produced non-finite gradients",
    )
    with torch.no_grad():
        direct = direct_dense_state(model.eval(), batch, output)
        changed = direct_dense_state(model, batch, output, slot_offset=0.1)
    zero_difference = max_difference(
        direct.feature, output["current_dense_readout"].feature
    )
    intervention = max(
        max_difference(changed.feature, direct.feature),
        max_difference(changed.assignment, direct.assignment),
    )
    swapped = dict(batch)
    swapped["future_features"] = batch["future_features"].roll(1, dims=2)
    with torch.no_grad():
        swapped_history = model.encode_history(swapped)
        swapped_current = decode_current_dense_readout(model, swapped, swapped_history)
    future_swap_difference = max_difference(direct.feature, swapped_current.feature)
    require(zero_difference < 2e-3, "identical object state changed the readout")
    require(intervention > 1e-5, "dense readout ignores object intervention")
    require(
        future_swap_difference < 1e-6,
        "future content reached the current dense readout",
    )
    report = {
        "status": "passed",
        "contract": DENSE_PREFLIGHT_CONTRACT,
        "git_commit": commit,
        "checkpoint_version": CHECKPOINT_VERSION,
        "data": os.path.abspath(args.data),
        "data_manifest_sha256": data_sha256,
        "teacher_sidecar_sha256": dataset.teacher_sidecar_sha256,
        "source_checkpoint": os.path.abspath(args.init_from),
        "source_checkpoint_sha256": source_sha256,
        "source_checkpoint_version": source_version,
        "base_gate_sha256": file_sha256(args.gate_report),
        "warm_start_missing": warm_start["missing"],
        "initial_token_reconstruction_max_difference": baseline_difference,
        "initial_token_reconstruction_rms_difference": baseline_rms,
        "identical_state_max_difference": zero_difference,
        "object_intervention_max_difference": intervention,
        "future_swap_max_difference": future_swap_difference,
        "loss": float(loss.detach()),
        "gradient_tensors": len(gradients),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
