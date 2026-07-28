#!/usr/bin/env python3
"""Fail-closed preflight for isolated v29 carrier training."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
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
from igsw.adaptive_gaussian_wm.carrier_contracts import (  # noqa: E402
    BASIS_GATE_CONTRACT,
    CARRIER_PREFLIGHT_CONTRACT,
)
from igsw.adaptive_gaussian_wm.config import AdaptiveGaussianWMConfig  # noqa: E402
from igsw.adaptive_gaussian_wm.loss_weights import (  # noqa: E402
    AdaptiveGaussianLossWeights,
)
from igsw.adaptive_gaussian_wm.readout_repair import first_query  # noqa: E402
from igsw.adaptive_gaussian_wm.readout_runtime import (  # noqa: E402
    current_background_feature,
    decode_gaussian_readout,
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
    parser.add_argument("--basis_gate_report", required=True)
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


def verify_zero_delta(model, batch: dict, output: dict) -> dict:
    tokens = output["history_token_states"][-1]
    slots = output["history_slot_states"][-1]
    conditioned = first_query(output["current_gaussian_readout"])
    direct = model.gaussian_readout.current_carrier(
        tokens, current_background_feature(batch)
    )
    fields = (
        "feature",
        "center",
        "covariance",
        "depth_order",
        "opacity",
        "activation",
        "background_feature",
    )
    differences = {
        name: max_difference(getattr(conditioned, name), getattr(direct, name))
        for name in fields
    }
    require(max(differences.values()) < 1e-5, "current object delta is not zero")

    changed, _ = decode_gaussian_readout(
        model,
        batch,
        tokens,
        slots,
        slots.slots[:, None] + 0.1,
        slots.center[:, None] + 0.05,
        predicted_relative_scale=slots.relative_scale[:, None] * 1.1,
        predicted_relative_disparity=slots.relative_disparity[:, None] + 0.1,
    )
    intervention = max(
        max_difference(getattr(first_query(changed), name), getattr(direct, name))
        for name in ("feature", "center", "covariance", "depth_order")
    )
    require(intervention > 1e-5, "carrier ignores an object-state intervention")
    return {
        "zero_delta_max_difference": max(differences.values()),
        "zero_delta_fields": differences,
        "object_intervention_max_difference": intervention,
    }


def main() -> None:
    args = parse_args()
    for name in (
        "data",
        "init_from",
        "gate_report",
        "basis_gate_report",
        "output",
    ):
        require(os.path.isabs(getattr(args, name)), f"--{name} must be absolute")
    require(torch.cuda.is_available(), "carrier preflight requires CUDA")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
    )
    require(not status.strip(), "carrier preflight requires a clean worktree")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    data_sha256 = verify_manifest(args.data)
    source_sha256 = file_sha256(args.init_from)
    basis_sha256 = file_sha256(args.basis_gate_report)
    base_gate = load_json(args.gate_report)
    basis_gate = load_json(args.basis_gate_report)
    expected_base = {
        "status": "passed",
        "architecture": ARCHITECTURE,
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_contract": "rolling_recovery_v1",
        "git_commit": commit,
        "data_manifest_sha256": data_sha256,
    }
    require(
        all(base_gate.get(key) == value for key, value in expected_base.items()),
        "base verifier report differs from the current contract",
    )
    selected = int(basis_gate.get("selected_children", 0))
    expected_basis = {
        "status": "passed",
        "contract": BASIS_GATE_CONTRACT,
        "git_commit": commit,
        "data_manifest_sha256": data_sha256,
        "source_checkpoint_sha256": source_sha256,
    }
    require(
        all(basis_gate.get(key) == value for key, value in expected_basis.items()),
        "basis gate report differs from the current contract",
    )
    require(selected in (2, 4, 8), "basis gate selected an invalid child count")

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
    cpu_batch = default_collate([dataset[0]])
    source = torch.load(
        args.init_from, map_location="cpu", weights_only=False, mmap=True
    )
    require(
        source.get("checkpoint_version") in (28, CHECKPOINT_VERSION),
        "carrier source must be a v28/v29 checkpoint",
    )
    require(source.get("phase") == "representation", "source is not representation")
    config = replace(
        AdaptiveGaussianWMConfig.object_memory_full(dataset.feature_dim),
        gaussian_children=selected,
        hierarchical_gaussian_carrier=True,
        dense_object_readout=False,
    )
    device = torch.device("cuda:0")
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    warm_start = warm_start_model(model, source)
    require(not warm_start["unexpected"], "warm start has unexpected parameters")
    require(not warm_start["shape_mismatch"], "warm start has shape mismatches")
    require(
        all(name.startswith("gaussian_readout.hierarchical.") for name in warm_start["missing"]),
        "warm start is missing parameters outside the new carrier",
    )
    model.requires_grad_(False)
    model.gaussian_readout.hierarchical.requires_grad_(True)
    model.train()
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
        readout_regularization=0.0,
        carrier_support=0.2,
        carrier_compact=0.001,
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
    require(bool(torch.isfinite(loss)), "carrier preflight loss is not finite")
    loss.backward()
    gradients = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    require(gradients, "carrier preflight produced no gradients")
    require(
        all(name.startswith("gaussian_readout.hierarchical.") for name, _ in gradients),
        "carrier preflight reached a frozen module",
    )
    require(
        all(bool(torch.isfinite(gradient).all()) for _, gradient in gradients),
        "carrier preflight produced non-finite gradients",
    )
    zero_delta = verify_zero_delta(model.eval(), batch, output)
    component_count = first_query(output["current_gaussian_readout"]).center.shape[2]
    parent_count = output["history_token_states"][-1].center.shape[1]
    require(
        component_count == parent_count * selected,
        "carrier component count does not match parent x children",
    )
    report = {
        "status": "passed",
        "contract": CARRIER_PREFLIGHT_CONTRACT,
        "git_commit": commit,
        "data_manifest_sha256": data_sha256,
        "source_checkpoint": os.path.abspath(args.init_from),
        "source_checkpoint_sha256": source_sha256,
        "basis_gate_sha256": basis_sha256,
        "selected_children": selected,
        "parent_count": parent_count,
        "component_count": component_count,
        "loss": float(loss.detach()),
        "gradient_tensors": len(gradients),
        "warm_start": warm_start,
        **zero_delta,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
