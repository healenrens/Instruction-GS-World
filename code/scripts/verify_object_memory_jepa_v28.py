#!/usr/bin/env python3
"""Server-only integrity and causal verifier for Object Memory JEPA v28."""
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

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.checkpointing import CHECKPOINT_VERSION  # noqa: E402
from igsw.adaptive_gaussian_wm.dynamics_runtime import run_object_dynamics  # noqa: E402
from igsw.adaptive_gaussian_wm.observed_action import posterior_from_targets  # noqa: E402
from igsw.adaptive_gaussian_wm.relative_geometry import (  # noqa: E402
    pairwise_relative_geometry,
)
from igsw.adaptive_gaussian_wm.scale import signed_gap_scale  # noqa: E402
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.sequence_contract import (  # noqa: E402
    EPISODE_MANIFEST_NAME,
    EPISODE_VERIFIED_NAME,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher_sidecar", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--history_frames", type=int, default=4)
    parser.add_argument("--future_frames", type=int, default=4)
    parser.add_argument("--sequence_anchors", default="3,5,8")
    parser.add_argument("--expected_local_gpus", type=int, default=8)
    return parser.parse_args()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def max_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def verify_repository() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
    )
    require(not status.strip(), "repository must be clean before verification")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def verify_data_manifest(data: str) -> str:
    manifest = os.path.join(data, EPISODE_MANIFEST_NAME)
    verified = os.path.join(data, EPISODE_VERIFIED_NAME)
    require(os.path.isfile(manifest), f"missing episode manifest: {manifest}")
    require(os.path.isfile(verified), f"missing verified digest: {verified}")
    with open(verified, encoding="utf-8") as handle:
        expected = handle.read().strip().split()[0]
    actual = file_sha256(manifest)
    require(actual == expected, "episode manifest verification digest differs")
    return actual


def build_batch(args):
    dataset = CausalVisualSequenceDataset(
        args.data,
        "train",
        history_frames=args.history_frames,
        future_frames=args.future_frames,
        anchors=args.sequence_anchors,
        max_items=2,
        load_rgb=False,
        teacher_sidecar=args.teacher_sidecar,
    )
    require(dataset.feature_dim > 0, "dataset has no DINO feature dimension")
    require(len(dataset) >= 1, "dataset has no verification sample")
    require(
        hasattr(dataset, "teacher_sidecar"),
        "v28 requires the dense episode backend",
    )
    if dataset.teacher_sidecar is not None:
        dataset.teacher_sidecar.verify_hashes()
    return dataset, default_collate([dataset[0]])


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def changed_future(batch: dict) -> dict:
    modified = dict(batch)
    modified["future_features"] = batch["future_features"].flip(-1)
    return modified


def changed_sidecar(batch: dict) -> dict:
    modified = dict(batch)
    for name, value in batch.items():
        if name.startswith("teacher_future_") and torch.is_tensor(value):
            modified[name] = value.flip(-1)
    return modified


def history_and_prior(model, batch: dict):
    history = model.encode_history(batch)
    history_scale = signed_gap_scale(
        batch["history_times"], model.config.gap_reference
    )
    future_scale = signed_gap_scale(
        batch["future_times"], model.config.gap_reference
    )
    prior = model.prior_context(history, future_scale, history_scale)
    return history, history_scale, future_scale, prior


def verify_geometry_invariance(device: torch.device) -> float:
    center = torch.tensor(
        [[[0.1, -0.2], [0.4, 0.3], [-0.5, 0.2]]], device=device
    )
    scale = torch.tensor([[0.2, 0.4, 0.3]], device=device)
    disparity = torch.tensor([[0.1, -0.2, 0.5]], device=device)
    visible = torch.tensor([[1.0, 0.8, 0.5]], device=device)
    original = pairwise_relative_geometry(center, scale, disparity, visible)
    transformed = pairwise_relative_geometry(
        center * 1.7 + torch.tensor([0.2, -0.1], device=device),
        scale * 1.7,
        disparity + 3.0,
        visible,
    )
    difference = max_difference(original, transformed)
    require(difference < 1e-5, "relative geometry is not scale/shift invariant")
    return difference


def verify_checkpoint(path: str, model) -> dict:
    if not path:
        return {"checkpoint": "not_provided"}
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False, mmap=True
    )
    require(
        checkpoint.get("checkpoint_version") == CHECKPOINT_VERSION,
        "checkpoint is not version 28",
    )
    require(
        checkpoint.get("config", {}).get("architecture") == "object_memory_v1",
        "checkpoint architecture differs",
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return {
        "checkpoint": os.path.abspath(path),
        "checkpoint_phase": checkpoint.get("phase"),
        "checkpoint_step": int(checkpoint.get("global_step", -1)),
    }


def verify_parameter_contract(model) -> dict:
    parameters = list(model.named_parameters())
    names = [name for name, _ in parameters]
    require(len(names) == len(set(names)), "model has duplicate parameter names")
    target_trainable = [
        name
        for name, parameter in parameters
        if name.startswith("target_") and parameter.requires_grad
    ]
    require(not target_trainable, "EMA target parameters are trainable")
    action_names = [
        name
        for name, _ in parameters
        if name.startswith("latent_actions.posterior.")
        or name.startswith("dynamics.action_")
    ]
    require(action_names, "DDP parameter set has no action branch")
    return {
        "parameter_tensors": len(parameters),
        "parameter_count": sum(parameter.numel() for _, parameter in parameters),
        "trainable_parameter_count": sum(
            parameter.numel()
            for _, parameter in parameters
            if parameter.requires_grad
        ),
        "action_parameter_tensors": len(action_names),
        "ema_target_trainable_tensors": len(target_trainable),
    }


@torch.no_grad()
def verify_model(model, batch: dict) -> dict:
    history, history_scale, future_scale, prior = history_and_prior(model, batch)
    future_batch = changed_future(batch)
    changed_history, _, _, changed_prior = history_and_prior(model, future_batch)
    history_difference = max_difference(history["slots"], changed_history["slots"])
    prior_difference = max_difference(prior, changed_prior)
    require(history_difference < 1e-6, "future content reached history encoder")
    require(prior_difference < 1e-6, "future content reached prior context")

    _, target_future = model.encode_targets(batch)
    posterior = posterior_from_targets(
        model, batch, history, target_future, future_scale, None
    )[0]
    _, changed_target = model.encode_targets(future_batch)
    changed_posterior = posterior_from_targets(
        model, future_batch, history, changed_target, future_scale, None
    )[0]
    posterior_difference = max_difference(posterior, changed_posterior)
    require(posterior_difference > 1e-6, "posterior ignored changed future")
    require(
        posterior.shape[-2:] == (4, 32),
        "continuous action does not have shape [4,32]",
    )

    empty_mask = torch.zeros_like(history["activity"], dtype=torch.bool)
    memory = dict(
        history_relative_scale=history["relative_scale"],
        history_relative_disparity=history["relative_disparity"],
        history_relations=history["relations"],
        history_existence=history["existence"],
    )
    zero = run_object_dynamics(
        model,
        history["slots"],
        history["activity"],
        history_scale,
        future_scale,
        torch.zeros_like(posterior),
        empty_mask,
        history["center"],
        None,
        **memory,
    )
    action = run_object_dynamics(
        model,
        history["slots"],
        history["activity"],
        history_scale,
        future_scale,
        posterior,
        empty_mask,
        history["center"],
        None,
        **memory,
    )
    zero_residual = float(zero.action_slot_residual.float().abs().max())
    base_difference = max_difference(zero.base_future_slots, action.base_future_slots)
    require(zero_residual == 0.0, "zero action produced a residual")
    require(base_difference < 1e-6, "action changed the action-free base")

    current = history["last_memory"]
    predicted = model.object_memory.predict(
        current, batch["future_times"][:, 0] - batch["history_times"][:, -1]
    )
    occluded = replace(current, activity=torch.zeros_like(current.activity))
    corrected = model.object_memory.correct(
        predicted, occluded, history["token_states"][-1]
    )
    existence_difference = max_difference(
        corrected.existence, predicted.existence
    )
    require(existence_difference < 1e-6, "occlusion deleted object existence")

    sidecar_prior_difference = 0.0
    if "teacher_sidecar_present" in batch:
        sidecar_batch = changed_sidecar(batch)
        sidecar_history, _, _, sidecar_prior = history_and_prior(
            model, sidecar_batch
        )
        require(
            max_difference(history["slots"], sidecar_history["slots"]) < 1e-6,
            "teacher sidecar reached history encoder",
        )
        sidecar_prior_difference = max_difference(prior, sidecar_prior)
        require(sidecar_prior_difference < 1e-6, "teacher sidecar reached prior")
    active_count = history["token_states"][-1].active_count.float()
    require(bool((active_count >= 64).all()), "token gate fell below 64")
    require(bool((active_count <= 256).all()), "token gate exceeded 256")
    return {
        "history_future_swap_max_difference": history_difference,
        "prior_future_swap_max_difference": prior_difference,
        "posterior_future_swap_max_difference": posterior_difference,
        "sidecar_prior_swap_max_difference": sidecar_prior_difference,
        "zero_action_residual_max": zero_residual,
        "action_free_base_max_difference": base_difference,
        "occlusion_existence_max_difference": existence_difference,
        "active_token_count": float(active_count.mean()),
    }


def main() -> None:
    args = parse_args()
    require(os.path.isabs(args.data), "--data must be absolute")
    require(os.path.isabs(args.output), "--output must be absolute")
    if args.teacher_sidecar:
        require(os.path.isabs(args.teacher_sidecar), "--teacher_sidecar must be absolute")
    require(torch.cuda.is_available(), "v28 verifier requires CUDA")
    local_gpus = torch.cuda.device_count()
    require(
        local_gpus == args.expected_local_gpus,
        f"visible GPU count is {local_gpus}, expected {args.expected_local_gpus}",
    )
    commit = verify_repository()
    data_sha256 = verify_data_manifest(args.data)
    dataset, cpu_batch = build_batch(args)
    device = torch.device("cuda:0")
    config = AdaptiveGaussianWMConfig.object_memory_full(dataset.feature_dim)
    require(config.condition_dim == 0, "v28 unexpectedly enables language")
    require(not config.rgb_supervision, "v28 unexpectedly enables RGB loss")
    model = AdaptiveGaussianObjectWorldModel(config).to(device).eval()
    checkpoint = verify_checkpoint(args.checkpoint, model)
    batch = to_device(cpu_batch, device)
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_bf16_supported()
        else nullcontext()
    )
    with amp:
        model_checks = verify_model(model, batch)
    report = {
        "status": "passed",
        "architecture": "object_memory_v1",
        "checkpoint_version": CHECKPOINT_VERSION,
        "git_commit": commit,
        "data": os.path.abspath(args.data),
        "data_manifest_sha256": data_sha256,
        "teacher_sidecar": "enabled" if args.teacher_sidecar else "disabled",
        "teacher_sidecar_sha256": getattr(dataset, "teacher_sidecar_sha256", ""),
        "disabled_teacher_losses": (
            [] if args.teacher_sidecar else ["relative_disparity", "visibility"]
        ),
        "local_gpu_count": local_gpus,
        "expected_global_ranks": 16,
        "relative_geometry_max_difference": verify_geometry_invariance(device),
        "ddp_parameter_contract": verify_parameter_contract(model),
        **checkpoint,
        **model_checks,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
