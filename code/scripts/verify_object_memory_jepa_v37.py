#!/usr/bin/env python3
"""Server gate for full-DINO background-separated Object Memory JEPA v37."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import math
import os
import subprocess
import sys

import torch
from torch.utils.data import default_collate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianLossWeights,
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.change_readout_diagnostics import (  # noqa: E402
    CHANGE_READOUT_REQUIRED_DIAGNOSTICS,
    change_residual_readout_diagnostics,
    validate_change_readout_diagnostic_contract,
)
from igsw.adaptive_gaussian_wm.checkpointing import CHECKPOINT_VERSION  # noqa: E402
from igsw.adaptive_gaussian_wm.diagnostic_statistics import (  # noqa: E402
    finalize_diagnostic_metrics,
)
from igsw.adaptive_gaussian_wm.episode_sequence_dataset import (  # noqa: E402
    CausalVisualEpisodeDataset,
)
from igsw.adaptive_gaussian_wm.observed_action import (  # noqa: E402
    posterior_from_targets,
)
from igsw.adaptive_gaussian_wm.relative_geometry import (  # noqa: E402
    pairwise_relative_geometry,
)
from igsw.adaptive_gaussian_wm.robotwin_lerobot_source import (  # noqa: E402
    LEROBOT_DEFAULT_ROOT,
    LEROBOT_DEFAULT_VARIANTS,
    LEROBOT_EXPECTED_FPS,
    LEROBOT_SOURCE_KIND,
)
from igsw.adaptive_gaussian_wm.scale import signed_gap_scale  # noqa: E402
from igsw.adaptive_gaussian_wm.sequence_contract import (  # noqa: E402
    EPISODE_CACHE_VERSION,
    EPISODE_MANIFEST_NAME,
    EPISODE_VERIFIED_NAME,
)


FEATURE_CONTRACT = "backbone_native_dinov2_l_1024"
READOUT_BACKEND = "change_only_object_residual"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def max_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher_sidecar", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--expected_local_gpus", default="auto")
    args = parser.parse_args()
    for name in ("data", "output"):
        require(os.path.isabs(getattr(args, name)), f"--{name} must be absolute")
    for name in ("teacher_sidecar", "checkpoint"):
        value = getattr(args, name)
        require(not value or os.path.isabs(value), f"--{name} must be absolute")
    return args


def verify_repository() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
    )
    require(not status.strip(), "v37 verifier requires a clean worktree")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def verify_manifest(data: str) -> tuple[str, dict]:
    manifest_path = os.path.join(data, EPISODE_MANIFEST_NAME)
    verified_path = os.path.join(data, EPISODE_VERIFIED_NAME)
    require(os.path.isfile(manifest_path), "episode manifest is missing")
    require(os.path.isfile(verified_path), "verified manifest checksum is missing")
    result = subprocess.run(
        ["sha256sum", "-c", "--status", EPISODE_VERIFIED_NAME],
        cwd=data,
        check=False,
    )
    require(result.returncode == 0, "episode manifest checksum failed")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    cache = manifest.get("cache", {})
    source = manifest.get("source", {})
    require(
        manifest.get("episode_cache_version") == EPISODE_CACHE_VERSION,
        "episode cache version is not v37 native DINO",
    )
    require(manifest.get("complete") is True, "episode cache is incomplete")
    require(cache.get("feature_contract") == "backbone_native", "DINO is projected")
    require(int(cache.get("feature_dim", 0)) == 1024, "DINO feature_dim is not 1024")
    require(
        cache.get("model") == "vit_large_patch14_dinov2.lvd142m",
        "cache backbone is not DINOv2-L",
    )
    require(source.get("kind") == LEROBOT_SOURCE_KIND, "source is not LeRobot-v3")
    require(
        os.path.realpath(str(source.get("path", "")))
        == os.path.realpath(LEROBOT_DEFAULT_ROOT),
        "RoboTwin source root is not the authoritative fanyupeng dataset",
    )
    require(
        tuple(source.get("variants", ())) == LEROBOT_DEFAULT_VARIANTS,
        "cache does not contain both clean and randomized splits",
    )
    require(
        float(source.get("expected_source_fps", 0.0)) == LEROBOT_EXPECTED_FPS
        and int(source.get("source_frame_stride", 0)) == 1
        and float(manifest.get("control_hz", 0.0)) == LEROBOT_EXPECTED_FPS,
        "RoboTwin cache is not native stride-1 30 Hz",
    )
    return file_sha256(manifest_path), manifest


def build_batch(args) -> tuple[CausalVisualEpisodeDataset, dict[str, torch.Tensor]]:
    dataset = CausalVisualEpisodeDataset(
        args.data,
        "train",
        history_frames=4,
        future_frames=4,
        anchors="3,5,8",
        max_items=1,
        teacher_sidecar=args.teacher_sidecar,
    )
    require(dataset.feature_dim == 1024, "dataset did not expose full DINO")
    require(dataset.feature_contract == "backbone_native", "dataset contract differs")
    if dataset.teacher_sidecar is not None:
        dataset.teacher_sidecar.verify_hashes()
    return dataset, default_collate([dataset[0]])


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def verify_checkpoint(path: str, model) -> dict:
    if not path:
        return {"checkpoint": "not_provided"}
    require(os.path.isfile(path), "checkpoint is missing")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    require(
        checkpoint.get("checkpoint_version") == CHECKPOINT_VERSION,
        f"checkpoint is not version {CHECKPOINT_VERSION}",
    )
    require(checkpoint.get("config") == model.config.to_dict(), "config differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    return {
        "checkpoint": os.path.abspath(path),
        "checkpoint_sha256": file_sha256(path),
        "checkpoint_phase": checkpoint.get("phase"),
        "checkpoint_phase_step": checkpoint.get("phase_step"),
    }


def representation_weights() -> AdaptiveGaussianLossWeights:
    return AdaptiveGaussianLossWeights(
        future=1.0,
        history=0.5,
        flow=0.0,
        feature=1.0,
        allocator=0.2,
        slot=0.2,
        action=0.0,
        action_specificity=0.0,
        geometry=0.25,
        rgb=0.0,
    )


def verify_causal_paths(model, batch: dict) -> dict[str, float]:
    history = model.encode_history(batch)
    future_scale = signed_gap_scale(batch["future_times"], model.config.gap_reference)
    history_scale = signed_gap_scale(batch["history_times"], model.config.gap_reference)
    prior = model.prior_context(history, future_scale, history_scale)
    _, target = model.encode_targets(batch)
    changed = dict(batch)
    changed["future_features"] = batch["future_features"].roll(1, dims=-1)
    changed_history = model.encode_history(changed)
    changed_prior = model.prior_context(changed_history, future_scale, history_scale)
    _, changed_target = model.encode_targets(changed)
    history_difference = max_difference(history["slots"], changed_history["slots"])
    prior_difference = max_difference(prior, changed_prior)
    target_difference = max_difference(target["slots"], changed_target["slots"])
    require(history_difference == 0.0, "future content reached history encoder")
    require(prior_difference == 0.0, "future content reached prior context")
    require(target_difference > 1e-6, "future target ignored changed future content")
    posterior = posterior_from_targets(
        model, batch, history, target, future_scale, None
    )[0]
    changed_posterior = posterior_from_targets(
        model, changed, history, changed_target, future_scale, None
    )[0]
    posterior_difference = max_difference(posterior, changed_posterior)
    require(posterior_difference > 1e-6, "posterior ignored changed future content")
    require(posterior.shape[-2:] == (4, 32), "latent action is not [4,32]")
    sidecar_difference = 0.0
    if "teacher_sidecar_present" in batch:
        changed_sidecar = dict(batch)
        for name, value in batch.items():
            if name.startswith("teacher_future_") and torch.is_tensor(value):
                changed_sidecar[name] = value.flip(-1)
        sidecar_history = model.encode_history(changed_sidecar)
        sidecar_prior = model.prior_context(
            sidecar_history, future_scale, history_scale
        )
        require(
            max_difference(history["slots"], sidecar_history["slots"]) == 0.0,
            "teacher sidecar reached the history encoder",
        )
        sidecar_difference = max_difference(prior, sidecar_prior)
        require(sidecar_difference == 0.0, "teacher sidecar reached the prior")
    return {
        "history_future_swap_max_difference": history_difference,
        "prior_future_swap_max_difference": prior_difference,
        "target_future_swap_max_difference": target_difference,
        "posterior_future_swap_max_difference": posterior_difference,
        "sidecar_prior_swap_max_difference": sidecar_difference,
    }


def verify_relative_geometry(device: torch.device) -> float:
    center = torch.tensor([[[0.1, -0.2], [0.4, 0.3], [-0.5, 0.2]]], device=device)
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


def verify_parameters(model) -> dict:
    parameters = list(model.named_parameters())
    names = [name for name, _ in parameters]
    require(len(names) == len(set(names)), "model has duplicate parameter names")
    target_trainable = [
        name
        for name, parameter in parameters
        if name.startswith("target_") and parameter.requires_grad
    ]
    require(not target_trainable, "EMA target parameters are trainable")
    legacy = [name for name in names if name.startswith("gaussian_readout.")]
    require(not legacy, "legacy Gaussian parameters remain in v37")
    return {
        "parameter_tensors": len(parameters),
        "parameter_count": sum(parameter.numel() for _, parameter in parameters),
        "trainable_parameter_count": sum(
            parameter.numel() for _, parameter in parameters if parameter.requires_grad
        ),
        "ema_target_trainable_tensors": len(target_trainable),
        "legacy_gaussian_parameter_tensors": len(legacy),
    }


def verify_forward_backward(model, batch: dict) -> tuple[dict, dict]:
    history_mask = torch.zeros(
        batch["history_features"].shape[:3],
        device=batch["history_features"].device,
        dtype=torch.bool,
    )
    result = model(
        batch,
        history_mask=history_mask,
        phase="object_memory_representation_loss",
        loss_weights=representation_weights(),
        collect_diagnostics=True,
    )
    require(bool(torch.isfinite(result["loss"])), "representation loss is not finite")
    require(
        result["rendered_future_features"].shape[-1] == 1024,
        "readout did not preserve full DINO dimension",
    )
    slots = result["history_slot_states"][-1]
    partition = slots.assignment.float().sum(dim=-1) + (
        slots.background_assignment.float()
    )
    partition_error = float((partition - 1.0).abs().max())
    require(partition_error < 1e-5, "foreground/background partition is invalid")
    identity_error = max_difference(
        result["residual_reference_features"],
        batch["history_features"][:, -1:].expand_as(
            result["residual_reference_features"]
        ),
    )
    require(identity_error == 0.0, "current field did not bypass the readout exactly")
    diagnostics = change_residual_readout_diagnostics(batch, result)
    diagnostic_values = {name: float(value) for name, value in diagnostics.items()}
    validate_change_readout_diagnostic_contract(diagnostic_values)
    require(
        CHANGE_READOUT_REQUIRED_DIAGNOSTICS.issubset(diagnostic_values),
        "change readout diagnostic contract is incomplete",
    )
    required_training = {
        "baseline_persistence_future",
        "dynamics_gain_over_persistence",
        "predictive_relative_gain_over_persistence",
        "geometry_relative_scale_gain_over_persistence",
        "memory_existence_brier",
        "token_count_vs_spatial_complexity_correlation",
        *CHANGE_READOUT_REQUIRED_DIAGNOSTICS,
    }
    training_metrics = finalize_diagnostic_metrics(
        {name: float(value.detach()) for name, value in result["parts"].items()}
    )
    missing_training = required_training.difference(training_metrics)
    require(not missing_training, f"missing W&B diagnostics: {missing_training}")
    nonfinite_training = {
        name for name in required_training if not math.isfinite(training_metrics[name])
    }
    require(not nonfinite_training, f"non-finite W&B diagnostics: {nonfinite_training}")
    zero_action_residual = float(
        result["predicted_action_slot_residual"].detach().float().abs().max()
    )
    require(zero_action_residual == 0.0, "zero action produced a residual")
    active_count = result["history_token_states"][-1].active_count.float()
    require(bool((active_count >= 64).all()), "token gate fell below 64")
    require(bool((active_count <= 256).all()), "token gate exceeded 256")

    current_memory = result["history_slot_states"][-1]
    predicted_memory = model.object_memory.predict(
        current_memory,
        batch["future_times"][:, 0] - batch["history_times"][:, -1],
    )
    occluded = replace(
        current_memory, activity=torch.zeros_like(current_memory.activity)
    )
    corrected = model.object_memory.correct(
        predicted_memory, occluded, result["history_token_states"][-1]
    )
    occlusion_difference = max_difference(
        corrected.existence, predicted_memory.existence
    )
    require(occlusion_difference == 0.0, "occlusion deleted object existence")

    result["loss"].backward()
    prefixes = ("allocator.", "object_aggregator.", "dynamics.", "change_readout.")
    counts = {prefix.rstrip("."): 0 for prefix in prefixes}
    nonfinite = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        for prefix in prefixes:
            if name.startswith(prefix):
                counts[prefix.rstrip(".")] += 1
    require(not nonfinite, f"non-finite gradients: {nonfinite}")
    require(
        all(value > 0 for value in counts.values()), f"missing core gradients: {counts}"
    )
    background_gradient = model.object_aggregator.background_score.weight.grad
    require(
        background_gradient is not None, "background assignment received no gradient"
    )
    require(
        bool(torch.isfinite(background_gradient).all()),
        "background assignment gradient is not finite",
    )
    metrics = {
        "representation_loss": float(result["loss"]),
        "foreground_background_partition_max_error": partition_error,
        "current_field_identity_max_error": identity_error,
        "background_gradient_norm": float(background_gradient.float().norm()),
        "gradient_tensors": counts,
        "representation_diagnostic_metrics": len(training_metrics),
        "zero_action_residual_max": zero_action_residual,
        "active_token_count": float(active_count.mean()),
        "occlusion_existence_max_difference": occlusion_difference,
        **diagnostic_values,
    }
    return result, metrics


def main() -> None:
    args = parse_args()
    require(torch.cuda.is_available(), "v37 verifier requires CUDA")
    local_gpus = torch.cuda.device_count()
    require(local_gpus > 0, "no visible CUDA GPU")
    if args.expected_local_gpus != "auto":
        require(
            args.expected_local_gpus.isdigit()
            and int(args.expected_local_gpus) == local_gpus,
            f"visible GPU count is {local_gpus}, expected {args.expected_local_gpus}",
        )
    commit = verify_repository()
    manifest_sha256, _ = verify_manifest(args.data)
    dataset, cpu_batch = build_batch(args)
    device = torch.device("cuda:0")
    config = AdaptiveGaussianWMConfig.object_memory_full(dataset.feature_dim)
    require(config.change_residual_readout, "v37 change readout is disabled")
    require(not config.dense_object_readout, "legacy dense readout is still enabled")
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    require(model.gaussian_readout is None, "legacy Gaussian readout is still present")
    checkpoint = verify_checkpoint(args.checkpoint, model)
    batch = to_device(cpu_batch, device)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if torch.cuda.is_bf16_supported()
        else nullcontext
    )
    model.eval()
    with torch.no_grad(), amp_context():
        causal = verify_causal_paths(model, batch)
    model.train()
    model.zero_grad(set_to_none=True)
    with amp_context():
        _, training = verify_forward_backward(model, batch)
    report = {
        "status": "passed",
        "architecture": "object_memory_v1",
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_contract": "rolling_recovery_v1",
        "readout_backend": READOUT_BACKEND,
        "feature_contract": FEATURE_CONTRACT,
        "feature_dim": dataset.feature_dim,
        "control_hz": float(dataset.control_hz),
        "git_commit": commit,
        "data": os.path.abspath(args.data),
        "data_manifest_sha256": manifest_sha256,
        "teacher_sidecar": "enabled" if args.teacher_sidecar else "disabled",
        "teacher_sidecar_sha256": getattr(dataset, "teacher_sidecar_sha256", ""),
        "local_gpu_count": local_gpus,
        "gpu_policy": args.expected_local_gpus,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "relative_geometry_max_difference": verify_relative_geometry(device),
        "ddp_parameter_contract": verify_parameters(model),
        **checkpoint,
        **causal,
        **training,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
