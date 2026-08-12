"""Executable server contract gate for predictive Object Tube JEPA v45."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import subprocess
import sys

import torch
from torch.utils.data import default_collate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.frozen_video_encoder import (  # noqa: E402
    FrozenDinoVideoRuntime,
)
from igsw.adaptive_gaussian_wm.predictive_object_tube_model import (  # noqa: E402
    PredictiveObjectTubeWorldModel,
)
from igsw.adaptive_gaussian_wm.temporal_object_dataset import (  # noqa: E402
    TEMPORAL_OBJECT_VIDEO_CONTRACT,
    TemporalObjectVideoDataset,
)
from igsw.adaptive_gaussian_wm.v45_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    PredictiveObjectTubeConfig,
)
from igsw.adaptive_gaussian_wm.v45_curriculum import curriculum_at  # noqa: E402
from igsw.adaptive_gaussian_wm.video_correspondence import (  # noqa: E402
    build_video_correspondence,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def maximum_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dino_frame_batch", type=int, default=16)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--chunk_length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def _state_at(output: dict, index: int, count: int) -> dict[str, torch.Tensor]:
    return {
        name: output["full_state"][name][:, index, :count]
        for name in (
            "semantic",
            "dynamic",
            "center",
            "log_scale",
            "presence",
            "visibility",
        )
    }


def endpoint_identity_error(model, output, frame_times) -> dict[str, float]:
    current = _state_at(output, 0, model.config.object_slots)
    zero = torch.zeros(
        len(frame_times),
        model.config.action_tokens,
        model.config.action_dim,
        device=frame_times.device,
    )
    identity = model.dynamics(current, zero, frame_times[:, -1] - frame_times[:, 0])
    errors = {
        name: maximum_difference(identity[name], current[name]) for name in current
    }
    require(max(errors.values()) == 0.0, "zero effect is not an exact identity")
    return errors


def gradient_contract(model, output) -> dict[str, int | float]:
    model.zero_grad(set_to_none=True)
    output["loss"].backward()
    prefixes = {
        "tokenizer": "tokenizer.",
        "posterior": "effect_posterior.",
        "dynamics": "dynamics.",
        "image_goal": "goal_effect_predictor.",
    }
    groups = {name: [] for name in prefixes}
    nonfinite, missing = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is None:
            missing.append(name)
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        for group, prefix in prefixes.items():
            if name.startswith(prefix):
                groups[group].append(parameter)
                break
    require(not nonfinite, f"v45 has non-finite gradients: {nonfinite}")
    require(not missing, f"v45 has unused trainable parameters: {missing}")
    result = {}
    for name, parameters in groups.items():
        count = sum(parameter.grad is not None for parameter in parameters)
        require(count == len(parameters), f"{name} has missing gradients")
        result[f"{name}_gradient_tensors"] = count
    result["loss"] = float(output["loss"].detach())
    model.zero_grad(set_to_none=True)
    return result


@torch.no_grad()
def causal_and_goal_contract(model, features, batch, amp_context) -> dict[str, float]:
    changed = features.patches.clone()
    midpoint = changed.shape[1] // 2
    changed[:, midpoint:] = changed[:, midpoint:].flip(2)
    with amp_context():
        reference = model(
            features.patches,
            features.coordinates,
            features.valid,
            batch["frame_times"],
            batch["observation_mask"],
            35_000,
        )
        altered = model(
            changed,
            features.coordinates,
            features.valid,
            batch["frame_times"],
            batch["observation_mask"],
            35_000,
        )
    prefix = maximum_difference(
        reference["full_state"]["dynamic"][:, :midpoint],
        altered["full_state"]["dynamic"][:, :midpoint],
    )
    posterior = maximum_difference(
        reference["trajectory_effect"], altered["trajectory_effect"]
    )
    goal = maximum_difference(reference["goal_effect"], altered["goal_effect"])
    goal_state = maximum_difference(
        reference["goal_target"]["dynamic"], altered["goal_target"]["dynamic"]
    )
    require(prefix < 1e-6, "future patches changed the causal object-state prefix")
    require(posterior > 1e-6, "future patches did not change the video posterior")
    require(goal_state > 1e-6, "future image did not change the image-goal state")
    require(goal > 1e-6, "future image did not change the image-goal effect")
    return {
        "future_swap_history_prefix_max_difference": prefix,
        "future_swap_posterior_max_difference": posterior,
        "future_swap_goal_state_max_difference": goal_state,
        "future_swap_image_goal_effect_max_difference": goal,
    }


def low_mass_contract(model, features, batch, amp_context) -> dict[str, float]:
    patches = features.patches[:1, :2]
    coordinates = features.coordinates[:1, :2]
    valid = torch.zeros_like(features.valid[:1, :2])
    times = batch["frame_times"][:1, :2]
    observed = torch.ones(1, 2, dtype=torch.bool, device=patches.device)
    correspondence = build_video_correspondence(
        patches.detach(),
        coordinates,
        valid,
        model.config.correspondence_temperature,
        model.config.correspondence_spatial_sigma,
    )
    model.zero_grad(set_to_none=True)
    with amp_context():
        state = model.tokenizer(
            patches, coordinates, valid, times, observed, correspondence
        )
        probe = state["semantic"].float().square().mean()
    probe.backward()
    maximum_gradient = max(
        float(parameter.grad.float().abs().max())
        for parameter in model.tokenizer.parameters()
        if parameter.grad is not None
    )
    correction = float(state["correction_gate"].float().abs().max())
    require(correction == 0.0, "zero-support slots received observation correction")
    require(maximum_gradient < 100.0, "low-mass slot update has an unstable gradient")
    require(bool(torch.isfinite(probe)), "low-mass slot state is non-finite")
    model.zero_grad(set_to_none=True)
    return {
        "zero_support_correction_gate": correction,
        "zero_support_max_gradient": maximum_gradient,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v45 verifier requires a visible CUDA device")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    config = PredictiveObjectTubeConfig()
    dataset = TemporalObjectVideoDataset(
        args.data, "train", max_items=64, seed=args.seed, record_manifest_hash=False
    )
    sample_a = dataset[(0, args.chunk_length)]
    sample_b = dataset[(1, args.chunk_length)]
    forbidden = {
        "instruction",
        "condition_feature",
        "teacher_sidecar",
        "segmentation",
        "action",
        "dino",
    }
    require(not forbidden.intersection(sample_a), "v45 dataset exposed supervision")
    batch = default_collate([sample_a, sample_b])
    batch = {
        name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }
    batch["observation_mask"][:, args.chunk_length // 2] = False
    encoder = FrozenDinoVideoRuntime(config, device, args.amp, args.dino_frame_batch)
    require(
        not any(parameter.requires_grad for parameter in encoder.backbone.parameters()),
        "pretrained DINO is not frozen",
    )
    features = encoder(batch)
    model = PredictiveObjectTubeWorldModel(config).to(device).train()
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    with amp_context():
        output = model(
            features.patches,
            features.coordinates,
            features.valid,
            batch["frame_times"],
            batch["observation_mask"],
            35_000,
        )
    require(bool(torch.isfinite(output["loss"])), "v45 verifier loss is non-finite")
    require(
        all(bool(torch.isfinite(value)) for value in output["parts"].values()),
        "v45 verifier metrics contain non-finite values",
    )
    assignment = output["full_state"]["assignment"].float()
    mass_error = float((assignment.sum(dim=2) - features.valid.float()).abs().max())
    require(mass_error < 1e-4, "object assignment does not partition patches")
    require(assignment.shape[2] == config.object_slots, "scene competes in assignment")
    require(
        output["full_state"]["tube_prediction"].shape
        == (2, args.chunk_length - 1, config.object_slots, config.patch_dim),
        "predictive tube shape differs",
    )
    require(output["short_effect"].shape == (2, 4, 32), "effect shape differs")
    effect_norm = float(output["short_effect"].float().norm(dim=-1).mean())
    require(effect_norm < 0.95, "fresh v45 effect is already unit-norm saturated")
    gradients = gradient_contract(model, output)
    identity = endpoint_identity_error(model, output, batch["frame_times"])
    low_mass = low_mass_contract(model, features, batch, amp_context)
    model.eval()
    causal = causal_and_goal_contract(model, features, batch, amp_context)
    curricula = {
        step: curriculum_at(step, config)
        for step in (0, 10_000, 12_000, 30_000, 32_000)
    }
    require(curricula[0].effect_weight == 0.0, "object stage enables effects")
    require(curricula[12_000].effect_weight == 1.0, "effect ramp differs")
    require(curricula[32_000].goal_weight == 1.0, "image-goal ramp differs")
    report = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "contract": TEMPORAL_OBJECT_VIDEO_CONTRACT,
        "git_commit": git_commit(),
        "data": os.path.abspath(args.data),
        "historical_checkpoint_used": False,
        "teacher_sidecar_used": False,
        "language_used": False,
        "explicit_action_used": False,
        "dino_model": config.dino_model_name,
        "dino_fully_frozen": True,
        "object_slots": config.object_slots,
        "scene_is_separate_context": True,
        "transient_is_visibility_confidence": True,
        "assignment_mass_error": mass_error,
        "fresh_effect_mean_norm": effect_norm,
        "zero_effect_identity_error": identity,
        **gradients,
        **low_mass,
        **causal,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = f"{output_path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output_path)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
