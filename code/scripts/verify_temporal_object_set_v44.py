"""Server-side executable contract gate for the v44 Temporal Object Set model."""

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
from igsw.adaptive_gaussian_wm.temporal_object_dataset import (  # noqa: E402
    TEMPORAL_OBJECT_VIDEO_CONTRACT,
    TemporalObjectVideoDataset,
)
from igsw.adaptive_gaussian_wm.temporal_object_set_model import (  # noqa: E402
    TemporalObjectSetWorldModel,
)
from igsw.adaptive_gaussian_wm.v44_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    TemporalObjectSetConfig,
)
from igsw.adaptive_gaussian_wm.v44_curriculum import curriculum_at  # noqa: E402


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


def endpoint_identity_error(model, output, frame_times) -> dict[str, float]:
    current = {
        name: output["full_state"][name][:, 0, : model.config.object_slots]
        for name in (
            "semantic",
            "dynamic",
            "center",
            "log_scale",
            "presence",
            "visibility",
        )
    }
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
    require(
        max(errors.values()) == 0.0, "zero effect is not an exact identity transition"
    )
    return errors


def gradient_contract(model, output) -> dict[str, int | float]:
    model.zero_grad(set_to_none=True)
    output["loss"].backward()
    groups = {"tokenizer": [], "posterior": [], "dynamics": [], "goal": []}
    nonfinite = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(
            torch.isfinite(parameter.grad).all()
        ):
            nonfinite.append(name)
        if name.startswith("tokenizer."):
            groups["tokenizer"].append(parameter)
        elif name.startswith("effect_posterior."):
            groups["posterior"].append(parameter)
        elif name.startswith("dynamics."):
            groups["dynamics"].append(parameter)
        elif name.startswith("goal_effect_predictor."):
            groups["goal"].append(parameter)
    require(not nonfinite, f"v44 has non-finite gradients: {nonfinite}")
    for name, parameters in groups.items():
        require(
            any(parameter.grad is not None for parameter in parameters),
            f"{name} has no gradient",
        )
    result = {
        f"{name}_gradient_tensors": sum(
            parameter.grad is not None for parameter in parameters
        )
        for name, parameters in groups.items()
    }
    result["loss"] = float(output["loss"].detach())
    model.zero_grad(set_to_none=True)
    return result


@torch.no_grad()
def causal_contract(model, features, batch, amp_context) -> dict[str, float]:
    changed = features.patches.clone()
    midpoint = changed.shape[1] // 2
    changed[:, midpoint:] = changed[:, midpoint:].flip(2)
    with amp_context():
        altered = model(
            changed,
            features.coordinates,
            features.valid,
            batch["frame_times"],
            batch["observation_mask"],
            35_000,
        )
    with amp_context():
        reference = model(
            features.patches,
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
    require(prefix < 1e-6, "future patches changed the causal object-state prefix")
    require(posterior > 1e-6, "future patches did not change the effect posterior")
    return {
        "future_swap_history_prefix_max_difference": prefix,
        "future_swap_posterior_max_difference": posterior,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v44 verifier requires a visible CUDA device")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    config = TemporalObjectSetConfig()
    dataset = TemporalObjectVideoDataset(
        args.data,
        "train",
        max_items=64,
        seed=args.seed,
    )
    sample = dataset[(0, args.chunk_length)]
    forbidden = {
        "instruction",
        "condition_feature",
        "teacher_sidecar",
        "segmentation",
        "action",
        "dino",
    }
    require(
        not forbidden.intersection(sample), "v44 dataset exposed forbidden supervision"
    )
    batch = default_collate([sample])
    batch = {
        name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }
    batch["observation_mask"][:, args.chunk_length // 2] = False
    encoder = FrozenDinoVideoRuntime(config, device, args.amp, args.dino_frame_batch)
    require(
        not any(parameter.requires_grad for parameter in encoder.backbone.parameters()),
        "standard pretrained DINO is not fully frozen",
    )
    features = encoder(batch)
    model = TemporalObjectSetWorldModel(config).to(device).train()
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
    require(bool(torch.isfinite(output["loss"])), "v44 verifier loss is non-finite")
    require(
        all(bool(torch.isfinite(value)) for value in output["parts"].values()),
        "v44 verifier metrics contain non-finite values",
    )
    assignment = output["full_state"]["assignment"].float()
    assignment_mass_error = float(
        (assignment.sum(dim=2) - features.valid.float()).abs().max()
    )
    require(
        assignment_mass_error < 1e-4,
        "competitive assignment does not partition patches",
    )
    require(output["short_effect"].shape == (1, 4, 32), "short effect shape differs")
    require(output["goal_effect"].shape == (1, 4, 32), "goal effect shape differs")
    gradients = gradient_contract(model, output)
    identity = endpoint_identity_error(model, output, batch["frame_times"])
    model.eval()
    causal = causal_contract(model, features, batch, amp_context)
    curricula = {
        step: curriculum_at(step, config)
        for step in (0, 10_000, 12_000, 30_000, 32_000)
    }
    require(
        curricula[0].effect_weight == 0.0 and curricula[0].goal_weight == 0.0,
        "v44 object stage enables later objectives",
    )
    require(curricula[12_000].effect_weight == 1.0, "v44 effect ramp differs")
    require(curricula[32_000].goal_weight == 1.0, "v44 goal ramp differs")
    report = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "contract": TEMPORAL_OBJECT_VIDEO_CONTRACT,
        "git_commit": git_commit(),
        "data": os.path.abspath(args.data),
        "data_manifest_sha256": dataset.data_sha256,
        "historical_checkpoint_used": False,
        "teacher_sidecar_used": False,
        "language_used": False,
        "explicit_action_used": False,
        "dino_model": config.dino_model_name,
        "dino_image_size": config.dino_image_size,
        "dino_fully_frozen": True,
        "chunk_shape": list(batch["video_rgb"].shape),
        "patch_shape": list(features.patches.shape),
        "object_slots": config.object_slots,
        "role_slots": {"scene": 1, "transient": 1},
        "assignment_mass_error": assignment_mass_error,
        "zero_effect_identity_error": identity,
        **gradients,
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
