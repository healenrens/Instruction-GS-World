"""GPU contract verifier for trajectory-anchored object-state learning."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys

import torch
from torch.utils.data import default_collate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.frozen_video_encoder import FrozenDinoVideoRuntime  # noqa: E402
from igsw.adaptive_gaussian_wm.gradient_health import clip_finite_grad_norm_  # noqa: E402
from igsw.adaptive_gaussian_wm.trajectory_object_dataset import (  # noqa: E402
    TRAJECTORY_OBJECT_VIDEO_CONTRACT,
    TrajectoryObjectVideoDataset,
)
from igsw.adaptive_gaussian_wm.trajectory_object_world_model import (  # noqa: E402
    TrajectoryObjectWorldModel,
    _frame_state,
)
from igsw.adaptive_gaussian_wm.trajectory_teacher import (  # noqa: E402
    build_trajectory_evidence,
)
from igsw.adaptive_gaussian_wm.v49_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    TrajectoryObjectStateConfig,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def maximum_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--dino_frame_batch", type=int, default=16)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--chunk_length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def build_evidence(config, features):
    return build_trajectory_evidence(
        features.patches,
        features.coordinates,
        features.valid,
        config.correspondence_temperature,
        config.correspondence_spatial_sigma,
        config.trajectory_confidence_floor,
    )


def gradient_contract(model, features, batch, evidence, amp_context) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    with amp_context():
        output = model(
            features.patches,
            features.coordinates,
            features.valid,
            batch["frame_times"],
            batch["observation_mask"],
            evidence,
            1.0,
        )
    require(bool(torch.isfinite(output["loss"])), "v49 loss is non-finite")
    output["loss"].backward()
    missing, nonfinite, gradients = [], [], []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        gradients.append(parameter.grad.detach().float().square().sum())
    require(not missing, f"v49 has unused parameters: {missing}")
    require(not nonfinite, f"v49 has non-finite gradients: {nonfinite}")
    norm = torch.stack(gradients).sum().sqrt()
    require(float(norm) > 0.0, "v49 produced zero gradient norm")
    preclip = clip_finite_grad_norm_(model.named_parameters(), 5.0)
    metrics = {
        "loss": float(output["loss"].detach()),
        "gradient_norm": float(norm),
        "preclip_gradient_norm": float(preclip),
        **{name: float(value) for name, value in output["parts"].items()},
    }
    model.zero_grad(set_to_none=True)
    return metrics


@torch.no_grad()
def structural_contract(model, features, batch, evidence, amp_context) -> dict[str, float]:
    with amp_context():
        output = model(
            features.patches,
            features.coordinates,
            features.valid,
            batch["frame_times"],
            batch["observation_mask"],
            evidence,
            0.0,
        )
    batch_size, frames, patches = features.valid.shape
    slots = model.config.object_slots
    state = output["full_state"]
    require(
        state["identity"].shape
        == (batch_size, frames, slots, model.config.identity_dim),
        "v49 identity state shape differs",
    )
    require(
        state["dynamic"].shape
        == (batch_size, frames, slots, model.config.dynamic_dim),
        "v49 dynamic state shape differs",
    )
    require(
        state["center"].shape == (batch_size, frames, slots, 2),
        "v49 geometry state shape differs",
    )
    expected_assignment = (batch_size, frames, patches, model.config.owner_count)
    require(
        state["assignment"].shape == expected_assignment,
        "v49 encoder owner assignment shape differs",
    )
    require(
        output["decoder_assignment"].shape == expected_assignment,
        "v49 decoder owner assignment shape differs",
    )
    encoder_partition = state["assignment"].sum(dim=-1)
    decoder_partition = output["decoder_assignment"].sum(dim=-1)
    encoder_error = maximum_difference(
        encoder_partition[features.valid], torch.ones_like(encoder_partition[features.valid])
    )
    decoder_error = maximum_difference(
        decoder_partition[features.valid], torch.ones_like(decoder_partition[features.valid])
    )
    require(encoder_error < 1e-5, "v49 encoder owners do not partition patches")
    require(decoder_error < 1e-5, "v49 decoder owners do not partition patches")
    require(
        output["effect"].shape
        == (batch_size, model.config.effect_factors, model.config.effect_dim),
        "v49 latent effect shape differs",
    )
    return {
        "encoder_owner_partition_max_error": encoder_error,
        "decoder_owner_partition_max_error": decoder_error,
        "object_slots": float(slots),
        "owner_count": float(model.config.owner_count),
        "patch_count": float(patches),
    }


@torch.no_grad()
def causal_contract(model, features, batch, evidence, amp_context) -> dict[str, float]:
    midpoint = features.patches.shape[1] // 2
    changed_patches = features.patches.clone()
    changed_patches[:, midpoint:] = changed_patches[:, midpoint:].flip(2)
    changed_evidence = build_trajectory_evidence(
        changed_patches,
        features.coordinates,
        features.valid,
        model.config.correspondence_temperature,
        model.config.correspondence_spatial_sigma,
        model.config.trajectory_confidence_floor,
    )
    with amp_context():
        reference = model(
            features.patches,
            features.coordinates,
            features.valid,
            batch["frame_times"],
            batch["observation_mask"],
            evidence,
            0.0,
        )
        changed = model(
            changed_patches,
            features.coordinates,
            features.valid,
            batch["frame_times"],
            batch["observation_mask"],
            changed_evidence,
            0.0,
        )
    prefix = maximum_difference(
        reference["full_state"]["identity"][:, :midpoint],
        changed["full_state"]["identity"][:, :midpoint],
    )
    require(prefix < 1e-6, "future content changed the causal v49 state prefix")
    hidden = ~batch["observation_mask"]
    masked_changed = features.patches.clone()
    masked_changed[hidden] = masked_changed[hidden].flip(1)
    masked_evidence = build_trajectory_evidence(
        masked_changed,
        features.coordinates,
        features.valid,
        model.config.correspondence_temperature,
        model.config.correspondence_spatial_sigma,
        model.config.trajectory_confidence_floor,
    )
    with amp_context():
        hidden_changed = model(
            masked_changed,
            features.coordinates,
            features.valid,
            batch["frame_times"],
            batch["observation_mask"],
            masked_evidence,
            0.0,
        )
    masked_difference = maximum_difference(
        reference["masked_state"]["identity"],
        hidden_changed["masked_state"]["identity"],
    )
    require(masked_difference < 1e-6, "hidden RGB changed the masked v49 state")
    evidence_difference = maximum_difference(evidence.forward, changed_evidence.forward)
    require(evidence_difference > 1e-6, "v49 trajectory teacher ignores changed video")
    return {
        "future_swap_prefix_max_difference": prefix,
        "hidden_content_state_max_difference": masked_difference,
        "trajectory_teacher_future_swap_difference": evidence_difference,
    }


@torch.no_grad()
def effect_contract(model, features, batch, amp_context) -> dict[str, float]:
    state = model.state_encoder(
        features.patches,
        features.coordinates,
        features.valid,
        batch["frame_times"],
        torch.ones_like(batch["observation_mask"]),
    )
    source = _frame_state(state, 0)
    target = _frame_state(state, -1)
    changed_target = {name: value.clone() for name, value in target.items()}
    changed_target["dynamic"] = changed_target["dynamic"].roll(1, dims=1)
    effect = model.effect_posterior(source, target)
    changed_effect = model.effect_posterior(source, changed_target)
    posterior_difference = maximum_difference(effect, changed_effect)
    require(posterior_difference > 1e-6, "v49 posterior ignores target object state")
    delta = batch["frame_times"][:, -1] - batch["frame_times"][:, 0]
    zero = model.dynamics(source, torch.zeros_like(effect), delta)
    zero_difference = max(
        maximum_difference(zero[name], source[name]) for name in source
    )
    require(zero_difference < 1e-6, "zero latent effect changes v49 object state")
    return {
        "posterior_target_swap_max_difference": posterior_difference,
        "zero_effect_state_max_difference": zero_difference,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v49 verifier requires a visible CUDA device")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    config = TrajectoryObjectStateConfig()
    dataset = TrajectoryObjectVideoDataset(
        args.data,
        "train",
        max_items=64,
        seed=args.seed,
        record_manifest_hash=False,
    )
    samples = [dataset[(index, args.chunk_length)] for index in range(2)]
    forbidden = {
        "instruction",
        "condition_feature",
        "teacher_sidecar",
        "segmentation",
        "action",
        "dino",
    }
    require(not forbidden.intersection(samples[0]), "v49 dataset exposed supervision")
    batch = default_collate(samples)
    batch = {
        name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }
    encoder = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    require(
        not any(parameter.requires_grad for parameter in encoder.backbone.parameters()),
        "v49 DINO teacher is not frozen",
    )
    features = encoder(batch)
    evidence = build_evidence(config, features)
    require(bool((evidence.confidence > 0).any()), "v49 found no confident trajectories")
    model = TrajectoryObjectWorldModel(config).to(device).train()
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    structural = structural_contract(model, features, batch, evidence, amp_context)
    gradients = gradient_contract(model, features, batch, evidence, amp_context)
    model.eval()
    causal = causal_contract(model, features, batch, evidence, amp_context)
    effect = effect_contract(model, features, batch, amp_context)
    report = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "contract": TRAJECTORY_OBJECT_VIDEO_CONTRACT,
        "git_commit": args.source_revision,
        "data": os.path.abspath(args.data),
        "historical_checkpoint_used": False,
        "trajectory_teacher": "frozen_dino_cycle_consistent_patch_transport",
        "language_used": False,
        "explicit_action_used": False,
        "instance_segmentation_used": False,
        "dino_fully_frozen": True,
        "core_objective": "trajectory_assignment_plus_disentangled_object_state",
        **structural,
        **gradients,
        **causal,
        **effect,
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

