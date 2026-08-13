"""Server gate for grounded object-state learning v47."""

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
from igsw.adaptive_gaussian_wm.length_metric_window import LengthMetricWindow  # noqa: E402
from igsw.adaptive_gaussian_wm.observation_complete_world_model import (  # noqa: E402
    ObservationCompleteWorldModel,
)
from igsw.adaptive_gaussian_wm.temporal_object_dataset import (  # noqa: E402
    TEMPORAL_OBJECT_VIDEO_CONTRACT, TemporalObjectVideoDataset,
)
from igsw.adaptive_gaussian_wm.v47_config import (  # noqa: E402
    ARCHITECTURE, CHECKPOINT_VERSION, ObservationCompleteConfig,
)
from igsw.adaptive_gaussian_wm.v47_curriculum import curriculum_at  # noqa: E402


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


def _state_at(output: dict, index: int) -> dict[str, torch.Tensor]:
    return {
        name: output["full_state"][name][:, index]
        for name in ("semantic", "dynamic", "center", "log_scale", "presence", "visibility")
    }


def _group_gradient_norms(model) -> dict[str, float]:
    groups = {"state": [], "effect": [], "goal": []}
    missing, nonfinite = [], []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        group = (
            "state" if name.startswith("state_encoder.") else
            "goal" if name.startswith("goal_effect_predictor.") else "effect"
        )
        groups[group].append(parameter.grad.detach().float().square().sum())
    require(not missing, f"v47 has unused parameters: {missing}")
    require(not nonfinite, f"v47 has non-finite gradients: {nonfinite}")
    return {
        name: float(torch.stack(values).sum().sqrt())
        for name, values in groups.items()
    }


def gradient_contract(model, features, batch, amp_context) -> dict[str, float]:
    result = {}
    stages = {
        "state": (0, {"state"}),
        "effect": (model.config.state_phase_steps, {"effect"}),
        "goal": (model.config.goal_phase_steps, {"effect", "goal"}),
    }
    for stage, (step, active) in stages.items():
        model.zero_grad(set_to_none=True)
        with amp_context():
            output = model(
                features.patches, features.coordinates, features.valid,
                batch["frame_times"], batch["observation_mask"], step,
            )
        require(bool(torch.isfinite(output["loss"])), f"{stage} loss is non-finite")
        output["loss"].backward()
        norms = _group_gradient_norms(model)
        for group, norm in norms.items():
            if group in active:
                require(norm > 0.0, f"{stage} did not train {group}")
            else:
                require(norm == 0.0, f"{stage} leaked gradients into {group}")
            result[f"{stage}_{group}_gradient_norm"] = norm
        if stage == "state":
            result["state_loss"] = float(output["loss"].detach())
            result["observation_query_count"] = float(
                output["parts"]["observation_query_count"].detach()
            )
    model.zero_grad(set_to_none=True)
    return result


@torch.no_grad()
def structural_contract(model, features, batch, amp_context) -> dict[str, float]:
    with amp_context():
        output = model(
            features.patches, features.coordinates, features.valid,
            batch["frame_times"], batch["observation_mask"], 0,
        )
    state = output["full_state"]
    owner_mass = state["assignment"].float().sum(dim=2) + state["scene_assignment"].float()
    partition_error = maximum_difference(owner_mass, features.valid.float())
    scene_fraction = float(
        (state["scene_assignment"].float() * features.valid.float()).sum()
        / features.valid.float().sum().clamp_min(1.0)
    )
    require(partition_error < 1e-5, "object plus scene owners do not partition patches")
    require(state["assignment"].shape[2] == model.config.object_slots, "object count differs")
    require(state["scene_coefficients"].shape[-2] == 6, "scene decoder is not low-rank")
    require(
        bool((state["observed_presence"] >= state["observed_visibility"]).all()),
        "observed presence does not preserve temporarily hidden tracks",
    )
    require(
        bool((state["presence"] + 1e-6 >= state["visibility"]).all()),
        "world-state visibility exceeds presence",
    )
    require(
        float(output["parts"]["observation_query_count"]) == model.config.observation_queries,
        "state objective does not supervise the configured patch query count",
    )
    require(
        all(bool(torch.isfinite(value)) for value in output["parts"].values()),
        "v47 diagnostics contain non-finite values",
    )
    identity_difference = maximum_difference(
        state["predicted_identity"], state["observation_identity"]
    )
    require(
        identity_difference > 1e-6,
        "track-observation retrieval collapsed to a same-tensor comparison",
    )
    require(
        float(output["parts"]["loss_masked_state"]) > 0.0,
        "masked compact-state supervision is inactive",
    )
    return {
        "owner_partition_max_error": partition_error,
        "scene_owner_fraction": scene_fraction,
        "effective_object_count": float(output["parts"]["object_effective_count"]),
        "observation_error": float(output["parts"]["loss_observation_complete"]),
        "scene_only_error": float(output["parts"]["diagnostic_scene_only_error"]),
        "masked_state_loss": float(output["parts"]["loss_masked_state"]),
        "supported_object_count": float(output["parts"]["object_supported_count"]),
        "object_owner_fraction": float(output["parts"]["object_owner_fraction"]),
        "presence_visibility_gap": float(output["parts"]["presence_visibility_gap"]),
        "track_observation_identity_max_difference": identity_difference,
    }


@torch.no_grad()
def masked_observation_contract(model, features, batch, amp_context) -> dict[str, float]:
    mask = batch["observation_mask"].clone()
    index = mask.shape[1] // 2
    mask[:, index] = False
    altered = features.patches.clone()
    altered[:, index] = torch.roll(altered[:, index], shifts=1, dims=1)
    with amp_context():
        reference = model.state_encoder(
            features.patches, features.coordinates, features.valid,
            batch["frame_times"], mask,
        )
        changed = model.state_encoder(
            altered, features.coordinates, features.valid, batch["frame_times"], mask,
        )
    differences = [
        maximum_difference(reference[name][:, index], changed[name][:, index])
        for name in ("semantic", "dynamic", "center", "log_scale", "presence", "visibility", "scene")
    ]
    difference = max(differences)
    require(difference < 1e-6, "masked state reads the hidden frame")
    return {"masked_frame_content_swap_max_difference": difference}


@torch.no_grad()
def causal_contract(model, features, batch, amp_context) -> dict[str, float]:
    changed = features.patches.clone()
    midpoint = changed.shape[1] // 2
    changed[:, midpoint:] = changed[:, midpoint:].flip(2)
    with amp_context():
        reference = model(
            features.patches, features.coordinates, features.valid,
            batch["frame_times"], batch["observation_mask"], model.config.goal_phase_steps,
        )
        altered = model(
            changed, features.coordinates, features.valid,
            batch["frame_times"], batch["observation_mask"], model.config.goal_phase_steps,
        )
    prefix = maximum_difference(
        reference["full_state"]["dynamic"][:, :midpoint],
        altered["full_state"]["dynamic"][:, :midpoint],
    )
    posterior = maximum_difference(reference["trajectory_effect"], altered["trajectory_effect"])
    goal = maximum_difference(reference["goal_effect"], altered["goal_effect"])
    require(prefix < 1e-6, "future content changed the causal state prefix")
    require(posterior > 1e-6, "future content did not change the posterior")
    require(goal > 1e-6, "future content did not change the image-goal effect")
    return {
        "future_swap_history_prefix_max_difference": prefix,
        "future_swap_posterior_max_difference": posterior,
        "future_swap_goal_effect_max_difference": goal,
    }


@torch.no_grad()
def zero_effect_contract(model, features, batch, amp_context) -> dict[str, float]:
    with amp_context():
        output = model(
            features.patches, features.coordinates, features.valid,
            batch["frame_times"], batch["observation_mask"], 0,
        )
    current = _state_at(output, 0)
    zero = torch.zeros(
        len(features.patches), model.config.action_tokens, model.config.action_dim,
        device=features.patches.device,
    )
    prediction = model.dynamics(
        current, zero, batch["frame_times"][:, -1] - batch["frame_times"][:, 0]
    )
    error = max(maximum_difference(prediction[name], current[name]) for name in current)
    require(error == 0.0, "zero effect is not an exact identity")
    return {"zero_effect_identity_max_error": error}


def metric_window_contract() -> dict[str, float]:
    window = LengthMetricWindow()
    lengths = (8, 16, 24, 32) * 5
    for length in lengths:
        window.add(
            {
                "loss_state": float(length),
                "track_observation_retrieval_top1": 1.0 / length,
            },
            float(length),
            1.0,
            0.8,
        )
    summary = window.summarize_and_reset()
    require(summary["history_length_coverage"] == 4.0, "history telemetry lost a length")
    require(summary["metric_window_microbatches"] == 20.0, "history telemetry window differs")
    for length in (8, 16, 24, 32):
        require(
            summary[f"history_h{length}_updates"] == 5.0,
            f"history h{length} telemetry count differs",
        )
        require(
            summary[f"history_h{length}_loss_state"] == float(length),
            f"history h{length} telemetry value differs",
        )
    return {
        "history_length_coverage": summary["history_length_coverage"],
        "history_metric_window_microbatches": summary["metric_window_microbatches"],
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v47 verifier requires a visible CUDA device")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    config = ObservationCompleteConfig()
    dataset = TemporalObjectVideoDataset(
        args.data, "train", max_items=64, seed=args.seed, record_manifest_hash=False
    )
    samples = [dataset[(index, args.chunk_length)] for index in range(2)]
    forbidden = {"instruction", "condition_feature", "teacher_sidecar", "segmentation", "action", "dino"}
    require(not forbidden.intersection(samples[0]), "v47 dataset exposed forbidden supervision")
    require(
        any(not bool(sample["observation_mask"][1:-1].all()) for sample in samples),
        "v47 dataset did not provide a hidden interior observation",
    )
    batch = default_collate(samples)
    batch = {
        name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }
    batch["observation_mask"][:, args.chunk_length // 2] = False
    encoder = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    require(
        not any(parameter.requires_grad for parameter in encoder.backbone.parameters()),
        "v47 DINO teacher is not frozen",
    )
    features = encoder(batch)
    model = ObservationCompleteWorldModel(config).to(device).train()
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16" else nullcontext
    )
    structural = structural_contract(model, features, batch, amp_context)
    gradients = gradient_contract(model, features, batch, amp_context)
    model.eval()
    masked = masked_observation_contract(model, features, batch, amp_context)
    causal = causal_contract(model, features, batch, amp_context)
    identity = zero_effect_contract(model, features, batch, amp_context)
    telemetry = metric_window_contract()
    curricula = {
        step: curriculum_at(step, config)
        for step in (0, config.state_phase_steps, config.goal_phase_steps)
    }
    require(curricula[0].state_weight == 1.0, "v47 initial state stage differs")
    require(curricula[config.state_phase_steps].state_weight == 0.0, "v47 state does not freeze")
    require(curricula[config.state_phase_steps].effect_weight > 0.0, "v47 effect ramp does not start")
    require(curricula[config.goal_phase_steps].goal_weight > 0.0, "v47 goal ramp does not start")
    report = {
        "status": "passed", "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE, "contract": TEMPORAL_OBJECT_VIDEO_CONTRACT,
        "git_commit": args.source_revision, "data": os.path.abspath(args.data),
        "historical_checkpoint_used": False, "teacher_sidecar_used": False,
        "language_used": False, "explicit_action_used": False,
        "instance_segmentation_used": False, "fixed_patch_correspondence_used": False,
        "dino_model": config.dino_model_name, "dino_fully_frozen": True,
        "dino_checkpoint": os.path.abspath(args.dino_checkpoint),
        "object_slots": config.object_slots, "scene_is_explicit_owner": True,
        "presence_visibility_are_separate": True,
        "identity_target": "predicted_track_to_aligned_observation",
        "masked_compact_state_supervision": True,
        "object_grounding_supervision": True,
        "state_freezes_before_effect_learning": True,
        **structural, **gradients, **masked, **causal, **identity, **telemetry,
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
