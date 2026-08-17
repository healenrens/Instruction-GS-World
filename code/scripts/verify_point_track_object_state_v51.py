"""GPU verifier for the v51 point-track Object State contract."""

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
from igsw.adaptive_gaussian_wm.point_track_alignment import match_trajectory_teacher_to_student  # noqa: E402
from igsw.adaptive_gaussian_wm.point_track_dataset import (  # noqa: E402
    POINT_TRACK_VIDEO_CONTRACT,
    PointTrackObjectVideoDataset,
)
from igsw.adaptive_gaussian_wm.point_track_teacher import (  # noqa: E402
    FrozenPointTrackerRuntime,
    PointTrackEvidence,
)
from igsw.adaptive_gaussian_wm.point_track_world_model import (  # noqa: E402
    PointTrackObjectWorldModel,
    frame_state,
)
from igsw.adaptive_gaussian_wm.trajectory_component_teacher import build_trajectory_component_teacher  # noqa: E402
from igsw.adaptive_gaussian_wm.trajectory_lifecycle import (  # noqa: E402
    LIFECYCLE_ABSENT,
    LIFECYCLE_OCCLUDED,
)
from igsw.adaptive_gaussian_wm.v51_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    PointTrackObjectStateConfig,
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
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--dino_frame_batch", type=int, default=16)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--chunk_length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def finite_gradient_contract(model, output, label: str) -> dict[str, float]:
    output["loss"].backward()
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    missing, nonfinite, gradients = [], [], []
    for name, parameter in trainable:
        if parameter.grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        gradients.append(parameter.grad.detach().float().square().sum())
    require(not missing, f"v51 {label} has unused parameters: {missing}")
    require(not nonfinite, f"v51 {label} has non-finite gradients: {nonfinite}")
    norm = torch.stack(gradients).sum().sqrt()
    require(float(norm) > 0.0, f"v51 {label} produced zero gradient norm")
    preclip = clip_finite_grad_norm_(trainable, 5.0)
    result = {
        f"{label}_loss": float(output["loss"].detach()),
        f"{label}_gradient_norm": float(norm),
        f"{label}_preclip_gradient_norm": float(preclip),
        f"{label}_trainable_parameter_tensors": float(len(trainable)),
    }
    model.zero_grad(set_to_none=True)
    return result


@torch.no_grad()
def structural_contract(model, features, evidence, output) -> dict[str, float]:
    state = output["state"]
    batch, frames, patches = features.valid.shape
    slots = model.config.object_slots
    require(state["identity"].shape == (batch, frames, slots, model.config.identity_dim), "v51 identity shape differs")
    require(state["dynamic"].shape == (batch, frames, slots, model.config.dynamic_dim), "v51 dynamic shape differs")
    require(state["support_shape"].shape == (batch, frames, slots, 3), "v51 support shape differs")
    require(state["unobserved_time"].shape == (batch, frames, slots), "v51 unobserved-time shape differs")
    require(state["assignment"].shape == (batch, frames, patches, model.config.owner_count), "v51 assignment shape differs")
    require(output["student_tracklets"].features.shape == features.patches.shape, "v51 student tracklet shape differs")
    require(output["student_tracklets"].temporal_residual.shape == features.patches.shape, "v51 temporal residual shape differs")
    require(float(output["student_tracklets"].temporal_residual[:, 0].abs().max()) < 1e-6, "v51 first-frame temporal residual is not zero")
    require(evidence.coordinates.shape == (batch, frames, model.config.tracker_queries, 2), "v51 point-track shape differs")
    require(evidence.visibility.shape == (batch, frames, model.config.tracker_queries), "v51 visibility shape differs")
    partition = state["assignment"].sum(dim=-1)
    partition_error = maximum_difference(partition[features.valid], torch.ones_like(partition[features.valid]))
    require(partition_error < 1e-5, "v51 encoder owners do not partition patches")
    decoder_partition = output["decoder_assignment"].sum(dim=-1)
    decoder_error = maximum_difference(
        decoder_partition[features.valid], torch.ones_like(decoder_partition[features.valid])
    )
    require(decoder_error < 1e-5, "v51 decoder owners do not partition patches")
    direction_norm = state["support_shape"][..., 1:].float().norm(dim=-1)
    shape_error = float((direction_norm - 1.0).abs().max())
    require(shape_error < 1e-3, "v51 support orientation is not normalized")
    require(len(evidence.query_times.unique()) >= 2, "v51 tracker does not use multiple anchor times")
    teacher_components = float(output["teacher"].component_valid.float().sum(dim=-1).mean())
    require(teacher_components >= 1.0, "v51 verifier found no trajectory teacher component")
    return {
        "object_slots": float(slots),
        "owner_count": float(model.config.owner_count),
        "patch_count": float(patches),
        "point_track_count": float(model.config.tracker_queries),
        "point_track_visible_fraction": float(evidence.visibility.float().mean()),
        "encoder_owner_partition_max_error": partition_error,
        "decoder_owner_partition_max_error": decoder_error,
        "support_orientation_max_norm_error": shape_error,
        "trajectory_teacher_component_count": teacher_components,
    }


@torch.no_grad()
def causal_contract(model, features, batch, amp_context) -> dict[str, float]:
    midpoint = features.patches.shape[1] // 2
    changed = features.patches.clone()
    changed[:, midpoint:] = changed[:, midpoint:].flip(2)
    with amp_context():
        _, reference = model.encode_student(
            features.patches, features.coordinates, features.valid, batch["frame_times"]
        )
        _, altered = model.encode_student(
            changed, features.coordinates, features.valid, batch["frame_times"]
        )
    difference = maximum_difference(
        reference["identity"][:, :midpoint], altered["identity"][:, :midpoint]
    )
    require(difference < 1e-6, "future RGB changed the causal v51 state prefix")
    return {"future_swap_prefix_max_difference": difference}


@torch.no_grad()
def external_track_contract(
    model, features, evidence, output, frame_times, amp_context
) -> dict[str, float]:
    match = output["match"]
    midpoint = evidence.coordinates.shape[1] // 2
    changed_coordinates = evidence.coordinates.clone()
    changed_coordinates[:, midpoint:] = changed_coordinates[:, midpoint:].roll(1, dims=2)
    changed_features = evidence.sampled_features.clone()
    changed_features[:, midpoint:] = changed_features[:, midpoint:].roll(1, dims=2)
    shifted = PointTrackEvidence(
        coordinates=changed_coordinates,
        visibility=evidence.visibility,
        residual_flow=evidence.residual_flow,
        motion_salience=evidence.motion_salience,
        query_times=evidence.query_times,
        sampled_features=changed_features,
    )
    changed_teacher = build_trajectory_component_teacher(
        shifted, model.config, frame_times
    )
    changed = match_trajectory_teacher_to_student(
        output["state"], shifted, changed_teacher, features.grid_hw, model.config.object_slots
    )
    with amp_context():
        changed_output = model(
            features.patches,
            features.coordinates,
            features.valid,
            frame_times,
            shifted,
            features.grid_hw,
        )
    student_difference = maximum_difference(
        output["state"]["identity"], changed_output["state"]["identity"]
    )
    assignment_difference = maximum_difference(
        match.target_student_owner, changed.target_student_owner
    )
    graph_difference = maximum_difference(
        output["teacher"].graph_affinity, changed_teacher.graph_affinity
    )
    require(graph_difference > 1e-6, "v51 teacher ignores external track identity")
    require(student_difference < 1e-6, "v51 deployment student reads teacher tracks")
    return {
        "external_track_target_owner_difference": assignment_difference,
        "external_track_graph_affinity_difference": graph_difference,
        "student_state_track_shuffle_max_difference": student_difference,
    }


@torch.no_grad()
def decoder_occlusion_contract(model, features, output, amp_context) -> dict[str, float]:
    state = {name: value[:, 0].clone() for name, value in output["state"].items() if name not in ("assignment", "mass")}
    with amp_context():
        _, reference = model.decoder(state, features.coordinates[:, 0], features.valid[:, 0])
        state["visibility"][:, 0] = 0.0
        _, hidden = model.decoder(state, features.coordinates[:, 0], features.valid[:, 0])
    decrease = float((reference[..., 0] - hidden[..., 0]).mean())
    require(decrease > 0.0, "occluded v51 object still participates in reconstruction")
    return {"occluded_object_assignment_mean_decrease": decrease}


@torch.no_grad()
def zero_effect_contract(model, output, frame_times, amp_context) -> dict[str, float]:
    source = frame_state(output["state"], 0)
    effect = torch.zeros(
        len(frame_times), model.config.effect_factors, model.config.effect_dim,
        device=frame_times.device,
    )
    with amp_context():
        predicted = model.dynamics(source, effect, frame_times[:, -1] - frame_times[:, 0])
    difference = max(maximum_difference(predicted[name], source[name]) for name in source)
    require(difference < 1e-6, "zero latent effect changes v51 object state")
    return {"zero_effect_state_max_difference": difference}


@torch.no_grad()
def teacher_semantics_contract(config, device) -> dict[str, float]:
    frames, points = 12, 6
    coordinates = torch.zeros(1, frames, points, 2, device=device)
    base = torch.tensor(
        [[-0.6, -0.4], [-0.5, -0.4], [-0.1, 0.0], [0.0, 0.0], [0.5, 0.4], [0.6, 0.4]],
        device=device,
    )
    coordinates[:] = base
    time = torch.arange(frames, device=device).float()
    coordinates[0, :, 2:4, 0] += 0.04 * time[:, None]
    visibility = torch.ones(1, frames, points, device=device, dtype=torch.bool)
    visibility[:, 2:5, 2:4] = False
    visibility[:, 5:, 4:6] = False
    features = torch.zeros(1, frames, points, config.patch_dim, device=device)
    features[..., 0:2, 0] = 1.0
    features[..., 2:4, 1] = 1.0
    features[..., 4:6, 2] = 1.0
    residual = coordinates[:, 1:] - coordinates[:, :-1]
    salience = residual.norm(dim=-1)
    evidence = PointTrackEvidence(
        coordinates=coordinates,
        visibility=visibility,
        residual_flow=residual,
        motion_salience=salience,
        query_times=torch.zeros(points, device=device, dtype=torch.long),
        sampled_features=features,
    )
    frame_times = time[None] / 30.0
    teacher = build_trajectory_component_teacher(evidence, config, frame_times)
    object_owner = teacher.track_owner[0, :, : config.object_slots].sum(dim=-1)
    static_is_object = float(object_owner[:2].mean())
    moving_is_object = float(object_owner[2:4].mean())
    absent = float((teacher.lifecycle_state == LIFECYCLE_ABSENT).sum())
    occluded = float((teacher.lifecycle_state == LIFECYCLE_OCCLUDED).sum())
    require(static_is_object == 1.0, "v51 static persistent tracks are not object candidates")
    require(moving_is_object == 1.0, "v51 moving persistent tracks are not object candidates")
    require(absent > 0.0, "v51 teacher produced no sustained-absence target")
    require(occluded > 0.0, "v51 teacher produced no gap/reappearance occlusion target")
    horizon_valid = teacher.relative_motion_valid.flatten(0, 2).any(dim=0)
    require(bool(horizon_valid.all()), "v51 teacher does not cover every motion horizon")
    geometry_horizon_valid = teacher.geometry_residual_valid.flatten(0, 2).any(dim=0)
    require(bool(geometry_horizon_valid.all()), "v51 teacher does not cover every geometry horizon")
    return {
        "static_teacher_object_fraction": static_is_object,
        "moving_teacher_object_fraction": moving_is_object,
        "teacher_absent_states": absent,
        "teacher_occluded_states": occluded,
        "teacher_motion_horizons": float(len(config.dynamic_horizons)),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v51 verifier requires a visible CUDA device")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    config = PointTrackObjectStateConfig()
    config.validate()
    semantics = teacher_semantics_contract(config, device)
    dataset = PointTrackObjectVideoDataset(
        args.data, "train", max_items=64, seed=args.seed
    )
    samples = [dataset[(index, args.chunk_length)] for index in range(1)]
    forbidden = {"instruction", "condition_feature", "teacher_sidecar", "segmentation", "action", "dino"}
    require(not forbidden.intersection(samples[0]), "v51 dataset exposed forbidden supervision")
    batch = default_collate(samples)
    batch = {name: value.to(device) if torch.is_tensor(value) else value for name, value in batch.items()}
    dino = FrozenDinoVideoRuntime(config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint)
    point_tracker = FrozenPointTrackerRuntime(config, device, args.tracker_checkpoint, 1)
    require(not any(parameter.requires_grad for parameter in dino.backbone.parameters()), "v51 DINO is not frozen")
    require(not any(parameter.requires_grad for parameter in point_tracker.model.parameters()), "v51 point tracker is not frozen")
    features = dino(batch)
    evidence = point_tracker(batch, features.patches, features.grid_hw)
    require(bool(evidence.visibility.any()), "v51 verifier found no visible point track")
    amp_context = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.amp == "bf16" else nullcontext
    model = PointTrackObjectWorldModel(config, "object_state").to(device).train()
    with amp_context():
        state_output = model(
            features.patches, features.coordinates, features.valid,
            batch["frame_times"], evidence, features.grid_hw,
        )
    require(bool(torch.isfinite(state_output["loss"])), "v51 Object State loss is non-finite")
    structural = structural_contract(model, features, evidence, state_output)
    state_gradients = finite_gradient_contract(model, state_output, "object_state")
    model.eval()
    causal = causal_contract(model, features, batch, amp_context)
    external = external_track_contract(
        model, features, evidence, state_output, batch["frame_times"], amp_context
    )
    occlusion = decoder_occlusion_contract(model, features, state_output, amp_context)
    zero = zero_effect_contract(model, state_output, batch["frame_times"], amp_context)
    model.set_stage("latent_effect")
    model.train()
    with amp_context():
        effect_output = model(
            features.patches, features.coordinates, features.valid,
            batch["frame_times"], None, features.grid_hw,
        )
    effect_gradients = finite_gradient_contract(model, effect_output, "latent_effect")
    report = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "contract": POINT_TRACK_VIDEO_CONTRACT,
        "git_commit": args.source_revision,
        "data": os.path.abspath(args.data),
        "historical_checkpoint_used": False,
        "trajectory_teacher": "frozen_cotracker3_component_set_v2",
        "object_candidate_contract": "persistent_appearance_geometry_not_motion_threshold",
        "lifecycle_contract": "visible_occluded_unknown_absent_from_track_history",
        "dynamic_supervision": "multi_horizon_relative_motion_and_geometry_residual",
        "dino_fully_frozen": True,
        "language_used": False,
        "explicit_action_used": False,
        "instance_segmentation_used": False,
        "object_state_then_latent_effect": True,
        **structural,
        **state_gradients,
        **effect_gradients,
        **causal,
        **external,
        **occlusion,
        **zero,
        **semantics,
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
