"""GPU verifier for the v52 learning-objective-first contract."""

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
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MULTISOURCE_POINT_TRACK_CONTRACT,
    MultiSourcePointTrackObjectVideoDataset,
)
from igsw.adaptive_gaussian_wm.object_state_target_v52 import (  # noqa: E402
    ObjectStatePredictions,
    object_state_target_terms,
)
from igsw.adaptive_gaussian_wm.point_track_dataset import (  # noqa: E402
    POINT_TRACK_VIDEO_CONTRACT,
    PointTrackObjectVideoDataset,
)
from igsw.adaptive_gaussian_wm.point_track_teacher import (  # noqa: E402
    FrozenPointTrackerRuntime,
    PointTrackEvidence,
)
from igsw.adaptive_gaussian_wm.point_track_world_model_v52 import (  # noqa: E402
    LearningObjectiveObjectWorldModel,
)
from igsw.adaptive_gaussian_wm.trajectory_relation_teacher import (  # noqa: E402
    build_trajectory_relation_teacher,
)
from igsw.adaptive_gaussian_wm.v52_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    LearningObjectiveObjectStateConfig,
)
from igsw.adaptive_gaussian_wm.v52_falsification import (  # noqa: E402
    build_synthetic_objective_contract,
    run_objective_falsification,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def maximum_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--data_index", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--dino_frame_batch", type=int, default=16)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--chunk_length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def finite_gradient_contract(model, output) -> dict[str, float]:
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
    require(not missing, f"v52 has unused trainable parameters: {missing}")
    require(not nonfinite, f"v52 has non-finite gradients: {nonfinite}")
    norm = torch.stack(gradients).sum().sqrt()
    require(float(norm) > 0.0, "v52 produced zero gradient norm")
    preclip = clip_finite_grad_norm_(trainable, 5.0)
    model.zero_grad(set_to_none=True)
    return {
        "object_state_loss": float(output["loss"].detach()),
        "object_state_gradient_norm": float(norm),
        "object_state_preclip_gradient_norm": float(preclip),
        "object_state_trainable_parameter_tensors": float(len(trainable)),
    }


@torch.no_grad()
def structural_contract(model, features, evidence, output) -> dict[str, float]:
    state = output["state"]
    batch, frames, patches = features.valid.shape
    slots = model.config.object_slots
    require(
        state["identity"].shape == (batch, frames, slots, model.config.identity_dim),
        "v52 identity shape differs",
    )
    require(
        state["dynamic"].shape == (batch, frames, slots, model.config.dynamic_dim),
        "v52 dynamic shape differs",
    )
    require(
        state["assignment"].shape == (batch, frames, patches, model.config.owner_count),
        "v52 assignment shape differs",
    )
    require(
        not hasattr(output["teacher"], "track_owner"),
        "v52 teacher exposed hard component owners",
    )
    encoder_partition = state["assignment"].sum(dim=-1)
    encoder_error = maximum_difference(
        encoder_partition[features.valid], torch.ones_like(encoder_partition[features.valid])
    )
    decoder_partition = output["decoder_assignment"].sum(dim=-1)
    decoder_error = maximum_difference(
        decoder_partition[features.valid], torch.ones_like(decoder_partition[features.valid])
    )
    require(encoder_error < 1e-5, "v52 encoder owners do not partition patches")
    require(decoder_error < 1e-5, "v52 decoder owners do not partition patches")
    require(not hasattr(model, "dynamics"), "v52 contains forbidden Dynamics")
    require(not hasattr(model, "effect_posterior"), "v52 contains forbidden latent effect")
    return {
        "object_slots": float(slots),
        "owner_count": float(model.config.owner_count),
        "patch_count": float(patches),
        "point_track_count": float(evidence.coordinates.shape[2]),
        "encoder_owner_partition_max_error": encoder_error,
        "decoder_owner_partition_max_error": decoder_error,
        "same_relation_evidence_mean": float(output["teacher"].same_confidence.mean()),
        "different_relation_evidence_mean": float(
            output["teacher"].different_confidence.mean()
        ),
        "lifecycle_known_fraction": float(
            output["teacher"].lifecycle_known.float().mean()
        ),
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
    require(difference < 1e-6, "future RGB changed the causal v52 state prefix")
    return {"future_swap_prefix_max_difference": difference}


@torch.no_grad()
def teacher_isolation_contract(model, features, batch, evidence, output, amp_context):
    midpoint = evidence.coordinates.shape[1] // 2
    changed = PointTrackEvidence(
        coordinates=evidence.coordinates.roll(1, dims=2),
        visibility=evidence.visibility,
        residual_flow=evidence.residual_flow.roll(1, dims=2),
        motion_salience=evidence.motion_salience.roll(1, dims=2),
        query_times=evidence.query_times,
        sampled_features=evidence.sampled_features.roll(1, dims=2),
    )
    changed_teacher = build_trajectory_relation_teacher(
        changed, model.config, batch["frame_times"]
    )
    with amp_context():
        changed_output = model(
            features.patches, features.coordinates, features.valid,
            batch["frame_times"], changed, features.grid_hw,
        )
    state_difference = maximum_difference(
        output["state"]["identity"], changed_output["state"]["identity"]
    )
    target_difference = maximum_difference(
        output["teacher"].relation_score, changed_teacher.relation_score
    )
    require(state_difference < 1e-6, "v52 deployment Student reads teacher tracks")
    require(target_difference > 1e-6, "v52 relation target ignores changed tracks")
    return {
        "student_state_track_swap_max_difference": state_difference,
        "teacher_relation_track_swap_max_difference": target_difference,
        "teacher_swap_midpoint": float(midpoint),
    }


def factorized_gradient_contract(config, device) -> dict[str, float]:
    teacher, evidence, base = build_synthetic_objective_contract(config, device)
    perturbed_identity = base.identity.detach().float().clone()
    perturbed_identity[:, perturbed_identity.shape[1] // 2 :] = perturbed_identity[
        :, perturbed_identity.shape[1] // 2 :
    ].roll(1, dims=-1)
    decoder_assignment = 0.90 * base.decoder_assignment.detach().float()
    decoder_assignment = decoder_assignment + 0.10 / config.owner_count
    tensors = {
        "assignment": base.assignment.detach().float().requires_grad_(True),
        "identity": perturbed_identity.requires_grad_(True),
        "motion": (base.motion.detach().float() + 0.05).requires_grad_(True),
        "center": (base.center.detach().float() + 0.05).requires_grad_(True),
        "visibility": base.visibility.detach().float().requires_grad_(True),
        "presence": base.presence.detach().float().requires_grad_(True),
        "decoder_assignment": decoder_assignment.requires_grad_(True),
    }
    prediction = ObjectStatePredictions(**tensors)
    terms = object_state_target_terms(prediction, teacher, evidence, config)
    expected = {
        "identity": ("identity",),
        "motion": ("motion",),
        "geometry": ("center",),
        "lifecycle": ("visibility", "presence"),
        "decoder_support": ("decoder_assignment",),
    }
    metrics = {}
    values = tuple(tensors.values())
    names = tuple(tensors)
    for term_name, expected_names in expected.items():
        gradients = torch.autograd.grad(
            terms[term_name], values, retain_graph=True, allow_unused=True
        )
        active = {
            name for name, gradient in zip(names, gradients)
            if gradient is not None and float(gradient.abs().sum()) > 0.0
        }
        require(active == set(expected_names), f"v52 {term_name} gradients reached {active}")
        metrics[f"{term_name}_gradient_field_count"] = float(len(active))
    return metrics


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v52 verifier requires a visible CUDA device")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    config = LearningObjectiveObjectStateConfig()
    config.validate()
    falsification = run_objective_falsification(config, device)
    require(falsification["status"] == "passed", "v52 objective falsification failed")
    dataset = (
        MultiSourcePointTrackObjectVideoDataset(
            args.data_index, "train", max_items=0, seed=args.seed
        )
        if args.data_index
        else PointTrackObjectVideoDataset(
            args.data, "train", max_items=64, seed=args.seed
        )
    )
    sample = dataset[(0, args.chunk_length)]
    forbidden = {
        "instruction", "condition_feature", "teacher_sidecar", "segmentation",
        "action", "dino",
    }
    require(not forbidden.intersection(sample), "v52 dataset exposed forbidden supervision")
    source_probe_count = 1
    if args.data_index:
        probes = [dataset[(index, args.chunk_length)] for index in dataset.source_probe_indices]
        require(len(probes) == len(dataset.source_names), "v52 did not probe every video source")
        require(
            all(tuple(item["video_rgb"].shape[-2:]) == (518, 518) for item in probes),
            "v52 multisource RGB preprocessing differs",
        )
        require(
            all(bool((item["frame_times"][1:] > item["frame_times"][:-1]).all()) for item in probes),
            "v52 multisource frame times are not strictly increasing",
        )
        sample = probes[0]
        source_probe_count = len(probes)
    batch = default_collate([sample])
    batch = {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    tracker = FrozenPointTrackerRuntime(config, device, args.tracker_checkpoint, 1)
    require(
        not any(parameter.requires_grad for parameter in dino.backbone.parameters()),
        "v52 DINO is not frozen",
    )
    require(
        not any(parameter.requires_grad for parameter in tracker.model.parameters()),
        "v52 point tracker is not frozen",
    )
    features = dino(batch)
    evidence = tracker(batch, features.patches, features.grid_hw)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16" else nullcontext
    )
    model = LearningObjectiveObjectWorldModel(config).to(device).train()
    with amp_context():
        output = model(
            features.patches, features.coordinates, features.valid,
            batch["frame_times"], evidence, features.grid_hw,
        )
    structure = structural_contract(model, features, evidence, output)
    factorization = factorized_gradient_contract(config, device)
    gradients = finite_gradient_contract(model, output)
    model.eval()
    causal = causal_contract(model, features, batch, amp_context)
    isolation = teacher_isolation_contract(
        model, features, batch, evidence, output, amp_context
    )
    report = {
        "status": "passed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "contract": (
            MULTISOURCE_POINT_TRACK_CONTRACT
            if args.data_index else POINT_TRACK_VIDEO_CONTRACT
        ),
        "git_commit": args.source_revision,
        "data": os.path.abspath(args.data),
        "data_index": os.path.abspath(args.data_index) if args.data_index else "",
        "source_names": list(getattr(dataset, "source_names", ("robotwin",))),
        "source_episode_counts": list(getattr(dataset, "source_episode_counts", ())),
        "source_task_counts": list(getattr(dataset, "source_task_counts", ())),
        "source_target_samples": list(getattr(dataset, "source_target_samples", ())),
        "source_probe_count": source_probe_count,
        "historical_checkpoint_used": False,
        "hard_component_pseudo_labels": False,
        "point_tracker_role": "training_only_correspondence_and_relation_evidence",
        "object_target": "persistent_compositional_relation_constrained_visual_entity",
        "objective_falsification_status": falsification["status"],
        "objective_falsification": falsification,
        "dynamics_present": False,
        "latent_effect_present": False,
        "dino_fully_frozen": True,
        "language_used": False,
        "explicit_action_used": False,
        "instance_segmentation_used": False,
        **structure,
        **factorization,
        **gradients,
        **causal,
        **isolation,
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
