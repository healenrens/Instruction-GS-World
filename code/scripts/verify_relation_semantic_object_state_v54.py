"""GPU verifier for the v54 relation-semantic Object State contract."""

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
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.point_track_teacher import (  # noqa: E402
    FrozenPointTrackerRuntime,
    PointTrackEvidence,
)
from igsw.adaptive_gaussian_wm.relation_semantic_object_state_v54 import (  # noqa: E402
    RelationSemanticObjectStateModel,
)
from igsw.adaptive_gaussian_wm.relation_semantic_objective_v54 import (  # noqa: E402
    semantic_alignment_terms,
)
from igsw.adaptive_gaussian_wm.trajectory_relation_teacher_v54 import (  # noqa: E402
    build_trajectory_relation_teacher_v54,
)
from igsw.adaptive_gaussian_wm.v52_falsification import (  # noqa: E402
    build_synthetic_objective_contract,
    run_objective_falsification,
)
from igsw.adaptive_gaussian_wm.v54_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    RelationSemanticObjectStateConfig,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def maximum_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--dino_frame_batch", type=int, default=16)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--chunk_lengths", default="3,4,6,8")
    parser.add_argument("--temporal_step_ms", default="100,200,400")
    parser.add_argument("--chunk_length", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def finite_gradient_contract(model, output) -> dict[str, float]:
    output["loss"].backward()
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    missing, nonfinite, gradients = [], [], []
    semantic_gradient = 0.0
    for name, parameter in trainable:
        if parameter.grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        magnitude = parameter.grad.detach().float().square().sum()
        gradients.append(magnitude)
        if name.startswith("identity_to_semantic."):
            semantic_gradient += float(magnitude)
    require(not missing, f"v54 has unused trainable parameters: {missing}")
    require(not nonfinite, f"v54 has non-finite gradients: {nonfinite}")
    norm = torch.stack(gradients).sum().sqrt()
    require(float(norm) > 0.0, "v54 produced zero gradient norm")
    require(semantic_gradient > 0.0, "v54 semantic head received no gradient")
    preclip = clip_finite_grad_norm_(trainable, 5.0)
    model.zero_grad(set_to_none=True)
    return {
        "object_state_loss": float(output["loss"].detach()),
        "object_state_gradient_norm": float(norm),
        "object_state_preclip_gradient_norm": float(preclip),
        "semantic_head_gradient_norm": semantic_gradient**0.5,
        "object_state_trainable_parameter_tensors": float(len(trainable)),
    }


@torch.no_grad()
def structural_contract(model, features, evidence, output) -> dict[str, float]:
    state = output["state"]
    batch, frames, patches = features.valid.shape
    slots = model.config.object_slots
    require(
        state["identity"].shape == (batch, frames, slots, model.config.identity_dim),
        "v54 identity shape differs",
    )
    require(
        state["dynamic"].shape == (batch, frames, slots, model.config.dynamic_dim),
        "v54 dynamic shape differs",
    )
    require(
        state["assignment"].shape == (batch, frames, patches, model.config.owner_count),
        "v54 assignment shape differs",
    )
    require(
        output["semantic_identity"].shape[-1] == model.config.patch_dim,
        "v54 semantic identity dimension differs",
    )
    encoder_partition = state["assignment"].sum(dim=-1)
    encoder_error = maximum_difference(
        encoder_partition[features.valid], torch.ones_like(encoder_partition[features.valid])
    )
    decoder_partition = output["decoder_assignment"].sum(dim=-1)
    decoder_error = maximum_difference(
        decoder_partition[features.valid], torch.ones_like(decoder_partition[features.valid])
    )
    require(encoder_error < 1e-5, "v54 encoder owners do not partition patches")
    require(decoder_error < 1e-5, "v54 decoder owners do not partition patches")
    require(not hasattr(model, "dynamics"), "v54 contains forbidden Dynamics")
    require(not hasattr(model, "effect_posterior"), "v54 contains forbidden latent effect")
    require(not hasattr(model, "point_tracker"), "v54 model contains the training teacher")
    return {
        "object_slots": float(slots),
        "owner_count": float(model.config.owner_count),
        "patch_count": float(patches),
        "point_track_count": float(evidence.coordinates.shape[2]),
        "encoder_owner_partition_max_error": encoder_error,
        "decoder_owner_partition_max_error": decoder_error,
        "same_relation_evidence_mean": float(output["teacher"].same_confidence.mean()),
        "different_relation_evidence_mean": float(output["teacher"].different_confidence.mean()),
        "teacher_object_confidence_mean": float(output["teacher"].object_confidence.mean()),
        "lifecycle_known_fraction": float(output["teacher"].lifecycle_known.float().mean()),
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
    require(difference < 1e-6, "future RGB changed the causal v54 state prefix")
    return {"future_swap_prefix_max_difference": difference}


@torch.no_grad()
def teacher_isolation_contract(model, features, batch, evidence, output, amp_context):
    changed = PointTrackEvidence(
        coordinates=evidence.coordinates.roll(1, dims=2),
        visibility=evidence.visibility,
        residual_flow=evidence.residual_flow.roll(1, dims=2),
        motion_salience=evidence.motion_salience.roll(1, dims=2),
        query_times=evidence.query_times,
        sampled_features=evidence.sampled_features.roll(1, dims=2),
    )
    changed_teacher = build_trajectory_relation_teacher_v54(
        changed, model.config, batch["frame_times"][:1]
    )
    with amp_context():
        changed_output = model(
            features.patches, features.coordinates, features.valid,
            batch["frame_times"], changed, torch.tensor([0], device=features.patches.device),
            features.grid_hw,
        )
    state_difference = maximum_difference(
        output["state"]["identity"], changed_output["state"]["identity"]
    )
    target_difference = maximum_difference(
        output["teacher"].relation_score, changed_teacher.relation_score
    )
    require(state_difference < 1e-6, "v54 deployment Student reads teacher tracks")
    require(target_difference > 1e-6, "v54 relation target ignores changed tracks")
    return {
        "student_state_track_swap_max_difference": state_difference,
        "teacher_relation_track_swap_max_difference": target_difference,
    }


def semantic_falsification(config, device) -> dict[str, float | bool]:
    teacher, _, _ = build_synthetic_objective_contract(config, device)
    target = teacher.track_identity[:, None].expand(-1, teacher.visibility.shape[1], -1, -1)
    reasonable = semantic_alignment_terms(target, teacher)["semantic_alignment"]
    swapped = target.roll(2, dims=2)
    corrupted = semantic_alignment_terms(swapped, teacher)["semantic_alignment"]
    margin = float(corrupted - reasonable)
    require(margin > config.objective_falsification_margin, "v54 semantic target ignores track identity")
    return {
        "semantic_reasonable_loss": float(reasonable),
        "semantic_swapped_loss": float(corrupted),
        "semantic_swap_margin": margin,
        "semantic_falsification_passed": True,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v54 verifier requires a visible CUDA device")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    config = RelationSemanticObjectStateConfig()
    config.validate()
    falsification = run_objective_falsification(config, device)
    require(falsification["status"] == "passed", "v54 base objective falsification failed")
    semantic = semantic_falsification(config, device)

    dataset = MultiSourceRobotVideoDataset(
        args.data_index,
        "train",
        chunk_lengths=args.chunk_lengths,
        temporal_step_ms=args.temporal_step_ms,
        max_items=0,
        seed=args.seed,
    )
    sample = dataset[(0, args.chunk_length)]
    forbidden = {
        "instruction", "condition_feature", "teacher_sidecar",
        "segmentation", "action", "dino",
    }
    require(not forbidden.intersection(sample), "v54 dataset exposed forbidden supervision")
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
        "v54 DINO is not frozen",
    )
    require(
        not any(parameter.requires_grad for parameter in tracker.model.parameters()),
        "v54 point tracker is not frozen",
    )
    features = dino(batch)
    evidence = tracker(batch, features.patches, features.grid_hw)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16" else nullcontext
    )
    model = RelationSemanticObjectStateModel(config).to(device).train()
    teacher_indices = torch.tensor([0], device=device)
    with amp_context():
        output = model(
            features.patches, features.coordinates, features.valid,
            batch["frame_times"], evidence, teacher_indices, features.grid_hw,
        )
    structure = structural_contract(model, features, evidence, output)
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
        "git_commit": args.source_revision,
        "data_index": os.path.abspath(args.data_index),
        "historical_checkpoint_used": False,
        "hard_component_pseudo_labels": False,
        "dense_affinity_objective": False,
        "point_tracker_role": "training_only_rotating_relation_teacher",
        "point_tracker_in_deployable_model": False,
        "dynamics_present": False,
        "latent_effect_present": False,
        "dino_fully_frozen": True,
        "language_used": False,
        "explicit_action_used": False,
        "instance_segmentation_used": False,
        "objective_falsification": falsification,
        **semantic,
        **structure,
        **gradients,
        **causal,
        **isolation,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
