"""Held-video teacher-agreement evaluation for v52 Object State."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.frozen_video_encoder import FrozenDinoVideoRuntime  # noqa: E402
from igsw.adaptive_gaussian_wm.point_track_dataset import PointTrackObjectVideoDataset  # noqa: E402
from igsw.adaptive_gaussian_wm.point_track_teacher import FrozenPointTrackerRuntime  # noqa: E402
from igsw.adaptive_gaussian_wm.point_track_world_model_v52 import (  # noqa: E402
    LearningObjectiveObjectWorldModel,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v52_checkpointing import validate_checkpoint_header  # noqa: E402
from igsw.adaptive_gaussian_wm.v52_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    LearningObjectiveObjectStateConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--evaluator_revision", required=True)
    parser.add_argument("--expected_step", type=int, default=22_500)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--splits", default="heldseed,heldtask")
    parser.add_argument("--chunk_lengths", default="8,16,24,32")
    parser.add_argument("--temporal_stride", type=int, default=1)
    parser.add_argument("--items", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dino_frame_batch", type=int, default=64)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--wandb_mode", choices=("disabled", "online", "offline"), default="online")
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="learning_objective_object_state_v52_step22500")
    parser.add_argument("--wandb_group", default="learning-objective-object-state-v52-eval")
    parser.add_argument("--wandb_dir", default="")
    return parser.parse_args()


def weighted(value, weight):
    return float((value.float() * weight.float()).sum() / weight.float().sum().clamp_min(1.0))


def batch_metrics(model, features, evidence, output, amp_context):
    prediction, teacher = output["prediction"], output["teacher"]
    assignment = prediction.assignment.float()
    visible = teacher.visibility.float()
    object_weight = visible * teacher.object_confidence[:, None]
    pair_visible = (
        visible[:, 1:]
        * visible[:, :-1]
        * teacher.object_confidence[:, None]
    )
    correct = F.cosine_similarity(assignment[:, 1:], assignment[:, :-1], dim=-1)
    shuffled = F.cosine_similarity(
        assignment[:, 1:], assignment[:, :-1].roll(1, dims=2), dim=-1
    )
    mean_owner = (assignment * visible[..., None]).sum(dim=1)
    mean_owner = mean_owner / visible.sum(dim=1).clamp_min(1.0)[..., None]
    object_mean = mean_owner[..., : model.config.object_slots]
    relation_similarity = torch.einsum("bpk,bqk->bpq", object_mean, object_mean)
    object_pair = (
        teacher.object_confidence[:, :, None]
        * teacher.object_confidence[:, None]
    )
    identity = F.normalize(prediction.identity.float(), dim=-1, eps=1e-6)
    mean_identity = (identity * visible[..., None]).sum(dim=1)
    mean_identity = F.normalize(
        mean_identity / visible.sum(dim=1).clamp_min(1.0)[..., None],
        dim=-1,
        eps=1e-6,
    )
    identity_temporal = (identity * mean_identity[:, None]).sum(dim=-1)
    identity_relation = torch.einsum("bpd,bqd->bpq", mean_identity, mean_identity)
    motion_error = F.smooth_l1_loss(
        prediction.motion.float(), teacher.motion.float(), reduction="none"
    ).mean(dim=-1)
    zero_motion = F.smooth_l1_loss(
        torch.zeros_like(prediction.motion), teacher.motion.float(), reduction="none"
    ).mean(dim=-1)
    motion_weight = teacher.motion_valid.float() * teacher.object_confidence[:, None, :, None]
    known = teacher.lifecycle_known.float() * teacher.object_confidence[:, None]
    visibility_accuracy = (
        (prediction.visibility >= 0.5) == (teacher.visibility >= 0.5)
    ).float()
    presence_accuracy = (
        (prediction.presence >= 0.5) == (teacher.presence >= 0.5)
    ).float()
    root_mass = torch.einsum(
        "btpk,btp->bk",
        assignment[..., : model.config.object_slots],
        visible * teacher.object_confidence[:, None],
    )
    root_share = root_mass / root_mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    effective = torch.exp(-(root_share * root_share.clamp_min(1e-8).log()).sum(dim=-1))
    values = {
        "track_consistency": weighted(correct, pair_visible),
        "track_shuffled_consistency": weighted(shuffled, pair_visible),
        "same_relation_similarity": weighted(
            relation_similarity, teacher.same_confidence * object_pair
        ),
        "different_relation_similarity": weighted(
            relation_similarity, teacher.different_confidence * object_pair
        ),
        "identity_temporal_cosine": weighted(
            identity_temporal, object_weight
        ),
        "identity_same_similarity": weighted(
            identity_relation, teacher.same_confidence * object_pair
        ),
        "identity_different_similarity": weighted(
            identity_relation, teacher.different_confidence * object_pair
        ),
        "motion_error": weighted(motion_error, motion_weight),
        "zero_motion_error": weighted(zero_motion, motion_weight),
        "visibility_accuracy": weighted(visibility_accuracy, known),
        "presence_accuracy": weighted(presence_accuracy, known),
        "effective_roots": float(effective.mean()),
        "maximum_root_share": float(root_share.amax(dim=-1).mean()),
        "student_object_fraction": weighted(
            assignment[..., : model.config.object_slots].sum(dim=-1), visible
        ),
        "teacher_object_confidence": weighted(
            teacher.object_confidence[:, None].expand_as(visible), visible
        ),
    }
    values.update(deletion_locality(model, features, evidence, output, amp_context))
    return values


@torch.no_grad()
def deletion_locality(model, features, evidence, output, amp_context):
    prediction, teacher, state = output["prediction"], output["teacher"], output["state"]
    candidate = teacher.object_confidence * teacher.visibility.sum(dim=1)
    if float(candidate.max()) <= 0.0:
        return {"deletion_inside": 0.0, "deletion_outside": 0.0, "deletion_items": 0.0}
    track = int(candidate[0].argmax())
    frame = int(teacher.visibility[0, :, track].argmax())
    slot = int(prediction.assignment[0, frame, track, : model.config.object_slots].argmax())
    frame_state = {
        name: value[0:1, frame]
        for name, value in state.items()
        if name not in ("assignment", "mass")
    }
    coordinates = features.coordinates[0:1, frame]
    valid = features.valid[0:1, frame]
    with amp_context():
        reference, _ = model.decoder(frame_state, coordinates, valid)
        object_valid = torch.ones(
            1, model.config.object_slots, device=reference.device, dtype=torch.bool
        )
        object_valid[:, slot] = False
        deleted, _ = model.decoder(frame_state, coordinates, valid, object_valid)
    change = 1.0 - F.cosine_similarity(reference.float(), deleted.float(), dim=-1)
    track_position = evidence.coordinates[0, frame, track]
    distance = (features.coordinates[0, frame] - track_position).norm(dim=-1)
    inside = (distance <= 0.18) & valid[0]
    outside = (distance >= 0.30) & valid[0]
    if not bool(inside.any()) or not bool(outside.any()):
        return {"deletion_inside": 0.0, "deletion_outside": 0.0, "deletion_items": 0.0}
    return {
        "deletion_inside": float(change[0, inside].mean()),
        "deletion_outside": float(change[0, outside].mean()),
        "deletion_items": 1.0,
    }


def evaluate_condition(model, loader, dino, tracker, device, amp_context):
    sums, count = {}, 0
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_to_device(cpu_batch, device)
            features = dino(batch)
            evidence = tracker(batch, features.patches, features.grid_hw)
            with amp_context():
                output = model(
                    features.patches, features.coordinates, features.valid,
                    batch["frame_times"], evidence, features.grid_hw,
                )
            values = batch_metrics(model, features, evidence, output, amp_context)
            batch_size = len(features.patches)
            count += batch_size
            for name, value in values.items():
                sums[name] = sums.get(name, 0.0) + value * batch_size
    metrics = {name: value / max(count, 1) for name, value in sums.items()}
    metrics["track_margin"] = metrics["track_consistency"] - metrics["track_shuffled_consistency"]
    metrics["relation_margin"] = metrics["same_relation_similarity"] - metrics["different_relation_similarity"]
    metrics["identity_margin"] = (
        metrics["identity_same_similarity"]
        - metrics["identity_different_similarity"]
    )
    metrics["motion_gain_over_zero"] = (
        metrics["zero_motion_error"] - metrics["motion_error"]
    ) / max(metrics["zero_motion_error"], 1e-6)
    metrics["deletion_locality_ratio"] = metrics["deletion_inside"] / max(
        metrics["deletion_outside"], 1e-8
    )
    checks = {
        "track_correspondence": metrics["track_margin"] >= 0.03,
        "relation_separation": metrics["relation_margin"] >= 0.10,
        "identity_temporal_stability": metrics["identity_temporal_cosine"] >= 0.90,
        "identity_separation": metrics["identity_margin"] >= 0.10,
        "motion_decodable": metrics["motion_gain_over_zero"] >= 0.10,
        "lifecycle_decodable": metrics["visibility_accuracy"] >= 0.75
        and metrics["presence_accuracy"] >= 0.75,
        "root_not_dominant": metrics["maximum_root_share"] <= 0.75,
        "deletion_supported": metrics["deletion_items"] >= 1.0,
        "deletion_local": metrics["deletion_locality_ratio"] >= 1.25,
    }
    return metrics, checks


def init_wandb(args):
    if args.wandb_mode == "disabled":
        return None
    if not args.wandb_dir or not args.wandb_project:
        raise ValueError("v52 evaluation W&B directory and project are required")
    import wandb

    os.makedirs(args.wandb_dir, exist_ok=True)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name,
        group=args.wandb_group,
        job_type="v52-object-state-held-teacher-agreement",
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        config=vars(args),
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v52 held evaluation requires CUDA")
    checkpoint = torch.load(
        os.path.abspath(args.checkpoint), map_location="cpu", weights_only=False, mmap=True
    )
    validate_checkpoint_header(checkpoint)
    if checkpoint.get("git_commit") != args.source_revision:
        raise ValueError("v52 evaluation source revision differs from checkpoint")
    if int(checkpoint.get("global_step", -1)) != args.expected_step:
        raise ValueError("v52 evaluation requires the step-22500 checkpoint")
    config = LearningObjectiveObjectStateConfig()
    if checkpoint.get("config") != config.to_dict():
        raise ValueError("v52 evaluation config differs from checkpoint")
    device = torch.device("cuda:0")
    model = LearningObjectiveObjectWorldModel(config).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    tracker = FrozenPointTrackerRuntime(config, device, args.tracker_checkpoint, 1)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16" else nullcontext
    )
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    lengths = [int(value) for value in args.chunk_lengths.split(",")]
    run = init_wandb(args)
    conditions, condition_index = {}, 0
    for split in splits:
        for length in lengths:
            dataset = PointTrackObjectVideoDataset(
                args.data, split, str(length), str(args.temporal_stride), args.items, args.seed
            )
            loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
            metrics, checks = evaluate_condition(
                model, loader, dino, tracker, device, amp_context
            )
            key = f"{split}/H{length}"
            conditions[key] = {"metrics": metrics, "checks": checks}
            print(json.dumps({"condition": key, "metrics": metrics, "checks": checks}, sort_keys=True), flush=True)
            if run is not None:
                run.log({
                    "evaluation/condition_index": condition_index,
                    "evaluation/history_length": length,
                    **{f"metric/{name}": value for name, value in metrics.items()},
                    **{f"check/{name}": int(value) for name, value in checks.items()},
                }, step=condition_index)
            condition_index += 1
    teacher_gate = all(
        all(condition["checks"].values()) for condition in conditions.values()
    )
    report = {
        "status": "completed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "training_git_commit": args.source_revision,
        "evaluator_git_commit": args.evaluator_revision,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_step": int(checkpoint["global_step"]),
        "evaluation_scope": "held_video_relation_teacher_agreement",
        "teacher_agreement_gate_passed": teacher_gate,
        "independent_object_truth_available": False,
        "deployment_promotion_ready": False,
        "conditions": conditions,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if run is not None:
        run.summary.update({
            "evaluation/status": report["status"],
            "evaluation/teacher_agreement_gate_passed": int(teacher_gate),
            "evaluation/independent_object_truth_available": 0,
            "evaluation/deployment_promotion_ready": 0,
            "evaluation/report": output_path,
        })
        run.finish()
    print(json.dumps({"report": output_path, "teacher_agreement_gate_passed": teacher_gate, "deployment_promotion_ready": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
