"""Evaluate v52 with external object truth and no point tracker."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.frozen_video_encoder import FrozenDinoVideoRuntime  # noqa: E402
from igsw.adaptive_gaussian_wm.independent_object_truth_v52 import (  # noqa: E402
    INDEPENDENT_OBJECT_TRUTH_CONTRACT,
    IndependentObjectTruthDataset,
    independent_deletion_metrics,
    independent_object_metrics,
)
from igsw.adaptive_gaussian_wm.point_track_world_model_v52 import (  # noqa: E402
    LearningObjectiveObjectWorldModel,
)
from igsw.adaptive_gaussian_wm.v52_checkpointing import validate_checkpoint_header  # noqa: E402
from igsw.adaptive_gaussian_wm.v52_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    LearningObjectiveObjectStateConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--truth_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--evaluator_revision", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--expected_step", type=int, default=22_500)
    parser.add_argument("--splits", default="heldseed,heldtask")
    parser.add_argument("--dino_frame_batch", type=int, default=64)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--minimum_items", type=int, default=32)
    parser.add_argument("--minimum_objects", type=int, default=64)
    parser.add_argument("--minimum_reappearance_cases", type=int, default=8)
    parser.add_argument("--minimum_occluded_cases", type=int, default=8)
    parser.add_argument("--minimum_absent_cases", type=int, default=8)
    parser.add_argument("--wandb_mode", choices=("disabled", "online", "offline"), default="online")
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="independent_object_state_v52_step22500")
    parser.add_argument("--wandb_group", default="independent-object-state-v52-eval")
    parser.add_argument("--wandb_dir", default="")
    return parser.parse_args()


def init_wandb(args):
    if args.wandb_mode == "disabled":
        return None
    if not args.wandb_dir or not args.wandb_project:
        raise ValueError("independent v52 evaluation requires W&B directory and project")
    import wandb

    os.makedirs(args.wandb_dir, exist_ok=True)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name,
        group=args.wandb_group,
        job_type="v52-independent-object-truth",
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        config=vars(args),
    )


def move_truth(sample: dict, device: torch.device) -> dict:
    return {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in sample.items()
    }


def add_statistics(totals: dict[str, list[float]], values: dict) -> None:
    for name, (value_sum, value_count) in values.items():
        current = totals.setdefault(name, [0.0, 0.0])
        current[0] += float(value_sum)
        current[1] += float(value_count)


def normalized_metrics(totals: dict[str, list[float]]) -> dict[str, float]:
    metrics = {
        name: value_sum / value_count if value_count > 0.0 else 0.0
        for name, (value_sum, value_count) in totals.items()
    }
    metrics["object_relation_margin"] = (
        metrics["same_object_root_similarity"]
        - metrics["different_object_root_similarity"]
    )
    metrics["decoder_object_relation_margin"] = (
        metrics["decoder_same_object_similarity"]
        - metrics["decoder_different_object_similarity"]
    )
    metrics["deletion_locality_ratio"] = metrics["deletion_inside"] / max(
        metrics["deletion_outside"], 1e-8
    )
    for name in (
        "independent_objects",
        "reappearance_cases",
        "occluded_cases",
        "absent_cases",
        "annotated_frames",
    ):
        metrics[f"{name}_total"] = totals[name][0]
    return metrics


def quality_checks(metrics: dict) -> dict[str, bool]:
    return {
        "foreground_routes_to_objects": metrics["foreground_object_routing"] >= 0.70,
        "background_routes_to_scene": metrics["background_scene_routing"] >= 0.70,
        "object_relation_separated": metrics["object_relation_margin"] >= 0.30,
        "object_representation_is_minimal": metrics["object_effective_roots"] <= 2.00,
        "decoder_foreground_routes_to_objects": (
            metrics["decoder_foreground_object_routing"] >= 0.70
        ),
        "decoder_background_routes_to_scene": (
            metrics["decoder_background_scene_routing"] >= 0.70
        ),
        "decoder_objects_are_separated": (
            metrics["decoder_object_relation_margin"] >= 0.30
        ),
        "assignment_persistent": (
            metrics["object_assignment_temporal_similarity"] >= 0.80
        ),
        "identity_persistent": metrics["identity_temporal_cosine"] >= 0.90,
        "identity_reappears": metrics["reappearance_identity_cosine"] >= 0.80,
        "relative_geometry_decodable": metrics["relative_center_error"] <= 0.20,
        "visibility_decodable": metrics["visibility_accuracy"] >= 0.75,
        "presence_decodable": metrics["presence_accuracy"] >= 0.75,
        "slot_deletion_changes_object": metrics["deletion_inside"] >= 0.01,
        "slot_deletion_is_local": metrics["deletion_locality_ratio"] >= 1.50,
    }


def promotion_checks(metrics: dict, items: int, args) -> dict[str, bool]:
    return {
        "coverage_items": items >= args.minimum_items,
        "coverage_objects": (
            metrics["independent_objects_total"] >= args.minimum_objects
        ),
        "coverage_reappearance": (
            metrics["reappearance_cases_total"] >= args.minimum_reappearance_cases
        ),
        "coverage_occlusion": metrics["occluded_cases_total"] >= args.minimum_occluded_cases,
        "coverage_absence": metrics["absent_cases_total"] >= args.minimum_absent_cases,
        **quality_checks(metrics),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("independent v52 evaluation requires CUDA")
    checkpoint = torch.load(
        os.path.abspath(args.checkpoint), map_location="cpu", weights_only=False, mmap=True
    )
    validate_checkpoint_header(checkpoint)
    if checkpoint.get("git_commit") != args.source_revision:
        raise ValueError("independent evaluator source differs from checkpoint")
    if int(checkpoint.get("global_step", -1)) != args.expected_step:
        raise ValueError("independent evaluator requires the step-22500 checkpoint")
    config = LearningObjectiveObjectStateConfig()
    if checkpoint.get("config") != config.to_dict():
        raise ValueError("independent evaluator config differs from checkpoint")
    splits = tuple(value.strip() for value in args.splits.split(",") if value.strip())
    dataset = IndependentObjectTruthDataset(args.data, args.truth_manifest, splits)
    device = torch.device("cuda:0")
    model = LearningObjectiveObjectWorldModel(config).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    dino = FrozenDinoVideoRuntime(
        config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint
    )
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16" else nullcontext
    )
    run = init_wandb(args)
    totals: dict[str, list[float]] = {}
    split_totals: dict[str, dict[str, list[float]]] = {}
    split_items: dict[str, int] = {}
    with torch.no_grad():
        for index in range(len(dataset)):
            truth = move_truth(dataset[index], device)
            batch = {
                "video_rgb": truth["video_rgb"][None],
                "video_pixel_valid": truth["video_pixel_valid"][None],
            }
            features = dino(batch)
            with amp_context():
                _, state = model.encode_student(
                    features.patches,
                    features.coordinates,
                    features.valid,
                    truth["frame_times"][None],
                )
                _, decoder_assignment = model.decode_sequence(
                    state, features.coordinates, features.valid
                )
            values = independent_object_metrics(
                model, features, state, decoder_assignment, truth
            )
            values.update(
                independent_deletion_metrics(model, features, state, truth, amp_context)
            )
            add_statistics(totals, values)
            split = truth["split"]
            add_statistics(split_totals.setdefault(split, {}), values)
            split_items[split] = split_items.get(split, 0) + 1
            if run is not None:
                item_metrics = normalized_metrics({
                    name: [value_sum, value_count]
                    for name, (value_sum, value_count) in values.items()
                })
                run.log(
                    {
                        "evaluation/item_index": index,
                        **{f"item/{name}": value for name, value in item_metrics.items()},
                    },
                    step=index,
                )
    metrics = normalized_metrics(totals)
    checks = promotion_checks(metrics, len(dataset), args)
    conditions = {
        split: {
            "items": split_items[split],
            "metrics": normalized_metrics(values),
        }
        for split, values in split_totals.items()
    }
    for condition in conditions.values():
        condition["checks"] = quality_checks(condition["metrics"])
    passed = all(checks.values()) and all(
        all(condition["checks"].values()) for condition in conditions.values()
    )
    report = {
        "status": "passed" if passed else "failed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "contract": INDEPENDENT_OBJECT_TRUTH_CONTRACT,
        "training_git_commit": args.source_revision,
        "evaluator_git_commit": args.evaluator_revision,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_step": int(checkpoint["global_step"]),
        "truth_manifest": os.path.abspath(args.truth_manifest),
        "truth_provenance": dataset.provenance,
        "evaluation_scope": "external_stable_object_masks_and_lifecycle",
        "point_tracker_used": False,
        "training_teacher_used": False,
        "independent_object_truth_available": True,
        "deployment_promotion_ready": passed,
        "items": len(dataset),
        "metrics": metrics,
        "checks": checks,
        "conditions": conditions,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if run is not None:
        summary = {
            "evaluation/status": report["status"],
            "evaluation/deployment_promotion_ready": int(passed),
            **{f"evaluation/{name}": value for name, value in metrics.items()},
            **{f"evaluation/check_{name}": int(value) for name, value in checks.items()},
        }
        for split, condition in conditions.items():
            summary.update({
                f"evaluation/{split}/{name}": value
                for name, value in condition["metrics"].items()
            })
            summary.update({
                f"evaluation/{split}/check_{name}": int(value)
                for name, value in condition["checks"].items()
            })
        run.summary.update(summary)
        run.finish()
    print(json.dumps(report, sort_keys=True), flush=True)
    if not passed:
        raise RuntimeError("v52 independent Object State promotion gate failed")


if __name__ == "__main__":
    main()
