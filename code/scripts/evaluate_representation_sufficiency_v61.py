"""Compare trained V61 Object State checkpoints on held representation utility."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.carrier_teacher_v61 import (  # noqa: E402
    FrozenSiglipObjectTeacherV61,
    build_object_components_v61,
)
from igsw.adaptive_gaussian_wm.continuous_carrier_world_model_v61 import (  # noqa: E402
    ContinuousCarrierObjectWorldModelV61,
)
from igsw.adaptive_gaussian_wm.frozen_video_encoder import (  # noqa: E402
    FrozenDinoVideoRuntime,
)
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourceRobotVideoDataset,
)
from igsw.adaptive_gaussian_wm.point_track_teacher import (  # noqa: E402
    FrozenPointTrackerRuntime,
)
from igsw.adaptive_gaussian_wm.representation_sufficiency_v61 import (  # noqa: E402
    condition_from_v61_output,
    evaluate_representation_sufficiency_v61,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.trajectory_relation_teacher_v56 import (  # noqa: E402
    build_trajectory_relation_teacher_v56,
)
from igsw.adaptive_gaussian_wm.v61_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    config_for_variant,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="variant=/absolute/path/latest.pt; pass once per compared variant",
    )
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--siglip_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk_lengths", default="4,6,8")
    parser.add_argument("--temporal_step_ms", default="100,200,400")
    parser.add_argument("--held_group_stride", type=int, default=20)
    parser.add_argument("--conditions_per_source", type=int, default=8)
    parser.add_argument("--student_frame_batch", type=int, default=32)
    parser.add_argument("--dino_frame_batch", type=int, default=32)
    parser.add_argument("--siglip_teacher_batch", type=int, default=32)
    parser.add_argument("--probe_sample_limit", type=int, default=16384)
    parser.add_argument("--probe_mlp_steps", type=int, default=100)
    parser.add_argument("--active_variance_floor", type=float, default=1e-4)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--wandb_mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", required=True)
    parser.add_argument("--wandb_group", default="v61-representation-sufficiency-eval")
    parser.add_argument("--wandb_dir", required=True)
    parser.add_argument("--evaluator_revision", default="")
    return parser.parse_args()


def parse_checkpoints(values):
    parsed = {}
    for value in values:
        variant, path = value.split("=", 1)
        if variant in parsed:
            raise ValueError(f"duplicate V61 evaluator variant: {variant}")
        parsed[variant] = os.path.abspath(path)
    return parsed


def load_models(args, paths, device):
    models, headers = {}, {}
    for variant, path in paths.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError(f"{variant} is not a version-61 checkpoint")
        if checkpoint.get("architecture") != ARCHITECTURE:
            raise ValueError(f"{variant} checkpoint architecture differs")
        if checkpoint["config"]["variant"] != variant:
            raise ValueError(f"{variant} checkpoint contains another variant")
        config = config_for_variant(variant)
        model = (
            ContinuousCarrierObjectWorldModelV61(
                config,
                args.dino_checkpoint,
                args.siglip_checkpoint,
                args.student_frame_batch,
            )
            .to(device)
            .eval()
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        models[variant] = model
        headers[variant] = {
            "path": path,
            "global_step": int(checkpoint["global_step"]),
            "git_commit": checkpoint.get("git_commit", ""),
            "seed": int(checkpoint["args"]["seed"]),
            "held_group_stride": int(checkpoint["args"]["held_group_stride"]),
        }
    return models, headers


def batch_sample(sample, device):
    return move_to_device(
        {
            name: value[None] if torch.is_tensor(value) else value
            for name, value in sample.items()
        },
        device,
    )


def init_wandb(args, paths, headers):
    if args.wandb_mode == "disabled":
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name,
        group=args.wandb_group,
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        job_type="representation-sufficiency-eval",
        config={
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "checkpoints": paths,
            "checkpoint_headers": headers,
            "data_index": os.path.abspath(args.data_index),
            "chunk_lengths": args.chunk_lengths,
            "temporal_step_ms": args.temporal_step_ms,
            "held_group_stride": args.held_group_stride,
            "conditions_per_source": args.conditions_per_source,
            "probe_sample_limit": args.probe_sample_limit,
            "probe_mlp_steps": args.probe_mlp_steps,
            "active_variance_floor": args.active_variance_floor,
            "truth_scope": "held_training_teacher_not_independent_object_truth",
            "evaluator_revision": args.evaluator_revision,
        },
    )


def comparison_metrics(results):
    variants = list(results)
    if len(variants) != 2:
        return {}
    base, candidate = variants
    shared = sorted(set(results[base]) & set(results[candidate]))
    return {
        f"comparison/{candidate}_minus_{base}/{name}": results[candidate][name]
        - results[base][name]
        for name in shared
    }


def breakdown_metrics(records, dataset, args):
    output = {"source": {}, "chunk_length": {}}
    for source_index, source_name in enumerate(dataset.source_names):
        selected = [item for item in records if item.source_index == source_index]
        output["source"][source_name] = evaluate_representation_sufficiency_v61(
            selected,
            variance_floor=args.active_variance_floor,
            sample_limit=args.probe_sample_limit,
            mlp_steps=args.probe_mlp_steps,
            include_probes=False,
        )
    for chunk_length in sorted({item.chunk_length for item in records}):
        selected = [item for item in records if item.chunk_length == chunk_length]
        output["chunk_length"][str(chunk_length)] = (
            evaluate_representation_sufficiency_v61(
                selected,
                variance_floor=args.active_variance_floor,
                sample_limit=args.probe_sample_limit,
                mlp_steps=args.probe_mlp_steps,
                include_probes=False,
            )
        )
    return output


def main():
    args = parse_args()
    paths = parse_checkpoints(args.checkpoint)
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    models, headers = load_models(args, paths, device)
    seeds = {header["seed"] for header in headers.values()}
    strides = {header["held_group_stride"] for header in headers.values()}
    if len(seeds) != 1 or strides != {args.held_group_stride}:
        raise ValueError("compared V61 checkpoints do not share seed and held split")
    dataset = MultiSourceRobotVideoDataset(
        args.data_index,
        "train",
        args.chunk_lengths,
        args.temporal_step_ms,
        max_items=0,
        seed=next(iter(seeds)),
        group_partition="held",
        held_group_stride=args.held_group_stride,
    )
    dino = FrozenDinoVideoRuntime(
        next(iter(models.values())).config,
        device,
        args.amp,
        args.dino_frame_batch,
        args.dino_checkpoint,
    )
    tracker = FrozenPointTrackerRuntime(
        next(iter(models.values())).config,
        device,
        args.tracker_checkpoint,
        sequence_batch=1,
    )
    semantic_teacher = FrozenSiglipObjectTeacherV61(
        args.siglip_checkpoint, device, args.siglip_teacher_batch
    )
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    run = init_wandb(args, paths, headers)
    records = {variant: [] for variant in models}
    chunk_lengths = tuple(int(value) for value in args.chunk_lengths.split(","))
    condition_index = 0
    with torch.inference_mode():
        for source_index in range(len(dataset.source_names)):
            indices = dataset.balanced_source_evaluation_indices(
                source_index, args.conditions_per_source
            )
            for base_index in indices:
                for chunk_length in chunk_lengths:
                    batch = batch_sample(dataset[(base_index, chunk_length)], device)
                    teacher_features = dino(batch)
                    evidence = tracker(
                        batch, teacher_features.patches, teacher_features.grid_hw
                    )
                    relation = build_trajectory_relation_teacher_v56(
                        evidence,
                        next(iter(models.values())).config,
                        batch["frame_times"],
                    )
                    components = build_object_components_v61(
                        batch,
                        evidence,
                        relation,
                        next(iter(models.values())).config.object_roots,
                        semantic_teacher,
                    )
                    progress = {
                        "progress/condition": condition_index + 1,
                        "progress/source_index": source_index,
                        "progress/chunk_length": chunk_length,
                    }
                    for variant, model in models.items():
                        with amp_context():
                            output = model(batch, evidence, relation, components)
                        records[variant].append(
                            condition_from_v61_output(output, evidence, relation, batch)
                        )
                        progress[f"progress/{variant}/object_state_loss"] = float(
                            output["parts"]["object_state_loss"]
                        )
                    if run is not None:
                        run.log(progress, step=condition_index)
                    condition_index += 1
    results = {
        variant: evaluate_representation_sufficiency_v61(
            values,
            variance_floor=args.active_variance_floor,
            sample_limit=args.probe_sample_limit,
            mlp_steps=args.probe_mlp_steps,
        )
        for variant, values in records.items()
    }
    breakdowns = {
        variant: breakdown_metrics(values, dataset, args)
        for variant, values in records.items()
    }
    comparison = comparison_metrics(results)
    report = {
        "status": "completed",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "evaluator_revision": args.evaluator_revision,
        "truth_scope": "held_training_teacher_not_independent_object_truth",
        "conditions_per_variant": condition_index,
        "source_names": dataset.source_names,
        "checkpoint_headers": headers,
        "metrics": results,
        "breakdowns": breakdowns,
        "comparison": comparison,
        "promotion_decision": "not_automatic_requires_g2_and_independent_g3_review",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if run is not None:
        flattened = {
            f"sufficiency/{variant}/{name}": value
            for variant, metrics in results.items()
            for name, value in metrics.items()
        }
        flattened.update(comparison)
        flattened.update(
            {
                f"breakdown/{variant}/{axis}/{key}/{name}": value
                for variant, variant_breakdown in breakdowns.items()
                for axis, groups in variant_breakdown.items()
                for key, metrics in groups.items()
                for name, value in metrics.items()
            }
        )
        run.log(flattened, step=condition_index)
        run.summary.update(flattened)
        run.summary.update(
            {
                "evaluation/status": "completed",
                "evaluation/truth_scope": report["truth_scope"],
                "evaluation/conditions_per_variant": condition_index,
                "evaluation/report": os.path.abspath(args.output),
                "evaluation/promotion_decision": report["promotion_decision"],
            }
        )
        run.finish()
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
