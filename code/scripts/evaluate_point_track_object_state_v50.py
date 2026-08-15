"""Held-video evaluation and promotion gates for both v50 stages."""

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
from igsw.adaptive_gaussian_wm.point_track_world_model import PointTrackObjectWorldModel  # noqa: E402
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v50_checkpointing import validate_checkpoint_header  # noqa: E402
from igsw.adaptive_gaussian_wm.v50_config import ARCHITECTURE, CHECKPOINT_VERSION, STAGES, PointTrackObjectStateConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--split", default="heldseed")
    parser.add_argument("--items", type=int, default=128)
    parser.add_argument("--chunk_length", type=int, default=16)
    parser.add_argument("--temporal_stride", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dino_frame_batch", type=int, default=32)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--wandb_mode", choices=("disabled", "online", "offline"), default="disabled")
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_group", default="point-track-object-state-v50-eval")
    parser.add_argument("--wandb_dir", default="")
    return parser.parse_args()


def weighted_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def track_identity_metrics(model, state, match):
    prediction = F.normalize(
        model.identity_readout(state["identity"].float()), dim=-1, eps=1e-6
    )
    target = match.identity[:, None]
    visible = match.visibility * match.component_valid[:, None].float()
    occluded_before = (match.lifecycle_state == 1).float().cumsum(dim=1) > 0
    weight = visible * occluded_before.float()
    correct = weighted_mean((prediction * target).sum(dim=-1), weight)
    shuffled = weighted_mean(
        (prediction * target.roll(1, dims=2)).sum(dim=-1), weight
    )
    return correct, shuffled, int(weight.sum())


def deletion_locality(model, features, evidence, output, amp_context):
    state = output["state"]
    frame_motion = torch.cat(
        (evidence.motion_salience, torch.zeros_like(evidence.motion_salience[:, :1])), dim=1
    ).sum(dim=2)
    frame = int(frame_motion[0].argmax())
    teacher_owner = output["match"].target_student_owner[0, :, : model.config.object_slots]
    track_motion = evidence.motion_salience[0].mean(dim=0)
    slot_motion = torch.einsum("p,pk->k", track_motion, teacher_owner)
    slot = int(slot_motion.argmax())
    frame_state = {
        name: value[:, frame]
        for name, value in state.items()
        if name not in ("assignment", "mass")
    }
    with amp_context():
        reference, _ = model.decoder(frame_state, features.coordinates[:, frame], features.valid[:, frame])
        object_valid = torch.ones(1, model.config.object_slots, device=reference.device, dtype=torch.bool)
        object_valid[:, slot] = False
        deleted, _ = model.decoder(
            frame_state, features.coordinates[:, frame], features.valid[:, frame], object_valid
        )
    change = 1.0 - F.cosine_similarity(reference.float(), deleted.float(), dim=-1, eps=1e-6)
    assigned_tracks = teacher_owner[:, slot] > 0.5
    assigned_tracks = assigned_tracks & evidence.visibility[0, frame]
    if not bool(assigned_tracks.any()):
        return change.new_tensor(0.0), change.new_tensor(0.0), 0
    track_positions = evidence.coordinates[0, frame, assigned_tracks]
    patch_positions = features.coordinates[0, frame]
    distance = (patch_positions[:, None] - track_positions[None]).norm(dim=-1).amin(dim=1)
    inside = (distance <= 0.18) & features.valid[0, frame]
    outside = (distance >= 0.30) & features.valid[0, frame]
    if not bool(inside.any()) or not bool(outside.any()):
        return change.new_tensor(0.0), change.new_tensor(0.0), 0
    inside_change = change[0, inside].mean()
    outside_change = change[0, outside].mean()
    return inside_change, outside_change, 1


def probe_examples(state, match):
    feature = torch.cat((state["identity"][:, :-1], state["dynamic"][:, :-1]), dim=-1).float()
    return (
        feature.reshape(-1, feature.shape[-1]),
        match.motion.reshape(-1, 2),
        match.motion_valid.reshape(-1).float(),
        match.visibility[:, :-1].reshape(-1, 1),
        match.lifecycle_known[:, :-1].reshape(-1).float(),
    )


def ridge_relative_gain(features, target, weight):
    keep = weight > 1e-4
    x, y = features[keep].float(), target[keep].float()
    if len(x) < 16:
        return -1.0
    order = torch.arange(len(x), device=x.device)
    train, test = order % 2 == 0, order % 2 == 1
    mean, scale = x[train].mean(0), x[train].std(0, unbiased=False).clamp_min(1e-4)
    normalized = (x - mean) / scale
    normalized = torch.cat((normalized, torch.ones(len(normalized), 1, device=x.device)), dim=1)
    xtx = normalized[train].T @ normalized[train]
    ridge = 1e-2 * torch.eye(len(xtx), device=x.device)
    coefficient = torch.linalg.solve(xtx + ridge, normalized[train].T @ y[train])
    prediction = normalized[test] @ coefficient
    mse = (prediction - y[test]).square().mean()
    baseline = (y[test] - y[train].mean(0)).square().mean().clamp_min(1e-8)
    return float((baseline - mse) / baseline)


def state_decision(metrics):
    checks = {
        "teacher_owner_gain": metrics["teacher_owner_gain_over_shuffled"] >= 0.05,
        "identity_gain": metrics["track_identity_gain_over_shuffled"] >= 0.05,
        "reappearance_present": metrics["reappearance_pairs"] >= 8,
        "reappearance_identity": metrics["reappearance_correct_cosine"] > metrics["reappearance_shuffled_cosine"],
        "deletion_locality": metrics["deletion_locality_ratio"] >= 1.25,
        "motion_probe": metrics["motion_probe_relative_gain"] >= 0.05,
        "visibility_probe": metrics["visibility_probe_relative_gain"] >= 0.05,
        "moving_tracks_use_objects": metrics["moving_track_object_fraction"] >= 0.50,
    }
    return checks, all(checks.values())


def evaluate(args, model, loader, dino, tracker, device, amp_context):
    sums, counts = {}, {}
    probe_x, probe_motion, probe_motion_weight, probe_visibility, probe_visibility_weight = [], [], [], [], []
    with torch.no_grad():
        for batch_index, cpu_batch in enumerate(loader):
            batch = move_to_device(cpu_batch, device)
            features = dino(batch)
            evidence = (
                tracker(batch, features.patches, features.grid_hw)
                if args.stage == "object_state"
                else None
            )
            with amp_context():
                output = model(
                    features.patches, features.coordinates, features.valid,
                    batch["frame_times"], evidence, features.grid_hw,
                )
            for name, value in output["parts"].items():
                sums[name] = sums.get(name, 0.0) + float(value)
                counts[name] = counts.get(name, 0) + 1
            if args.stage == "latent_effect":
                continue
            correct, shuffled, pairs = track_identity_metrics(
                model, output["state"], output["match"]
            )
            inside, outside, deletion_valid = deletion_locality(model, features, evidence, output, amp_context)
            motion = torch.cat((evidence.motion_salience, torch.zeros_like(evidence.motion_salience[:, :1])), dim=1)
            moving = motion >= 0.5
            object_fraction = output["match"].sampled_student_assignment[..., : model.config.object_slots].sum(-1)
            for name, value in (
                ("reappearance_correct_cosine", float(correct)),
                ("reappearance_shuffled_cosine", float(shuffled)),
            ):
                sums[name] = sums.get(name, 0.0) + value * pairs
                counts[name] = counts.get(name, 0) + pairs
            sums["reappearance_pairs"] = sums.get("reappearance_pairs", 0.0) + pairs
            counts["reappearance_pairs"] = 1
            for name, value in (
                ("deletion_inside_change", float(inside)),
                ("deletion_outside_change", float(outside)),
            ):
                sums[name] = sums.get(name, 0.0) + value * deletion_valid
                counts[name] = counts.get(name, 0) + deletion_valid
            sums["deletion_valid_items"] = sums.get("deletion_valid_items", 0.0) + deletion_valid
            counts["deletion_valid_items"] = 1
            moving_fraction = float(weighted_mean(object_fraction, moving.float()))
            sums["moving_track_object_fraction"] = sums.get("moving_track_object_fraction", 0.0) + moving_fraction
            counts["moving_track_object_fraction"] = counts.get("moving_track_object_fraction", 0) + 1
            x, motion_target, motion_weight, visibility_target, visibility_weight = probe_examples(
                output["state"], output["match"]
            )
            probe_x.append(x.cpu())
            probe_motion.append(motion_target.cpu())
            probe_motion_weight.append(motion_weight.cpu())
            probe_visibility.append(visibility_target.cpu())
            probe_visibility_weight.append(visibility_weight.cpu())
            if batch_index + 1 >= args.items:
                break
    metrics = {name: sums[name] / max(counts[name], 1) for name in sums}
    if args.stage == "object_state":
        x = torch.cat(probe_x)
        metrics["motion_probe_relative_gain"] = ridge_relative_gain(x, torch.cat(probe_motion), torch.cat(probe_motion_weight))
        metrics["visibility_probe_relative_gain"] = ridge_relative_gain(x, torch.cat(probe_visibility), torch.cat(probe_visibility_weight))
        metrics["deletion_locality_ratio"] = metrics["deletion_inside_change"] / max(metrics["deletion_outside_change"], 1e-8)
        metrics["track_identity_gain_over_shuffled"] = (
            metrics["reappearance_correct_cosine"] - metrics["reappearance_shuffled_cosine"]
        ) / max(1.0 - metrics["reappearance_shuffled_cosine"], 1e-6)
    return metrics


def log_wandb(args, report, step):
    if args.wandb_mode == "disabled":
        return
    if not args.wandb_project or not args.wandb_dir:
        raise ValueError("v50 evaluation W&B requires project and directory")
    import wandb
    os.makedirs(args.wandb_dir, exist_ok=True)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name or None,
        group=args.wandb_group or None,
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        config={name: value for name, value in vars(args).items()},
        job_type=f"v50-{args.stage}-held-evaluation",
    )
    payload = {
        f"eval/{name}": float(value) if isinstance(value, (int, float, bool)) else value
        for name, value in report["metrics"].items()
    }
    payload.update({f"gate/{name}": int(value) for name, value in report["checks"].items()})
    run.log(payload, step=step)
    run.summary.update({"gate/status": report["status"], "gate/checkpoint": report["checkpoint"]})
    run.finish()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v50 held evaluation requires CUDA")
    device = torch.device("cuda:0")
    checkpoint_path = os.path.abspath(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    validate_checkpoint_header(checkpoint)
    if checkpoint.get("stage") != args.stage:
        raise ValueError("v50 evaluation stage differs from checkpoint")
    if checkpoint.get("git_commit") != args.source_revision:
        raise ValueError("v50 evaluation source revision differs from checkpoint")
    config = PointTrackObjectStateConfig()
    if checkpoint.get("config") != config.to_dict():
        raise ValueError("v50 evaluation config differs from checkpoint")
    dataset = PointTrackObjectVideoDataset(
        args.data, args.split, str(args.chunk_length), str(args.temporal_stride), args.items, args.seed
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    model = PointTrackObjectWorldModel(config, args.stage).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    dino = FrozenDinoVideoRuntime(config, device, args.amp, args.dino_frame_batch, args.dino_checkpoint)
    tracker = (
        FrozenPointTrackerRuntime(config, device, args.tracker_checkpoint, 1)
        if args.stage == "object_state"
        else None
    )
    amp_context = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.amp == "bf16" else nullcontext
    metrics = evaluate(args, model, loader, dino, tracker, device, amp_context)
    if args.stage == "object_state":
        checks, passed = state_decision(metrics)
    else:
        checks = {
            "correct_effect_beats_zero_by_10pct": metrics["effect_gain_over_zero"] >= 0.10,
            "correct_effect_beats_shuffled_by_10pct": metrics["effect_gain_over_shuffled"] >= 0.10,
        }
        passed = all(checks.values())
    report = {
        "status": "passed" if passed else "failed",
        "evaluation_stage": args.stage,
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "git_commit": args.source_revision,
        "checkpoint": checkpoint_path,
        "checkpoint_step": int(checkpoint["global_step"]),
        "data": os.path.abspath(args.data),
        "split": args.split,
        "items": min(args.items, len(dataset)),
        "evaluation_scope": "held_teacher_consistency_not_independent_object_proof",
        "deployable_student_reads_point_tracks": False,
        "checks": checks,
        "metrics": metrics,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = f"{output}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)
    log_wandb(args, report, int(checkpoint["global_step"]))
    print(json.dumps(report, sort_keys=True), flush=True)
    if not passed:
        raise RuntimeError(f"v50 {args.stage} held gate failed")


if __name__ == "__main__":
    main()
