#!/usr/bin/env python3
"""Structural contract verification for the v66 G0 object motion field."""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.object_motion_field_v66 import (  # noqa: E402
    fit_object_motion_models_v66,
    predict_affine_v66,
    predict_motion_field_v66,
    predict_translation_v66,
    roll_motion_field_fit_v66,
)
from igsw.adaptive_gaussian_wm.v66_config import (  # noqa: E402
    CONTRACT,
    ObjectMotionFieldAuditConfigV66,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def synthetic_evidence(device):
    batch, frames, points = 4, 10, 12
    angle = torch.arange(points, device=device).float() * (2.0 * torch.pi / points)
    radius = 0.19 + 0.035 * torch.cos(3.0 * angle)
    base = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1)
    base = base[None].expand(batch, -1, -1).clone()
    phase = torch.arange(batch, device=device).float() * 0.55
    core_base = base[0, ::2]
    core_relative = core_base - core_base.mean(dim=0, keepdim=True)
    selected = [0]
    minimum_distance = (core_relative - core_relative[0]).square().sum(dim=-1)
    for _ in range(2):
        minimum_distance[selected] = -1.0
        index = int(minimum_distance.argmax().item())
        selected.append(index)
        distance = (core_relative - core_relative[index]).square().sum(dim=-1)
        minimum_distance = torch.minimum(minimum_distance.clamp_min(0.0), distance)
    centers = core_relative[selected]
    mode_scale = core_relative.square().sum(dim=-1).mean().sqrt() * 0.75
    relative = base - base[:, ::2].mean(dim=1, keepdim=True)
    distance = (relative[:, :, None] - centers[None, None]).square().sum(dim=-1)
    radial = torch.softmax(-distance / (2.0 * mode_scale**2), dim=-1)
    coefficients = []
    for offset in (0.0, 2.1, 4.2):
        coefficients.append(
            0.020
            * torch.stack(
                (
                    torch.cos(phase + offset),
                    torch.sin(phase + offset),
                ),
                dim=-1,
            )
        )
    coefficients = torch.stack(coefficients, dim=1)
    local_velocity = torch.einsum("bpk,bkd->bpd", radial, coefficients)
    translation = 0.012 * torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1)
    velocity = local_velocity + translation[:, None]
    time = torch.arange(frames, device=device).float()
    coordinates = base[:, None] + time[None, :, None, None] * velocity[:, None]
    visibility = torch.ones(batch, frames, points, device=device, dtype=torch.bool)
    reliability = torch.ones(batch, points, device=device)
    evidence = SimpleNamespace(
        coordinates=coordinates,
        visibility=visibility,
        reliability=reliability,
    )
    core = torch.zeros(batch, points, device=device, dtype=torch.bool)
    core[:, ::2] = True
    holdout = ~core
    return evidence, core, holdout


def masked_error(prediction, target, membership):
    weight = membership[:, None].float()
    error = (prediction.float() - target.float()).norm(dim=-1)
    weight = weight.expand_as(error)
    return (error * weight).sum(dim=(1, 2)) / weight.sum(dim=(1, 2))


def max_fit_difference(first, second):
    fields = (
        "translation",
        "affine",
        "mode_centers",
        "mode_scale",
        "mode_coefficients",
        "reference_center",
    )
    return max(
        float((getattr(first, name) - getattr(second, name)).abs().max())
        for name in fields
    )


def verify(device):
    config = ObjectMotionFieldAuditConfigV66(
        carrier_count=2,
        tracker_grid_side=4,
        tracker_anchor_fractions=(0.0,),
        core_tracks=6,
        core_candidate_tracks=8,
        minimum_component_tracks=4,
        minimum_holdout_tracks=2,
        local_motion_modes=3,
        transition_ridge=1e-5,
        motion_field_ridge=1e-5,
    )
    config.validate()
    evidence, core, holdout = synthetic_evidence(device)
    split = evidence.coordinates.shape[1] // 2
    fit = fit_object_motion_models_v66(evidence, core, 0, split, config)
    source = evidence.coordinates[:, split - 1]
    target_index = torch.as_tensor(
        [split - 1 + horizon for horizon in config.transition_horizons],
        device=device,
    )
    target = evidence.coordinates.index_select(1, target_index)
    persistence = source[:, None].expand_as(target)
    translation = predict_translation_v66(source, fit)
    affine = predict_affine_v66(source, fit)
    field = predict_motion_field_v66(source, fit)
    shuffled = predict_motion_field_v66(source, roll_motion_field_fit_v66(fit))
    errors = {
        "persistence": masked_error(persistence, target, holdout),
        "translation": masked_error(translation, target, holdout),
        "affine": masked_error(affine, target, holdout),
        "motion_field": masked_error(field, target, holdout),
        "shuffled_field": masked_error(shuffled, target, holdout),
    }

    holdout_corrupted = SimpleNamespace(
        coordinates=evidence.coordinates.clone(),
        visibility=evidence.visibility,
        reliability=evidence.reliability,
    )
    holdout_corrupted.coordinates[:, :split, holdout[0]] += 4.0
    holdout_fit = fit_object_motion_models_v66(
        holdout_corrupted, core, 0, split, config
    )
    future_corrupted = SimpleNamespace(
        coordinates=evidence.coordinates.clone(),
        visibility=evidence.visibility,
        reliability=evidence.reliability,
    )
    future_corrupted.coordinates[:, split:] -= 3.0
    future_fit = fit_object_motion_models_v66(
        future_corrupted, core, 0, split, config
    )
    holdout_isolation = max_fit_difference(fit, holdout_fit)
    future_isolation = max_fit_difference(fit, future_fit)
    means = {name: float(value.mean()) for name, value in errors.items()}
    checks = {
        "four_gpu_contract": int(os.environ.get("WORLD_SIZE", "1")) in (1, 4),
        "core_and_holdout_are_disjoint": not bool((core & holdout).any()),
        "shared_three_mode_capacity": fit.mode_coefficients.shape[2:] == (3, 2),
        "all_fits_are_valid": bool(fit.field_valid.all()),
        "all_outputs_are_finite": all(bool(value.isfinite().all()) for value in errors.values()),
        "holdout_tracks_do_not_change_fit": holdout_isolation < 1e-7,
        "future_frames_do_not_change_fit": future_isolation < 1e-7,
        "field_fits_prefix_better_than_affine": bool(
            (fit.field_prefix_error < fit.affine_prefix_error).all()
        ),
        "field_beats_persistence_on_holdout": means["motion_field"] < means["persistence"],
        "field_beats_translation_on_holdout": means["motion_field"] < means["translation"],
        "field_beats_affine_on_holdout": means["motion_field"] < means["affine"],
        "correct_field_beats_shuffled_sample": means["motion_field"] < means["shuffled_field"],
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "contract": CONTRACT,
        "checks": checks,
        "holdout_coordinate_errors": means,
        "prefix_errors": {
            "translation": float(fit.translation_prefix_error.mean()),
            "affine": float(fit.affine_prefix_error.mean()),
            "motion_field": float(fit.field_prefix_error.mean()),
        },
        "holdout_fit_isolation_max_difference": holdout_isolation,
        "future_fit_isolation_max_difference": future_isolation,
        "motion_field_parameterization": "one affine plus three shared local RBF residual modes",
    }


def main():
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    report = verify(device)
    report.update(
        {
            "source_revision": args.source_revision,
            "world_size": world_size,
            "device": str(device),
        }
    )
    local_pass = torch.tensor(
        int(report["status"] == "passed"), device=device, dtype=torch.int32
    )
    if world_size > 1:
        dist.all_reduce(local_pass, op=dist.ReduceOp.MIN)
    report["status"] = "passed" if int(local_pass) == 1 else "failed"
    if local_rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps(report, sort_keys=True))
    if world_size > 1:
        dist.destroy_process_group()
    if report["status"] != "passed":
        raise RuntimeError("v66 object motion-field structural contract failed")


if __name__ == "__main__":
    main()
