"""Evaluate posterior, deterministic branch, and flow-prior coverage."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.latent_particle_wm.action_field import (  # noqa: E402
    ActionFieldConfig,
    CorrelatedActionField,
)
from igsw.latent_particle_wm.action_metrics import (  # noqa: E402
    MetricAccumulator,
    add_prediction_metrics,
    add_prior_coverage_metrics,
    point_errors,
)
from igsw.latent_particle_wm.pair_data import CausalPairDataset  # noqa: E402
from igsw.latent_particle_wm.rendering import render_dino, render_rgb  # noqa: E402


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def add_render_metrics(
    accumulator: MetricAccumulator,
    prefix: str,
    model: CorrelatedActionField,
    batch: dict,
    dense_field: torch.Tensor,
    render_height: int,
    render_width: int,
) -> None:
    rendered, alpha = render_rgb(batch, dense_field, render_height, render_width)
    target = F.interpolate(
        batch["rgb1"],
        size=(render_height, render_width),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    mse = (rendered - target).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    l1 = (rendered - target).abs().mean(dim=(1, 2, 3))
    for value in 10.0 * torch.log10(1.0 / mse):
        accumulator.scalar(f"{prefix}/rgb_psnr_db", float(value))
    for value in l1:
        accumulator.scalar(f"{prefix}/rgb_l1", float(value))
    for value in alpha.mean(dim=(1, 2, 3)):
        accumulator.scalar(f"{prefix}/alpha_mean", float(value))
    if model.config.dino_dim:
        rendered_dino, _ = render_dino(batch, dense_field, model.config.dino_dim)
        similarity = (
            F.normalize(rendered_dino, dim=-1)
            * F.normalize(batch["dino1"], dim=-1)
        ).sum(dim=-1).mean(dim=(1, 2))
        for value in similarity:
            accumulator.scalar(f"{prefix}/dino_cosine", float(value))


def update_grouped(
    sums: dict,
    counts: dict,
    group: str,
    batch: dict,
    posterior: torch.Tensor,
    deterministic: torch.Tensor,
    prior_samples: torch.Tensor,
) -> None:
    posterior_error, _ = point_errors(batch, posterior)
    deterministic_error, _ = point_errors(batch, deterministic)
    prior_error, _ = point_errors(batch, prior_samples)
    valid = batch["motion_valid"]
    for index in range(len(valid)):
        mask = valid[index]
        if not bool(mask.any()):
            continue
        prior_clip_oracle = prior_error[:, index, mask].mean(dim=-1).min()
        values = {
            "posterior_flow_epe_px": posterior_error[index, mask].mean(),
            "deterministic_flow_epe_px": deterministic_error[index, mask].mean(),
            "prior_best_point_flow_epe_px": prior_error[:, index, mask].amin(dim=0).mean(),
            "prior_best_clip_flow_epe_px": prior_clip_oracle,
        }
        key = str(batch[group][index]) if group == "task" else str(int(batch[group][index]))
        for metric, value in values.items():
            sums[key][metric] += float(value)
            counts[key][metric] += 1


def finalize_grouped(sums: dict, counts: dict) -> dict:
    return {
        group: {
            metric: sums[group][metric] / counts[group][metric]
            for metric in sorted(sums[group])
            if counts[group][metric]
        }
        for group in sorted(sums)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--split", choices=("train", "heldseed", "heldtask"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--render_batches", type=int, default=8)
    parser.add_argument("--render_height", type=int, default=98)
    parser.add_argument("--render_width", type=int, default=130)
    parser.add_argument("--seed", type=int, default=117)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("samples must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ActionFieldConfig(**checkpoint["config"])
    model = CorrelatedActionField(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = CausalPairDataset(
        args.data,
        args.split,
        config.control_rows,
        config.control_cols,
        checkpoint["args"]["active_count"],
        args.dino,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    accumulator = MetricAccumulator()
    task_sums: dict = defaultdict(lambda: defaultdict(float))
    task_counts: dict = defaultdict(lambda: defaultdict(int))
    horizon_sums: dict = defaultdict(lambda: defaultdict(float))
    horizon_counts: dict = defaultdict(lambda: defaultdict(int))
    task_histogram = Counter()
    horizon_histogram = Counter()
    causal_report = {}
    with torch.inference_mode():
        for batch_index, cpu_batch in enumerate(loader):
            batch = move_to_device(cpu_batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                posterior = model.forward_posterior(batch)
                center = model.sample_prior(batch, 1, stochastic=False)
                samples = model.sample_prior(batch, args.samples, stochastic=True)
            zero = torch.zeros_like(posterior["control_field"])
            add_prediction_metrics(accumulator, "zero", batch, zero)
            add_prediction_metrics(
                accumulator,
                "deterministic",
                batch,
                posterior["deterministic_control_field"],
            )
            add_prediction_metrics(
                accumulator,
                "posterior",
                batch,
                posterior["control_field"],
            )
            add_prediction_metrics(
                accumulator,
                "prior_center",
                batch,
                center["control_field"][0],
            )
            add_prior_coverage_metrics(accumulator, batch, samples["control_field"])

            posterior_error, _ = point_errors(batch, posterior["control_field"])
            deterministic_error, _ = point_errors(
                batch,
                posterior["deterministic_control_field"],
            )
            valid = batch["motion_valid"]
            for clip in range(len(valid)):
                if bool(valid[clip].any()):
                    delta = (
                        posterior_error[clip, valid[clip]].mean()
                        - deterministic_error[clip, valid[clip]].mean()
                    )
                    accumulator.scalar("paired/posterior_minus_deterministic_px", float(delta))
            update_grouped(
                task_sums,
                task_counts,
                "task",
                batch,
                posterior["control_field"],
                posterior["deterministic_control_field"],
                samples["control_field"],
            )
            update_grouped(
                horizon_sums,
                horizon_counts,
                "horizon",
                batch,
                posterior["control_field"],
                posterior["deterministic_control_field"],
                samples["control_field"],
            )
            task_histogram.update(batch["task"])
            horizon_histogram.update(int(value) for value in batch["horizon"])

            if batch_index < args.render_batches:
                add_render_metrics(
                    accumulator,
                    "posterior",
                    model,
                    batch,
                    posterior["dense_field"],
                    args.render_height,
                    args.render_width,
                )
                add_render_metrics(
                    accumulator,
                    "deterministic",
                    model,
                    batch,
                    posterior["deterministic_dense_field"],
                    args.render_height,
                    args.render_width,
                )
                add_render_metrics(
                    accumulator,
                    "prior_center",
                    model,
                    batch,
                    center["dense_field"][0],
                    args.render_height,
                    args.render_width,
                )

            if not causal_report:
                order = torch.arange(len(batch["state"]) - 1, -1, -1, device=device)
                shuffled = dict(batch)
                for key in (
                    "target",
                    "motion_valid",
                    "visible",
                    "matched",
                    "target_xyz",
                    "rgb1",
                    "dino1",
                ):
                    shuffled[key] = batch[key][order]
                first_context = model.encode_current(batch)["prior_context"]
                second_context = model.encode_current(shuffled)["prior_context"]
                second_posterior = model.forward_posterior(shuffled)
                causal_report = {
                    "prior_context_future_swap_max_abs_difference": float(
                        (first_context - second_context).abs().max()
                    ),
                    "posterior_future_swap_max_abs_difference": float(
                        (posterior["actions"] - second_posterior["actions"]).abs().max()
                    ),
                }
            if (batch_index + 1) % 25 == 0:
                print(f"[evaluate-action-field] {batch_index + 1}/{len(loader)}", flush=True)

    metrics = accumulator.result()
    coverage = [
        metrics[f"prior_best_{count}/flow_epe_px_point"]
        for count in (1, 2, 4, 8, 16)
        if count <= args.samples
    ]
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "data": os.path.abspath(args.data),
        "split": args.split,
        "count": len(dataset),
        "samples": args.samples,
        "metrics": metrics,
        "by_task": finalize_grouped(task_sums, task_counts),
        "by_horizon": finalize_grouped(horizon_sums, horizon_counts),
        "task_counts": dict(sorted(task_histogram.items())),
        "horizon_counts": dict(sorted(horizon_histogram.items())),
        "coverage_monotonic_nonincreasing": all(
            later <= earlier + 1e-8
            for earlier, later in zip(coverage, coverage[1:])
        ),
        "causal": causal_report,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    temporary = f"{args.out}.tmp.{os.getpid()}"
    with open(temporary, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    os.replace(temporary, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
