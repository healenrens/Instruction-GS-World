"""Measure best-of-N coverage of a trained global latent-action prior."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.latent_particle_wm.metrics import prediction_xyz  # noqa: E402
from igsw.latent_particle_wm.models import ParticleWorldModel, WorldModelConfig  # noqa: E402
from igsw.latent_particle_wm.probe_data import ParticleProbeDataset  # noqa: E402


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def update(
    totals: dict[str, float],
    squares: dict[str, float],
    counts: dict[str, int],
    batch: dict,
    samples: torch.Tensor,
    baseline_prediction: torch.Tensor,
    sample_counts: tuple[int, ...],
    prefix: str,
) -> None:
    valid = batch["motion_valid"]
    mover = (batch["target"][..., :2].norm(dim=-1) > 0.01) & valid
    height = batch["image_hw"][:, 0].float()[None, :, None]
    width = batch["image_hw"][:, 1].float()[None, :, None]
    scale = torch.stack((width.expand_as(height), height), dim=-1)
    target_flow = batch["target"][None, ..., :2] * scale
    sample_flow = samples[..., :2] * scale
    flow_error = (sample_flow - target_flow).norm(dim=-1)
    xyz = torch.stack([prediction_xyz(batch, sample) for sample in samples])
    xyz_error = (xyz - batch["target_xyz"][None]).norm(dim=-1) * 100.0
    weight = mover.float()[None]
    mover_count = weight.sum(dim=-1)
    keep = mover_count[0] > 0
    flow_per_clip = (flow_error * weight).sum(dim=-1) / mover_count.clamp_min(1.0)
    xyz_per_clip = (xyz_error * weight).sum(dim=-1) / mover_count.clamp_min(1.0)
    zero_per_clip = target_flow.norm(dim=-1)
    zero_per_clip = (zero_per_clip * weight).sum(dim=-1)[0] / mover_count[0].clamp_min(1.0)
    baseline_flow = baseline_prediction[..., :2] * scale[0]
    baseline_flow_error = (baseline_flow - target_flow[0]).norm(dim=-1)
    baseline_per_clip = (
        baseline_flow_error * mover.float()
    ).sum(dim=-1) / mover.float().sum(dim=-1).clamp_min(1.0)
    baseline_values = baseline_per_clip[keep]
    baseline_key = f"{prefix}/baseline/flow_epe_mover_px"
    totals[baseline_key] += float(baseline_values.sum())
    squares[baseline_key] += float(baseline_values.square().sum())
    counts[baseline_key] += len(baseline_values)
    baseline_point = baseline_flow_error[mover]
    baseline_point_key = f"{prefix}/baseline/flow_epe_mover_point_weighted_px"
    totals[baseline_point_key] += float(baseline_point.sum())
    squares[baseline_point_key] += float(baseline_point.square().sum())
    counts[baseline_point_key] += len(baseline_point)
    for count in sample_counts:
        best_flow_values, best_flow_index = flow_per_clip[:count].min(dim=0)
        best_xyz_values, best_xyz_index = xyz_per_clip[:count].min(dim=0)
        best_flow = best_flow_values[keep]
        best_xyz = best_xyz_values[keep]
        delta = best_flow - zero_per_clip[keep]
        baseline_delta = best_flow - baseline_per_clip[keep]
        for metric, value in (
            ("flow_epe_mover_px", best_flow),
            ("xyz_epe_mover_cm", best_xyz),
            ("flow_delta_vs_zero_px", delta),
            ("flow_delta_vs_baseline_px", baseline_delta),
        ):
            key = f"{prefix}/best_of_{count:02d}/{metric}"
            totals[key] += float(value.sum())
            squares[key] += float(value.square().sum())
            counts[key] += len(value)
        row = torch.arange(samples.shape[1], device=samples.device)
        selected_flow_error = flow_error[best_flow_index, row][mover]
        selected_xyz_error = xyz_error[best_xyz_index, row][mover]
        for metric, value in (
            ("flow_epe_mover_point_weighted_px", selected_flow_error),
            ("xyz_epe_mover_point_weighted_cm", selected_xyz_error),
        ):
            key = f"{prefix}/best_of_{count:02d}/{metric}"
            totals[key] += float(value.sum())
            squares[key] += float(value.square().sum())
            counts[key] += len(value)


@torch.no_grad()
def evaluate(
    model: ParticleWorldModel,
    cache: dict,
    split: str,
    sample_counts: tuple[int, ...],
    device: torch.device,
    baseline: ParticleWorldModel,
) -> dict:
    dataset = ParticleProbeDataset(cache, split)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    totals: dict[str, float] = defaultdict(float)
    squares: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    maximum = max(sample_counts)
    torch.manual_seed(17)
    for cpu_batch in loader:
        batch = to_device(cpu_batch, device)
        samples, _ = model.predict_prior(batch, samples=maximum, sample=True)
        baseline_prediction, _ = baseline.predict_prior(batch, samples=1, sample=False)
        update(
            totals,
            squares,
            counts,
            batch,
            samples,
            baseline_prediction[0],
            sample_counts,
            "all",
        )
        for horizon in batch["horizon"].unique().tolist():
            keep = batch["horizon"] == horizon
            sliced = {key: value[keep] for key, value in batch.items()}
            update(
                totals,
                squares,
                counts,
                sliced,
                samples[:, keep],
                baseline_prediction[0, keep],
                sample_counts,
                f"horizon_{int(horizon):02d}",
            )
    result = {}
    for key in sorted(totals):
        count = max(counts[key], 1)
        mean = totals[key] / count
        variance = max(squares[key] / count - mean * mean, 0.0)
        result[key] = mean
        result[f"{key}__se"] = (variance / count) ** 0.5
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample_counts", default="1,2,4,8,16,32")
    args = parser.parse_args()
    sample_counts = tuple(int(value) for value in args.sample_counts.split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ParticleWorldModel(WorldModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    baseline_checkpoint = torch.load(
        args.baseline_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    baseline = ParticleWorldModel(WorldModelConfig(**baseline_checkpoint["config"])).to(device)
    baseline.load_state_dict(baseline_checkpoint["model"])
    baseline.eval()
    result = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "baseline_checkpoint": os.path.abspath(args.baseline_checkpoint),
        "sample_counts": sample_counts,
        "splits": {
            split: evaluate(model, cache, split, sample_counts, device, baseline)
            for split in ("heldseed", "heldtask")
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
