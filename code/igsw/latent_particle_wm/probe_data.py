"""Build a compact, strict-causal particle-motion probe from RoboTwin clips.

The current particle state always comes from rt2_causal_v1: a fixed 48x48
single-frame VGGT grid. Full-video SpaTracker trajectories and future RGB are
targets only. The cache keeps all proposal particles used by the probe; a
current-image-only GPSToken mask marks the smaller renderer subset.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from igsw.gaussians.gpstoken import gpstoken_init, grad_mag


@dataclass(frozen=True)
class ProbeRoots:
    plan: str
    causal: str
    source: str
    tracked: str


def fixed_grid_indices(grid: int, particle_side: int) -> torch.Tensor:
    """Uniform persistent identities sampled from the full fixed proposal grid."""
    axis = torch.linspace(0, grid - 1, particle_side).round().long().unique()
    if len(axis) != particle_side:
        raise ValueError(f"particle_side={particle_side} does not map uniquely to grid={grid}")
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return (yy * grid + xx).reshape(-1)


def _stratified(entries: list[dict], limit: int, seed: int) -> list[dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_task[entry["task"]].append(entry)
    rng = random.Random(seed)
    for task_entries in by_task.values():
        rng.shuffle(task_entries)
    tasks = sorted(by_task)
    selected: list[dict] = []
    cursor = 0
    while len(selected) < limit and tasks:
        task = tasks[cursor % len(tasks)]
        if by_task[task]:
            selected.append(by_task[task].pop())
        if not by_task[task]:
            tasks.remove(task)
            cursor = 0
        else:
            cursor += 1
    return selected


def select_plan_entries(
    roots: ProbeRoots,
    limits: dict[str, int],
    seed: int,
) -> list[dict]:
    with open(roots.plan) as handle:
        plan = json.load(handle)
    available = []
    for entry in plan:
        filename = f"{entry['name']}_{entry['split']}.pt"
        if os.path.exists(os.path.join(roots.causal, filename)) and os.path.exists(
            os.path.join(roots.tracked, filename)
        ):
            copied = dict(entry)
            copied["filename"] = filename
            available.append(copied)
    selected = []
    for offset, split in enumerate(("train", "heldseed", "heldtask")):
        split_entries = [entry for entry in available if entry["split"] == split]
        selected.extend(_stratified(split_entries, min(limits[split], len(split_entries)), seed + offset))
    return selected


def _resize_rgb(rgb: torch.Tensor, height: int, width: int) -> torch.Tensor:
    image = rgb.permute(2, 0, 1).float()[None]
    return F.interpolate(image, size=(height, width), mode="bilinear", align_corners=False)[0] / 255.0


def _sample_map(image: torch.Tensor, uv: torch.Tensor, height: int, width: int) -> torch.Tensor:
    u = (uv[:, 0] + 0.5) / width * 2.0 - 1.0
    v = (uv[:, 1] + 0.5) / height * 2.0 - 1.0
    grid = torch.stack((u, v), dim=-1)[None, None]
    sampled = F.grid_sample(image[None], grid, mode="bilinear", align_corners=False)
    return sampled[0, :, 0].T.contiguous()


def _camera_xyz(xyz: torch.Tensor, viewmat: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(*xyz.shape[:-1], 1, dtype=xyz.dtype)
    return (torch.cat((xyz, ones), dim=-1) @ viewmat.T)[..., :3]


def _project(xyz_camera: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    normalized = xyz_camera / xyz_camera[..., 2:3].clamp_min(1e-5)
    return (normalized @ intrinsics.T)[..., :2]


def _gradient_map(rgb: torch.Tensor) -> torch.Tensor:
    gray = (rgb * rgb.new_tensor([0.299, 0.587, 0.114])[:, None, None]).sum(0)
    gx = gray[:, 2:] - gray[:, :-2]
    gy = gray[2:, :] - gray[:-2, :]
    gx = F.pad(gx, (1, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 1, 1))
    return torch.sqrt(gx.square() + gy.square()).clamp(0, 1)[None]


def _depth_derivatives(depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = F.pad((depth[:, 2:] - depth[:, :-2]) * 0.5, (1, 1, 0, 0))
    dy = F.pad((depth[2:, :] - depth[:-2, :]) * 0.5, (0, 0, 1, 1))
    return dx, dy


def _active_mask(rgb: torch.Tensor, uv: torch.Tensor, count: int) -> torch.Tensor:
    image = (rgb.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    _, gaussians = gpstoken_init(grad_mag(gray), count)
    centers = torch.tensor([[item[0], item[1]] for item in gaussians], dtype=torch.float32)
    nearest = torch.cdist(centers, uv).argmin(dim=1).unique()
    mask = torch.zeros(len(uv), dtype=torch.bool)
    mask[nearest] = True
    return mask


def _state_features(
    causal: dict,
    selected: torch.Tensor,
    particle_side: int,
    active_count: int,
) -> dict[str, torch.Tensor]:
    height, width = int(causal["H"]), int(causal["W"])
    viewmat = causal["viewmat"].float()
    intrinsics = causal["K_intr"].float()
    means = causal["means"].float()
    xyz_camera_full = _camera_xyz(means, viewmat)
    uv = causal["uv"].float()[selected]
    xyz = xyz_camera_full[selected]
    z = xyz[:, 2].clamp_min(1e-5)
    rgb = _resize_rgb(causal["gt_rgb"][0], height, width)
    color = _sample_map(rgb, uv, height, width)
    gradient = _sample_map(_gradient_map(rgb), uv, height, width)

    grid = round(len(means) ** 0.5)
    if grid * grid != len(means):
        raise ValueError(f"expected square proposal grid, got {len(means)}")
    depth = xyz_camera_full[:, 2].reshape(grid, grid)
    dx, dy = _depth_derivatives(depth)
    median_z = z.median().clamp_min(1e-5)
    center = xyz.median(dim=0).values
    radius = (xyz - center).norm(dim=-1).quantile(0.95).clamp_min(1e-4)
    sigma = torch.full((len(selected), 2), 1.0 / particle_side)
    state = torch.cat(
        (
            torch.stack((uv[:, 0] / width * 2 - 1, uv[:, 1] / height * 2 - 1), dim=-1),
            (z.log() - median_z.log())[:, None],
            (xyz - center) / radius,
            color * 2 - 1,
            gradient,
            torch.stack((dx.reshape(-1)[selected], dy.reshape(-1)[selected]), dim=-1) / median_z,
            sigma,
        ),
        dim=-1,
    )
    return {
        "state": state,
        "active": _active_mask(rgb, uv, active_count),
        "uv0": uv,
        "z0": z,
        "intrinsics": intrinsics,
        "viewmat": viewmat,
        "image_hw": torch.tensor([height, width], dtype=torch.long),
        "color0": color,
        "center": center,
        "radius": radius,
    }


def _target_features(
    causal: dict,
    source: dict,
    base: dict[str, torch.Tensor],
    selected: torch.Tensor,
    horizon: int,
) -> dict[str, torch.Tensor]:
    height, width = base["image_hw"].tolist()
    target_world = causal["traj"][horizon].float()[selected]
    target_xyz = _camera_xyz(target_world, base["viewmat"])
    target_uv = _project(target_xyz, base["intrinsics"])
    target_z = target_xyz[:, 2].clamp_min(1e-5)
    future_rgb = _resize_rgb(source["gt_rgb"][horizon], height, width)
    future_color = _sample_map(future_rgb, target_uv, height, width)
    motion = torch.stack(
        (
            (target_uv[:, 0] - base["uv0"][:, 0]) / width,
            (target_uv[:, 1] - base["uv0"][:, 1]) / height,
            (target_z / base["z0"]).log(),
        ),
        dim=-1,
    )
    target = torch.cat((motion, future_color - base["color0"]), dim=-1)
    valid = causal["geom_valid"].bool()[selected]
    visible = causal["vis"][horizon].bool()[selected] & valid
    path_world = causal["traj"][: horizon + 1].float()[:, selected]
    path_xyz = _camera_xyz(path_world, base["viewmat"])
    path_uv = _project(path_xyz, base["intrinsics"])
    step_scale = target.new_tensor([width, height])
    max_image_step = ((path_uv[1:] - path_uv[:-1]) / step_scale).norm(dim=-1).amax(dim=0)
    path_z = path_xyz[..., 2].clamp_min(1e-5)
    max_depth_step = (path_z[1:] / path_z[:-1]).log().abs().amax(dim=0)
    path_finite = torch.isfinite(path_uv).all(dim=(0, 2)) & torch.isfinite(path_z).all(dim=0)
    plausible = path_finite & (max_image_step < 0.12) & (max_depth_step < 0.35)
    plausible &= torch.isfinite(target).all(dim=-1) & (target_z > 1e-5)
    motion_valid = valid & visible & plausible
    return {
        "target": target,
        "valid": valid,
        "visible": visible,
        "motion_valid": motion_valid,
        "target_xyz": target_xyz,
    }


def build_probe_cache(
    roots: ProbeRoots,
    output: str,
    limits: dict[str, int],
    horizons: tuple[int, ...] = (1, 3, 6, 9, 12),
    grid: int = 48,
    particle_side: int = 16,
    active_count: int = 96,
    seed: int = 17,
) -> dict:
    entries = select_plan_entries(roots, limits, seed)
    selected = fixed_grid_indices(grid, particle_side)
    clips: dict[str, list] = defaultdict(list)
    records: dict[str, list] = defaultdict(list)
    split_to_id = {"train": 0, "heldseed": 1, "heldtask": 2}

    for clip_index, entry in enumerate(entries):
        filename = entry["filename"]
        causal = torch.load(os.path.join(roots.causal, filename), map_location="cpu", weights_only=False)
        source = torch.load(os.path.join(roots.source, filename), map_location="cpu", weights_only=False)
        if causal.get("causal_geometry_version") != "vggt_t1_grid48_v1":
            raise ValueError(f"unexpected causal geometry in {filename}")
        base = _state_features(causal, selected, particle_side, active_count)
        for key, value in base.items():
            if key not in {"color0", "center", "radius", "viewmat"}:
                clips[key].append(value)
        clips["split"].append(split_to_id[entry["split"]])
        clips["task"].append(entry["task"])
        clips["name"].append(filename)
        clips["radius"].append(base["radius"])
        for horizon in horizons:
            target = _target_features(causal, source, base, selected, horizon)
            records["clip_index"].append(clip_index)
            records["horizon"].append(horizon)
            for key, value in target.items():
                records[key].append(value)
        if (clip_index + 1) % 100 == 0 or clip_index + 1 == len(entries):
            print(f"[probe-cache] {clip_index + 1}/{len(entries)}", flush=True)

    cache = {
        "version": "strict_causal_particle_probe_v2",
        "roots": roots.__dict__,
        "config": {
            "limits": limits,
            "horizons": list(horizons),
            "grid": grid,
            "particle_side": particle_side,
            "active_count": active_count,
            "seed": seed,
        },
        "clips": {
            key: torch.stack(value) if value and torch.is_tensor(value[0]) else value
            for key, value in clips.items()
        },
        "records": {
            key: torch.stack(value)
            if value and torch.is_tensor(value[0])
            else torch.tensor(value, dtype=torch.long)
            for key, value in records.items()
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.save(cache, output)
    return cache


class ParticleProbeDataset(Dataset):
    def __init__(self, cache: dict, split: str):
        split_id = {"train": 0, "heldseed": 1, "heldtask": 2}[split]
        clip_indices = cache["records"]["clip_index"]
        clip_split = cache["clips"]["split"]
        keep = torch.tensor([clip_split[int(index)] == split_id for index in clip_indices])
        self.record_indices = keep.nonzero(as_tuple=False).flatten()
        self.cache = cache

    def __len__(self) -> int:
        return len(self.record_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index = int(self.record_indices[index])
        clip_index = int(self.cache["records"]["clip_index"][record_index])
        clips = self.cache["clips"]
        records = self.cache["records"]
        return {
            "state": clips["state"][clip_index],
            "active": clips["active"][clip_index],
            "uv0": clips["uv0"][clip_index],
            "z0": clips["z0"][clip_index],
            "intrinsics": clips["intrinsics"][clip_index],
            "image_hw": clips["image_hw"][clip_index],
            "radius": clips["radius"][clip_index],
            "target": records["target"][record_index],
            "valid": records["valid"][record_index],
            "visible": records["visible"][record_index],
            "motion_valid": records["motion_valid"][record_index],
            "target_xyz": records["target_xyz"][record_index],
            "horizon": records["horizon"][record_index],
            "clip_index": torch.tensor(clip_index),
        }
