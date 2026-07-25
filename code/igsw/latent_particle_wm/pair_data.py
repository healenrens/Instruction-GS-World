"""Training dataset for strict-causal arbitrary-frame particle pairs."""
from __future__ import annotations

import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from igsw.causal_geometry import project_xyz
from igsw.gaussians.gpstoken import gpstoken_init, grad_mag

from .pair_targets import CAUSAL_PAIR_VERSION

PAIR_STATE_DIM = 16


def control_grid_indices(grid: int, rows: int, cols: int) -> torch.Tensor:
    if rows > grid or cols > grid:
        raise ValueError("control grid cannot exceed the dense proposal grid")
    ys = torch.linspace(0, grid - 1, rows).round().long()
    xs = torch.linspace(0, grid - 1, cols).round().long()
    if len(ys.unique()) != rows or len(xs.unique()) != cols:
        raise ValueError("control grid indices are not unique")
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return (yy * grid + xx).flatten()


def sample_image(rgb_chw: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    height, width = rgb_chw.shape[-2:]
    grid = torch.stack(
        (
            (uv[:, 0] + 0.5) / width * 2.0 - 1.0,
            (uv[:, 1] + 0.5) / height * 2.0 - 1.0,
        ),
        dim=-1,
    )[None, None]
    sampled = F.grid_sample(
        rgb_chw[None],
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled[0, :, 0].T.contiguous()


def gaussian_base_scales(
    means: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    grid: int,
    footprint: float = 0.72,
) -> torch.Tensor:
    margin = width // 64
    spacing_x = (width - 2 * margin) / (grid - 1)
    spacing_y = (height - 2 * margin) / (grid - 1)
    depth = means[:, 2].clamp_min(1e-4)
    scale_x = depth / intrinsics[0, 0] * spacing_x * footprint
    scale_y = depth / intrinsics[1, 1] * spacing_y * footprint
    scale_z = torch.minimum(scale_x, scale_y) * 0.1
    return torch.stack((scale_x, scale_y, scale_z), dim=-1)


def current_active_mask(rgb_hwc: torch.Tensor, uv: torch.Tensor, count: int) -> torch.Tensor:
    image = rgb_hwc.numpy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    _, gaussians = gpstoken_init(grad_mag(gray), count)
    centers = torch.tensor(
        [[item[0], item[1]] for item in gaussians],
        dtype=torch.float32,
    )
    nearest = torch.cdist(centers, uv.float()).argmin(dim=1).unique()
    active = torch.zeros(len(uv), dtype=torch.bool)
    active[nearest] = True
    return active


def _gradient_map(rgb_chw: torch.Tensor) -> torch.Tensor:
    gray = (rgb_chw * rgb_chw.new_tensor([0.299, 0.587, 0.114])[:, None, None]).sum(0)
    dx = F.pad(gray[:, 2:] - gray[:, :-2], (1, 1, 0, 0))
    dy = F.pad(gray[2:] - gray[:-2], (0, 0, 1, 1))
    return torch.sqrt(dx.square() + dy.square()).clamp(0.0, 1.0)[None]


def _depth_derivatives(depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = F.pad((depth[:, 2:] - depth[:, :-2]) * 0.5, (1, 1, 0, 0))
    dy = F.pad((depth[2:] - depth[:-2, :]) * 0.5, (0, 0, 1, 1))
    return dx, dy


def pair_features(
    pair: dict,
    control_indices: torch.Tensor,
    active_count: int,
) -> dict[str, torch.Tensor]:
    means = pair["means"].float()
    uv = pair["uv"].float()
    intrinsics = pair["K_intr"].float()
    height, width = int(pair["H"]), int(pair["W"])
    grid = round(len(means) ** 0.5)
    if grid * grid != len(means):
        raise ValueError("pair proposals must form a square grid")
    rgb0_hwc = pair["rgb_path"][0]
    rgb1_hwc = pair["rgb_path"][-1]
    rgb0 = rgb0_hwc.permute(2, 0, 1).float() / 255.0
    rgb1 = rgb1_hwc.permute(2, 0, 1).float() / 255.0
    dense_color = sample_image(rgb0, uv)
    dense_scales = gaussian_base_scales(means, intrinsics, height, width, grid)
    active = current_active_mask(rgb0_hwc, uv, active_count)

    depth = means[:, 2].reshape(grid, grid)
    depth_dx, depth_dy = _depth_derivatives(depth)
    gradient = sample_image(_gradient_map(rgb0), uv)
    center = means.median(dim=0).values
    radius = (means - center).norm(dim=-1).quantile(0.95).clamp_min(1e-4)
    median_depth = means[:, 2].median().clamp_min(1e-4)
    median_scale = dense_scales[:, :2].median(dim=0).values.clamp_min(1e-6)
    state_dense = torch.cat(
        (
            torch.stack((uv[:, 0] / width * 2 - 1, uv[:, 1] / height * 2 - 1), dim=-1),
            (means[:, 2].log() - median_depth.log())[:, None],
            (means - center) / radius,
            dense_color * 2 - 1,
            gradient,
            torch.stack((depth_dx.flatten(), depth_dy.flatten()), dim=-1) / median_depth,
            (dense_scales[:, :2] / median_scale).log(),
            torch.full((len(means), 1), 0.99),
            active.float()[:, None],
        ),
        dim=-1,
    )
    if state_dense.shape[-1] != PAIR_STATE_DIM:
        raise ValueError(f"unexpected state dimension: {state_dense.shape[-1]}")

    target_xyz = pair["traj"][-1].float()
    target_uv = project_xyz(target_xyz, intrinsics)
    target_depth = target_xyz[:, 2].clamp_min(1e-5)
    motion = torch.stack(
        (
            (target_uv[:, 0] - uv[:, 0]) / width,
            (target_uv[:, 1] - uv[:, 1]) / height,
            (target_depth / means[:, 2].clamp_min(1e-5)).log(),
        ),
        dim=-1,
    )
    future_color = sample_image(rgb1, target_uv)
    target = torch.cat((motion, future_color - dense_color), dim=-1)

    path = pair["traj"].float()
    path_uv = project_xyz(path, intrinsics)
    step_scale = target.new_tensor([width, height])
    image_jump = ((path_uv[1:] - path_uv[:-1]) / step_scale).norm(dim=-1).amax(dim=0)
    path_depth = path[..., 2].clamp_min(1e-5)
    depth_jump = (path_depth[1:] / path_depth[:-1]).log().abs().amax(dim=0)
    finite_path = torch.isfinite(path).all(dim=(0, 2)) & torch.isfinite(path_uv).all(dim=(0, 2))
    plausible = finite_path & (image_jump < 0.12) & (depth_jump < 0.35)
    plausible &= torch.isfinite(target).all(dim=-1)
    matched = torch.isfinite(pair["geom_match_distance"])
    visible = pair["vis"][-1].bool()
    motion_valid = pair["geom_valid"].bool() & visible & plausible

    selected = control_indices
    return {
        "state": state_dense[selected],
        "target": target[selected],
        "matched": matched[selected],
        "visible": visible[selected],
        "motion_valid": motion_valid[selected],
        "target_xyz": target_xyz[selected],
        "control_means": means[selected],
        "control_uv": uv[selected],
        "tracker_image_jump": image_jump[selected],
        "tracker_depth_jump": depth_jump[selected],
        "dense_means": means,
        "dense_uv": uv,
        "dense_color": dense_color,
        "dense_scales": dense_scales,
        "dense_active": active,
        "intrinsics": intrinsics,
        "viewmat": pair["viewmat"].float(),
        "image_hw": torch.tensor([height, width], dtype=torch.long),
        "rgb0": rgb0,
        "rgb1": rgb1,
        "horizon": torch.tensor(int(pair["horizon"]), dtype=torch.long),
    }


class CausalPairDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        control_rows: int = 16,
        control_cols: int = 16,
        active_count: int = 192,
        dino_root: str | None = None,
    ):
        paths = sorted(glob.glob(os.path.join(root, "*.pt")))
        self.paths = [
            path
            for path in paths
            if f"_{split}_t" in os.path.basename(path)
        ]
        if not self.paths:
            raise ValueError(f"no {split} pairs in {root}")
        first = torch.load(self.paths[0], map_location="cpu", weights_only=False)
        if first.get("pair_version") != CAUSAL_PAIR_VERSION:
            raise ValueError("causal pair version mismatch")
        grid = round(first["means"].shape[0] ** 0.5)
        self.control_indices = control_grid_indices(grid, control_rows, control_cols)
        self.active_count = active_count
        self.dino_root = dino_root

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        pair = torch.load(path, map_location="cpu", weights_only=False)
        if pair.get("pair_version") != CAUSAL_PAIR_VERSION:
            raise ValueError(f"causal pair version mismatch: {path}")
        item = pair_features(pair, self.control_indices, self.active_count)
        if self.dino_root:
            dino_path = os.path.join(self.dino_root, os.path.basename(path))
            dino = torch.load(dino_path, map_location="cpu", weights_only=False)
            if (
                dino["source_name"] != pair["source_name"]
                or int(dino["start"]) != int(pair["start"])
                or int(dino["end"]) != int(pair["end"])
            ):
                raise ValueError(f"DINO sidecar mismatch: {dino_path}")
            item["dino0"] = dino["dino0"].float()
            item["dino1"] = dino["dino1"].float()
        item["source_name"] = pair["source_name"]
        item["task"] = pair["task"]
        item["pair_path"] = path
        return item
