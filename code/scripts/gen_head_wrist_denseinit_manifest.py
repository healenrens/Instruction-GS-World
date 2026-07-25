"""Generate a head+wrist D3DGS manifest with multi-frame backwarped init cloud.

The official Dynamic3DGaussians trainer is unchanged. This only improves
`init_pt_cld.npz`: RGB-D points from selected video frames are backprojected to
world; points on moving entities are mapped back to their timestep-0 pose using
simulator entity poses, while static points stay in world coordinates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.convert_head_wrist_to_d3dgs import convert  # noqa: E402
from scripts.gen_head_wrist_d3dgs_manifest import (  # noqa: E402
    _generate_episode,
    _moving_ids,
    _save_png,
    _w2c,
)
from scripts.maniskill_gt import (  # noqa: E402
    EXCLUDE_NAMES,
    backproject_to_world,
    extrinsic_to_viewmat,
    pose_to_matrix,
)


def _entity_name_by_sid(rec) -> dict[int, str]:
    return {
        int(sid): getattr(ent, "name", str(sid))
        for sid, ent in rec.env.unwrapped.segmentation_id_map.items()
    }


def _dense_init_cloud(rec, moving: set[int], frame_stride: int, pixel_stride: int,
                      depth_max: float) -> np.ndarray:
    names = _entity_name_by_sid(rec)
    exclude = set(EXCLUDE_NAMES)
    frame_ids = list(range(0, len(rec.frames), int(frame_stride)))
    if 0 not in frame_ids:
        frame_ids.insert(0, 0)
    rows = []
    for t in frame_ids:
        frame = rec.frames[t]
        poses_t = rec.entity_poses[t]
        poses_0 = rec.entity_poses[0]
        for uid in ("head", "wrist"):
            cam = frame["cams"][uid]
            K = cam["K"].float()
            view = extrinsic_to_viewmat(cam["extrinsic_cv"].float())
            depth = cam["depth"].float() / 1000.0
            seg = cam["seg"].long()
            rgb = cam["rgb"].float() / 255.0
            pts, valid_depth = backproject_to_world(depth, K, view)
            keep = valid_depth & (depth < float(depth_max)) & (seg > 0)
            if int(pixel_stride) > 1:
                yy, xx = torch.meshgrid(torch.arange(seg.shape[0]), torch.arange(seg.shape[1]), indexing="ij")
                keep &= (yy % int(pixel_stride) == 0) & (xx % int(pixel_stride) == 0)
            for sid, name in names.items():
                if name in exclude:
                    keep &= seg != int(sid)
            xyz = pts[keep]
            color = rgb[keep]
            sid_vals = seg[keep]
            fg = torch.isin(sid_vals, torch.tensor(sorted(moving), dtype=seg.dtype)).float()[:, None]
            if moving:
                xyz = _backwarp_moving_points(xyz, sid_vals, moving, poses_t, poses_0)
            rows.append(torch.cat([xyz, color, fg], dim=1))
    data = torch.cat(rows, dim=0).cpu().numpy().astype(np.float32)
    if data.shape[0] == 0:
        raise RuntimeError("empty dense init cloud")
    return data


def _backwarp_moving_points(xyz: torch.Tensor, sid_vals: torch.Tensor, moving: set[int],
                            poses_t: dict[int, torch.Tensor], poses_0: dict[int, torch.Tensor]) -> torch.Tensor:
    out = xyz.clone()
    for sid in sorted(moving):
        mask = sid_vals == int(sid)
        if not bool(mask.any()):
            continue
        if sid not in poses_t or sid not in poses_0:
            raise KeyError(f"missing pose for moving segmentation id {sid}")
        T_t = pose_to_matrix(poses_t[sid].float())
        T_0 = pose_to_matrix(poses_0[sid].float())
        R = (T_0 @ torch.linalg.inv(T_t))[:3, :3]
        tr = (T_0 @ torch.linalg.inv(T_t))[:3, 3]
        out[mask] = out[mask] @ R.T + tr
    return out


def write_dense_manifest(rec, instruction: str, out_dir: Path, env_id: str, seed: int,
                         frame_stride: int, max_frames: int, depth_max: float,
                         move_threshold_m: float, init_frame_stride: int,
                         init_pixel_stride: int) -> Path:
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    moving = _moving_ids(rec, move_threshold_m)
    init = _dense_init_cloud(rec, moving, init_frame_stride, init_pixel_stride, depth_max)
    init_path = raw_dir / "init_pt_cld.npz"
    np.savez(init_path, data=init)

    indices = list(range(0, len(rec.frames), int(frame_stride)))
    if max_frames and max_frames > 0:
        indices = indices[:int(max_frames)]
    if len(indices) < 2:
        raise ValueError("need at least two frames for dynamic reconstruction")

    frames = []
    for out_t, idx in enumerate(indices):
        frame = rec.frames[idx]
        entry = {}
        for uid, cam_id in (("head", 0), ("wrist", 1)):
            cam = frame["cams"][uid]
            rgb = cam["rgb"].numpy().astype(np.uint8)
            seg = cam["seg"].numpy()
            mask = np.isin(seg, np.array(sorted(moving), dtype=seg.dtype)).astype(np.uint8) * 255
            entry[uid] = {
                "rgb": _save_png(raw_dir / f"{uid}_{out_t:06d}.png", rgb),
                "seg": _save_png(raw_dir / f"{uid}_{out_t:06d}_seg.png", mask),
                "K": cam["K"].float().cpu().numpy().astype(float).tolist(),
                "w2c": _w2c(cam["extrinsic_cv"]),
                "cam_id": cam_id,
            }
        frames.append(entry)

    manifest = {
        "init_pt_cld": str(init_path),
        "instruction": instruction,
        "env": env_id,
        "seed": int(seed),
        "moving_seg_ids": sorted(int(x) for x in moving),
        "init_frame_stride": int(init_frame_stride),
        "init_pixel_stride": int(init_pixel_stride),
        "frames": frames,
    }
    manifest_path = out_dir / "episode_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PickCube-v1", choices=["PickCube-v1", "PushCube-v1", "StackCube-v1"])
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--cam", type=int, default=256)
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--depth_max", type=float, default=2.0)
    ap.add_argument("--move_threshold_m", type=float, default=0.01)
    ap.add_argument("--max_retries", type=int, default=8)
    ap.add_argument("--init_frame_stride", type=int, default=16)
    ap.add_argument("--init_pixel_stride", type=int, default=4)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--convert_out_root", default="")
    ap.add_argument("--sequence", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.init_frame_stride <= 0 or args.init_pixel_stride <= 0:
        raise ValueError("init strides must be positive")

    rec, instruction = _generate_episode(args.env, args.seed, args.cam, args.max_retries)
    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest_path = write_dense_manifest(
        rec, instruction, out_dir, args.env, args.seed, args.frame_stride, args.max_frames,
        args.depth_max, args.move_threshold_m, args.init_frame_stride, args.init_pixel_stride,
    )
    result = {"manifest": str(manifest_path), "frames_recorded": len(rec.frames)}
    if args.convert_out_root and args.sequence:
        result["converted"] = convert(
            manifest_path, Path(args.convert_out_root).expanduser().resolve(),
            args.sequence, args.overwrite, jpeg_quality=95,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
