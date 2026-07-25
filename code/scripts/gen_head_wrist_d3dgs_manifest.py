"""Generate a ManiSkill head+wrist episode manifest for official D3DGS.

This is a reference-format fixture generator: fixed head camera, wrist camera
mounted on the robot TCP, all selected video frames, calibrated K/w2c, foreground
masks, and an init point cloud from real rendered depth. Dynamic reconstruction
is still done by the official Dynamic3DGaussians trainer.
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
from scripts.maniskill_gt import (  # noqa: E402
    EXCLUDE_NAMES,
    INSTRUCTIONS,
    _ObsRecorder,
    _manip_object,
    _script_pick,
    _script_push,
    backproject_to_world,
    extrinsic_to_viewmat,
)


def _make_env_class(env_id: str):
    import sapien
    from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
    from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
    from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils

    base_cls = {
        "PickCube-v1": PickCubeEnv,
        "PushCube-v1": PushCubeEnv,
        "StackCube-v1": StackCubeEnv,
    }[env_id]

    class HeadWristEnv(base_cls):
        cam_size = 256

        @property
        def _default_sensor_configs(self):
            target = [0.0, 0.0, 0.05]
            head_pose = sapien_utils.look_at(eye=[0.38, 0.0, 0.62], target=target)
            wrist_pose = sapien_utils.look_at(eye=[-0.08, 0.0, 0.04], target=[0.08, 0.0, 0.0])
            return [
                CameraConfig("head", head_pose, self.cam_size, self.cam_size, np.pi / 2, 0.01, 100),
                CameraConfig("wrist", wrist_pose, self.cam_size, self.cam_size, np.pi / 2, 0.01, 100,
                             mount=self.agent.tcp),
            ]

    return HeadWristEnv


class HeadWristRecorder(_ObsRecorder):
    def _record(self, obs):
        rec = {"cams": {}}
        for uid in ("head", "wrist"):
            cam = obs["sensor_data"][uid]
            prm = obs["sensor_param"][uid]
            rec["cams"][uid] = {
                "rgb": cam["rgb"][0].detach().cpu().clone(),
                "depth": cam["depth"][0, ..., 0].detach().cpu().clone(),
                "seg": cam["segmentation"][0, ..., 0].detach().cpu().clone(),
                "K": prm["intrinsic_cv"][0].detach().cpu().clone(),
                "extrinsic_cv": prm["extrinsic_cv"][0].detach().cpu().clone(),
            }
        self.frames.append(rec)
        self.entity_poses.append(self._entity_pose_snapshot())


def _generate_episode(env_id: str, seed: int, cam: int, max_retries: int):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    cls = _make_env_class(env_id)
    cls.cam_size = int(cam)
    reg_id = f"HeadWrist{env_id}"
    if reg_id not in gym.envs.registry:
        gym.register(reg_id, entry_point=cls)
    base = gym.make(
        reg_id, obs_mode="rgb+depth+segmentation", control_mode="pd_ee_delta_pose",
        render_mode="rgb_array", sim_backend="cpu", num_envs=1,
    )
    policy = _script_push if env_id == "PushCube-v1" else _script_pick
    rec = HeadWristRecorder(base)
    s = int(seed)
    for _ in range(int(max_retries)):
        rec.reset(seed=s)
        obj = _manip_object(base.unwrapped)
        start = obj.pose.p[0].cpu().numpy().copy()
        policy(base, rec)
        moved = float(np.linalg.norm(obj.pose.p[0].cpu().numpy() - start))
        ok = moved > 0.04 and len(rec.frames) > 4
        print(f"[head-wrist] env={env_id} seed={s} moved={moved:.3f} frames={len(rec.frames)} "
              f"{'OK' if ok else 'retry'}", flush=True)
        if ok:
            return rec, INSTRUCTIONS[env_id]
        s += 1
    raise RuntimeError(f"failed to generate moving head/wrist episode for {env_id} seed={seed}")


def _moving_ids(rec: HeadWristRecorder, threshold_m: float) -> set[int]:
    first, last = rec.entity_poses[0], rec.entity_poses[-1]
    moving = set()
    for sid, p0 in first.items():
        p1 = last.get(sid)
        if p1 is None:
            continue
        if float(torch.linalg.norm(p1[:3] - p0[:3])) >= float(threshold_m):
            moving.add(int(sid))
    return moving


def _save_png(path: Path, arr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return str(path)


def _init_cloud_from_frame(rec: HeadWristRecorder, moving: set[int], depth_max: float) -> np.ndarray:
    f0 = rec.frames[0]
    rows = []
    exclude = set(EXCLUDE_NAMES)
    name_by_sid = {
        int(sid): getattr(ent, "name", str(sid))
        for sid, ent in rec.env.unwrapped.segmentation_id_map.items()
    }
    for uid in ("head", "wrist"):
        cam = f0["cams"][uid]
        K = cam["K"].float()
        view = extrinsic_to_viewmat(cam["extrinsic_cv"].float())
        depth = cam["depth"].float() / 1000.0
        seg = cam["seg"].long()
        rgb = cam["rgb"].float() / 255.0
        pts, valid_depth = backproject_to_world(depth, K, view)
        keep = valid_depth & (depth < float(depth_max)) & (seg > 0)
        for sid, name in name_by_sid.items():
            if name in exclude:
                keep &= seg != int(sid)
        xyz = pts[keep]
        color = rgb[keep]
        fg = torch.isin(seg[keep], torch.tensor(sorted(moving), dtype=seg.dtype)).float()[:, None]
        rows.append(torch.cat([xyz, color, fg], dim=1))
    data = torch.cat(rows, dim=0).cpu().numpy().astype(np.float32)
    if data.shape[0] == 0:
        raise RuntimeError("empty init cloud")
    return data


def _w2c(extrinsic_cv: torch.Tensor) -> list[list[float]]:
    mat = extrinsic_to_viewmat(extrinsic_cv.float()).cpu().numpy()
    return mat.astype(float).tolist()


def write_manifest(rec: HeadWristRecorder, instruction: str, out_dir: Path, env_id: str, seed: int,
                   frame_stride: int, max_frames: int, depth_max: float, move_threshold_m: float) -> Path:
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    moving = _moving_ids(rec, move_threshold_m)
    init = _init_cloud_from_frame(rec, moving, depth_max)
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
            rgb_path = _save_png(raw_dir / f"{uid}_{out_t:06d}.png", rgb)
            seg_path = _save_png(raw_dir / f"{uid}_{out_t:06d}_seg.png", mask)
            entry[uid] = {
                "rgb": rgb_path,
                "seg": seg_path,
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
    ap.add_argument("--max_frames", type=int, default=0, help="0 means all selected episode frames")
    ap.add_argument("--depth_max", type=float, default=2.0)
    ap.add_argument("--move_threshold_m", type=float, default=0.01)
    ap.add_argument("--max_retries", type=int, default=8)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--convert_out_root", default="")
    ap.add_argument("--sequence", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rec, instruction = _generate_episode(args.env, args.seed, args.cam, args.max_retries)
    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest_path = write_manifest(
        rec, instruction, out_dir, args.env, args.seed, args.frame_stride, args.max_frames,
        args.depth_max, args.move_threshold_m,
    )
    result = {"manifest": str(manifest_path), "frames_recorded": len(rec.frames)}
    if args.convert_out_root and args.sequence:
        summary = convert(manifest_path, Path(args.convert_out_root).expanduser().resolve(),
                          args.sequence, args.overwrite, jpeg_quality=95)
        result["converted"] = summary
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
