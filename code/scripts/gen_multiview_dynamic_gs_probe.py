"""Probe fixed-identity DynamicGS data from a multi-view canonical scan.

This is a small Stage-B prototype for notes/data_dynamic3dgs_plan.md:
build g0 once from several calibrated RGB-D/seg cameras at the clip start, keep
Gaussian IDs fixed, and generate the whole trajectory from simulator entity poses.
Only PickCube is covered here because it is a data-contract probe, not the final
multi-task generator.
"""
from __future__ import annotations

import argparse
import os
import sys

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv  # noqa: E402
from mani_skill.sensors.camera import CameraConfig  # noqa: E402
from mani_skill.utils import sapien_utils  # noqa: E402

from igsw.gaussians.render import render_gaussianset, psnr  # noqa: E402
from igsw.lifting.to_gaussians import points_to_gaussians  # noqa: E402
from scripts.maniskill_gt import (  # noqa: E402
    INSTRUCTIONS,
    EXCLUDE_NAMES,
    _ObsRecorder,
    _manip_object,
    _script_pick,
    _moving_seg_ids,
    apply_traj_to_gaussians,
    backproject_to_world,
    extrinsic_to_viewmat,
    pose_to_matrix,
)


class PickCubeScanEnv(PickCubeEnv):
    @property
    def _default_sensor_configs(self):
        target = [0.0, 0.0, 0.05]
        eyes = [
            [0.30, 0.00, 0.60],
            [0.00, 0.36, 0.56],
            [0.00, -0.36, 0.56],
            [-0.34, 0.00, 0.58],
        ]
        cams = []
        for i, eye in enumerate(eyes):
            pose = sapien_utils.look_at(eye=eye, target=target)
            cams.append(CameraConfig(f"scan_{i}", pose, self.scan_w, self.scan_h, np.pi / 2, 0.01, 100))
        return cams


class MultiViewRecorder(_ObsRecorder):
    def _record(self, obs):
        rec = {"cams": {}}
        for uid, cam in obs["sensor_data"].items():
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


def generate_scan_episode(seed: int, cam: int):
    PickCubeScanEnv.scan_w = cam
    PickCubeScanEnv.scan_h = cam
    env_id = "PickCubeScanProbe-v1"
    if env_id not in gym.envs.registry:
        gym.register(env_id, entry_point=PickCubeScanEnv)
    base = gym.make(
        env_id, obs_mode="rgb+depth+segmentation", control_mode="pd_ee_delta_pose",
        render_mode="rgb_array", sim_backend="cpu", num_envs=1,
    )
    rec = MultiViewRecorder(base)
    rec.reset(seed=seed)
    obj = _manip_object(base.unwrapped)
    start = obj.pose.p[0].cpu().numpy().copy()
    _script_pick(base, rec)
    moved = float(np.linalg.norm(obj.pose.p[0].cpu().numpy() - start))
    print(f"[mv] seed={seed} moved={moved:.3f} frames={len(rec.frames)}", flush=True)
    return rec, INSTRUCTIONS["PickCube-v1"]


def filter_gaussians(gs, keep):
    features = None if gs.features is None else gs.features[keep]
    return gs.__class__(
        gs.means[keep], gs.quats[keep], gs.scales[keep],
        gs.opacities[keep], gs.colors[keep], features,
    )


def project_to_camera(points, K, viewmat, H=None, W=None):
    ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
    cam = torch.cat([points, ones], -1) @ viewmat.T
    z_raw = cam[:, 2]
    z = z_raw.clamp_min(1e-6)
    u = K[0, 0] * (cam[:, 0] / z) + K[0, 2]
    v = K[1, 1] * (cam[:, 1] / z) + K[1, 2]
    uv = torch.stack([u, v], -1)
    if H is None or W is None:
        return uv
    valid = (z_raw > 1e-6) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    return uv, valid


def build_multiview_clip(rec, instruction, K_frames, device, depth_max,
                         window_steps=None, policy_visible_only=False):
    n_sim = len(rec.frames)
    win = n_sim - 1 if window_steps is None else int(min(max(1, window_steps), n_sim - 1))
    idx = np.linspace(0, win, K_frames + 1).round().astype(int)
    frames = [rec.frames[i] for i in idx]
    poses = [rec.entity_poses[i] for i in idx]
    f0 = frames[0]
    u = rec.env.unwrapped
    name_by_sid = {int(sid): getattr(ent, "name", str(sid)) for sid, ent in u.segmentation_id_map.items()}
    exclude = {sid for sid, nm in name_by_sid.items() if nm in EXCLUDE_NAMES}

    pts, imgs, masks, segs = [], [], [], []
    cam_ids = sorted(f0["cams"])
    for cid in cam_ids:
        c = f0["cams"][cid]
        K = c["K"].to(device).float()
        viewmat = extrinsic_to_viewmat(c["extrinsic_cv"].to(device).float())
        depth = c["depth"].to(device).float() / 1000.0
        seg = c["seg"].to(device).long()
        rgb = (c["rgb"].to(device).float() / 255.0).permute(2, 0, 1)
        p, dmask = backproject_to_world(depth, K, viewmat)
        keep = dmask & (depth < depth_max) & (seg > 0)
        for sid in exclude:
            keep &= (seg != int(sid))
        pts.append(p)
        imgs.append(rgb)
        masks.append(keep)
        segs.append(seg)

    g0, _uv_native = points_to_gaussians(
        torch.stack(pts, 0), torch.stack(imgs, 0), torch.stack(masks, 0),
        opacity_init=0.9, scale_factor=0.6, scale_pct=(0.01, 0.7), return_uv=True,
    )
    seg_per_g = torch.stack(segs, 0)[torch.stack(masks, 0)]
    g0 = g0.to(device)

    base = f0["cams"]["scan_0"]
    K0 = base["K"].to(device).float()
    view0 = extrinsic_to_viewmat(base["extrinsic_cv"].to(device).float())
    H, W = int(base["depth"].shape[0]), int(base["depth"].shape[1])
    uv_policy, uv_valid = project_to_camera(g0.means, K0, view0, H, W)
    uv_total = int(uv_valid.numel())
    uv_visible = int(uv_valid.sum().item())
    if policy_visible_only:
        if uv_visible == 0:
            raise RuntimeError("policy-visible multiview filter removed every Gaussian")
        g0 = filter_gaussians(g0, uv_valid)
        seg_per_g = seg_per_g[uv_valid]
        uv_policy = uv_policy[uv_valid]
        uv_valid = uv_valid[uv_valid]

    X0 = g0.means
    traj = torch.empty(len(idx), X0.shape[0], 3, device=device)
    traj[0] = X0
    uniq = torch.unique(seg_per_g).tolist()
    for t in range(1, len(idx)):
        Xt = X0.clone()
        for sid in uniq:
            sid = int(sid)
            rp0 = poses[0].get(sid)
            rpt = poses[t].get(sid)
            if rp0 is None or rpt is None:
                continue
            m = seg_per_g == sid
            Trel = pose_to_matrix(rpt.to(device).float()) @ torch.linalg.inv(pose_to_matrix(rp0.to(device).float()))
            Xh = torch.cat([X0[m], torch.ones(int(m.sum()), 1, device=device)], -1)
            Xt[m] = (Xh @ Trel.T)[:, :3]
        traj[t] = Xt

    return {
        "g0": g0, "uv": uv_policy, "seg_per_g": seg_per_g, "traj": traj,
        "uv_valid": uv_valid,
        "K_intr": K0, "viewmat": view0, "H": H, "W": W, "Kf": len(idx) - 1,
        "frames": frames, "poses": poses, "name_by_sid": name_by_sid, "uniq_seg": uniq,
        "instruction": instruction, "scan_views": len(cam_ids),
        "uv_visible_count": uv_visible, "uv_total_before_filter": uv_total,
        "policy_visible_only": policy_visible_only,
    }


@torch.no_grad()
def validate_base_view(clip, device, out_dir=None):
    g0, traj = clip["g0"], clip["traj"]
    K, viewmat, H, W = clip["K_intr"], clip["viewmat"], clip["H"], clip["W"]
    base_rgb0 = clip["frames"][0]["cams"]["scan_0"]["rgb"].to(device).float() / 255.0
    full = []
    for t in range(clip["Kf"] + 1):
        gt = clip["frames"][t]["cams"]["scan_0"]["rgb"].to(device).float() / 255.0
        g = apply_traj_to_gaussians(g0, traj[t], clip["seg_per_g"], clip["poses"][t], clip["poses"][0], device)
        colors, alpha, _ = render_gaussianset(g, viewmat[None], K[None], W, H)
        pred = (colors[0] + (1 - alpha[0]) * base_rgb0).clamp(0, 1)
        full.append(psnr(pred, gt))
    return full


def save_clip(clip, out, instruction, val_psnr):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save = {
        "means": clip["g0"].means.cpu(), "quats": clip["g0"].quats.cpu(),
        "scales": clip["g0"].scales.cpu(), "opacities": clip["g0"].opacities.cpu(),
        "colors": clip["g0"].colors.cpu(), "uv": clip["uv"].cpu(),
        "uv_valid": clip["uv_valid"].cpu(),
        "seg_per_g": clip["seg_per_g"].cpu(), "traj": clip["traj"].cpu(),
        "K_intr": clip["K_intr"].cpu(), "viewmat": clip["viewmat"].cpu(),
        "H": clip["H"], "W": clip["W"], "Kf": clip["Kf"], "instruction": instruction,
        "env": "PickCubeScanProbe-v1", "seed": 0, "split": "train",
        "scan_views": clip["scan_views"], "val_psnr": val_psnr,
        "uv_visible_count": clip["uv_visible_count"],
        "uv_total_before_filter": clip["uv_total_before_filter"],
        "policy_visible_only": clip["policy_visible_only"],
        "poses": [{int(k): v.cpu() for k, v in p.items()} for p in clip["poses"]],
        "name_by_sid": {int(k): v for k, v in clip["name_by_sid"].items()},
        "gt_rgb": torch.stack([f["cams"]["scan_0"]["rgb"] for f in clip["frames"]], 0),
    }
    torch.save(save, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--cam", type=int, default=256)
    ap.add_argument("--window_steps", type=int, default=80)
    ap.add_argument("--depth_max", type=float, default=2.0)
    ap.add_argument("--policy_visible_only", type=int, default=0)
    ap.add_argument("--out", default="data/multiview_probe/pickcube_mv.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    rec, instruction = generate_scan_episode(args.seed, args.cam)
    clip = build_multiview_clip(
        rec, instruction, args.K, device, args.depth_max, args.window_steps,
        policy_visible_only=bool(args.policy_visible_only),
    )
    full = validate_base_view(clip, device)
    save_clip(clip, args.out, instruction, full)
    disp = (clip["traj"][-1] - clip["traj"][0]).norm(dim=-1)
    print(f"[mv] saved {args.out}", flush=True)
    print(f"[mv] views={clip['scan_views']} N={len(clip['g0'])} traj={tuple(clip['traj'].shape)}", flush=True)
    print(f"[mv] policy_uv_visible={clip['uv_visible_count']}/{clip['uv_total_before_filter']} "
          f"visible_only={clip['policy_visible_only']}", flush=True)
    print(f"[mv] psnr mean(t>=1)={float(np.mean(full[1:])):.2f} min={float(np.min(full[1:])):.2f}", flush=True)
    print(f"[mv] moving_frac={(disp > 0.01).float().mean().item():.3f}", flush=True)


if __name__ == "__main__":
    main()
