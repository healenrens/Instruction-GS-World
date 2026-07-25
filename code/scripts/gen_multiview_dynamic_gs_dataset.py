"""Stage-B fixed-ID DynamicGS dataset generator with complete multiview g0.

This keeps all canonical scan Gaussians and stores per-view projection metadata
for future multi-view Qwen grounding. It does not filter to policy-visible points.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.gaussians.types import GaussianSet  # noqa: E402
from igsw.lifting.to_gaussians import points_to_gaussians  # noqa: E402
from scripts.gen_multiview_dynamic_gs_probe import (  # noqa: E402
    project_to_camera,
    validate_base_view,
)
from scripts.multiview_scan_envs import generate_scan_episode  # noqa: E402
from scripts.maniskill_gt import (  # noqa: E402
    EXCLUDE_NAMES,
    backproject_to_world,
    extrinsic_to_viewmat,
    pose_to_matrix,
)


def _dedupe_gaussians(g0, seg_per_g, source_view, source_uv, voxel_m):
    q = torch.round(g0.means / float(voxel_m)).long()
    key = torch.cat([seg_per_g.long()[:, None], q], dim=1)
    uniq, inv = torch.unique(key, return_inverse=True, dim=0)
    n = uniq.shape[0]
    count = torch.bincount(inv, minlength=n).to(g0.means.dtype).clamp_min(1.0)

    means = torch.zeros(n, 3, device=g0.means.device, dtype=g0.means.dtype)
    colors = torch.zeros(n, 3, device=g0.colors.device, dtype=g0.colors.dtype)
    scales = torch.zeros(n, 3, device=g0.scales.device, dtype=g0.scales.dtype)
    opacities = torch.zeros(n, device=g0.opacities.device, dtype=g0.opacities.dtype)
    means.index_add_(0, inv, g0.means)
    colors.index_add_(0, inv, g0.colors)
    scales.index_add_(0, inv, g0.scales)
    opacities.index_add_(0, inv, g0.opacities)
    means = means / count[:, None]
    colors = colors / count[:, None]
    scales = scales / count[:, None]
    opacities = opacities / count

    order = torch.argsort(inv)
    sorted_inv = inv[order]
    starts = torch.cat([
        torch.zeros(1, device=order.device, dtype=torch.long),
        (sorted_inv[1:] != sorted_inv[:-1]).nonzero(as_tuple=True)[0] + 1,
    ])
    first = order[starts]
    quats = g0.quats[first]
    gs = GaussianSet(means, quats, scales, opacities, colors, None).validate()
    return gs, uniq[:, 0].long(), source_view[first], source_uv[first], count.long()


def _project_depth_visible(points, K, viewmat, depth_m, tol_m):
    H, W = int(depth_m.shape[0]), int(depth_m.shape[1])
    uv, in_frame = project_to_camera(points, K, viewmat, H, W)
    ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
    cam = torch.cat([points, ones], -1) @ viewmat.T
    z = cam[:, 2]
    px = uv[:, 0].round().long().clamp(0, W - 1)
    py = uv[:, 1].round().long().clamp(0, H - 1)
    zbuf = depth_m[py, px]
    depth_ok = torch.isfinite(zbuf) & (zbuf > 1e-4) & ((zbuf - z).abs() <= float(tol_m))
    return uv, in_frame & depth_ok


def build_stage_b_clip(rec, instruction, K_frames, device, depth_max, voxel_m,
                       vis_tol_m, window_steps=None):
    n_sim = len(rec.frames)
    win = n_sim - 1 if window_steps is None else int(min(max(1, window_steps), n_sim - 1))
    idx = np.linspace(0, win, K_frames + 1).round().astype(int)
    frames = [rec.frames[i] for i in idx]
    poses = [rec.entity_poses[i] for i in idx]
    f0 = frames[0]
    cam_ids = sorted(k for k in f0["cams"] if str(k).startswith("scan_"))
    if not cam_ids:
        raise RuntimeError("Stage-B clip requires scan_* cameras")
    name_by_sid = {
        int(sid): getattr(ent, "name", str(sid))
        for sid, ent in rec.env.unwrapped.segmentation_id_map.items()
    }
    exclude = {sid for sid, nm in name_by_sid.items() if nm in EXCLUDE_NAMES}

    pts, imgs, masks, segs = [], [], [], []
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
            keep &= seg != int(sid)
        pts.append(p)
        imgs.append(rgb)
        masks.append(keep)
        segs.append(seg)

    mask_stack = torch.stack(masks, 0)
    g0_raw, uv_native = points_to_gaussians(
        torch.stack(pts, 0), torch.stack(imgs, 0), mask_stack,
        opacity_init=0.9, scale_factor=0.6, scale_pct=(0.01, 0.7), return_uv=True,
    )
    seg_per_g = torch.stack(segs, 0)[mask_stack].long()
    view_grid = torch.arange(len(cam_ids), device=device)[:, None, None].expand_as(mask_stack)
    source_view = view_grid[mask_stack].long()
    g0_raw = g0_raw.to(device)
    g0, seg_per_g, source_view, source_uv, merge_count = _dedupe_gaussians(
        g0_raw, seg_per_g, source_view, uv_native.to(device), voxel_m)

    scan_K, scan_viewmat, scan_rgb, uv_by_view, valid_by_view = [], [], [], [], []
    for cid in cam_ids:
        c = f0["cams"][cid]
        K = c["K"].to(device).float()
        viewmat = extrinsic_to_viewmat(c["extrinsic_cv"].to(device).float())
        depth = c["depth"].to(device).float() / 1000.0
        uv, valid = _project_depth_visible(g0.means, K, viewmat, depth, vis_tol_m)
        scan_K.append(K)
        scan_viewmat.append(viewmat)
        scan_rgb.append(c["rgb"])
        uv_by_view.append(uv)
        valid_by_view.append(valid)

    uv_by_view = torch.stack(uv_by_view, 1)
    valid_by_view = torch.stack(valid_by_view, 1)
    any_valid = valid_by_view.any(1)
    first_valid = valid_by_view.float().argmax(1).long()
    best_view = torch.where(any_valid, first_valid, source_view)
    best_uv = uv_by_view[torch.arange(len(g0), device=device), best_view]
    policy_view = 0
    H, W = int(f0["cams"][cam_ids[policy_view]]["depth"].shape[0]), int(f0["cams"][cam_ids[policy_view]]["depth"].shape[1])

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
        "g0": g0, "seg_per_g": seg_per_g, "traj": traj,
        "uv": uv_by_view[:, policy_view], "uv_valid": valid_by_view[:, policy_view],
        "uv_by_view": uv_by_view, "uv_valid_by_view": valid_by_view,
        "source_view": source_view, "source_uv": source_uv,
        "best_view": best_view, "best_uv": best_uv, "merge_count": merge_count,
        "K_intr": scan_K[policy_view], "viewmat": scan_viewmat[policy_view],
        "scan_K_intr": torch.stack(scan_K, 0), "scan_viewmat": torch.stack(scan_viewmat, 0),
        "scan_rgb": torch.stack(scan_rgb, 0), "scan_cam_ids": cam_ids,
        "H": H, "W": W, "Kf": len(idx) - 1, "frames": frames, "poses": poses,
        "instruction": instruction, "name_by_sid": name_by_sid, "uniq_seg": uniq,
        "scan_views": len(cam_ids), "policy_view": policy_view,
        "contract_version": "dynamic_gs_stage_b_v1",
    }


def save_stage_b_clip(clip, out, env, seed, split, val_psnr):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save = {
        "contract_version": clip["contract_version"],
        "means": clip["g0"].means.cpu(), "quats": clip["g0"].quats.cpu(),
        "scales": clip["g0"].scales.cpu(), "opacities": clip["g0"].opacities.cpu(),
        "colors": clip["g0"].colors.cpu(), "seg_per_g": clip["seg_per_g"].cpu(),
        "traj": clip["traj"].cpu(), "uv": clip["uv"].cpu(), "uv_valid": clip["uv_valid"].cpu(),
        "uv_by_view": clip["uv_by_view"].cpu(), "uv_valid_by_view": clip["uv_valid_by_view"].cpu(),
        "source_view": clip["source_view"].cpu(), "source_uv": clip["source_uv"].cpu(),
        "best_view": clip["best_view"].cpu(), "best_uv": clip["best_uv"].cpu(),
        "merge_count": clip["merge_count"].cpu(), "K_intr": clip["K_intr"].cpu(),
        "viewmat": clip["viewmat"].cpu(), "scan_K_intr": clip["scan_K_intr"].cpu(),
        "scan_viewmat": clip["scan_viewmat"].cpu(), "scan_rgb": clip["scan_rgb"].cpu(),
        "scan_cam_ids": clip["scan_cam_ids"], "H": clip["H"], "W": clip["W"], "Kf": clip["Kf"],
        "instruction": clip["instruction"], "env": env, "seed": seed, "split": split,
        "scan_views": clip["scan_views"], "policy_view": clip["policy_view"],
        "val_psnr": val_psnr, "poses": [{int(k): v.cpu() for k, v in p.items()} for p in clip["poses"]],
        "name_by_sid": {int(k): v for k, v in clip["name_by_sid"].items()},
        "gt_rgb": torch.stack([f["cams"][clip["scan_cam_ids"][clip["policy_view"]]]["rgb"] for f in clip["frames"]], 0),
    }
    torch.save(save, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PickCube-v1")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--split", default="train")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--cam", type=int, default=256)
    ap.add_argument("--window_steps", type=int, default=80)
    ap.add_argument("--depth_max", type=float, default=2.0)
    ap.add_argument("--voxel_m", type=float, default=0.003)
    ap.add_argument("--vis_tol_m", type=float, default=0.025)
    ap.add_argument("--validate_render", type=int, default=1)
    ap.add_argument("--out", default="data/multiview_stage_b/pickcube_s1000_train.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    rec, instruction = generate_scan_episode(args.env, args.seed, args.cam)
    clip = build_stage_b_clip(
        rec, instruction, args.K, device, args.depth_max, args.voxel_m,
        args.vis_tol_m, window_steps=args.window_steps)
    if args.validate_render:
        if device == "cpu":
            raise RuntimeError("render validation uses gsplat CUDA; rerun with --device cuda or set --validate_render 0")
        val_psnr = validate_base_view(clip, device)
    else:
        val_psnr = []
    save_stage_b_clip(clip, args.out, args.env, args.seed, args.split, val_psnr)
    valid = clip["uv_valid_by_view"]
    print(f"[stage-b] saved {args.out}", flush=True)
    print(f"[stage-b] N={len(clip['g0'])} views={clip['scan_views']} traj={tuple(clip['traj'].shape)}", flush=True)
    print(f"[stage-b] any_view_visible={int(valid.any(1).sum())}/{valid.shape[0]} "
          f"policy_visible={int(clip['uv_valid'].sum())}/{valid.shape[0]}", flush=True)
    if val_psnr:
        print(f"[stage-b] psnr mean(t>=1)={float(np.mean(val_psnr[1:])):.2f} "
              f"min={float(np.min(val_psnr[1:])):.2f}", flush=True)
    else:
        print("[stage-b] render validation skipped by --validate_render 0", flush=True)


if __name__ == "__main__":
    main()
