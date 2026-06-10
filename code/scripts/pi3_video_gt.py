"""PURE-VIDEO data-gen, Pi3 backend — the twin of video_gt.py (§50 St4RTrack backend).

Same pipeline shape (LIBERO episode -> window -> frame-0 3DGS + per-entity rigid motion -> clip dict),
but every geometric quantity comes from Pi3 (third_party/Pi3, checkpoints/Pi3/model.safetensors):
  - Pi3 runs ONCE over the K+1 subsampled frames and returns, per frame, a CAMERA-FRAME pointmap
    (`local_points`, z = depth), a confidence map, and a cam2world pose (OpenCV).
  - GAUGE: canonical frame = the frame-0 camera. Every per-frame quantity is mapped through inv(T0).
    LIBERO's agentview camera is STATIC, so ||translation(inv(T0)@T_t)|| ~ 0 is a free sanity check.
  - INTRINSICS: Pi3 has no intrinsics head -> recover the focal from the frame-0 local pointmap
    (per-pixel f_u=(u-cx)z/x, f_v=(v-cy)z/y, medians; on anisotropy trust the vertical axis, same
    anti-anisotropy recipe as video_gt) and REBUILD frame-0 geometry by clean pinhole backprojection
    so the cloud reprojects to its own pixels.
  - MOTION (the Pi3 advantage): Pi3 gives PER-FRAME depth, so a CoTracker 2D track at frame t lifts
    DIRECTLY to 3D (pinhole + bilinear depth + inv(T0)@T_t) — no PnP, no monocular apparent-size
    depth cue needed. Each entity/arm-cluster pose is a TRIMMED KABSCH fit (drop worst 25% residuals,
    refit) between its frame-0 3D points and the lifted frame-t points of the still-visible tracks,
    with video_gt's teleport guard + hold-last-pose fallback.
  - Arm (mask id 8) keeps video_gt's motion-clustering structure (static bucket by max track disp,
    kmeans on [end,mid] displacement, tracking-failure infill, k=5-majority Gaussian assignment,
    synthetic seg ids 50+c). Hole fill = frame-LAST rebuilt pointmap, same acceptance rules.

Emits the EXACT same clip dict as video_gt.py (means/quats/scales/opacities/colors/uv/seg_per_g/
traj[Kf+1,N,3]/K_intr/viewmat=I/H/W/Kf/instruction/gt_rgb/val_psnr/is_obj/n_fill/split) plus
focal and backend="pi3", so SimClipDataset + train_sim.py are untouched.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("HF_HOME", "/mnt/pfs/public/xuhaoming/hf_cache")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # code/  (igsw)
sys.path.insert(0, os.path.dirname(__file__))                        # scripts/ (video_gt)

import cv2
import numpy as np
import torch

from video_gt import (load_episode_full, find_object_id, pick_window,   # noqa: E402
                      cotracker_grid, validate, _fit_rigid)
from igsw.gaussians.types import GaussianSet                            # noqa: E402
from igsw.lifting.pi3_lifter import Pi3Lifter                           # noqa: E402
from igsw.lifting.to_gaussians import points_to_gaussians               # noqa: E402


# --------------------------------------------------------------------------- #
# Pi3 geometry helpers
# --------------------------------------------------------------------------- #
def estimate_focal_pi3(local0: np.ndarray):
    """Focal (px, model res) from the frame-0 CAMERA-FRAME pointmap, pp = image centre.
    Per-pixel f_u=(u-cx)z/x and f_v=(v-cy)z/y; medians. Pi3 (like St4R on sim renders) can be
    anisotropic in x -> if the two medians disagree >10%, trust the VERTICAL one (f_v).
    Returns (focal, fu_med, fv_med)."""
    Hm, Wm = local0.shape[:2]
    x = local0[..., 0].astype(np.float32)
    y = local0[..., 1].astype(np.float32)
    z = local0[..., 2].astype(np.float32)
    uu = (np.arange(Wm, dtype=np.float32)[None, :] - Wm / 2.0)   # [1,Wm]
    vv = (np.arange(Hm, dtype=np.float32)[:, None] - Hm / 2.0)   # [Hm,1]
    with np.errstate(divide="ignore", invalid="ignore"):
        fu_per = (uu * z) / x
        fv_per = (vv * z) / y
    gu = np.isfinite(fu_per) & (np.abs(x) > 1e-3) & (z > 1e-4) & (np.abs(np.broadcast_to(uu, z.shape)) > 8)
    gv = np.isfinite(fv_per) & (np.abs(y) > 1e-3) & (z > 1e-4) & (np.abs(np.broadcast_to(vv, z.shape)) > 8)
    fu = float(np.median(fu_per[gu])) if int(gu.sum()) > 64 else float("nan")
    fv = float(np.median(fv_per[gv])) if int(gv.sum()) > 64 else float("nan")
    if not np.isfinite(fv):                                       # degenerate; fall back to fu
        return fu, fu, fv
    if not np.isfinite(fu) or abs(fu - fv) / abs(fv) > 0.10:
        return fv, fu, fv
    return 0.5 * (fu + fv), fu, fv


def _pinhole(zmap: np.ndarray, focal: float):
    """Clean pinhole backprojection of a depth map -> [Hm,Wm,3] camera-frame points."""
    Hm, Wm = zmap.shape
    uu = (np.arange(Wm, dtype=np.float32)[None, :] - Wm / 2.0)
    vv = (np.arange(Hm, dtype=np.float32)[:, None] - Hm / 2.0)
    return np.stack([(uu * zmap) / focal, (vv * zmap) / focal, zmap], axis=-1).astype(np.float32)


def _bilinear_z(zmap: np.ndarray, xy: np.ndarray):
    """Bilinear depth sample. zmap [Hm,Wm]; xy [Q,2] model px (x,y). NaN where out-of-bounds
    or any of the 4 neighbours is invalid (non-finite or z<=1e-4) — depth edges must not blend."""
    Hq, Wq = zmap.shape
    out = np.full(len(xy), np.nan, np.float32)
    x = xy[:, 0]; y = xy[:, 1]
    fin = np.isfinite(x) & np.isfinite(y)
    x0 = np.floor(np.where(fin, x, 0)).astype(np.int64)
    y0 = np.floor(np.where(fin, y, 0)).astype(np.int64)
    inb = fin & (x0 >= 0) & (x0 < Wq - 1) & (y0 >= 0) & (y0 < Hq - 1)
    if not inb.any():
        return out
    xi = x0[inb]; yi = y0[inb]
    fx = (x[inb] - xi).astype(np.float32); fy = (y[inb] - yi).astype(np.float32)
    z00 = zmap[yi, xi]; z01 = zmap[yi, xi + 1]; z10 = zmap[yi + 1, xi]; z11 = zmap[yi + 1, xi + 1]
    ok = (np.isfinite(z00) & np.isfinite(z01) & np.isfinite(z10) & np.isfinite(z11)
          & (z00 > 1e-4) & (z01 > 1e-4) & (z10 > 1e-4) & (z11 > 1e-4))
    zz = z00 * (1 - fx) * (1 - fy) + z01 * fx * (1 - fy) + z10 * (1 - fx) * fy + z11 * fx * fy
    zz[~ok] = np.nan
    out[inb] = zz
    return out


def _trimmed_kabsch(X0v: np.ndarray, Xtv: np.ndarray, trim: float = 0.25):
    """Kabsch, then drop the worst `trim` fraction of residuals and refit (CoTracker drift /
    bad depth samples are heavy-tailed; a plain LSQ fit chases them)."""
    R, t = _fit_rigid(X0v, Xtv)
    resid = np.linalg.norm(X0v @ R.T + t - Xtv, axis=1)
    thr = np.percentile(resid, 100.0 * (1.0 - trim))
    keep = resid <= thr
    if int(keep.sum()) >= 3:
        R, t = _fit_rigid(X0v[keep], Xtv[keep])
    return R, t


def entity_pose_traj_pi3(X0q, te, ve, zmaps, rel, K_model, s_to_model, zmax_t,
                         step_clamp: float = 0.15):
    """Per-frame rigid pose of one entity/cluster, Pi3 style (direct 3D lift, NO PnP).

    X0q [Q,3]     canonical (frame-0 camera) 3D of the entity's tracked queries
    te  [T,Q,2]   CoTracker 2D tracks (SOURCE px); ve [T,Q] visibility
    zmaps [T,Hm,Wm] Pi3 per-frame depth (camera frame, == rebuilt depth's z)
    rel [T,4,4]   inv(T0) @ T_t : camera_t coords -> canonical
    zmax_t [T]    per-frame p99.5 depth (visibility gate)
    Returns (Rt list[(R,t)] len T with frame0=identity, Xtq [T,Q,3], n_held)."""
    T, Q, _ = te.shape
    f = float(K_model[0, 0]); cx = float(K_model[0, 2]); cy = float(K_model[1, 2])
    Rt = [(np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))]
    Xtq = np.broadcast_to(X0q[None], (T, Q, 3)).copy()
    R_last, t_last = Rt[0]
    n_held = 0
    for t in range(1, T):
        R_app, t_app = R_last, t_last
        accept = False
        px = te[t] * s_to_model                                  # src -> model px
        z = _bilinear_z(zmaps[t], px)
        vis = ((ve[t] > 0.5) & np.isfinite(px).all(1) & np.isfinite(z)
               & (z > 0.1) & (z < zmax_t[t]))
        nv = int(vis.sum())
        if nv >= max(12, int(np.ceil(0.10 * Q))):
            u = px[vis, 0]; v = px[vis, 1]; zv = z[vis]
            Xc = np.stack([(u - cx) * zv / f, (v - cy) * zv / f, zv], 1)   # camera_t frame
            P = rel[t]
            Xcan = Xc @ P[:3, :3].T + P[:3, 3]                   # -> canonical
            Rc, tc = _trimmed_kabsch(X0q[vis].astype(np.float64), Xcan.astype(np.float64))
            cand = X0q @ Rc.T + tc
            # teleport guard (== video_gt): one subsampled step is ~0.3s; centroid jumps
            # >step_clamp are kinematically impossible -> hold the last solved pose.
            if float(np.linalg.norm(cand.mean(0) - Xtq[t - 1].mean(0))) <= step_clamp:
                R_app, t_app = Rc, tc
                accept = True
        Xtq[t] = X0q @ R_app.T + t_app
        Rt.append((R_app, t_app))
        if accept:
            R_last, t_last = R_app, t_app
        else:
            n_held += 1
    return Rt, Xtq, n_held


# --------------------------------------------------------------------------- #
# build one clip
# --------------------------------------------------------------------------- #
ID_OBJ_OV = 1                                                     # open-vocab target id (== GT named-object id)


def _mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def build_clip(epi, K, win, device, lifter, model_ct, window_mode="center", seg_mode="gt"):
    rgb_all, msk_all, ooi_all, instruction, n = load_episode_full(epi)
    obj_id, obj_path = find_object_id(msk_all, n)
    print(f"[dbg] manipulated object id={obj_id} (centroid path {obj_path:.0f}px) window_mode={window_mode}", flush=True)
    widx = pick_window((msk_all == obj_id).astype(np.uint8), n, win,
                       mode=window_mode, rng=np.random.default_rng(epi))
    # ---- mask SOURCE: GT, or open-vocab (GroundingDINO+SAM2) for the two frames we actually use ----
    # The pipeline only consumes masks at widx[0] (seg_per_g + g0-keep) and widx[-1] (hole-fill exclusion).
    # Windowing / target-id selection stays on GT motion — generation-time label arbitration the plan allows.
    if seg_mode == "openvocab":
        from openvocab_seg import segment_frame_amg

        def _gt_centroid(frame):                                  # gen-time GT-motion arbitration (plan-allowed)
            ys, xs = np.where(msk_all[frame] == obj_id)
            return (float(xs.mean()), float(ys.mean())) if xs.size else None
        mask_w0 = segment_frame_amg(rgb_all[widx[0]], instruction, device=device,
                                    target_xy=_gt_centroid(widx[0]))
        mask_wL = segment_frame_amg(rgb_all[widx[-1]], instruction, device=device,
                                    target_xy=_gt_centroid(widx[-1]))
        gt0 = msk_all[widx[0]]
        tcov = float(((mask_w0 == ID_OBJ_OV) & (gt0 == obj_id)).sum() / max(1, int((gt0 == obj_id).sum())))
        iou_t = _mask_iou(mask_w0 == ID_OBJ_OV, gt0 == obj_id)
        print(f"[ov] open-vocab seg: target-cov={tcov:.2f} target-IoU={iou_t:.2f} vs GT "
              f"(point-prompted at GT mover centroid; mask is SAM2-quality)", flush=True)
    else:
        mask_w0 = msk_all[widx[0]]
        mask_wL = msk_all[widx[-1]]
    sub = np.linspace(0, len(widx) - 1, K + 1).round().astype(int)
    sub = np.unique(sub)
    Kf = len(sub) - 1
    rgb_win = rgb_all[widx]                                       # [Tw,256,256,3] consecutive
    H0, W0 = rgb_win.shape[1:3]
    rgb_sub = np.stack([rgb_all[widx[i]] for i in sub])           # [Kf+1,256,256,3]

    # ---- Pi3 ONCE over the K+1 subsampled frames -------------------------------------------
    res = lifter.lift(rgb_sub, conf_thr=0.1, edge_rtol=0.0)       # edge filter off; we gate ourselves
    local = res["local_points"].numpy()                           # [T,Hm,Wm,3] camera frame (z=depth)
    confs = res["conf"].numpy()                                   # [T,Hm,Wm] sigmoid in (0,1)
    poses = res["camera_poses"].numpy().astype(np.float64)        # [T,4,4] cam2world OpenCV
    imgs = res["images"]                                          # [T,3,Hm,Wm] float in [0,1]
    Hm, Wm = local.shape[1], local.shape[2]
    print(f"[dbg] pi3 processing res {Wm}x{Hm} (src {W0}x{H0}), frames={local.shape[0]}", flush=True)
    s_to_model = np.array([Wm / float(W0), Hm / float(H0)], np.float32)

    # ---- gauge: canonical = frame-0 camera --------------------------------------------------
    T0inv = np.linalg.inv(poses[0])
    rel = np.stack([T0inv @ poses[t] for t in range(Kf + 1)]).astype(np.float32)  # cam_t -> canonical
    cam_t = np.linalg.norm(rel[:, :3, 3], axis=1)
    print(f"[dbg] camera-static sanity: max_t ||translation(inv(T0)@T_t)|| = {cam_t.max():.4f} m "
          f"(LIBERO static cam: expect <0.02)", flush=True)

    # ---- intrinsics from frame-0 local pointmap + clean pinhole REBUILD ---------------------
    local0 = local[0]
    focal, fu_med, fv_med = estimate_focal_pi3(local0)
    print(f"[dbg] focal: f_u median={fu_med:.1f} f_v median={fv_med:.1f} -> chosen {focal:.1f} "
          f"(model px)", flush=True)
    K_model = np.array([[focal, 0, Wm / 2.0], [0, focal, Hm / 2.0], [0, 0, 1]], np.float32)
    z0 = local0[..., 2].astype(np.float32)
    pts0 = _pinhole(z0, focal)                                    # canonical == frame-0 camera frame

    # ---- g0 keep mask: valid geometry & (confident | tracked-entity pixel) ------------------
    ent0 = np.isin(mask_w0, (obj_id, 8, 10)).astype(np.uint8)
    ent_model = cv2.resize(ent0, (Wm, Hm), interpolation=cv2.INTER_NEAREST) > 0
    z0fin = z0[np.isfinite(z0)]
    zmax0 = np.percentile(z0fin, 99.5) if z0fin.size else 2.0
    valid = np.isfinite(pts0).all(-1) & (z0 > 1e-4) & (z0 < zmax0)
    keep = valid & ((confs[0] > 0.1) | ent_model)
    g0, uv = points_to_gaussians(torch.from_numpy(pts0).float()[None], imgs[0:1],
                                 torch.from_numpy(keep)[None], opacity_init=0.9,
                                 scale_factor=0.6, scale_pct=(0.01, 0.7), return_uv=True)
    g0 = g0.to(device)
    uv = uv.to(device)                                            # [N,2] MODEL px (x,y)
    N = len(g0)

    # ---- seg_per_g from GT mask (§46-allowed shortcut) at each Gaussian's source pixel ------
    msk0 = mask_w0
    uv_src = (uv.cpu().numpy() / s_to_model)
    ui = np.clip(uv_src[:, 0].round().astype(int), 0, W0 - 1)
    vi = np.clip(uv_src[:, 1].round().astype(int), 0, H0 - 1)
    seg_per_g = torch.from_numpy(msk0[vi, ui].astype(np.int64)).to(device)
    is_obj = seg_per_g == obj_id

    # ---- per-ENTITY rigid motion (object + gripper + motion-clustered arm), Pi3 lift --------
    traj = torch.empty(Kf + 1, N, 3, device=device, dtype=torch.float32)
    traj[:] = g0.means[None]                                      # default: STATIC background
    pts0_fin = pts0[np.isfinite(pts0).all(-1)].reshape(-1, 3)
    scene_r = float(np.linalg.norm(pts0_fin - pts0_fin.mean(0), axis=1).std() + 1e-9)
    ENT_TRACK = (obj_id, 8, 10)
    ENT_CAP = {obj_id: 3000, 8: 2500, 10: 2500}
    seg_np = seg_per_g.cpu().numpy()
    rs = np.random.RandomState(epi)
    ent_idx = []
    for e in ENT_TRACK:
        ii = np.where(seg_np == e)[0]
        if len(ii) > ENT_CAP[e]:
            ii = rs.choice(ii, ENT_CAP[e], replace=False)
        ent_idx.append(ii)
    q_all = np.concatenate([uv_src[ii] for ii in ent_idx], 0)     # ONE CoTracker pass for all
    tr2d, vis2d = cotracker_grid(model_ct, rgb_win, q_all, device=device)   # [Tw,Q,2],[Tw,Q]
    tr2d_sub, vis2d_sub = tr2d[sub], vis2d[sub]

    zmaps = local[..., 2].astype(np.float32)                      # [T,Hm,Wm] (rebuild keeps z as-is)
    zmax_t = np.empty(Kf + 1, np.float32)
    for t in range(Kf + 1):
        zt = zmaps[t][np.isfinite(zmaps[t]) & (zmaps[t] > 1e-4)]
        zmax_t[t] = np.percentile(zt, 99.5) if zt.size else 2.0

    off = 0
    n_mov = 0
    all_means_np = g0.means.cpu().numpy()
    for e, ii in zip(ENT_TRACK, ent_idx):
        Q = len(ii)
        if Q < 6:
            off += Q
            continue
        te = tr2d_sub[:, off:off + Q]; ve = vis2d_sub[:, off:off + Q]
        off += Q
        d2d = np.linalg.norm(te[-1] - te[0], axis=-1)
        fin = np.isfinite(d2d)
        med2d = float(np.median(d2d[fin])) if fin.any() else 0.0
        max2d = float(np.max(d2d[fin])) if fin.any() else 0.0
        ii_all = np.where(seg_np == e)[0]                         # apply to the WHOLE entity
        if e != 8:
            if med2d <= 2.0:
                print(f"[dbg] ent{e}: Q={Q} med2d={med2d:.1f}px static", flush=True)
                continue
            X0q = all_means_np[ii]
            Rt, Xtq, n_held = entity_pose_traj_pi3(X0q, te, ve, zmaps, rel, K_model,
                                                   s_to_model, zmax_t)
            X0a = all_means_np[ii_all]
            out_a = np.broadcast_to(X0a[None], (Kf + 1, len(ii_all), 3)).copy()
            for t in range(1, Kf + 1):
                R, tv = Rt[t]
                out_a[t] = X0a @ R.T + tv
            traj[:, ii_all] = torch.from_numpy(out_a).float().to(device)
            n_mov += len(ii_all)
            print(f"[dbg] ent{e}: Q={Q} med2d={med2d:.1f}px TRACKED "
                  f"(applied to {len(ii_all)}; held {n_held}/{Kf} frames)", flush=True)
        else:
            # ARM (id8): base+links share one mask id -> motion-cluster the tracks and solve one
            # rigid part per cluster (same structure as video_gt's id8 branch; only the per-frame
            # pose solve is the Pi3 lift+Kabsch instead of PnP).
            if max2d <= 4.0:
                print(f"[dbg] ent8: Q={Q} max2d={max2d:.1f}px static", flush=True)
                continue
            feat = np.concatenate([te[Kf] - te[0], te[Kf // 2] - te[0]], 1)   # [Q,4] disp traj
            feat = np.nan_to_num(feat)
            dmax_q = np.nanmax(np.linalg.norm(te - te[0:1], axis=-1), axis=0)  # [Q] max-over-frames
            stat = np.nan_to_num(dmax_q) < 4.0
            mvq = ~stat
            labels = np.zeros(Q, np.int64)                                     # 0 = static part
            n_clu = 0
            if int(mvq.sum()) >= 40:
                from scipy.cluster.vq import kmeans2
                k_arm = 2 if int(mvq.sum()) < 400 else 3
                cen, lab = kmeans2(feat[mvq].astype(np.float64), k_arm, minit="++", seed=epi)
                labels[mvq] = lab + 1
                n_clu = int(lab.max()) + 1
                # tracking-failure infill: a "static" query within 14 src px of a moving query is a
                # lost track (dark low-texture joints), not a static part -> adopt its cluster.
                from scipy.spatial import cKDTree as _KD
                q0 = uv_src[ii]
                mv_i = np.where(labels > 0)[0]
                st_i = np.where(labels == 0)[0]
                if len(mv_i) and len(st_i):
                    dmv, jmv = _KD(q0[mv_i]).query(q0[st_i], k=1)
                    adopt = dmv < 14.0
                    labels[st_i[adopt]] = labels[mv_i[jmv[adopt]]]
            # every id8 Gaussian follows the MAJORITY cluster of its k=5 nearest tracked queries
            from scipy.spatial import cKDTree
            tree = cKDTree(uv_src[ii])
            _, nq5 = tree.query(uv_src[ii_all], k=5)
            lab5 = labels[nq5]
            lab_all = np.array([np.bincount(r).argmax() for r in lab5], np.int64)
            X0a = all_means_np[ii_all]
            out_a = np.broadcast_to(X0a[None], (Kf + 1, len(ii_all), 3)).copy()
            for c in range(1, n_clu + 1):
                qm = labels == c
                am = lab_all == c
                if int(qm.sum()) < 12 or int(am.sum()) < 1:
                    continue
                X0q = all_means_np[ii[qm]]
                Rt, Xtq, n_held = entity_pose_traj_pi3(X0q, te[:, qm], ve[:, qm], zmaps, rel,
                                                       K_model, s_to_model, zmax_t)
                for t in range(1, Kf + 1):
                    R, tv = Rt[t]
                    out_a[t][am] = X0a[am] @ R.T + tv
                # synthetic per-part seg id (50+c) for entity-LBS / gate pooling / sem prototypes
                seg_per_g[torch.from_numpy(ii_all[am]).to(device)] = 50 + c
                n_mov += int(am.sum())
            traj[:, ii_all] = torch.from_numpy(out_a).float().to(device)
            print(f"[dbg] ent8: Q={Q} max2d={max2d:.1f}px clusters={n_clu} "
                  f"moved={int((lab_all > 0).sum())}/{len(ii_all)}", flush=True)
    seg_np = seg_per_g.cpu().numpy()                              # refresh (arm sub-parts)
    mover_frac = float(n_mov) / float(max(1, N))

    # ---- OCCLUSION-HOLE FILL from the frame-LAST rebuilt pointmap ----------------------------
    # The single-frame g0 has no Gaussians behind the arm/object; the LAST frame sees that
    # background. Keep last-frame points that are NOT a tracked entity at the last frame, project
    # (frame-0 K) into a frame-0 tracked-entity region (= the future hole), have ok confidence,
    # stay within the scene body's depth (z <= p92 of g0 z — ghost-arm guard), and are voxel-fresh
    # vs g0 -> append as STATIC background Gaussians (traj = X0).
    n_fill = 0
    try:
        zL = zmaps[Kf]
        confL = confs[Kf]
        ptsL_cam = _pinhole(zL, focal)
        PL = rel[Kf]
        ptsL = ptsL_cam @ PL[:3, :3].T + PL[:3, 3]                  # camera_Kf -> canonical
        mskL = cv2.resize(mask_wL, (Wm, Hm), interpolation=cv2.INTER_NEAREST)
        msk0_m = cv2.resize(msk0, (Wm, Hm), interpolation=cv2.INTER_NEAREST)
        cand = (np.isfinite(ptsL).all(-1) & (zL > 1e-4) & (ptsL[..., 2] > 1e-4)
                & (confL > 0.1) & (~np.isin(mskL, ENT_TRACK)))
        hole0 = np.isin(msk0_m, ENT_TRACK)
        colL = imgs[Kf].permute(1, 2, 0).numpy()
        pb = ptsL[cand]; cb = colL[cand]; sb = mskL[cand].astype(np.int64)
        u0p = np.clip(np.round(focal * pb[:, 0] / pb[:, 2] + Wm / 2.0).astype(int), 0, Wm - 1)
        v0p = np.clip(np.round(focal * pb[:, 1] / pb[:, 2] + Hm / 2.0).astype(int), 0, Hm - 1)
        in_hole = hole0[v0p, u0p]
        pb, cb, sb, u0p, v0p = pb[in_hole], cb[in_hole], sb[in_hole], u0p[in_hole], v0p[in_hole]
        if len(pb) > 64:
            g0_np = g0.means.cpu().numpy()
            z_fill_max = float(np.percentile(g0_np[:, 2], 92))
            okz = pb[:, 2] <= z_fill_max
            pb, cb, sb, u0p, v0p = pb[okz], cb[okz], sb[okz], u0p[okz], v0p[okz]
            vox = 0.004

            def _vk(P):
                q = np.floor(P / vox).astype(np.int64)
                return q[:, 0] * 73856093 + q[:, 1] * 19349663 + q[:, 2] * 83492791

            fresh = ~np.isin(_vk(pb), np.unique(_vk(g0_np)))
            pb, cb, sb, u0p, v0p = pb[fresh], cb[fresh], sb[fresh], u0p[fresh], v0p[fresh]
            if len(pb) > 40000:
                sel = rs.choice(len(pb), 40000, replace=False)
                pb, cb, sb, u0p, v0p = pb[sel], cb[sel], sb[sel], u0p[sel], v0p[sel]
            n_fill = len(pb)
            if n_fill > 0:
                dev_ = g0.means.device
                mfill = torch.from_numpy(pb).float().to(dev_)
                med_scale = g0.scales.median(0).values[None].expand(n_fill, 3).contiguous()
                qfill = torch.zeros(n_fill, 4, device=dev_); qfill[:, 0] = 1.0
                ofill = torch.full((n_fill,), 0.9, device=dev_)
                cfill = torch.from_numpy(cb).float().to(dev_).clamp(0, 1)
                g0 = GaussianSet(torch.cat([g0.means, mfill]), torch.cat([g0.quats, qfill]),
                                 torch.cat([g0.scales, med_scale]), torch.cat([g0.opacities, ofill]),
                                 torch.cat([g0.colors, cfill]), None)
                uv = torch.cat([uv, torch.from_numpy(np.stack([u0p, v0p], 1)).float().to(dev_)])
                seg_per_g = torch.cat([seg_per_g, torch.from_numpy(sb).to(dev_)])
                is_obj = torch.cat([is_obj, torch.zeros(n_fill, dtype=torch.bool, device=dev_)])
                traj = torch.cat([traj, mfill[None].expand(Kf + 1, n_fill, 3)], dim=1)
                N = len(g0)
    except Exception as ex:                                       # fill is best-effort
        print(f"[dbg] hole-fill failed: {type(ex).__name__}: {ex}", flush=True)
    print(f"[dbg] hole-fill added {n_fill} static background Gaussians (N={N})", flush=True)

    viewmat = torch.eye(4, device=device)                         # canonical == cam0, static cam
    K_intr = torch.from_numpy(K_model).float().to(device)
    gt_rgb = torch.from_numpy(np.stack([cv2.resize(rgb_all[widx[i]], (Wm, Hm)) for i in sub])
                              ).to(torch.uint8)                   # [Kf+1,Hm,Wm,3] model res

    return dict(g0=g0, uv=uv, seg_per_g=seg_per_g, traj=traj, K_intr=K_intr,
                viewmat=viewmat, H=Hm, W=Wm, Kf=Kf, instruction=instruction,
                gt_rgb=gt_rgb, is_obj=is_obj, focal=float(focal),
                widx=[int(widx[i]) for i in sub], scene_r=scene_r,
                mover_frac=mover_frac, n_obj=int(is_obj.sum()), n_fill=n_fill)


# --------------------------------------------------------------------------- #
# review renders (modeled on _libero_g0check.py / _libero_motioncheck.py)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def render_g0_check(pt_path, out_png, scale_mult=4.0, device="cuda"):
    """REAL frame-0 | g0 render side-by-side (scales x scale_mult so the cloud reads opaque)."""
    import imageio.v2 as iio
    from igsw.gaussians.render import render_gaussianset
    c = torch.load(pt_path, map_location=device, weights_only=False)
    real = c["gt_rgb"][0].to(device).float() / 255.0
    g = GaussianSet(c["means"].to(device), c["quats"].to(device),
                    (c["scales"] * scale_mult).to(device), c["opacities"].to(device),
                    c["colors"].to(device), None)
    col, _, _ = render_gaussianset(g, c["viewmat"][None].to(device).float(),
                                   c["K_intr"][None].to(device).float(), int(c["W"]), int(c["H"]))
    side = torch.cat([real, col[0].clamp(0, 1)], 1)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    iio.imwrite(out_png, (side * 255).to(torch.uint8).cpu().numpy())
    mn = c["means"]
    print(f"[render] g0 check -> {out_png}  focal={float(c['K_intr'][0, 0]):.1f} N={len(mn)} "
          f"x[{mn[:, 0].min():.2f},{mn[:, 0].max():.2f}] y[{mn[:, 1].min():.2f},{mn[:, 1].max():.2f}] "
          f"z[{mn[:, 2].min():.2f},{mn[:, 2].max():.2f}]", flush=True)


@torch.no_grad()
def render_motion_check(pt_path, out_png, scale_mult=4.0, device="cuda"):
    """Rows t=0 / Kf//2 / Kf, cols REAL | GT-motion render (means=traj[t], scales x scale_mult)."""
    import imageio.v2 as iio
    from igsw.gaussians.render import render_gaussianset
    c = torch.load(pt_path, map_location=device, weights_only=False)
    tr = c["traj"].to(device).float(); Kf = int(c["Kf"]); isobj = c["is_obj"].to(device).bool()
    disp = (tr[Kf] - tr[0]).norm(dim=-1)
    print(f"[render] N={disp.numel()} n_obj={int(isobj.sum())} movers(>1cm)={int((disp > 0.01).sum())} "
          f"ALL disp max {float(disp.max()):.3f} p99 {float(disp.quantile(0.99)):.3f}", flush=True)
    if int(isobj.sum()) > 0:
        od = disp[isobj]
        print(f"[render] OBJ disp: mean {float(od.mean()):.3f} max {float(od.max()):.3f} "
              f"p50 {float(od.median()):.3f}", flush=True)
    sc = (c["scales"] * scale_mult).to(device)
    vm = c["viewmat"][None].to(device).float(); Ki = c["K_intr"][None].to(device).float()
    W = int(c["W"]); H = int(c["H"])

    def _r(mn):
        g = GaussianSet(mn, c["quats"].to(device), sc, c["opacities"].to(device),
                        c["colors"].to(device), None)
        col, _, _ = render_gaussianset(g, vm, Ki, W, H)
        return col[0].clamp(0, 1).cpu().numpy()

    rows = []
    for t in [0, Kf // 2, Kf]:
        real = (c["gt_rgb"][t].float() / 255.0).cpu().numpy()
        rows.append(np.concatenate([real, _r(tr[t])], 1))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    iio.imwrite(out_png, (np.concatenate(rows, 0) * 255).astype(np.uint8))
    print(f"[render] motion check -> {out_png} (rows t=0/{Kf // 2}/{Kf}, cols REAL | GT-motion)",
          flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epi", type=int, default=0)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--win", type=int, default=48, help="consecutive-frame window length tracked")
    ap.add_argument("--out", default="data/libero_pi3/clip.pt")
    ap.add_argument("--valdir", default="")
    ap.add_argument("--split", default="train")
    ap.add_argument("--window_mode", default="center", choices=["center", "early"],
                    help="§54: 'early' starts the window PRE-contact (gripper far) to break the shortcut")
    ap.add_argument("--seg", default="gt", choices=["gt", "openvocab"],
                    help="mask source: GT sim masks, or open-vocab GroundingDINO+SAM2 (honest inference-time seg)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--g0_png", default="", help="optional REAL|g0 side-by-side render path")
    ap.add_argument("--motion_png", default="", help="optional REAL|GT-motion grid render path")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    print(f"[pi3_video_gt] epi={args.epi} K={args.K} win={args.win} window_mode={args.window_mode}", flush=True)
    lifter = Pi3Lifter(device=dev)
    from cotracker.predictor import CoTrackerPredictor
    model_ct = CoTrackerPredictor(checkpoint="checkpoints/cotracker/scaled_offline.pth",
                                  v2=False, offline=True).to(dev)
    clip = build_clip(args.epi, args.K, args.win, dev, lifter, model_ct,
                      window_mode=args.window_mode, seg_mode=args.seg)
    print(f"[pi3_video_gt] instruction={clip['instruction']!r}", flush=True)
    print(f"[pi3_video_gt] N={len(clip['g0'])} Kf={clip['Kf']} focal={clip['focal']:.1f} "
          f"n_obj_gauss={clip['n_obj']} scene_r={clip['scene_r']:.3f} "
          f"mover_frac={clip['mover_frac']:.3f}", flush=True)
    if clip["n_obj"] > 0:
        od = (clip["traj"][clip["Kf"]] - clip["traj"][0]).norm(dim=-1)[clip["is_obj"]]
        print(f"[pi3_video_gt] obj 3D endpoint disp: median={float(od.median()):.3f}m "
              f"mean={float(od.mean()):.3f}m max={float(od.max()):.3f}m (expect ~0.08-0.40)", flush=True)

    val = validate(clip, dev, out_dir=(args.valdir or None))
    print("[pi3_video_gt] val PSNR/frame: " + " ".join(f"{p:.1f}" for p in val), flush=True)
    print(f"[pi3_video_gt] val frame0={val[0]:.2f} mean(t>=1)={np.mean(val[1:]):.2f} "
          f"min(t>=1)={np.min(val[1:]):.2f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save = {
        "means": clip["g0"].means.cpu(), "quats": clip["g0"].quats.cpu(),
        "scales": clip["g0"].scales.cpu(), "opacities": clip["g0"].opacities.cpu(),
        "colors": clip["g0"].colors.cpu(), "uv": clip["uv"].cpu(),
        "seg_per_g": clip["seg_per_g"].cpu(), "traj": clip["traj"].cpu(),
        "K_intr": clip["K_intr"].cpu(), "viewmat": clip["viewmat"].cpu(),
        "H": clip["H"], "W": clip["W"], "Kf": clip["Kf"],
        "instruction": clip["instruction"], "epi": args.epi, "split": args.split,
        "val_psnr": val, "gt_rgb": clip["gt_rgb"],
        "is_obj": clip["is_obj"].cpu(), "n_fill": int(clip.get("n_fill", 0)),
        "focal": float(clip["focal"]), "backend": "pi3",
    }
    torch.save(save, args.out)
    print(f"[pi3_video_gt] saved -> {args.out}", flush=True)

    if args.g0_png:
        render_g0_check(args.out, args.g0_png, device=dev)
    if args.motion_png:
        render_motion_check(args.out, args.motion_png, device=dev)


if __name__ == "__main__":
    main()
