"""STAGE-2 PURE-VIDEO data-gen (agent.md §46, notes/research_pure_video_4d.md §5 P1/P3) — the twin of
maniskill_gt.py, but every geometric quantity is ESTIMATED from the RGB video (no GT depth/camera used to
GENERATE; GT mask is used only as the §46-allowed first-run `seg_per_g` shortcut, and GT is used only to
VALIDATE in st4r_stage1.py / the val PSNR here is render-vs-real-RGB which needs no GT).

Stage-1 finding (decisive, see report): St4RTrack gives EXCELLENT pure-video geometry+camera (frame-0 depth
corr 0.95 vs GT, focal err 7.7%) but its tracking branch FAILS to follow the small manipulated object
(object 2D motion ~0 vs GT 25-49 px). CoTracker3 (also pure RGB) tracks the object almost perfectly
(2D corr 0.995, vis 0.98). So this pipeline is the HYBRID the research note argued for:
  - GEOMETRY + CAMERA + frame-0 3DGS      <- St4RTrack (consistent, occlusion-aware depth)
  - per-object 3D MOTION (the traj)        <- CoTracker3 2D tracks lifted to 3D via PER-OBJECT rigid PnP
        (2D CoTracker track @ frame t  <->  frame-0 3D object point X0  +  St4R intrinsics  =>  object pose
         change T_{e,t} in the camera frame; X_t = T_{e,t} X0). This is the §38-killer "rigid-fit from the
         VISIBLE tracks" recovered from pure video, and it is robust to the gripper occluding the object
         (PnP-RANSAC on the still-visible object points). Static Gaussians (background) get traj==X0.

Emits the EXACT maniskill clip dict (means/quats/scales/opacities/colors/uv/seg_per_g/traj[Kf+1,N,3]/
K_intr/viewmat/H/W/Kf/instruction/gt_rgb/val_psnr/split) so SimClipDataset + train_sim.py are UNTOUCHED.

World gauge = frame-0 camera (== St4R gauge: world==cam0), matching the schema convention. The camera is the
STATIC LIBERO agentview, so viewmat = identity-rotation world->cam with St4R's estimated focal; all motion
lives in `traj`. Run as an offline step in the main venv (St4R + CoTracker both import under torch 2.8)."""
from __future__ import annotations

import argparse
import io
import json
import os

os.environ.setdefault("HF_HOME", "/mnt/pfs/public/xuhaoming/hf_cache")
import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "co-tracker"))

import st4r_lib
from igsw.gaussians.types import GaussianSet
from igsw.gaussians.render import render_gaussianset, psnr
from igsw.lifting.to_gaussians import points_to_gaussians


REPO = "binhng/libero_object_lerobot_mask_depth"


# --------------------------------------------------------------------------- #
# LIBERO loading
# --------------------------------------------------------------------------- #
def _decode(cell):
    from PIL import Image
    if isinstance(cell, dict) and cell.get("bytes") is not None:
        return np.array(Image.open(io.BytesIO(cell["bytes"])))
    return np.array(cell)


def load_episode_full(epi: int):
    """Return all consecutive frames of the episode (RGB[T,256,256,3], mask[T,...], ooi[T,...], instruction)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    t = pq.read_table(hf_hub_download(REPO, f"data/chunk-000/episode_{epi:06d}.parquet", repo_type="dataset"))
    n = t.num_rows
    ti = int(t["task_index"][0].as_py())
    tasks = {}
    for line in open(hf_hub_download(REPO, "meta/tasks.jsonl", repo_type="dataset")):
        d = json.loads(line); tasks[d["task_index"]] = d["task"]
    rgb = np.stack([_decode(t["observation.images.image"][i].as_py()) for i in range(n)])
    msk = np.stack([_decode(t["observation.images.image_mask"][i].as_py())[..., 0] for i in range(n)])
    ooi = np.stack([_decode(t["observation.images.object_of_interest_mask"][i].as_py())[..., 0] for i in range(n)])
    return rgb, msk, ooi, tasks.get(ti, ""), n


def find_object_id(msk_all, n, exclude=(0, 8, 10), stride: int = 2):
    """§50: the manipulated-object mask id is EPISODE-DEPENDENT (ids are per-object-TYPE: each scene
    object keeps its id across episodes; which one the task moves varies). Pick the non-robot,
    non-background id with the LARGEST centroid path over the episode. Returns (obj_id, path_px)."""
    ids = [int(i) for i in np.unique(msk_all[0]) if int(i) not in exclude]
    best, bid = -1.0, (ids[0] if ids else 1)
    for i in ids:
        cs = []
        for t in range(0, n, stride):
            m = msk_all[t] == i
            cs.append(np.argwhere(m)[:, [1, 0]].mean(0) if m.sum() > 10 else np.array([np.nan, np.nan]))
        cs = np.array(cs)
        sp = float(np.nansum(np.linalg.norm(np.diff(cs, axis=0), axis=1)))
        if sp > best:
            best, bid = sp, i
    return bid, best


def pick_window(ooi_all, n, win: int, mode: str = "center", rng=None):
    """Choose a `win`-frame CONSECUTIVE window from the object mask.

    `center` (default): centred on the strongest object motion (the clip is dominated by the
      manipulation; frame-0 already has the gripper ON the object -> gripper-proximity predicts the
      mover, so language gets no gradient — the §52a shortcut).
    `early` (§54): START 10-30 frames BEFORE motion ONSET, so frame-0 is PRE-contact (gripper still far
      from the target). Now 'move the object nearest the gripper' no longer identifies the target and
      the instruction is the ONLY signal that selects which object will move -> language must carry it."""
    cs = []
    for i in range(n):
        oo = ooi_all[i] > 0
        cs.append(np.argwhere(oo)[:, [1, 0]].mean(0) if oo.sum() > 0 else np.array([np.nan, np.nan]))
    cs = np.array(cs)
    if mode == "early":
        valid = np.where(~np.isnan(cs[:, 0]))[0]
        ref = cs[valid[0]] if len(valid) else np.array([0.0, 0.0])
        disp = np.nan_to_num(np.linalg.norm(cs - ref[None], axis=1))      # cumulative move from start
        onset = next((t for t in range(n) if disp[t] > 2.0), max(0, n - win))
        back = int(rng.integers(10, 31)) if rng is not None else 20
        bi = max(0, min(onset - back, max(0, n - win)))
        return list(range(bi, min(bi + win, n)))
    speed = np.zeros(n)
    speed[1:] = np.linalg.norm(np.diff(cs, axis=0), axis=1)
    speed = np.nan_to_num(speed)
    best, bi = -1, 0                                                       # window with max total motion
    for s in range(0, max(1, n - win)):
        tot = speed[s:s + win].sum()
        if tot > best:
            best, bi = tot, s
    return list(range(bi, min(bi + win, n)))


# --------------------------------------------------------------------------- #
# CoTracker dense 2D tracks
# --------------------------------------------------------------------------- #
def cotracker_grid(model_ct, rgb_win, query_xy, device="cuda"):
    """rgb_win [T,H0,W0,3] uint8; query_xy [Q,2] (x,y) at frame 0 (source res). Returns tracks[T,Q,2] (src
    px) and vis[T,Q]."""
    vid = torch.from_numpy(rgb_win).permute(0, 3, 1, 2)[None].float().to(device)   # [1,T,3,H,W]
    q = np.concatenate([np.zeros((len(query_xy), 1)), query_xy.astype(np.float32)], 1)  # (t=0,x,y)
    q = torch.from_numpy(q).float().to(device)[None]
    with torch.no_grad():
        tr, vis = model_ct(vid, queries=q)
    return tr[0].cpu().numpy(), vis[0].cpu().numpy()


# --------------------------------------------------------------------------- #
# per-object rigid motion from CoTracker-2D + St4R-3D via PnP
# --------------------------------------------------------------------------- #
def _fit_rigid(X, Y):
    """Kabsch: rigid (R, t) with Y ≈ X @ R.T + t. X,Y [P,3] numpy."""
    cx = X.mean(0); cy = Y.mean(0)
    H = (X - cx).T @ (Y - cy)
    U, _, Vt = np.linalg.svd(H.astype(np.float64))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = (Vt.T @ D @ U.T).astype(np.float32)
    t = (cy - cx @ R.T).astype(np.float32)
    return R, t


def _size_depth_correct(Xt_t, X0_obj, tracks2d_obj, vis_obj, t, ratio_clamp=(0.6, 1.6)):
    """§50 data-fix-2: monocular APPARENT-SIZE depth cue. Under a STATIC camera a moving object's
    depth is untriangulatable (degenerate); PnP leaves t_z weakly constrained (observed 1.7-2.5x
    overshoot). For a rigid object the 2D track spread scales as f*r/z, so z_t = z_0 * s_0/s_t with
    s computed on the COMMON-visible track subset (occlusion-robust). Correct by translating the
    solved entity ALONG ITS CENTER VIEW RAY (keeps the center's reprojection exact, object rigid)."""
    common = ((vis_obj[t] > 0.5) & (vis_obj[0] > 0.5)
              & np.isfinite(tracks2d_obj[t]).all(-1) & np.isfinite(tracks2d_obj[0]).all(-1))
    if int(common.sum()) < 8:
        return Xt_t
    p0 = tracks2d_obj[0][common]; pt = tracks2d_obj[t][common]
    s0 = float(np.sqrt(((p0 - p0.mean(0)) ** 2).sum(-1).mean()))
    st = float(np.sqrt(((pt - pt.mean(0)) ** 2).sum(-1).mean()))
    if s0 < 2.0 or st < 2.0:
        return Xt_t
    z0c = float(X0_obj[:, 2].mean())
    c = Xt_t.mean(0)
    z_tgt = float(np.clip(z0c * (s0 / st), z0c * ratio_clamp[0], z0c * ratio_clamp[1]))
    ray = c / max(float(np.linalg.norm(c)), 1e-6)
    if abs(float(ray[2])) < 0.2:
        return Xt_t
    d = (z_tgt - float(c[2])) / float(ray[2])
    return Xt_t + d * ray[None, :]


def object_pose_traj(X0_obj, tracks2d_obj, vis_obj, K_model, scale_to_model, size_z: bool = True):
    """X0_obj [No,3] frame-0 3D (St4R, camera frame) of an object's Gaussians; tracks2d_obj [T,No,2] their
    CoTracker 2D positions (SOURCE px); vis_obj [T,No]; K_model 3x3 (model res); scale_to_model maps src px
    -> model px. Solve, per frame t, the object's rigid transform T_t (cam frame) by PnP-RANSAC of the
    VISIBLE 2D-3D correspondences, then X_t = T_t @ X0. size_z applies the §50 apparent-size depth
    correction (compact rigid entities; disable for the large articulated arm). Returns Xt_obj [T,No,3]."""
    import cv2
    T = tracks2d_obj.shape[0]
    No = X0_obj.shape[0]
    Xt = np.broadcast_to(X0_obj[None], (T, No, 3)).copy()
    K = K_model.astype(np.float32)
    obj3d = X0_obj.astype(np.float32)
    last_rvec = last_tvec = None
    for t in range(1, T):
        px = (tracks2d_obj[t] * scale_to_model).astype(np.float32)        # -> model px
        good = (vis_obj[t] > 0.5) & np.isfinite(px).all(1) & (obj3d[:, 2] > 1e-5)
        if good.sum() < max(12, 0.10 * No):                              # too few/occluded -> hold pose
            # not enough visible -> hold the last solved pose (object fully occluded)
            if last_rvec is not None:
                R, _ = cv2.Rodrigues(last_rvec)
                Xt[t] = (obj3d @ R.T) + last_tvec[:, 0]
            continue
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj3d[good], px[good], K, None,
            iterationsCount=100, reprojectionError=4.0,
            flags=cv2.SOLVEPNP_ITERATIVE,
            rvec=last_rvec.copy() if last_rvec is not None else None,
            tvec=last_tvec.copy() if last_tvec is not None else None,
            useExtrinsicGuess=last_rvec is not None)
        if not ok:
            if last_rvec is not None:
                R, _ = cv2.Rodrigues(last_rvec); Xt[t] = (obj3d @ R.T) + last_tvec[:, 0]
            continue
        # PnP solves cam_T_obj for a STATIC cam observing the moved object; but our 3D is the FRAME-0 object
        # in the (static) camera frame and the 2D is the object AT frame t in the SAME camera. So PnP returns
        # the transform mapping frame-0 object points to where they reproject at frame t = exactly T_t. Apply.
        R, _ = cv2.Rodrigues(rvec)
        cand = (obj3d @ R.T) + tvec[:, 0]
        if size_z:
            cand = _size_depth_correct(cand, obj3d, tracks2d_obj, vis_obj, t)
        # TELEPORT GUARD (visual audit: a cluster with occluded/out-of-frame tracks fits a garbage
        # pose late in the window -> its Gaussians scatter). One subsampled step is ~0.3s; a centroid
        # jump > step_clamp is kinematically impossible for tabletop manipulation -> hold last pose.
        step_clamp = 0.15
        if float(np.linalg.norm(cand.mean(0) - Xt[t - 1].mean(0))) > step_clamp:
            Xt[t] = Xt[t - 1]
            continue
        Xt[t] = cand
        last_rvec, last_tvec = rvec, tvec
    return Xt


# --------------------------------------------------------------------------- #
# build one clip
# --------------------------------------------------------------------------- #
def build_clip(epi, K, win, device, model_st4r, model_ct,
               mover_2d_thresh_frac=0.04, max_obj_gauss=4000, window_mode="center"):
    rgb_all, msk_all, ooi_all, instruction, n = load_episode_full(epi)
    # §50: the manipulated object's mask id is EPISODE-dependent (ids are per-object-type; epi400's
    # mover is NOT id1) — detect it as the most-moving non-robot id, then centre the window on it.
    obj_id, obj_path = find_object_id(msk_all, n)
    print(f"[dbg] manipulated object id={obj_id} (centroid path {obj_path:.0f}px) window_mode={window_mode}", flush=True)
    widx = pick_window((msk_all == obj_id).astype(np.uint8), n, win,
                       mode=window_mode, rng=np.random.default_rng(epi))
    # subsample to K+1 frames across the window (CoTracker tracks the FULL window; we slice K+1 for the clip)
    sub = np.linspace(0, len(widx) - 1, K + 1).round().astype(int)
    sub = np.unique(sub)
    Kf = len(sub) - 1
    rgb_win = rgb_all[widx]                                               # [Tw,256,256,3] consecutive
    Tw = rgb_win.shape[0]
    H0, W0 = rgb_win.shape[1:3]

    # ---- St4R geometry + camera (pure RGB) on the SUBSAMPLED frames (for frame-0 g0 + intrinsics) ----
    rgb_sub = [rgb_all[widx[i]] for i in sub]
    res = st4r_lib.run_st4rtrack(model_st4r, rgb_sub, device=device, size=512)
    Hm, Wm = res["H"], res["W"]
    s_to_model = np.array([Wm / float(W0), Hm / float(H0)], np.float32)   # src px -> model px
    # St4R's raw pointmap is ANISOTROPICALLY DISTORTED on sim renders: its x is compressed ~2.4x vs the
    # depth/vertical axis (fit: v=fy*y/z is clean at fy~618 / resid 1.7px, but u needs fx~1499 / resid 28px).
    # estimate_focal averages the two -> biased (672) and a vertical-blob reconstruction. The VERTICAL axis is
    # clean, so estimate the focal from v=f*y/z (per-pixel median, distortion-free) and REBUILD the geometry by
    # clean pinhole backprojection (pixel grid + St4R depth). The rebuilt pointmap reprojects to its own pixels
    # => renders that match the RGB. (depth z is kept as-is: St4R depth is faithful, ~0.95 corr vs GT.)
    pts0_raw = res["pts0"]                                                # [Hm,Wm,3] distorted x
    zmap = pts0_raw[..., 2].astype(np.float32)
    ymap = pts0_raw[..., 1].astype(np.float32)
    vv = (np.arange(Hm, dtype=np.float32)[:, None] - Hm / 2.0)           # row offset from centre [Hm,1]
    uu = (np.arange(Wm, dtype=np.float32)[None, :] - Wm / 2.0)           # col offset from centre [1,Wm]
    fper = (vv * zmap) / ymap                                            # focal implied per pixel by clean v-axis
    good = np.isfinite(fper) & (np.abs(ymap) > 1e-3) & (zmap > 1e-4) & (np.abs(vv) > 8)
    focal = float(np.median(fper[good])) if int(good.sum()) > 64 else st4r_lib.estimate_focal(pts0_raw)
    K_model = np.array([[focal, 0, Wm / 2.0], [0, focal, Hm / 2.0], [0, 0, 1]], np.float32)

    # frame-0 3DGS from the REBUILT frame-0 pointmap (one Gaussian per kept pixel)
    pts0 = np.stack([(uu * zmap) / focal, (vv * zmap) / focal, zmap], axis=-1).astype(np.float32)  # [Hm,Wm,3]
    col0 = res["colors0"]                                                 # [Hm,Wm,3] in [0,1]
    conf0 = res["conf"][0]
    # keep finite, in-front points; drop the far wall (depth>p98) to bound scene radius. St4R confidence is
    # LOW on small manipulated objects, and a flat conf>1.2 gate was silently dropping the SALAD itself (the
    # very object we must track -> dead motion). So FORCE-KEEP the object-of-interest pixels (resized to model
    # res) regardless of confidence; only require valid (finite, in-front, non-far) geometry there.
    # force-keep ALL tracked-entity pixels (object id1, arm id8, gripper id10 — §50) plus ooi:
    # St4R confidence is low exactly on the small/moving things we must track.
    ent0 = np.isin(msk_all[widx[0]], (obj_id, 8, 10)).astype(np.uint8)
    ent_model = cv2.resize(ent0, (Wm, Hm), interpolation=cv2.INTER_NEAREST) > 0
    ooi_model = cv2.resize((ooi_all[widx[0]] > 0).astype(np.uint8), (Wm, Hm),
                           interpolation=cv2.INTER_NEAREST) > 0
    z = pts0[..., 2]
    # p99.5 (was p98): the p98 cut sliced the BACK WALL into fragments that "appear" when the arm
    # moves away (visual audit: the fragment cloud at t12 was kept-wall pieces, not a motion bug).
    # The wall is part of the scene -> keep it whole; only the far outlier tail is dropped.
    zmax = np.percentile(z[np.isfinite(z)], 99.5)
    valid = np.isfinite(pts0).all(-1) & (z > 1e-4) & (z < zmax)
    keep = valid & ((conf0 > 1.2) | ooi_model | ent_model)
    pts_t = torch.from_numpy(pts0).float()[None]                         # [1,Hm,Wm,3]
    img_t = torch.from_numpy(col0).float().permute(2, 0, 1)[None]        # [1,3,Hm,Wm]
    keep_t = torch.from_numpy(keep)[None]
    g0, uv = points_to_gaussians(pts_t, img_t, keep_t, opacity_init=0.9,
                                 scale_factor=0.6, scale_pct=(0.01, 0.7), return_uv=True)
    g0 = g0.to(device)
    uv = uv.to(device)                                                   # [N,2] MODEL px (x,y)
    N = len(g0)

    # ---- seg_per_g from GT mask (§46-allowed first-run shortcut) projected to Gaussians via uv ----
    msk0 = msk_all[widx[0]]                                              # [256,256]
    ooi0 = ooi_all[widx[0]] > 0
    # sample mask at each Gaussian's source pixel
    uv_src = (uv.cpu().numpy() / s_to_model)                            # -> src px
    ui = np.clip(uv_src[:, 0].round().astype(int), 0, W0 - 1)
    vi = np.clip(uv_src[:, 1].round().astype(int), 0, H0 - 1)
    seg_per_g = torch.from_numpy(msk0[vi, ui].astype(np.int64)).to(device)
    is_obj = seg_per_g == obj_id                                         # §50: the DETECTED manipulated id
    # object (mask audit §49: ooi = 63% basket + 29% gripper + 7.7% object -> ooi-based tracking fused
    # the gripper+basket into the "object" rigid fit; per-entity ids fix it).

    # ---- §50 per-ENTITY rigid motion: object(id1) + arm body(id8) + gripper(id10), each its own
    # CoTracker track set + rigid PnP (+ apparent-size depth correction for the compact entities).
    # This (a) un-freezes the ARM (data-fix-1: it was static background before), (b) un-pollutes the
    # object trajectory (gripper no longer in its fit), (c) keeps the basket/table truly static.
    traj = torch.empty(Kf + 1, N, 3, device=device, dtype=torch.float32)
    traj[:] = g0.means[None]                                             # default: STATIC (background holds X0)
    mover_frac = 0.0
    scene_r = float(np.linalg.norm(pts0.reshape(-1, 3) - pts0.reshape(-1, 3).mean(0), axis=1).std() + 1e-9)
    ENT_TRACK = (obj_id, 8, 10)                                          # object (detected), arm body, gripper
    ENT_CAP = {obj_id: 3000, 8: 2500, 10: 2500}
    seg_np = seg_per_g.cpu().numpy()
    rs = np.random.RandomState(epi)
    ent_idx = []
    for e in ENT_TRACK:
        ii = np.where(seg_np == e)[0]
        if len(ii) > ENT_CAP[e]:
            ii = rs.choice(ii, ENT_CAP[e], replace=False)
        ent_idx.append(ii)
    q_all = np.concatenate([uv_src[ii] for ii in ent_idx], 0)            # one CoTracker pass for all
    tr2d, vis2d = cotracker_grid(model_ct, rgb_win, q_all, device=device)  # [Tw,Q,2],[Tw,Q]
    tr2d_sub, vis2d_sub = tr2d[sub], vis2d[sub]                          # slice to the K+1 frames
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
        ii_all = np.where(seg_np == e)[0]                                # APPLY to the WHOLE entity
        # (bug fixed: the fit was applied only to the capped track subset -> the rest of the
        # entity's Gaussians stayed frozen = leftover ghosts.)
        if e != 8:
            if med2d <= 2.0:
                print(f"[dbg] ent{e}: Q={Q} med2d={med2d:.1f}px static", flush=True)
                continue
            X0q = g0.means[ii].cpu().numpy()
            Xtq = object_pose_traj(X0q, te, ve, K_model, s_to_model, size_z=True)
            # re-apply each frame's rigid fit (Kabsch between X0q and Xtq) to ALL entity Gaussians
            X0a = all_means_np[ii_all]
            out_a = np.broadcast_to(X0a[None], (Kf + 1, len(ii_all), 3)).copy()
            for t in range(1, Kf + 1):
                R, tv = _fit_rigid(X0q, Xtq[t])
                out_a[t] = X0a @ R.T + tv
            traj[:, ii_all] = torch.from_numpy(out_a).float().to(device)
            n_mov += len(ii_all)
            print(f"[dbg] ent{e}: Q={Q} med2d={med2d:.1f}px TRACKED (applied to {len(ii_all)})", flush=True)
        else:
            # §50 ARM (id8): base + links lumped in ONE mask id; a single rigid fit is dominated by
            # the STATIC base (median wins) and freezes the moving forearm (the "ghost arm" in the
            # visual audit). MOTION-CLUSTER the tracks (static bucket + k-means on the displacement
            # trajectory) and solve ONE RIGID PART PER CLUSTER; every id8 Gaussian follows its
            # nearest track's cluster. Sub-parts get synthetic seg ids 50+c (kept <64 for the
            # sem-prototype bank) so entity-LBS / gate-pooling treat them as separate rigid parts.
            if max2d <= 4.0:
                print(f"[dbg] ent8: Q={Q} max2d={max2d:.1f}px static", flush=True)
                continue
            feat = np.concatenate([te[Kf] - te[0], te[Kf // 2] - te[0]], 1)   # [Q,4] disp traj
            feat = np.nan_to_num(feat)
            # static bucket by MAX-over-frames displacement (end-only mislabels out-and-back
            # mid-arm sections as static -> frozen fragments, visual audit)
            dmax_t = np.nanmax(np.linalg.norm(te - te[0:1], axis=-1), axis=0)  # [Q]
            stat = np.nan_to_num(dmax_t) < 4.0
            mvq = ~stat
            labels = np.zeros(Q, np.int64)                                    # 0 = static part
            n_clu = 0
            if int(mvq.sum()) >= 40:
                from scipy.cluster.vq import kmeans2
                k_arm = 2 if int(mvq.sum()) < 400 else 3
                cen, lab = kmeans2(feat[mvq].astype(np.float64), k_arm, minit="++", seed=epi)
                labels[mvq] = lab + 1
                n_clu = int(lab.max()) + 1
                # TRACKING-FAILURE INFILL (visual audit: CoTracker loses dark low-texture joint
                # sections -> their tracks read static -> frozen fragments while the arm leaves).
                # A "static" query SURROUNDED by moving queries is a tracking failure, not a static
                # part: adopt the cluster of its nearest moving query (within 14 src px).
                from scipy.spatial import cKDTree as _KD
                q0 = uv_src[ii]
                mv_i = np.where(labels > 0)[0]
                st_i = np.where(labels == 0)[0]
                if len(mv_i) and len(st_i):
                    dmv, jmv = _KD(q0[mv_i]).query(q0[st_i], k=1)
                    adopt = dmv < 14.0
                    labels[st_i[adopt]] = labels[mv_i[jmv[adopt]]]
            # assign EVERY id8 Gaussian by the MAJORITY cluster of its k=5 nearest tracked queries
            # (k=1 misrouted boundary Gaussians -> stray static fragments in the visual audit)
            from scipy.spatial import cKDTree
            tree = cKDTree(uv_src[ii])
            _, nq5 = tree.query(uv_src[ii_all], k=5)
            lab5 = labels[nq5]                                                # [|ii_all|,5]
            lab_all = np.array([np.bincount(r).argmax() for r in lab5], np.int64)
            X0a = all_means_np[ii_all]
            out_a = np.broadcast_to(X0a[None], (Kf + 1, len(ii_all), 3)).copy()
            for c in range(1, n_clu + 1):
                qm = labels == c
                am = lab_all == c
                if int(qm.sum()) < 12 or int(am.sum()) < 1:
                    continue
                X0q = g0.means[ii[qm]].cpu().numpy()
                Xtq = object_pose_traj(X0q, te[:, qm], ve[:, qm], K_model, s_to_model, size_z=False)
                for t in range(1, Kf + 1):
                    R, tv = _fit_rigid(X0q, Xtq[t])
                    out_a[t][am] = X0a[am] @ R.T + tv
                # synthetic per-part seg id (50+c) for entity-LBS / gate pooling / sem prototypes
                seg_per_g[torch.from_numpy(ii_all[am]).to(device)] = 50 + c
                n_mov += int(am.sum())
            traj[:, ii_all] = torch.from_numpy(out_a).float().to(device)
            print(f"[dbg] ent8: Q={Q} max2d={max2d:.1f}px clusters={n_clu} "
                  f"moved={int((lab_all > 0).sum())}/{len(ii_all)}", flush=True)
    seg_np = seg_per_g.cpu().numpy()                                     # refresh (arm sub-parts)
    mover_frac = float(n_mov) / float(max(1, N))

    # ---- §50 data-fix-3: OCCLUSION-HOLE FILL — second St4R pass anchored at the LAST frame.
    # The single-frame g0 has no Gaussians behind the arm/object; when they move, black holes open.
    # The LAST frame SEES that background (the movers vacated it). Reconstruct the last frame's
    # geometry (same v-axis-focal pinhole rebuild), keep its points that (a) are NOT a tracked
    # entity at the last frame, (b) project into a frame-0 tracked-entity region (= the future
    # hole), (c) are voxel-fresh vs g0 -> append as STATIC background Gaussians (seg = last-frame
    # mask id; traj = X0). Static camera => the two passes share the camera frame canonically.
    n_fill = 0
    try:
        resB = st4r_lib.run_st4rtrack(model_st4r, [rgb_sub[i] for i in range(len(rgb_sub) - 1, -1, -1)],
                                      device=device, size=512)
        ptsB_raw = resB["pts0"]
        zB = ptsB_raw[..., 2].astype(np.float32); yB = ptsB_raw[..., 1].astype(np.float32)
        fperB = (vv * zB) / yB
        goodB = np.isfinite(fperB) & (np.abs(yB) > 1e-3) & (zB > 1e-4) & (np.abs(vv) > 8)
        focalB = float(np.median(fperB[goodB])) if int(goodB.sum()) > 64 else focal
        ptsB = np.stack([(uu * zB) / focalB, (vv * zB) / focalB, zB], -1).astype(np.float32)
        colB = resB["colors0"]; confB = resB["conf"][0]
        mskL = cv2.resize(msk_all[widx[-1]], (Wm, Hm), interpolation=cv2.INTER_NEAREST)
        msk0_m = cv2.resize(msk0, (Wm, Hm), interpolation=cv2.INTER_NEAREST)
        zBfin = zB[np.isfinite(zB)]
        zBmax = np.percentile(zBfin, 98) if zBfin.size else 2.0
        cand = (np.isfinite(ptsB).all(-1) & (zB > 1e-4) & (zB < zBmax) & (confB > 1.2)
                & (~np.isin(mskL, ENT_TRACK)))
        hole0 = np.isin(msk0_m, ENT_TRACK)
        pb = ptsB[cand]; cb = colB[cand]; sb = mskL[cand].astype(np.int64)
        u0p = np.clip(np.round(focal * pb[:, 0] / pb[:, 2] + Wm / 2.0).astype(int), 0, Wm - 1)
        v0p = np.clip(np.round(focal * pb[:, 1] / pb[:, 2] + Hm / 2.0).astype(int), 0, Hm - 1)
        in_hole = hole0[v0p, u0p]
        pb, cb, sb, u0p, v0p = pb[in_hole], cb[in_hole], sb[in_hole], u0p[in_hole], v0p[in_hole]
        if len(pb) > 64:
            g0_np = g0.means.cpu().numpy()
            # GHOST-ARM guard (visual audit): the region behind the arm is the FAR WALL, which g0
            # itself drops (z<p98) — filling it creates an isolated arm-silhouette shell floating on
            # black (a "ghost arm"). Fill must (a) stay within the scene body's depth (<= p92 of g0
            # z) and (b) EXTEND existing geometry (a g0 neighbour within 3cm), never create
            # disconnected shells.
            z_fill_max = float(np.percentile(g0_np[:, 2], 92))
            okz = pb[:, 2] <= z_fill_max
            pb, cb, sb, u0p, v0p = pb[okz], cb[okz], sb[okz], u0p[okz], v0p[okz]
            # (a 3cm-neighbour criterion was tried and KILLED legitimate patch interiors — a hole's
            # centre is >3cm from existing geometry by definition; the z-restriction alone removes
            # the far-wall ghost while keeping table/floor reveals.)
            vox = 0.004

            def _vk(P):
                q = np.floor(P / vox).astype(np.int64)
                return q[:, 0] * 73856093 + q[:, 1] * 19349663 + q[:, 2] * 83492791

            fresh = ~np.isin(_vk(pb), np.unique(_vk(g0_np)))
            pb, cb, sb, u0p, v0p = pb[fresh], cb[fresh], sb[fresh], u0p[fresh], v0p[fresh]
            if len(pb) > 40000:                                          # cap fill size
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
                from igsw.gaussians.types import GaussianSet as _GS
                g0 = _GS(torch.cat([g0.means, mfill]), torch.cat([g0.quats, qfill]),
                         torch.cat([g0.scales, med_scale]), torch.cat([g0.opacities, ofill]),
                         torch.cat([g0.colors, cfill]), None)
                uv = torch.cat([uv, torch.from_numpy(np.stack([u0p, v0p], 1)).float().to(dev_)])
                seg_per_g = torch.cat([seg_per_g, torch.from_numpy(sb).to(dev_)])
                is_obj = torch.cat([is_obj, torch.zeros(n_fill, dtype=torch.bool, device=dev_)])
                traj = torch.cat([traj, mfill[None].expand(Kf + 1, n_fill, 3)], dim=1)
                N = len(g0)
    except Exception as ex:                                              # fill is best-effort
        print(f"[dbg] hole-fill failed: {type(ex).__name__}: {ex}", flush=True)
    print(f"[dbg] hole-fill added {n_fill} static background Gaussians (N={N})", flush=True)

    viewmat = torch.eye(4, device=device)                              # world==cam0 (St4R gauge), static cam
    K_intr = torch.from_numpy(K_model).float().to(device)
    # store gt_rgb at the MODEL/render resolution (Hm,Wm = St4R size 512), NOT the native 256 — else the
    # trainer's render-loss (renders at H,W) mismatches gt_rgb. (validate() resized on-the-fly; the stored
    # clip must too.) cv2.resize takes (W,H).
    gt_rgb = torch.from_numpy(np.stack([cv2.resize(rgb_all[widx[i]], (Wm, Hm)) for i in sub])).to(torch.uint8)  # [Kf+1,Hm,Wm,3]

    return dict(g0=g0, uv=uv, seg_per_g=seg_per_g, traj=traj, K_intr=K_intr,
                viewmat=viewmat, H=Hm, W=Wm, Kf=Kf, instruction=instruction,
                gt_rgb=gt_rgb, is_obj=is_obj, focal=float(focal),
                widx=[int(widx[i]) for i in sub], scene_r=scene_r,
                mover_frac=mover_frac, n_obj=int(is_obj.sum()), n_fill=n_fill)


# --------------------------------------------------------------------------- #
# validation: render moved Gaussians vs real future RGB (needs NO GT depth/camera)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(clip, device, out_dir=None):
    """Render the frame-0 Gaussians MOVED by `traj` (means=traj[t]) from the static camera, composite over
    the frame-0 RGB background, and PSNR vs the REAL frame-t RGB (resized to model res). Reports full-frame
    + object-region PSNR (the object region = where GT-mask object IS, projected/resized)."""
    import cv2
    g0 = clip["g0"]; traj = clip["traj"]; K = clip["K_intr"]; viewmat = clip["viewmat"]
    H, W = clip["H"], clip["W"]; Kf = clip["Kf"]; gt = clip["gt_rgb"]
    is_obj = clip["is_obj"]
    full, objp = [], []
    bg0 = torch.from_numpy(cv2.resize(gt[0].numpy(), (W, H))).float().to(device) / 255.0
    for t in range(Kf + 1):
        g = g0.clone(); g.means = traj[t]
        colors, alphas, _ = render_gaussianset(g, viewmat[None], K[None], W, H)
        pred = colors[0].clamp(0, 1); a = alphas[0].clamp(0, 1)
        pred_c = pred + (1.0 - a) * bg0
        gtt = torch.from_numpy(cv2.resize(gt[t].numpy(), (W, H))).float().to(device) / 255.0
        full.append(psnr(pred_c, gtt))
        if out_dir is not None:
            import imageio.v3 as iio
            os.makedirs(out_dir, exist_ok=True)
            comp = (torch.cat([gtt, pred_c], 1).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            iio.imwrite(os.path.join(out_dir, f"val_t{t:02d}_psnr{full[-1]:.1f}.png"), comp)
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epi", type=int, default=0)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--win", type=int, default=48, help="consecutive-frame window length tracked")
    ap.add_argument("--out", default="data/libero_video/clip.pt")
    ap.add_argument("--valdir", default="")
    ap.add_argument("--split", default="train")
    ap.add_argument("--window_mode", default="center", choices=["center", "early"],
                    help="§54: 'early' starts the window PRE-contact (gripper far) to break the shortcut")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    print(f"[video_gt] epi={args.epi} K={args.K} win={args.win} window_mode={args.window_mode}", flush=True)
    model_st4r = st4r_lib.load_model(device=dev)
    from cotracker.predictor import CoTrackerPredictor
    model_ct = CoTrackerPredictor(checkpoint="checkpoints/cotracker/scaled_offline.pth",
                                  v2=False, offline=True).to(dev)
    clip = build_clip(args.epi, args.K, args.win, dev, model_st4r, model_ct, window_mode=args.window_mode)
    print(f"[video_gt] instruction={clip['instruction']!r}", flush=True)
    print(f"[video_gt] N={len(clip['g0'])} Kf={clip['Kf']} focal={clip['focal']:.1f} "
          f"n_obj_gauss={clip['n_obj']} scene_r={clip['scene_r']:.3f} mover_frac={clip['mover_frac']:.3f}", flush=True)

    val = validate(clip, dev, out_dir=(args.valdir or None))
    print(f"[video_gt] val PSNR/frame: " + " ".join(f"{p:.1f}" for p in val), flush=True)
    print(f"[video_gt] val frame0={val[0]:.2f} mean(t>=1)={np.mean(val[1:]):.2f} min(t>=1)={np.min(val[1:]):.2f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
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
    }
    torch.save(save, args.out)
    print(f"[video_gt] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
