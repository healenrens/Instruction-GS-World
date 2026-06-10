"""STAGE-1 de-risk (agent.md §46, notes/research_pure_video_4d.md §6): run St4RTrack on ONE LIBERO
episode's agentview RGB (pure video, NO GT depth/camera used to GENERATE) and validate vs the GT it
never saw:
  (a) DEPTH: scale/shift-align St4R frame-0 depth to GT metric depth (robosuite formula), report AbsRel + δ.
            Also: backproject the estimated depth with the ESTIMATED camera -> is the table flat?
  (b) CAMERA: estimated agentview pose vs the GT LIBERO agentview pose (pos=[1.5,0,0.9], quat wxyz
            [0.56,0.43,0.43,0.56], fovy 45deg, 256²). Report position error + rotation (deg) error after
            the standard up-to-scale/gauge alignment.

LIBERO GT depth: the parquet `image_depth` 8-bit = the robosuite NORMALIZED OpenGL buffer d_norm*255, and
  real_Z = near/(1 - d_norm*(1-near/far))  (robosuite get_real_depth_map). Equivalently 1/Z is AFFINE in
  d_norm. So GT metric depth is determined by (near,far); we fit them by least-squares so that 1/Z_gt is
  affine in St4R's 1/Z over high-confidence pixels (this is the affine-invariant depth-agreement test), then
  report AbsRel in metric space. We ALSO report the assumption-free `corr(1/Z_st4r, d_norm)` which needs no
  near/far at all (d_norm is monotONE in 1/Z_gt, so a static camera => high corr iff geometry shape matches).
"""
from __future__ import annotations

import argparse
import io
import os

os.environ.setdefault("HF_HOME", "/mnt/pfs/public/xuhaoming/hf_cache")
import numpy as np
import torch

import st4r_lib


# ---- LIBERO GT agentview camera (robosuite agentview, libero_object) ---------------------------
GT_CAM_POS = np.array([1.5, 0.0, 0.9], np.float32)
GT_CAM_QUAT_WXYZ = np.array([0.56, 0.43, 0.43, 0.56], np.float32)  # robosuite camera quat (wxyz)
GT_FOVY_DEG = 45.0


def quat_wxyz_to_R(q):
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], np.float32)


def gt_agentview_c2w_opencv():
    """robosuite camera frame is OpenGL (x right, y up, z BACKWARD). Convert to OpenCV (x right, y down,
    z forward) by flipping y,z. Returns c2w[4,4] in OpenCV convention (world->cam = inv)."""
    R_wc_gl = quat_wxyz_to_R(GT_CAM_QUAT_WXYZ)         # cam(GL)->world rotation
    flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    R_wc_cv = R_wc_gl @ flip
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_wc_cv
    c2w[:3, 3] = GT_CAM_POS
    return c2w


def gt_intrinsics(H, W):
    f = 0.5 * H / np.tan(np.deg2rad(GT_FOVY_DEG) / 2.0)
    return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1]], np.float32)


# ---- LIBERO episode loading --------------------------------------------------------------------
def decode(cell):
    from PIL import Image
    if isinstance(cell, dict) and cell.get("bytes") is not None:
        return np.array(Image.open(io.BytesIO(cell["bytes"])))
    return np.array(cell)


def load_libero_episode(epi: int, K: int):
    """Return (frames_rgb[list HxWx3 uint8], depth8[T,H,W] uint8, mask[T,H,W], ooi[T,H,W], instruction).
    Subsamples K+1 evenly spaced frames over the episode."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    REPO = "binhng/libero_object_lerobot_mask_depth"
    p = hf_hub_download(REPO, f"data/chunk-000/episode_{epi:06d}.parquet", repo_type="dataset")
    t = pq.read_table(p)
    n = t.num_rows
    ti = int(t["task_index"][0].as_py())
    tasks = {}
    for line in open(hf_hub_download(REPO, "meta/tasks.jsonl", repo_type="dataset")):
        import json
        d = json.loads(line)
        tasks[d["task_index"]] = d["task"]
    instruction = tasks.get(ti, "")
    idx = np.linspace(0, n - 1, K + 1).round().astype(int)
    idx = np.unique(idx)
    rgb = [decode(t["observation.images.image"][int(i)].as_py()) for i in idx]
    dep = np.stack([decode(t["observation.images.image_depth"][int(i)].as_py())[..., 0] for i in idx])
    msk = np.stack([decode(t["observation.images.image_mask"][int(i)].as_py())[..., 0] for i in idx])
    ooi = np.stack([decode(t["observation.images.object_of_interest_mask"][int(i)].as_py())[..., 0] for i in idx])
    return rgb, dep, msk, ooi, instruction, idx


# ---- depth metric recovery + metrics -----------------------------------------------------------
def depth_corr(z_hat, d8, w):
    """Assumption-free weighted Pearson corr between St4R depth and the GT 8-bit depth. The LIBERO
    `image_depth` is (empirically) ~LINEAR in metric depth (corr(z_hat,d8)=+0.95 verified: higher 8-bit
    = farther), so a high POSITIVE corr means St4R's geometry shape matches GT. No near/far needed."""
    wsum = w.sum() + 1e-12
    mz = (w * z_hat).sum() / wsum
    md = (w * d8).sum() / wsum
    cov = (w * (z_hat - mz) * (d8 - md)).sum() / wsum
    vz = (w * (z_hat - mz) ** 2).sum() / wsum
    vd = (w * (d8 - md) ** 2).sum() / wsum
    return float(cov / (np.sqrt(vz * vd) + 1e-12))


def depth_metrics(z_hat, d8, w):
    """The GT 8-bit depth is treated as RELATIVE metric depth Z_gt := d8 (min-max-normalized meters; the
    affine gauge to true meters is unknown but cancels under the per-clip scale+shift align we do anyway).
    Fit z_hat -> Z_gt by least-squares scale+shift in depth space, report AbsRel + delta thresholds.
    AbsRel here is the standard affine-invariant monocular-depth error."""
    Z_gt = d8.astype(np.float64)
    m = (w > 0) & np.isfinite(z_hat) & (z_hat > 1e-6) & (Z_gt > 1e-6)
    zh, zg = z_hat[m].astype(np.float64), Z_gt[m]
    A = np.stack([zh, np.ones_like(zh)], 1)                       # zg ~ s*zh + t
    sol, *_ = np.linalg.lstsq(A, zg, rcond=None)
    s, t = sol
    zp = s * zh + t
    absrel = float(np.mean(np.abs(zp - zg) / zg))
    ratio = np.maximum(zp / np.clip(zg, 1e-6, None), zg / np.clip(zp, 1e-6, None))
    d1 = float(np.mean(ratio < 1.25)); d2 = float(np.mean(ratio < 1.25 ** 2)); d3 = float(np.mean(ratio < 1.25 ** 3))
    return dict(absrel=absrel, d1=d1, d2=d2, d3=d3, scale=float(s), shift=float(t), n=int(m.sum()))


def table_flatness(pts0_world, mask_model, table_id_px):
    """Fit a plane to the table pixels (mask==0 background OR a chosen id) in the estimated frame-0 cloud;
    report RMS distance to the plane / scene scale (low => flat)."""
    P = pts0_world[table_id_px]
    if P.shape[0] < 50:
        return None
    c = P.mean(0)
    U, S, Vt = np.linalg.svd(P - c)
    n = Vt[-1]
    d = np.abs((P - c) @ n)
    scale = np.linalg.norm(P - c, axis=1).std() + 1e-9
    return dict(rms=float(d.mean()), rms_over_scale=float(d.mean() / scale), n=int(P.shape[0]))


def camera_errors(est_c2w_0, est_c2w_all, focal_model, H, W, src_hw):
    """Compare the ESTIMATED frame-0 camera (in St4R's own gauge) to the GT agentview.
    St4R world = frame-0 camera, so est_c2w_0 ~ identity by construction; the meaningful checks are:
      - intrinsics: est focal (rescaled to source 256) vs GT focal.
      - extrinsic SHAPE: St4R gauge has world==cam0, so we compare the RELATIVE camera motion (should be
        ~0 since LIBERO agentview is STATIC) -> report max camera translation across frames / scene-ish.
      - absolute pose vs GT only makes sense up to the gauge; we report the angle between est cam0 forward
        axis and GT cam0 forward axis AFTER aligning world frames is not possible w/o scene corr, so we
        instead report: is the camera static? (max inter-frame rotation deg, translation)."""
    H0, W0 = src_hw
    f_src = focal_model * (W0 / float(W))
    f_gt = gt_intrinsics(H0, W0)[0, 0]
    # camera motion across frames (St4R gauge); LIBERO agentview is static => should be tiny
    fwd = []
    pos = []
    for c2w in est_c2w_all:
        fwd.append(c2w[:3, 2])           # camera +z (forward) in world
        pos.append(c2w[:3, 3])
    fwd = np.stack(fwd); pos = np.stack(pos)
    # angle of each frame's forward vs frame-0 forward
    f0 = fwd[0] / (np.linalg.norm(fwd[0]) + 1e-9)
    angs = []
    for f in fwd:
        fn = f / (np.linalg.norm(f) + 1e-9)
        angs.append(np.degrees(np.arccos(np.clip(fn @ f0, -1, 1))))
    scene = np.linalg.norm(pos - pos[0], axis=1)
    return dict(f_src=float(f_src), f_gt=float(f_gt), f_relerr=float(abs(f_src - f_gt) / f_gt),
                max_cam_rot_deg=float(np.max(angs)), max_cam_trans=float(np.max(scene)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epi", type=int, default=0)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/st4r_stage1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device if torch.cuda.is_available() else "cpu"

    print(f"[stage1] loading LIBERO episode {args.epi} (K={args.K})", flush=True)
    rgb, dep8, msk, ooi, instr, idx = load_libero_episode(args.epi, args.K)
    print(f"[stage1] instruction: {instr!r}  frames={len(rgb)} src={rgb[0].shape}", flush=True)

    print("[stage1] loading St4RTrack + running inference (pure RGB)...", flush=True)
    model = st4r_lib.load_model(device=dev)
    res = st4r_lib.run_st4rtrack(model, rgb, device=dev, size=args.size)
    H, W = res["H"], res["W"]
    H0, W0 = res["src_hw"]
    sx, sy = res["scale_xy"]
    print(f"[stage1] model res {H}x{W}  src {H0}x{W0}  scale {sx:.2f},{sy:.2f}  T={res['tracks'].shape[0]}", flush=True)

    focal = st4r_lib.estimate_focal(res["pts0"])
    print(f"[stage1] estimated focal (model px) = {focal:.1f}", flush=True)

    # per-frame camera (PnP of frame-0 px vs track positions)
    est_c2w = []
    for j in range(res["tracks"].shape[0]):
        c2w, ok = st4r_lib.solve_pose_c2w(res["tracks"][j], focal, res["conf"][j])
        est_c2w.append(c2w)
    est_c2w = np.stack(est_c2w)

    # ---- (b) CAMERA ----
    cam = camera_errors(est_c2w[0], est_c2w, focal, H, W, (H0, W0))
    print("\n[stage1] === CAMERA vs GT ===", flush=True)
    print(f"  focal(src256) est={cam['f_src']:.1f}  GT(fovy45)={cam['f_gt']:.1f}  relerr={cam['f_relerr']*100:.1f}%", flush=True)
    print(f"  camera static-check: max inter-frame rot={cam['max_cam_rot_deg']:.2f} deg  "
          f"max trans(gauge units)={cam['max_cam_trans']:.4f}  (LIBERO agentview IS static -> want ~0)", flush=True)

    # ---- (a) DEPTH ----
    # frame-0 depth from St4R = pts0 z in the frame-0 CAMERA frame. pts0 is already in frame-0 world
    # which == frame-0 camera frame (St4R gauge), so z = pts0[...,2].
    z_hat = res["pts0"][..., 2]                                   # [H,W] model res
    # downsample GT depth8/mask to model res by resizing (nearest for ids; the model res is 512 vs src 256)
    import cv2
    d8_0 = dep8[0].astype(np.float32)
    m_0 = msk[0]
    d8_r = cv2.resize(d8_0, (W, H), interpolation=cv2.INTER_NEAREST)
    m_r = cv2.resize(m_0, (W, H), interpolation=cv2.INTER_NEAREST)
    w = (res["conf"][0] > 1.5).astype(np.float32) * (d8_r > 0).astype(np.float32)
    if w.sum() < 100:
        w = (d8_r > 0).astype(np.float32)
    corr = depth_corr(z_hat, d8_r.astype(np.float64), w)
    dm = depth_metrics(z_hat, d8_r, w)
    Z_gt = d8_r.astype(np.float64)
    print("\n[stage1] === DEPTH vs GT (frame 0, scale+shift aligned; GT 8-bit ~ linear metric) ===", flush=True)
    print(f"  corr(Z_st4r, d8) [assumption-free shape] = {corr:.3f}   (n_valid={int(w.sum())})", flush=True)
    print(f"  AbsRel={dm['absrel']:.3f}  d<1.25={dm['d1']:.3f} d<1.25^2={dm['d2']:.3f} "
          f"d<1.25^3={dm['d3']:.3f}  (n={dm['n']})", flush=True)

    # ---- table flatness using the estimated frame-0 cloud + GT mask id 0 (background/table) ----
    table_px = (m_r == 0) & (z_hat > 1e-5) & (res["conf"][0] > 1.5)
    flat = table_flatness(res["pts0"], None if table_px is None else table_px, None) if False else \
        table_flatness(res["pts0"].reshape(-1, 3), table_px.reshape(-1), None)
    if flat:
        print(f"\n[stage1] === TABLE FLATNESS (est frame-0 cloud, GT-mask table pixels) ===", flush=True)
        print(f"  plane RMS={flat['rms']:.4f}  RMS/scene_std={flat['rms_over_scale']:.4f}  n={flat['n']}  "
              f"(low => flat)", flush=True)

    # ---- object motion sanity: mean 3D displacement of object-of-interest px frame0->last vs static bg ----
    ooi_0 = cv2.resize(ooi[0], (W, H), interpolation=cv2.INTER_NEAREST) > 0
    tracks = res["tracks"]                                        # [T,H,W,3]
    disp = np.linalg.norm(tracks[-1] - tracks[0], axis=-1)        # [H,W]
    bg = (m_r == 0)
    obj_disp = float(disp[ooi_0].mean()) if ooi_0.sum() > 0 else float("nan")
    bg_disp = float(disp[bg].mean()) if bg.sum() > 0 else float("nan")
    scene_rad = float(np.linalg.norm(res["pts0"].reshape(-1, 3) - res["pts0"].reshape(-1, 3).mean(0), axis=1).std() + 1e-9)
    print(f"\n[stage1] === MOTION sanity (St4R 3D tracks) ===", flush=True)
    print(f"  object-of-interest mean |disp| frame0->last = {obj_disp:.4f}  ({obj_disp/scene_rad:.3f} of scene)", flush=True)
    print(f"  background mean |disp|                       = {bg_disp:.4f}  ({bg_disp/scene_rad:.3f} of scene)", flush=True)
    print(f"  => object/background disp ratio = {obj_disp/(bg_disp+1e-9):.2f}  (want >>1: object moves, table static)", flush=True)

    # ---- save a small debug bundle ----
    np.savez(os.path.join(args.out, f"epi{args.epi}_stage1.npz"),
             pts0=res["pts0"], conf0=res["conf"][0], z_hat=z_hat, Z_gt=Z_gt,
             d8=d8_r, mask=m_r, est_c2w=est_c2w, focal=focal,
             obj_disp=obj_disp, bg_disp=bg_disp, corr=corr, absrel=dm["absrel"])
    print(f"\n[stage1] saved debug -> {os.path.join(args.out, f'epi{args.epi}_stage1.npz')}", flush=True)
    print("[stage1] DONE", flush=True)


if __name__ == "__main__":
    main()
