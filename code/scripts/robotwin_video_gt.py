"""Path 1 (POSE-FREE) GT from RoboTwin VIDEO — the real-world path (reality has no poses).

Pi3 (depth/pointmaps, no intrinsics needed) + CoTracker (dense grid track) -> per-point 3D trajectories,
purely from the head_camera RGB stream. NO simulator poses, NO object masks (Path 1 = raw per-point).
Emits the SAME clip dict the trainer expects (means/uv/traj[Kf+1,N,3]/K_intr/viewmat/H/W/Kf/instruction/
gt_rgb), so train_gpstoken_wm.py is untouched.

Gauge note: everything stays in Pi3's affine-invariant WORLD gauge. That is fine because the trainer's
img_loss target is IMAGE-NORMALIZED (Du/W, Dv/H) + depth is D log z -> both SCALE-INVARIANT, so Pi3's
unknown global scale never enters the loss.

Usage: robotwin_video_gt.py --hdf5 <ep.hdf5> --out clip.pt --K 12 [--ngrid 2048]
"""
import argparse, os, sys, io
import numpy as np, torch, h5py
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))                       # for pi3_video_gt helpers
from igsw.lifting.pi3_lifter import Pi3Lifter                       # noqa: E402
from igsw.lifting.tracking import CoTrackerTracker, sample_pointmaps_at  # noqa: E402

INSTR = {
    "pick_dual_bottles": "Pick up the two bottles and move them to the target positions.",
    "beat_block_hammer": "Pick up the hammer and beat the block.",
    "move_stapler_pad": "Move the stapler onto the pad.",
    "move_pillbottle_pad": "Move the pill bottle onto the pad.",
    "grab_roller": "Grab the roller.",
    "click_bell": "Click the bell.",
    "handover_block": "Hand over the block from one arm to the other.",
    "move_can_pot": "Move the can next to the pot.",
}


def load_robotwin_frames(hdf5_path, cam="head_camera"):
    with h5py.File(hdf5_path, "r") as h:
        rgb = h[f"observation/{cam}/rgb"]
        T = rgb.shape[0]
        frames = np.stack([np.array(Image.open(io.BytesIO(bytes(rgb[t]))).convert("RGB")) for t in range(T)]).astype(np.uint8)
    task = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(hdf5_path))))
    return frames, INSTR.get(task, task.replace("_", " "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cam", default="head_camera")
    ap.add_argument("--K", type=int, default=12, help="Kf+1 subsampled frames over the whole episode")
    ap.add_argument("--ngrid", type=int, default=2048, help="# tracked grid points = clip tokens (N)")
    ap.add_argument("--conf_thr", type=float, default=0.1)
    ap.add_argument("--vis_keep", type=float, default=0.6, help="keep only tracks CoTracker marks visible in >= this fraction of frames (drops lost/occluded/drifting -> cleaner GT)")
    args = ap.parse_args(); dev = "cuda"

    frames, instruction = load_robotwin_frames(args.hdf5, args.cam)
    T = frames.shape[0]
    sub = np.unique(np.linspace(0, T - 1, args.K + 1).astype(int))    # K+1 frames over the episode
    Kf = len(sub) - 1
    rgb_sub = frames[sub]                                            # [Kf+1,H0,W0,3]

    lifter = Pi3Lifter()
    res = lifter.lift(rgb_sub, conf_thr=args.conf_thr, edge_rtol=0.0)
    glob = res["points"].numpy().astype(np.float32)                 # [T,Hm,Wm,3] joint WORLD gauge
    local = res["local_points"].numpy().astype(np.float32)          # [T,Hm,Wm,3] camera frame (z=depth)
    conf = res["conf"].numpy().astype(np.float32)                   # [T,Hm,Wm] in (0,1)
    poses = res["camera_poses"].numpy().astype(np.float64)          # [T,4,4] cam2world
    imgs = res["images"]                                            # [T,3,Hm,Wm] float[0,1]
    Hm, Wm = local.shape[1], local.shape[2]

    # focal from frame-0 local pointmap (per-pixel f=(u-cx)z/x), median
    z0 = local[0, ..., 2]
    yy, xx = np.mgrid[0:Hm, 0:Wm].astype(np.float32)
    valid0 = np.isfinite(z0) & (z0 > 1e-4)
    fu = ((xx - Wm / 2.0) * z0) / np.where(np.abs(local[0, ..., 0]) > 1e-6, local[0, ..., 0], np.nan)
    fv = ((yy - Hm / 2.0) * z0) / np.where(np.abs(local[0, ..., 1]) > 1e-6, local[0, ..., 1], np.nan)
    focal = float(np.nanmedian(np.concatenate([fu[valid0], fv[valid0]])))
    K_intr = torch.tensor([[focal, 0, Wm / 2.0], [0, focal, Hm / 2.0], [0, 0, 1]], dtype=torch.float32)
    # STATIC-CAMERA gauge: canonical = the fixed camera frame, viewmat = I. We do NOT use Pi3's camera
    # poses: on dynamic manipulation scenes Pi3 (SfM, assumes static scene) MISATTRIBUTES object motion as
    # camera ego-motion (verified: real cam motion 0.000m, Pi3 est 0.30 + 9.8deg). Using per-frame LOCAL
    # pointmaps in the fixed camera frame => per-point motion = true OBJECT motion, no spurious global drift.
    viewmat = torch.eye(4, dtype=torch.float32)

    # frame-0 grid of valid+confident pixels -> tracked queries (these become the clip's N tokens)
    keep = valid0 & (conf[0] > args.conf_thr)
    vs, us = np.where(keep)
    if len(vs) > args.ngrid:
        sel = np.random.RandomState(0).choice(len(vs), args.ngrid, replace=False)
        vs, us = vs[sel], us[sel]
    queries = np.stack([us, vs], 1).astype(np.float32)              # [N,2] (x,y) model px
    N = len(queries)

    # CoTracker over the model-res frames, querying the grid at frame 0
    tracker = CoTrackerTracker()
    vid = (imgs if torch.is_tensor(imgs) else torch.from_numpy(imgs)).float()       # [T,3,Hm,Wm]
    tracks, vis = tracker.track(vid, torch.from_numpy(queries).to(dev))             # [T,N,2],[T,N]

    # lift tracks -> per-point 3D trajectory in the FIXED CAMERA frame (Path 1: raw, no rigid fit).
    # Use LOCAL (per-frame camera-frame) pointmaps, NOT global, to avoid Pi3's ego-motion misattribution.
    traj_w = sample_pointmaps_at(torch.from_numpy(local).to(dev), tracks.to(dev))   # [T,N,3] camera frame
    # vis-filter: drop tracks lost/occluded/drifting in too many frames (task-dependent noise) -> cleaner GT
    vf = vis.float().mean(0)                                                         # [N] visible fraction
    ki = torch.where(vf >= args.vis_keep)[0]
    if len(ki) >= 128:
        traj_w = traj_w[:, ki]; vis = vis[:, ki]; queries = queries[ki.cpu().numpy()]
    N = traj_w.shape[1]
    means = traj_w[0].clone()                                                       # frame0 3D (camera frame)
    # uv in MODEL px (frame0 query) — the trainer places tokens by these + the rgb
    uv = torch.from_numpy(queries).float()

    gt_rgb = torch.from_numpy(np.stack([
        np.asarray(Image.fromarray(frames[sub[i]]).resize((Wm, Hm))) for i in range(Kf + 1)]).astype(np.uint8))

    clip = {
        "means": means.cpu(), "uv": uv.cpu(), "traj": traj_w.cpu().float(),
        "K_intr": K_intr, "viewmat": viewmat, "H": Hm, "W": Wm, "Kf": Kf,
        "instruction": instruction, "gt_rgb": gt_rgb, "n_fill": 0,
        "backend": "pi3_robotwin_p1", "vis": vis.cpu(), "conf0": float((conf[0] > args.conf_thr).mean()),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(clip, args.out)
    disp = (traj_w[Kf] - traj_w[0]).norm(dim=-1)
    print(f"[robotwin-p1] {os.path.basename(args.out)} N={N} Kf={Kf} {Wm}x{Hm} focal={focal:.0f} "
          f"mover(>1%scene): med_disp={float(disp.median()):.3f} max={float(disp.max()):.3f} "
          f"vis_frac={float(vis.float().mean()):.2f}", flush=True)


if __name__ == "__main__":
    main()
