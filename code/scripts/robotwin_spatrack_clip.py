"""SpaTrackerV2 GT clip producer (Path 1, alternative tracker to Pi3+CoTracker).

SpaTrackerV2 does joint 2D+3D point tracking, dynamic-scene aware. We emit the SAME clip dict the trainer
expects (means/uv/traj[Kf+1,N,3]/K_intr/viewmat/H/W/Kf/instruction/gt_rgb/vis) so train_gpstoken_wm.py is
untouched. Key construction (verified by robotwin_spt_probe.py):
  - fixed_cam=True : RoboTwin head camera is static -> SpaTracker returns c2w=I for all frames -> track3d
    is in ONE fixed camera frame -> per-point 3D motion = pure OBJECT motion (no Pi3-style ego-motion
    misattribution).
  - traj_cam[t,n] = unproject(track2d[t,n], z=track3d[t,n].z, K) with viewmat=I. This makes
    project_to_uv(traj, K, I) == track2d EXACTLY (clean image-flow GT) while using SpaTracker's 3D depth.
    The trainer's img_loss is image-normalized 2D flow (Du/W,Dv/H) + scale-invariant depth (Dlogz), so the
    unknown global scale never enters the loss.

Batch: loads VGGT+Predictor ONCE and loops a glob (resumable: skips existing out files).
Usage: robotwin_spatrack_clip.py --pt_glob "data/rtvid_v1/*_train.pt" --out_dir data/rtvid_v2 [--grid 48] [--vis_keep 0.3]
       robotwin_spatrack_clip.py --pt <clip.pt> --out <out.pt>            (single clip)
"""
import argparse, sys, os, io, glob as _glob
for _k, _v in {"HF_HOME": "/mnt/pfs/public/xuhaoming/hf_cache", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "XDG_CACHE_HOME": "/mnt/pfs/public/xuhaoming/.cache"}.items():
    os.environ.setdefault(_k, _v)                                                 # never cache to the 30GB home (~/)
import numpy as np, torch
from PIL import Image
SPT = "/mnt/pfs/public/xuhaoming/SpaTrackerV2"; sys.path.insert(0, SPT)
from models.SpaTrackV2.models.predictor import Predictor                          # noqa: E402
from models.SpaTrackV2.models.utils import get_points_on_a_grid                   # noqa: E402
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track        # noqa: E402
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image    # noqa: E402


def np3(x):
    x = x.float().cpu().numpy()
    while x.ndim > 3: x = x[0]
    return x                                                                      # [T,N,C]


def load_pt_frames(pt):
    c = torch.load(pt, weights_only=False); g = c["gt_rgb"]
    frames = (g.numpy() if hasattr(g, "numpy") else np.asarray(g)).astype(np.uint8)
    return frames, c.get("instruction", "")


def load_hdf5_frames(hdf5, K=12):
    import h5py
    with h5py.File(hdf5, "r") as h:
        rgb = h["observation/head_camera/rgb"]; T0 = rgb.shape[0]
        idx = np.linspace(0, T0 - 1, K + 1).astype(int)
        return np.stack([np.array(Image.open(io.BytesIO(bytes(rgb[t]))).convert("RGB")) for t in idx]).astype(np.uint8)


def process_one(frames, instruction, vggt, model, args, out):
    vt = torch.from_numpy(frames).permute(0, 3, 1, 2).float()                     # [T,3,H0,W0]
    vt_p = preprocess_image(vt)[None]                                             # [1,T,3,h,w] (pure resize)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        pred = vggt(vt_p.cuda() / 255)
        intrs = pred["intrs"]; depth = pred["points_map"][..., 2]; unc = pred["unc_metric"]
    depth_t = depth.squeeze().cpu().numpy()

    vt_in = vt_p[0]; H, W = int(vt_in.shape[-2]), int(vt_in.shape[-1])
    grid = get_points_on_a_grid(args.grid, (H, W), device="cpu")
    qxyt = torch.cat([torch.zeros_like(grid[:, :, :1]), grid], dim=2)[0].numpy()  # [N,3] (t=0,x,y)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        out_t = model.forward(vt_in, depth=depth_t, intrs=intrs.squeeze(0) if intrs.dim() > 3 else intrs,
                              extrs=None, queries=qxyt, fps=1, full_point=False, iters_track=args.iters_track,
                              query_no_BA=True, fixed_cam=True, stage=1, unc_metric=unc,
                              support_frame=len(vt_in) - 1, replace_ratio=0.2)
    c2w, intr2, pmap, confd, track3d, track2d, vis, conf, video = out_t
    t2 = np3(track2d)[..., :2]; t3 = np3(track3d)[..., :3]                        # [T,N,2],[T,N,3]
    T, N0 = t2.shape[0], t2.shape[1]; Kf = T - 1
    vv = np.squeeze(vis.float().cpu().numpy())                                    # -> [T,N]
    if vv.ndim == 1 and vv.size == N0: vv = np.broadcast_to(vv[None], (T, N0)).copy()
    if vv.shape != (T, N0): vv = np.ones((T, N0), dtype=np.float32)               # fallback: all visible
    K = (intr2.float().cpu().numpy()); K = K[0] if K.ndim == 3 else K            # fixed cam -> frame0 intrinsics
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    z = np.clip(t3[..., 2], 1e-3, None)                                           # [T,N] camera-frame depth
    xc = (t2[..., 0] - cx) / fx * z
    yc = (t2[..., 1] - cy) / fy * z
    traj = np.stack([xc, yc, z], -1).astype(np.float32)                           # [T,N,3] (reprojects to track2d)

    visf = np.clip(vv, 0.0, 1.0); visible = (visf > 0.5); vfrac = visible.mean(0)
    keep = np.where(vfrac >= args.vis_keep)[0]
    if len(keep) < 128: keep = np.arange(N0)                                      # safety: keep all if too aggressive
    traj = traj[:, keep]; visible = visible[:, keep]; uv = t2[0, keep]
    N = traj.shape[1]
    gt_rgb = torch.from_numpy(np.stack([
        np.asarray(Image.fromarray(frames[i]).resize((W, H))) for i in range(T)]).astype(np.uint8))

    traj_t = torch.from_numpy(traj)
    clip = {
        "means": traj_t[0].clone(), "uv": torch.from_numpy(uv).float(), "traj": traj_t.float(),
        "K_intr": torch.from_numpy(K.astype(np.float32)), "viewmat": torch.eye(4, dtype=torch.float32),
        "H": H, "W": W, "Kf": Kf, "instruction": instruction,
        "gt_rgb": gt_rgb, "n_fill": 0, "backend": "spatrack_p1",
        "vis": torch.from_numpy(visible), "conf0": float(visf[0].mean()),
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(clip, out)
    disp = np.linalg.norm(traj[Kf] - traj[0], axis=1)
    print(f"[spt-clip] {os.path.basename(out)} N={N} Kf={Kf} {W}x{H} f={fx:.0f} "
          f"med_disp={np.median(disp):.3f} max={disp.max():.3f} movers(>.01)={(disp>0.01).sum()} "
          f"vis_frac={visf.mean():.2f} z[{z.min():.2f},{z.max():.2f}]", flush=True)
    del out_t, track3d, track2d, vis; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt"); ap.add_argument("--hdf5"); ap.add_argument("--out")
    ap.add_argument("--pt_glob"); ap.add_argument("--out_dir")
    ap.add_argument("--instruction", default="")
    ap.add_argument("--grid", type=int, default=48, help="frame-0 grid side -> grid^2 candidate tokens")
    ap.add_argument("--vis_keep", type=float, default=0.3, help="keep tracks visible in >= this frame-fraction")
    ap.add_argument("--iters_track", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()

    tasks = []                                                                    # (clip_path_or_None, out_path)
    if args.pt_glob:
        assert args.out_dir, "--pt_glob needs --out_dir"
        for cp in sorted(_glob.glob(args.pt_glob)):
            tasks.append((cp, os.path.join(args.out_dir, os.path.basename(cp))))
    elif args.pt:
        tasks.append((args.pt, args.out))
    else:
        tasks.append((None, args.out))                                           # frames from --hdf5
    if args.nshard > 1: tasks = tasks[args.shard::args.nshard]                    # split across GPUs
    print(f"[spt-clip] {len(tasks)} clip(s) to process; grid={args.grid} vis_keep={args.vis_keep}", flush=True)

    vggt = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().cuda()
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline").eval().cuda()
    done = 0
    for cp, out in tasks:
        if out and os.path.exists(out) and not args.overwrite:
            print(f"[spt-clip] skip-exists {os.path.basename(out)}", flush=True); done += 1; continue
        try:
            if cp:
                frames, instruction = load_pt_frames(cp)
            else:
                frames, instruction = load_hdf5_frames(args.hdf5), ""
            if args.instruction: instruction = args.instruction
            process_one(frames, instruction, vggt, model, args, out)
            done += 1
        except Exception as e:
            print(f"[spt-clip] SKIP {os.path.basename(out)}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
    print(f"[spt-clip] FINISHED {done}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
