"""Probe: determine the coordinate frame of SpaTrackerV2's track3d so the GT clip's traj reprojects to
track2d via project_to_uv(traj, K, viewmat). Runs the SpaTracker pipeline on one .pt clip's frames and
prints per-frame reprojection error under two hypotheses:
  A) track3d is in the (per-frame) CAMERA frame  -> uv = K @ (t3 / t3.z)
  B) track3d is in a WORLD frame                 -> uv = K @ (w2c @ [t3;1] / z)
Whichever error ~0 px tells us how to build traj. Usage: robotwin_spt_probe.py <clip.pt> [grid]
"""
import sys, os
for _k, _v in {"HF_HOME": "/mnt/pfs/public/xuhaoming/hf_cache", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "XDG_CACHE_HOME": "/mnt/pfs/public/xuhaoming/.cache"}.items():
    os.environ.setdefault(_k, _v)
import numpy as np, torch
SPT = "/mnt/pfs/public/xuhaoming/SpaTrackerV2"; sys.path.insert(0, SPT)
from models.SpaTrackV2.models.predictor import Predictor                          # noqa: E402
from models.SpaTrackV2.models.utils import get_points_on_a_grid                   # noqa: E402
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track        # noqa: E402
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image    # noqa: E402


def np3(x):
    x = x.float().cpu().numpy()
    while x.ndim > 3: x = x[0]
    return x


def main():
    clip = sys.argv[1]; grid_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    fixed_cam = (sys.argv[3].lower() in ("1", "true")) if len(sys.argv) > 3 else False
    print(f"fixed_cam={fixed_cam}", flush=True)
    c = torch.load(clip, weights_only=False); g = c["gt_rgb"]
    frames = (g.numpy() if hasattr(g, "numpy") else np.asarray(g)).astype(np.uint8)  # [T,H,W,3]
    vt = torch.from_numpy(frames).permute(0, 3, 1, 2).float()                        # [T,3,H,W]

    vggt = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().cuda()
    vt_p = preprocess_image(vt)[None]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        pred = vggt(vt_p.cuda() / 255)
        intrs = pred["intrs"]; depth = pred["points_map"][..., 2]; unc = pred["unc_metric"]
    depth_t = depth.squeeze().cpu().numpy()
    del vggt; torch.cuda.empty_cache()

    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline").eval().cuda()
    vt_in = vt_p[0]; H, W = vt_in.shape[-2:]
    grid = get_points_on_a_grid(grid_size, (H, W), device="cpu")
    qxyt = torch.cat([torch.zeros_like(grid[:, :, :1]), grid], dim=2)[0].numpy()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        out_t = model.forward(vt_in, depth=depth_t, intrs=intrs.squeeze(0) if intrs.dim() > 3 else intrs,
                              extrs=None, queries=qxyt, fps=1, full_point=False, iters_track=4,
                              query_no_BA=True, fixed_cam=fixed_cam, stage=1, unc_metric=unc,
                              support_frame=len(vt_in) - 1, replace_ratio=0.2)
    c2w, intr2, pmap, confd, track3d, track2d, vis, conf, video = out_t
    t2 = np3(track2d)[..., :2]; t3 = np3(track3d)[..., :3]; T = t2.shape[0]
    I2 = intr2.float().cpu().numpy(); C2W = c2w.float().cpu().numpy()
    print(f"shapes: track2d={t2.shape} track3d={t3.shape} intr2={I2.shape} c2w={C2W.shape} HxW={H}x{W}", flush=True)
    print(f"track3d z: min={t3[...,2].min():.3f} max={t3[...,2].max():.3f} med={np.median(t3[...,2]):.3f}", flush=True)
    print(f"c2w[0]=\n{np.round(C2W[0] if C2W.ndim==3 else C2W,3)}", flush=True)

    def proj(p3, K):
        z = np.clip(p3[..., 2:3], 1e-4, None); uvh = p3 / z
        return (uvh @ K.T)[..., :2]
    eA, eB = [], []
    for t in range(T):
        Kt = I2[t] if I2.ndim == 3 else I2
        Ct = C2W[t] if C2W.ndim == 3 else C2W
        pA = proj(t3[t], Kt)
        w2c = np.linalg.inv(Ct)
        cam = (np.concatenate([t3[t], np.ones((t3.shape[1], 1))], 1) @ w2c.T)[:, :3]
        pB = proj(cam, Kt)
        eA.append(np.median(np.linalg.norm(pA - t2[t], axis=1)))
        eB.append(np.median(np.linalg.norm(pB - t2[t], axis=1)))
    print(f"reproj err (median px) per frame:\n  A camera-frame: {np.round(eA,2)}", flush=True)
    print(f"  B world-frame:  {np.round(eB,2)}", flush=True)
    print(f"VERDICT: {'A=camera-frame' if np.mean(eA)<np.mean(eB) else 'B=world-frame'} "
          f"(meanA={np.mean(eA):.2f} meanB={np.mean(eB):.2f})", flush=True)
    # also: does unproject(track2d, depthmap@track2d) match? (the Pi3-style lift)
    print("track2d[0,:3]=", np.round(t2[0, :3], 1), " track2d[-1,:3]=", np.round(t2[-1, :3], 1), flush=True)


if __name__ == "__main__":
    main()
