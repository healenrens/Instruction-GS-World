"""SpaTrackerV2 GT producer + flow viz — alternative to Pi3+CoTracker (Path 1). SpaTrackerV2 does 3D
point tracking (tracking + depth jointly, dynamic-scene aware) -> per-point 3D + 2D trajectories in ONE
model. We run its RGB pipeline (VGGT4Track estimates depth/intrinsics -> Predictor tracks a grid) on a
RoboTwin clip and draw the GT image-flow (frame0->frameK arrows, top movers) for visual comparison.

Usage: robotwin_spatrack_gt.py <hdf5> <out_png> [grid_size]
"""
import sys, io, os
for _k, _v in {"HF_HOME": "/mnt/pfs/public/xuhaoming/hf_cache", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "XDG_CACHE_HOME": "/mnt/pfs/public/xuhaoming/.cache"}.items():
    os.environ.setdefault(_k, _v)                                                 # never cache to the 30GB home (~/)
import numpy as np, torch, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
SPT = "/mnt/pfs/public/xuhaoming/SpaTrackerV2"; sys.path.insert(0, SPT)
from models.SpaTrackV2.models.predictor import Predictor                          # noqa: E402
from models.SpaTrackV2.models.utils import get_points_on_a_grid                   # noqa: E402
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track        # noqa: E402
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image    # noqa: E402


def load_frames(hdf5, K=12):
    if hdf5.endswith(".pt"):                                                      # drive from a .pt clip = SAME frames as Pi3 (fair)
        c = torch.load(hdf5, weights_only=False); g = c["gt_rgb"]
        g = g.numpy() if hasattr(g, "numpy") else np.asarray(g)
        if g.dtype != np.uint8: g = g * 255 if float(g.max()) <= 1.01 else g
        return g.astype(np.uint8)                                                 # [Kf+1,H,W,3]
    with h5py.File(hdf5, "r") as h:
        rgb = h["observation/head_camera/rgb"]; T = rgb.shape[0]
        idx = np.linspace(0, T - 1, K + 1).astype(int)
        frames = np.stack([np.array(Image.open(io.BytesIO(bytes(rgb[t]))).convert("RGB")) for t in idx])
    return frames.astype(np.uint8)                                                # [K+1,H,W,3]


def main():
    hdf5, out = sys.argv[1], sys.argv[2]
    grid_size = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    Kf = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    frames = load_frames(hdf5, Kf)
    vt = torch.from_numpy(frames).permute(0, 3, 1, 2).float()                     # [T,3,H,W]

    # --- RGB mode: VGGT4Track estimates depth + intrinsics ---
    vggt = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().cuda()
    vt_p = preprocess_image(vt)[None]                                             # [1,T,3,h,w]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        pred = vggt(vt_p.cuda() / 255)
        intrs = pred["intrs"]; depth = pred["points_map"][..., 2]; unc = pred["unc_metric"]
    depth_t = depth.squeeze().cpu().numpy()
    del vggt; torch.cuda.empty_cache()

    # --- Predictor: 3D point tracking of a frame-0 grid ---
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline").eval().cuda()
    vt_in = vt_p[0]                                                               # [T,3,h,w] processed
    H, W = vt_in.shape[-2:]
    grid = get_points_on_a_grid(grid_size, (H, W), device="cpu")
    qxyt = torch.cat([torch.zeros_like(grid[:, :, :1]), grid], dim=2)[0].numpy()  # [N,3] = (t=0,x,y)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        out_t = model.forward(vt_in, depth=depth_t, intrs=intrs.squeeze(0) if intrs.dim() > 3 else intrs,
                              extrs=None, queries=qxyt, fps=1, full_point=False, iters_track=4,
                              query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc,
                              support_frame=len(vt_in) - 1, replace_ratio=0.2)
    c2w, intr2, pmap, confd, track3d, track2d, vis, conf, video = out_t

    def np2(x):
        x = x.float().cpu().numpy()
        while x.ndim > 3: x = x[0]
        return x                                                                  # [T,N,C]
    t2 = np2(track2d)[..., :2]; t3 = np2(track3d)[..., :3]; vv = np2(vis[..., None])[..., 0]
    T = t2.shape[0]; Kf = T - 1
    img0 = video[0].float().cpu().numpy() if hasattr(video, "float") else np.asarray(video)[0]
    if img0.shape[0] in (3, 4): img0 = img0.transpose(1, 2, 0)
    img0 = (img0[..., :3] * (255 if img0.max() <= 1.01 else 1)).clip(0, 255).astype(np.uint8)

    d3 = np.linalg.norm(t3[Kf] - t3[0], axis=1)                                   # 3D disp per track
    d2 = np.linalg.norm(t2[Kf] - t2[0], axis=1)
    fig, ax = plt.subplots(figsize=(8, 6)); ax.imshow(img0)
    mv = np.argsort(-d2)[:120]                                                    # sort by 2D flow (same as Pi3 viz, fair)
    for i in mv:
        if d2[i] < 1: continue
        ax.annotate("", xy=(t2[Kf, i, 0], t2[Kf, i, 1]), xytext=(t2[0, i, 0], t2[0, i, 1]),
                    arrowprops=dict(arrowstyle="->", color=plt.cm.turbo(min(d2[i] / 80, 1)), lw=0.9, alpha=0.8))
    ax.set_title(f"SpaTrackerV2 GT flow ({os.path.basename(hdf5)}): top-120 movers, "
                 f"med2d {np.median(d2):.0f}px vis {vv.mean():.2f}", fontsize=9); ax.axis("off")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    np.savez(out.replace(".png", ".npz"), track2d=t2, track3d=t3, vis=vv)
    print(f"[spatrack] {os.path.basename(out)} N={t2.shape[1]} Kf={Kf} {W}x{H} med2d={np.median(d2):.1f}px vis={vv.mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
