"""Profile the SpaTrackerV2 GT producer stage-by-stage on a few windows from the shared plan.

Times: HDF5 JPEG decode | VGGT forward (depth+intrinsics) | Predictor tracking | postproc/save.
Also measures the per-EPISODE VGGT-amortization opportunity: how many distinct windows share an episode
and how much VGGT time we'd save by running depth ONCE on the full episode-frame set.

Usage: rt2_profile_producer.py --plan data/rt2_win/window_plan.json --n 8
"""
import argparse, io, json, os, sys, time
for _k, _v in {"HF_HOME": "/mnt/pfs/public/xuhaoming/hf_cache", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "XDG_CACHE_HOME": "/mnt/pfs/public/xuhaoming/.cache"}.items():
    os.environ.setdefault(_k, _v)
import numpy as np, torch
from PIL import Image
SPT = "/mnt/pfs/public/xuhaoming/SpaTrackerV2"; sys.path.insert(0, SPT)
from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.models.utils import get_points_on_a_grid
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image


def cuda_sync():
    torch.cuda.synchronize()


def decode_frames(hdf5, s, win, kf):
    idx = np.linspace(s, s + win, kf + 1).astype(int)
    import h5py
    with h5py.File(hdf5, "r") as h:
        rgb_ds = h["observation/head_camera/rgb"]
        frames = np.stack([np.array(Image.open(io.BytesIO(bytes(rgb_ds[t]))).convert("RGB"))
                           for t in idx]).astype(np.uint8)
    return frames


def vggt_depth(vggt, frames):
    vt = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    vt_p = preprocess_image(vt)[None]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        pred = vggt(vt_p.cuda() / 255)
        intrs = pred["intrs"]; depth = pred["points_map"][..., 2]; unc = pred["unc_metric"]
    return vt_p, intrs, depth, unc


def track(model, vt_p, intrs, depth, unc, grid_side, iters_track):
    vt_in = vt_p[0]; H, W = int(vt_in.shape[-2]), int(vt_in.shape[-1])
    depth_t = depth.squeeze().cpu().numpy()
    grid = get_points_on_a_grid(grid_side, (H, W), device="cpu")
    qxyt = torch.cat([torch.zeros_like(grid[:, :, :1]), grid], dim=2)[0].numpy()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        out_t = model.forward(vt_in, depth=depth_t, intrs=intrs.squeeze(0) if intrs.dim() > 3 else intrs,
                              extrs=None, queries=qxyt, fps=1, full_point=False, iters_track=iters_track,
                              query_no_BA=True, fixed_cam=True, stage=1, unc_metric=unc,
                              support_frame=len(vt_in) - 1, replace_ratio=0.2)
    return out_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--iters_track", type=int, default=4)
    args = ap.parse_args()
    plan = json.load(open(args.plan))[: args.n]

    t0 = time.time()
    vggt = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().cuda()
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline").eval().cuda()
    print(f"[load] models loaded in {time.time()-t0:.1f}s", flush=True)

    # warmup (first call has cudnn autotune / lazy init overhead)
    f0 = decode_frames(plan[0]["hdf5"], plan[0]["s"], plan[0]["win"], plan[0]["kf"])
    vt_p, intrs, depth, unc = vggt_depth(vggt, f0); cuda_sync()
    _ = track(model, vt_p, intrs, depth, unc, args.grid, args.iters_track); cuda_sync()
    print("[warmup] done", flush=True)

    agg = {"decode": 0.0, "vggt": 0.0, "track": 0.0}
    for i, w in enumerate(plan):
        t = time.time(); frames = decode_frames(w["hdf5"], w["s"], w["win"], w["kf"]); td = time.time() - t
        cuda_sync(); t = time.time(); vt_p, intrs, depth, unc = vggt_depth(vggt, frames); cuda_sync(); tv = time.time() - t
        cuda_sync(); t = time.time(); _ = track(model, vt_p, intrs, depth, unc, args.grid, args.iters_track); cuda_sync(); tt = time.time() - t
        agg["decode"] += td; agg["vggt"] += tv; agg["track"] += tt
        print(f"[win {i}] {w['name']}: decode={td:.2f} vggt={tv:.2f} track={tt:.2f} total={td+tv+tt:.2f}", flush=True)
        torch.cuda.empty_cache()
    n = len(plan)
    print(f"\n[AVG over {n}] decode={agg['decode']/n:.2f} vggt={agg['vggt']/n:.2f} track={agg['track']/n:.2f} "
          f"total={(agg['decode']+agg['vggt']+agg['track'])/n:.2f} s/window", flush=True)

    # episode-amortization opportunity
    from collections import Counter
    full = json.load(open(args.plan))
    by_ep = Counter((w["task"], w["ep"]) for w in full)
    wins = list(by_ep.values())
    print(f"[amortize] {len(full)} windows over {len(by_ep)} episodes -> "
          f"avg {np.mean(wins):.2f} windows/episode (min={min(wins)} max={max(wins)})", flush=True)
    print(f"[amortize] if VGGT runs ONCE/episode: VGGT cost drops ~{np.mean(wins):.1f}x "
          f"-> est new s/window ~ {agg['decode']/n + agg['vggt']/n/np.mean(wins) + agg['track']/n:.2f}", flush=True)


if __name__ == "__main__":
    main()
