"""Test whether VGGT depth/intrinsics for a window's frames are the SAME when VGGT is run on JUST that
window's 13 frames vs. when run as part of a larger episode-level frame batch. VGGT is a multi-view model,
so depth at a frame index may depend on the OTHER frames in the input set. If they differ, per-episode
VGGT amortization (slicing depth from an episode run) would NOT be numerically equivalent to baseline.
"""
import io, json, os, sys
for _k, _v in {"HF_HOME": "/mnt/pfs/public/xuhaoming/hf_cache", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "XDG_CACHE_HOME": "/mnt/pfs/public/xuhaoming/.cache"}.items():
    os.environ.setdefault(_k, _v)
import numpy as np, torch, h5py
from PIL import Image
SPT = "/mnt/pfs/public/xuhaoming/SpaTrackerV2"; sys.path.insert(0, SPT)
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image


def frames_at(hdf5, idx):
    with h5py.File(hdf5, "r") as h:
        rgb_ds = h["observation/head_camera/rgb"]
        return np.stack([np.array(Image.open(io.BytesIO(bytes(rgb_ds[t]))).convert("RGB"))
                         for t in idx]).astype(np.uint8)


def run_vggt(vggt, frames):
    vt = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    vt_p = preprocess_image(vt)[None]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        pred = vggt(vt_p.cuda() / 255)
    return pred["intrs"].float().cpu(), pred["points_map"][..., 2].float().cpu()


def main():
    plan = json.load(open("data/rt2_win/window_plan.json"))
    # two overlapping windows in the same episode
    ws = [w for w in plan if w["task"] == "adjust_bottle" and w["ep"] == 0]
    w0, w1 = ws[0], ws[1]
    hdf5 = w0["hdf5"]
    idx0 = np.linspace(w0["s"], w0["e"], w0["kf"] + 1).astype(int)
    idx1 = np.linspace(w1["s"], w1["e"], w1["kf"] + 1).astype(int)
    print("w0 idx:", idx0.tolist()); print("w1 idx:", idx1.tolist())
    # union (episode-level run)
    union = np.array(sorted(set(idx0.tolist()) | set(idx1.tolist())))
    print("union idx:", union.tolist())

    vggt = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().cuda()

    K0_solo, D0_solo = run_vggt(vggt, frames_at(hdf5, idx0))
    Ku, Du = run_vggt(vggt, frames_at(hdf5, union))

    # map w0 frames into union positions
    pos = {f: i for i, f in enumerate(union.tolist())}
    sel = [pos[f] for f in idx0.tolist()]
    D0_fromunion = Du[0, sel] if Du.dim() == 4 else Du[sel]
    K0_fromunion = (Ku.squeeze(0)[sel] if Ku.dim() > 3 else Ku[sel])

    D0_solo_ = D0_solo.squeeze()
    D0_fromunion_ = D0_fromunion.squeeze()
    print("D0_solo shape", tuple(D0_solo_.shape), "D0_fromunion shape", tuple(D0_fromunion_.shape))
    dd = (D0_solo_ - D0_fromunion_).abs()
    print(f"DEPTH abs-diff: max={dd.max():.6f} mean={dd.mean():.6f}  (rel-to-median-depth "
          f"{dd.max()/D0_solo_.median():.4%})")
    K0s = K0_solo.squeeze(); K0u = K0_fromunion.squeeze()
    kd = (K0s - K0u).abs()
    print(f"INTRS abs-diff: max={kd.max():.6f}")
    print("VERDICT:", "EQUIVALENT (amortize is safe)" if dd.max() < 1e-2 else
          "NOT EQUIVALENT (depth depends on input frame set -> amortize changes GT)")


if __name__ == "__main__":
    main()
