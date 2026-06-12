"""Verify CURRENT training-data fidelity: render [ real RGB | pseudo-GT recon ] side by side.
The recon panel = the clip's canonical Gaussians moved by the EXACT pseudo-GT `traj` and rendered from
the clip camera (isotropic-ish => moving means is a faithful render). No model, no prediction — pure GT.
If real and recon match, the training target is faithful; if the object scatters or fails to rotate while
the real one does, the DATA is the problem (not the model). Directly tests whether rotation is captured.

Outputs per clip: an mp4 (playable) + a montage PNG (rows = REAL / GT-recon, cols = timesteps; inline-viewable).
Usage: python code/scripts/viz_traindata_verify.py <out_tag> <clip1.pt> [clip2.pt ...]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "code")
import imageio.v2 as iio  # noqa: E402
from igsw.gaussians import GaussianSet, render_gaussianset  # noqa: E402

try:
    import cv2  # noqa: E402
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

dev = "cuda"
OUT = "outputs/traindata_verify"
os.makedirs(OUT, exist_ok=True)
TAG = sys.argv[1]
CLIPS = sys.argv[2:]


def kabsch_deg(P, Q):
    """rotation angle (deg) of the SE(3) that best maps P->Q (numpy [n,3])."""
    if len(P) < 8:
        return float("nan")
    Pm, Qm = P.mean(0), Q.mean(0)
    H = (P - Pm).T @ (Q - Qm)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def label(img, text, color=(255, 255, 255)):
    if HAVE_CV2:
        cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return img


for ci, p in enumerate(CLIPS):
    c = torch.load(p, map_location=dev, weights_only=False)
    g0 = GaussianSet(c["means"], c["quats"], c["scales"], c["opacities"], c["colors"], None)
    traj = c["traj"].to(dev).float()
    K = int(c["Kf"])
    H, W = int(c["H"]), int(c["W"])
    Ki = c["K_intr"].to(dev).float()
    vm = c["viewmat"].to(dev).float()
    gt = c["gt_rgb"].to(dev).float() / 255.0
    is_obj = c["is_obj"].cpu().numpy() if "is_obj" in c else None
    instr = c.get("instruction", "")
    backend = c.get("backend", "?")

    # object-motion stats over the window (pseudo-GT)
    tn = traj.cpu().numpy()
    if is_obj is not None and is_obj.sum() >= 8:
        rot = kabsch_deg(tn[0][is_obj], tn[-1][is_obj])
        disp = float(np.linalg.norm(tn[-1][is_obj].mean(0) - tn[0][is_obj].mean(0)) * 100)
    else:
        rot, disp = float("nan"), float("nan")
    tagc = os.path.basename(p).replace(".pt", "")
    print(f"[{ci}] {tagc} | {backend} | rot={rot:.1f}deg disp={disp:.1f}cm | {instr[:60]!r}")

    oi = torch.from_numpy(is_obj).to(dev) if is_obj is not None else None
    reals, recons, recons_obj = [], [], []
    for t in range(K + 1):
        gm = GaussianSet(traj[t], g0.quats, g0.scales, g0.opacities, g0.colors, None)
        rec, alpha, _ = render_gaussianset(gm, vm[None], Ki[None], W, H)
        rec = (rec[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        real = (gt[t].cpu().numpy() * 255).astype(np.uint8)
        reals.append(real)
        recons.append(rec)
        if oi is not None and int(oi.sum()) >= 8:
            gmo = GaussianSet(traj[t][oi], g0.quats[oi], g0.scales[oi], g0.opacities[oi], g0.colors[oi], None)
            reco, _, _ = render_gaussianset(gmo, vm[None], Ki[None], W, H)
            recons_obj.append((reco[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))

    # mp4: [ real | recon ] side by side
    wv = iio.get_writer(f"{OUT}/{TAG}_{tagc}.mp4", fps=6)
    for r, rc in zip(reals, recons):
        wv.append_data(np.concatenate([r, rc], axis=1))
    wv.close()

    # montage PNG: rows = REAL / GT-recon, cols = sampled timesteps
    cols = sorted(set([0, K // 4, K // 2, (3 * K) // 4, K]))
    sc = 256
    def small(im):
        if HAVE_CV2:
            return cv2.resize(im, (sc, sc), interpolation=cv2.INTER_AREA)
        idx = (np.linspace(0, im.shape[0] - 1, sc)).astype(int)
        return im[idx][:, idx]
    row_real = np.concatenate([label(small(reals[t]).copy(), f"REAL t{t}", (90, 255, 90)) for t in cols], axis=1)
    row_rec = np.concatenate([label(small(recons[t]).copy(), f"full-recon t{t}", (120, 200, 255)) for t in cols], axis=1)
    rows = [row_real, row_rec]
    if recons_obj:
        row_obj = np.concatenate([label(small(recons_obj[t]).copy(), f"OBJ-only t{t}", (255, 210, 90)) for t in cols], axis=1)
        rows.append(row_obj)
    montage = np.concatenate(rows, axis=0)
    hdr = f"{tagc}  [{backend}]  rot={rot:.0f}deg disp={disp:.0f}cm  |  {instr[:70]}"
    if HAVE_CV2:
        bar = np.zeros((28, montage.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, hdr, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        montage = np.concatenate([bar, montage], axis=0)
    iio.imwrite(f"{OUT}/{TAG}_{tagc}_montage.png", montage)
    print(f"    -> {OUT}/{TAG}_{tagc}.mp4  +  _montage.png")

print("DONE")
