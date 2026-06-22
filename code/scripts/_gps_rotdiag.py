"""Rotation-failure diagnostic (PURE GT geometry, no model). For each held mover entity, decompose
the GT token motion into translation (centroid shift) + rotation (Kabsch about centroid) + non-rigid
residual, and report what FRACTION of total motion is rotational. Answers the strategic question:
does rotation-via-translation fail because the rotational SIGNAL is weak (data/translation-dominated)
or because the model can't learn an available signal? Usage: python _gps_rotdiag.py data/mix_v15 held
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.gpstoken_wm.losses import kabsch_R  # noqa: E402


def rot_deg(R):
    c = (R.diagonal().sum() - 1) / 2
    return float(torch.arccos(torch.clamp(c, -1, 1)) * 180 / np.pi)


data = sys.argv[1] if len(sys.argv) > 1 else "data/mix_v15"
split = sys.argv[2] if len(sys.argv) > 2 else "held"
clips = sorted(glob.glob(f"{data}/*{split}*.pt"))
rows = []
for cp in clips:
    c = torch.load(cp, map_location="cpu", weights_only=False)
    traj = c["traj"].float()
    K = int(c["Kf"])
    if "seg_per_g" not in c:
        continue
    seg = c["seg_per_g"].long()
    x0all, x1all = traj[0], traj[K]
    disp = (x1all - x0all).norm(dim=-1)
    for s in torch.unique(seg).tolist():
        if s == 0:
            continue
        sel = (seg == s) & (disp > 0.01)
        if sel.sum() < 6:
            continue
        x0, x1 = x0all[sel], x1all[sel]
        c0 = x0.mean(0)
        t = x1.mean(0) - c0                      # centroid translation
        R = kabsch_R(x0, x1)                      # best-fit rotation about centroid
        rotvec = (R @ (x0 - c0).T).T - (x0 - c0)  # per-point displacement caused PURELY by rotation
        rigid = (R @ (x0 - c0).T).T + c0 + t
        rows.append((
            rot_deg(R),                           # GT rotation angle (deg)
            t.norm().item(),                      # centroid translation magnitude
            rotvec.norm(dim=-1).mean().item(),    # mean rotational displacement
            (x1 - rigid).norm(dim=-1).mean().item(),  # non-rigid residual
            (x1 - x0).norm(dim=-1).mean().item(),     # total motion
            (x0 - c0).norm(dim=-1).mean().item(),     # object radius
            int(sel.sum()),
        ))

A = np.array(rows)
ang, tr, rd, rs, tot, rad, npts = A.T
fr_rot = rd / np.clip(tot, 1e-6, None)
fr_res = rs / np.clip(tot, 1e-6, None)


def pct(x, th):
    return 100 * np.mean(x > th)


print(f"n_clips={len(clips)}  n_entities={len(A)}  (held movers, >=6 pts moving >1cm)")
print(f"GT rotation angle: median {np.median(ang):.1f}deg  p75 {np.percentile(ang,75):.1f}deg  "
      f">10deg:{pct(ang,10):.0f}%  >20deg:{pct(ang,20):.0f}%")
print(f"object radius median {np.median(rad)*100:.1f}cm")
print(f"motion(cm): total {np.median(tot)*100:.1f}  | trans {np.median(tr)*100:.1f}  "
      f"rotdisp {np.median(rd)*100:.2f}  nonrigid {np.median(rs)*100:.2f}")
print(f"** rotation share of motion (rotdisp/total): median {np.median(fr_rot)*100:.0f}%  "
      f"p75 {np.percentile(fr_rot,75)*100:.0f}% **")
print(f"   nonrigid share (resid/total): median {np.median(fr_res)*100:.0f}%")
hi = ang > 15
if hi.sum():
    print(f"--- HIGH-rot subset (GT>15deg, n={int(hi.sum())}) ---")
    print(f"   rotdisp median {np.median(rd[hi])*100:.2f}cm  total {np.median(tot[hi])*100:.1f}cm  "
          f"rotation share median {np.median(fr_rot[hi])*100:.0f}%")
# 5deg5cm context: rotation that the readout demands the model recover
print(f"NOTE: 5deg5cm readout asks model to recover rotation; median rotdisp={np.median(rd)*100:.2f}cm "
      f"vs EPE~6.5cm noise floor -> {'SIGNAL < NOISE' if np.median(rd)*100 < 6.5 else 'signal above noise'}")
