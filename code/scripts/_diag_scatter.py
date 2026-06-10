"""Quantify whether the PREDICTED object motion is COHERENT (rigid translation) or SCATTERING.
For the mover object's control points:
  extent = RMS distance of points from their own centroid (the object's spatial 'size')
  extent_ratio = extent(end) / extent(start)   -> 1.0 = rigid;  >>1 = Gaussians fly apart
  scatter      = RMS(per-point disp - mean disp) / ||mean disp||  -> 0 = rigid; ~1 = incoherent
Compares GT vs the model's prediction (correct instruction)."""
import sys, os, glob, numpy as np, torch
sys.path.insert(0, "code"); sys.path.insert(0, "code/scripts")
from scripts.eval_langswap import build_model, uniform_controls, run, INSTR, NOUNS

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/libero_v9lang/ckpt_last.pt"
DATA = sys.argv[2] if len(sys.argv) > 2 else "data/libero_pi3_v2"


def extent(X):                                   # RMS distance from centroid
    return float(np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean()))


def stats(p0, pK):
    disp = pK - p0
    md = disp.mean(0)
    scatter = float(np.sqrt(((disp - md) ** 2).sum(1).mean()) / (np.linalg.norm(md) + 1e-9))
    return extent(p0), extent(pK), extent(pK) / (extent(p0) + 1e-9), scatter


ck = torch.load(CKPT, map_location="cpu", weights_only=False)
mdl = build_model(ck)
clips = sorted(glob.glob(f"{DATA}/*_c_heldseed.pt"))[:4]
print(f"model={CKPT}")
print(f"{'clip':<26} {'GT ext0->extK (ratio)':<26} {'PRED ext0->extK (ratio)':<27} {'GT scat':<8} {'PRED scat'}")
for cp in clips:
    c = torch.load(cp, map_location="cuda", weights_only=False)
    seg = c["seg_per_g"].cuda().long(); N = len(seg); nkeep = N - int(c.get("n_fill", 0))
    K = int(c["Kf"]); tr = c["traj"].cuda().float()
    ci = uniform_controls(seg, nkeep, ck.get("M", 2048)); seg_c = seg[ci]
    gt_disp = (tr[K] - tr[0]).norm(dim=-1)[ci]
    obj_es = [e for e in torch.unique(seg_c).tolist() if 1 <= e <= 7]
    mv_e = max(obj_es, key=lambda e: float(gt_disp[seg_c == e].mean()))
    mv = (seg_c == mv_e).cpu().numpy()
    instr = c["instruction"]
    gt0 = tr[0][ci].cpu().numpy()[mv]; gtK = tr[K][ci].cpu().numpy()[mv]
    epT, init = run(mdl, c, ci, seg, instr, K)
    init = init.cpu().numpy()[mv]; epT = epT.cpu().numpy()[mv]
    g0e, gKe, gr, gs = stats(gt0, gtK)
    p0e, pKe, pr, ps = stats(init, epT)
    print(f"{os.path.basename(cp):<26} {g0e*100:5.1f}->{gKe*100:5.1f}cm ({gr:4.2f})       "
          f"{p0e*100:5.1f}->{pKe*100:5.1f}cm ({pr:4.2f})        {gs:5.2f}    {ps:5.2f}")
