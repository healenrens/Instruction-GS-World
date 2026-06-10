"""READ-ONLY survey for the rigid-motion redesign (no model changes).
Across ALL heldseed + first-N train clips, per mover-entity:
  [data]  GT rigid residual (Kabsch fit of GT motion; expect ~0 -> GT is rigid by construction)
          controls-per-entity (Kabsch feasibility; need >=4)
  [model] pred extent-ratio (scatter), pred Kabsch residual,
          dir-cos raw vs dir-cos after rigid projection (does projection preserve direction?)
"""
import sys, os, glob, numpy as np, torch
sys.path.insert(0, "code"); sys.path.insert(0, "code/scripts")
from scripts.eval_langswap import build_model, uniform_controls, run

CKPT = "checkpoints/libero_v9lang/ckpt_last.pt"; DATA = "data/libero_pi3_v2"


def kabsch_fit(P, Q):
    Pm, Qm = P.mean(0), Q.mean(0); Pc, Qc = P - Pm, Q - Qm
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    Qf = (R @ P.T).T + (Qm - R @ Pm)
    return Qf, float(np.linalg.norm(Q - Qf, axis=1).mean())


def ext(X):
    return float(np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean()))


def dc(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


ck = torch.load(CKPT, map_location="cpu", weights_only=False); mdl = build_model(ck)
clips = sorted(glob.glob(f"{DATA}/*_heldseed.pt")) + sorted(glob.glob(f"{DATA}/*_train.pt"))[:8]
rows = []
for cp in clips:
    c = torch.load(cp, map_location="cuda", weights_only=False)
    seg = c["seg_per_g"].cuda().long(); N = len(seg); nkeep = N - int(c.get("n_fill", 0))
    K = int(c["Kf"]); tr = c["traj"].cuda().float()
    ci = uniform_controls(seg, nkeep, ck.get("M", 2048)); seg_c = seg[ci]
    gt_disp = (tr[K] - tr[0]).norm(dim=-1)[ci]
    obj_es = [e for e in torch.unique(seg_c).tolist() if 1 <= e <= 7]
    if not obj_es:
        continue
    mv_e = max(obj_es, key=lambda e: float(gt_disp[seg_c == e].mean()))
    mv = (seg_c == mv_e).cpu().numpy()
    npts = int(mv.sum())
    gt0 = tr[0][ci].cpu().numpy()[mv]; gtK = tr[K][ci].cpu().numpy()[mv]
    _, gt_res = kabsch_fit(gt0, gtK)                       # GT rigidity (expect ~0)
    epT, init = run(mdl, c, ci, seg, c["instruction"], K)
    init = init.cpu().numpy()[mv]; epT = epT.cpu().numpy()[mv]
    rig, pred_res = kabsch_fit(init, epT)
    gd = gtK.mean(0) - gt0.mean(0)
    dir_raw = dc(epT.mean(0) - init.mean(0), gd)
    dir_rig = dc(rig.mean(0) - init.mean(0), gd)
    rows.append((os.path.basename(cp), npts, gt_res * 100, ext(epT) / ext(init),
                 pred_res * 100, dir_raw, dir_rig))
    print(f"{os.path.basename(cp):<26} pts={npts:<4} GTres={gt_res*100:5.2f}cm  "
          f"predExt x{ext(epT)/ext(init):4.2f}  predRes={pred_res*100:5.2f}cm  "
          f"dir {dir_raw:+.2f} -> rigid {dir_rig:+.2f}")

a = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows], float)
print(f"\nSUMMARY over {len(rows)} clips (mover entity):")
print(f"  controls/entity      min={a[:,0].min():.0f}  median={np.median(a[:,0]):.0f}")
print(f"  GT rigid residual    median={np.median(a[:,1]):.2f}cm  max={a[:,1].max():.2f}cm  (rigid GT => ~0)")
print(f"  PRED extent-ratio    median={np.median(a[:,2]):.2f}  >1.2 in {(a[:,2]>1.2).sum()}/{len(rows)}  max={a[:,2].max():.2f}")
print(f"  PRED rigid residual  median={np.median(a[:,3]):.2f}cm  max={a[:,3].max():.2f}cm")
print(f"  dir-cos raw->rigid   {np.median(a[:,4]):+.2f} -> {np.median(a[:,5]):+.2f}  (projection must NOT hurt direction)")
