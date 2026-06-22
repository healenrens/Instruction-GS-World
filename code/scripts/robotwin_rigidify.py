"""Path 2 (SYNTHESIZE POSES) — denoise Path-1 per-point video GT into per-OBJECT rigid motion.

Operates on the already-saved Path-1 clips (no Pi3/CoTracker re-run): for each clip's per-point track
trajectory, motion-cluster the movers, fit a per-frame TRIMMED-KABSCH rigid transform per cluster, and
replace each cluster's trajectory with its rigid transform (averages out per-point CoTracker/depth noise).
Static background -> frozen at frame0 (static camera). This is "synthesize the object pose from observation
via another model" (the tracker), the cleaner-GT path.

Usage: robotwin_rigidify.py --indir data/rtvid_v1 --outdir data/rtvid_v1_rigid [--mover_pct 0.35 --k 3]
"""
import argparse, glob, os
import numpy as np, torch


def kabsch(A, B):
    """Rigid fit R,t with R@A.T + t ~= B. A,B [M,3]."""
    cA, cB = A.mean(0), B.mean(0)
    H = (A - cA).T @ (B - cB)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cB - R @ cA


def rigid_traj(traj, mover_pct=0.35, k=3, trim=0.25, min_cluster=8):
    """traj [T,N,3] raw per-point tracks -> denoised per-object rigid trajectory.
    Movers (top mover_pct by total displacement) are clustered by motion + rigid-fit per frame; the rest
    (background) is frozen at frame0 (static camera)."""
    T, N, _ = traj.shape
    out = np.tile(traj[0][None], (T, 1, 1)).astype(np.float32)        # default: static at frame0
    disp = np.linalg.norm(traj[-1] - traj[0], axis=1)
    thr = np.quantile(disp, 1.0 - mover_pct)
    mov = np.where(disp > max(thr, 1e-9))[0]
    if len(mov) < min_cluster:
        return traj.astype(np.float32)                                # no clear movers -> keep raw
    feat = np.concatenate([traj[-1, mov] - traj[0, mov], traj[T // 2, mov] - traj[0, mov]], 1)  # [M,6]
    try:
        from sklearn.cluster import KMeans
        kk = int(min(k, max(1, len(mov) // 30)))
        lab = KMeans(kk, n_init=4, random_state=0).fit_predict(feat) if kk > 1 else np.zeros(len(mov), int)
    except Exception:
        lab = np.zeros(len(mov), int)
    for c in np.unique(lab):
        ci = mov[lab == c]
        if len(ci) < min_cluster:
            out[:, ci] = traj[:, ci]                                  # too small to fit -> keep raw
            continue
        A = traj[0, ci]
        for t in range(1, T):
            B = traj[t, ci]
            R, tt = kabsch(A, B)
            res = np.linalg.norm((A @ R.T + tt) - B, axis=1)
            keep = res <= np.quantile(res, 1.0 - trim)
            if keep.sum() >= 6:
                R, tt = kabsch(A[keep], B[keep])
            out[t, ci] = A @ R.T + tt
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True); ap.add_argument("--outdir", required=True)
    ap.add_argument("--mover_pct", type=float, default=0.35); ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    files = sorted(glob.glob(f"{args.indir}/*.pt"))
    n = 0
    for f in files:
        c = torch.load(f, weights_only=False)
        traj = c["traj"].numpy().astype(np.float32)
        rj = rigid_traj(traj, mover_pct=args.mover_pct, k=args.k)
        # report denoise: per-mover-point track jitter vs rigid (residual of raw to rigid)
        c["traj"] = torch.from_numpy(rj).float()
        c["backend"] = "pi3_robotwin_p2_rigid"
        torch.save(c, os.path.join(args.outdir, os.path.basename(f)))
        n += 1
    print(f"[rigidify] {n} clips -> {args.outdir} (mover_pct={args.mover_pct} k={args.k})", flush=True)


if __name__ == "__main__":
    main()
