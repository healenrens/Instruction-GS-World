"""Quantify the pseudo-GT MOTION quality of the (real-video) clips the model learns from — is the weak
real-video dcos a model problem or a data ceiling? Per clip: object motion magnitude, DIRECTION
COHERENCE (do the object's tokens move consistently, or is the GT a noisy cloud?), and RIGID RESIDUAL
(is the GT object motion rigid, or does it deform/scatter?). Low coherence + high residual = noisy GT
= the model can't do better than the supervision. Usage: python _gps_real_gtqual.py data/mix_v15 heldreal"""
import glob
import os
import sys

import numpy as np
import torch


def kabsch_R(P, Q):
    Pm, Qm = P.mean(0), Q.mean(0)
    H = (P - Pm).T @ (Q - Qm)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


data, split = sys.argv[1], sys.argv[2]
cohs, rigs, disps = [], [], []
for cp in sorted(glob.glob(f"{data}/*_{split}.pt")):
    c = torch.load(cp, map_location="cpu", weights_only=False)
    means = c["means"].numpy()
    traj = c["traj"].numpy()
    K = int(c["Kf"])
    is_obj = c["is_obj"].numpy().astype(bool) if "is_obj" in c else None
    if is_obj is None or is_obj.sum() < 6:
        continue
    P0, PK = traj[0][is_obj], traj[K][is_obj]
    d = PK - P0                                              # per-object-token displacement
    disp_med = float(np.linalg.norm(d, axis=-1).mean(0)) * 100   # NOTE: magnitude of MEAN object motion
    mover = np.linalg.norm(d, axis=-1) > 0.01
    if mover.sum() < 6:
        continue
    dm = d[mover]
    meandir = dm.mean(0)
    meandir = meandir / max(np.linalg.norm(meandir), 1e-6)
    unit = dm / np.clip(np.linalg.norm(dm, axis=-1, keepdims=True), 1e-6, None)
    coh = float((unit @ meandir).mean())                    # direction coherence in [-1,1]
    R = kabsch_R(P0[mover], PK[mover])
    pred = (P0[mover] - P0[mover].mean(0)) @ R.T + PK[mover].mean(0)
    rigres = float(np.median(np.linalg.norm(pred - PK[mover], axis=-1))) * 100   # cm, rigid-fit residual
    gtmag = float(np.median(np.linalg.norm(dm, axis=-1))) * 100
    cohs.append(coh); rigs.append(rigres); disps.append(gtmag)
    print(f"{os.path.basename(cp):34s} obj_g={int(is_obj.sum()):5d} GTdisp_med={gtmag:5.1f}cm "
          f"dir_COHERENCE={coh:+.2f}  rigid_residual={rigres:5.1f}cm  {c.get('instruction','')[:32]!r}")

if cohs:
    print(f"\n[{split}] medians: dir-coherence {np.median(cohs):+.2f}  rigid-residual {np.median(rigs):.1f}cm  "
          f"GTdisp {np.median(disps):.1f}cm  (n={len(cohs)})")
    print("  read: coherence~1 + residual~0 = clean rigid GT (learnable); coherence low + residual high = noisy GT (data ceiling)")
