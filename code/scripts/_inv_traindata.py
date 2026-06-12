"""Inventory current training data: counts per split, train clips by backend, heldgoal instructions,
and a suggested representative set for fidelity rendering. Print-only."""
import glob
import os

import numpy as np
import torch


def kdeg(P, Q):
    if len(P) < 8:
        return float("nan")
    Pm, Qm = P.mean(0), Q.mean(0)
    H = (P - Pm).T @ (Q - Qm)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


D = "data/mix_v15"
print("=== mix_v15 split counts ===")
for s in ["train", "heldseed", "heldreal", "held90", "heldgoal"]:
    print(f"  {s}: {len(glob.glob(f'{D}/*_{s}.pt'))}")

print("=== train clips by backend ===")
cats = {}
for f in sorted(glob.glob(f"{D}/*_train.pt")):
    c = torch.load(f, map_location="cpu", weights_only=False)
    b = c.get("backend", "?")
    io = c["is_obj"].numpy()
    tn = c["traj"].numpy()
    rot = kdeg(tn[0][io], tn[-1][io]) if io.sum() >= 8 else float("nan")
    disp = float(np.linalg.norm(tn[-1][io].mean(0) - tn[0][io].mean(0)) * 100) if io.sum() >= 8 else float("nan")
    cats.setdefault(b, []).append((os.path.basename(f), c.get("instruction", "")[:42], int(io.sum()), rot, disp))
for k, v in cats.items():
    print(f"--- backend={k}: {len(v)} clips ---")
    for fn, ins, no, rot, disp in v[:6]:
        print(f"    {fn} | obj_g={no} rot={rot:.0f} disp={disp:.0f}cm | {ins!r}")

print("=== heldgoal (rotation eval) clips ===")
for f in sorted(glob.glob(f"{D}/*_heldgoal.pt")):
    c = torch.load(f, map_location="cpu", weights_only=False)
    io = c["is_obj"].numpy()
    tn = c["traj"].numpy()
    rot = kdeg(tn[0][io], tn[-1][io]) if io.sum() >= 8 else float("nan")
    print(f"  {os.path.basename(f)} | {c.get('backend','?')} rot={rot:.0f} | {c.get('instruction','')[:50]!r}")

print("=== turn rotation clips (data/_agibot) ===")
tc = sorted(glob.glob("data/_agibot/clip_stv2_turn_ep*.pt"))
print(f"  count={len(tc)}")
for f in tc[:8]:
    c = torch.load(f, map_location="cpu", weights_only=False)
    io = c["is_obj"].numpy()
    tn = c["traj"].numpy()
    rot = kdeg(tn[0][io], tn[-1][io]) if io.sum() >= 8 else float("nan")
    print(f"  {os.path.basename(f)} | rot={rot:.0f} obj_g={int(io.sum())} | {c.get('instruction','')[:45]!r}")
