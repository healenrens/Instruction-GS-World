"""§84 R4 entry: assemble the v12 SIM+REAL co-training mix.
- Audits all AgiBot StV2 clips with the quality gate (target disp 5-80 StV2-cm, 500-30k obj gaussians).
- PASS clips -> epi9{ep:05d}_r_{train|heldreal}.pt names in data/mix_v12 (every 5th PASS -> heldreal).
- Symlinks ALL LIBERO sim clips (train/heldtask/heldseed splits preserved).
Run: .venv/bin/python code/scripts/prep_mix_v12.py
"""
import glob
import os

import torch

WS = "/mnt/pfs/public/xuhaoming/instruct_gs_world"
SIM = os.path.join(WS, "data/libero_pi3_v2")
REAL = os.path.join(WS, "data/_agibot")
OUT = os.path.join(WS, "data/mix_v12")


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "*.pt")):
        os.remove(f)
    n_sim = 0
    for f in sorted(glob.glob(os.path.join(SIM, "*.pt"))):
        os.symlink(f, os.path.join(OUT, os.path.basename(f)))
        n_sim += 1
    passing = []
    for f in sorted(glob.glob(os.path.join(REAL, "clip_stv2_ep*.pt")),
                    key=lambda x: int(x.split("ep")[1].split(".")[0])):
        c = torch.load(f, map_location="cpu", weights_only=False)
        ep = int(f.split("ep")[1].split(".")[0])
        io = c["is_obj"]
        d = (c["traj"][-1][io] - c["traj"][0][io]).norm(dim=-1)
        med = float(d.median()) if int(io.sum()) else 0.0
        if (0.05 <= med <= 0.8) and (500 <= int(io.sum()) <= 30000):
            passing.append((ep, f, med))
    n_tr = n_hr = 0
    for k, (ep, f, med) in enumerate(passing):
        split = "heldreal" if (k % 5 == 4) else "train"
        dst = os.path.join(OUT, f"epi9{ep:05d}_r_{split}.pt")
        os.symlink(f, dst)
        if split == "train":
            n_tr += 1
        else:
            n_hr += 1
    print(f"[mix_v12] sim={n_sim} | real PASS={len(passing)} -> train={n_tr} heldreal={n_hr}")
    for ep, _, med in passing:
        print(f"  real ep{ep}: disp={med*100:.1f}")


if __name__ == "__main__":
    main()
