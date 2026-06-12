"""§90 R4 final: v15 mix = LIBERO sim + AgiBot real + libero_90 BOOK (WIN=96 vocab) +
libero_goal ROTATION (WIN=96, drawer/knob = large clean rotation, §89 data-lever).
Quality-gated; 4 held splits for the comprehensive final test:
  heldseed (sim guard) | heldreal (real guard) | held90 (unseen-noun vocab) | heldgoal (rotation 5deg5cm).
"""
import glob
import os

import torch

WS = "/mnt/pfs/public/xuhaoming/instruct_gs_world"
SIM = os.path.join(WS, "data/libero_pi3_v2")
REAL = os.path.join(WS, "data/_agibot")
OUT = os.path.join(WS, "data/mix_v15")


def gated(pattern, disp_lo=0.05, disp_hi=0.8, n_lo=500, n_hi=40000):
    out = []
    for f in sorted(glob.glob(os.path.join(REAL, pattern)),
                    key=lambda x: int(x.split("ep")[1].split(".")[0])):
        c = torch.load(f, map_location="cpu", weights_only=False)
        io = c["is_obj"]
        if int(io.sum()) == 0:
            continue
        med = float((c["traj"][-1][io] - c["traj"][0][io]).norm(dim=-1).median())
        if disp_lo <= med <= disp_hi and n_lo <= int(io.sum()) <= n_hi:
            out.append((int(f.split("ep")[1].split(".")[0]), f, med))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "*.pt")):
        os.remove(f)
    n = {"sim": 0}
    for f in sorted(glob.glob(os.path.join(SIM, "*.pt"))):
        os.symlink(f, os.path.join(OUT, os.path.basename(f)))
        n["sim"] += 1
    # source -> (glob, held-split-name, id-prefix); every 5th PASS clip -> its held split
    for tag, (pat, held, pfx) in {
        "real": ("clip_stv2_ep*.pt", "heldreal", "epi80"),
        "book": ("clip_stv2_lib90_ep*.pt", "held90", "epi81"),
        "goal": ("clip_stv2_goal_ep*.pt", "heldgoal", "epi82"),
    }.items():
        clips = gated(pat)
        tr = hd = 0
        for k, (ep, f, med) in enumerate(clips):
            split = held if (k % 5 == 4) else "train"
            os.symlink(f, os.path.join(OUT, f"{pfx}{ep:04d}_r_{split}.pt"))
            tr += split == "train"
            hd += split == held
        n[tag] = f"{len(clips)} -> train{tr}/{held}{hd}"
    print(f"[mix_v15] {n}")
    print(f"[mix_v15] train={len(glob.glob(os.path.join(OUT, '*_train.pt')))}")


if __name__ == "__main__":
    main()
