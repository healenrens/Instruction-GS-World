"""Summarize the STAGE-1 RoboTwin2 datasets: split sizes, task coverage, mover stats for the backbone
flow-GT clips, and the action chunk shape. Run after production.

Usage: rt2_summary.py --win data/rt2_win --act data/rt2_act
"""
import argparse
import glob
import os
from collections import Counter, defaultdict

import numpy as np
import torch


def split_of(path):
    b = os.path.basename(path)
    for s in ("heldtask", "heldseed", "train"):
        if b.endswith(f"_{s}.pt"):
            return s
    return "?"


def task_of(path):
    b = os.path.basename(path).rsplit("_ep", 1)[0]
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", default="data/rt2_win")
    ap.add_argument("--act", default="data/rt2_act")
    args = ap.parse_args()

    print("=" * 60)
    print("BACKBONE flow-GT clips (data/rt2_win, SpaTracker):")
    wf = [f for f in glob.glob(f"{args.win}/*.pt") if "_plan" not in f and "_select" not in f]
    bs = Counter(split_of(f) for f in wf)
    bt = defaultdict(Counter)
    for f in wf:
        bt[task_of(f)][split_of(f)] += 1
    print(f"  total clips = {len(wf)}   " + "  ".join(f"{k}={bs[k]}" for k in ("train", "heldseed", "heldtask")))
    print(f"  tasks covered = {len(bt)}")
    # mover stats over a sample
    samp = wf[:: max(1, len(wf) // 80)]
    movs, Ns = [], []
    for f in samp:
        c = torch.load(f, map_location="cpu", weights_only=False)
        tr = c["traj"]; K = int(c["Kf"])
        disp = (tr[K] - tr[0]).norm(dim=-1)
        movs.append(int((disp > 0.01).sum())); Ns.append(tr.shape[1])
    if movs:
        print(f"  per-clip movers(>.01): median {int(np.median(movs))}  p25 {int(np.percentile(movs,25))}  p75 {int(np.percentile(movs,75))}  (sample {len(samp)})")
        print(f"  per-clip N tokens: median {int(np.median(Ns))}")

    print("=" * 60)
    print("ACTION dataset (data/rt2_act):")
    af = [f for f in glob.glob(f"{args.act}/*.pt") if "norm_stats" not in f]
    asp = Counter(split_of(f) for f in af)
    at = set(task_of(f) for f in af)
    print(f"  total windows = {len(af)}   " + "  ".join(f"{k}={asp[k]}" for k in ("train", "heldseed", "heldtask")))
    print(f"  tasks covered = {len(at)}")
    c0 = torch.load(af[0], map_location="cpu", weights_only=False)
    print(f"  per-window: dq {tuple(c0['dq'].shape)}  anchor {tuple(c0['anchor'].shape)}  frame0 {tuple(c0['frame0'].shape)}  control_hz {c0['control_hz']:.2f}")
    ns = os.path.join(args.act, "norm_stats.pt")
    if os.path.exists(ns):
        st = torch.load(ns, weights_only=False)
        print(f"  norm_stats.pt: {st['n_windows']} train windows, {st['n_steps']} steps")


if __name__ == "__main__":
    main()
