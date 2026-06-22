"""Compute per-dim Δqpos mean/std over the TRAIN split of the action dataset -> norm_stats.pt.

The 14 dims = [L_arm0..5, L_gripper, R_arm0..5, R_gripper]. Grippers (dims 6,13) are near-binary in
RoboTwin (open=1/closed=0), so most steps have dq=0 with rare large jumps -> report them explicitly.

Usage: rt2_norm_stats.py --data data/rt2_act --out data/rt2_act/norm_stats.pt
"""
import argparse
import glob
import os

import numpy as np
import torch

DIM_NAMES = ([f"L_arm{i}" for i in range(6)] + ["L_grip"] +
             [f"R_arm{i}" for i in range(6)] + ["R_grip"])
GRIPPER_DIMS = [6, 13]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = args.out or os.path.join(args.data, "norm_stats.pt")
    files = sorted(glob.glob(f"{args.data}/*_train.pt"))
    assert files, f"no *_train.pt in {args.data}"
    n = 0
    s1 = np.zeros(14, dtype=np.float64)
    s2 = np.zeros(14, dtype=np.float64)
    amax = np.zeros(14, dtype=np.float64)
    nz = np.zeros(14, dtype=np.float64)        # nonzero-step count per dim (for gripper sparsity)
    for f in files:
        dq = torch.load(f, map_location="cpu", weights_only=False)["dq"].numpy().astype(np.float64)  # [WIN,14]
        s1 += dq.sum(0); s2 += (dq * dq).sum(0)
        amax = np.maximum(amax, np.abs(dq).max(0))
        nz += (np.abs(dq) > 1e-6).sum(0)
        n += dq.shape[0]
    mean = s1 / n
    var = np.maximum(s2 / n - mean ** 2, 0.0)
    std = np.sqrt(var)
    std_safe = np.where(std < 1e-6, 1.0, std)   # avoid /0 for dead dims at train/inference time
    stats = {
        "mean": torch.from_numpy(mean.astype(np.float32)),
        "std": torch.from_numpy(std.astype(np.float32)),
        "std_safe": torch.from_numpy(std_safe.astype(np.float32)),
        "abs_max": torch.from_numpy(amax.astype(np.float32)),
        "nonzero_frac": torch.from_numpy((nz / n).astype(np.float32)),
        "n_steps": int(n), "n_windows": len(files), "dim_names": DIM_NAMES,
        "gripper_dims": GRIPPER_DIMS,
    }
    torch.save(stats, out)
    print(f"[norm] {len(files)} train windows, {n} steps -> {out}")
    print(f"{'dim':<8} {'mean':>10} {'std':>10} {'abs_max':>10} {'nz_frac':>8}")
    for i, name in enumerate(DIM_NAMES):
        tag = "  <-grip" if i in GRIPPER_DIMS else ""
        print(f"{name:<8} {mean[i]:>10.5f} {std[i]:>10.5f} {amax[i]:>10.4f} {nz[i]/n:>8.4f}{tag}")


if __name__ == "__main__":
    main()
