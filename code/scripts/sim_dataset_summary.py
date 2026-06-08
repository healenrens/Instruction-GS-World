"""Summarize the generated sim dataset for the report: per-task/per-split counts + the val_psnr
and movefrac distributions. Run after generation.
  python code/scripts/sim_dataset_summary.py --data data/maniskill
"""
import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/pfs/public/xuhaoming/instruct_gs_world/data/maniskill")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.data, "*.pt")))
    by_task = defaultdict(int); by_split = defaultdict(int); by_ts = defaultdict(int)
    vpsnr = []; movefrac = []; nclips = 0
    for p in paths:
        try:
            c = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        env = c.get("env", "?"); split = c.get("split", "train")
        by_task[env] += 1; by_split[split] += 1; by_ts[(env, split)] += 1
        vp = c.get("val_psnr")
        if vp is not None and len(vp) > 1:
            vpsnr.append(float(np.mean(vp[1:])))
        if "movefrac" in c:
            movefrac.append(float(c["movefrac"]))
        nclips += 1
    print(f"=== sim dataset: {nclips} clips under {args.data} ===")
    print("by task:", dict(by_task))
    print("by split:", dict(by_split))
    print("by (task,split):", {f"{k[0]}/{k[1]}": v for k, v in sorted(by_ts.items())})
    if vpsnr:
        v = np.array(vpsnr)
        print(f"val_psnr mean(t>=1): n={len(v)} mean={v.mean():.1f} median={np.median(v):.1f} "
              f"min={v.min():.1f} max={v.max():.1f} p10={np.percentile(v,10):.1f}")
    if movefrac:
        m = np.array(movefrac)
        print(f"movefrac(>0.02r): mean={m.mean():.3f} median={np.median(m):.3f} min={m.min():.3f} max={m.max():.3f}")


if __name__ == "__main__":
    main()
