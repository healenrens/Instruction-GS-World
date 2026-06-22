"""Build the CHEAP action dataset (no SpaTracker) for ALL windows in the shared plan.

Per window store a compact .pt:
  frame0     : uint8 [H,W,3]   head-cam rgb at step s
  instruction: str             one 'seen' phrasing for the episode (random among seen, seeded by name)
  dq         : float32 [WIN,14] per-step delta-qpos, dq[i] = vector[s+i+1] - vector[s+i], i=0..WIN-1
  anchor     : float32 [14]    absolute joint_action vector[s] (the window's starting qpos)
  K_intr     : float32 [3,3]   head-cam intrinsics at step s (static cam -> constant)
  task, ep, s, split, control_hz

This is what the later DiT trains on (frame0 + instruction -> Δqpos chunk). frame0 here is the SAME step s
the backbone clip's frame0 spans, so the two datasets are window-aligned.

Usage: rt2_build_action.py --plan data/rt2_win/window_plan.json --out data/rt2_act \
         [--shard 0 --nshard 4]  (sharding for parallelism)
"""
import argparse
import io
import json
import os
import random

import h5py
import numpy as np
import torch
from PIL import Image

CONTROL_HZ = 250.0 / 15.0


def pick_seen_instruction(json_path, seed_name):
    j = json.load(open(json_path))
    seen = j.get("seen") or j.get("unseen") or []
    if not seen:
        return ""
    rng = random.Random(seed_name)
    return rng.choice(seen)


def build_one(w, out_dir, overwrite=False):
    out = os.path.join(out_dir, f"{w['name']}_{w['split']}.pt")
    if os.path.exists(out) and not overwrite:
        return "skip"
    s, win = int(w["s"]), int(w["win"])
    with h5py.File(w["hdf5"], "r") as h:
        vec = np.asarray(h["joint_action/vector"], dtype=np.float64)         # [T,14]
        rgb_ds = h["observation/head_camera/rgb"]
        frame0 = np.array(Image.open(io.BytesIO(bytes(rgb_ds[s]))).convert("RGB")).astype(np.uint8)
        K = np.asarray(h["observation/head_camera/intrinsic_cv"][s], dtype=np.float32)  # [3,3]
    seg = vec[s:s + win + 1]                                                 # [WIN+1,14]
    dq = (seg[1:] - seg[:-1]).astype(np.float32)                            # [WIN,14] per-step deltas
    anchor = seg[0].astype(np.float32)                                       # [14]
    instr = pick_seen_instruction(w["json"], w["name"])
    clip = {
        "frame0": torch.from_numpy(frame0),
        "instruction": instr,
        "dq": torch.from_numpy(dq),
        "anchor": torch.from_numpy(anchor),
        "K_intr": torch.from_numpy(K),
        "task": w["task"], "ep": int(w["ep"]), "s": s, "win": win,
        "split": w["split"], "control_hz": float(CONTROL_HZ),
    }
    os.makedirs(out_dir, exist_ok=True)
    torch.save(clip, out)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="for quick verification: only first N windows")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    plan = json.load(open(args.plan))
    if args.limit:
        plan = plan[:args.limit]
    plan = plan[args.shard::args.nshard]
    ok = sk = err = 0
    for i, w in enumerate(plan):
        try:
            r = build_one(w, args.out, args.overwrite)
            ok += (r == "ok"); sk += (r == "skip")
        except Exception as e:
            err += 1
            print(f"[act] ERR {w['name']}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 200 == 0:
            print(f"[act] shard{args.shard}: {i+1}/{len(plan)} ok={ok} skip={sk} err={err}", flush=True)
    print(f"[act] shard{args.shard} DONE: ok={ok} skip={sk} err={err}", flush=True)


if __name__ == "__main__":
    main()
