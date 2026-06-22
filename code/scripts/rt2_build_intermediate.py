"""Build the lightweight SpaTracker-INPUT intermediates for the backbone subset.

robotwin_spatrack_clip.py --pt_glob consumes clip .pt files via load_pt_frames(): it only reads
c["gt_rgb"] (a [Kf+1,H,W,3] uint8 stack) and c["instruction"]. So for each chosen backbone WINDOW we
emit exactly that: KF+1 head-cam frames EVENLY spanning [s, s+win] of the episode + the instruction.
SpaTracker then turns it into the full trainer clip (means/uv/traj/.../vis) written to data/rt2_win.

We choose a TRACTABLE SUBSET (target ~1500 clips) balanced across tasks: take up to --per_task windows
per (trainable) task from the shared plan, drawing from train+heldseed (and the heldtask tasks too, so
the backbone has heldtask flow-GT to eval on). Same windows the action set covers (subset of the plan).

Usage: rt2_build_intermediate.py --plan data/rt2_win/window_plan.json --out data/rt2_win_src \
         --per_task 40 [--shard 0 --nshard 4]
"""
import argparse
import io
import json
import os
import random
from collections import defaultdict

import h5py
import numpy as np
import torch
from PIL import Image


def pick_seen_instruction(json_path, seed_name):
    j = json.load(open(json_path))
    seen = j.get("seen") or j.get("unseen") or []
    if not seen:
        return ""
    return random.Random(seed_name).choice(seen)


def select_subset(plan, per_task, seed=0):
    """Up to per_task windows per task, sampled deterministically. Keep split balance per task by sampling
    within each split proportionally; prioritize train then heldseed then heldtask."""
    by_task = defaultdict(list)
    for w in plan:
        by_task[w["task"]].append(w)
    rng = random.Random(seed)
    chosen = []
    for task in sorted(by_task):
        ws = by_task[task]
        # group by split
        by_split = defaultdict(list)
        for w in ws:
            by_split[w["split"]].append(w)
        for v in by_split.values():
            rng.shuffle(v)
        if "heldtask" in by_split:               # heldtask task: just take per_task from it
            chosen += by_split["heldtask"][:per_task]
            continue
        # trainable task: aim ~85% train, ~15% heldseed of the per_task budget
        n_held = max(1, int(round(per_task * 0.15))) if by_split.get("heldseed") else 0
        n_train = per_task - n_held
        chosen += by_split.get("train", [])[:n_train]
        chosen += by_split.get("heldseed", [])[:n_held]
    return chosen


def build_one(w, out_dir, overwrite=False):
    out = os.path.join(out_dir, f"{w['name']}_{w['split']}.pt")
    if os.path.exists(out) and not overwrite:
        return "skip"
    s, win, kf = int(w["s"]), int(w["win"]), int(w["kf"])
    idx = np.linspace(s, s + win, kf + 1).astype(int)                       # KF+1 frames spanning the window
    with h5py.File(w["hdf5"], "r") as h:
        rgb_ds = h["observation/head_camera/rgb"]
        frames = np.stack([np.array(Image.open(io.BytesIO(bytes(rgb_ds[t]))).convert("RGB"))
                           for t in idx]).astype(np.uint8)                   # [KF+1,H,W,3]
    instr = pick_seen_instruction(w["json"], w["name"])
    clip = {"gt_rgb": torch.from_numpy(frames), "instruction": instr,
            "task": w["task"], "ep": int(w["ep"]), "s": s, "win": win, "split": w["split"]}
    os.makedirs(out_dir, exist_ok=True)
    torch.save(clip, out)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_task", type=int, default=40)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dump_select", default="", help="write the chosen subset plan json here (rank0 only)")
    args = ap.parse_args()
    plan = json.load(open(args.plan))
    subset = select_subset(plan, args.per_task)
    if args.dump_select and args.shard == 0:
        os.makedirs(os.path.dirname(args.dump_select) or ".", exist_ok=True)
        json.dump(subset, open(args.dump_select, "w"))
        from collections import Counter
        print(f"[src] subset {len(subset)} windows  " +
              "  ".join(f"{k}={v}" for k, v in Counter(w['split'] for w in subset).items()), flush=True)
    sub = subset[args.shard::args.nshard]
    ok = sk = err = 0
    for i, w in enumerate(sub):
        try:
            r = build_one(w, args.out, args.overwrite)
            ok += (r == "ok"); sk += (r == "skip")
        except Exception as e:
            err += 1
            print(f"[src] ERR {w['name']}: {type(e).__name__}: {e}", flush=True)
    print(f"[src] shard{args.shard} DONE: ok={ok} skip={sk} err={err} (of {len(sub)})", flush=True)


if __name__ == "__main__":
    main()
