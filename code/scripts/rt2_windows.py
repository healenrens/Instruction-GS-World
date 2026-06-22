"""RoboTwin2 windowed STAGE-1 data plan: the SINGLE source of truth for windows + splits, shared by
BOTH the backbone flow-GT dataset and the action dataset so they stay consistent.

A WINDOW = WIN (=50) consecutive control steps starting at step s. We slide with STRIDE over each episode.
The backbone clip takes KF+1 (=13) frames evenly spanning [s, s+WIN] and runs SpaTracker; the action clip
stores frame0 + the per-step Δqpos chunk over the same [s, s+WIN].

Splits (per task):
  - heldtask : a fixed list of whole tasks held out entirely (zero-shot task generalization).
  - heldseed : on the remaining tasks, ~HELDSEED_FRAC of EPISODES per task held out (unseen episodes).
  - train    : the rest.
The episode->split assignment is DETERMINISTIC (seeded by task name) so it is identical for both datasets.

RoboTwin2 facts (verified): sim timestep 1/250 s, save_freq 15 -> control freq = 250/15 = 16.67 Hz.
WIN=50 steps = 50 * 15/250 = 3.0 s. Head camera is STATIC (cam2world_gl constant) -> fixed_cam=True valid.
"""
import argparse
import hashlib
import json
import os

import h5py
import numpy as np

DATA_ROOT = "/root/xuhaoming/public/Cosmos-3-Finetune/data/RoboTwin2"
# Only 22/50 tasks ship inline instructions; this overlay (used by the Cosmos dataloader) covers ALL 50.
INSTR_OVERLAY = "/mnt/pfs/public/xuhaoming/Cosmos-3-Finetune/data/robotwin2_instructions_overlay"


def instruction_json(task, ep):
    """Resolve the per-episode instruction json: prefer the inline file, fall back to the overlay (covers
    all 50 tasks). Returns the first existing path, or the inline path (may not exist -> caller handles)."""
    inline = os.path.join(DATA_ROOT, task, "demo_clean", "instructions", f"episode{ep}.json")
    if os.path.exists(inline):
        return inline
    overlay = os.path.join(INSTR_OVERLAY, task, "demo_clean", "instructions", f"episode{ep}.json")
    return overlay if os.path.exists(overlay) else inline
WIN = 50            # window length in control steps
KF = 12            # KF+1 = 13 frames spanning the window for SpaTracker
STRIDE = 25        # 25 = 50% overlap -> ~2x windows per episode vs non-overlap
HELDSEED_FRAC = 0.20
CONTROL_HZ = 250.0 / 15.0     # 16.667 Hz
# 3 whole tasks held out entirely (heldtask). Picked to be representative + reliable movers.
HELDTASK = ["handover_block", "place_object_basket", "stack_blocks_two"]


def _episode_split(task, ep_idx, n_eps):
    """Deterministic per-(task,episode) split. heldtask tasks -> all 'heldtask'. Else ~HELDSEED_FRAC of
    episodes -> 'heldseed' (chosen by a stable hash of task+ep so it never moves), rest -> 'train'."""
    if task in HELDTASK:
        return "heldtask"
    h = int(hashlib.md5(f"{task}:{ep_idx}".encode()).hexdigest(), 16)
    return "heldseed" if (h % 1000) < int(HELDSEED_FRAC * 1000) else "train"


def list_tasks():
    return sorted(d for d in os.listdir(DATA_ROOT)
                  if os.path.isdir(os.path.join(DATA_ROOT, d, "demo_clean", "data")))


def episode_length(hdf5_path):
    with h5py.File(hdf5_path, "r") as h:
        return int(h["joint_action/vector"].shape[0])


def plan_windows(tasks=None, win=WIN, stride=STRIDE, kf=KF, min_len=None):
    """Return a list of window dicts: {task, ep, hdf5, json, T, s, e, split, name}. e = s+win (inclusive of
    s..s+win for KF frames; the action chunk uses deltas s..s+win-1 -> win deltas). Windows need T > s+win."""
    if tasks is None:
        tasks = list_tasks()
    if min_len is None:
        min_len = win + 1
    out = []
    for task in tasks:
        dd = os.path.join(DATA_ROOT, task, "demo_clean", "data")
        idir = os.path.join(DATA_ROOT, task, "demo_clean", "instructions")
        eps = sorted(int(f[len("episode"):-len(".hdf5")]) for f in os.listdir(dd)
                     if f.startswith("episode") and f.endswith(".hdf5"))
        n_eps = len(eps)
        for ep in eps:
            hp = os.path.join(dd, f"episode{ep}.hdf5")
            jp = instruction_json(task, ep)
            T = episode_length(hp)
            if T < min_len:
                continue
            split = _episode_split(task, ep, n_eps)
            # last valid start: need s+win <= T-1 (so frame s+win exists)
            last_s = T - 1 - win
            if last_s < 0:
                continue
            starts = list(range(0, last_s + 1, stride))
            if starts[-1] != last_s:
                starts.append(last_s)            # always include a tail window covering the end
            for wi, s in enumerate(starts):
                out.append({
                    "task": task, "ep": ep, "hdf5": hp, "json": jp, "T": T,
                    "s": s, "e": s + win, "win": win, "kf": kf, "split": split,
                    "name": f"{task}_ep{ep:02d}_w{s:04d}",
                })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/rt2_win/window_plan.json")
    ap.add_argument("--win", type=int, default=WIN)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--kf", type=int, default=KF)
    ap.add_argument("--summary_only", action="store_true")
    args = ap.parse_args()

    plan = plan_windows(win=args.win, stride=args.stride, kf=args.kf)
    from collections import Counter, defaultdict
    by_split = Counter(w["split"] for w in plan)
    eps_by_split = defaultdict(set)
    for w in plan:
        eps_by_split[w["split"]].add((w["task"], w["ep"]))
    tasks = sorted(set(w["task"] for w in plan))
    print(f"control freq = {CONTROL_HZ:.2f} Hz  ->  WIN={args.win} steps = {args.win/CONTROL_HZ:.2f} s")
    print(f"tasks total = {len(tasks)}  heldtask = {HELDTASK}")
    print(f"windows: total={len(plan)}  " + "  ".join(f"{k}={by_split[k]}" for k in ("train", "heldseed", "heldtask")))
    print("episodes: " + "  ".join(f"{k}={len(eps_by_split[k])}" for k in ("train", "heldseed", "heldtask")))
    wpe = len(plan) / max(1, sum(len(v) for v in eps_by_split.values()))
    print(f"avg windows/episode = {wpe:.2f}")
    if not args.summary_only:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(plan, open(args.out, "w"))
        print(f"wrote {args.out}  ({len(plan)} windows)")


if __name__ == "__main__":
    main()
