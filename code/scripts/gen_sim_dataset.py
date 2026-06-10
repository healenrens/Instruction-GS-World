"""Parallel clean-GT sim dataset generation for the GENERALIZATION phase (agent.md §39 -> scale-up).

The overfit (§39) PROVED the architecture localizes motion on ONE clean sim clip (corr 0.95 with
per-control spatial-grounding). To test GENERALIZATION we need MANY clean clips with diversity:
multiple tasks x many seeds (randomized object positions -> varied motion + instruction).

This driver generates a list of (env, seed) jobs and runs `maniskill_gt.generate_episode + build_clip
+ validate` for each, writing one `.pt` clip per job (same schema as maniskill_gt.main). It is the
PER-WORKER body: the launcher (gen_sim_launch.sh) starts N copies, each pinned to a GPU and handed a
disjoint slice of the job list (--shard i / --nshards N), so all 4 GPUs + many CPU-sim workers run in
parallel. Clips below a val_psnr threshold (the GT self-check) are DROPPED (not saved).

A clip's filename encodes task+seed+split so the trainer can do the held-out split by filename:
   {env}_s{seed:04d}_{split}.pt          split in {train, heldseed, heldtask}
Held-out = a fraction of seeds (heldseed) AND one task held entirely (heldtask).

Usage (one worker):
  python code/scripts/gen_sim_dataset.py --out data/maniskill --shard 0 --nshards 16 \
      --tasks PickCube-v1,PushCube-v1,StackCube-v1 --seeds 200 --held_task StackCube-v1 \
      --held_seed_frac 0.15 --K 16 --cam 512 --min_val_psnr 16
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.maniskill_gt import generate_episode, build_clip, validate, _moving_seg_ids  # noqa: E402


def build_jobs(tasks, n_seeds, held_task, held_seed_frac, seed_base):
    """Deterministic (env, seed, split) job list. Held-out = the whole `held_task` (heldtask split)
    plus a fixed `held_seed_frac` of EACH non-held task's seeds (heldseed split). The split is keyed
    on (env, seed) by a stable hash so train/held membership is identical across workers + reruns."""
    import hashlib
    jobs = []
    for env in tasks:
        for i in range(n_seeds):
            seed = seed_base + i
            if env == held_task:
                split = "heldtask"
            else:
                h = int(hashlib.md5(f"{env}:{seed}".encode()).hexdigest(), 16) % 1000
                split = "heldseed" if h < int(held_seed_frac * 1000) else "train"
            jobs.append((env, seed, split))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/maniskill")
    ap.add_argument("--tasks", default="PickCube-v1,PushCube-v1,StackCube-v1")
    ap.add_argument("--seeds", type=int, default=200, help="seeds per task (0..seeds-1 + seed_base)")
    ap.add_argument("--seed_base", type=int, default=1000, help="offset so we don't collide with the 2 hand clips")
    ap.add_argument("--held_task", default="StackCube-v1", help="this task is ENTIRELY held out (task-level generalization)")
    ap.add_argument("--held_seed_frac", type=float, default=0.15, help="fraction of each non-held task's seeds held out")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--cam", type=int, default=512)
    ap.add_argument("--depth_max", type=float, default=2.0)
    ap.add_argument("--window_sec", type=float, default=0.0,
                    help=">0: each clip spans this many SECONDS of motion from a (random) start; 0=whole episode (legacy)")
    ap.add_argument("--control_freq", type=int, default=20,
                    help="sim control Hz (PickCube/PushCube=20) to map window_sec -> sim steps")
    ap.add_argument("--random_start", type=int, default=0,
                    help="1: random clip start within the episode (deterministic per env+seed); needs episodes longer than the window")
    ap.add_argument("--fuse_stride", type=int, default=0,
                    help=">0: WHOLE-VIDEO temporal fusion of the canonical G0 (every Nth frame registered into canonical via known poses) -> complete, low-uncertainty geometry; 0=single-frame G0 (legacy)")
    ap.add_argument("--min_val_psnr", type=float, default=16.0, help="drop clips whose mean(t>=1) full PSNR is below this")
    ap.add_argument("--min_movefrac", type=float, default=0.02, help="drop clips with too little moving geometry (frac>0.02r)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", type=int, default=0, help="0=skip clips already on disk (resumable)")
    args = ap.parse_args()

    import hashlib
    dev = args.device if torch.cuda.is_available() else "cpu"
    tasks = [t for t in args.tasks.split(",") if t]
    window_steps = int(round(args.window_sec * args.control_freq)) if args.window_sec > 0 else None
    jobs = build_jobs(tasks, args.seeds, args.held_task, args.held_seed_frac, args.seed_base)
    mine = jobs[args.shard::args.nshards]
    os.makedirs(args.out, exist_ok=True)
    print(f"[gen shard {args.shard}/{args.nshards}] {len(mine)}/{len(jobs)} jobs | dev={dev} "
          f"tasks={tasks} held_task={args.held_task}", flush=True)

    n_ok = n_drop = n_skip = n_fail = 0
    t0 = time.time()
    for ji, (env, seed, split) in enumerate(mine):
        short = env.replace("-v1", "").lower()
        out = os.path.join(args.out, f"{short}_s{seed:04d}_{split}.pt")
        if os.path.isfile(out) and not args.overwrite:
            n_skip += 1
            continue
        try:
            rec, instruction, success = generate_episode(env, seed, args.cam, args.cam)
            if not success:
                print(f"[gen {args.shard}] {env} s{seed}: NO motion -> drop", flush=True)
                n_drop += 1
                continue
            sseed = int(hashlib.md5(f"{env}:{seed}:start".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(sseed) if args.random_start else None
            clip = build_clip(rec, args.K, dev, depth_max=args.depth_max,
                              window_steps=window_steps, rng=rng, fuse_stride=args.fuse_stride)
            full, dyn, mover_names = validate(clip, dev, out_dir=None)
            mean_full = float(np.mean(full[1:]))
            # moving-geometry fraction (normalized by workspace radius over movers)
            X0 = clip["traj"][0]; XK = clip["traj"][clip["Kf"]]
            disp = (XK - X0).norm(dim=-1)
            move_ids = _moving_seg_ids(clip, dev)
            mm = torch.zeros(len(clip["seg_per_g"]), dtype=torch.bool, device=dev)
            for sid in move_ids:
                mm |= (clip["seg_per_g"] == sid)
            ref = X0[mm] if mm.any() else X0
            radius = (ref - ref.mean(0)).norm(dim=-1).quantile(0.9).clamp_min(1e-6)
            movefrac = float((disp / radius > 0.02).float().mean())
            if mean_full < args.min_val_psnr or movefrac < args.min_movefrac:
                print(f"[gen {args.shard}] {env} s{seed}: DROP psnr={mean_full:.1f} movefrac={movefrac:.3f}", flush=True)
                n_drop += 1
                continue
            save = {
                "means": clip["g0"].means.cpu(), "quats": clip["g0"].quats.cpu(),
                "scales": clip["g0"].scales.cpu(), "opacities": clip["g0"].opacities.cpu(),
                "colors": clip["g0"].colors.cpu(),
                "uv": clip["uv"].cpu(), "seg_per_g": clip["seg_per_g"].cpu(),
                "traj": clip["traj"].cpu(), "K_intr": clip["K_intr"].cpu(),
                "viewmat": clip["viewmat"].cpu(), "H": clip["H"], "W": clip["W"], "Kf": clip["Kf"],
                "instruction": instruction, "env": env, "seed": seed, "split": split,
                "val_psnr": full, "movefrac": movefrac,
                "window_sec": args.window_sec, "start_idx": clip.get("start"),
                "n_sim": clip.get("n_sim"), "win_steps": clip.get("win"),
                "fuse_stride": args.fuse_stride,
                "gt_rgb": torch.stack([f["rgb"] for f in clip["frames"]], 0),
            }
            torch.save(save, out)
            n_ok += 1
            rate = (ji + 1) / (time.time() - t0 + 1e-6)
            print(f"[gen {args.shard}] OK {env} s{seed} [{split}] psnr={mean_full:.1f} "
                  f"movefrac={movefrac:.3f} N={len(clip['g0'])} ({n_ok} ok / {ji+1} done, {rate:.2f}job/s) -> {os.path.basename(out)}",
                  flush=True)
        except Exception as e:
            n_fail += 1
            print(f"[gen {args.shard}] FAIL {env} s{seed}: {type(e).__name__}: {e}", flush=True)
            continue
    print(f"[gen shard {args.shard}] DONE: {n_ok} saved, {n_drop} dropped, {n_skip} skipped, "
          f"{n_fail} failed in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
