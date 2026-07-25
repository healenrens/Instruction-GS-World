"""Build the 13-frame SpaTracker-INPUT intermediates for the FULL plan (all 20642 windows).

Same per-window output as rt2_build_intermediate.build_one (gt_rgb=[kf+1,H,W,3] uint8 + instruction +
metadata), but iterates the ENTIRE window_plan instead of select_subset's per-task budget. Sharded for
4-GPU/4-process parallelism. Resumable (skips existing).

Usage: rt2_joint_intermediate.py --plan data/rt2_win/window_plan.json --out data/rt2_joint_src \
         [--shard 0 --nshard 4]
"""
import argparse, io, json, os, random
import h5py, numpy as np, torch
from PIL import Image


def pick_seen_instruction(json_path, seed_name):
    j = json.load(open(json_path))
    seen = j.get("seen") or j.get("unseen") or []
    if not seen:
        return ""
    return random.Random(seed_name).choice(seen)


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
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    plan = json.load(open(args.plan))
    if args.limit:
        plan = plan[:args.limit]
    sub = plan[args.shard::args.nshard]
    ok = sk = err = 0
    for i, w in enumerate(sub):
        try:
            r = build_one(w, args.out, args.overwrite)
            ok += (r == "ok"); sk += (r == "skip")
        except Exception as e:
            err += 1
            print(f"[jsrc] ERR {w['name']}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 200 == 0:
            print(f"[jsrc] shard{args.shard}: {i+1}/{len(sub)} ok={ok} skip={sk} err={err}", flush=True)
    print(f"[jsrc] shard{args.shard} DONE: ok={ok} skip={sk} err={err} (of {len(sub)})", flush=True)


if __name__ == "__main__":
    main()
