"""AgiBot static-camera GATE (batch). For each (task, episode) sample, decode a sub-window, run the
SpaTracker producer core with fixed_cam=False, inspect the returned per-frame c2w + mover-fraction, and
apply the prior agent's gate rule:
    STATIC  iff  c2w translation < 2% of median scene depth  AND  mover-fraction < 50%.

Loads VGGT+Predictor ONCE and loops over a JSON job list. Writes a JSONL with one line per episode so we
can build the CONFIRMED-STATIC episode list for GT production. Reuses agibot_spatrack_eval's loaders/core.

Usage:
  agibot_gate.py --jobs jobs.json --out gate_results.jsonl [--kf 12 --grid 40 --win_start 0.3 --win_frac 0.4]
where jobs.json = [["task_327", 0], ["task_327", 5], ...]
"""
import argparse, sys, os, json
for _k, _v in {"HF_HOME": "/mnt/pfs/public/xuhaoming/hf_cache", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "XDG_CACHE_HOME": "/mnt/pfs/public/xuhaoming/.cache"}.items():
    os.environ.setdefault(_k, _v)
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agibot_spatrack_eval import (agibot_head_video, task_label, decode_window, run_core,
                                  cam_motion_stats, reproj_err, quality)
SPT = "/mnt/pfs/public/xuhaoming/SpaTrackerV2"; sys.path.insert(0, SPT)
from models.SpaTrackV2.models.predictor import Predictor                          # noqa: E402
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kf", type=int, default=12)
    ap.add_argument("--grid", type=int, default=40)
    ap.add_argument("--win_start", type=float, default=0.30)
    ap.add_argument("--win_frac", type=float, default=0.40)
    ap.add_argument("--iters_track", type=int, default=4)
    ap.add_argument("--trans_frac_thr", type=float, default=0.02)
    ap.add_argument("--mover_frac_thr", type=float, default=0.50)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()

    jobs = json.load(open(args.jobs))
    if args.nshard > 1: jobs = jobs[args.shard::args.nshard]
    print(f"[gate] {len(jobs)} episodes to gate (shard {args.shard}/{args.nshard})", flush=True)

    vggt = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().cuda()
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline").eval().cuda()

    fout = open(args.out, "a")
    done = 0
    for task, ep in jobs:
        try:
            vid = agibot_head_video(task, ep)
            lbl = task_label(task)
            frames, idx = decode_window(vid, args.kf, args.win_start, args.win_frac)
            res = run_core(frames, vggt, model, args.grid, args.iters_track, fixed_cam=False)
            q = quality(res); tr, rot, _ = cam_motion_stats(res["C2W"])
            re = reproj_err(res)
            zmed = q["zmed"]; tr_frac = tr / max(zmed, 1e-6)
            is_static = (tr_frac < args.trans_frac_thr) and (q["mover_frac"] < args.mover_frac_thr)
            rec = dict(task=task, episode=int(ep), label=lbl[:90], src=[idx[0], idx[-1]],
                       trans_frac=round(tr_frac, 4), rot_deg=round(rot, 3),
                       mover_frac=round(q["mover_frac"], 3), movers=q["movers"],
                       mover_med2d=round(q["mover_med2d"], 2), vis=round(q["vis"], 3),
                       zmed=round(zmed, 3), reproj_med=round(float(np.median(re)), 3),
                       static=bool(is_static))
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(f"[gate] {task} ep{ep}: trans={100*tr_frac:.1f}% rot={rot:.2f} "
                  f"mvfrac={100*q['mover_frac']:.0f}% reproj={np.median(re):.2f}px STATIC={is_static}", flush=True)
            done += 1
            del res; torch.cuda.empty_cache()
        except Exception as e:
            print(f"[gate] SKIP {task} ep{ep}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
    fout.close()
    print(f"[gate] FINISHED {done}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
