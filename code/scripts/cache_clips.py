"""Offline clip caching — the large-scale data-prep step.

For each (task, episode, clip) it: decodes K+1 head frames, Pi3-lifts them jointly
(consistent gauge), builds the dense canonical Gaussian set G0, per-frame cameras
(Ks, viewmats), GT frames, and the FROZEN Qwen3-VL hidden states for the
instruction (+ first frame). Everything is saved as one .pt so the trainer needs
neither Pi3 nor the VLM in-loop.

Resumable (skips existing) and shardable: --shard i --num_shards n splits the clip
list across GPUs/processes. Run one per GPU in the background to parallelize.

Storage/clip ~ 12 MB (G0 fp16 + GT uint8 + hidden bf16). 2 TB budget is ample.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.data import AgiBotLeRobotTask, list_tasks  # noqa: E402
from igsw.lifting import Pi3Lifter, points_to_gaussians  # noqa: E402
from igsw.gaussians import intrinsics_from_local_points, viewmat_from_pose  # noqa: E402


def enumerate_clips(tasks, eps_per_task, clips_per_ep, K, stride, fps_margin=30):
    """Yield (task_root, ep, f0) clip specs deterministically."""
    specs = []
    for tr in tasks:
        try:
            t = AgiBotLeRobotTask(tr)
        except Exception:
            continue
        eps = t.episode_indices[:eps_per_task]
        for ep in eps:
            length = t.episode_meta(ep).length
            span = K * stride
            last_f0 = length - span - fps_margin
            if last_f0 <= fps_margin:
                continue
            f0s = np.linspace(fps_margin, last_f0, clips_per_ep).astype(int).tolist()
            for f0 in sorted(set(f0s)):
                specs.append((tr, ep, int(f0)))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/mnt/pfs/public/xuhaoming/instruct_gs_world/data/clips_v1")
    ap.add_argument("--tasks", default="all", help="'all' or comma-separated task roots")
    ap.add_argument("--n_tasks", type=int, default=20)
    ap.add_argument("--eps_per_task", type=int, default=10)
    ap.add_argument("--clips_per_ep", type=int, default=4)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max_gaussians", type=int, default=120000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--no_vlm", action="store_true", help="skip caching Qwen hidden (store raw text only)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of clips (0=no cap)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda"

    all_tasks = list_tasks() if args.tasks == "all" else args.tasks.split(",")
    all_tasks = all_tasks[: args.n_tasks]
    specs = enumerate_clips(all_tasks, args.eps_per_task, args.clips_per_ep, args.K, args.stride)
    specs = specs[args.shard :: args.num_shards]
    if args.limit:
        specs = specs[: args.limit]
    print(f"[cache] shard {args.shard}/{args.num_shards}: {len(specs)} clips -> {args.out_dir}", flush=True)

    lifter = Pi3Lifter(device=dev)
    enc = None
    if not args.no_vlm:
        from igsw.dynamics.conditioning import QwenVLEncoder
        enc = QwenVLEncoder(device=dev)

    done = skip = fail = 0
    t0 = time.time()
    for i, (tr, ep, f0) in enumerate(specs):
        tname = os.path.basename(os.path.dirname(tr))
        out = os.path.join(args.out_dir, f"{tname}_ep{ep:04d}_f{f0:05d}_K{args.K}s{args.stride}.pt")
        if os.path.exists(out):
            skip += 1
            continue
        try:
            t = AgiBotLeRobotTask(tr)
            lang = t.language(ep)
            idx = [f0 + j * args.stride for j in range(args.K + 1)]
            frames = t.decode_frames(ep, "observation.images.head", idx)  # [K+1,H,W,3]
            res = lifter.lift(frames, conf_thr=0.1, edge_rtol=0.03)
            pts = res["points"].to(dev); imgs = res["images"].to(dev)
            mask = res["mask"].to(dev); local = res["local_points"].to(dev); poses = res["camera_poses"].to(dev)
            Kf1, _, H, W = imgs.shape
            g0 = points_to_gaussians(pts[:1], imgs[:1], mask[:1], opacity_init=0.9, scale_factor=1.0)
            if len(g0) > args.max_gaussians:
                from igsw.gaussians.sampling import downsample_gaussians
                g0 = downsample_gaussians(g0, args.max_gaussians)
            Ks = torch.stack([intrinsics_from_local_points(local[k]) for k in range(Kf1)], 0)
            viewmats = torch.stack([viewmat_from_pose(poses[k]) for k in range(Kf1)], 0)

            rec = {
                "g0": {
                    "means": g0.means.half().cpu(), "quats": g0.quats.half().cpu(),
                    "scales": g0.scales.half().cpu(), "opacities": g0.opacities.half().cpu(),
                    "colors": g0.colors.half().cpu(),
                },
                "gt": (imgs.permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8).cpu(),  # [K+1,H,W,3]
                "Ks": Ks.cpu(), "viewmats": viewmats.cpu(),
                "H": H, "W": W, "K": args.K, "stride": args.stride,
                "instruction": lang, "task": tname, "ep": ep, "f0": f0,
            }
            if enc is not None:
                h, m = enc.encode(lang, image=frames[0])
                rec["lang_hidden"] = h.to(torch.bfloat16).cpu()   # [L,2048]
                rec["lang_mask"] = m.cpu()
            torch.save(rec, out)
            done += 1
            if done % 20 == 0:
                rate = (done + skip) / (time.time() - t0 + 1e-6)
                print(f"  [{i+1}/{len(specs)}] done={done} skip={skip} fail={fail} "
                      f"{rate:.2f} clips/s  last={os.path.basename(out)}", flush=True)
        except Exception as e:
            fail += 1
            print(f"  [FAIL] {tname} ep{ep} f{f0}: {type(e).__name__}: {e}", flush=True)
    print(f"[cache] DONE shard {args.shard}: done={done} skip={skip} fail={fail} "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
