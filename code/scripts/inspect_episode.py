"""Smoke-test the AgiBot LeRobot reader: language, proprio shapes, frame decode.

Usage:
    python scripts/inspect_episode.py [--task <root>] [--ep N] [--out DIR]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.data import AgiBotLeRobotTask, RGB_CAMERAS, list_tasks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="LeRobot task root (…/task_XXX/task_XXX)")
    ap.add_argument("--ep", type=int, default=None)
    ap.add_argument("--out", default="outputs/inspect")
    args = ap.parse_args()

    if args.task is None:
        tasks = list_tasks()
        print(f"[discovery] found {len(tasks)} task roots; using first: {tasks[0]}")
        args.task = tasks[0]

    t = AgiBotLeRobotTask(args.task)
    eps = t.episode_indices
    ep = args.ep if args.ep is not None else eps[0]
    print(f"[task] root={t.root}")
    print(f"[task] fps={t.fps} chunks_size={t.chunks_size} n_episodes={len(eps)}")
    print(f"[task] video_keys={t.video_keys}")
    em = t.episode_meta(ep)
    print(f"[ep {ep}] length={em.length} (~{em.length / t.fps:.1f}s)  tasks={em.tasks}")
    print(f"[ep {ep}] language = {t.language(ep)!r}")

    # proprio / actions
    cols = t.read_parquet(ep)
    print(f"[ep {ep}] parquet columns ({len(cols)}):")
    for k in sorted(cols):
        v = cols[k]
        extra = ""
        if v.ndim <= 2 and v.size <= v.shape[0] * 8:
            extra = f" first={np.asarray(v[0]).ravel()[:8]}"
        print(f"    {k:42s} {str(v.shape):16s} {v.dtype}{extra}")

    # frame decode sanity for the head camera + count check vs parquet length
    cam = "observation.images.head"
    nvf = t.num_video_frames(ep, cam)
    print(f"[ep {ep}] head video frames = {nvf}; parquet rows = {cols['timestamp'].shape[0]}")

    # decode 5 evenly-spaced frames from every RGB camera that exists
    os.makedirs(args.out, exist_ok=True)
    from PIL import Image

    for cam in RGB_CAMERAS:
        if cam not in t.video_keys:
            continue
        n = t.num_video_frames(ep, cam)
        idx = list(np.linspace(0, n - 1, 5).astype(int))
        frames = t.decode_frames(ep, cam, idx)
        short = cam.split(".")[-1]
        print(f"    [{short}] decoded {frames.shape} idx={idx} "
              f"mean={frames.mean():.1f}")
        # save the middle frame for visual sanity
        mid = frames[len(frames) // 2]
        Image.fromarray(mid).save(os.path.join(args.out, f"ep{ep}_{short}.png"))
    print(f"[done] sample frames written to {args.out}/")


if __name__ == "__main__":
    main()
