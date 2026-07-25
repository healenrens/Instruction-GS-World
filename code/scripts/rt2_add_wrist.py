"""Patch rt2_joint clips with wrist frame0 (left_rgb/right_rgb) from the raw hdf5, for --wrist conditioning.
Adds ONLY left_rgb/right_rgb (uint8 [h,w,3]) per clip; the head 3D/GPSToken path is untouched. Idempotent.

  rt2_add_wrist.py --data data/rt2_joint --plan data/rt2_win/window_plan.json [--tasks grab_roller,click_bell]
                   [--shard i --nshard N]
"""
import argparse, glob, io, json, os
import h5py, numpy as np, torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rt2_joint")
    ap.add_argument("--plan", default="data/rt2_win/window_plan.json")
    ap.add_argument("--tasks", default="", help="comma list to limit (empty = all)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    plan = json.load(open(a.plan))
    ep2hdf5 = {(w["task"], int(w["ep"])): w["hdf5"] for w in plan}        # (task,ep) -> raw hdf5
    files = [f for f in sorted(glob.glob(f"{a.data}/*.pt")) if "norm_stats" not in f]
    if a.tasks:
        ts = tuple(t + "_ep" for t in a.tasks.split(","))
        files = [f for f in files if os.path.basename(f).startswith(ts)]
    files = files[a.shard::a.nshard]
    ok = sk = err = 0
    open_h = {"path": None, "h": None}                                   # reuse one open hdf5 across an episode

    def rgb_at(h, cam, s):
        ds = h[f"observation/{cam}/rgb"]
        return np.array(Image.open(io.BytesIO(bytes(ds[s]))).convert("RGB")).astype(np.uint8)

    for i, f in enumerate(files):
        try:
            c = torch.load(f, map_location="cpu", weights_only=False)
            if "left_rgb" in c and "right_rgb" in c and not a.overwrite:
                sk += 1; continue
            hp = ep2hdf5.get((c["task"], int(c["ep"])))
            if hp is None:
                err += 1; continue
            if open_h["path"] != hp:
                if open_h["h"] is not None:
                    open_h["h"].close()
                open_h["h"] = h5py.File(hp, "r"); open_h["path"] = hp
            s = int(c["s"])
            c["left_rgb"] = torch.from_numpy(rgb_at(open_h["h"], "left_camera", s))
            c["right_rgb"] = torch.from_numpy(rgb_at(open_h["h"], "right_camera", s))
            torch.save(c, f)
            ok += 1
            if ok == 1:
                print(f"[wrist] first: left {tuple(c['left_rgb'].shape)} right {tuple(c['right_rgb'].shape)} "
                      f"head {tuple(c['gt_rgb'].shape)}", flush=True)
        except Exception as e:
            err += 1
            if err <= 5:
                print(f"[wrist] ERR {os.path.basename(f)}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 500 == 0:
            print(f"[wrist] {i+1}/{len(files)} ok={ok} skip={sk} err={err}", flush=True)
    if open_h["h"] is not None:
        open_h["h"].close()
    print(f"[wrist] DONE shard{a.shard}: ok={ok} skip={sk} err={err} / {len(files)}", flush=True)


if __name__ == "__main__":
    main()
