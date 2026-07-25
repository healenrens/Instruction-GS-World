"""Build the strict-causal RoboTwin VLA dataset from existing local artifacts.

Inputs use only the current head RGB through single-frame VGGT on a fixed 48x48 grid.
Full-video SpaTracker outputs are optional geometry labels. All action windows remain in
the dataset, including windows without a valid tracker result.
"""
import argparse
import glob
import io
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
from PIL import Image

torch.set_num_threads(int(os.environ.get("RT2_CAUSAL_CPU_THREADS", "2")))
torch.set_num_interop_threads(1)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.causal_geometry import (CAUSAL_GEOMETRY_VERSION, causal_geometry_from_prediction,
                                  preprocessed_rgb, tracker_targets_on_grid)


def resolve_hdf5(path: str) -> str:
    if os.path.exists(path):
        return path
    old = "/root/xuhaoming/public/"
    if path.startswith(old):
        candidate = "/mnt/pfs/public/xuhaoming/" + path[len(old):]
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(path)


def camera_rgb(hdf5: h5py.File, camera: str, frame: int) -> torch.Tensor:
    raw = bytes(hdf5[f"observation/{camera}/rgb"][frame])
    rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(rgb.copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/rt2_joint_src")
    ap.add_argument("--actions", default="data/rt2_act")
    ap.add_argument("--flow", default="data/rt2_joint")
    ap.add_argument("--plan", default="data/rt2_win/window_plan.json")
    ap.add_argument("--out", default="data/rt2_causal_v1")
    ap.add_argument("--spatrack_root", default="/mnt/pfs/public/xuhaoming/SpaTrackerV2")
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--kf", type=int, default=12)
    ap.add_argument("--max_match_px", type=float, default=2.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.grid != 48:
        raise ValueError(f"{CAUSAL_GEOMETRY_VERSION} requires --grid 48")
    if not (0 <= a.shard < a.nshard):
        raise ValueError("invalid shard")

    for key, value in {"HF_HOME": "/mnt/pfs/public/xuhaoming/hf_cache", "HF_HUB_OFFLINE": "1",
                       "TRANSFORMERS_OFFLINE": "1",
                       "XDG_CACHE_HOME": "/mnt/pfs/public/xuhaoming/.cache"}.items():
        os.environ.setdefault(key, value)
    sys.path.insert(0, a.spatrack_root)
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
    from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

    plan = json.load(open(a.plan))
    episode_paths = {(w["task"], int(w["ep"])): resolve_hdf5(w["hdf5"]) for w in plan}
    files = sorted(glob.glob(os.path.join(a.source, "*.pt")))[a.shard::a.nshard]
    if a.limit:
        files = files[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().cuda()

    open_path = None
    open_hdf5 = None
    written = skipped = with_flow = valid_labels = 0
    started = time.time()
    for index, source_path in enumerate(files, 1):
        name = os.path.basename(source_path)
        out_path = os.path.join(a.out, name)
        if os.path.exists(out_path) and not a.overwrite:
            skipped += 1
            continue
        action_path = os.path.join(a.actions, name)
        if not os.path.exists(action_path):
            raise FileNotFoundError(action_path)
        action = torch.load(action_path, map_location="cpu", weights_only=False)
        frame = action["frame0"].permute(2, 0, 1).float()[None]
        video = preprocess_image(frame)[None]
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(video.cuda() / 255.0)
        geom = causal_geometry_from_prediction(pred["points_map"], pred["intrs"], a.grid)
        means = geom["means"].cpu()
        uv = geom["uv"].cpu()
        K = geom["K_intr"].cpu()
        kf = a.kf

        flow_path = os.path.join(a.flow, name)
        if os.path.exists(flow_path):
            flow = torch.load(flow_path, map_location="cpu", weights_only=False)
            labels = tracker_targets_on_grid(means, uv, K, flow["uv"], flow["traj"],
                                              flow["K_intr"], flow.get("vis"), a.max_match_px)
            if int(flow["Kf"]) != kf:
                raise ValueError(f"Kf mismatch for {name}")
            with_flow += 1
        else:
            labels = {"traj": means[None].repeat(kf + 1, 1, 1),
                      "vis": torch.zeros(kf + 1, len(means), dtype=torch.bool),
                      "geom_valid": torch.zeros(len(means), dtype=torch.bool),
                      "match_distance": torch.full((len(means),), float("inf")),
                      "matched_tracks": 0}
        valid_count = int(labels["geom_valid"].sum())
        valid_labels += valid_count

        episode_path = episode_paths[(action["task"], int(action["ep"]))]
        if open_path != episode_path:
            if open_hdf5 is not None:
                open_hdf5.close()
            open_hdf5 = h5py.File(episode_path, "r")
            open_path = episode_path
        frame_index = int(action["s"])
        clip = {
            "means": means, "uv": uv, "traj": labels["traj"].cpu(),
            "K_intr": K, "viewmat": torch.eye(4), "H": int(geom["H"]), "W": int(geom["W"]),
            "Kf": kf, "instruction": action["instruction"],
            "gt_rgb": preprocessed_rgb(video).cpu()[None], "vis": labels["vis"].cpu(),
            "geom_valid": labels["geom_valid"].cpu(),
            "geom_match_distance": labels["match_distance"].cpu(),
            "geometry_valid_count": valid_count,
            "geometry_matched_tracks": int(labels["matched_tracks"]),
            "causal_geometry_version": CAUSAL_GEOMETRY_VERSION,
            "geometry_input_source": "current_head_rgb_only",
            "geometry_target_source": "spatrack_full_video" if os.path.exists(flow_path) else "none",
            "dq": action["dq"].float(), "anchor": action["anchor"].float(),
            "control_hz": float(action["control_hz"]), "task": action["task"],
            "ep": int(action["ep"]), "s": frame_index, "win": int(action["win"]),
            "split": action["split"],
            "left_rgb": camera_rgb(open_hdf5, "left_camera", frame_index),
            "right_rgb": camera_rgb(open_hdf5, "right_camera", frame_index),
        }
        tmp_path = f"{out_path}.tmp.{os.getpid()}"
        torch.save(clip, tmp_path)
        os.replace(tmp_path, out_path)
        written += 1
        if index == 1 or index % 100 == 0:
            rate = written / max(time.time() - started, 1e-6)
            print(f"[causal-data] shard {a.shard}/{a.nshard} {index}/{len(files)} written={written} "
                  f"skip={skipped} flow={with_flow} valid={valid_count}/{len(means)} "
                  f"rate={rate:.2f}/s", flush=True)
    if open_hdf5 is not None:
        open_hdf5.close()
    mean_valid = valid_labels / max(with_flow, 1)
    print(f"[causal-data] DONE shard {a.shard}/{a.nshard}: written={written} skipped={skipped} "
          f"flow={with_flow} mean_valid_per_flow={mean_valid:.1f} out={a.out}", flush=True)


if __name__ == "__main__":
    main()
