"""Render-check an official Dynamic3DGaussians reconstruction.

This script does not train or modify official code. It loads an official
`output/<exp>/<seq>/params.npz`, renders selected timesteps/cameras through the
official rasterizer helpers, compares against the converted RGB frames, and
writes side-by-side PNGs plus a JSON summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _parse_frames(spec: str, num_timesteps: int) -> list[int]:
    if spec == "default":
        return sorted({0, min(1, num_timesteps - 1), num_timesteps - 1})
    if spec == "all":
        return list(range(num_timesteps))
    frames = [int(item) for item in spec.split(",") if item]
    bad = [frame for frame in frames if frame < 0 or frame >= num_timesteps]
    if bad:
        raise ValueError(f"frame indices out of range: {bad}")
    return frames


def _parse_cameras(spec: str, num_cameras: int) -> list[int]:
    if spec == "all":
        return list(range(num_cameras))
    cameras = [int(item) for item in spec.split(",") if item]
    bad = [cam for cam in cameras if cam < 0 or cam >= num_cameras]
    if bad:
        raise ValueError(f"camera indices out of range: {bad}")
    return cameras


def _load_timestep_params(params_npz: np.lib.npyio.NpzFile, timestep: int) -> dict[str, torch.Tensor]:
    params = {}
    for key in params_npz.files:
        arr = params_npz[key]
        if key in {"means3D", "rgb_colors", "unnorm_rotations"} and arr.ndim >= 3:
            arr = arr[timestep]
        params[key] = torch.tensor(arr).cuda().float().contiguous()
    return params


def _to_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def _metrics(render: np.ndarray, target: np.ndarray) -> dict[str, float]:
    render_f = render.astype(np.float32) / 255.0
    target_f = target.astype(np.float32) / 255.0
    diff = render_f - target_f
    mse = float(np.mean(diff * diff))
    psnr = float(-10.0 * np.log10(max(mse, 1e-12)))
    return {
        "psnr": psnr,
        "l1": float(np.mean(np.abs(diff))),
        "mse": mse,
    }


def _comparison(render: np.ndarray, target: np.ndarray) -> np.ndarray:
    diff = np.clip(np.abs(render.astype(np.int16) - target.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
    return np.concatenate([target, render, diff], axis=1)


def _camera_stats(results: list[dict]) -> dict[str, dict[str, float]]:
    stats = {}
    for camera in sorted({item["camera"] for item in results}):
        rows = [item for item in results if item["camera"] == camera]
        psnr = np.array([item["psnr"] for item in rows], dtype=np.float64)
        l1 = np.array([item["l1"] for item in rows], dtype=np.float64)
        worst = min(rows, key=lambda item: item["psnr"])
        best = max(rows, key=lambda item: item["psnr"])
        stats[str(camera)] = {
            "count": int(len(rows)),
            "mean_psnr": float(psnr.mean()),
            "median_psnr": float(np.median(psnr)),
            "min_psnr": float(psnr.min()),
            "max_psnr": float(psnr.max()),
            "p10_psnr": float(np.percentile(psnr, 10)),
            "p90_psnr": float(np.percentile(psnr, 90)),
            "mean_l1": float(l1.mean()),
            "worst_timestep": int(worst["timestep"]),
            "worst_fn": worst["fn"],
            "best_timestep": int(best["timestep"]),
            "best_fn": best["fn"],
        }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official_repo", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--frames", default="default", help="'default', 'all', or comma-separated indices")
    ap.add_argument("--cameras", default="all", help="'all' or comma-separated camera indices")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--near", type=float, default=1.0)
    ap.add_argument("--far", type=float, default=100.0)
    args = ap.parse_args()
    if args.near <= 0.0 or args.far <= args.near:
        raise ValueError("--near must be positive and --far must be greater than --near")

    official_repo = Path(args.official_repo).expanduser().resolve()
    sys.path.insert(0, str(official_repo))
    from diff_gaussian_rasterization import GaussianRasterizer as Renderer
    from helpers import params2rendervar, setup_camera

    meta = json.loads((official_repo / "data" / args.sequence / "train_meta.json").read_text())
    params_path = official_repo / "output" / args.exp / args.sequence / "params.npz"
    if not params_path.exists():
        raise FileNotFoundError(params_path)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    params_npz = np.load(params_path)
    frames = _parse_frames(args.frames, len(meta["fn"]))
    cameras = _parse_cameras(args.cameras, len(meta["fn"][0]))
    results = []
    with torch.no_grad():
        for timestep in frames:
            params = _load_timestep_params(params_npz, timestep)
            for cam_idx in cameras:
                fn = meta["fn"][timestep][cam_idx]
                cam = setup_camera(
                    meta["w"],
                    meta["h"],
                    meta["k"][timestep][cam_idx],
                    meta["w2c"][timestep][cam_idx],
                    near=args.near,
                    far=args.far,
                )
                render, _, _ = Renderer(raster_settings=cam)(**params2rendervar(params))
                render = torch.exp(params["cam_m"][cam_idx])[:, None, None] * render
                render = render + params["cam_c"][cam_idx][:, None, None]
                render_img = _to_image(render)
                target_img = np.array(Image.open(official_repo / "data" / args.sequence / "ims" / fn).convert("RGB"))
                item = {
                    "timestep": timestep,
                    "camera": cam_idx,
                    "fn": fn,
                    **_metrics(render_img, target_img),
                }
                out_name = f"t{timestep:04d}_c{cam_idx}_{Path(fn).stem}.png"
                Image.fromarray(_comparison(render_img, target_img)).save(out_dir / out_name)
                item["comparison"] = str(out_dir / out_name)
                results.append(item)
    summary = {
        "sequence": args.sequence,
        "exp": args.exp,
        "params": str(params_path),
        "frames": frames,
        "cameras": cameras,
        "near": args.near,
        "far": args.far,
        "mean_psnr": float(np.mean([item["psnr"] for item in results])),
        "mean_l1": float(np.mean([item["l1"] for item in results])),
        "camera_stats": _camera_stats(results),
        "results": results,
    }
    (out_dir / "render_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
