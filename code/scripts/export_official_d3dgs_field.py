"""Export official Dynamic3DGaussians params into a model-learning tensor pack.

The official `params.npz` is the reconstruction artifact. This script only
repackages it into the local fixed-ID `g0 + traj` contract used by the IGSW
training/eval scripts, while preserving the official time-varying state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


TIME_VARYING = ("means3D", "rgb_colors", "unnorm_rotations")
PERSISTENT = ("log_scales", "logit_opacities", "seg_colors")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("unnorm_rotations contains zero-length quaternions")
    return q / norm


def _load_params(path: Path) -> dict[str, np.ndarray]:
    params = dict(np.load(path))
    missing = [key for key in (*TIME_VARYING, *PERSISTENT) if key not in params]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    params = {key: np.asarray(value, dtype=np.float32) for key, value in params.items()}
    means = params["means3D"]
    if means.ndim != 3 or means.shape[-1] != 3:
        raise ValueError(f"means3D must be [T,N,3], got {means.shape}")
    t_count, n_gauss = means.shape[:2]
    expected = {
        "rgb_colors": (t_count, n_gauss, 3),
        "unnorm_rotations": (t_count, n_gauss, 4),
        "log_scales": (n_gauss, 3),
        "logit_opacities": (n_gauss, 1),
        "seg_colors": (n_gauss, 3),
    }
    for key, shape in expected.items():
        if params[key].shape != shape:
            raise ValueError(f"{key} expected {shape}, got {params[key].shape}")
    for key in (*TIME_VARYING, *PERSISTENT):
        if not np.isfinite(params[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    return params


def _load_meta(sequence_dir: Path | None) -> tuple[dict, dict]:
    if sequence_dir is None:
        return {}, {}
    meta_path = sequence_dir / "train_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text())
    camera_meta = {
        "sequence_dir": str(sequence_dir),
        "width": int(meta["w"]),
        "height": int(meta["h"]),
        "fn": meta["fn"],
        "cam_id": meta.get("cam_id"),
        "num_cameras": len(meta["fn"][0]),
        "num_observations": sum(len(row) for row in meta["fn"]),
    }
    compact = {
        "sequence_dir": str(sequence_dir),
        "width": camera_meta["width"],
        "height": camera_meta["height"],
        "num_cameras": camera_meta["num_cameras"],
        "num_observations": camera_meta["num_observations"],
        "first_fn": meta["fn"][0],
        "last_fn": meta["fn"][-1],
        "first_cam_id": meta.get("cam_id", [None])[0],
    }
    return camera_meta, compact


def _to_tensor(arr: np.ndarray, dtype: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(arr))
    if dtype == "float16":
        return tensor.half()
    if dtype == "float32":
        return tensor.float()
    raise ValueError(f"unsupported dtype: {dtype}")


def export(params_path: Path, out_path: Path, sequence_dir: Path | None, dtype: str) -> dict:
    params = _load_params(params_path)
    means = params["means3D"]
    quats = _normalize_quat(params["unnorm_rotations"])
    colors = np.clip(params["rgb_colors"], 0.0, 1.0)
    scales = np.exp(params["log_scales"])
    opacities = _sigmoid(params["logit_opacities"])[:, 0]
    seg = params["seg_colors"][:, 0] > 0.5
    disp = np.linalg.norm(means - means[0:1], axis=-1)
    radius = float(np.linalg.norm(means[0].max(axis=0) - means[0].min(axis=0)))
    move_eps = max(radius * 0.01, 1e-6)
    summary = {
        "contract": "official_d3dgs_dynamic_field_v1",
        "source_params": str(params_path),
        "timesteps": int(means.shape[0]),
        "gaussians": int(means.shape[1]),
        "dtype": dtype,
        "foreground_count": int(seg.sum()),
        "background_count": int((~seg).sum()),
        "scene_radius_proxy": radius,
        "move_eps": move_eps,
        "moving_fraction": float((disp.max(axis=0) > move_eps).mean()),
        "mean_final_displacement": float(disp[-1].mean()),
        "max_final_displacement": float(disp[-1].max()),
    }
    camera_meta, compact_meta = _load_meta(sequence_dir)
    summary.update(compact_meta)
    pack = {
        "contract": summary["contract"],
        "summary": summary,
        "means": _to_tensor(means[0], dtype),
        "quats": _to_tensor(quats[0], dtype),
        "scales": _to_tensor(scales, dtype),
        "opacities": _to_tensor(opacities, dtype),
        "colors": _to_tensor(colors[0], dtype),
        "traj": _to_tensor(means, dtype),
        "rot_traj": _to_tensor(quats, dtype),
        "color_traj": _to_tensor(colors, dtype),
        "seg": torch.from_numpy(seg),
        "official_keys": sorted(params.keys()),
    }
    if camera_meta:
        pack["camera_meta"] = camera_meta
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, out_path)
    summary["out_path"] = str(out_path)
    summary["out_bytes"] = out_path.stat().st_size
    (out_path.with_suffix(out_path.suffix + ".json")).write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="Official output/<exp>/<seq>/params.npz")
    ap.add_argument("--out", required=True, help="Output .pt path")
    ap.add_argument("--sequence_dir", help="Optional official data/<seq> directory for camera metadata")
    ap.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    args = ap.parse_args()
    summary = export(
        Path(args.params).expanduser().resolve(),
        Path(args.out).expanduser().resolve(),
        Path(args.sequence_dir).expanduser().resolve() if args.sequence_dir else None,
        args.dtype,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
