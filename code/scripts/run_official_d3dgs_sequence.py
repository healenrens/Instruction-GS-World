"""Run the official Dynamic3DGaussians trainer on a converted sequence.

The official repo hardcodes `./data/<seq>` and exposes `train(seq, exp)` from
`train.py`. This wrapper verifies the converted head+wrist sequence, links it
into the official repo, and calls that official function.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


REQUIRED_META = ("w", "h", "fn", "k", "w2c")


def _check_matrix_array(value, shape_tail: tuple[int, int], name: str) -> None:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[-2:] != shape_tail:
        raise ValueError(f"{name} must be [T,C,{shape_tail[0]},{shape_tail[1]}], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")


def _load_meta(path: Path) -> dict:
    meta = json.loads(path.read_text())
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    fns = meta["fn"]
    if not fns or not all(fns_t for fns_t in fns):
        raise ValueError(f"{path} has empty fn entries")
    n_cam = len(fns[0])
    if n_cam < 2:
        raise ValueError(f"{path} must contain at least head+wrist cameras")
    if any(len(row) != n_cam for row in fns):
        raise ValueError(f"{path} has inconsistent camera count")
    _check_matrix_array(meta["k"], (3, 3), f"{path}.k")
    _check_matrix_array(meta["w2c"], (4, 4), f"{path}.w2c")
    if np.asarray(meta["k"]).shape[:2] != (len(fns), n_cam):
        raise ValueError(f"{path}.k shape does not match fn")
    if np.asarray(meta["w2c"]).shape[:2] != (len(fns), n_cam):
        raise ValueError(f"{path}.w2c shape does not match fn")
    return meta


def verify_sequence(seq_dir: Path) -> dict:
    train_meta = _load_meta(seq_dir / "train_meta.json")
    if (seq_dir / "test_meta.json").exists():
        _load_meta(seq_dir / "test_meta.json")
    init = np.load(seq_dir / "init_pt_cld.npz")["data"]
    if init.ndim != 2 or init.shape[1] != 7 or init.shape[0] == 0:
        raise ValueError(f"init_pt_cld.npz data must be non-empty [N,7], got {init.shape}")
    if not np.isfinite(init).all():
        raise ValueError("init_pt_cld contains non-finite values")
    ims = seq_dir / "ims"
    seg = seq_dir / "seg"
    for t, row in enumerate(train_meta["fn"]):
        for fn in row:
            image = ims / fn
            mask = seg / fn.replace(".jpg", ".png")
            if not image.exists():
                raise FileNotFoundError(image)
            if not mask.exists():
                raise FileNotFoundError(mask)
    return {
        "timesteps": len(train_meta["fn"]),
        "cameras": len(train_meta["fn"][0]),
        "init_points": int(init.shape[0]),
    }


def link_sequence(official_repo: Path, seq_dir: Path, sequence: str, replace_link: bool) -> Path:
    data_dir = official_repo / "data"
    data_dir.mkdir(exist_ok=True)
    dst = data_dir / sequence
    if dst.exists() or dst.is_symlink():
        if not replace_link:
            raise FileExistsError(f"{dst} exists; pass --replace_link for symlink replacement")
        if not dst.is_symlink():
            raise FileExistsError(f"{dst} exists and is not a symlink; refusing to remove")
        dst.unlink()
    dst.symlink_to(seq_dir.resolve(), target_is_directory=True)
    return dst


def run_train(official_repo: Path, sequence: str, exp: str, python: str) -> Path:
    code = f"from train import train; train({sequence!r}, {exp!r})"
    subprocess.run([python, "-c", code], cwd=official_repo, check=True)
    return official_repo / "output" / exp / sequence / "params.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official_repo", required=True, help="Path to JonathonLuiten/Dynamic3DGaussians")
    ap.add_argument("--sequence_dir", required=True, help="Converted data/<sequence> directory")
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--exp", default="head_wrist_ref")
    ap.add_argument("--python", default="python")
    ap.add_argument("--replace_link", action="store_true")
    ap.add_argument("--verify_only", action="store_true")
    args = ap.parse_args()

    official_repo = Path(args.official_repo).expanduser().resolve()
    sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    if not (official_repo / "train.py").exists():
        raise FileNotFoundError(official_repo / "train.py")
    summary = verify_sequence(sequence_dir)
    linked = link_sequence(official_repo, sequence_dir, args.sequence, args.replace_link)
    summary["linked_sequence"] = str(linked)
    if args.verify_only:
        print(json.dumps(summary, indent=2))
        return
    params_path = run_train(official_repo, args.sequence, args.exp, args.python)
    summary["params_path"] = str(params_path)
    summary["params_exists"] = params_path.exists()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
