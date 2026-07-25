"""Convert calibrated head+wrist videos into Dynamic3DGaussians data layout.

This script only packages verified observations for the official D3DGS/4-LEGS
training code. It does not estimate camera poses, synthesize views, initialize
geometry, or train a Gaussian model.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


STREAMS = ("head", "wrist")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _matrix(value, name: str, shape: tuple[int, int]) -> list[list[float]]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr.tolist()


def _stream_cfg(manifest: dict, frame: dict, stream: str, key: str, t: int):
    frame_stream = frame.get(stream, {})
    if key in frame_stream:
        return frame_stream[key]
    cameras = manifest.get("cameras", {})
    camera = cameras.get(stream, {})
    if key in camera:
        return camera[key]
    raise KeyError(f"missing {stream}.{key} for frame {t}")


def _cam_id(manifest: dict, frame: dict, stream: str, default: int) -> int:
    value = frame.get(stream, {}).get("cam_id")
    if value is None:
        value = manifest.get("cameras", {}).get(stream, {}).get("cam_id", default)
    return int(value)


def _frame_path(manifest_path: Path, frame: dict, stream: str, key: str, t: int) -> Path:
    frame_stream = frame.get(stream, {})
    if key not in frame_stream:
        raise KeyError(f"missing {stream}.{key} path for frame {t}")
    path = _resolve(manifest_path.parent, frame_stream[key])
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_init_cloud(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)["data"]
    elif path.suffix == ".npy":
        data = np.load(path)
    else:
        data = np.loadtxt(path)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 7:
        raise ValueError(f"init cloud must be [N,7] xyz/rgb/seg, got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("init cloud contains non-finite values")
    if data.shape[0] == 0:
        raise ValueError("init cloud is empty")
    rgb = data[:, 3:6]
    seg = data[:, 6]
    if rgb.min() < 0.0 or rgb.max() > 1.0:
        raise ValueError("init cloud RGB must be normalized to [0,1]")
    if seg.min() < 0.0 or seg.max() > 1.0:
        raise ValueError("init cloud foreground seg must be in [0,1]")
    return data


def _write_rgb(src: Path, dst: Path, expected_size: tuple[int, int] | None, quality: int) -> tuple[int, int]:
    img = Image.open(src).convert("RGB")
    size = img.size
    if expected_size is not None and size != expected_size:
        raise ValueError(f"{src} has size {size}, expected {expected_size}")
    img.save(dst, quality=quality, subsampling=1)
    return size


def _write_seg(src: Path, dst: Path, expected_size: tuple[int, int]) -> None:
    img = Image.open(src).convert("L")
    if img.size != expected_size:
        raise ValueError(f"{src} has size {img.size}, expected {expected_size}")
    arr = (np.asarray(img) > 0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(dst)


def _prepare_out(seq_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    if seq_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{seq_dir} exists; pass --overwrite to replace")
        shutil.rmtree(seq_dir)
    ims = seq_dir / "ims"
    seg = seq_dir / "seg"
    ims.mkdir(parents=True)
    seg.mkdir(parents=True)
    return ims, seg


def _build_meta(manifest_path: Path, manifest: dict, frames: list[dict], ims_dir: Path,
                seg_dir: Path, jpeg_quality: int, name_prefix: str = "") -> dict:
    meta = {"w": None, "h": None, "fn": [], "k": [], "w2c": [], "cam_id": []}
    expected_size: tuple[int, int] | None = None
    for t, frame in enumerate(frames):
        fns, ks, w2cs, cam_ids = [], [], [], []
        for c, stream in enumerate(STREAMS):
            rgb_src = _frame_path(manifest_path, frame, stream, "rgb", t)
            seg_src = _frame_path(manifest_path, frame, stream, "seg", t)
            stem = f"{name_prefix}{stream}_{t:06d}"
            rgb_name = f"{stem}.jpg"
            seg_name = f"{stem}.png"
            size = _write_rgb(rgb_src, ims_dir / rgb_name, expected_size, jpeg_quality)
            if expected_size is None:
                expected_size = size
                meta["w"], meta["h"] = int(size[0]), int(size[1])
            _write_seg(seg_src, seg_dir / seg_name, size)
            fns.append(rgb_name)
            ks.append(_matrix(_stream_cfg(manifest, frame, stream, "K", t), f"{stream}.K[{t}]", (3, 3)))
            w2cs.append(_matrix(_stream_cfg(manifest, frame, stream, "w2c", t), f"{stream}.w2c[{t}]", (4, 4)))
            cam_ids.append(_cam_id(manifest, frame, stream, c))
        meta["fn"].append(fns)
        meta["k"].append(ks)
        meta["w2c"].append(w2cs)
        meta["cam_id"].append(cam_ids)
    return meta


def convert(manifest_path: Path, out_root: Path, sequence: str, overwrite: bool, jpeg_quality: int) -> dict:
    manifest = json.loads(manifest_path.read_text())
    frames = manifest.get("frames", [])
    if not frames:
        raise ValueError("manifest.frames is empty")

    seq_dir = out_root / sequence
    ims_dir, seg_dir = _prepare_out(seq_dir, overwrite)
    init_path = _resolve(manifest_path.parent, manifest["init_pt_cld"])
    init_cloud = _load_init_cloud(init_path)

    meta = _build_meta(manifest_path, manifest, frames, ims_dir, seg_dir, jpeg_quality)
    test_source = "train_copy"
    test_frames = manifest.get("test_frames")
    test_meta = meta
    if test_frames:
        test_source = "manifest.test_frames"
        test_meta = _build_meta(manifest_path, manifest, test_frames, ims_dir, seg_dir, jpeg_quality, "test_")

    np.savez(seq_dir / "init_pt_cld.npz", data=init_cloud)
    (seq_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    (seq_dir / "test_meta.json").write_text(json.dumps(test_meta, indent=2))
    summary = {
        "sequence": sequence,
        "timesteps": len(frames),
        "test_timesteps": len(test_meta["fn"]),
        "test_meta_source": test_source,
        "streams": list(STREAMS),
        "observations": len(frames) * len(STREAMS),
        "width": meta["w"],
        "height": meta["h"],
        "init_points": int(init_cloud.shape[0]),
        "layout": "Dynamic3DGaussians data/<sequence>",
    }
    (seq_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSON manifest with head/wrist rgb, seg, K, w2c and init_pt_cld")
    ap.add_argument("--out_root", required=True, help="Directory that will contain data/<sequence>-style folder")
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    args = ap.parse_args()

    summary = convert(
        Path(args.manifest).expanduser().resolve(),
        Path(args.out_root).expanduser().resolve(),
        args.sequence,
        args.overwrite,
        args.jpeg_quality,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
