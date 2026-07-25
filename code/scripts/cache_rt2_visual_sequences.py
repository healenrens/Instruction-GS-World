"""Cache language-free RoboTwin visual sequences with DINO grids and physical time."""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
import time

import torch
import torch.nn.functional as F
from torchvision.io import encode_jpeg

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.rgb_supervision import resize_and_pad_rgb  # noqa: E402
from igsw.adaptive_gaussian_wm.sequence_contract import (  # noqa: E402
    CONTROL_HZ,
    EXPECTED_FRAME_COUNT,
    SEQUENCE_CACHE_VERSION,
    control_frame_indices,
    preprocess_vggt_rgb,
)
from igsw.gpstoken_wm.dino_features import DinoFeatures  # noqa: E402


def projection_matrix(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    if output_dim > input_dim:
        raise ValueError("DINO projection dimension cannot exceed backbone dimension")
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(input_dim, output_dim, generator=generator)
    return torch.linalg.qr(matrix, mode="reduced").Q


def projected_sequence(
    extractor: DinoFeatures,
    projection: torch.Tensor,
    frames: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    for start in range(0, len(frames), batch_size):
        grid, _ = extractor.grid_batch(
            frames[start : start + batch_size]
        )
        grid = F.normalize(grid, dim=-1)
        outputs.append(
            F.normalize(grid @ projection, dim=-1).to(torch.bfloat16).cpu()
        )
    return torch.cat(outputs)


def cache_rgb(
    frames: torch.Tensor,
    short_side: int,
    pad_multiple: int,
    jpeg_quality: int,
) -> dict:
    resized, valid = resize_and_pad_rgb(frames, short_side, pad_multiple)
    content_height = int(valid[0].any(dim=-1).sum())
    content_width = int(valid[0].any(dim=-2).sum())
    content = resized[:, :, :content_height, :content_width]
    return {
        "jpeg_frames": [
            encode_jpeg(frame.contiguous(), quality=jpeg_quality)
            for frame in content
        ],
        "content_height": content_height,
        "content_width": content_width,
        "padded_height": int(resized.shape[-2]),
        "padded_width": int(resized.shape[-1]),
        "short_side": short_side,
        "pad_multiple": pad_multiple,
        "jpeg_quality": jpeg_quality,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="vit_large_patch14_dinov2.lvd142m")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--projection_seed", type=int, default=17)
    parser.add_argument("--frame_batch", type=int, default=4)
    parser.add_argument("--rgb_short_side", type=int, default=256)
    parser.add_argument("--rgb_pad_multiple", type=int, default=16)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard < args.nshard:
        raise ValueError("invalid shard")
    if args.frame_batch < 1 or not 1 <= args.jpeg_quality <= 100:
        raise ValueError("invalid cache batch or JPEG quality")
    for key, value in {
        "HF_HOME": "/mnt/pfs/public/xuhaoming/hf_cache",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": "/mnt/pfs/public/xuhaoming/.cache",
    }.items():
        os.environ.setdefault(key, value)

    all_files = sorted(glob.glob(os.path.join(args.source, "*.pt")))
    if args.limit:
        all_files = all_files[: args.limit]
    files = all_files[args.shard :: args.nshard]
    if not files:
        raise ValueError("sequence cache shard is empty")
    os.makedirs(args.out, exist_ok=True)
    pending = [
        path
        for path in files
        if args.overwrite
        or not os.path.exists(os.path.join(args.out, os.path.basename(path)))
    ]
    if not pending:
        print(
            f"[visual-sequence] DONE shard={args.shard}/{args.nshard} "
            f"pending=0 out={os.path.abspath(args.out)}"
        )
        return

    torch.set_float32_matmul_precision("high")
    extractor = DinoFeatures(args.model, args.image_size).cuda().bfloat16().eval()
    projection = projection_matrix(
        extractor.embed_dim,
        args.feature_dim,
        args.projection_seed,
    ).cuda()
    projection_hash = hashlib.sha256(
        projection.cpu().numpy().tobytes()
    ).hexdigest()
    started_at = time.time()
    bytes_written = 0
    for index, source_path in enumerate(pending, 1):
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        frames = source.get("gt_rgb")
        if (
            not torch.is_tensor(frames)
            or frames.dtype != torch.uint8
            or frames.ndim != 4
            or frames.shape[-1] != 3
            or len(frames) != EXPECTED_FRAME_COUNT
        ):
            raise ValueError(f"invalid source RGB sequence: {source_path}")
        processed = preprocess_vggt_rgb(frames, args.image_size)
        features = projected_sequence(
            extractor,
            projection,
            processed,
            args.frame_batch,
        )
        controls = control_frame_indices(
            int(source["s"]),
            int(source["win"]),
            len(frames),
        )
        cache = {
            "sequence_version": SEQUENCE_CACHE_VERSION,
            "source_name": os.path.basename(source_path),
            "split": source["split"],
            "source_start": int(source["s"]),
            "source_window": int(source["win"]),
            "control_hz": CONTROL_HZ,
            "frame_control_indices": controls,
            "dino": features,
            "model": args.model,
            "image_size": args.image_size,
            "feature_dim": args.feature_dim,
            "projection_seed": args.projection_seed,
            "projection_sha256": projection_hash,
            "visual_preprocess": "spatracker_vggt_crop_width518",
            "rgb": cache_rgb(
                processed,
                args.rgb_short_side,
                args.rgb_pad_multiple,
                args.jpeg_quality,
            ),
        }
        forbidden = {
            "instruction",
            "condition_feature",
            "condition_tokens",
            "task",
            "task_index",
            "language",
        }
        if forbidden.intersection(cache):
            raise ValueError("semantic fields are forbidden in visual sequence caches")
        out_path = os.path.join(args.out, os.path.basename(source_path))
        temporary = f"{out_path}.tmp.{os.getpid()}"
        torch.save(cache, temporary)
        os.replace(temporary, out_path)
        bytes_written += os.path.getsize(out_path)
        if index == 1 or index % 20 == 0 or index == len(pending):
            elapsed = max(time.time() - started_at, 1e-6)
            print(
                f"[visual-sequence] shard={args.shard}/{args.nshard} "
                f"{index}/{len(pending)} clips={index / elapsed:.2f}/s "
                f"frames={len(frames) * index / elapsed:.2f}/s "
                f"written_gib={bytes_written / 2**30:.2f}",
                flush=True,
            )
    print(
        f"[visual-sequence] DONE shard={args.shard}/{args.nshard} "
        f"written={len(pending)} out={os.path.abspath(args.out)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
