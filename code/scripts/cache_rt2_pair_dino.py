"""Cache compact frozen DINOv2 grids for strict-causal frame pairs."""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
import time

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.gpstoken_wm.dino_features import DinoFeatures  # noqa: E402
from igsw.latent_particle_wm.pair_targets import CAUSAL_PAIR_VERSION  # noqa: E402


def projection_matrix(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    if output_dim > input_dim:
        raise ValueError("DINO projection dimension cannot exceed the backbone dimension")
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(input_dim, output_dim, generator=generator)
    return torch.linalg.qr(matrix, mode="reduced").Q


def projected_grid(
    extractor: DinoFeatures,
    projection: torch.Tensor,
    rgb: torch.Tensor,
) -> torch.Tensor:
    grid, _ = extractor.grid(rgb.numpy())
    grid = F.normalize(grid, dim=-1)
    return F.normalize(grid @ projection, dim=-1).to(torch.bfloat16).cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="vit_large_patch14_dinov2.lvd142m")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not (0 <= args.shard < args.nshard):
        raise ValueError("invalid shard")
    for key, value in {
        "HF_HOME": "/mnt/pfs/public/xuhaoming/hf_cache",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": "/mnt/pfs/public/xuhaoming/.cache",
    }.items():
        os.environ.setdefault(key, value)

    files = sorted(glob.glob(os.path.join(args.data, "*.pt")))[args.shard :: args.nshard]
    if args.limit:
        files = files[: args.limit]
    os.makedirs(args.out, exist_ok=True)
    pending = [
        path
        for path in files
        if args.overwrite
        or not os.path.exists(os.path.join(args.out, os.path.basename(path)))
    ]
    if not pending:
        print(f"[pair-dino] DONE pending=0 out={os.path.abspath(args.out)}")
        return

    torch.set_float32_matmul_precision("high")
    extractor = DinoFeatures(args.model, args.image_size).cuda().bfloat16().eval()
    projection = projection_matrix(extractor.embed_dim, args.feature_dim, args.seed).cuda()
    projection_hash = hashlib.sha256(projection.cpu().numpy().tobytes()).hexdigest()
    frame_cache: dict[tuple[str, int], torch.Tensor] = {}
    cached_source = None
    started_at = time.time()
    for index, path in enumerate(pending, 1):
        pair = torch.load(path, map_location="cpu", weights_only=False)
        if pair.get("pair_version") != CAUSAL_PAIR_VERSION:
            raise ValueError(f"causal pair version mismatch: {path}")
        source_name = pair["source_name"]
        if source_name != cached_source:
            frame_cache.clear()
            cached_source = source_name
        start_key = (source_name, int(pair["start"]))
        end_key = (source_name, int(pair["end"]))
        if start_key not in frame_cache:
            frame_cache[start_key] = projected_grid(
                extractor,
                projection,
                pair["rgb_path"][0],
            )
        if end_key not in frame_cache:
            frame_cache[end_key] = projected_grid(
                extractor,
                projection,
                pair["rgb_path"][-1],
            )
        sidecar = {
            "source_name": source_name,
            "start": int(pair["start"]),
            "end": int(pair["end"]),
            "horizon": int(pair["horizon"]),
            "dino0": frame_cache[start_key],
            "dino1": frame_cache[end_key],
            "model": args.model,
            "image_size": args.image_size,
            "feature_dim": args.feature_dim,
            "projection_seed": args.seed,
            "projection_sha256": projection_hash,
        }
        out_path = os.path.join(args.out, os.path.basename(path))
        temporary = f"{out_path}.tmp.{os.getpid()}"
        torch.save(sidecar, temporary)
        os.replace(temporary, out_path)
        if index == 1 or index % 100 == 0 or index == len(pending):
            rate = index / max(time.time() - started_at, 1e-6)
            print(
                f"[pair-dino] shard={args.shard}/{args.nshard} "
                f"{index}/{len(pending)} rate={rate:.2f}/s",
                flush=True,
            )
    print(f"[pair-dino] DONE written={len(pending)} out={os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
