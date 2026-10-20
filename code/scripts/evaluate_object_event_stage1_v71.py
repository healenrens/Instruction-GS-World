#!/usr/bin/env python3
"""Single-GPU held diagnosis for the V71 joint stage, without a language prior."""

import argparse
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igsw.object_event_wm.codec import FrozenVisualCodec
from igsw.object_event_wm.config import ModelConfig
from igsw.object_event_wm.data import build_video_dataset
from igsw.object_event_wm.model import ObjectEventWorldModel
from igsw.object_event_wm.stage1_evaluation import evaluate_stage1
from igsw.object_event_wm.training import checkpoint_path, read_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--partition", default="development")
    parser.add_argument("--items", type=int, default=200)
    parser.add_argument("--video-cases", type=int, default=12)
    parser.add_argument("--frame-batch", type=int, default=4)
    parser.add_argument("--decode-batch", type=int, default=2)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--resume", action="store_true", help="Reuse completed cases in the same output directory.")
    args = parser.parse_args()
    path = checkpoint_path(args.checkpoint)
    saved = read_checkpoint(path)
    config = ModelConfig(**saved["model_config"])
    model = ObjectEventWorldModel(config).cuda().eval()
    model.load_state_dict(saved["model"])
    info = {"checkpoint": str(path), "checkpoint_step": saved["step"],
            "training_revision": saved["args"].get("source_revision"),
            "evaluation_revision": os.environ.get("SOURCE_REVISION", ""),
            "model_config": saved["model_config"], "partition": args.partition}
    del saved
    codec = FrozenVisualCodec(args.assets, args.frame_batch, args.decode_batch).cuda().eval()
    dataset = build_video_dataset(args.manifest, partition=args.partition)
    report = evaluate_stage1(model, codec, dataset, args.out, info, args.items, args.video_cases, args.precision, args.resume)
    print(f'[stage1-held-v71] cases={report["cases"]} scenes={report["scenes"]} report={args.out}/report.json', flush=True)


if __name__ == "__main__":
    main()
