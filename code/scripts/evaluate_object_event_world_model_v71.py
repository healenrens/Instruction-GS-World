#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igsw.object_event_wm.codec import FrozenVisualCodec
from igsw.object_event_wm.config import ModelConfig
from igsw.object_event_wm.data import ObjectEventVideoDataset
from igsw.object_event_wm.evaluation import evaluate
from igsw.object_event_wm.model import ObjectEventWorldModel
from igsw.object_event_wm.training import read_checkpoint
from igsw.object_event_wm.prior_training import load_prior_model
from igsw.object_event_wm.text import TextFeatureStore


def main():
    parser = argparse.ArgumentParser(description="Paired posterior reconstruction and optional conditional generation.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--prior-checkpoint", default="")
    parser.add_argument("--text-cache", default="")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--partition", default="development")
    parser.add_argument("--items-per-source", type=int, default=8)
    parser.add_argument("--videos-per-source", type=int, default=1)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    prior, text_store = None, None
    if args.prior_checkpoint:
        model, prior, saved = load_prior_model(args.prior_checkpoint, torch.device("cuda"))
        text_cache = args.text_cache or saved["args"]["text_cache"]
        text_store = TextFeatureStore(text_cache) if text_cache else None
    else:
        saved = read_checkpoint(args.checkpoint)
        model = ObjectEventWorldModel(ModelConfig(**saved["model_config"])).cuda().eval()
        model.load_state_dict(saved["model"])
    codec = FrozenVisualCodec(args.assets).cuda().eval()
    data = ObjectEventVideoDataset(args.manifest, partition=args.partition,
                                  sampling="language_windows" if prior is not None else None)
    evaluate(model, codec, data, args.out, args.items_per_source, args.precision, args.videos_per_source,
             prior=prior, text_store=text_store, samples=args.samples)


if __name__ == "__main__":
    main()
