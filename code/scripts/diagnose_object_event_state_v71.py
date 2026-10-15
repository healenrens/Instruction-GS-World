#!/usr/bin/env python3
"""Record per-frame numerical scales after a real end-to-end instability report."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igsw.object_event_wm.codec import FrozenVisualCodec, letterbox_video
from igsw.object_event_wm.config import ModelConfig
from igsw.object_event_wm.data import ObjectEventVideoDataset, collate_object_event_video, write_json
from igsw.object_event_wm.model import ObjectEventWorldModel
from igsw.object_event_wm.training import EpisodeBatchSampler, device_batch, read_checkpoint


def scale(value):
    value = value.detach().float()
    return {"rms": float(value.square().mean().sqrt()), "abs_max": float(value.abs().max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    saved = read_checkpoint(args.checkpoint) if args.checkpoint else None
    config = ModelConfig(**saved["model_config"]) if saved else ModelConfig()
    model = ObjectEventWorldModel(config).cuda().eval()
    if saved:
        model.load_state_dict(saved["model"])
    codec = FrozenVisualCodec(args.assets).cuda().eval()
    dataset = ObjectEventVideoDataset(args.manifest, seed=args.seed)
    item = next(iter(EpisodeBatchSampler(len(dataset), 1, 0, 1, args.seed)))[0]
    sample = dataset[item]
    batch = device_batch(collate_object_event_video([sample]), torch.device("cuda"))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        images, mask = letterbox_video(batch, model.view_size(codec))
        features = codec.encode(images)
        states = model.observe(features, codec.patch_valid(mask), batch["times"])
        th = batch["history_frames"]
        previous = features.new_zeros((1, config.slots, config.effect_dim))
        rows = []
        for index, state in enumerate(states):
            row = {"frame": index, "seconds": float(state.time[0]),
                   "encoder_features": scale(features[:, index]), "state_tokens": scale(state.tokens),
                   "centers": scale(state.centers)}
            if index >= th:
                effect = model.posterior(states[index-1], state, previous,
                                         first=index == th, deterministic=True)
                previous = effect["value"]
                row.update(posterior_mean=scale(effect["mean"]), effect_kl=float(effect["kl"].mean()))
            rows.append(row)
    write_json(args.output, {"case_id": sample["case_id"], "source": sample["source"],
                            "checkpoint": args.checkpoint or "random_initialization", "frames": rows,
                            "purpose": "numerical scale localization, not object quality"})
    print(f"numerical_trace={args.output}", flush=True)


if __name__ == "__main__":
    main()
