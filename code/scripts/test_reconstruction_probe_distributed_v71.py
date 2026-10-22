#!/usr/bin/env python3
"""Compare complete fixed-set optimization and resume on two CPU ranks."""

from argparse import Namespace
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igsw.object_event_wm.config import ModelConfig
from igsw.object_event_wm.model import ObjectEventWorldModel
from igsw.object_event_wm.reconstruction_probe import train_probe


class FixtureCodec(torch.nn.Module):
    image_size, grid_size = 16, 2

    def decode(self, tokens):
        pictures = tokens.reshape(-1, 4, 3).transpose(1, 2).reshape(-1, 3, 2, 2)
        return F.interpolate(pictures, (16, 16)).reshape(*tokens.shape[:-2], 3, 16, 16)


def run_case(root, name, steps, resume=""):
    payload = torch.load(root/"inputs.pt", map_location="cpu", weights_only=False)
    model = ObjectEventWorldModel(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    args = Namespace(out=str(root/name), state_source="free", checkpoint="fixture",
        initialization="checkpoint", source_revision="fixture", batch=1, precision="fp32",
        lr=.0002, weight_decay=.01, log_every=2, save_every=2, steps=steps)
    checkpoint = torch.load(root/resume/"latest.pt", map_location="cpu", weights_only=False) if resume else None
    train_probe(model, FixtureCodec(), payload["bank"], args, resume=checkpoint)


def worker(rank, root, name, steps, resume):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{root}/{name}_{steps}.init", rank=rank, world_size=2)
    run_case(Path(root), name, steps, resume)
    dist.destroy_process_group()


def main():
    import json

    torch.set_num_threads(1)
    torch.manual_seed(17)
    config = ModelConfig(feature_dim=3, width=8, heads=2, object_slots=2, local_tokens=2,
                         observation_iterations=1, dynamics_layers=1, effect_dim=2, view_size=16)
    model = ObjectEventWorldModel(config)
    images = torch.rand(4, 3, 16, 16)
    features = F.adaptive_avg_pool2d(images, (2, 2)).flatten(2).transpose(1, 2)
    labels = np.zeros((16, 16), dtype=np.int64)
    labels[3:9, 4:10] = 1
    bank = {"images": images, "features": features,
        "pixel_valid": torch.ones(4, 1, 16, 16, dtype=torch.bool),
        "patch_valid": torch.ones(4, 4, dtype=torch.bool),
        "metadata": [{"case_id": f"fixture_{i}", "scene_seed": i//2, "partition": "fixture",
                      "native_hw": [16, 16], "seconds": float(i)} for i in range(4)],
        "truth": [{"rgb": (image.permute(1, 2, 0).numpy()*255).round().astype(np.uint8),
                   "anchor_labels": labels, "labels": labels, "actors": [1], "robots": []} for image in images]}
    with tempfile.TemporaryDirectory(prefix="v71_ddp_probe_") as directory:
        root = Path(directory)
        torch.save({"model": model.state_dict(), "config": vars(config), "bank": bank}, root/"inputs.pt")
        for name in ("serial", "ddp", "resumed"):
            (root/name).mkdir()
        run_case(root, "serial", 4)
        mp.spawn(worker, args=(str(root), "ddp", 4, ""), nprocs=2, join=True)
        mp.spawn(worker, args=(str(root), "resumed", 2, ""), nprocs=2, join=True)
        mp.spawn(worker, args=(str(root), "resumed", 4, "resumed"), nprocs=2, join=True)
        serial, distributed, recovered = [torch.load(root/name/"latest.pt", map_location="cpu", weights_only=False)
                                           for name in ("serial", "ddp", "resumed")]
        def tensors(saved):
            return {**{f"{module}.{key}": value for module, weights in saved["model"].items()
                       for key, value in weights.items()},
                    **{"free."+key: value for key, value in saved["free_states"].items()}}
        first, second, third = map(tensors, (serial, distributed, recovered))
        serial_difference = max(float((first[key]-second[key]).abs().max()) for key in first)
        resume_difference = max(float((second[key]-third[key]).abs().max()) for key in second)
        assert all(torch.allclose(first[key], second[key], atol=2e-5, rtol=1e-5) for key in first)
        assert resume_difference == 0
        assert distributed["world_size"] == 2 and len(distributed["rank_rng"]) == 2
        report = json.loads((root/"ddp"/"report.json").read_text())
        assert report["effective_batch"] == 4 and report["local_frame_counts"] == [2, 2]
        assert len((root/"ddp"/"train.jsonl").read_text().splitlines()) == 3
        print(json.dumps({"status": "passed", "world_size": 2, "fixed_frames": 4,
            "ddp_vs_serial_max_difference": serial_difference, "resume_max_difference": resume_difference,
            "scope": "CPU/Gloo complete probe; fixture codec, no GPU or pretrained weights"}))


if __name__ == "__main__":
    main()
