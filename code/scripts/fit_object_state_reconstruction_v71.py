#!/usr/bin/env python3
"""Fit a small fixed set of train frames using only the V71 observed-state path."""

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igsw.object_event_wm.codec import FrozenVisualCodec
from igsw.object_event_wm.config import ModelConfig
from igsw.object_event_wm.data import build_video_dataset
from igsw.object_event_wm.model import ObjectEventWorldModel
from igsw.object_event_wm.reconstruction_probe import load_encoder_comparison, prepare_frames, train_probe
from igsw.object_event_wm.training import checkpoint_path, read_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Fixed failed joint checkpoint; its config is also used for fresh init.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--initialization", choices=("checkpoint", "fresh"), default="checkpoint")
    parser.add_argument("--state-source", choices=("encoder", "free"), default="encoder")
    parser.add_argument("--reference-run", default="", help="Reuse this probe's exact frames and show its fitted encoder reconstruction.")
    parser.add_argument("--resume", default="", help="Explicit probe checkpoint, not the original joint run.")
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--frame-seconds", default="0,2,5,10")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--frame-batch", type=int, default=4)
    parser.add_argument("--decode-batch", type=int, default=2)
    parser.add_argument("--swanlab-project", default=os.environ.get("PROBE_SWANLAB_PROJECT", os.environ.get("SWANLAB_PROJECT", "")))
    parser.add_argument("--swanlab-workspace", default=os.environ.get("PROBE_SWANLAB_WORKSPACE", os.environ.get("SWANLAB_WORKSPACE", "")))
    parser.add_argument("--source-revision", default=os.environ.get("SOURCE_REVISION", ""))
    args = parser.parse_args()
    # Pass project/workspace through init(), not the SDK's nested JSON environment fields.
    os.environ.pop("SWANLAB_PROJECT", None)
    os.environ.pop("SWANLAB_WORKSPACE", None)
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if resume is not None:
        saved_args = resume["args"]
        args.state_source = resume.get("state_source", "encoder")
        args.reference_run = saved_args.get("reference_run", "")
        for name in ("checkpoint", "initialization", "manifest", "assets", "scenes", "frame_seconds",
                     "batch", "precision", "lr", "weight_decay", "seed", "frame_batch", "decode_batch"):
            setattr(args, name, saved_args[name])
    args.checkpoint = str(checkpoint_path(args.checkpoint))
    saved = resume if resume is not None else read_checkpoint(args.checkpoint)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", local_rank)
    model = ObjectEventWorldModel(ModelConfig(**saved["model_config"])).to(device)
    if resume is None and args.initialization == "checkpoint":
        model.load_state_dict(saved["model"])
    del saved
    codec = FrozenVisualCodec(args.assets, args.frame_batch, args.decode_batch).to(device).eval()
    if resume is not None:
        bank = torch.load(out/"fixed_frames.pt", map_location="cpu", weights_only=False)
    elif rank == 0:
        if args.reference_run:
            bank = torch.load(Path(args.reference_run)/"fixed_frames.pt", map_location="cpu", weights_only=False)
        else:
            dataset = build_video_dataset(args.manifest, partition="train", seed=args.seed)
            bank = prepare_frames(dataset, codec, model.view_size(codec), args.scenes,
                                  [float(value) for value in args.frame_seconds.split(",")], device, args.precision)
        # The fixed set is the experiment input; an existing run is resumed explicitly.
        with (out/"selection.json").open("x") as stream:
            json.dump({"args": vars(args), "frames": bank["metadata"], "skipped": bank["skipped"]},
                      stream, ensure_ascii=False, indent=2)
        torch.save(bank, out/"fixed_frames.pt")
    if world > 1:
        dist.barrier()
        if rank != 0:
            bank = torch.load(out/"fixed_frames.pt", map_location="cpu", weights_only=False)
    comparison = (load_encoder_comparison(model, codec, bank, Path(args.reference_run)/"latest.pt",
                                          args.batch, args.precision)
                  if args.reference_run and resume is None and rank == 0 else None)
    tracker = None
    if args.swanlab_project and rank == 0:
        import swanlab
        run_id = resume.get("swanlab_run_id") if resume else None
        tracker = swanlab.init(project=args.swanlab_project, workspace=args.swanlab_workspace or None,
            name=out.name, config=vars(args), log_dir=str(out/"swanlog"),
            id=run_id, resume="must" if run_id else None)
    if rank == 0:
        print(json.dumps({"event": "reconstruction_probe_start", "frames": len(bank["images"]),
            "micro_batch": args.batch, "steps": args.steps, "initialization": args.initialization,
            "state_source": args.state_source, "reference_run": args.reference_run, "world_size": world,
            "base_checkpoint": args.checkpoint, "out": str(out)}), flush=True)
    report = train_probe(model, codec, bank, args, tracker, resume, comparison)
    if rank == 0:
        print(json.dumps({"event": "reconstruction_probe_complete", "step": report["step"],
            "before": report["before"], "after": report["after"], "review": str(out/"index.html")}), flush=True)
    if tracker is not None:
        import swanlab
        swanlab.finish()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
