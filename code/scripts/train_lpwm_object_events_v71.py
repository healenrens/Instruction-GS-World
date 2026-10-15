#!/usr/bin/env python3
"""Train upstream LPWM with its ELBO/Adam and the shared episode sampling contract."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igsw.object_event_baselines.lpwm import LPWMConfig, LPWMVideoModel, local_reconstruction_loss, make_config
from igsw.object_event_wm.data import build_video_dataset, collate_object_event_video, write_json
from igsw.object_event_wm.training import EpisodeBatchSampler, device_batch, read_checkpoint, restore_rng, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "manifest", "text-cache", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--partition", default="train")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--snapshot-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", default="")
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--swanlab-project", default="")
    parser.add_argument("--swanlab-workspace", default="")
    args = parser.parse_args()
    saved = read_checkpoint(args.resume) if args.resume else None
    if saved is not None:
        overrides = {name: getattr(args, name) for name in ("out", "resume", "stop_after", "source_revision")}
        args = argparse.Namespace(**{**saved["args"], **overrides})
    rank, local_rank, world = (int(os.environ.get(name, default)) for name, default in
                               (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    random.seed(args.seed+rank)
    np.random.seed(args.seed+rank)
    torch.manual_seed(args.seed+rank)
    torch.cuda.manual_seed_all(args.seed+rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    assets = json.loads(Path(args.assets).read_text())
    dataset = build_video_dataset(args.manifest, partition=args.partition, seed=args.seed,
                                  text_cache=args.text_cache, sampling="language_windows")
    text_metadata = json.loads((Path(args.text_cache)/"index.json").read_text())
    config, changes = make_config(assets, dataset.manifest["sequence"], text_metadata)
    if saved is not None:
        config = LPWMConfig(**saved["model_config"])
    native = config.native
    args.batch = native["batch_size"] if args.batch is None else args.batch
    args.epochs = native["num_epochs"] if args.epochs is None else args.epochs
    args.warmup_epochs = native["warmup_epoch"] if args.warmup_epochs is None else args.warmup_epochs
    global_batch = args.batch*args.accum*world
    args.steps = args.steps or math.ceil(len(dataset)*args.epochs/global_batch)
    model = LPWMVideoModel(config, assets["repository"]).to(device)
    if saved is not None:
        model.load_state_dict(saved["model"])
    criterion = local_reconstruction_loss(assets, native["normalize_rgb"]).to(device).eval()
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True) if world > 1 else model
    optimizer = torch.optim.Adam(model.parameters(), lr=native["lr"], betas=tuple(native["adam_betas"]),
                                 eps=native["adam_eps"], weight_decay=native["weight_decay"])
    scheduler = None
    if native["use_scheduler"]:
        schedule_type = importlib.import_module("utils.util_func").LinearWithWarmupScheduler
        scheduler = schedule_type(optimizer, gamma=native["scheduler_gamma"], verbose=False,
            steps=(max(args.warmup_epochs, 1), max(args.warmup_epochs, 1)+1),
            factors=(1.0, 1.0, native["scheduler_gamma"]))
    step, cursor = 0, 0
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(saved["scheduler"])
        step, cursor = saved["step"], saved["cursor"]
        restore_rng(saved["rng"][rank])
    loader = DataLoader(dataset, batch_sampler=EpisodeBatchSampler(len(dataset), args.batch, rank, world, args.seed, cursor),
                        collate_fn=collate_object_event_video, num_workers=args.workers, pin_memory=True,
                        generator=torch.Generator().manual_seed(args.seed+rank), persistent_workers=args.workers > 0)
    stream, out = iter(loader), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out/"train.jsonl").open("a" if saved is not None else "x") if rank == 0 else None
    tracker, run_id = None, saved.get("swanlab_run_id") if saved else None
    if rank == 0 and args.swanlab_project:
        import swanlab
        tracker = swanlab.init(project=args.swanlab_project, workspace=args.swanlab_workspace or None,
            name=out.name, config=vars(args), log_dir=str(out/"swanlog"), id=run_id, resume="must" if run_id else None)
        run_id = tracker.id
    metadata = {"format": "lpwm_object_events_v71", "baseline_assets": assets,
                "native_config_changes": changes, "loss": "upstream LPWM ELBO with local VGG/LPIPS weights"}
    if rank == 0:
        write_json(out/"run.json", {"args": vars(args), "config": asdict(config), "assets": assets,
                    "inventory": model.inventory(), "native_config_changes": changes,
                    "dataset_episodes": len(dataset), "sequence": dataset.manifest["sequence"],
                    "dataset_contract": dataset.manifest["contract"],
                    "sources": sorted({entry[0]["source"] for entry in dataset.episodes}),
                    "global_batch": global_batch, "world_size": world, "precision": "float32",
                    "loss": metadata["loss"], "explicit_actions_used": False,
                    "warmup_unit": "episode-uniform sample epochs",
                    "criterion_parameters_frozen": all(not p.requires_grad for p in criterion.parameters())})
    model.train()
    completed_epochs = cursor*args.batch*world//len(dataset)
    while step < args.steps and (not args.stop_after or step < args.stop_after):
        started, sums, cases, accepted, data_seconds = time.perf_counter(), {}, [], 0, 0.
        optimizer.zero_grad(set_to_none=True)
        while accepted < args.accum:
            before_data = time.perf_counter()
            cpu = next(stream)
            data_seconds += time.perf_counter()-before_data
            epoch = cursor*args.batch*world/len(dataset)
            cursor += 1
            bad = torch.tensor(int("decode_errors" in cpu), device=device)
            if world > 1:
                dist.all_reduce(bad, op=dist.ReduceOp.MAX)
            if bad.item():
                print(json.dumps({"event": "decode_batch_skipped", "rank": rank, "cursor": cursor,
                                  "errors": cpu.get("decode_errors", [])}), flush=True)
                continue
            batch = device_batch(cpu, device)
            sync = wrapped.no_sync() if world > 1 and accepted+1 < args.accum else nullcontext()
            with sync:
                result = wrapped(batch, criterion, warmup=epoch < args.warmup_epochs)
                (result["loss"]/args.accum).backward()
            for name, value in result["parts"].items():
                scalar = float(value.detach()) if torch.is_tensor(value) else float(value)
                sums[name] = sums.get(name, 0.)+scalar/args.accum
            cases.extend(cpu["case_id"])
            accepted += 1
        gradient_norm = torch.linalg.vector_norm(torch.stack([parameter.grad.detach().float().norm()
                                    for parameter in model.parameters() if parameter.grad is not None]))
        optimizer.step()
        next_epochs = cursor*args.batch*world//len(dataset)
        if scheduler is not None:
            for _ in range(next_epochs-completed_epochs):
                scheduler.step()
        completed_epochs = next_epochs
        step += 1
        values = torch.tensor([sums[key] for key in sorted(sums)], dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(values)
            values /= world
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        row = {**dict(zip(sorted(sums), values.tolist())), "step": step, "cursor": cursor,
               "data_epoch": cursor*args.batch*world/len(dataset), "global_batch": global_batch,
               "grad_norm": float(gradient_norm), "lr": optimizer.param_groups[0]["lr"],
               "seconds": seconds, "data_seconds": data_seconds, "samples_per_second": global_batch/seconds,
               "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30}
        if rank == 0:
            log.write(json.dumps({**row, "case_ids_rank0": cases})+"\n")
            log.flush()
            if step % args.log_every == 0 or step == 1:
                print(json.dumps(row), flush=True)
                if tracker is not None:
                    swanlab.log(row, step=step)
        if step % args.save_every == 0 or step == args.steps or step == args.stop_after:
            save_checkpoint(out, model, optimizer, scheduler, step, cursor, args, rank, world, run_id, metadata)
    if rank == 0:
        write_json(out/"completion.json", {"step": step, "cursor": cursor, "scheduled_steps": args.steps})
        log.close()
        if tracker is not None:
            swanlab.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
