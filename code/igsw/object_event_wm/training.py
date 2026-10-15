"""Foreground single-GPU/DDP training with episode-uniform resumable sampling."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler

from .codec import FrozenVisualCodec
from .config import ModelConfig
from .data import build_video_dataset, collate_object_event_video, write_json
from .model import ObjectEventWorldModel


class EpisodeBatchSampler(Sampler):
    def __init__(self, count, batch, rank, world, seed, start=0):
        self.count, self.batch, self.rank, self.world = count, batch, rank, world
        self.seed, self.start = seed, start

    def __iter__(self):
        micro, epoch, order = self.start, -1, None
        while True:
            items = []
            for offset in range(self.batch):
                visit = micro*self.world*self.batch + self.rank*self.batch + offset
                current_epoch, within = divmod(visit, self.count)
                if epoch != current_epoch:
                    epoch = current_epoch
                    order = torch.randperm(self.count, generator=torch.Generator().manual_seed(self.seed+epoch)).tolist()
                items.append((order[within], epoch, visit))
            yield items
            micro += 1


def device_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def save_checkpoint(out, model, optimizer, scheduler, step, cursor, args, rank, world, run_id, metadata=None):
    local_rng = rng_state()
    states = [None]*world
    if world > 1:
        dist.all_gather_object(states, local_rng)
    else:
        states[0] = local_rng
    if rank == 0:
        payload = {"format": "object_event_wm_v71", "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "step": step, "cursor": cursor, "rng": states, "world_size": world,
                   "args": vars(args), "model_config": asdict(model.config), "swanlab_run_id": run_id}
        payload.update(metadata or {})
        path = out / f"recovery_{step:08d}.pt"
        temporary = path.with_suffix(".partial")
        torch.save(payload, temporary)
        temporary.replace(path)
        write_json(out / "latest.json", {"checkpoint": path.name, "step": step})
        if args.snapshot_every and step % args.snapshot_every == 0:
            snapshot = out / f"step_{step:08d}.pt"
            if not snapshot.exists():
                os.link(path, snapshot)
        for old in sorted(out.glob("recovery_*.pt"))[:-2]:
            old.unlink()
    if world > 1:
        dist.barrier()


def checkpoint_path(path):
    path = Path(path).resolve()
    if path.suffix == ".json":
        path = path.parent / json.loads(path.read_text())["checkpoint"]
    return path


def read_checkpoint(path):
    return torch.load(checkpoint_path(path), map_location="cpu", weights_only=False)


def parse_args(default_stage="joint"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--partition", default="train")
    parser.add_argument("--assets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-config", default="")
    parser.add_argument("--stage", choices=("joint", "prior"), default=default_stage)
    parser.add_argument("--base-checkpoint", default="", help="Fixed world-model checkpoint for prior labels.")
    parser.add_argument("--text-cache", default="")
    parser.add_argument("--prior-layers", type=int, default=4)
    parser.add_argument("--language-dropout", type=float, default=.1)
    parser.add_argument("--effect-mode", choices=("global_step", "object_step", "object_segment"), default="object_step")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--rollout-ramp", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--snapshot-every", type=int, default=2500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--frame-batch", type=int, default=4)
    parser.add_argument("--decode-batch", type=int, default=2)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--resume", default="")
    parser.add_argument("--init-from", default="")
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--swanlab-project", default="")
    parser.add_argument("--swanlab-workspace", default="")
    return parser.parse_args()


def train(args):
    checkpoint = read_checkpoint(args.resume) if args.resume else None
    if checkpoint is not None:
        overrides = {key: getattr(args, key) for key in ("resume", "out", "stop_after", "source_revision")}
        args = argparse.Namespace(**{**vars(args), **checkpoint["args"], **overrides})
        args.stage = "prior" if checkpoint["format"] == "object_event_prior_v71" else "joint"
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
    options = json.loads(Path(args.model_config).read_text()) if args.model_config else {}
    config = ModelConfig(**{**options, "effect_mode": args.effect_mode})
    if checkpoint is not None:
        config = ModelConfig(**checkpoint["model_config"])
    teacher, forward_arguments, checkpoint_metadata = None, {}, {"stage": args.stage}
    if args.stage == "prior":
        from .prior_training import prepare_prior
        model, teacher, checkpoint_metadata = prepare_prior(args, checkpoint, rank, world, device)
        config = model.config
        forward_arguments["teacher"] = teacher
    else:
        model = ObjectEventWorldModel(config).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
    elif args.init_from:
        model.load_state_dict(read_checkpoint(args.init_from)["model"])
    codec = FrozenVisualCodec(args.assets, args.frame_batch, args.decode_batch,
                              load_decoder=args.stage == "joint").to(device)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank]) if world > 1 else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def schedule(step):
        warm = min((step+1)/max(args.warmup, 1), 1.)
        progress = min(max((step-args.warmup)/max(args.steps-args.warmup, 1), 0.), 1.)
        return warm*(.1+.9*.5*(1+math.cos(math.pi*progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    step, cursor = 0, 0
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step, cursor = checkpoint["step"], checkpoint["cursor"]
        restore_rng(checkpoint["rng"][rank])
    dataset = build_video_dataset(args.manifest, partition=args.partition, seed=args.seed,
                                      sampling="language_windows" if args.stage == "prior" else None,
                                      text_cache=args.text_cache if args.stage == "prior" else "")
    sampler = EpisodeBatchSampler(len(dataset), args.batch, rank, world, args.seed, cursor)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                        collate_fn=collate_object_event_video, pin_memory=True,
                        generator=torch.Generator().manual_seed(args.seed+rank),
                        persistent_workers=args.workers > 0)
    stream = iter(loader)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "train.jsonl").open("a" if args.resume else "x") if rank == 0 else None
    tracker = None
    run_id = checkpoint.get("swanlab_run_id") if checkpoint else None
    if args.swanlab_project and rank == 0:
        import swanlab
        tracker = swanlab.init(project=args.swanlab_project, workspace=args.swanlab_workspace or None,
                               name=out.name, config=vars(args), log_dir=str(out / "swanlog"),
                               id=run_id, resume="must" if run_id else None)
        run_id = tracker.id
    if rank == 0:
        write_json(out / "run.json", {"args": vars(args), "model_config": asdict(config),
                                     "world_size": world, "dataset_episodes": len(dataset),
                                     "dataset_contract": dataset.manifest["contract"],
                                     "sequence": dataset.manifest["sequence"],
                                     "time_unit": dataset.manifest.get("time_unit", "seconds"),
                                     "trainable_parameters": sum(p.numel() for p in model.parameters()),
                                     "module_parameters": {name: sum(p.numel() for p in module.parameters())
                                                           for name, module in model.named_children()},
                                     "state_token_budget": config.slots*config.local_tokens,
                                     "effect_scalars_per_step": config.slots*config.effect_dim,
                                     "frozen_codec_parameters": sum(p.numel() for p in codec.parameters()),
                                     "frozen_teacher_parameters": sum(p.numel() for p in teacher.parameters()) if teacher else 0,
                                     "assets": codec.asset_config, "checkpoint_metadata": checkpoint_metadata})
    model.train()
    while step < args.steps and (not args.stop_after or step < args.stop_after):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        sums, accepted, data_seconds, last_cases = {}, 0, 0., []
        while accepted < args.accum:
            before_data = time.perf_counter()
            cpu_batch = next(stream)
            cursor += 1
            data_seconds += time.perf_counter()-before_data
            bad = torch.tensor(int("decode_errors" in cpu_batch), device=device)
            if world > 1:
                dist.all_reduce(bad, op=dist.ReduceOp.MAX)
            if bad.item():
                print(json.dumps({"event": "decode_batch_skipped", "rank": rank, "cursor": cursor,
                                  "errors": cpu_batch.get("decode_errors", [])}), flush=True)
                continue
            batch = device_batch(cpu_batch, device)
            sync = (wrapped.no_sync() if world > 1 and accepted+1 < args.accum else nullcontext())
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                    ramp = min((step+1)/max(args.rollout_ramp, 1), 1.)
                    result = wrapped(batch, codec, rollout_weight=ramp, **forward_arguments)
                (result["loss"]/args.accum).backward()
            for key, value in {"loss": result["loss"], **result["parts"]}.items():
                sums[key] = sums.get(key, 0.)+value.detach().float()/args.accum
            last_cases = batch["case_id"]
            accepted += 1
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        scheduler.step()
        step += 1
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        if step == 1 or step % args.log_every == 0:
            keys = list(sums)
            metrics = torch.stack([sums[key] for key in keys])
            if world > 1:
                dist.all_reduce(metrics)
                metrics /= world
            row = {key: float(value) for key, value in zip(keys, metrics)}
            row.update(step=step, cursor=cursor, grad_norm=float(norm), lr=optimizer.param_groups[0]["lr"],
                       seconds=seconds, data_seconds=data_seconds,
                       global_batch=args.batch*args.accum*world,
                       samples_per_second=args.batch*args.accum*world/seconds,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            if rank == 0:
                log.write(json.dumps({**row, "case_ids_rank0": last_cases}, allow_nan=False)+"\n")
                log.flush()
                print(json.dumps(row, allow_nan=False), flush=True)
                if tracker is not None:
                    import swanlab
                    swanlab.log(row, step=step)
        if step % args.save_every == 0:
            save_checkpoint(out, model, optimizer, scheduler, step, cursor, args, rank, world, run_id, checkpoint_metadata)
    if step % args.save_every != 0:
        save_checkpoint(out, model, optimizer, scheduler, step, cursor, args, rank, world, run_id, checkpoint_metadata)
    if rank == 0:
        write_json(out / "completion.json", {"step": step, "cursor": cursor,
                                            "scheduled_steps": args.steps, "world_size": world})
        log.close()
        if tracker is not None:
            import swanlab
            swanlab.finish()
    if world > 1:
        dist.destroy_process_group()


def main():
    train(parse_args())
