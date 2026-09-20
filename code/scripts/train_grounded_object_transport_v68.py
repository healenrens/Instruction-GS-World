#!/usr/bin/env python3
"""Foreground DDP training on an immutable offline teacher manifest."""

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from igsw.distributed import init_torchrun
from igsw.adaptive_gaussian_wm.grounded_motion_dataset_v68 import GroundedMotionDatasetV68, GroundedMotionSamplerV68, collate_grounded_motion_v68
from igsw.adaptive_gaussian_wm.grounded_object_transport_v68 import GroundedObjectTransportV68, ObjectTransportConfigV68
from igsw.adaptive_gaussian_wm.grounded_appearance_teacher_v68 import GroundedAppearanceTeacherV68
from igsw.adaptive_gaussian_wm.v67_config import ContinuousPredictiveObjectFieldConfigV67
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import write_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stage", choices=("state", "dynamics"), default="state")
    p.add_argument("--state_checkpoint", default="")
    p.add_argument("--resume", default="")
    p.add_argument("--source_revision", default="local-unversioned")
    p.add_argument("--dino_checkpoint", required=True)
    p.add_argument("--siglip_checkpoint", required=True)
    p.add_argument("--dino_frame_batch", type=int, default=96)
    p.add_argument("--siglip_frame_batch", type=int, default=96)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--global_batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=2500)
    p.add_argument("--recovery_every", type=int, default=500)
    p.add_argument("--wandb_project", default="instruct-gs-world")
    p.add_argument("--wandb_entity", default="healenrenss-university-of-chinese-acadmic-and-science")
    p.add_argument("--wandb_name", default="grounded_object_transport_v68")
    p.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    return p.parse_args()


def main():
    args = parse_args()
    context = init_torchrun()
    device = torch.device(context.device)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if checkpoint:
        assert checkpoint["checkpoint_version"] == 68 and checkpoint["world_size"] == context.world_size, "strict resume requires the saved v68 rank topology"
        assert checkpoint["args"]["source_revision"] == args.source_revision, "strict resume requires the saved source revision"
        # Strict resume restores, rather than changes, the saved run specification.
        runtime = {"resume": args.resume, "workers": args.workers}
        args = argparse.Namespace(**{**checkpoint["args"], **runtime})
    torch.manual_seed(args.seed + context.rank)
    random.seed(args.seed + context.rank)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frozen_manifest = out / "dataset.json"
    if context.is_main and checkpoint is None:
        with frozen_manifest.open("x", encoding="utf-8") as stream:
            json.dump(json.loads(Path(args.manifest).read_text()), stream)
    if context.distributed:
        dist.barrier()
    dataset = GroundedMotionDatasetV68(frozen_manifest, args.points, args.seed)
    sampler = GroundedMotionSamplerV68(dataset, context.rank, context.world_size, args.seed, args.batch)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                        collate_fn=collate_grounded_motion_v68, pin_memory=True, generator=generator)
    config = ObjectTransportConfigV68(**checkpoint["config"]) if checkpoint else ObjectTransportConfigV68()
    model = GroundedObjectTransportV68(config, args.stage).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    elif args.stage == "dynamics":
        state = torch.load(args.state_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        model.target_encoder.load_state_dict(model.encoder.state_dict(), strict=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    warmup = max(1, round(args.steps * .05))
    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        ratio = min(1., (step - warmup) / max(1, args.steps - warmup))
        return .1 + .9 * .5 * (1 + math.cos(math.pi * ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    step, epoch, cursor = 0, 0, 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step, epoch, cursor = checkpoint["step"], checkpoint["epoch"], checkpoint["cursor"]
    accum = checkpoint["grad_accum"] if checkpoint else max(1, math.ceil(args.global_batch / (args.batch * context.world_size)))
    if context.is_main and checkpoint is None:
        write_json(out / "run.json", {"args": vars(args), "config": config.to_dict(),
                   "world_size": context.world_size, "grad_accum": accum})
    teacher = None
    if args.stage == "state":
        teacher = GroundedAppearanceTeacherV68(ContinuousPredictiveObjectFieldConfigV67(), device, "bf16",
                    args.dino_checkpoint, args.siglip_checkpoint, args.dino_frame_batch, args.siglip_frame_batch)
    wrapped = DistributedDataParallel(model, device_ids=[context.local_rank], broadcast_buffers=False) if context.distributed else model
    run = None
    if context.is_main and args.wandb_mode != "disabled":
        import wandb
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ.pop("WANDB_RESUME", None)
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=args.wandb_name,
                         group="grounded-object-transport-v68", mode=args.wandb_mode,
                         id=checkpoint["wandb_id"] if checkpoint else None, resume="must" if checkpoint else None,
                         config={**vars(args), **config.to_dict(), "grad_accum": accum, "world_size": context.world_size,
                                 "effective_batch": args.batch * context.world_size * accum})
    if context.distributed:
        dist.barrier()

    def save(kind):
        local_rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device), "python": random.getstate()}
        states = [None] * context.world_size
        if context.distributed:
            dist.all_gather_object(states, local_rng)
        else:
            states[0] = local_rng
        if context.is_main:
            value = {"checkpoint_version": 68, "architecture": "grounded_object_transport_v1", "config": config.to_dict(),
                     "args": vars(args), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "step": step, "epoch": epoch, "cursor": cursor,
                     "rng": states, "world_size": context.world_size, "grad_accum": accum, "wandb_id": run.id if run else None}
            temporary = out / "latest.tmp.pt"
            torch.save(value, temporary)
            temporary.replace(out / "latest.pt")
            if kind == "numbered":
                torch.save(value, out / f"step_{step:07d}.pt")
            write_json(out / "progress.json", {"step": step, "epoch": epoch, "cursor": cursor, "stage": args.stage, "checkpoint": str(out / "latest.pt")})
            print(f"[object-transport-v68] saved step={step} path={out / 'latest.pt'}", flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if checkpoint:
        rng = checkpoint["rng"][context.rank]
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state(rng["cuda"], device)
        random.setstate(rng["python"])
    pending = 0
    accumulated = None
    print(f"[object-transport-v68] stage={args.stage} step={step} examples={len(dataset)} world={context.world_size} batch={args.batch} accum={accum}", flush=True)
    while step < args.steps:
        sampler.epoch = epoch
        generator.manual_seed(args.seed + epoch)
        for batch_index, cpu_batch in enumerate(loader):
            if batch_index < cursor:
                continue
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in cpu_batch.items()}
            targets = teacher(batch) if teacher else None
            pending += 1
            boundary = pending == accum
            sync = nullcontext() if boundary or not context.distributed else wrapped.no_sync()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = wrapped(batch, targets)
                    loss = output["loss"] / accum
                loss.backward()
            part_values = torch.stack([output["loss"].detach().float(), *[v.detach().float() for v in output["parts"].values()]])
            accumulated = part_values if accumulated is None else accumulated + part_values
            cursor = batch_index + 1
            if not boundary:
                continue
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            model.update_target()
            step += 1
            pending = 0
            step_values = accumulated / accum
            accumulated = None
            if step % args.log_every == 0 or step == 1:
                names = ["loss", "gradient_norm", *output["parts"]]
                values = torch.cat((step_values[:1], grad.float().reshape(1), step_values[1:]))
                if context.distributed:
                    dist.all_reduce(values)
                    values /= context.world_size
                metrics = dict(zip(names, values.cpu().tolist()))
                if "epe_px" in output:
                    local_rows = [{"source": batch["source"][i], "case": batch["case_id"][i],
                                   "epe_px": output["epe_px"][i][output["epe_valid"][i]].detach().cpu().tolist()} for i in range(len(batch["source"]))]
                    gathered = [None] * context.world_size
                    if context.distributed:
                        dist.all_gather_object(gathered, local_rows)
                    else:
                        gathered[0] = local_rows
                    values_px = [v for rank_rows in gathered for row in rank_rows for v in row["epe_px"]]
                    if values_px:
                        quantiles = torch.tensor(values_px).quantile(torch.tensor([.5, .9, .95])).tolist()
                        metrics.update(dict(zip(("epe_px_p50", "epe_px_p90", "epe_px_p95"), quantiles)))
                    if context.is_main and run:
                        table = wandb.Table(columns=["source", "case", "epe_px_per_point"])
                        for rank_rows in gathered:
                            for row in rank_rows:
                                table.add_data(row["source"], row["case"], row["epe_px"])
                        run.log({"transport/cases": table}, step=step, commit=False)
                if context.is_main:
                    metrics.update(step=step, lr=optimizer.param_groups[0]["lr"])
                    print(json.dumps(metrics), flush=True)
                    if run:
                        run.log(metrics, step=step)
            if step % args.recovery_every == 0 or step % args.save_every == 0:
                save("numbered" if step % args.save_every == 0 else "recovery")
            if step >= args.steps:
                break
        if step < args.steps:
            epoch += 1
            cursor = 0
    save("numbered")
    if run:
        run.finish()
    if context.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
