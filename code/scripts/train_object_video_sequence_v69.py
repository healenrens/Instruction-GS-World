#!/usr/bin/env python3
"""Full-capacity object-video training with explicit state/dynamics stages and resume."""

import argparse
from collections import Counter
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from igsw.distributed import init_torchrun
from igsw.adaptive_gaussian_wm.v69_config import ObjectVideoConfigV69, parameter_inventory
from igsw.adaptive_gaussian_wm.v69_runtime import add_v69_arguments, config_from_args, batch_is_readable, case_metrics_v69, append_case_records
from igsw.adaptive_gaussian_wm.v69_resume_diagnostics import configure_reproducibility_v69, step_inputs_v69, save_step_trace_v69
from igsw.adaptive_gaussian_wm.episode_uniform_sampler_v69 import EpisodeUniformSamplerV69
from igsw.adaptive_gaussian_wm.episode_uniform_sampler_v69 import episode_key
from igsw.adaptive_gaussian_wm.object_video_sequence_dataset_v69 import ObjectVideoSequenceDatasetV69, collate_object_video_v69, move_batch_v69
from igsw.adaptive_gaussian_wm.pretrained_visual_encoder_v69 import PretrainedVisualEncoderV69
from igsw.adaptive_gaussian_wm.object_video_world_model_v69 import ObjectVideoWorldModelV69
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import write_json


def main():
    args = add_v69_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if checkpoint is not None:
        args = argparse.Namespace(**{"deterministic": False, "resume_trace": False, **checkpoint["args"],
                                     "resume": args.resume, "workers": args.workers, "stop_after": args.stop_after})
    numerical_mode = configure_reproducibility_v69(args.deterministic)
    context = init_torchrun()
    device = torch.device(context.device)
    state_checkpoint = torch.load(args.state_checkpoint, map_location="cpu", weights_only=False) if args.stage == "dynamics" and checkpoint is None else None
    inherited = checkpoint if checkpoint is not None else state_checkpoint
    config = ObjectVideoConfigV69(**inherited["config"]) if inherited is not None else config_from_args(args)
    torch.manual_seed(args.seed + context.rank)
    random.seed(args.seed + context.rank)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "dataset.json"
    if context.is_main and checkpoint is None:
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump(json.loads(Path(args.manifest).read_text()), stream)
        source_repository = state_checkpoint["args"]["encoder_repository"] if state_checkpoint is not None else args.encoder_repository
        shutil.copytree(source_repository, out / "encoder_source", ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    if checkpoint is None:
        args.encoder_repository = str(out / "encoder_source")
    if context.distributed:
        dist.barrier()
    dataset = ObjectVideoSequenceDatasetV69(manifest, config, args.seed, "train")
    sampler = EpisodeUniformSamplerV69(dataset, context.rank, context.world_size, args.batch, args.seed)
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                        collate_fn=collate_object_video_v69, pin_memory=True, generator=loader_generator)
    saved_backbone = inherited["perception"] if inherited is not None else None
    perception = PretrainedVisualEncoderV69(config.encoder, args.encoder_repository, args.encoder_weights,
                     args.encoder_frame_batch, history_seconds=config.history_seconds, saved_backbone=saved_backbone).to(device)
    model = ObjectVideoWorldModelV69(config, args.stage).to(device)
    if inherited is not None:
        model.load_state_dict(inherited["model"], strict=True)
    inventory = parameter_inventory({"perception": perception, "object_memory": model.encoder, "EMA_memory": model.target_encoder,
                                     "readout": model.readout, "posterior": model.posterior, "dynamics": model.dynamics})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=.01)
    warmup = max(1, round(args.steps*.05))
    def rate(step):
        if step < warmup:
            return (step+1)/warmup
        progress = min(1., (step-warmup)/max(1, args.steps-warmup))
        return .1 + .45*(1+math.cos(math.pi*progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    step, epoch, cursor = 0, 0, 0
    accum = max(1, math.ceil(args.global_batch/(args.batch*context.world_size)))
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step, epoch, cursor, accum = checkpoint["step"], checkpoint["epoch"], checkpoint["cursor"], checkpoint["grad_accum"]
    if context.is_main:
        write_json(out / "model_inventory.json", {"modules": inventory, "config": config.to_dict(), "perception": perception.provenance})
        episodes = set(episode_key(entry) for entry in dataset.entries)
        sampling = {"unique_episodes": len(episodes), "source_episodes": dict(Counter(key[0] for key in episodes)),
                    "source_clips": dict(Counter(entry["source"] for entry in dataset.entries)),
                    "padded_epoch_visits": len(sampler)*context.world_size,
                    "sampling": "one uniform clip per episode per epoch; DDP tail padding repeats episodes"}
        if checkpoint is None:
            write_json(out / "run.json", {"args": vars(args), "config": config.to_dict(), "world_size": context.world_size,
                       "grad_accum": accum, "effective_batch": accum*args.batch*context.world_size, "sampling": sampling,
                       "numerical_mode": numerical_mode})
        print(json.dumps({"event": "v69_model_inventory", "modules": inventory, "config": config.to_dict()}), flush=True)
    wrapped = DistributedDataParallel(model, device_ids=[context.local_rank], broadcast_buffers=False) if context.distributed else model
    run = None
    if context.is_main and args.wandb_mode != "disabled":
        import wandb
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ.pop("WANDB_RESUME", None)
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=args.wandb_name,
                         group="object-video-sequence-v69", job_type=args.stage, mode=args.wandb_mode,
                         id=checkpoint["wandb_id"] if checkpoint is not None else None,
                         resume="must" if checkpoint is not None and checkpoint["wandb_id"] else None,
                         config={**vars(args), **config.to_dict(), "model_inventory": inventory,
                                 "effective_batch": accum*args.batch*context.world_size, "sampling": sampling, "numerical_mode": numerical_mode})
    model.train()
    perception.eval()
    optimizer.zero_grad(set_to_none=True)
    if checkpoint is not None:
        rng = checkpoint["rng"][context.rank]
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state(rng["cuda"], device)
        random.setstate(rng["python"])
    del checkpoint, inherited, state_checkpoint, saved_backbone

    def save(numbered=False):
        rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device), "python": random.getstate()}
        states = [None]*context.world_size
        if context.distributed:
            dist.all_gather_object(states, rng)
        else:
            states[0] = rng
        if context.is_main:
            value = {"checkpoint_version": 69, "architecture": config.architecture, "args": vars(args), "config": config.to_dict(),
                     "model": model.state_dict(), "perception": perception.backbone.state_dict(), "perception_provenance": perception.provenance,
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": states,
                     "step": step, "epoch": epoch, "cursor": cursor, "grad_accum": accum,
                     "world_size": context.world_size, "wandb_id": run.id if run else None, "numerical_mode": numerical_mode}
            temporary = out / "latest.tmp.pt"
            torch.save(value, temporary)
            temporary.replace(out / "latest.pt")
            if numbered:
                path = out / f"step_{step:07d}.pt"
                torch.save(value, path.with_suffix(".tmp.pt"))
                path.with_suffix(".tmp.pt").replace(path)
            write_json(out / "progress.json", {"step": step, "epoch": epoch, "cursor": cursor, "stage": args.stage, "checkpoint": str(out / "latest.pt")})
            print(f"[object-video-v69] checkpoint={out / 'latest.pt'} step={step}", flush=True)

    pending, accumulated = 0, None
    begin = time.monotonic()
    finish_step = min(args.steps, args.stop_after) if args.stop_after else args.steps
    while step < finish_step:
        sampler.epoch, sampler.start = epoch, cursor*args.batch
        loader_generator.manual_seed(args.seed+epoch)
        start_cursor = cursor
        for relative_batch, cpu_batch in enumerate(loader):
            cursor = start_cursor + relative_batch + 1
            if not batch_is_readable(cpu_batch, context, device):
                continue
            batch = move_batch_v69(cpu_batch, device)
            perception_start, perception_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            perception_start.record()
            fields = perception(batch["rgb"], batch["pixel_valid"], batch["times"], batch["native_hw"])
            perception_end.record()
            trace_inputs = step_inputs_v69(batch, device) if args.resume_trace else None
            pending += 1
            boundary = pending == accum
            synchronization = nullcontext() if boundary or not context.distributed else wrapped.no_sync()
            with synchronization:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = wrapped(fields, batch)
                    loss = output["loss"]/accum
                loss.backward()
            if args.resume_trace:
                save_step_trace_v69(out / f"resume_trace_rank{context.rank:04d}", step+1, pending, trace_inputs, output, model)
            values = torch.stack([output["loss"].detach().float(), *[p.detach().float() for p in output["parts"].values()]])
            accumulated = values if accumulated is None else accumulated+values
            if not boundary:
                continue
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            missing_gradients = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            model.update_target()
            step += 1
            pending = 0
            values, accumulated = accumulated/accum, None
            if step == 1 or step % args.log_every == 0:
                if context.distributed:
                    dist.all_reduce(values)
                    values /= context.world_size
                metrics = dict(zip(["loss", *output["parts"]], values.cpu().tolist()))
                perception_end.synchronize()
                metrics.update(step=step, lr=optimizer.param_groups[0]["lr"], gradient_norm=float(grad),
                               trainable_parameters_without_gradient=missing_gradients,
                               perception_gpu_seconds_last_microbatch=perception_start.elapsed_time(perception_end)/1000,
                               elapsed_seconds=time.monotonic()-begin,
                               peak_memory_gb=torch.cuda.max_memory_allocated(device)/1024**3,
                               teacher_target_fraction=float(batch["teacher"]["valid"].float().mean()),
                               known_observation_fraction=float((batch["teacher"]["observation"] >= 0).float().mean()))
                rows = case_metrics_v69(output, batch, config, args.stage)
                append_case_records(out / f"cases_rank{context.rank:04d}.jsonl", [{"step": step, **row} for row in rows])
                gathered = [None]*context.world_size
                if context.distributed:
                    dist.all_gather_object(gathered, rows)
                else:
                    gathered[0] = rows
                if context.is_main:
                    append_case_records(out / "metrics.jsonl", [metrics])
                    print(json.dumps(metrics), flush=True)
                    if run:
                        table = wandb.Table(columns=["case", "source", "seconds", "valid_points", "transport_points", "all_p50", "all_p90", "transport_p50", "transport_p90", "selection_status"])
                        for rank_rows in gathered:
                            for row in rank_rows:
                                table.add_data(*[row[key] for key in table.columns])
                        run.log({**metrics, "sequence/cases": table}, step=step)
            if step % args.recovery_every == 0 or step % args.save_every == 0:
                save(step % args.save_every == 0)
            if step >= finish_step:
                break
        if step < finish_step:
            epoch, cursor = epoch+1, 0
    save(True)
    if run:
        run.finish()
    if context.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
