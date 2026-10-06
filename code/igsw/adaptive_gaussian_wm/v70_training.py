"""V70 text/expert training with native FSDP, accumulation and explicit recovery."""

import argparse
from functools import partial
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader

from igsw.distributed import read_torchrun_context
from .episode_uniform_sampler_v69 import EpisodeUniformSamplerV69
from .v70_checkpoint import (
    capture_rng_v70, load_checkpoint_v70, load_model_checkpoint_v70,
    read_checkpoint_metadata_v70, resolve_checkpoint_v70, restore_rng_v70, save_checkpoint_v70,
)
from .v70_tracking import (
    add_tracking_arguments_v70, finish_tracking_v70,
    log_training_scalars_v70, start_tracking_v70,
)


def add_training_arguments_v70(parser):
    parser.add_argument("--manifest", default="")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--teacher_checkpoint", "--state_checkpoint", default="")
    parser.add_argument("--out", default="outputs/language_object_effect_v70")
    parser.add_argument("--mode", choices=("flow", "regression"), default="flow")
    parser.add_argument("--expert_kwargs", type=json.loads, default={})
    parser.add_argument("--visual_tokens", type=int, default=4096)
    parser.add_argument("--text_tokens", type=int, default=512)
    parser.add_argument("--frame_batch", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", "--grad_accum", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr_text", type=float, default=1e-5)
    parser.add_argument("--lr_expert", type=float, default=1e-4)
    parser.add_argument("--warmup_fraction", type=float, default=.05)
    parser.add_argument("--lr_floor", type=float, default=.1)
    parser.add_argument("--weight_decay", type=float, default=.01)
    parser.add_argument("--clip", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--snapshot_every", type=int, default=2500)
    parser.add_argument("--retain", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=10)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", default="")
    initialization.add_argument("--init_from", default="")
    parser.add_argument("--stop_after", type=int, default=0)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace_values", type=int, default=8)
    parser.add_argument("--trace_every", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    return add_tracking_arguments_v70(parser)


def restore_training_arguments_v70(args):
    """Only stopping/tracing controls can override saved training arguments."""
    if not args.resume:
        return args, None
    metadata = read_checkpoint_metadata_v70(args.resume)
    overrides = {name: getattr(args, name) for name in (
        "resume", "stop_after", "trace", "trace_values", "trace_every")}
    return argparse.Namespace(**{**metadata["args"], **overrides}), metadata


def model_config_v70(args):
    return {"mode": args.mode, "expert_kwargs": args.expert_kwargs,
            "visual_tokens": args.visual_tokens, "text_tokens": args.text_tokens,
            "precision": "bf16", "sharding": "FULL_SHARD", "use_orig_params": True,
            "activation_checkpointing": "native_non_reentrant",
            "accumulation": "synchronize_each_microbatch"}


def wrap_model_fsdp_v70(model, device, block_classes=None):
    """FP32 master trainable parameters, BF16 compute, transformer-block shards."""
    device = torch.device(device)
    classes = tuple(model.fsdp_wrap_classes if block_classes is None else block_classes)
    ignored = tuple(model.fsdp_ignored_modules)
    # Disable the model's internal checkpoint loops; native wrappers own recomputation.
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing_disable"):
            module.gradient_checkpointing_disable()
    model.expert.activation_checkpointing = False
    model.float()
    for module in ignored:
        module.to(device=device)
        for parameter in module.parameters():
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
    wrapped = FSDP(
        model, device_id=device, sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True, sync_module_states=True, ignored_modules=ignored,
        auto_wrap_policy=partial(transformer_auto_wrap_policy,
                                 transformer_layer_cls=set(classes)),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16,
            buffer_dtype=None, cast_root_forward_inputs=False),
        limit_all_gathers=True,
    )
    apply_activation_checkpointing(
        wrapped,
        checkpoint_wrapper_fn=partial(checkpoint_wrapper,
                                      checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda module: isinstance(module, classes),
    )
    return wrapped


def make_optimizer_v70(model, args):
    return torch.optim.AdamW([
        {"params": [p for p in model.conditioner.parameters() if p.requires_grad],
         "lr": args.lr_text, "name": "text"},
        {"params": [p for p in model.expert.parameters() if p.requires_grad],
         "lr": args.lr_expert, "name": "expert"},
    ], weight_decay=args.weight_decay)


def make_scheduler_v70(optimizer, args):
    warmup = round(args.steps * args.warmup_fraction)

    def rate(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1., (step - warmup) / max(1, args.steps - warmup - 1))
        return args.lr_floor + (1 - args.lr_floor) * .5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, rate)


def move_batch_v70(batch, device):
    """Keep instruction strings and native uint8 RGB unchanged."""
    return {name: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor)
            else value for name, value in batch.items()}


def _trace_values(tensor, count):
    return tensor.detach().flatten()[:count].float().cpu().tolist()


def _parameter_counts(module):
    return {"total": sum(p.numel() for p in module.parameters()),
            "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad),
            "frozen": sum(p.numel() for p in module.parameters() if not p.requires_grad)}


def _update_probe(module, preferred_suffix):
    candidates = [(name, p) for name, p in module.named_parameters()
                  if p.requires_grad and p.numel() and p.grad is not None]
    name, parameter = sorted(candidates, key=lambda pair: not pair[0].endswith(preferred_suffix))[0]
    return name, parameter, parameter.detach().float().cpu().clone()


def _update_measurement(probe):
    name, parameter, before = probe
    after = parameter.detach().float().cpu()
    delta = after - before
    return {"parameter": name, "local_elements": before.numel(),
            "before_first8": before.flatten()[:8].tolist(),
            "after_first8": after.flatten()[:8].tolist(),
            "delta_first8": delta.flatten()[:8].tolist(),
            "delta_l2": float(delta.norm()), "delta_max_abs": float(delta.abs().max()),
            "changed_elements": int(torch.count_nonzero(delta))}


def train_v70(args, *, model=None, dataset=None, collate_fn=None, history_encoder=None,
              block_classes=None, device=None, trace_callback=None, fsdp=True):
    """Expose the full training loop for parent-owned end-to-end fixtures.

    Explicit injections allow small fixtures. ``fsdp=False`` is an unsharded
    fixture only; the CLI always uses native FSDP. ``stop_after`` is an absolute
    optimizer step and saves a resumable checkpoint before returning. Trace
    JSONL/callback events contain sampler tuples, effect noise, tau and numerics.
    The caller owns process-group shutdown.
    """
    args, metadata = restore_training_arguments_v70(args)
    initialization = None
    if metadata is None and args.init_from:
        args.init_from = str(resolve_checkpoint_v70(args.init_from))
        initialization = read_checkpoint_metadata_v70(args.init_from)
        # Reuse the model and its fixed label space, but start a new optimizer/run.
        for name in ("manifest", "model_path", "teacher_checkpoint", "mode",
                     "expert_kwargs", "visual_tokens", "text_tokens"):
            setattr(args, name, initialization["args"][name])
    context = read_torchrun_context()
    device = torch.device(device or context.device)
    world = metadata["world_size"] if metadata is not None else context.world_size
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    if not dist.is_initialized():
        if device.type == "cuda":
            torch.cuda.set_device(device)
        # Physical ranks come from torchrun. Recovery's topology-specific RNG
        # files are read before DCP; another world size has no training state.
        rendezvous = None if context.distributed else tempfile.TemporaryDirectory(prefix="igsw-v70-")
        init_method = "env://" if context.distributed else f"file://{rendezvous.name}/store"
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo",
                                rank=context.rank, world_size=context.world_size,
                                init_method=init_method,
                                device_id=device if device.type == "cuda" else None)
    rank = dist.get_rank()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    out = Path(args.out).resolve()
    args.out = str(out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"event": "v70_training_config", "args": vars(args),
                          "world_size": world, "launch_world_size": context.world_size,
                          "global_batch": args.batch * args.accum * world,
                          "resumed": metadata is not None}), flush=True)
        if metadata is not None and world != dist.get_world_size():
            print("V70 training resume requires the saved world size; "
                  "different-world loading is model-only inference.", flush=True)
    dist.barrier()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    config = metadata["config"] if metadata is not None else model_config_v70(args)
    if initialization is not None:
        config = initialization["config"]
    if dataset is None:
        from .language_effect_dataset_v70 import LanguageEffectDatasetV70, collate_language_effect_v70
        dataset = LanguageEffectDatasetV70(args.manifest, partition="train")
        collate_fn = collate_language_effect_v70
    if model is None:
        from .language_effect_model_v70 import LanguageEffectModelV70
        model = LanguageEffectModelV70(
            args.model_path, mode=config["mode"], expert_kwargs=config["expert_kwargs"],
            visual_tokens=config["visual_tokens"], text_tokens=config["text_tokens"],
        )
    if history_encoder is None:
        from .frozen_object_teacher_v70 import FrozenHistoryStateV70
        if not args.teacher_checkpoint:
            args.teacher_checkpoint = json.loads(Path(args.manifest).read_text())["teacher_checkpoint"]
        history_encoder = FrozenHistoryStateV70(args.teacher_checkpoint, device, args.frame_batch)
    source_metadata = metadata if metadata is not None else initialization
    teacher = source_metadata["fixed_teacher"] if source_metadata is not None else history_encoder.teacher
    history_encoder.requires_grad_(False).to(device).eval()
    conditioner_counts = _parameter_counts(model.conditioner)
    inventory = {"model": _parameter_counts(model), "conditioner": conditioner_counts,
                 "text": {"total": conditioner_counts["trainable"],
                          "trainable": conditioner_counts["trainable"], "frozen": 0},
                 "expert": _parameter_counts(model.expert),
                 "frozen_history": _parameter_counts(history_encoder),
                 "frozen_vision": [_parameter_counts(module) for module in model.fsdp_ignored_modules]}
    wrapped = wrap_model_fsdp_v70(model, device, block_classes) if fsdp else model.to(device)
    if initialization is not None:
        load_model_checkpoint_v70(wrapped, args.init_from)
        if rank == 0:
            print(json.dumps({"event": "v70_model_warm_start", "checkpoint": args.init_from,
                              "source_step": initialization["step"],
                              "source_world_size": initialization["world_size"],
                              "world_size": world, "optimizer_restored": False}), flush=True)
    optimizer = make_optimizer_v70(model, args)
    optimizer_parameters = {id(p) for group in optimizer.param_groups for p in group["params"]}
    exclusions = {
        "history_optimizer_parameters": sum(id(p) in optimizer_parameters for p in history_encoder.parameters()),
        "vision_optimizer_parameters": sum(id(p) in optimizer_parameters
            for module in model.fsdp_ignored_modules for p in module.parameters()),
        "history_trainable_parameters": inventory["frozen_history"]["trainable"],
        "vision_trainable_parameters": sum(row["trainable"] for row in inventory["frozen_vision"]),
    }
    if rank == 0:
        record = {"args": vars(args), "config": config, "world_size": world,
                  "effective_batch": args.batch * args.accum * world,
                  "inventory": inventory, "optimizer_exclusions": exclusions,
                  "fixed_teacher": teacher,
                  "source_revision": os.environ.get("SOURCE_REVISION", ""),
                  "initialization": ({"checkpoint": args.init_from,
                      "source_step": initialization["step"],
                      "source_world_size": initialization["world_size"]}
                      if initialization is not None else None)}
        with (out / "run.json").open("x" if metadata is None else "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        print(json.dumps({"event": "v70_inventory", "inventory": inventory,
                          "optimizer_exclusions": exclusions}), flush=True)
    scheduler = make_scheduler_v70(optimizer, args)
    wrapped.train()
    sampler = EpisodeUniformSamplerV69(dataset, rank, world, args.batch, args.seed)
    loader_generator = torch.Generator().manual_seed(args.seed + rank)
    loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler,
                        num_workers=args.workers, collate_fn=collate_fn,
                        pin_memory=device.type == "cuda", generator=loader_generator)
    step, epoch, cursor, loader_epoch_rng = 0, 0, 0, None
    checkpoint_path, saved_rng = None, None
    if metadata is not None:
        trainer, saved_rng = load_checkpoint_v70(args.resume, wrapped, optimizer, scheduler)
        step, epoch, cursor = trainer["step"], trainer["epoch"], trainer["cursor"]
        loader_epoch_rng = saved_rng["loader_epoch"]
        checkpoint_path = args.resume
    run, tracking = None, None
    if rank == 0:
        run, tracking = start_tracking_v70(args, {"args": vars(args), "model": config,
                                                "fixed_teacher": teacher},
                                            metadata["tracking"] if metadata is not None else None)
    tracking_values = [tracking]
    dist.broadcast_object_list(tracking_values, src=0)
    tracking = tracking_values[0]
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if saved_rng is not None:
        restore_rng_v70(saved_rng, device, loader_generator)
    iterator, draws = None, None
    last_saved = step if metadata is not None else -1
    final_step = min(args.steps, args.stop_after) if args.stop_after else args.steps
    trace_path = out / f"trace_rank_{rank:05d}.jsonl"
    trace_enabled = args.trace or trace_callback is not None

    def emit(event):
        if args.trace:
            with trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
        if trace_callback is not None:
            trace_callback(event)

    while step < final_step:
        began = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        metrics = {}
        tracing = trace_enabled and (step + 1) % args.trace_every == 0
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        for microbatch in range(args.accum):
            if iterator is None:
                sampler.epoch, sampler.start = epoch, cursor
                # Recreate this epoch's original worker seed without consuming an
                # extra loader-generator draw when restarting mid-epoch.
                current_rng = loader_generator.get_state()
                if loader_epoch_rng is not None:
                    loader_generator.set_state(loader_epoch_rng)
                loader_epoch_rng = loader_generator.get_state()
                iterator = iter(loader)
                if cursor:
                    loader_generator.set_state(current_rng)
                epoch_samples = cursor + len(sampler)
                draws = iter(sampler) if trace_enabled else None
            batch = move_batch_v70(next(iterator), device)
            count = batch["target_mean"].shape[0]
            sample_keys = [next(draws) for _ in range(count)] if trace_enabled else None
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16,
                                                 enabled=device.type == "cuda"):
                history = history_encoder(batch)
            vlm_inputs = model.prepare_inputs(batch)
            target = batch["target_mean"].float()
            noise = torch.randn_like(target) if args.mode == "flow" else torch.zeros_like(target)
            tau = (torch.rand(count, device=device) if args.mode == "flow"
                   else torch.zeros(count, device=device))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = wrapped(vlm_inputs, history, target, batch["query_valid"], noise=noise, tau=tau)
                loss = output["loss"] / args.accum
            loss.backward()
            for name, value in {**output["metrics"], "loss": output["loss"].detach()}.items():
                scalar = torch.as_tensor(value, device=device).detach().float()
                metrics[name] = metrics.get(name, torch.zeros_like(scalar)) + scalar / args.accum
            if tracing:
                emit({"event": "microbatch", "rank": rank, "step": step + 1,
                      "microbatch": microbatch, "epoch": epoch, "cursor": cursor,
                      "samples": sample_keys, "loss": float(output["loss"].detach()),
                      "noise": _trace_values(noise, args.trace_values),
                      "tau": _trace_values(tau, args.trace_values),
                      "target_mean": _trace_values(target, args.trace_values),
                      "history_times": _trace_values(history["times"], args.trace_values)})
            cursor += count
            if cursor == epoch_samples:
                epoch, cursor = epoch + 1, 0
                iterator, draws, loader_epoch_rng = None, None, None
            del output, loss, history, vlm_inputs, target, noise, tau, batch
        grad_norm = (wrapped.clip_grad_norm_(args.clip) if fsdp
                     else torch.nn.utils.clip_grad_norm_(wrapped.parameters(), args.clip))
        probes = ({"text": _update_probe(model.text_blocks, "q_proj.weight"),
                   "expert": _update_probe(model.expert, "effect_output.weight")}
                  if step == 0 else None)
        optimizer.step()
        scheduler.step()
        step += 1
        if probes is not None:
            local_report = {"rank": rank, "step": step,
                            "finite_grad_norm": bool(torch.isfinite(grad_norm)),
                            "grad_norm": float(grad_norm),
                            "updates": {name: _update_measurement(probe) for name, probe in probes.items()}}
            reports = [None] * dist.get_world_size()
            dist.all_gather_object(reports, local_report)
            if rank == 0:
                report = {"step": step, "inventory": inventory,
                          "optimizer_exclusions": exclusions, "ranks": reports}
                (out / "module_update_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(json.dumps({"event": "v70_module_update", **report}), flush=True)
            del probes, local_report, reports
        values = torch.stack([metrics[name] for name in sorted(metrics)])
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= world
        scalars = {f"train/{name}": float(value) for name, value in zip(sorted(metrics), values)}
        elapsed = time.perf_counter() - began
        global_batch = args.batch * args.accum * world
        scalars.update({"train/grad_norm": float(grad_norm), "train/lr_text": learning_rates[0],
                        "train/lr_expert": learning_rates[1], "train/step_seconds": elapsed,
                        "train/global_batch": global_batch, "train/samples_per_second": global_batch / elapsed})
        if device.type == "cuda":
            scalars.update({"train/peak_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
                            "train/peak_memory_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3})
        if tracing:
            emit({"event": "optimizer", "rank": rank, "step": step,
                  "epoch": epoch, "cursor": cursor,
                  "scalars": {name: value for name, value in scalars.items()
                              if name not in ("train/step_seconds", "train/samples_per_second",
                                              "train/peak_memory_allocated_gb", "train/peak_memory_reserved_gb")}})
        if rank == 0 and (step == 1 or step % args.log_every == 0 or step == final_step):
            logging_rng = capture_rng_v70(device, loader_generator, loader_epoch_rng)
            log_training_scalars_v70(run, scalars, step)
            print(json.dumps({"event": "v70_step", "step": step, **scalars}), flush=True)
            restore_rng_v70(logging_rng, device, loader_generator)
        if step % args.checkpoint_every == 0 or (args.snapshot_every and step % args.snapshot_every == 0) or step == final_step:
            checkpoint_path = save_checkpoint_v70(
                out, wrapped, optimizer, scheduler, step=step, epoch=epoch, cursor=cursor,
                args=args, config=config, teacher=teacher, tracking=tracking,
                device=device, loader_generator=loader_generator, loader_epoch_rng=loader_epoch_rng,
                retain=args.retain, snapshot_every=args.snapshot_every,
            )
            last_saved = step
    if last_saved != step:
        checkpoint_path = save_checkpoint_v70(
            out, wrapped, optimizer, scheduler, step=step, epoch=epoch, cursor=cursor,
            args=args, config=config, teacher=teacher, tracking=tracking,
            device=device, loader_generator=loader_generator, loader_epoch_rng=loader_epoch_rng,
            retain=args.retain, snapshot_every=args.snapshot_every,
        )
    finish_tracking_v70(run)
    dist.barrier()
    return {"step": step, "epoch": epoch, "cursor": cursor,
            "checkpoint": str(checkpoint_path), "args": vars(args), "config": config,
            "world_size": world, "tracking": tracking}
