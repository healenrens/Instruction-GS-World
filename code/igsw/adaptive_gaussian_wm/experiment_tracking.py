"""Rank-zero Weights & Biases tracking for adaptive WM training."""
from __future__ import annotations

import argparse
import os


_PROGRESS_KEYS = {
    "phase",
    "phase_step",
    "global_step",
    "sampler_epoch",
    "data_epoch",
    "phase_samples_seen",
    "samples_seen",
}
_RUNTIME_KEYS = {
    "lr",
    "grad_norm",
    "steps_per_second",
    "samples_per_second",
    "wall_time_seconds",
    "micro_batch",
    "grad_accum",
    "world_size",
    "effective_batch",
    "updates_per_epoch",
}
_SYSTEM_KEYS = {
    "peak_memory_gb",
    "peak_reserved_memory_gb",
    "memory_headroom_fraction",
    "memory_reserved_headroom_fraction",
}


def add_wandb_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wandb_mode",
        choices=("disabled", "online", "offline"),
        default="disabled",
    )
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_tags", default="")
    parser.add_argument("--wandb_run_id", default="")
    parser.add_argument("--wandb_dir", default="")


def validate_wandb_arguments(args: argparse.Namespace) -> None:
    if args.wandb_mode == "disabled":
        return
    if not args.wandb_project:
        raise ValueError("W&B tracking requires --wandb_project")
    if not args.wandb_dir:
        raise ValueError("W&B tracking requires --wandb_dir")


def _read_run_id(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as handle:
        return handle.read().strip()


def _write_run_id(path: str, run_id: str) -> None:
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(run_id + "\n")
    os.replace(temporary, path)


def _metric_name(name: str) -> str:
    if name in _PROGRESS_KEYS:
        return f"progress/{name}"
    if name in _RUNTIME_KEYS or name.startswith("lr_"):
        return f"runtime/{name}"
    if name in _SYSTEM_KEYS:
        return f"system/{name}"
    return f"train/{name}"


class WandbTracker:
    def __init__(self, run) -> None:
        self.run = run

    def log(self, record: dict) -> None:
        step = int(record["global_step"])
        payload = {
            _metric_name(name): value
            for name, value in record.items()
        }
        self.run.log(payload, step=step)

    def finish(self) -> None:
        self.run.finish()


def init_wandb_tracker(
    args: argparse.Namespace,
    context,
    tracking_config: dict,
) -> WandbTracker | None:
    if args.wandb_mode == "disabled" or not context.is_main:
        return None

    import wandb

    os.makedirs(args.wandb_dir, exist_ok=True)
    run_id_path = os.path.join(args.out, "wandb_run_id.txt")
    saved_run_id = _read_run_id(run_id_path)
    requested_run_id = args.wandb_run_id.strip()
    if saved_run_id and requested_run_id and saved_run_id != requested_run_id:
        raise ValueError("requested W&B run id differs from output metadata")
    run_id = requested_run_id or saved_run_id
    if args.resume and not run_id:
        raise ValueError("checkpoint resume requires an existing W&B run id")
    if not run_id:
        run_id = wandb.util.generate_id()

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name or None,
        group=args.wandb_group or None,
        tags=tags or None,
        id=run_id,
        resume="allow" if saved_run_id or args.resume else "never",
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        config=tracking_config,
    )
    if run is None:
        raise RuntimeError("wandb.init returned no run")
    _write_run_id(run_id_path, run.id)
    return WandbTracker(run)
