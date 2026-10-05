"""Native DCP model/optimizer shards and optimizer-boundary V70 recovery."""

import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_state_dict,
    set_model_state_dict, set_state_dict,
)


def resolve_checkpoint_v70(path):
    """Accept a shard directory, its run directory, or an explicit latest.json."""
    path = Path(path).resolve()
    if path.is_file():
        return path.parent / json.loads(path.read_text())["checkpoint"]
    if (path / "metadata.json").is_file():
        return path
    return path / json.loads((path / "latest.json").read_text())["checkpoint"]


def read_checkpoint_metadata_v70(path):
    return json.loads((resolve_checkpoint_v70(path) / "metadata.json").read_text())


def capture_rng_v70(device, loader_generator, loader_epoch_rng):
    device = torch.device(device)
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "loader": loader_generator.get_state(), "loader_epoch": loader_epoch_rng}


def restore_rng_v70(state, device, loader_generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], torch.device(device))
    loader_generator.set_state(state["loader"])


def _atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint_v70(out, model, optimizer, scheduler, *, step, epoch, cursor,
                        args, config, teacher, tracking, device, loader_generator,
                        loader_epoch_rng, retain=2, snapshot_every=2500):
    """All ranks call after optimizer.step; only complete directories are published."""
    out = Path(out)
    rank, world = dist.get_rank(), dist.get_world_size()
    name = f"step_{step:07d}"
    temporary, destination = out / (name + ".tmp"), out / name
    if rank == 0:
        temporary.mkdir(parents=True, exist_ok=True)
        (temporary / f"rng_world_{world:05d}").mkdir(exist_ok=True)
    dist.barrier()
    rng = capture_rng_v70(device, loader_generator, loader_epoch_rng)
    model_state, optimizer_state = get_state_dict(
        model, optimizer, options=StateDictOptions(cpu_offload=True))
    trainer = {"step": step, "epoch": epoch, "cursor": cursor,
               "scheduler": scheduler.state_dict()}
    dcp.save({"model": model_state, "optimizer": optimizer_state, "trainer": trainer},
             checkpoint_id=temporary / "shards")
    torch.save(rng, temporary / f"rng_world_{world:05d}" / f"rank_{rank:05d}.pt")
    if rank == 0:
        _atomic_json(temporary / "metadata.json", {
            "format": "language_object_effect_v70_dcp", "step": step,
            "args": dict(vars(args)), "config": config, "world_size": world,
            "fixed_teacher": teacher, "model_path": args.model_path,
            "tracking": tracking,
            "snapshot": bool(snapshot_every and step % snapshot_every == 0),
        })
    dist.barrier()
    if rank == 0:
        os.replace(temporary, destination)
    dist.barrier()
    if rank == 0:
        _atomic_json(out / "latest.json", {"checkpoint": name, "step": step})
        rolling = [path for path in sorted(out.glob("step_*"))
                   if path.is_dir() and not path.name.endswith(".tmp")
                   and not json.loads((path / "metadata.json").read_text())["snapshot"]]
        for path in rolling[:-retain]:
            shutil.rmtree(path)
    dist.barrier()
    # Checkpoint collectives must not change the next training draw.
    restore_rng_v70(rng, device, loader_generator)
    return destination


def load_checkpoint_v70(path, model, optimizer, scheduler):
    """Restore training shards; return cursor and rank RNG for late restoration."""
    path = resolve_checkpoint_v70(path)
    # Rank-local recovery is tied to the saved topology. A different launch has
    # no RNG directory and fails through the native file read, not a custom gate.
    rng = torch.load(path / f"rng_world_{dist.get_world_size():05d}" /
                     f"rank_{dist.get_rank():05d}.pt", map_location="cpu", weights_only=False)
    model_state, optimizer_state = get_state_dict(
        model, optimizer, options=StateDictOptions(cpu_offload=True))
    trainer = {"step": 0, "epoch": 0, "cursor": 0,
               "scheduler": scheduler.state_dict()}
    state = {"model": model_state, "optimizer": optimizer_state, "trainer": trainer}
    dcp.load(state, checkpoint_id=path / "shards")
    set_state_dict(model, optimizer, model_state_dict=state["model"],
                   optim_state_dict=state["optimizer"],
                   options=StateDictOptions(cpu_offload=True))
    scheduler.load_state_dict(state["trainer"]["scheduler"])
    return state["trainer"], rng


def load_model_checkpoint_v70(model, path):
    """Model-only inference load, including unwrapped/single-rank and resharded models.

    Construct the model from metadata's model_path/config first. All ranks of an
    initialized evaluation process group call this helper. No optimizer/RNG is read.
    """
    path = resolve_checkpoint_v70(path)
    options = StateDictOptions()
    state = {"model": get_model_state_dict(model, options=options)}
    dcp.load(state, checkpoint_id=path / "shards")
    set_model_state_dict(model, state["model"], options=options)
    return read_checkpoint_metadata_v70(path)


def load_inference_model_v70(checkpointdir, device):
    """Instantiate the saved V70 architecture and load only DCP model shards."""
    from .language_effect_model_v70 import LanguageEffectModelV70

    metadata = read_checkpoint_metadata_v70(checkpointdir)
    config = metadata["config"]
    model = LanguageEffectModelV70(
        metadata["model_path"], mode=config["mode"],
        expert_kwargs=config["expert_kwargs"],
        visual_tokens=config["visual_tokens"], text_tokens=config["text_tokens"],
    ).to(device=torch.device(device))
    # Keep rotary-frequency buffers in FP32 when using BF16 model parameters.
    for parameter in model.parameters():
        parameter.data = parameter.data.to(dtype=torch.bfloat16)
    load_model_checkpoint_v70(model, checkpointdir)
    model.requires_grad_(False).eval()
    return model, metadata
