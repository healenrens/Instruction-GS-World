"""Locate the first representation objective with non-finite allocator gradients."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.pair_dataset import (  # noqa: E402
    CausalPairFeatureDataset,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    move_to_device,
    validate_data_model_contract,
)
from igsw.distributed import init_torchrun  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--condition_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data_world_size", type=int, default=4)
    parser.add_argument("--data_rank", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--anomaly_objective", default="")
    parser.add_argument("--anomaly_batch", type=int, default=-1)
    return parser.parse_args()


def gradient_summary(
    gradients: tuple[torch.Tensor | None, ...],
    names: list[str],
) -> dict:
    nonfinite = []
    maximum = 0.0
    square_sum = 0.0
    for name, gradient in zip(names, gradients, strict=True):
        if gradient is None:
            continue
        finite = torch.isfinite(gradient)
        if not bool(finite.all()):
            nonfinite.append(name)
            continue
        value = gradient.detach().float()
        maximum = max(maximum, float(value.abs().max()))
        square_sum += float(value.square().sum())
    return {
        "finite": not nonfinite,
        "nonfinite_parameters": nonfinite,
        "max_abs": maximum,
        "norm": math.sqrt(square_sum),
    }


def assignment_summary(model, batch, amp_context) -> list[dict[str, float]]:
    with torch.no_grad(), amp_context():
        history = model.encode_history(batch)
    reports = []
    for state in history["token_states"]:
        assignment = state.assignment.float()
        mass = assignment.sum(dim=-1)
        normalized = assignment / mass[..., None].clamp_min(1e-6)
        reports.append(
            {
                "assignment_zero_fraction": float((assignment == 0).float().mean()),
                "mass_min": float(mass.min()),
                "mass_max": float(mass.max()),
                "normalized_max": float(normalized.max()),
                "center_abs_max": float(state.center.float().abs().max()),
            }
        )
    return reports


def output_path(base: str, rank: int, world_size: int) -> str:
    absolute = os.path.abspath(base)
    if world_size == 1:
        return absolute
    stem, extension = os.path.splitext(absolute)
    return f"{stem}.rank{rank}{extension or '.json'}"


def main() -> None:
    args = parse_args()
    context = init_torchrun()
    if not torch.cuda.is_available():
        raise RuntimeError("failure diagnosis must run on the remote CUDA host")
    if context.distributed and context.world_size != args.data_world_size:
        raise ValueError("torchrun world size must equal --data_world_size")
    data_rank = context.rank if context.distributed else args.data_rank
    if not 0 <= data_rank < args.data_world_size:
        raise ValueError("--data_rank is outside --data_world_size")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    saved_args = checkpoint["args"]
    dataset = CausalPairFeatureDataset(
        args.data,
        args.dino,
        "train",
        max_items=int(saved_args["max_train_items"]),
        condition_cache=args.condition_cache,
        load_rgb=True,
        rgb_short_side=config.rgb_short_side,
        rgb_pad_multiple=config.rgb_pad_multiple,
    )
    validate_data_model_contract(config, dataset, True, True)
    sampler = DistributedSampler(
        dataset,
        num_replicas=args.data_world_size,
        rank=data_rank,
        shuffle=True,
        seed=int(saved_args["seed"]),
        drop_last=True,
    )
    sampler.set_epoch(int(saved_args["seed"]) + int(checkpoint["phase_step"]))
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    device = torch.device(context.device)
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.train()
    names = []
    parameters = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and name.startswith("allocator."):
            names.append(name)
            parameters.append(parameter)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else torch.enable_grad
    )

    batches = []
    for batch_index, cpu_batch in enumerate(loader):
        batch = move_to_device(cpu_batch, device)
        with amp_context():
            result = model(batch, phase="representation")
        parts = result["parts"]
        objectives = {
            "allocator": parts["allocator"],
            "feature": parts["feature"],
            "slot": 0.5 * parts["slot"],
            "rgb": config.rgb_loss_weight * parts["rgb_current"],
            "rgb_object": config.rgb_loss_weight * parts["rgb_object"],
        }
        objective_reports = {}
        selected = args.anomaly_objective
        for objective_name, objective in objectives.items():
            if selected and objective_name != selected:
                continue
            use_anomaly = (
                objective_name == selected
                and batch_index == args.anomaly_batch
            )
            if use_anomaly:
                with torch.autograd.detect_anomaly(check_nan=True):
                    gradients = torch.autograd.grad(
                        objective,
                        parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
            else:
                gradients = torch.autograd.grad(
                    objective,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
            objective_reports[objective_name] = {
                "loss": float(objective.detach().float()),
                **gradient_summary(gradients, names),
            }
            del gradients
        batches.append(
            {
                "batch_index": batch_index,
                "objectives": objective_reports,
                "history_assignments": assignment_summary(
                    model,
                    batch,
                    amp_context,
                ),
            }
        )
        del result, parts, objectives, batch

    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "phase_step": int(checkpoint["phase_step"]),
        "data_rank": data_rank,
        "data_world_size": args.data_world_size,
        "amp": args.amp,
        "spatial_precision": {
            "min": float(
                torch.nn.functional.softplus(
                    model.allocator.spatial_precision.detach()
                ).min()
            ),
            "max": float(
                torch.nn.functional.softplus(
                    model.allocator.spatial_precision.detach()
                ).max()
            ),
        },
        "batches": batches,
    }
    path = output_path(args.output, data_rank, args.data_world_size)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "ok", "output": path}, sort_keys=True))
    if context.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
