"""Shared runtime helpers for adaptive Gaussian world-model training."""
from __future__ import annotations

import math

import torch
import torch.distributed as dist


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup: int,
    total: int,
    floor_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < warmup:
            return max(step, 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return floor_ratio + (1.0 - floor_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def reduce_metrics(
    parts: dict[str, torch.Tensor],
    world_size: int,
) -> dict[str, float]:
    result = {}
    for name, value in parts.items():
        reduced = value.detach().float()
        if world_size > 1:
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            reduced /= world_size
        result[name] = float(reduced)
    return result


def cuda_memory_metrics(device: torch.device) -> dict[str, float]:
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    return {
        "peak_memory_gb": allocated / 1024**3,
        "peak_reserved_memory_gb": reserved / 1024**3,
        "memory_headroom_fraction": 1.0 - allocated / total,
        "memory_reserved_headroom_fraction": 1.0 - reserved / total,
    }


def validate_data_model_contract(
    config,
    dataset,
    language_enabled: bool,
    rgb_enabled: bool,
) -> None:
    expected_condition_dim = dataset.condition_dim if language_enabled else 0
    mismatches = {}
    if config.feature_dim != dataset.feature_dim:
        mismatches["feature_dim"] = (config.feature_dim, dataset.feature_dim)
    if config.full_dino_features and getattr(dataset, "feature_contract", "") != (
        "backbone_native"
    ):
        mismatches["feature_contract"] = (
            "backbone_native",
            getattr(dataset, "feature_contract", "<missing>"),
        )
    if config.condition_dim != expected_condition_dim:
        mismatches["condition_dim"] = (
            config.condition_dim,
            expected_condition_dim,
        )
    if config.rgb_supervision != rgb_enabled:
        mismatches["rgb_supervision"] = (config.rgb_supervision, rgb_enabled)
    if (
        config.token_conditioned_prior
        and (
            dataset.condition_store is None
            or dataset.condition_store.token_features is None
        )
    ):
        mismatches["token_conditioned_prior"] = (
            True,
            "condition cache has no token features",
        )
    if mismatches:
        raise ValueError(f"checkpoint and runtime data contract differ: {mismatches}")
