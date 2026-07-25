"""Measure whether representation objectives cooperate on shared model parameters."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
from torch.utils.data import DataLoader

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
from igsw.adaptive_gaussian_wm.training import (  # noqa: E402
    representation_pretrain_loss,
)


REPRESENTATION_PREFIXES = (
    "allocator.",
    "object_aggregator.",
    "gaussian_readout.",
)


def group_name(parameter_name: str) -> str:
    return parameter_name.split(".", maxsplit=1)[0]


def gradient_norms(
    gradients: tuple[torch.Tensor | None, ...],
    names: list[str],
) -> dict[str, float]:
    squares: dict[str, float] = {"all": 0.0}
    for name, gradient in zip(names, gradients, strict=True):
        if gradient is None:
            continue
        value = float(gradient.detach().float().square().sum().cpu())
        group = group_name(name)
        squares["all"] += value
        squares[group] = squares.get(group, 0.0) + value
    return {name: math.sqrt(value) for name, value in squares.items()}


def cosine_with_reference(
    reference: list[torch.Tensor | None],
    gradients: tuple[torch.Tensor | None, ...],
    names: list[str],
) -> dict[str, float]:
    statistics: dict[str, list[float]] = {}
    for name, base, gradient in zip(names, reference, gradients, strict=True):
        if base is None or gradient is None:
            continue
        current = gradient.detach().float().cpu()
        group = group_name(name)
        dot = float((base * current).sum())
        base_square = float(base.square().sum())
        current_square = float(current.square().sum())
        for key in ("all", group):
            values = statistics.setdefault(key, [0.0, 0.0, 0.0])
            values[0] += dot
            values[1] += base_square
            values[2] += current_square
    return {
        name: dot / max(math.sqrt(base_square * current_square), 1e-12)
        for name, (dot, base_square, current_square) in statistics.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--condition_cache", default="")
    parser.add_argument("--max_items", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.max_items < args.batch or args.batch <= 0:
        raise ValueError("max_items must be at least one positive batch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    if not config.rgb_supervision:
        raise ValueError("gradient diagnosis requires an RGB-enabled checkpoint")
    dataset = CausalPairFeatureDataset(
        args.data,
        args.dino,
        "train",
        max_items=args.max_items,
        condition_cache=args.condition_cache if config.condition_dim > 0 else "",
        load_rgb=True,
        rgb_short_side=config.rgb_short_side,
        rgb_pad_multiple=config.rgb_pad_multiple,
    )
    validate_data_model_contract(config, dataset, config.condition_dim > 0, True)
    device = torch.device(args.device)
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.train()
    batch = move_to_device(
        next(
            iter(
                DataLoader(
                    dataset,
                    batch_size=args.batch,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=True,
                )
            )
        ),
        device,
    )
    names = []
    parameters = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and name.startswith(REPRESENTATION_PREFIXES):
            names.append(name)
            parameters.append(parameter)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else torch.enable_grad
    )
    with amp_context():
        _, parts = representation_pretrain_loss(model, batch)
    objectives = {
        "feature": parts["feature"],
        "allocator": parts["allocator"],
        "slot": 0.5 * parts["slot"],
        "rgb": config.rgb_loss_weight * parts["rgb_current"],
    }
    feature_gradients = torch.autograd.grad(
        objectives["feature"],
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    feature_cpu = [
        None if gradient is None else gradient.detach().float().cpu()
        for gradient in feature_gradients
    ]
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "batch": args.batch,
        "losses": {
            name: float(value.detach().float().cpu())
            for name, value in objectives.items()
        },
        "gradient_norm": {
            "feature": gradient_norms(feature_gradients, names),
        },
        "cosine_with_feature": {"feature": {"all": 1.0}},
    }
    del feature_gradients
    for name in ("allocator", "slot", "rgb"):
        gradients = torch.autograd.grad(
            objectives[name],
            parameters,
            retain_graph=name != "rgb",
            allow_unused=True,
        )
        report["gradient_norm"][name] = gradient_norms(gradients, names)
        report["cosine_with_feature"][name] = cosine_with_reference(
            feature_cpu,
            gradients,
            names,
        )
        del gradients
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
