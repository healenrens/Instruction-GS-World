"""Test whether frozen object representations retain recoverable RGB appearance."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
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
from igsw.adaptive_gaussian_wm.readout_runtime import (  # noqa: E402
    object_rgb_from_micro,
)
from igsw.adaptive_gaussian_wm.rgb_supervision import (  # noqa: E402
    current_micro_rgb,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    move_to_device,
    validate_data_model_contract,
)


class RGBProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        layers = (
            [
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 3),
            ]
            if hidden_dim > 0
            else [nn.LayerNorm(input_dim), nn.Linear(input_dim, 3)]
        )
        self.network = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--condition_cache", required=True)
    parser.add_argument("--max_items", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibrated_checkpoint", default="")
    return parser.parse_args()


@torch.no_grad()
def extract_split(
    model,
    config,
    args,
    split: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    dataset = CausalPairFeatureDataset(
        args.data,
        args.dino,
        split,
        max_items=args.max_items,
        condition_cache=args.condition_cache,
        load_rgb=True,
        rgb_short_side=config.rgb_short_side,
        rgb_pad_multiple=config.rgb_pad_multiple,
    )
    validate_data_model_contract(config, dataset, True, True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    collected = {
        "slots": [],
        "features": [],
        "target": [],
        "weight": [],
        "existing": [],
    }
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            history = model.encode_history(batch)
            tokens = history["token_states"][-1]
            slots = history["slot_states"][-1]
            micro_rgb = current_micro_rgb(tokens, batch)
            target = object_rgb_from_micro(
                micro_rgb,
                slots.assignment,
                tokens.activation,
            )
            existing = torch.sigmoid(
                model.object_aggregator.decode_rgb_logits(slots.slots)
            )
        collected["slots"].append(slots.slots.float())
        collected["features"].append(slots.feature.float())
        collected["target"].append(target.float())
        collected["weight"].append(slots.activity.float())
        collected["existing"].append(existing.float())
    return {
        name: torch.cat(values)
        for name, values in collected.items()
    }


def weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    error = (prediction - target).square().mean(dim=-1)
    return (error * weight).sum() / weight.sum().clamp_min(1e-6)


def fit_probe(
    train_input: torch.Tensor,
    train_target: torch.Tensor,
    train_weight: torch.Tensor,
    hidden_dim: int,
    steps: int,
    lr: float,
) -> RGBProbe:
    probe = RGBProbe(train_input.shape[-1], hidden_dim).to(train_input.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        loss = weighted_mse(
            probe(train_input),
            train_target,
            train_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return probe.eval()


@torch.no_grad()
def metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    global_color: torch.Tensor,
) -> dict[str, float]:
    baseline = global_color.expand_as(target)
    weighted = float(weighted_mse(prediction, target, weight))
    baseline_weighted = float(weighted_mse(baseline, target, weight))
    return {
        "weighted_mse": weighted,
        "unweighted_mse": float((prediction - target).square().mean()),
        "global_color_weighted_mse": baseline_weighted,
        "relative_improvement_vs_global": (
            baseline_weighted - weighted
        ) / max(baseline_weighted, 1e-8),
        "prediction_variance": float(prediction.var(unbiased=False)),
        "target_variance": float(target.var(unbiased=False)),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("RGB recoverability probe must run on the remote CUDA host")
    if args.steps <= 0 or args.lr <= 0.0:
        raise ValueError("steps and lr must be positive")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    splits = {
        split: extract_split(model, config, args, split, device)
        for split in ("train", "heldseed", "heldtask")
    }
    train = splits["train"]
    global_color = (
        train["target"] * train["weight"][..., None]
    ).sum(dim=(0, 1), keepdim=True) / train["weight"].sum().clamp_min(1e-6)
    probes = {
        "slot_linear": fit_probe(
            train["slots"],
            train["target"],
            train["weight"],
            0,
            args.steps,
            args.lr,
        ),
        "slot_mlp": fit_probe(
            train["slots"],
            train["target"],
            train["weight"],
            config.object_dim,
            args.steps,
            args.lr,
        ),
        "feature_linear": fit_probe(
            train["features"],
            train["target"],
            train["weight"],
            0,
            args.steps,
            args.lr,
        ),
        "feature_mlp": fit_probe(
            train["features"],
            train["target"],
            train["weight"],
            config.object_dim,
            args.steps,
            args.lr,
        ),
    }
    report_splits = {}
    with torch.no_grad():
        for split, values in splits.items():
            report_splits[split] = {
                "existing_head": metrics(
                    values["existing"],
                    values["target"],
                    values["weight"],
                    global_color,
                ),
                **{
                    name: metrics(
                        probe(
                            values[
                                "slots" if name.startswith("slot_") else "features"
                            ]
                        ),
                        values["target"],
                        values["weight"],
                        global_color,
                    )
                    for name, probe in probes.items()
                },
            }
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "steps": args.steps,
        "lr": args.lr,
        "splits": report_splits,
    }
    if args.calibrated_checkpoint:
        calibrated_path = os.path.abspath(args.calibrated_checkpoint)
        calibrated_state = probes["slot_linear"].network.state_dict()
        model.object_aggregator.rgb_head.load_state_dict(calibrated_state)
        model.target_object_aggregator.rgb_head.load_state_dict(
            calibrated_state
        )
        artifact = {
            "checkpoint_version": checkpoint["checkpoint_version"],
            "parallelism": "rgb_head_calibrated_model_state_dict",
            "model": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "config": checkpoint["config"],
            "args": checkpoint["args"],
            "phase": checkpoint["phase"],
            "phase_step": checkpoint["phase_step"],
            "global_step": checkpoint["global_step"],
            "calibration": {
                "source": os.path.abspath(args.checkpoint),
                "steps": args.steps,
                "lr": args.lr,
                "resumable": False,
            },
        }
        os.makedirs(os.path.dirname(calibrated_path), exist_ok=True)
        temporary = f"{calibrated_path}.tmp.{os.getpid()}"
        torch.save(artifact, temporary)
        os.replace(temporary, calibrated_path)
        report["calibrated_checkpoint"] = calibrated_path
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
