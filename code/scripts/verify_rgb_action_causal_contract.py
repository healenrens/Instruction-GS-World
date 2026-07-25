"""Verify that observed future RGB reaches only the training-time action target."""
from __future__ import annotations

import argparse
import json
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
from igsw.adaptive_gaussian_wm.checkpointing import (  # noqa: E402
    CHECKPOINT_VERSION,
)
from igsw.adaptive_gaussian_wm.observed_action import (  # noqa: E402
    rgb_logit_action,
)
from igsw.adaptive_gaussian_wm.pair_dataset import (  # noqa: E402
    CausalPairFeatureDataset,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    move_to_device,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _max_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--condition_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    _require(
        checkpoint.get("checkpoint_version") == CHECKPOINT_VERSION,
        "checkpoint version does not match the active RGB-action contract",
    )
    _require(config.rgb_semantic_action, "checkpoint is not RGB-action")
    dataset = CausalPairFeatureDataset(
        args.data,
        args.dino,
        "train",
        max_items=2,
        condition_cache=args.condition_cache,
        load_rgb=True,
        rgb_short_side=config.rgb_short_side,
        rgb_pad_multiple=config.rgb_pad_multiple,
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    device = torch.device(args.device)
    batch = move_to_device(batch, device)
    swapped = dict(batch)
    swapped["future_rgb"] = batch["future_rgb"].roll(1, dims=0)
    swapped["future_rgb_valid"] = batch["future_rgb_valid"].roll(1, dims=0)

    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    history_mask = torch.zeros(
        2,
        batch["history_features"].shape[1],
        config.object_slots,
        device=device,
        dtype=torch.bool,
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        history = model.encode_history(batch)
        baseline = model(batch, history_mask=history_mask)
        counterfactual = model(swapped, history_mask=history_mask)

    history_difference = _max_difference(
        baseline["online_history_slots"],
        counterfactual["online_history_slots"],
    )
    prior_difference = _max_difference(
        baseline["prior_context"],
        counterfactual["prior_context"],
    )
    rgb_action_difference = _max_difference(
        baseline["posterior_actions"][..., 3:6],
        counterfactual["posterior_actions"][..., 3:6],
    )
    expected = rgb_logit_action(
        baseline["current_object_rgb"],
        baseline["target_future_object_rgb"],
    )
    if config.canonical_activity_gate:
        confidence = (
            history["activity"][:, -1, None]
            * baseline["target_future_activity"]
        ).float().clamp(0.0, 1.0).pow(config.canonical_activity_power)
        expected = expected * confidence[..., None]
    anchor_error = _max_difference(
        baseline["posterior_actions"][..., 3:6],
        expected,
    )
    _require(history_difference < 1e-6, "future RGB changed history encoding")
    _require(prior_difference < 1e-6, "future RGB changed Prior context")
    _require(rgb_action_difference > 1e-4, "future RGB did not change action")
    _require(anchor_error < 1e-6, "posterior RGB action is not exact")

    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_version": checkpoint["checkpoint_version"],
        "canonical_activity_gate": config.canonical_activity_gate,
        "canonical_activity_power": config.canonical_activity_power,
        "canonical_center_gate": config.canonical_center_gate,
        "action_residual_dim": config.action_residual_dim,
        "history_max_difference": history_difference,
        "prior_context_max_difference": prior_difference,
        "rgb_action_max_difference": rgb_action_difference,
        "rgb_action_anchor_max_error": anchor_error,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
