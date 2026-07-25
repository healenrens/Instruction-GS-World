"""Evaluate a deployable language-free image-goal Prior on visual sequences."""
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
from igsw.adaptive_gaussian_wm.goal_conditioning import (  # noqa: E402
    GOAL_CONDITION_VERSION,
    ObjectGoalConditioner,
)
from igsw.adaptive_gaussian_wm.goal_eval_bank import (  # noqa: E402
    build_goal_bank,
)
from igsw.adaptive_gaussian_wm.goal_prior_checkpointing import (  # noqa: E402
    GOAL_PRIOR_CHECKPOINT_VERSION,
    GOAL_PRIOR_CHECKPOINT_KIND,
    file_sha256,
    load_goal_prior_delta,
)
from igsw.adaptive_gaussian_wm.goal_prior_evaluation import (  # noqa: E402
    evaluate_goal_prior,
)
from igsw.adaptive_gaussian_wm.goal_time_evaluation import (  # noqa: E402
    evaluate_goal_time_counterfactual,
)
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    validate_data_model_contract,
)


def _load_models(delta_path: str, device: torch.device):
    delta = torch.load(
        delta_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if delta.get("checkpoint_kind") != GOAL_PRIOR_CHECKPOINT_KIND:
        raise ValueError("checkpoint is not an image-goal Prior delta")
    if delta.get("checkpoint_version") != GOAL_PRIOR_CHECKPOINT_VERSION:
        raise ValueError("image-goal Prior checkpoint version differs")
    if delta.get("goal_condition_version") != GOAL_CONDITION_VERSION:
        raise ValueError("image-goal conditioner contract version differs")
    source_path = delta["source_checkpoint"]
    if file_sha256(source_path) != delta["source_checkpoint_sha256"]:
        raise ValueError("image-goal Prior base checkpoint SHA256 differs")
    source = torch.load(
        source_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = AdaptiveGaussianWMConfig(**source["config"])
    if config.to_dict() != delta["source_model_config"]:
        raise ValueError("image-goal Prior source model config differs")
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(source["model"], strict=True)
    conditioner = ObjectGoalConditioner(config).to(device)
    load_goal_prior_delta(
        delta,
        model,
        conditioner,
        delta["active_model_parameters"],
    )
    model.eval()
    conditioner.eval()
    return delta, model, conditioner, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "heldseed", "heldtask"),
        required=True,
    )
    parser.add_argument("--max_items", type=int, default=48)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--wrong_goal_scope",
        choices=("held", "batch"),
        default="held",
    )
    parser.add_argument("--time_counterfactual", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (
        args.batch < 2
        or (args.max_items > 0 and args.max_items < args.batch)
    ):
        raise ValueError("goal Prior evaluation requires at least two samples")
    device = torch.device(args.device)
    delta, model, conditioner, config = _load_models(args.checkpoint, device)
    saved = delta["args"]
    dataset = CausalVisualSequenceDataset(
        args.data,
        args.split,
        history_frames=saved["history_frames"],
        future_frames=saved["future_frames"],
        anchors=saved["sequence_anchors"],
        max_items=args.max_items,
        load_rgb=config.rgb_supervision,
        explicit_goal=True,
        rgb_short_side=config.rgb_short_side,
        rgb_pad_multiple=config.rgb_pad_multiple,
    )
    validate_data_model_contract(config, dataset, False, config.rgb_supervision)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    goal_bank = (
        build_goal_bank(model, loader, device)
        if args.wrong_goal_scope == "held" or args.time_counterfactual
        else None
    )
    evaluation = evaluate_goal_prior(
        model,
        conditioner,
        loader,
        device,
        wrong_goal_scope=args.wrong_goal_scope,
        action_activity_floor=saved["action_activity_floor"],
        goal_bank=goal_bank,
    )
    time_counterfactual = (
        evaluate_goal_time_counterfactual(
            model,
            conditioner,
            loader,
            device,
            action_activity_floor=saved["action_activity_floor"],
            query_time_bank=goal_bank["future_times"],
            goal_time_bank=goal_bank["goal_time"],
        )
        if args.time_counterfactual
        else None
    )
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "source_checkpoint": delta["source_checkpoint"],
        "source_checkpoint_sha256": delta["source_checkpoint_sha256"],
        "data": os.path.abspath(args.data),
        "split": args.split,
        "history_frames": dataset.history_frames,
        "future_frames": dataset.future_frames,
        "anchors": list(dataset.anchors),
        "language_condition": "off",
        "goal_condition": "explicit final image plus physical goal time",
        "evaluation_batch": args.batch,
        "wrong_goal_scope": args.wrong_goal_scope,
        "evaluation": evaluation,
        "time_counterfactual": time_counterfactual,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
