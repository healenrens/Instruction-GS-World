"""Evaluate current-frame representation on the dense visual episode contract."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "scripts"))

from evaluate_adaptive_gaussian_representation import evaluate  # noqa: E402
from evaluate_visual_sequence_temporal_regions import (  # noqa: E402
    _available_samples,
    saved_or_override,
)
from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.task_group_evidence import (  # noqa: E402
    selected_task_group_layout,
)
from igsw.adaptive_gaussian_wm.temporal_slot_evaluation import (  # noqa: E402
    evaluate_temporal_slots,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    validate_data_model_contract,
)


def _sequence_cluster_ids(dataset) -> torch.Tensor:
    if not hasattr(dataset, "_locate"):
        raise ValueError("representation evaluation requires dense episode data")
    return torch.tensor(
        [dataset._locate(index)[0].episode_index for index in range(len(dataset))],
        dtype=torch.long,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "heldseed", "heldtask"),
        required=True,
    )
    parser.add_argument("--max_items", type=int, default=144)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--history_frames", type=int, default=0)
    parser.add_argument("--future_frames", type=int, default=0)
    parser.add_argument("--sequence_anchors", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batch < 2 or args.max_items < args.batch:
        raise ValueError("representation evaluation requires at least two samples")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    checkpoint_step = int(checkpoint.get("global_step", -1))
    if checkpoint_step < 1 or checkpoint.get("phase") != "joint":
        raise ValueError("representation evaluation requires a joint checkpoint")
    saved = checkpoint.get("args", {})
    if saved.get("data_format") != "sequence":
        raise ValueError("checkpoint was not trained with visual sequences")
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    if config.condition_dim != 0 or not config.rgb_supervision:
        raise ValueError("representation evaluation requires RGB and no language")
    dataset = CausalVisualSequenceDataset(
        args.data,
        args.split,
        history_frames=saved_or_override(
            saved,
            "history_frames",
            args.history_frames,
        ),
        future_frames=saved_or_override(
            saved,
            "future_frames",
            args.future_frames,
        ),
        anchors=saved_or_override(
            saved,
            "sequence_anchors",
            args.sequence_anchors,
        ),
        max_items=args.max_items,
        load_rgb=True,
        rgb_short_side=config.rgb_short_side,
        rgb_pad_multiple=config.rgb_pad_multiple,
    )
    validate_data_model_contract(config, dataset, False, True)
    clusters = _sequence_cluster_ids(dataset)
    device = torch.device(args.device)
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    representation_metrics = evaluate(model, loader, device, args.amp, clusters)
    temporal_slots = None
    if args.split in ("heldseed", "heldtask"):
        layout = selected_task_group_layout(dataset, args.data, args.split)
        temporal_slots = evaluate_temporal_slots(
            model, loader, device, args.amp, layout
        )
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_global_step": checkpoint_step,
        "checkpoint_phase": checkpoint["phase"],
        "checkpoint_version": checkpoint.get("checkpoint_version"),
        "data": os.path.abspath(args.data),
        "data_sha256": dataset.data_sha256,
        "split": args.split,
        "amp": args.amp,
        "requested_max_items": args.max_items,
        "available_samples": _available_samples(dataset),
        "samples": len(dataset),
        "source_clusters": len(torch.unique(clusters)),
        "history_frames": dataset.history_frames,
        "future_frames": dataset.future_frames,
        "anchors": list(dataset.anchors),
        "metrics": representation_metrics,
        "temporal_slot_stability": temporal_slots,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
