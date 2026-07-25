"""Run causal, component, and object-local latent-action evaluation."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.action_evidence_evaluation import (  # noqa: E402
    cross_episode_donors,
    evaluate,
)
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.task_group_evidence import (  # noqa: E402
    selected_task_group_layout,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    validate_data_model_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("heldseed", "heldtask"), required=True)
    parser.add_argument("--max_items", type=int, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if any(
        not os.path.isabs(path)
        for path in (args.checkpoint, args.data, args.output)
    ):
        raise ValueError("action-evidence paths must be absolute")
    if args.batch < 2 or args.max_items < args.batch:
        raise ValueError("action evidence requires at least two samples")
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    saved = checkpoint.get("args", {})
    if (
        checkpoint.get("phase") != "joint"
        or saved.get("data_format") != "sequence"
        or not config.object_aligned_actions
        or config.canonical_action_dim != 6
        or config.action_residual_dim <= 0
        or not config.rgb_supervision
        or config.condition_dim != 0
    ):
        raise ValueError("checkpoint does not satisfy the object-action evidence contract")
    dataset = CausalVisualSequenceDataset(
        args.data,
        args.split,
        history_frames=int(saved["history_frames"]),
        future_frames=int(saved["future_frames"]),
        anchors=str(saved["sequence_anchors"]),
        max_items=args.max_items,
        load_rgb=True,
        rgb_short_side=config.rgb_short_side,
        rgb_pad_multiple=config.rgb_pad_multiple,
    )
    validate_data_model_contract(config, dataset, False, True)
    layout = selected_task_group_layout(dataset, args.data, args.split)
    device = torch.device(args.device)
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    loader_args = {
        "batch_size": args.batch,
        "num_workers": args.workers,
        "pin_memory": True,
    }
    loader = DataLoader(dataset, shuffle=False, **loader_args)
    clusters = torch.tensor(
        [dataset._locate(index)[0].episode_index for index in range(len(dataset))]
    )
    donor_indices = cross_episode_donors(clusters)
    donor_loader = DataLoader(
        Subset(dataset, donor_indices.tolist()), shuffle=False, **loader_args
    )
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "data": os.path.abspath(args.data),
        "data_sha256": dataset.data_sha256,
        "task_source_index_sha256": layout.source_index_sha256,
        "split": args.split,
        "requested_max_items": args.max_items,
        "samples": len(dataset),
        "history_frames": dataset.history_frames,
        "future_frames": dataset.future_frames,
        "anchors": list(dataset.anchors),
        "action_layout": {
            "tokens": config.action_tokens,
            "dimensions": config.action_dim,
            "canonical_dimensions": config.canonical_action_dim,
            "residual_dimensions": config.action_residual_dim,
        },
        "evaluation": evaluate(
            model, loader, donor_loader, device, args.amp, layout
        ),
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = f"{output}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
