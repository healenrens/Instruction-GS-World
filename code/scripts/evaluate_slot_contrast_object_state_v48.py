#!/usr/bin/env python3
"""Held-video evaluation for the completed v48 slot-state checkpoint."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.frozen_video_encoder import (  # noqa: E402
    FrozenDinoVideoRuntime,
)
from igsw.adaptive_gaussian_wm.goal_eval_statistics import (  # noqa: E402
    clustered_paired_comparison,
)
from igsw.adaptive_gaussian_wm.slot_contrast_world_model import (  # noqa: E402
    SlotContrastObjectWorldModel,
)
from igsw.adaptive_gaussian_wm.temporal_object_dataset import (  # noqa: E402
    TemporalObjectVideoDataset,
    parse_int_choices,
)
from igsw.adaptive_gaussian_wm.v48_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    SlotContrastConfig,
)
from igsw.adaptive_gaussian_wm.v48_cross_episode_evaluation import (  # noqa: E402
    cross_episode_consistency,
)
from igsw.adaptive_gaussian_wm.v48_eval_visualization import (  # noqa: E402
    save_assignment_visualizations,
)
from igsw.adaptive_gaussian_wm.v48_held_metrics import (  # noqa: E402
    base_state_metrics,
    encode_fully_observed,
    slot_deletion_metrics,
    synthetic_blackout_metrics,
    temporal_order_metrics,
)


class FixedChunkView(Dataset):
    def __init__(self, dataset: TemporalObjectVideoDataset, chunk_length: int):
        self.dataset = dataset
        self.chunk_length = int(chunk_length)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.dataset[(index, self.chunk_length)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--split", choices=("heldseed", "heldtask"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--visualization_dir", default="")
    parser.add_argument("--max_items", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dino_frame_batch", type=int, default=128)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--chunk_lengths", default="8,16,24,32")
    parser.add_argument("--temporal_strides", default="1,2,3,4")
    parser.add_argument("--cross_episode_pairs", type=int, default=2000)
    parser.add_argument("--qualitative_items", type=int, default=8)
    return parser.parse_args()


def _move(batch: dict, device: torch.device) -> dict:
    return {
        name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def _cluster_summary(values: torch.Tensor, clusters: torch.Tensor) -> dict:
    if values.ndim != 1 or clusters.shape != values.shape:
        raise ValueError("v48 held metric and episode ids differ")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("v48 held metric contains a non-finite value")
    unique = torch.unique(clusters, sorted=True)
    episode_means = torch.stack(
        [values[clusters == cluster].mean() for cluster in unique]
    )
    standard_error = episode_means.std(unbiased=False) / math.sqrt(len(episode_means))
    mean = episode_means.mean()
    radius = 1.96 * standard_error
    return {
        "samples": len(values),
        "episodes": len(unique),
        "sample_mean": float(values.mean()),
        "episode_balanced_mean": float(mean),
        "episode_standard_error": float(standard_error),
        "ci95_lower": float(mean - radius),
        "ci95_upper": float(mean + radius),
    }


def _collect_metrics(storage: dict[str, list[torch.Tensor]], metrics: dict) -> None:
    for name, value in metrics.items():
        if value.ndim != 1:
            raise ValueError(f"v48 held metric {name} is not sample-wise")
        storage.setdefault(name, []).append(value.detach().float().cpu())


def _comparison(
    values: dict[str, torch.Tensor],
    prediction: str,
    reference: str,
    clusters: torch.Tensor,
) -> dict:
    return clustered_paired_comparison(values[prediction], values[reference], clusters)


def _group_id_lookup(dataset: TemporalObjectVideoDataset) -> torch.Tensor:
    names = sorted(
        {str(entry["sampling_group"]) for entry in dataset.manifest["episodes"]}
    )
    indices = {name: index for index, name in enumerate(names)}
    return torch.tensor(
        [
            indices[str(entry["sampling_group"])]
            for entry in dataset.manifest["episodes"]
        ],
        dtype=torch.long,
    )


def evaluate_length(
    model,
    encoder,
    dataset,
    group_lookup: torch.Tensor,
    chunk_length: int,
    args,
    device: torch.device,
    amp_context,
) -> dict:
    loader = DataLoader(
        FixedChunkView(dataset, chunk_length),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    collected: dict[str, list[torch.Tensor]] = {}
    episode_chunks, group_chunks = [], []
    slot_chunks, active_chunks = [], []
    visualization_paths = []
    item_offset = 0
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move(cpu_batch, device)
            features = encoder(batch)
            with amp_context():
                state = encode_fully_observed(
                    model,
                    features.patches,
                    features.coordinates,
                    features.valid,
                    batch["frame_times"],
                )
                metrics = base_state_metrics(
                    model, features.patches, features.valid, state
                )
                metrics.update(
                    temporal_order_metrics(
                        model,
                        features.patches,
                        features.coordinates,
                        features.valid,
                        batch["frame_times"],
                        args.seed + item_offset,
                    )
                )
                metrics.update(
                    synthetic_blackout_metrics(
                        model,
                        features.patches,
                        features.coordinates,
                        features.valid,
                        batch["frame_times"],
                        state,
                    )
                )
                metrics.update(
                    slot_deletion_metrics(
                        model,
                        features.patches,
                        features.coordinates,
                        features.valid,
                        state,
                    )
                )
            _collect_metrics(collected, metrics)
            episode_ids = batch["sequence_index"].long().cpu()
            episode_chunks.append(episode_ids)
            group_chunks.append(group_lookup[episode_ids])
            slot_chunks.append(state["contrast_slots"][:, -1].float().cpu())
            active_chunks.append(
                (state["activity"][:, -1] >= model.config.active_slot_fraction).cpu()
            )
            if args.visualization_dir and item_offset < args.qualitative_items:
                visualization_paths.extend(
                    save_assignment_visualizations(
                        cpu_batch["video_rgb"],
                        state["assignment"].float().cpu(),
                        features.grid_hw,
                        args.visualization_dir,
                        chunk_length,
                        item_offset,
                        args.qualitative_items,
                    )
                )
            item_offset += len(episode_ids)
            print(
                json.dumps(
                    {
                        "event": "v48_held_progress",
                        "split": args.split,
                        "chunk_length": chunk_length,
                        "items": item_offset,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    values = {name: torch.cat(chunks) for name, chunks in collected.items()}
    episodes = torch.cat(episode_chunks)
    groups = torch.cat(group_chunks)
    summaries = {
        name: _cluster_summary(value, episodes) for name, value in values.items()
    }
    comparisons = {
        "reconstruction_vs_frame_mean": _comparison(
            values, "reconstruction_error", "frame_mean_error", episodes
        ),
        "ordered_vs_reversed": _comparison(
            values, "ordered_target_error", "reversed_target_error", episodes
        ),
        "ordered_vs_shuffled": _comparison(
            values, "ordered_target_error", "shuffled_target_error", episodes
        ),
        "ordered_vs_persistence": _comparison(
            values, "ordered_target_error", "persistence_target_error", episodes
        ),
        "blackout_prediction_vs_persistence": _comparison(
            values,
            "blackout_prediction_error",
            "blackout_persistence_error",
            episodes,
        ),
    }
    cross_episode = cross_episode_consistency(
        torch.cat(slot_chunks),
        torch.cat(active_chunks),
        episodes,
        groups,
        args.cross_episode_pairs,
    )
    chance = 1.0 / model.config.object_slots
    checks = {
        "reconstruction_beats_frame_mean_ci95": comparisons[
            "reconstruction_vs_frame_mean"
        ]["positive_ci95_lower"],
        "ordered_history_beats_reversed_ci95": comparisons["ordered_vs_reversed"][
            "positive_ci95_lower"
        ],
        "ordered_history_beats_shuffled_ci95": comparisons["ordered_vs_shuffled"][
            "positive_ci95_lower"
        ],
        "ordered_prediction_beats_persistence_ci95": comparisons[
            "ordered_vs_persistence"
        ]["positive_ci95_lower"],
        "blackout_prediction_beats_persistence_ci95": comparisons[
            "blackout_prediction_vs_persistence"
        ]["positive_ci95_lower"],
        "slot_deletion_has_positive_utility": summaries["deletion_error_increase"][
            "ci95_lower"
        ]
        > 0.0,
        "slot_deletion_is_spatially_enriched": summaries[
            "deletion_locality_enrichment"
        ]["ci95_lower"]
        > 1.0,
        "reappearance_identity_beats_chance": summaries["reappearance_retrieval_top1"][
            "ci95_lower"
        ]
        > chance,
        "same_group_proxy_exceeds_different_group": (
            cross_episode["same_group_margin_over_different"] is not None
            and cross_episode["same_group_margin_over_different"] > 0.0
        ),
    }
    return {
        "items": len(episodes),
        "episodes": len(torch.unique(episodes)),
        "metrics": summaries,
        "paired_comparisons": comparisons,
        "cross_episode_proxy": cross_episode,
        "checks": checks,
        "visualizations": visualization_paths,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v48 held evaluator requires a visible CUDA device")
    if (
        min(
            args.max_items,
            args.batch,
            args.workers + 1,
            args.dino_frame_batch,
            args.cross_episode_pairs,
        )
        < 1
    ):
        raise ValueError("v48 held evaluator received a non-positive runtime argument")
    checkpoint_path = os.path.abspath(args.checkpoint)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("v48 held evaluator requires a version-48 checkpoint")
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("v48 held evaluator checkpoint architecture differs")
    config = SlotContrastConfig(**checkpoint["config"])
    config.validate()
    chunk_lengths = parse_int_choices(args.chunk_lengths, "chunk lengths")
    dataset = TemporalObjectVideoDataset(
        os.path.abspath(args.data),
        args.split,
        args.chunk_lengths,
        args.temporal_strides,
        observation_mask_probability=0.20,
        max_items=args.max_items,
        seed=args.seed,
        record_manifest_hash=False,
    )
    if any(value not in dataset.dynamic_history_lengths for value in chunk_lengths):
        raise ValueError("v48 held chunk lengths differ from the dataset contract")
    device = torch.device("cuda:0")
    model = SlotContrastObjectWorldModel(config).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    encoder = FrozenDinoVideoRuntime(
        config,
        device,
        args.amp,
        args.dino_frame_batch,
        os.path.abspath(args.dino_checkpoint),
    )
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.amp == "bf16"
        else nullcontext
    )
    group_lookup = _group_id_lookup(dataset)
    by_length = {}
    for chunk_length in chunk_lengths:
        print(
            json.dumps(
                {
                    "event": "v48_held_length_start",
                    "split": args.split,
                    "chunk_length": chunk_length,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        by_length[str(chunk_length)] = evaluate_length(
            model,
            encoder,
            dataset,
            group_lookup,
            chunk_length,
            args,
            device,
            amp_context,
        )
    all_checks = [
        value for length in by_length.values() for value in length["checks"].values()
    ]
    report = {
        "status": "completed",
        "contract": "v48_held_object_state_v1",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "checkpoint": checkpoint_path,
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "checkpoint_git_commit": checkpoint.get("git_commit"),
        "data": os.path.abspath(args.data),
        "split": args.split,
        "chunk_lengths": list(chunk_lengths),
        "evaluation_by_chunk_length": by_length,
        "all_checks_passed": all(all_checks),
        "semantic_object_correspondence_verified": False,
        "natural_occlusion_ground_truth_used": False,
        "interpretation_boundary": (
            "The evaluator measures compact-state capacity, temporal sensitivity, "
            "slot-local deletion effects, synthetic observation-dropout recovery, "
            "and task-group consistency. It does not supply object labels."
        ),
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = f"{output_path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output_path)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
