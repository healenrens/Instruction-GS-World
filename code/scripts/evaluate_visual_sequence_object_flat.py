"""Compare the object posterior core with a matched unstructured baseline."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
)
from igsw.adaptive_gaussian_wm.flat_baseline_checkpointing import (  # noqa: E402
    flat_parameter_metrics,
)
from igsw.adaptive_gaussian_wm.goal_eval_statistics import (  # noqa: E402
    clustered_relative_noninferiority_test,
)
from igsw.adaptive_gaussian_wm.goal_prior_checkpointing import (  # noqa: E402
    file_sha256,
)
from igsw.adaptive_gaussian_wm.matched_flat_world_model import (  # noqa: E402
    MatchedFlatLatentWorldModel,
)
from igsw.adaptive_gaussian_wm.matched_flat_evaluation import (  # noqa: E402
    COMPARISONS as FLAT_COMPARISONS,
    VARIANTS,
    causal_probe,
    change_feature_mse_by_query,
    change_rgb_distance_by_query,
    clustered_comparisons as comparisons,
    feature_mse_by_query,
    RGB_REGIONS,
    rgb_distance_by_query,
    rgb_region_errors_by_query,
)
from igsw.adaptive_gaussian_wm.matched_flat_contract import (  # noqa: E402
    available_samples,
    validate_object_flat_checkpoints,
)
from igsw.adaptive_gaussian_wm.matched_flat_training_contract import (  # noqa: E402
    training_contract_summary,
)
from igsw.adaptive_gaussian_wm.matched_flat_rgb import (  # noqa: E402
    batch_rgb_grids,
    feature_grid_shape,
    render_rgb_grid,
)
from igsw.adaptive_gaussian_wm.scale import signed_gap_scale  # noqa: E402
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.task_group_evidence import (  # noqa: E402
    object_flat_task_evidence, selected_task_group_layout,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    move_to_device,
    validate_data_model_contract,
)
from igsw.adaptive_gaussian_wm.temporal_region_evaluation import (  # noqa: E402
    TemporalRegionConfig,
)


@torch.no_grad()
def evaluate(
    object_model: AdaptiveGaussianObjectWorldModel,
    flat: MatchedFlatLatentWorldModel,
    loader: DataLoader,
    device: torch.device,
    amp: str,
    gap_reference: float,
    rgb_ssim_weight: float,
    task_layout,
) -> dict:
    values = {
        metric: {name: [] for name in VARIANTS}
        for metric in (
            "feature_mse",
            "change_weighted_feature_mse",
            "rgb_distance",
            "change_weighted_rgb_distance",
        )
    }
    clusters = []
    query_times = []
    object_actions = []
    flat_actions = []
    region_values = {
        region: {name: [] for name in VARIANTS}
        for region in RGB_REGIONS
    }
    region_clusters = {region: [] for region in RGB_REGIONS}
    region_coverage = {region: [] for region in RGB_REGIONS}
    region_config = TemporalRegionConfig()
    causal = None
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if amp == "bf16"
        else nullcontext
    )
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        object_mask = torch.zeros(
            batch["history_features"].shape[0],
            batch["history_features"].shape[1],
            object_model.config.object_slots,
            dtype=torch.bool,
            device=device,
        )
        history_scale = signed_gap_scale(batch["history_times"], gap_reference)
        future_scale = signed_gap_scale(batch["future_times"], gap_reference)
        history_rgb_grid, future_rgb_grid = batch_rgb_grids(batch)
        grid_height, grid_width = feature_grid_shape(batch)
        with amp_context():
            object_output = object_model(batch, history_mask=object_mask)
            flat_output = flat(
                batch["history_features"],
                batch["history_coordinates"],
                history_scale,
                history_rgb_grid,
                batch["future_features"],
                batch["future_coordinates"],
                future_scale,
                future_rgb_grid,
            )
            feature_predictions = {
                "object_posterior": object_output["rendered_future_features"],
                "flat_posterior": flat_output["posterior_feature_prediction"],
                "flat_history": flat_output["history_feature_prediction"],
                "copy": batch["history_features"][:, -1:].expand_as(
                    flat_output["posterior_feature_prediction"]
                ),
            }
            flat_posterior_rgb = render_rgb_grid(
                flat_output["posterior_rgb_grid_prediction"],
                batch["future_rgb_valid"],
                grid_height,
                grid_width,
            )
            flat_history_rgb = render_rgb_grid(
                flat_output["history_rgb_grid_prediction"],
                batch["future_rgb_valid"],
                grid_height,
                grid_width,
            )
            if object_output["rendered_future_rgb"] is None:
                raise ValueError("object RGB prediction is missing")
            rgb_predictions = {
                "object_posterior": object_output["rendered_future_rgb"],
                "flat_posterior": flat_posterior_rgb,
                "flat_history": flat_history_rgb,
                "copy": (batch["history_rgb"][:, -1:].float() / 255.0).expand_as(
                    flat_posterior_rgb
                ),
            }
        region_batch = rgb_region_errors_by_query(
            rgb_predictions,
            batch,
            region_config,
        )
        frame_clusters = batch["sequence_index"][:, None].expand(
            -1,
            batch["future_times"].shape[1],
        )
        for region in RGB_REGIONS:
            nonempty = region_batch[region]["nonempty"]
            region_clusters[region].append(frame_clusters[nonempty].cpu())
            region_coverage[region].append(
                region_batch[region]["coverage"].cpu()
            )
            for name in VARIANTS:
                region_values[region][name].append(
                    region_batch[region][name][nonempty].cpu()
                )
        if causal is None:
            with amp_context():
                causal = causal_probe(
                    flat,
                    batch,
                    history_rgb_grid,
                    future_rgb_grid,
                    history_scale,
                    future_scale,
                )
        for name, prediction in feature_predictions.items():
            values["feature_mse"][name].append(
                feature_mse_by_query(
                    prediction,
                    batch["future_features"],
                    batch["future_valid"],
                ).cpu()
            )
            values["change_weighted_feature_mse"][name].append(
                change_feature_mse_by_query(prediction, batch).cpu()
            )
        for name, prediction in rgb_predictions.items():
            values["rgb_distance"][name].append(
                rgb_distance_by_query(
                    prediction,
                    batch["future_rgb"],
                    batch["future_rgb_valid"],
                    rgb_ssim_weight,
                ).cpu()
            )
            values["change_weighted_rgb_distance"][name].append(
                change_rgb_distance_by_query(prediction, batch).cpu()
            )
        object_actions.append(object_output["posterior_actions"].float().cpu())
        flat_actions.append(flat_output["posterior_actions"].float().cpu())
        clusters.append(batch["sequence_index"].long().cpu())
        query_times.append(batch["future_times"].float().cpu())
    tensors = {
        metric: {name: torch.cat(chunks) for name, chunks in variants.items()}
        for metric, variants in values.items()
    }
    clusters = torch.cat(clusters)
    query_times = torch.cat(query_times)
    action_tensors = {
        "object_posterior": torch.cat(object_actions),
        "flat_posterior": torch.cat(flat_actions),
    }
    region_tensors = {
        region: {
            name: torch.cat(chunks)
            for name, chunks in variants.items()
        }
        for region, variants in region_values.items()
    }
    region_cluster_tensors = {
        region: torch.cat(chunks)
        for region, chunks in region_clusters.items()
    }
    rgb_regions = {}
    for region in RGB_REGIONS:
        if len(torch.unique(region_cluster_tensors[region])) < 2:
            raise ValueError(f"insufficient {region} RGB region clusters")
        rgb_regions[region] = {
            "frames": len(region_cluster_tensors[region]),
            "clusters": len(torch.unique(region_cluster_tensors[region])),
            "mean_coverage": float(torch.cat(region_coverage[region]).mean()),
            "mean": {
                name: float(value.mean())
                for name, value in region_tensors[region].items()
            },
            "comparison": comparisons(
                region_tensors[region],
                region_cluster_tensors[region],
            ),
        }
    overall = {
        metric: comparisons(
            {name: value.mean(dim=1) for name, value in variants.items()},
            clusters,
        )
        for metric, variants in tensors.items()
    }
    noninferiority = {
        metric: clustered_relative_noninferiority_test(
            variants["object_posterior"].mean(dim=1),
            variants["flat_posterior"].mean(dim=1),
            clusters,
            0.05,
        )
        for metric, variants in tensors.items()
    }
    return {
        "samples": len(clusters),
        "clusters": len(torch.unique(clusters)),
        "mean": {
            metric: {
                name: float(value.mean()) for name, value in variants.items()
            }
            for metric, variants in tensors.items()
        },
        "comparison": overall,
        "task_group_evidence": object_flat_task_evidence(
            tensors, clusters, task_layout, region_tensors,
            region_cluster_tensors, FLAT_COMPARISONS),
        "object_vs_flat_noninferiority_5pct": noninferiority,
        "rgb_regions": {
            "mask_source": "ground_truth_current_and_future_rgb_for_evaluation_only",
            "config": asdict(region_config),
            "regions": rgb_regions,
            "object_vs_flat_static_noninferiority_5pct": (
                clustered_relative_noninferiority_test(
                    region_tensors["static"]["object_posterior"],
                    region_tensors["static"]["flat_posterior"],
                    region_cluster_tensors["static"],
                    0.05,
                )
            ),
        },
        "action_statistics": {
            name: {
                "rms": float(action.square().mean().sqrt()),
                "sample_std_mean": float(action.flatten(1).std(dim=0).mean()),
            }
            for name, action in action_tensors.items()
        },
        "causal_probe": causal,
        "by_future_query": {
            str(index): {
                "time_seconds": {
                    "min": float(query_times[:, index].min()),
                    "mean": float(query_times[:, index].mean()),
                    "max": float(query_times[:, index].max()),
                },
                "mean": {
                    metric: {
                        name: float(value[:, index].mean())
                        for name, value in variants.items()
                    }
                    for metric, variants in tensors.items()
                },
                "comparison": {
                    metric: comparisons(
                        {
                            name: value[:, index]
                            for name, value in variants.items()
                        },
                        clusters,
                    )
                    for metric, variants in tensors.items()
                },
            }
            for index in range(query_times.shape[1])
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_checkpoint", required=True)
    parser.add_argument("--flat_checkpoint", required=True)
    parser.add_argument("--required_step", type=int, default=12000)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("heldseed", "heldtask"), required=True)
    parser.add_argument("--max_items", type=int, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = (args.object_checkpoint, args.flat_checkpoint, args.data, args.output)
    if any(not os.path.isabs(path) for path in paths):
        raise ValueError("object-flat paths must be absolute")
    if args.batch < 2 or args.max_items < args.batch or args.required_step <= 0:
        raise ValueError("object-flat evaluation size is invalid")
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite evaluation: {args.output}")
    object_checkpoint = torch.load(
        args.object_checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    flat_checkpoint = torch.load(
        args.flat_checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    object_sha256 = file_sha256(args.object_checkpoint)
    flat_sha256 = file_sha256(args.flat_checkpoint)
    object_args = object_checkpoint.get("args", {})
    dataset = CausalVisualSequenceDataset(
        args.data,
        args.split,
        history_frames=int(object_args.get("history_frames", 0)),
        future_frames=int(object_args.get("future_frames", 0)),
        anchors=str(object_args.get("sequence_anchors", "")),
        max_items=args.max_items,
        load_rgb=True,
        rgb_short_side=int(object_checkpoint["config"]["rgb_short_side"]),
        rgb_pad_multiple=int(object_checkpoint["config"]["rgb_pad_multiple"]),
    )
    config, architecture = validate_object_flat_checkpoints(
        object_checkpoint,
        flat_checkpoint,
        object_sha256,
        dataset,
        args.required_step,
    )
    validate_data_model_contract(config, dataset, False, True)
    task_layout = selected_task_group_layout(dataset, args.data, args.split)
    device = torch.device(args.device)
    object_model = AdaptiveGaussianObjectWorldModel(config).to(device)
    object_model.load_state_dict(object_checkpoint["model"], strict=True)
    object_model.eval()
    flat = MatchedFlatLatentWorldModel(**architecture).to(device)
    flat.load_state_dict(flat_checkpoint["model"], strict=True)
    flat.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    object_parameters = sum(p.numel() for p in object_model.parameters())
    flat_parameters = flat_parameter_metrics(flat)
    capacity_ratio = flat_parameters["total_parameters"] / object_parameters
    report = {
        "status": "ok",
        "object_checkpoint": os.path.abspath(args.object_checkpoint),
        "object_checkpoint_sha256": object_sha256,
        "object_checkpoint_global_step": int(object_checkpoint["global_step"]),
        "flat_checkpoint": os.path.abspath(args.flat_checkpoint),
        "flat_checkpoint_sha256": flat_sha256,
        "flat_checkpoint_global_step": int(flat_checkpoint["global_step"]),
        "data": os.path.abspath(args.data),
        "data_sha256": dataset.data_sha256,
        "task_source_index_sha256": task_layout.source_index_sha256,
        "split": args.split,
        "requested_max_items": args.max_items,
        "available_samples": available_samples(dataset),
        "history_frames": dataset.history_frames,
        "future_frames": dataset.future_frames,
        "anchors": list(dataset.anchors),
        "action_contract": {
            "type": "continuous",
            "tokens": architecture["action_tokens"],
            "dimensions": architecture["action_dim"],
            "layout": "dino_effect_3_plus_rgb_logit_effect_3_plus_residual_8",
            "state_tokens": architecture["state_tokens"],
            "object_source": "future_conditioned_object_posterior_oracle",
            "flat_source": "future_conditioned_unstructured_latent_posterior_oracle",
            "dynamics_future_access": "latent_action_only",
            "flat_posterior_future_modalities": ["dino", "rgb"],
        },
        "parameter_count": {
            "object_model": object_parameters,
            **flat_parameters,
            "flat_to_object_ratio": capacity_ratio,
        },
        "training_contract": training_contract_summary(
            object_checkpoint, flat_checkpoint, args.required_step
        ),
        "comparison_scope": {
            "matched": [
                "history_and_future_dino_inputs",
                "history_and_future_rgb_inputs",
                "dino_and_rgb_future_supervision",
                "future_query_times_and_coordinates",
                "continuous_action_token_count_and_width",
                "future_visible_only_to_posterior",
                "dynamics_width_layers_and_heads",
                "shared_scale_modulated_dynamics_initialization",
                "effective_global_batch",
            ],
            "not_matched": [
                "cumulative_pre_reference_optimization",
                "object_center_rgb_anchor_vs_dense_dino_rgb_effect_anchor",
                "object_gaussian_readout_vs_dense_grid_readout",
                "object_specific_latent_and_assignment_auxiliaries",
            ],
            "claim": "internal_representation_sanity_not_final_paper_superiority",
            "deployability": "both_posterior_paths_are_non_deployable_oracles",
        },
        "evaluation": evaluate(
            object_model,
            flat,
            loader,
            device,
            args.amp,
            float(flat_checkpoint["args"]["gap_reference"]),
            config.rgb_ssim_weight,
            task_layout,
        ),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
