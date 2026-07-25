"""Fit train-split action-only probes and evaluate absolute-state leakage."""
from __future__ import annotations

import argparse
import hashlib
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
from igsw.adaptive_gaussian_wm.action_probe import (  # noqa: E402
    explained_fraction,
    fit_ridge,
    gain_fraction,
    per_row_group_mse,
    state_groups,
)
from igsw.adaptive_gaussian_wm.goal_eval_statistics import (  # noqa: E402
    clustered_paired_comparison,
)
from igsw.adaptive_gaussian_wm.observed_action import (  # noqa: E402
    posterior_from_targets,
)
from igsw.adaptive_gaussian_wm.scale import signed_gap_scale  # noqa: E402
from igsw.adaptive_gaussian_wm.sequence_dataset import (  # noqa: E402
    CausalVisualSequenceDataset,
)
from igsw.adaptive_gaussian_wm.task_group_evidence import (  # noqa: E402
    selected_task_group_layout,
    task_group_comparison,
)
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    move_to_device,
    validate_data_model_contract,
)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def amp_context(device: torch.device, amp: str):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and amp == "bf16"
        else torch.no_grad()
    )


def object_state(features, centers, activity) -> torch.Tensor:
    if features.shape[:-1] != centers.shape[:-1]:
        raise ValueError("object feature and center shapes differ")
    if activity.shape != features.shape[:-1]:
        raise ValueError("object activity shape differs")
    return torch.cat((features.float(), centers.float(), activity[..., None].float()), dim=-1)


@torch.no_grad()
def collect(model, loader, device, amp) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "action",
            "history",
            "absolute",
            "delta",
            "time",
            "cluster",
        )
    }
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        with amp_context(device, amp):
            history = model.encode_history(batch)
            target_history, target_future = model.encode_targets(batch)
            future_scale = signed_gap_scale(
                batch["future_times"], model.config.gap_reference
            )
            actions, _ = posterior_from_targets(
                model,
                batch,
                history,
                target_future,
                future_scale,
                None,
            )
        current = object_state(
            target_history["feature"][:, -1],
            target_history["center"][:, -1],
            target_history["activity"][:, -1],
        )
        future = object_state(
            target_future["feature"],
            target_future["center"],
            target_future["activity"],
        )
        query_count = future.shape[1]
        chunks["action"].append(actions.float().flatten(2).cpu())
        chunks["history"].append(
            current.flatten(1)[:, None].expand(-1, query_count, -1).cpu()
        )
        chunks["absolute"].append(future.flatten(2).cpu())
        chunks["delta"].append(
            (future - current[:, None]).flatten(2).cpu()
        )
        chunks["time"].append(batch["future_times"].float().cpu()[..., None])
        chunks["cluster"].append(batch["sequence_index"].long().cpu())
    return {name: torch.cat(values) for name, values in chunks.items()}


def probe_inputs(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    action = torch.cat((data["action"], data["time"]), dim=-1)
    history = torch.cat((data["history"], data["time"]), dim=-1)
    return {
        "action": action,
        "history": history,
        "history_action": torch.cat(
            (data["history"], data["action"], data["time"]), dim=-1
        ),
    }


def flatten_rows(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.flatten(0, 1).to(device, non_blocking=True)


def fit_probes(train, device, penalty):
    inputs = probe_inputs(train)
    absolute = flatten_rows(train["absolute"], device)
    delta = flatten_rows(train["delta"], device)
    probes = {}
    for source in ("action", "history", "history_action"):
        probes[(source, "absolute")] = fit_ridge(
            flatten_rows(inputs[source], device), absolute, penalty
        )
    for source in ("action", "history_action"):
        probes[(source, "delta")] = fit_ridge(
            flatten_rows(inputs[source], device), delta, penalty
        )
    return probes


def evaluate_probes(probes, train, evaluation, device, feature_dim, object_count, layout):
    inputs = probe_inputs(evaluation)
    targets = {
        name: flatten_rows(evaluation[name], device)
        for name in ("absolute", "delta")
    }
    predictions = {
        key: probe.predict(flatten_rows(inputs[key[0]], device))
        for key, probe in probes.items()
    }
    train_means = {
        name: train[name].flatten(0, 1).mean(dim=0, keepdim=True).to(device)
        for name in ("absolute", "delta")
    }
    baselines = {
        "absolute_mean": train_means["absolute"].expand_as(targets["absolute"]),
        "delta_mean": train_means["delta"].expand_as(targets["delta"]),
        "copy": flatten_rows(evaluation["history"], device),
    }
    groups = state_groups(feature_dim, object_count)
    rows = len(targets["absolute"])
    sample_count, query_count = evaluation["absolute"].shape[:2]
    if rows != sample_count * query_count:
        raise RuntimeError("probe rows do not match samples and queries")
    errors: dict[str, dict[str, torch.Tensor]] = {
        group: {} for group in groups
    }
    for group, indices in groups.items():
        target_errors = {}
        for key, prediction in predictions.items():
            target_errors[f"{key[0]}_{key[1]}"] = per_row_group_mse(
                prediction, targets[key[1]], {group: indices}
            )[group]
        target_errors["absolute_mean"] = per_row_group_mse(
            baselines["absolute_mean"], targets["absolute"], {group: indices}
        )[group]
        target_errors["delta_mean"] = per_row_group_mse(
            baselines["delta_mean"], targets["delta"], {group: indices}
        )[group]
        target_errors["copy"] = per_row_group_mse(
            baselines["copy"], targets["absolute"], {group: indices}
        )[group]
        errors[group] = {
            name: value.reshape(sample_count, query_count).cpu()
            for name, value in target_errors.items()
        }
    clusters = evaluation["cluster"]
    report = {}
    for group, values in errors.items():
        sample_values = {name: value.mean(dim=1) for name, value in values.items()}
        comparisons = {
            "history_adds_to_action_absolute": clustered_paired_comparison(
                sample_values["history_action_absolute"],
                sample_values["action_absolute"],
                clusters,
            ),
            "action_adds_to_history_absolute": clustered_paired_comparison(
                sample_values["history_action_absolute"],
                sample_values["history_absolute"],
                clusters,
            ),
            "action_delta_over_mean": clustered_paired_comparison(
                sample_values["action_delta"],
                sample_values["delta_mean"],
                clusters,
            ),
        }
        task_evidence = {
            "history_adds_to_action_absolute": task_group_comparison(
                sample_values["history_action_absolute"],
                sample_values["action_absolute"],
                clusters,
                layout,
            ),
            "action_adds_to_history_absolute": task_group_comparison(
                sample_values["history_action_absolute"],
                sample_values["history_absolute"],
                clusters,
                layout,
            ),
            "action_delta_over_mean": task_group_comparison(
                sample_values["action_delta"],
                sample_values["delta_mean"],
                clusters,
                layout,
            ),
        }
        report[group] = {
            "mean_mse": {name: float(value.mean()) for name, value in values.items()},
            "absolute_action_explained_fraction_vs_mean": explained_fraction(
                sample_values["action_absolute"], sample_values["absolute_mean"]
            ),
            "delta_action_explained_fraction_vs_mean": explained_fraction(
                sample_values["action_delta"], sample_values["delta_mean"]
            ),
            "absolute_action_gain_fraction_of_history_action_vs_copy": gain_fraction(
                sample_values["action_absolute"],
                sample_values["history_action_absolute"],
                sample_values["copy"],
            ),
            "comparison": comparisons,
            "task_group_evidence": task_evidence,
            "by_future_query": {
                str(index): {
                    "mean_mse": {
                        name: float(value[:, index].mean())
                        for name, value in values.items()
                    },
                    "history_adds_to_action_absolute": clustered_paired_comparison(
                        values["history_action_absolute"][:, index],
                        values["action_absolute"][:, index],
                        clusters,
                    ),
                }
                for index in range(query_count)
            },
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--eval_split", choices=("heldseed", "heldtask"), required=True)
    parser.add_argument("--train_items", type=int, required=True)
    parser.add_argument("--eval_items", type=int, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ridge_penalty", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if any(not os.path.isabs(path) for path in (args.checkpoint, args.data, args.output)):
        raise ValueError("action-probe paths must be absolute")
    if min(args.train_items, args.eval_items, args.batch) <= 0:
        raise ValueError("action-probe sample and batch sizes must be positive")
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
        or config.condition_dim != 0
    ):
        raise ValueError("checkpoint does not satisfy the action-probe contract")
    common = {
        "cache_root": args.data,
        "history_frames": int(saved["history_frames"]),
        "future_frames": int(saved["future_frames"]),
        "anchors": str(saved["sequence_anchors"]),
        "load_rgb": config.rgb_supervision,
        "rgb_short_side": config.rgb_short_side,
        "rgb_pad_multiple": config.rgb_pad_multiple,
    }
    train_dataset = CausalVisualSequenceDataset(
        split="train", max_items=args.train_items, **common
    )
    eval_dataset = CausalVisualSequenceDataset(
        split=args.eval_split, max_items=args.eval_items, **common
    )
    validate_data_model_contract(config, train_dataset, False, config.rgb_supervision)
    validate_data_model_contract(config, eval_dataset, False, config.rgb_supervision)
    layout = selected_task_group_layout(eval_dataset, args.data, args.eval_split)
    device = torch.device(args.device)
    model = AdaptiveGaussianObjectWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    loader_args = {
        "batch_size": args.batch,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": True,
    }
    train = collect(model, DataLoader(train_dataset, **loader_args), device, args.amp)
    evaluation = collect(model, DataLoader(eval_dataset, **loader_args), device, args.amp)
    probes = fit_probes(train, device, args.ridge_penalty)
    result = evaluate_probes(
        probes,
        train,
        evaluation,
        device,
        config.feature_dim,
        config.object_slots,
        layout,
    )
    report = {
        "status": "ok",
        "probe_scope": "diagnostic_linear_ridge_not_a_model_training_objective",
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "data": os.path.abspath(args.data),
        "data_sha256": train_dataset.data_sha256,
        "task_source_index_sha256": layout.source_index_sha256,
        "train_split": "train",
        "eval_split": args.eval_split,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "future_queries": eval_dataset.future_frames,
        "ridge_penalty": args.ridge_penalty,
        "targets": {
            "absolute": "future_object_feature_center_activity",
            "delta": "future_minus_current_object_feature_center_activity",
        },
        "evaluation": result,
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
