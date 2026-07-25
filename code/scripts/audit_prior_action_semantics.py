"""Audit instruction-conditioned canonical actions by task and channel."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm import (  # noqa: E402
    AdaptiveGaussianObjectWorldModel,
    AdaptiveGaussianWMConfig,
)
from igsw.adaptive_gaussian_wm.instruction_groups import (  # noqa: E402
    build_condition_task_bank,
    select_different_task_condition,
)
from igsw.adaptive_gaussian_wm.pair_dataset import (  # noqa: E402
    CausalPairFeatureDataset,
)
from igsw.adaptive_gaussian_wm.scale import signed_gap_scale  # noqa: E402
from igsw.adaptive_gaussian_wm.train_runtime import (  # noqa: E402
    move_to_device,
    validate_data_model_contract,
)

DIMENSION_NAMES = (
    "center_dx",
    "center_dy",
    "center_norm",
    "rgb_r",
    "rgb_g",
    "rgb_b",
)


def _history_from_output(output: dict) -> dict[str, torch.Tensor]:
    return {
        "slots": output["online_history_slots"],
        "activity": torch.stack(
            [state.activity for state in output["history_slot_states"]],
            dim=1,
        ),
        "center": output["online_history_centers"],
    }


def _load_metadata(paths: list[str]) -> list[dict]:
    metadata = []
    for path in paths:
        pair = torch.load(path, map_location="cpu", weights_only=False)
        metadata.append(
            {
                "path": os.path.abspath(path),
                "task": pair["task"],
                "instruction": pair["instruction"],
                "horizon": float(pair["horizon"]),
            }
        )
    return metadata


def _vector_mean(rows: list[dict], key: str) -> list[float]:
    values = torch.tensor([row[key] for row in rows], dtype=torch.float64)
    return values.mean(dim=0).tolist()


def _scalar_mean(rows: list[dict], key: str) -> float:
    return float(sum(row[key] for row in rows) / len(rows))


def _summary(rows: list[dict]) -> dict:
    prior_mse = torch.tensor(
        _vector_mean(rows, "prior_mse"),
        dtype=torch.float64,
    )
    wrong_mse = torch.tensor(
        _vector_mean(rows, "wrong_mse"),
        dtype=torch.float64,
    )
    zero_mse = torch.tensor(
        _vector_mean(rows, "zero_mse"),
        dtype=torch.float64,
    )
    instruction_improvement = wrong_mse - prior_mse
    zero_improvement = zero_mse - prior_mse
    return {
        "samples": len(rows),
        "target_mean": _vector_mean(rows, "target_mean"),
        "target_rms": _vector_mean(rows, "target_rms"),
        "prior_mean": _vector_mean(rows, "prior_mean"),
        "wrong_mean": _vector_mean(rows, "wrong_mean"),
        "prior_mse": prior_mse.tolist(),
        "wrong_mse": wrong_mse.tolist(),
        "zero_mse": zero_mse.tolist(),
        "prior_vs_wrong_absolute_improvement": instruction_improvement.tolist(),
        "prior_vs_zero_absolute_improvement": zero_improvement.tolist(),
        "prior_vs_wrong_positive_dimensions": int(
            (instruction_improvement > 0.0).sum()
        ),
        "prior_vs_zero_positive_dimensions": int(
            (zero_improvement > 0.0).sum()
        ),
        "prior_target_cosine": _scalar_mean(rows, "prior_target_cosine"),
        "wrong_target_cosine": _scalar_mean(rows, "wrong_target_cosine"),
        "correct_wrong_action_rms": _scalar_mean(
            rows,
            "correct_wrong_action_rms",
        ),
    }


def _group_summaries(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {
        name: _summary(group)
        for name, group in sorted(groups.items())
    }


@torch.no_grad()
def audit(
    model: AdaptiveGaussianObjectWorldModel,
    loader: DataLoader,
    metadata: list[dict],
    device: torch.device,
    condition_bank: torch.Tensor,
    condition_token_bank: torch.Tensor | None,
    condition_token_valid_bank: torch.Tensor | None,
    condition_task_bank: torch.Tensor,
    wrong_task_rank: int,
) -> list[dict]:
    rows = []
    offset = 0
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        batch_size = batch["history_features"].shape[0]
        history_mask = torch.zeros(
            batch_size,
            batch["history_features"].shape[1],
            model.config.object_slots,
            device=device,
            dtype=torch.bool,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch, history_mask=history_mask)
            prior = model.latent_actions.prior.sample(
                output["prior_context"],
                sample_count=1,
                stochastic=False,
            )[0]
            wrong_index = select_different_task_condition(
                batch["condition_index"].cpu(),
                batch["task_index"].cpu(),
                condition_task_bank,
                condition_bank,
                wrong_task_rank,
            )
            wrong_condition = model.language_condition(
                condition_bank[wrong_index].to(device, non_blocking=True)
            )
            history_scale = signed_gap_scale(
                batch["history_times"],
                model.config.gap_reference,
            )
            future_scale = signed_gap_scale(
                batch["future_times"],
                model.config.gap_reference,
            )
            wrong_context = model.prior_context(
                _history_from_output(output),
                future_scale,
                history_scale,
                wrong_condition,
                (
                    condition_token_bank[wrong_index].to(
                        device,
                        non_blocking=True,
                    )
                    if condition_token_bank is not None
                    else None
                ),
                (
                    condition_token_valid_bank[wrong_index].to(
                        device,
                        non_blocking=True,
                    )
                    if condition_token_valid_bank is not None
                    else None
                ),
            )
            wrong = model.latent_actions.prior.sample(
                wrong_context,
                sample_count=1,
                stochastic=False,
            )[0]
        target = output["posterior_actions"].float()
        prior = prior.float()
        wrong = wrong.float()
        for index in range(batch_size):
            sample_target = target[index]
            sample_prior = prior[index]
            sample_wrong = wrong[index]
            record = dict(metadata[offset + index])
            record.update(
                {
                    "target_mean": sample_target.mean(dim=(0, 1)).cpu().tolist(),
                    "target_rms": sample_target.square().mean(
                        dim=(0, 1)
                    ).sqrt().cpu().tolist(),
                    "prior_mean": sample_prior.mean(dim=(0, 1)).cpu().tolist(),
                    "wrong_mean": sample_wrong.mean(dim=(0, 1)).cpu().tolist(),
                    "prior_mse": (
                        sample_prior - sample_target
                    ).square().mean(dim=(0, 1)).cpu().tolist(),
                    "wrong_mse": (
                        sample_wrong - sample_target
                    ).square().mean(dim=(0, 1)).cpu().tolist(),
                    "zero_mse": sample_target.square().mean(
                        dim=(0, 1)
                    ).cpu().tolist(),
                    "prior_target_cosine": float(
                        F.cosine_similarity(
                            sample_prior.flatten(),
                            sample_target.flatten(),
                            dim=0,
                        ).cpu()
                    ),
                    "wrong_target_cosine": float(
                        F.cosine_similarity(
                            sample_wrong.flatten(),
                            sample_target.flatten(),
                            dim=0,
                        ).cpu()
                    ),
                    "correct_wrong_action_rms": float(
                        (sample_prior - sample_wrong)
                        .square().mean().sqrt().cpu()
                    ),
                }
            )
            rows.append(record)
        offset += batch_size
    if offset != len(metadata):
        raise AssertionError("loader and metadata lengths differ")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--condition_cache", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "heldseed", "heldtask"),
        required=True,
    )
    parser.add_argument("--max_items", type=int, default=144)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--wrong_task_rank", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batch < 2 or args.max_items % args.batch:
        raise ValueError("max_items must be divisible by batch >= 2")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = AdaptiveGaussianWMConfig(**checkpoint["config"])
    if not config.canonical_semantic_action or config.action_dim != 6:
        raise ValueError("audit requires canonical-only six-dimensional actions")
    dataset = CausalPairFeatureDataset(
        args.data,
        args.dino,
        args.split,
        max_items=args.max_items,
        condition_cache=args.condition_cache,
        load_rgb=True,
        rgb_short_side=config.rgb_short_side,
        rgb_pad_multiple=config.rgb_pad_multiple,
    )
    validate_data_model_contract(config, dataset, True, True)
    if dataset.condition_store is None:
        raise ValueError("action semantic audit requires condition cache")
    condition_task_bank = build_condition_task_bank(
        dataset.all_paths,
        dataset.condition_store,
    )
    model = AdaptiveGaussianObjectWorldModel(config).cuda()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    rows = audit(
        model,
        loader,
        _load_metadata(dataset.paths),
        torch.device("cuda"),
        dataset.condition_store.features,
        dataset.condition_store.token_features,
        dataset.condition_store.token_valid,
        condition_task_bank,
        args.wrong_task_rank,
    )
    report = {
        "status": "ok",
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": args.split,
        "wrong_task_rank": args.wrong_task_rank,
        "dimensions": DIMENSION_NAMES,
        "overall": _summary(rows),
        "by_task": _group_summaries(rows, "task"),
        "by_instruction": _group_summaries(rows, "instruction"),
        "samples": rows,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in (
        "status",
        "checkpoint",
        "split",
        "wrong_task_rank",
        "dimensions",
        "overall",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
