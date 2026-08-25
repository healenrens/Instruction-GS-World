"""Exact sufficient statistics for v59 transition evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .object_transition_objective_v59 import (
    persistence_transition_errors_v59,
    transition_prediction_errors_v59,
)


BASELINES = ("zero", "shuffled", "persistence")


@dataclass
class TransitionEvaluationAccumulator:
    horizons: tuple[int, ...]
    active_count: float = 0.0
    valid_count: float = 0.0
    sample_count: int = 0
    error_sums: dict[str, float] = field(default_factory=dict)
    component_sums: dict[str, float] = field(default_factory=dict)
    horizon_active: dict[int, float] = field(default_factory=dict)
    horizon_error_sums: dict[tuple[int, str], float] = field(default_factory=dict)
    effect_sum: torch.Tensor | None = None
    effect_square_sum: torch.Tensor | None = None
    effect_abs_sum: float = 0.0
    effect_count: int = 0
    prediction_delta_square_sum: float = 0.0
    prediction_delta_count: int = 0

    def update(self, output, target, config, sample_mask: torch.Tensor | None = None):
        batch = target.pair_valid.shape[0]
        selected = (
            torch.ones(batch, dtype=torch.bool, device=target.pair_valid.device)
            if sample_mask is None
            else sample_mask.bool()
        )
        active = target.motion_active & selected[:, None]
        valid = target.pair_valid & selected[:, None]
        errors = {
            "correct": transition_prediction_errors_v59(
                output["correct"], target, config
            ),
            "zero": transition_prediction_errors_v59(output["zero"], target, config),
            "shuffled": transition_prediction_errors_v59(
                output["shuffled"], target, config
            ),
            "persistence": persistence_transition_errors_v59(target, config),
        }
        active_float = active.float()
        valid_float = valid.float()
        self.active_count += float(active_float.sum())
        self.valid_count += float(valid_float.sum())
        self.sample_count += int(selected.sum())
        for name, values in errors.items():
            self.error_sums[name] = self.error_sums.get(name, 0.0) + float(
                (values[0] * active_float).sum()
            )
        for name, values in zip(
            ("semantic", "geometry", "lifecycle"), errors["correct"][1:]
        ):
            self.component_sums[name] = self.component_sums.get(name, 0.0) + float(
                (values * valid_float).sum()
            )
        for index, horizon in enumerate(self.horizons):
            weight = active_float[:, index]
            self.horizon_active[horizon] = self.horizon_active.get(
                horizon, 0.0
            ) + float(weight.sum())
            for name, values in errors.items():
                key = (horizon, name)
                self.horizon_error_sums[key] = self.horizon_error_sums.get(
                    key, 0.0
                ) + float((values[0][:, index] * weight).sum())

        effect = output["effect"][selected].float().flatten(0, 1).flatten(0, 1).cpu()
        if effect.numel():
            total = effect.sum(dim=0)
            square = effect.square().sum(dim=0)
            self.effect_sum = (
                total if self.effect_sum is None else self.effect_sum + total
            )
            self.effect_square_sum = (
                square
                if self.effect_square_sum is None
                else self.effect_square_sum + square
            )
            self.effect_abs_sum += float(effect.abs().sum())
            self.effect_count += effect.shape[0]
        delta = (
            output["correct"].semantic[selected].float()
            - output["zero"].semantic[selected].float()
        )
        self.prediction_delta_square_sum += float(delta.square().sum())
        self.prediction_delta_count += delta.numel()

    @staticmethod
    def _gain(correct: float, baseline: float) -> float:
        return (baseline - correct) / max(baseline, 1e-6)

    def finalize(self) -> dict[str, float]:
        metrics = {
            "sample_count": float(self.sample_count),
            "motion_active_count": self.active_count,
            "transition_valid_count": self.valid_count,
            "motion_active_fraction": self.active_count
            / max(self.sample_count * len(self.horizons), 1),
            "transition_valid_fraction": self.valid_count
            / max(self.sample_count * len(self.horizons), 1),
        }
        for name, total in self.error_sums.items():
            metrics[f"{name}_active_error"] = total / max(self.active_count, 1.0)
        correct = metrics["correct_active_error"]
        for baseline in BASELINES:
            metrics[f"gain_over_{baseline}"] = self._gain(
                correct, metrics[f"{baseline}_active_error"]
            )
        for name, total in self.component_sums.items():
            metrics[f"correct_{name}_error"] = total / max(self.valid_count, 1.0)
        if self.effect_sum is not None and self.effect_square_sum is not None:
            mean = self.effect_sum / self.effect_count
            variance = self.effect_square_sum / self.effect_count - mean.square()
            metrics["effect_std"] = float(variance.clamp_min(0.0).sqrt().mean())
            metrics["effect_abs_mean"] = self.effect_abs_sum / (
                self.effect_count * self.effect_sum.numel()
            )
        metrics["effect_prediction_delta"] = (
            self.prediction_delta_square_sum / max(self.prediction_delta_count, 1)
        ) ** 0.5
        for horizon in self.horizons:
            count = self.horizon_active.get(horizon, 0.0)
            metrics[f"h{horizon}_motion_active_count"] = count
            for name in ("correct", *BASELINES):
                metrics[f"h{horizon}_{name}_active_error"] = (
                    self.horizon_error_sums.get((horizon, name), 0.0) / max(count, 1.0)
                )
            horizon_correct = metrics[f"h{horizon}_correct_active_error"]
            for baseline in BASELINES:
                metrics[f"h{horizon}_gain_over_{baseline}"] = self._gain(
                    horizon_correct,
                    metrics[f"h{horizon}_{baseline}_active_error"],
                )
        return metrics


def macro_metrics_v59(records: list[dict[str, float]]) -> dict[str, float]:
    keys = records[0].keys()
    metrics = {
        key: sum(record[key] for record in records) / len(records) for key in keys
    }
    correct = metrics["correct_active_error"]
    for baseline in BASELINES:
        metrics[f"gain_over_{baseline}"] = (
            metrics[f"{baseline}_active_error"] - correct
        ) / max(metrics[f"{baseline}_active_error"], 1e-6)
    horizons = sorted(
        int(key.split("_", maxsplit=1)[0][1:])
        for key in keys
        if key.startswith("h") and key.endswith("_correct_active_error")
    )
    for horizon in horizons:
        horizon_correct = metrics[f"h{horizon}_correct_active_error"]
        for baseline in BASELINES:
            reference = metrics[f"h{horizon}_{baseline}_active_error"]
            metrics[f"h{horizon}_gain_over_{baseline}"] = (
                reference - horizon_correct
            ) / max(reference, 1e-6)
    return metrics


def bootstrap_macro_gains_v59(
    records: list[dict[str, float]], samples: int, seed: int
) -> dict[str, dict[str, float]]:
    generator = torch.Generator().manual_seed(seed)
    correct = torch.tensor([row["correct_active_error"] for row in records])
    result = {}
    for baseline in BASELINES:
        reference = torch.tensor([row[f"{baseline}_active_error"] for row in records])
        indices = torch.randint(
            len(records), (samples, len(records)), generator=generator
        )
        sampled_correct = correct[indices].mean(dim=1)
        sampled_reference = reference[indices].mean(dim=1)
        gains = (sampled_reference - sampled_correct) / sampled_reference.clamp_min(
            1e-6
        )
        result[baseline] = {
            "estimate": float(
                (reference.mean() - correct.mean()) / reference.mean().clamp_min(1e-6)
            ),
            "ci95_low": float(torch.quantile(gains, 0.025)),
            "ci95_high": float(torch.quantile(gains, 0.975)),
        }
    return result
