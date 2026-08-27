"""Exact unseen-window diagnostics for v60 transition factorization."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .latent_object_transition_v59 import ObjectTransitionPrediction
from .object_transition_objective_v59 import (
    persistence_transition_errors_v59,
    transition_prediction_errors_v59,
)


BASELINES = ("zero", "shuffled", "persistence")
DIAGNOSTIC_ROUTES = (
    "correct",
    "student_oracle_gate",
    "teacher_predicted_gate",
    "teacher_oracle_gate",
)
PREDICTION_ROUTES = (*DIAGNOSTIC_ROUTES, "zero", "shuffled")
GROUPS = ("active", "low_change", "high_change", "change_weighted")


def teacher_source_prediction(target) -> ObjectTransitionPrediction:
    horizons = target.future_semantic.shape[1]
    visibility = target.source_visibility.float().clamp(1e-5, 1.0 - 1e-5)
    return ObjectTransitionPrediction(
        semantic=target.source_semantic[:, None].expand(-1, horizons, -1),
        geometry=target.source_geometry[:, None].expand(-1, horizons, -1),
        visibility_logits=torch.logit(visibility)[:, None].expand(-1, horizons),
    )


def compose_transition(base, residual, gate) -> ObjectTransitionPrediction:
    scale = gate.float()
    return ObjectTransitionPrediction(
        semantic=F.normalize(
            base.semantic.float() + scale[..., None] * residual.semantic_delta.float(),
            dim=-1,
            eps=1e-6,
        ),
        geometry=(
            base.geometry.float() + scale[..., None] * residual.geometry_delta.float()
        ),
        visibility_logits=(
            base.visibility_logits.float() + scale * residual.visibility_delta.float()
        ),
    )


def diagnostic_routes_v60(output, target) -> dict[str, ObjectTransitionPrediction]:
    teacher = teacher_source_prediction(target)
    residual = output["correct_residual"]
    return {
        "correct": output["correct"],
        "student_oracle_gate": compose_transition(
            output["base"], residual, target.change_strength
        ),
        "teacher_predicted_gate": compose_transition(
            teacher, residual, output["change_gate"]
        ),
        "teacher_oracle_gate": compose_transition(
            teacher, residual, target.change_strength
        ),
        "zero": output["zero"],
        "shuffled": output["shuffled"],
    }


def _gain(candidate: float, reference: float) -> float:
    return (reference - candidate) / max(reference, 1e-6)


def _correlation(sums: dict[str, float]) -> float:
    count = max(sums["count"], 1.0)
    gate_mean = sums["gate"] / count
    target_mean = sums["target"] / count
    covariance = sums["cross"] / count - gate_mean * target_mean
    gate_variance = sums["gate_square"] / count - gate_mean**2
    target_variance = sums["target_square"] / count - target_mean**2
    denominator = max(gate_variance, 0.0) ** 0.5 * max(target_variance, 0.0) ** 0.5
    return covariance / max(denominator, 1e-6)


@dataclass
class GatedTransitionEvaluationAccumulator:
    horizons: tuple[int, ...]
    sample_count: int = 0
    weight_sums: dict[str, float] = field(default_factory=dict)
    error_sums: dict[tuple[str, str], float] = field(default_factory=dict)
    horizon_weights: dict[int, float] = field(default_factory=dict)
    horizon_errors: dict[tuple[int, str], float] = field(default_factory=dict)
    component_sums: dict[str, float] = field(default_factory=dict)
    gate_sums: dict[str, float] = field(default_factory=dict)
    source_base_error_sum: float = 0.0
    effect_sum: torch.Tensor | None = None
    effect_square_sum: torch.Tensor | None = None
    effect_abs_sum: float = 0.0
    effect_count: int = 0

    def _weights(self, target, selected, config):
        valid = target.pair_valid & selected[:, None]
        strength = target.change_strength.float() * valid.float()
        return {
            "valid": valid.float(),
            "active": (target.motion_active & selected[:, None]).float(),
            "low_change": (
                valid & (target.change_strength <= config.low_change_threshold)
            ).float(),
            "high_change": (
                valid & (target.change_strength >= config.high_change_threshold)
            ).float(),
            "change_weighted": strength,
        }

    @torch.no_grad()
    def update(self, output, target, config, sample_mask: torch.Tensor | None = None):
        batch = target.pair_valid.shape[0]
        selected = (
            torch.ones(batch, dtype=torch.bool, device=target.pair_valid.device)
            if sample_mask is None
            else sample_mask.bool()
        )
        routes = diagnostic_routes_v60(output, target)
        errors = {
            name: transition_prediction_errors_v59(prediction, target, config)
            for name, prediction in routes.items()
        }
        errors["persistence"] = persistence_transition_errors_v59(target, config)
        weights = self._weights(target, selected, config)
        self.sample_count += int(selected.sum())
        for group, weight in weights.items():
            self.weight_sums[group] = self.weight_sums.get(group, 0.0) + float(
                weight.sum()
            )
            if group == "valid":
                continue
            for route, values in errors.items():
                key = (group, route)
                self.error_sums[key] = self.error_sums.get(key, 0.0) + float(
                    (values[0] * weight).sum()
                )

        valid = weights["valid"]
        for name, values in zip(
            ("semantic", "geometry", "lifecycle"), errors["correct"][1:]
        ):
            self.component_sums[name] = self.component_sums.get(name, 0.0) + float(
                (values * valid).sum()
            )
        source_target = type(
            "SourceTarget",
            (),
            {
                "future_semantic": target.source_semantic[:, None].expand_as(
                    target.future_semantic
                ),
                "future_geometry": target.source_geometry[:, None].expand_as(
                    target.future_geometry
                ),
                "future_visibility": target.source_visibility[:, None].expand_as(
                    target.future_visibility
                ),
            },
        )()
        source_error = transition_prediction_errors_v59(
            output["base"], source_target, config
        )[0]
        self.source_base_error_sum += float((source_error * valid).sum())

        active = weights["active"]
        for index, horizon in enumerate(self.horizons):
            weight = active[:, index]
            self.horizon_weights[horizon] = self.horizon_weights.get(
                horizon, 0.0
            ) + float(weight.sum())
            for route, values in errors.items():
                key = (horizon, route)
                self.horizon_errors[key] = self.horizon_errors.get(key, 0.0) + float(
                    (values[0][:, index] * weight).sum()
                )

        gate = output["change_gate"].float()
        target_gate = target.change_strength.float()
        self.gate_sums["count"] = self.gate_sums.get("count", 0.0) + float(valid.sum())
        for name, value in (
            ("gate", gate),
            ("target", target_gate),
            ("gate_square", gate.square()),
            ("target_square", target_gate.square()),
            ("cross", gate * target_gate),
            ("absolute_error", (gate - target_gate).abs()),
        ):
            self.gate_sums[name] = self.gate_sums.get(name, 0.0) + float(
                (value * valid).sum()
            )

        effect = output["effect"][selected].float().flatten(0, 2).cpu()
        if effect.numel():
            total, square = effect.sum(dim=0), effect.square().sum(dim=0)
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

    def finalize(self) -> dict[str, float]:
        horizon_count = len(self.horizons)
        metrics = {
            "sample_count": float(self.sample_count),
            "transition_valid_count": self.weight_sums.get("valid", 0.0),
            "motion_active_count": self.weight_sums.get("active", 0.0),
            "low_change_count": self.weight_sums.get("low_change", 0.0),
            "high_change_count": self.weight_sums.get("high_change", 0.0),
        }
        for name in ("transition_valid", "motion_active", "low_change", "high_change"):
            count_key = f"{name}_count"
            metrics[f"{name}_fraction"] = metrics[count_key] / max(
                self.sample_count * horizon_count, 1
            )
        for group in GROUPS:
            count = self.weight_sums.get(group, 0.0)
            for route in (*PREDICTION_ROUTES, "persistence"):
                metrics[f"{route}_{group}_error"] = self.error_sums.get(
                    (group, route), 0.0
                ) / max(count, 1.0)
            correct = metrics[f"correct_{group}_error"]
            for baseline in BASELINES:
                metrics[f"{group}_gain_over_{baseline}"] = _gain(
                    correct, metrics[f"{baseline}_{group}_error"]
                )
            for route in DIAGNOSTIC_ROUTES[1:]:
                metrics[f"{route}_{group}_gain_over_persistence"] = _gain(
                    metrics[f"{route}_{group}_error"],
                    metrics[f"persistence_{group}_error"],
                )
                metrics[f"{route}_{group}_gain_over_correct"] = _gain(
                    metrics[f"{route}_{group}_error"], correct
                )
        for baseline in BASELINES:
            metrics[f"gain_over_{baseline}"] = metrics[f"active_gain_over_{baseline}"]

        valid_count = max(self.weight_sums.get("valid", 0.0), 1.0)
        metrics["source_base_error"] = self.source_base_error_sum / valid_count
        for name, total in self.component_sums.items():
            metrics[f"correct_{name}_error"] = total / valid_count
        gate_count = max(self.gate_sums.get("count", 0.0), 1.0)
        metrics.update(
            {
                "change_gate_mean": self.gate_sums.get("gate", 0.0) / gate_count,
                "change_strength_mean": self.gate_sums.get("target", 0.0) / gate_count,
                "change_gate_mae": self.gate_sums.get("absolute_error", 0.0)
                / gate_count,
                "change_gate_correlation": _correlation(self.gate_sums),
            }
        )
        if self.effect_sum is not None and self.effect_square_sum is not None:
            mean = self.effect_sum / self.effect_count
            variance = self.effect_square_sum / self.effect_count - mean.square()
            metrics["effect_std"] = float(variance.clamp_min(0.0).sqrt().mean())
            metrics["effect_abs_mean"] = self.effect_abs_sum / (
                self.effect_count * self.effect_sum.numel()
            )
        for horizon in self.horizons:
            count = self.horizon_weights.get(horizon, 0.0)
            metrics[f"h{horizon}_motion_active_count"] = count
            for route in (*PREDICTION_ROUTES, "persistence"):
                metrics[f"h{horizon}_{route}_active_error"] = self.horizon_errors.get(
                    (horizon, route), 0.0
                ) / max(count, 1.0)
            correct = metrics[f"h{horizon}_correct_active_error"]
            for baseline in BASELINES:
                metrics[f"h{horizon}_gain_over_{baseline}"] = _gain(
                    correct, metrics[f"h{horizon}_{baseline}_active_error"]
                )
            for route in DIAGNOSTIC_ROUTES[1:]:
                metrics[f"h{horizon}_{route}_gain_over_correct"] = _gain(
                    metrics[f"h{horizon}_{route}_active_error"], correct
                )
        return metrics


def macro_metrics_v60(records: list[dict[str, float]]) -> dict[str, float]:
    metrics = {
        key: sum(record[key] for record in records) / len(records) for key in records[0]
    }
    for group in GROUPS:
        correct = metrics[f"correct_{group}_error"]
        for baseline in BASELINES:
            metrics[f"{group}_gain_over_{baseline}"] = _gain(
                correct, metrics[f"{baseline}_{group}_error"]
            )
        for route in DIAGNOSTIC_ROUTES[1:]:
            metrics[f"{route}_{group}_gain_over_persistence"] = _gain(
                metrics[f"{route}_{group}_error"],
                metrics[f"persistence_{group}_error"],
            )
            metrics[f"{route}_{group}_gain_over_correct"] = _gain(
                metrics[f"{route}_{group}_error"], correct
            )
    for baseline in BASELINES:
        metrics[f"gain_over_{baseline}"] = metrics[f"active_gain_over_{baseline}"]
    for horizon in sorted(metrics_horizons(metrics)):
        correct = metrics[f"h{horizon}_correct_active_error"]
        for baseline in BASELINES:
            metrics[f"h{horizon}_gain_over_{baseline}"] = _gain(
                correct, metrics[f"h{horizon}_{baseline}_active_error"]
            )
        for route in DIAGNOSTIC_ROUTES[1:]:
            metrics[f"h{horizon}_{route}_gain_over_correct"] = _gain(
                metrics[f"h{horizon}_{route}_active_error"], correct
            )
    return metrics


def metrics_horizons(metrics: dict[str, float]) -> set[int]:
    return {
        int(key.split("_", maxsplit=1)[0][1:])
        for key in metrics
        if key.startswith("h") and key.endswith("_correct_active_error")
    }


def bootstrap_gains_v60(records, samples: int, seed: int, group: str = "active"):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(records), (samples, len(records)), generator=generator)
    result = {}
    for route in (*DIAGNOSTIC_ROUTES, "zero", "shuffled"):
        candidate = torch.tensor([row[f"{route}_{group}_error"] for row in records])
        reference = torch.tensor([row[f"persistence_{group}_error"] for row in records])
        sampled_candidate = candidate[indices].mean(dim=1)
        sampled_reference = reference[indices].mean(dim=1)
        gains = (sampled_reference - sampled_candidate) / sampled_reference.clamp_min(
            1e-6
        )
        result[route] = {
            "estimate": float(_gain(float(candidate.mean()), float(reference.mean()))),
            "ci95_low": float(torch.quantile(gains, 0.025)),
            "ci95_high": float(torch.quantile(gains, 0.975)),
        }
    return result


def bootstrap_standard_gains_v60(records, samples: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(records), (samples, len(records)), generator=generator)
    correct = torch.tensor([row["correct_active_error"] for row in records])
    result = {}
    for baseline in BASELINES:
        reference = torch.tensor([row[f"{baseline}_active_error"] for row in records])
        sampled_correct = correct[indices].mean(dim=1)
        sampled_reference = reference[indices].mean(dim=1)
        gains = (sampled_reference - sampled_correct) / sampled_reference.clamp_min(
            1e-6
        )
        result[baseline] = {
            "estimate": float(_gain(float(correct.mean()), float(reference.mean()))),
            "ci95_low": float(torch.quantile(gains, 0.025)),
            "ci95_high": float(torch.quantile(gains, 0.975)),
        }
    return result


def factorization_diagnosis_v60(metrics):
    gate_recovery = metrics["student_oracle_gate_active_gain_over_correct"]
    base_recovery = metrics["teacher_predicted_gate_active_gain_over_correct"]
    joint_recovery = metrics["teacher_oracle_gate_active_gain_over_correct"]
    return {
        "student_source_base_error": metrics["source_base_error"],
        "oracle_gate_gain_over_standard": gate_recovery,
        "teacher_additive_base_gain_over_standard": base_recovery,
        "teacher_additive_base_oracle_gate_gain_over_standard": joint_recovery,
        "additive_base_bottleneck_larger_than_gate_bottleneck": base_recovery
        > gate_recovery,
        "oracle_route_beats_persistence": metrics[
            "teacher_oracle_gate_active_gain_over_persistence"
        ]
        > 0.0,
    }
