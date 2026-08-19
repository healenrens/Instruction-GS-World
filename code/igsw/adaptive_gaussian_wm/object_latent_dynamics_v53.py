"""Continuous latent effects and permutation-invariant object-token dynamics."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .distributed_statistics import roll_batch_with_grad
from .v53_config import SemanticObjectWorldModelConfig


def _time_features(delta_seconds: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    frequency = torch.exp(
        torch.linspace(0.0, -math.log(10_000.0), half, device=delta_seconds.device)
    )
    phase = torch.log1p(delta_seconds.clamp_min(0.0))[:, None] * frequency[None]
    features = torch.cat((phase.sin(), phase.cos()), dim=-1)
    if features.shape[-1] < width:
        features = F.pad(features, (0, width - features.shape[-1]))
    return features


class ContinuousObjectEffectPosterior(nn.Module):
    def __init__(self, config: SemanticObjectWorldModelConfig):
        super().__init__()
        width = config.slot_dim
        self.action_dim = config.action_dim
        self.transition = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.pool_query = nn.Parameter(torch.randn(width) * 0.02)
        self.time_projection = nn.Linear(width, width)
        self.distribution = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, config.action_dim * 2),
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        delta_seconds: torch.Tensor,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transition = self.transition(
            torch.cat((source, target, target - source), dim=-1)
        )
        scores = torch.einsum("bkd,d->bk", transition, self.pool_query)
        pooled = torch.einsum("bk,bkd->bd", scores.softmax(dim=-1), transition)
        pooled = pooled + self.time_projection(
            _time_features(delta_seconds, pooled.shape[-1])
        )
        mean, log_variance = self.distribution(pooled).chunk(2, dim=-1)
        log_variance = log_variance.clamp(-6.0, 2.0)
        if sample:
            effect = mean + torch.randn_like(mean) * (0.5 * log_variance).exp()
        else:
            effect = mean
        return effect, mean, log_variance


class ObjectLevelDynamics(nn.Module):
    def __init__(self, config: SemanticObjectWorldModelConfig):
        super().__init__()
        width = config.slot_dim
        self.object_slots = config.object_slots
        self.state_projection = nn.Linear(width, width)
        self.effect_projection = nn.Linear(config.action_dim, width)
        self.time_projection = nn.Linear(width, width)
        self.scene_projection = nn.Linear(width, width)
        self.slot_position = nn.Parameter(
            torch.randn(1, config.object_slots, width) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.dynamics_heads,
            dim_feedforward=width * config.dynamics_mlp_ratio,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, config.dynamics_depth)
        self.residual = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )

    def forward(
        self,
        source: torch.Tensor,
        scene: torch.Tensor,
        effect: torch.Tensor,
        delta_seconds: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.effect_projection(effect)
        condition = condition + self.time_projection(
            _time_features(delta_seconds, source.shape[-1])
        )
        condition = condition + self.scene_projection(scene.mean(dim=1))
        hidden = self.state_projection(source) + self.slot_position + condition[:, None]
        hidden = self.transformer(hidden)
        return source + self.residual(hidden)


def _sinkhorn_plan(
    prediction: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    iterations: int,
) -> torch.Tensor:
    prediction = F.normalize(prediction.float(), dim=-1, eps=1e-6)
    target = F.normalize(target.float(), dim=-1, eps=1e-6)
    log_plan = torch.einsum("bkd,bjd->bkj", prediction, target) / temperature
    for _ in range(iterations):
        log_plan = log_plan - torch.logsumexp(log_plan, dim=-1, keepdim=True)
        log_plan = log_plan - torch.logsumexp(log_plan, dim=-2, keepdim=True)
    return log_plan.exp()


def _state_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cosine = 1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    normalized_prediction = F.layer_norm(prediction.float(), (prediction.shape[-1],))
    normalized_target = F.layer_norm(target.float(), (target.shape[-1],))
    squared = (normalized_prediction - normalized_target).square().mean(dim=-1)
    return cosine + 0.25 * squared


def latent_dynamics_objective(
    config: SemanticObjectWorldModelConfig,
    posterior: ContinuousObjectEffectPosterior,
    dynamics: ObjectLevelDynamics,
    source: torch.Tensor,
    target: torch.Tensor,
    scene: torch.Tensor,
    delta_seconds: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    effect, mean, log_variance = posterior(
        source, target.detach(), delta_seconds, sample=True
    )
    correct = dynamics(source, scene, effect, delta_seconds)
    zero = dynamics(source, scene, torch.zeros_like(effect), delta_seconds)
    shuffled_effect = roll_batch_with_grad(effect).detach()
    shuffled = dynamics(source, scene, shuffled_effect, delta_seconds)
    with torch.no_grad():
        plan = _sinkhorn_plan(
            correct, target, config.sinkhorn_temperature, config.sinkhorn_iterations
        )
        matched_target = torch.einsum("bkj,bjd->bkd", plan, target.float())
    correct_per_slot = _state_error(correct, matched_target)
    zero_per_slot = _state_error(zero, matched_target)
    shuffled_per_slot = _state_error(shuffled, matched_target)
    change = 1.0 - F.cosine_similarity(source.float(), matched_target, dim=-1)
    change_weight = 1.0 + change.detach() / change.detach().mean().clamp_min(1e-4)
    correct_error = (correct_per_slot * change_weight).mean()
    zero_error = (zero_per_slot * change_weight).mean()
    shuffled_error = (shuffled_per_slot * change_weight).mean()
    delta_prediction = correct - source
    delta_target = matched_target - source.float()
    delta_error = F.smooth_l1_loss(delta_prediction.float(), delta_target)
    kl = -0.5 * (1.0 + log_variance - mean.square() - log_variance.exp()).mean()
    counterfactual = 0.5 * (
        F.relu(config.counterfactual_margin + correct_error - zero_error)
        + F.relu(config.counterfactual_margin + correct_error - shuffled_error)
    )
    loss = (
        correct_error
        + config.delta_state_weight * delta_error
        + config.action_kl_weight * kl
        + config.counterfactual_weight * counterfactual
    )
    parts = {
        "loss": loss.detach(),
        "dynamics_correct_object_error": correct_error.detach(),
        "dynamics_zero_effect_error": zero_error.detach(),
        "dynamics_shuffled_effect_error": shuffled_error.detach(),
        "dynamics_correct_gain_over_zero": (
            (zero_error - correct_error) / zero_error.clamp_min(1e-6)
        ).detach(),
        "dynamics_correct_gain_over_shuffled": (
            (shuffled_error - correct_error) / shuffled_error.clamp_min(1e-6)
        ).detach(),
        "dynamics_delta_state_error": delta_error.detach(),
        "effect_kl": kl.detach(),
        "effect_counterfactual_margin_loss": counterfactual.detach(),
        "effect_mean_norm": mean.norm(dim=-1).mean().detach(),
        "effect_sample_std": effect.float().std(dim=0).mean().detach(),
        "object_target_change": change.mean().detach(),
    }
    outputs = {
        "effect": effect,
        "effect_mean": mean,
        "effect_log_variance": log_variance,
        "prediction": correct,
        "zero_prediction": zero,
        "shuffled_prediction": shuffled,
        "matched_target": matched_target,
    }
    return loss, parts, outputs
