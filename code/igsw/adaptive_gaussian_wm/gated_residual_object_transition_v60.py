"""Source-anchored gated residual Dynamics for v60."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributed_statistics import roll_batch_with_grad
from .latent_object_transition_v59 import (
    ContinuousObjectEffectPosterior,
    ObjectTransitionPrediction,
    TimeCondition,
    TransitionBlock,
)
from .query_persistent_state_v58 import QueryPersistentStateEncoder


@dataclass(frozen=True)
class GatedObjectEffect:
    content: torch.Tensor
    gate_logits: torch.Tensor
    gate: torch.Tensor


@dataclass(frozen=True)
class ResidualTransitionOutput:
    prediction: ObjectTransitionPrediction
    semantic_delta: torch.Tensor
    geometry_delta: torch.Tensor
    visibility_delta: torch.Tensor


def expand_prediction(
    prediction: ObjectTransitionPrediction, horizons: int
) -> ObjectTransitionPrediction:
    return ObjectTransitionPrediction(
        semantic=prediction.semantic[:, None].expand(-1, horizons, -1),
        geometry=prediction.geometry[:, None].expand(-1, horizons, -1),
        visibility_logits=prediction.visibility_logits[:, None].expand(-1, horizons),
    )


class ContinuousChangeGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config.effect_factors * config.effect_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, 1),
        )

    def forward(self, effect: torch.Tensor) -> GatedObjectEffect:
        logits = self.network(effect.flatten(-2)).squeeze(-1).float()
        return GatedObjectEffect(
            content=effect,
            gate_logits=logits,
            gate=torch.sigmoid(logits),
        )


class SourceStateReconstruction(nn.Module):
    """Reconstruct the teacher source state without time or future inputs."""

    def __init__(self, config):
        super().__init__()
        self.semantic_input = nn.Sequential(
            nn.LayerNorm(config.patch_dim),
            nn.Linear(config.patch_dim, config.model_dim),
        )
        self.identity_input = nn.Sequential(
            nn.LayerNorm(config.identity_dim),
            nn.Linear(config.identity_dim, config.model_dim),
        )
        self.geometry_input = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.hidden = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, 2 * config.model_dim),
            nn.GELU(),
            nn.Linear(2 * config.model_dim, config.model_dim),
        )
        self.semantic_correction = nn.Linear(config.model_dim, config.patch_dim)
        self.geometry_correction = nn.Linear(
            config.model_dim, config.target_geometry_dim
        )
        self.visibility_correction = nn.Linear(config.model_dim, 1)
        self._initialize_corrections()

    def _initialize_corrections(self):
        for layer in (
            self.semantic_correction,
            self.geometry_correction,
            self.visibility_correction,
        ):
            nn.init.normal_(layer.weight, std=1e-3)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def source_geometry(source):
        covariance = source.covariance[:, -1]
        return torch.stack(
            (
                source.center[:, -1, 0],
                source.center[:, -1, 1],
                covariance[:, 0, 0],
                covariance[:, 1, 1],
                covariance[:, 0, 1],
            ),
            dim=-1,
        ).float()

    @staticmethod
    def auxiliary(source):
        return torch.cat(
            (
                SourceStateReconstruction.source_geometry(source),
                source.visibility[:, -1, None].float(),
            ),
            dim=-1,
        )

    def forward(self, source):
        source_semantic = F.normalize(
            source.pooled_semantic[:, -1].float(), dim=-1, eps=1e-6
        )
        source_geometry = self.source_geometry(source)
        hidden = (
            self.semantic_input(source_semantic)
            + self.identity_input(source.identity.float())
            + self.geometry_input(self.auxiliary(source))
        )
        hidden = hidden + self.hidden(hidden)
        prediction = ObjectTransitionPrediction(
            semantic=F.normalize(
                source_semantic + self.semantic_correction(hidden), dim=-1, eps=1e-6
            ),
            geometry=source_geometry + self.geometry_correction(hidden).float(),
            visibility_logits=(
                source.visibility_logits[:, -1].float()
                + self.visibility_correction(hidden).squeeze(-1).float()
            ),
        )
        return prediction, hidden


class GatedResidualObjectDynamics(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base = SourceStateReconstruction(config)
        self.effect = nn.Linear(config.effect_dim, config.model_dim, bias=False)
        self.effect_position = nn.Parameter(
            torch.randn(config.effect_factors, config.model_dim) * 0.02
        )
        self.time = TimeCondition(config.model_dim)
        self.blocks = nn.ModuleList(
            TransitionBlock(config) for _ in range(config.transition_layers)
        )
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.semantic_delta = nn.Linear(config.model_dim, config.patch_dim)
        self.geometry_delta = nn.Linear(config.model_dim, config.target_geometry_dim)
        self.visibility_delta = nn.Linear(config.model_dim, 1)
        for layer in (
            self.semantic_delta,
            self.geometry_delta,
            self.visibility_delta,
        ):
            nn.init.normal_(layer.weight, std=1e-3)
            nn.init.zeros_(layer.bias)

    def base_prediction(self, source, horizons: int):
        base, hidden = self.base(source)
        return expand_prediction(base, horizons), hidden

    def forward(self, source, effect, delta_seconds, gate):
        batch, horizons = delta_seconds.shape
        if effect.shape[:2] != (batch, horizons):
            raise ValueError("v60 effect and time horizons differ")
        if gate.shape != (batch, horizons):
            raise ValueError("v60 change gate must have shape [B,K]")
        base, source_hidden = self.base_prediction(source, horizons)
        source_token = source_hidden[:, None].expand(batch, horizons, -1)
        effect_token = self.effect(effect.float()) + self.effect_position
        time_token = self.time(delta_seconds)
        tokens = torch.cat(
            (source_token[:, :, None], effect_token, time_token[:, :, None]), dim=2
        )
        flat = tokens.flatten(0, 1)
        for block in self.blocks:
            flat = block(flat)
        hidden = self.output_norm(flat[:, 0]).reshape(batch, horizons, -1)
        semantic_delta = self.semantic_delta(hidden)
        geometry_delta = self.geometry_delta(hidden).float()
        visibility_delta = self.visibility_delta(hidden).squeeze(-1).float()
        scale = gate.float()
        prediction = ObjectTransitionPrediction(
            semantic=F.normalize(
                base.semantic.float() + scale[..., None] * semantic_delta.float(),
                dim=-1,
                eps=1e-6,
            ),
            geometry=base.geometry.float() + scale[..., None] * geometry_delta,
            visibility_logits=(
                base.visibility_logits.float() + scale * visibility_delta
            ),
        )
        return base, ResidualTransitionOutput(
            prediction=prediction,
            semantic_delta=semantic_delta,
            geometry_delta=geometry_delta,
            visibility_delta=visibility_delta,
        )


class GatedResidualObjectTransitionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = QueryPersistentStateEncoder(config)
        self.encoder.requires_grad_(False)
        self.posterior = ContinuousObjectEffectPosterior(config)
        self.change_gate = ContinuousChangeGate(config)
        self.dynamics = GatedResidualObjectDynamics(config)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def encode_source(self, patches, coordinates, valid, frame_times, query_coordinate):
        with torch.no_grad():
            return self.encoder(
                patches, coordinates, valid, frame_times, query_coordinate
            )

    def forward(
        self,
        patches,
        coordinates,
        valid,
        frame_times,
        query_coordinate,
        transition_target,
    ):
        source = self.encode_source(
            patches, coordinates, valid, frame_times, query_coordinate
        )
        effect = self.posterior(transition_target)
        gated = self.change_gate(effect)
        base, correct = self.dynamics(
            source, effect, transition_target.delta_seconds, gated.gate
        )
        zero = base
        shuffled_effect = roll_batch_with_grad(effect.detach())
        shuffled_gate = roll_batch_with_grad(gated.gate.detach())
        _, shuffled = self.dynamics(
            source,
            shuffled_effect,
            transition_target.delta_seconds,
            shuffled_gate,
        )
        return {
            "source": source,
            "effect": effect,
            "change_gate_logits": gated.gate_logits,
            "change_gate": gated.gate,
            "base": base,
            "correct": correct.prediction,
            "zero": zero,
            "shuffled": shuffled.prediction,
            "correct_residual": correct,
            "shuffled_change_gate": shuffled_gate,
        }
