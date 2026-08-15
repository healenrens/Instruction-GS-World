"""Causal object state with separate identity, dynamics, geometry and lifecycle."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .recurrent_slot_state import _coordinate_basis, evidence_normalized_attention
from .stable_normalization import stable_rms_normalize, stable_unit_normalize
from .v49_config import TrajectoryObjectStateConfig


class ObjectPatchAdapter(nn.Module):
    def __init__(self, config: TrajectoryObjectStateConfig):
        super().__init__()
        self.appearance = nn.Sequential(
            nn.LayerNorm(config.patch_dim),
            nn.Linear(config.patch_dim, config.identity_dim),
            nn.GELU(),
            nn.Linear(config.identity_dim, config.identity_dim),
        )
        self.dynamic = nn.Sequential(
            nn.LayerNorm(config.patch_dim),
            nn.Linear(config.patch_dim, config.dynamic_dim),
            nn.GELU(),
            nn.Linear(config.dynamic_dim, config.dynamic_dim),
        )
        self.position = nn.Sequential(
            nn.Linear(21, config.state_dim),
            nn.GELU(),
            nn.Linear(config.state_dim, config.state_dim),
        )
        self.key = nn.Linear(config.state_dim, config.state_dim, bias=False)
        self.value = nn.Linear(config.state_dim, config.state_dim, bias=False)

    def forward(
        self, patches: torch.Tensor, coordinates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        appearance = self.appearance(patches)
        dynamic = self.dynamic(patches)
        combined = torch.cat((appearance, dynamic), dim=-1)
        combined = combined + self.position(_coordinate_basis(coordinates))
        return appearance, dynamic, self.key(combined), self.value(combined)


class CausalObjectStatePredictor(nn.Module):
    def __init__(self, config: TrajectoryObjectStateConfig):
        super().__init__()
        self.config = config
        input_dim = config.state_dim + 4
        self.input = nn.Linear(input_dim, config.state_dim)
        self.time = nn.Sequential(
            nn.Linear(5, config.state_dim),
            nn.SiLU(),
            nn.Linear(config.state_dim, config.state_dim),
        )
        self.norm = nn.LayerNorm(config.state_dim)
        self.attention = nn.MultiheadAttention(
            config.state_dim,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, 2 * config.state_dim),
            nn.GELU(),
            nn.Linear(2 * config.state_dim, config.dynamic_dim + 4),
        )

    def forward(
        self,
        identity: torch.Tensor,
        dynamic: torch.Tensor,
        center: torch.Tensor,
        log_scale: torch.Tensor,
        presence: torch.Tensor,
        delta_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dt = delta_time.float().clamp_min(0.0)
        time = torch.stack(
            (
                torch.log1p(dt),
                torch.sin(dt),
                torch.cos(dt),
                torch.sin(0.25 * dt),
                torch.cos(0.25 * dt),
            ),
            dim=-1,
        )
        state = torch.cat(
            (
                identity.float(),
                dynamic.float(),
                center.float(),
                log_scale.float()[..., None],
                presence.float()[..., None],
            ),
            dim=-1,
        )
        hidden = self.input(state) + self.time(time)[:, None]
        attended, _ = self.attention(
            self.norm(hidden), self.norm(hidden), self.norm(hidden), need_weights=False
        )
        update = self.output(hidden + attended)
        dynamic_delta, center_delta, scale_delta, presence_delta = update.split(
            (self.config.dynamic_dim, 2, 1, 1), dim=-1
        )
        predicted_dynamic = stable_rms_normalize(
            dynamic.float() + 0.25 * torch.tanh(dynamic_delta)
        )
        predicted_center = (
            center.float() + 0.10 * torch.tanh(center_delta) * dt[:, None, None]
        ).clamp(-1.25, 1.25)
        predicted_scale = (
            log_scale.float() + 0.10 * torch.tanh(scale_delta.squeeze(-1))
        ).clamp(-3.0, 0.7)
        survival = torch.exp(
            -math.log(2.0) * dt[:, None] / self.config.presence_half_life_seconds
        )
        predicted_presence = (
            presence.float() * survival + 0.05 * torch.tanh(presence_delta.squeeze(-1))
        ).clamp(0.0, 1.0)
        return (
            predicted_dynamic.to(dynamic.dtype),
            predicted_center,
            predicted_scale,
            predicted_presence,
        )


class DisentangledObjectStateEncoder(nn.Module):
    def __init__(self, config: TrajectoryObjectStateConfig):
        super().__init__()
        self.config = config
        self.initial_identity = nn.Parameter(
            torch.empty(1, config.object_slots, config.identity_dim)
        )
        self.initial_dynamic = nn.Parameter(
            torch.empty(1, config.object_slots, config.dynamic_dim)
        )
        nn.init.normal_(self.initial_identity, std=0.02)
        nn.init.normal_(self.initial_dynamic, std=0.02)
        side = math.ceil(config.object_slots**0.5)
        axis = torch.linspace(-0.75, 0.75, side)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        seeds = torch.stack((x, y), dim=-1).reshape(-1, 2)[: config.object_slots]
        self.initial_center = nn.Parameter(seeds[None])
        self.initial_log_scale = nn.Parameter(
            torch.full((1, config.object_slots), -1.25)
        )
        self.initial_presence = nn.Parameter(
            torch.full((1, config.object_slots), -1.5)
        )
        self.scene_state = nn.Parameter(torch.empty(1, 1, config.state_dim))
        self.transient_state = nn.Parameter(torch.empty(1, 1, config.state_dim))
        nn.init.normal_(self.scene_state, std=0.02)
        nn.init.normal_(self.transient_state, std=0.02)
        self.adapter = ObjectPatchAdapter(config)
        self.predictor = CausalObjectStatePredictor(config)
        self.object_query = nn.Linear(config.state_dim, config.state_dim, bias=False)
        self.nuisance_query = nn.Linear(config.state_dim, config.state_dim, bias=False)
        self.identity_update = nn.GRUCell(config.identity_dim, config.identity_dim)
        self.dynamic_update = nn.GRUCell(config.dynamic_dim, config.dynamic_dim)
        self.scene_update = nn.GRUCell(config.state_dim, config.state_dim)
        self.transient_update = nn.GRUCell(config.state_dim, config.state_dim)

    @staticmethod
    def _detach_state(values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(value.detach() for value in values)

    def _correct(
        self,
        identity: torch.Tensor,
        dynamic: torch.Tensor,
        center: torch.Tensor,
        log_scale: torch.Tensor,
        presence: torch.Tensor,
        scene: torch.Tensor,
        patches: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        appearance, dynamic_input, keys, values = self.adapter(patches, coordinates)
        object_state = torch.cat((identity, dynamic), dim=-1)
        object_query = self.object_query(object_state.float())
        nuisance = torch.cat((scene, self.transient_state.expand(len(scene), -1, -1)), dim=1)
        nuisance_query = self.nuisance_query(nuisance.float())
        object_logits = torch.einsum("bkd,bnd->bkn", object_query, keys.float())
        distance = (
            coordinates[:, None].float() - center[:, :, None].float()
        ).square().sum(dim=-1)
        variance = (2.0 * log_scale.float()).exp()[:, :, None].clamp_min(1e-3)
        object_logits = object_logits / self.config.state_dim**0.5
        object_logits = object_logits - 0.25 * distance / variance
        object_logits = object_logits + presence.float().clamp_min(0.05).log()[..., None]
        nuisance_logits = torch.einsum(
            "bod,bnd->bon", nuisance_query, keys.float()
        ) / self.config.state_dim**0.5
        logits = torch.cat((object_logits, nuisance_logits), dim=1)
        logits = logits.masked_fill(~valid[:, None], -1e4)
        owner = logits.softmax(dim=1) * valid[:, None].float()
        object_competition = owner[:, : self.config.object_slots]
        normalized, support = evidence_normalized_attention(object_competition)
        object_mass = object_competition.sum(dim=2)
        appearance_observation = torch.einsum(
            "bkn,bnd->bkd", normalized, appearance.float()
        )
        dynamic_observation = torch.einsum(
            "bkn,bnd->bkd", normalized, dynamic_input.float()
        )
        identity_candidate = self.identity_update(
            appearance_observation.flatten(0, 1), identity.float().flatten(0, 1)
        ).reshape_as(identity)
        dynamic_candidate = self.dynamic_update(
            dynamic_observation.flatten(0, 1), dynamic.float().flatten(0, 1)
        ).reshape_as(dynamic)
        visibility = 1.0 - torch.exp(-object_mass.float() / 2.0)
        identity_gate = support.squeeze(-1) * self.config.identity_update_rate
        corrected_identity = stable_unit_normalize(
            torch.lerp(identity.float(), identity_candidate.float(), identity_gate[..., None])
        )
        corrected_dynamic = stable_rms_normalize(
            torch.lerp(dynamic.float(), dynamic_candidate.float(), support)
        )
        center_observation = torch.einsum(
            "bkn,bnd->bkd", normalized, coordinates.float()
        )
        offset = coordinates[:, None].float() - center_observation[:, :, None]
        spatial_variance = torch.einsum(
            "bkn,bknd->bkd", normalized, offset.square()
        ).mean(dim=-1)
        scale_observation = 0.5 * spatial_variance.clamp_min(1e-4).log()
        corrected_center = torch.lerp(center.float(), center_observation, support)
        corrected_scale = torch.lerp(
            log_scale.float(), scale_observation, support.squeeze(-1)
        ).clamp(-3.0, 0.7)
        corrected_presence = torch.maximum(presence.float(), visibility).clamp(0.0, 1.0)
        nuisance_normalized, _ = evidence_normalized_attention(owner[:, -2:])
        nuisance_observation = torch.einsum(
            "bon,bnd->bod", nuisance_normalized, values.float()
        )
        corrected_scene = self.scene_update(
            nuisance_observation[:, 0], scene[:, 0].float()
        )[:, None]
        transient = self.transient_update(
            nuisance_observation[:, 1], self.transient_state[:, 0].expand(len(scene), -1).float()
        )[:, None]
        return {
            "identity": corrected_identity.to(identity.dtype),
            "dynamic": corrected_dynamic.to(dynamic.dtype),
            "center": corrected_center,
            "log_scale": corrected_scale,
            "presence": corrected_presence,
            "visibility": visibility,
            "scene": corrected_scene.to(scene.dtype),
            "transient": transient.to(scene.dtype),
            "assignment": owner.transpose(1, 2),
            "mass": object_mass,
        }

    def forward(
        self,
        patches: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        frame_times: torch.Tensor,
        observation_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if patches.ndim != 4 or coordinates.shape != (*patches.shape[:3], 2):
            raise ValueError("v49 patch and coordinate shapes differ")
        if valid.shape != patches.shape[:3] or frame_times.shape != patches.shape[:2]:
            raise ValueError("v49 validity or time shape differs")
        if observation_mask.shape != patches.shape[:2]:
            raise ValueError("v49 observation mask shape differs")
        batch, frames = patches.shape[:2]
        identity = stable_unit_normalize(self.initial_identity.expand(batch, -1, -1))
        dynamic = stable_rms_normalize(self.initial_dynamic.expand(batch, -1, -1))
        center = self.initial_center.expand(batch, -1, -1).float()
        log_scale = self.initial_log_scale.expand(batch, -1).float()
        presence = self.initial_presence.sigmoid().expand(batch, -1).float()
        scene = self.scene_state.expand(batch, -1, -1)
        history: dict[str, list[torch.Tensor]] = {}
        previous_time = frame_times[:, 0]
        for index in range(frames):
            if index and index % self.config.bptt_span == 0:
                identity, dynamic, center, log_scale, presence, scene = self._detach_state(
                    (identity, dynamic, center, log_scale, presence, scene)
                )
            delta = frame_times[:, index] - previous_time if index else torch.zeros_like(previous_time)
            if index:
                dynamic, center, log_scale, presence = self.predictor(
                    identity, dynamic, center, log_scale, presence, delta
                )
            corrected = self._correct(
                identity,
                dynamic,
                center,
                log_scale,
                presence,
                scene,
                patches[:, index],
                coordinates[:, index],
                valid[:, index],
            )
            observed = observation_mask[:, index]
            state_mask = observed[:, None, None]
            scalar_mask = observed[:, None]
            identity = torch.where(state_mask, corrected["identity"], identity)
            dynamic = torch.where(state_mask, corrected["dynamic"], dynamic)
            center = torch.where(state_mask, corrected["center"], center)
            log_scale = torch.where(scalar_mask, corrected["log_scale"], log_scale)
            presence = torch.where(scalar_mask, corrected["presence"], presence)
            visibility = torch.where(
                scalar_mask, corrected["visibility"], torch.zeros_like(presence)
            )
            scene = torch.where(state_mask[:, :1], corrected["scene"], scene)
            transient = torch.where(
                state_mask[:, :1],
                corrected["transient"],
                torch.zeros_like(corrected["transient"]),
            )
            assignment = torch.where(
                observed[:, None, None],
                corrected["assignment"],
                torch.zeros_like(corrected["assignment"]),
            )
            values = {
                "identity": identity,
                "dynamic": dynamic,
                "center": center,
                "log_scale": log_scale,
                "presence": presence,
                "visibility": visibility,
                "scene": scene[:, 0],
                "transient": transient[:, 0],
                "assignment": assignment,
                "mass": corrected["mass"],
            }
            for name, value in values.items():
                history.setdefault(name, []).append(value)
            previous_time = frame_times[:, index]
        return {name: torch.stack(values, dim=1) for name, values in history.items()}

