"""Causal object tracks with an explicit, observation-complete scene owner."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .object_state_association import (
    LearnedSlotAssociation,
    align_scalars,
    align_slots,
)
from .v46_config import ObservationCompleteConfig


class _SlotBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(value)
        value = value + self.attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        return value + self.mlp(self.norm2(value))


def _valid_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.float()
    return (value.float() * weight[..., None]).sum(dim=-2) / weight.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0)


def scene_basis(coordinates: torch.Tensor) -> torch.Tensor:
    x, y = coordinates.float().unbind(dim=-1)
    return torch.stack((torch.ones_like(x), x, y, x * y, x.square(), y.square()), -1)


class ObservationCompleteObjectState(nn.Module):
    """Predict persistent tracks, then correct them with independent observations."""

    def __init__(self, config: ObservationCompleteConfig):
        super().__init__()
        config.validate()
        self.config = config
        dim = config.model_dim
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(config.patch_dim),
            nn.Linear(config.patch_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(dim),
        )
        self.initial_slots = nn.Parameter(torch.randn(config.object_slots, dim) / dim**0.5)
        self.track_identity = nn.Parameter(torch.randn(config.object_slots, dim) / dim**0.5)
        self.observation_queries = nn.Parameter(
            torch.randn(config.object_slots, dim) / dim**0.5
        )
        self.initial_scene = nn.Parameter(torch.zeros(dim))
        self.scene_projection = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.scene_update = nn.GRUCell(dim, dim)
        self.predictor = nn.ModuleList(
            [_SlotBlock(dim, config.heads, config.dropout) for _ in range(2)]
        )
        self.time_projection = nn.Sequential(
            nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.scene_query = nn.Linear(dim, dim, bias=False)
        self.semantic_observation = nn.Linear(dim, config.semantic_dim)
        self.dynamic_observation = nn.Linear(dim + 4, config.dynamic_dim)
        self.dynamic_update = nn.GRUCell(config.dynamic_dim, config.dynamic_dim)
        self.geometry_prediction = nn.Linear(dim, 5)
        self.lifecycle_prediction = nn.Linear(dim, 2)
        self.association = LearnedSlotAssociation(config)
        self.object_decoder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 2, config.patch_dim),
        )
        self.scene_decoder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, config.scene_basis_dim * config.patch_dim),
        )
        self.spatial_strength = nn.Parameter(torch.tensor(1.0))
        self.scene_logit_bias = nn.Parameter(torch.tensor(0.0))
        side = math.ceil(config.object_slots**0.5)
        grid = torch.linspace(-0.75, 0.75, side)
        y, x = torch.meshgrid(grid, grid, indexing="ij")
        centers = torch.stack((x, y), dim=-1).reshape(-1, 2)[: config.object_slots]
        self.initial_centers = nn.Parameter(centers)
        self.initial_log_scales = nn.Parameter(
            torch.full((config.object_slots, 2), math.log(0.55))
        )

    def _predict(
        self,
        slots: torch.Tensor,
        scene: torch.Tensor,
        center: torch.Tensor,
        log_scale: torch.Tensor,
        presence: torch.Tensor,
        visibility: torch.Tensor,
        delta_time: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        semantic = slots[..., : self.config.semantic_dim]
        time = torch.stack((delta_time, torch.log1p(delta_time)), dim=-1)
        hidden = slots + self.time_projection(time)[:, None]
        hidden = hidden + self.scene_projection(scene)[:, None]
        for block in self.predictor:
            hidden = block(hidden)
        hidden = torch.cat((semantic, hidden[..., self.config.semantic_dim :]), -1)
        geometry = self.geometry_prediction(hidden).float()
        lifecycle = self.lifecycle_prediction(hidden).float()
        center = (center + 0.10 * torch.tanh(geometry[..., :2])).clamp(-1.25, 1.25)
        log_scale = (log_scale + 0.05 * torch.tanh(geometry[..., 2:4])).clamp(-3.0, 0.7)
        presence = torch.sigmoid(
            torch.logit(presence.clamp(1e-4, 1 - 1e-4)) + lifecycle[..., 0]
        )
        visibility = torch.sigmoid(
            torch.logit(visibility.clamp(1e-4, 1 - 1e-4)) + lifecycle[..., 1]
        )
        return hidden, center, log_scale, presence, visibility

    def _owner_assignment(
        self,
        slots: torch.Tensor,
        scene: torch.Tensor,
        patches: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        center: torch.Tensor,
        log_scale: torch.Tensor,
        presence: torch.Tensor,
        read_appearance: torch.Tensor,
    ) -> torch.Tensor:
        patch_key = self.key(patches)
        appearance = torch.einsum("bkd,bnd->bkn", self.query(slots), patch_key)
        appearance = appearance / slots.shape[-1] ** 0.5
        appearance = appearance * read_appearance[:, None, None].to(appearance.dtype)
        difference = coordinates[:, None].float() - center[:, :, None]
        precision = torch.exp(-2.0 * log_scale.float()).clamp_max(100.0)
        spatial = (difference.square() * precision[:, :, None]).sum(dim=-1)
        object_logits = appearance.float() - F.softplus(self.spatial_strength.float()) * spatial
        object_logits = object_logits + presence.float().clamp_min(1e-4).log()[:, :, None]
        scene_logits = torch.einsum(
            "bd,bnd->bn", self.scene_query(scene), patch_key
        ).float() / slots.shape[-1] ** 0.5
        scene_logits = scene_logits * read_appearance[:, None].float()
        scene_logits = scene_logits + self.scene_logit_bias.float()
        logits = torch.cat((object_logits, scene_logits[:, None]), dim=1)
        logits = logits.masked_fill(~valid[:, None], -torch.finfo(logits.dtype).max)
        assignment = logits.softmax(dim=1) * valid[:, None].float()
        return assignment

    def _candidate_observation(
        self,
        patches: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        scene: torch.Tensor,
        observed: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch = len(patches)
        candidates = self.observation_queries[None].expand(batch, -1, -1)
        assignment = self._owner_assignment(
            candidates,
            scene,
            patches,
            coordinates,
            valid,
            self.initial_centers[None].expand(batch, -1, -1),
            self.initial_log_scales[None].expand(batch, -1, -1),
            torch.ones(batch, self.config.object_slots, device=patches.device),
            observed,
        )[:, : self.config.object_slots]
        support = assignment * observed[:, None, None].float()
        mass = support.sum(dim=-1)
        denominator = mass + self.config.observation_mass_tau
        pooled = torch.einsum(
            "bkn,bnd->bkd", support.to(patches.dtype), self.value(patches)
        ) / denominator[..., None].to(patches.dtype)
        center = torch.einsum(
            "bkn,bnd->bkd", support, coordinates.float()
        ) / denominator[..., None]
        offset = coordinates[:, None].float() - center[:, :, None]
        log_scale = 0.5 * torch.log(
            torch.einsum("bkn,bknd->bkd", support, offset.square())
            / denominator[..., None]
            + 1e-3
        )
        nominal = valid.sum(dim=-1, keepdim=True).float() / self.config.object_slots
        activity = (mass / nominal.clamp_min(1.0)).clamp(0.0, 1.0)
        semantic = F.normalize(
            self.semantic_observation(pooled).float(), dim=-1, eps=1e-6
        ).to(pooled.dtype)
        geometry = torch.cat((center, log_scale), dim=-1).to(pooled.dtype)
        dynamic = self.dynamic_observation(torch.cat((pooled, geometry), dim=-1))
        return torch.cat((semantic, dynamic), -1), center, log_scale, activity, mass

    def forward(
        self,
        frozen_patches: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        frame_times: torch.Tensor,
        observation_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if frozen_patches.ndim != 4 or frame_times.shape != frozen_patches.shape[:2]:
            raise ValueError("v46 state inputs have incompatible shapes")
        batch, frames = frozen_patches.shape[:2]
        if observation_mask.shape != (batch, frames) or observation_mask.dtype != torch.bool:
            raise ValueError("v46 observation mask must be boolean [B,T]")
        patches = self.patch_projection(frozen_patches)
        slots = self.initial_slots[None].expand(batch, -1, -1) + self.track_identity[None]
        scene = self.initial_scene[None].expand(batch, -1)
        center = self.initial_centers[None].expand(batch, -1, -1).float()
        log_scale = self.initial_log_scales[None].expand(batch, -1, -1).float()
        presence = torch.full((batch, self.config.object_slots), 0.5, device=patches.device)
        visibility = presence.clone()
        names = (
            "semantic", "dynamic", "center", "log_scale", "presence", "visibility",
            "predicted_presence", "predicted_visibility", "observed_presence",
            "assignment", "scene_assignment", "slot_mass", "correction_gate", "scene",
            "association_entropy", "association_unmatched", "association_appearance",
        )
        history: dict[str, list[torch.Tensor]] = {name: [] for name in names}
        for time_index in range(frames):
            delta = frame_times[:, time_index] if time_index == 0 else (
                frame_times[:, time_index] - frame_times[:, time_index - 1]
            )
            predicted = self._predict(
                slots, scene, center, log_scale, presence, visibility, delta.float()
            )
            predicted_slots, predicted_center, predicted_scale = predicted[:3]
            predicted_presence, predicted_visibility = predicted[3:]
            observed = observation_mask[:, time_index]
            scene_observation = _valid_mean(patches[:, time_index], valid[:, time_index])
            scene_candidate = self.scene_update(
                scene_observation.to(scene.dtype), scene
            )
            scene_rate = 1.0 if time_index == 0 else self.config.scene_update_rate
            scene_gate = observed[:, None].float() * scene_rate
            scene = torch.lerp(scene.float(), scene_candidate.float(), scene_gate).to(scene.dtype)
            candidates = self._candidate_observation(
                patches[:, time_index], coordinates[:, time_index], valid[:, time_index],
                scene, observed,
            )
            candidate_slots, candidate_center, candidate_scale, activity, candidate_mass = candidates
            association = self.association(
                predicted_slots, predicted_center, predicted_scale, predicted_presence,
                candidate_slots, candidate_center, candidate_scale, activity,
            )
            weights = association.normalized_transport
            aligned_slots = align_slots(weights, candidate_slots)
            aligned_center = align_slots(weights, candidate_center)
            aligned_scale = align_slots(weights, candidate_scale)
            aligned_activity = align_scalars(weights, activity)
            aligned_mass = align_scalars(weights, candidate_mass)
            correction = association.match_probability * aligned_activity * observed[:, None]
            previous_semantic, previous_dynamic = predicted_slots.split(
                (self.config.semantic_dim, self.config.dynamic_dim), dim=-1
            )
            observed_semantic, observed_dynamic = aligned_slots.split(
                (self.config.semantic_dim, self.config.dynamic_dim), dim=-1
            )
            semantic_rate = 1.0 if time_index == 0 else self.config.semantic_update_rate
            semantic = F.normalize(
                torch.lerp(
                    previous_semantic.float(), observed_semantic.float(),
                    (correction * semantic_rate)[..., None],
                ), dim=-1, eps=1e-6,
            ).to(predicted_slots.dtype)
            dynamic_candidate = self.dynamic_update(
                observed_dynamic.reshape(-1, self.config.dynamic_dim),
                previous_dynamic.reshape(-1, self.config.dynamic_dim),
            ).reshape(batch, self.config.object_slots, self.config.dynamic_dim)
            dynamic = torch.lerp(
                previous_dynamic.float(), dynamic_candidate.float(), correction[..., None]
            ).to(predicted_slots.dtype)
            center = torch.lerp(predicted_center, aligned_center, correction[..., None])
            log_scale = torch.lerp(
                predicted_scale, aligned_scale, correction[..., None]
            ).clamp(-3.0, 0.7)
            presence = torch.lerp(predicted_presence, aligned_activity, correction).clamp(0.0, 1.0)
            visibility = torch.lerp(
                predicted_visibility, aligned_activity, correction
            ).clamp(0.0, 1.0)
            slots = torch.cat((semantic, dynamic), dim=-1)
            owners = self._owner_assignment(
                slots, scene, patches[:, time_index], coordinates[:, time_index],
                valid[:, time_index], center, log_scale, presence, observed,
            )
            values = {
                "semantic": semantic, "dynamic": dynamic, "center": center,
                "log_scale": log_scale, "presence": presence, "visibility": visibility,
                "predicted_presence": predicted_presence,
                "predicted_visibility": predicted_visibility,
                "observed_presence": aligned_activity,
                "assignment": owners[:, : self.config.object_slots],
                "scene_assignment": owners[:, self.config.object_slots],
                "slot_mass": aligned_mass,
                "correction_gate": correction,
                "scene": scene,
                "association_entropy": association.entropy,
                "association_unmatched": association.unmatched_probability,
                "association_appearance": association.appearance_similarity,
            }
            for name, value in values.items():
                history[name].append(value)
        output = {name: torch.stack(values, dim=1) for name, values in history.items()}
        state = torch.cat((output["semantic"], output["dynamic"]), dim=-1)
        output["decoded_objects"] = self.object_decoder(state)
        output["scene_coefficients"] = self.scene_decoder(output["scene"]).reshape(
            batch, frames, self.config.scene_basis_dim, self.config.patch_dim
        )
        output["patch_target"] = frozen_patches
        output["identity_key"] = F.normalize(output["semantic"].float(), dim=-1, eps=1e-6)
        return output
