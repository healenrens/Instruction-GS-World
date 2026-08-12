"""Causal object tubes with a separate global scene state."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .v45_config import PredictiveObjectTubeConfig
from .video_correspondence import VideoCorrespondence


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
        attended = self.attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        value = value + attended
        return value + self.mlp(self.norm2(value))


def _valid_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.float()
    return (value.float() * weight[..., None]).sum(dim=-2) / weight.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0)


class PredictiveObjectTubeTokenizer(nn.Module):
    """Build persistent objects from scene-residual patch observations."""

    def __init__(self, config: PredictiveObjectTubeConfig):
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
        self.initial_slots = nn.Parameter(
            torch.randn(config.object_slots, dim) / dim**0.5
        )
        self.object_identity = nn.Parameter(
            torch.randn(config.object_slots, dim) / dim**0.5
        )
        self.initial_scene = nn.Parameter(torch.zeros(dim))
        self.scene_projection = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.predictor = nn.ModuleList(
            [_SlotBlock(dim, config.heads, config.dropout) for _ in range(2)]
        )
        self.time_projection = nn.Sequential(
            nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.semantic_observation = nn.Linear(dim, config.semantic_dim)
        self.dynamic_observation = nn.Linear(dim + 2, config.dynamic_dim)
        self.dynamic_update = nn.GRUCell(config.dynamic_dim, config.dynamic_dim)
        self.geometry_prediction = nn.Linear(dim, 5)
        self.lifecycle_prediction = nn.Linear(dim, 2)
        self.tube_time = nn.Sequential(
            nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.tube_decoder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 2, config.patch_dim),
        )
        self.spatial_strength = nn.Parameter(torch.tensor(1.0))
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
        feature: torch.Tensor,
        scene: torch.Tensor,
        center: torch.Tensor,
        log_scale: torch.Tensor,
        presence: torch.Tensor,
        visibility: torch.Tensor,
        delta_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        persistent_semantic = feature[..., : self.config.semantic_dim]
        time = torch.stack((delta_time, torch.log1p(delta_time)), dim=-1)
        feature = feature + self.time_projection(time)[:, None]
        feature = feature + self.scene_projection(scene)[:, None]
        for block in self.predictor:
            feature = block(feature)
        feature = torch.cat(
            (persistent_semantic, feature[..., self.config.semantic_dim :]), dim=-1
        )
        geometry = self.geometry_prediction(feature).float()
        lifecycle = self.lifecycle_prediction(feature).float()
        center = (center + 0.10 * torch.tanh(geometry[..., :2])).clamp(-1.25, 1.25)
        log_scale = (log_scale + 0.05 * torch.tanh(geometry[..., 2:4])).clamp(
            -3.0, 0.7
        )
        presence = torch.sigmoid(
            torch.logit(presence.clamp(1e-4, 1 - 1e-4)) + lifecycle[..., 0]
        )
        visibility = torch.sigmoid(
            torch.logit(visibility.clamp(1e-4, 1 - 1e-4)) + lifecycle[..., 1]
        )
        return feature, center, log_scale, presence, visibility

    def _assignment(
        self,
        slots: torch.Tensor,
        residual_patches: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        center: torch.Tensor,
        log_scale: torch.Tensor,
    ) -> torch.Tensor:
        logits = (
            torch.einsum(
                "bsd,bnd->bsn", self.query(slots), self.key(residual_patches)
            )
            / slots.shape[-1] ** 0.5
        )
        difference = coordinates[:, None].float() - center[:, :, None]
        precision = torch.exp(-2.0 * log_scale.float()).clamp_max(100.0)
        spatial = (difference.square() * precision[:, :, None]).sum(dim=-1)
        logits = logits - F.softplus(self.spatial_strength) * spatial
        logits = logits.masked_fill(~valid[:, None], -torch.finfo(logits.dtype).max)
        return logits.softmax(dim=1) * valid[:, None]

    def _tube_prediction(
        self,
        semantic: torch.Tensor,
        dynamic: torch.Tensor,
        frame_times: torch.Tensor,
    ) -> torch.Tensor:
        delta = frame_times[:, 1:] - frame_times[:, :-1]
        time = torch.stack((delta.float(), torch.log1p(delta.float())), dim=-1)
        state = torch.cat((semantic[:, :-1], dynamic[:, :-1]), dim=-1)
        state = state + self.tube_time(time)[:, :, None].to(state.dtype)
        return self.tube_decoder(state)

    def forward(
        self,
        frozen_patches: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        frame_times: torch.Tensor,
        observation_mask: torch.Tensor,
        correspondence: VideoCorrespondence,
    ) -> dict[str, torch.Tensor]:
        if frozen_patches.ndim != 4 or frame_times.shape != frozen_patches.shape[:2]:
            raise ValueError("v45 tokenizer temporal inputs have incompatible shapes")
        batch, frames, patch_count = frozen_patches.shape[:3]
        if observation_mask.shape != (batch, frames) or observation_mask.dtype != torch.bool:
            raise ValueError("v45 observation mask must be boolean [B,T]")
        patches = self.patch_projection(frozen_patches)
        projected_scene = _valid_mean(patches, valid).to(patches.dtype)
        frozen_scene = _valid_mean(frozen_patches, valid).to(frozen_patches.dtype)
        projected_residual = patches - projected_scene[:, :, None]
        frozen_residual = frozen_patches - frozen_scene[:, :, None]
        slots = self.initial_slots[None].expand(batch, -1, -1)
        slots = slots + self.object_identity[None]
        scene = self.initial_scene[None].expand(batch, -1)
        center = self.initial_centers[None].expand(batch, -1, -1).float()
        log_scale = self.initial_log_scales[None].expand(batch, -1, -1).float()
        presence = torch.full(
            (batch, self.config.object_slots), 0.5, device=patches.device
        )
        visibility = presence.clone()
        names = (
            "semantic",
            "dynamic",
            "center",
            "log_scale",
            "presence",
            "visibility",
            "predicted_presence",
            "predicted_visibility",
            "observed_presence",
            "assignment",
            "slot_mass",
            "correction_gate",
            "scene",
        )
        histories: dict[str, list[torch.Tensor]] = {name: [] for name in names}
        tau = self.config.observation_mass_tau
        for time in range(frames):
            delta = (
                frame_times[:, time]
                if time == 0
                else frame_times[:, time] - frame_times[:, time - 1]
            )
            predicted = self._predict(
                slots, scene, center, log_scale, presence, visibility, delta.float()
            )
            predicted_slots, predicted_center, predicted_scale = predicted[:3]
            predicted_presence, predicted_visibility = predicted[3:]
            observed = observation_mask[:, time]
            current_scene = torch.where(
                observed[:, None], projected_scene[:, time], scene
            )
            assignment = self._assignment(
                predicted_slots,
                projected_residual[:, time],
                coordinates[:, time],
                valid[:, time],
                predicted_center,
                predicted_scale,
            )
            cycle = (
                torch.zeros(batch, patch_count, device=patches.device)
                if time == 0
                else correspondence.cycle_error[:, time - 1]
            )
            confidence = (1.0 - cycle.float()).clamp(0.0, 1.0)
            confidence = confidence * valid[:, time].float() * observed[:, None]
            support = assignment.float() * confidence[:, None]
            mass = support.sum(dim=-1)
            denominator = mass + tau
            correction = mass / denominator
            pooled = torch.einsum(
                "bsn,bnd->bsd", support, self.value(projected_residual[:, time])
            ) / denominator[..., None]
            observed_center = torch.einsum(
                "bsn,bnd->bsd", support, coordinates[:, time].float()
            ) / denominator[..., None]
            offset = coordinates[:, time, None].float() - observed_center[:, :, None]
            observed_scale = 0.5 * torch.log(
                torch.einsum("bsn,bsnd->bsd", support, offset.square())
                / denominator[..., None]
                + 1e-3
            )
            semantic_previous, dynamic_previous = predicted_slots.split(
                (self.config.semantic_dim, self.config.dynamic_dim), dim=-1
            )
            semantic_observation = F.normalize(
                self.semantic_observation(pooled).float(), dim=-1, eps=1e-6
            ).to(pooled.dtype)
            rate = 1.0 if time == 0 else self.config.semantic_update_rate
            semantic_gate = (correction * rate)[..., None]
            semantic = F.normalize(
                torch.lerp(
                    semantic_previous.float(),
                    semantic_observation.float(),
                    semantic_gate.float(),
                ),
                dim=-1,
                eps=1e-6,
            ).to(pooled.dtype)
            motion = torch.zeros_like(confidence) if time == 0 else correspondence.residual_motion[:, time - 1].float()
            motion_summary = torch.stack(
                (
                    (support * motion[:, None]).sum(-1) / denominator,
                    (support * cycle[:, None]).sum(-1) / denominator,
                ),
                dim=-1,
            )
            dynamic_input = self.dynamic_observation(
                torch.cat((pooled, motion_summary.to(pooled.dtype)), dim=-1)
            )
            dynamic_candidate = self.dynamic_update(
                dynamic_input.reshape(-1, self.config.dynamic_dim),
                dynamic_previous.reshape(-1, self.config.dynamic_dim),
            ).reshape(batch, self.config.object_slots, self.config.dynamic_dim)
            dynamic = torch.lerp(
                dynamic_previous.float(),
                dynamic_candidate.float(),
                correction[..., None],
            ).to(dynamic_candidate.dtype)
            nominal_mass = valid[:, time].sum(-1, keepdim=True).float()
            nominal_mass = nominal_mass / self.config.object_slots
            observed_presence = (mass / nominal_mass.clamp_min(1.0)).clamp(0.0, 1.0)
            center = torch.lerp(
                predicted_center, observed_center, correction[..., None]
            )
            log_scale = torch.lerp(
                predicted_scale, observed_scale, correction[..., None]
            ).clamp(-3.0, 0.7)
            presence = torch.lerp(
                predicted_presence, observed_presence, correction
            ).clamp(0.0, 1.0)
            visibility = torch.lerp(
                predicted_visibility, observed_presence, correction
            ).clamp(0.0, 1.0)
            slots = torch.cat((semantic, dynamic), dim=-1)
            scene = current_scene
            values = {
                "semantic": semantic,
                "dynamic": dynamic,
                "center": center,
                "log_scale": log_scale,
                "presence": presence,
                "visibility": visibility,
                "predicted_presence": predicted_presence,
                "predicted_visibility": predicted_visibility,
                "observed_presence": observed_presence,
                "assignment": assignment,
                "slot_mass": mass,
                "correction_gate": correction,
                "scene": scene,
            }
            for name, value in values.items():
                histories[name].append(value)
        output = {
            name: torch.stack(values, dim=1) for name, values in histories.items()
        }
        state = torch.cat((output["semantic"], output["dynamic"]), dim=-1)
        output["decoded_slots"] = self.tube_decoder(state)
        output["patch_embedding"] = frozen_residual
        output["tube_target_patches"] = frozen_residual
        output["tube_prediction"] = self._tube_prediction(
            output["semantic"], output["dynamic"], frame_times
        )
        return output
