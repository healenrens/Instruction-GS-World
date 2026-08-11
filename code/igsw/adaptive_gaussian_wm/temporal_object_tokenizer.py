"""Recurrent competitive object slots with scene and transient roles."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .v44_config import TemporalObjectSetConfig
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


class TemporalObjectTokenizer(nn.Module):
    """Turn a causal patch stream into persistent object-level world states."""

    def __init__(self, config: TemporalObjectSetConfig):
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
            torch.randn(config.total_slots, dim) / dim**0.5
        )
        self.role_embedding = nn.Parameter(
            torch.randn(config.total_slots, dim) / dim**0.5
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
        self.semantic_decoder = nn.Linear(config.semantic_dim, dim)
        self.spatial_strength = nn.Parameter(torch.tensor(1.0))
        self.object_motion_strength = nn.Parameter(torch.tensor(1.0))
        self.scene_motion_strength = nn.Parameter(torch.tensor(1.0))
        self.transient_cycle_strength = nn.Parameter(torch.tensor(1.0))
        centers = torch.zeros(config.total_slots, 2)
        side = math.ceil(config.object_slots**0.5)
        grid = torch.linspace(-0.75, 0.75, side)
        y, x = torch.meshgrid(grid, grid, indexing="ij")
        centers[: config.object_slots] = torch.stack((x, y), dim=-1).reshape(-1, 2)[
            : config.object_slots
        ]
        self.initial_centers = nn.Parameter(centers)
        self.initial_log_scales = nn.Parameter(
            torch.full((config.total_slots, 2), math.log(0.55))
        )

    def _predict(
        self,
        feature: torch.Tensor,
        center: torch.Tensor,
        log_scale: torch.Tensor,
        presence: torch.Tensor,
        visibility: torch.Tensor,
        delta_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        time = torch.stack((delta_time, torch.log1p(delta_time)), dim=-1)
        feature = feature + self.time_projection(time)[:, None]
        for block in self.predictor:
            feature = block(feature)
        geometry = self.geometry_prediction(feature).float()
        lifecycle = self.lifecycle_prediction(feature).float()
        center = (center + 0.10 * torch.tanh(geometry[..., :2])).clamp(-1.25, 1.25)
        log_scale = (log_scale + 0.05 * torch.tanh(geometry[..., 2:4])).clamp(-3.0, 0.7)
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
        patches: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        center: torch.Tensor,
        log_scale: torch.Tensor,
        residual_motion: torch.Tensor,
        cycle_error: torch.Tensor,
    ) -> torch.Tensor:
        logits = (
            torch.einsum("bsd,bnd->bsn", self.query(slots), self.key(patches))
            / slots.shape[-1] ** 0.5
        )
        object_count = self.config.object_slots
        difference = coordinates[:, None].float() - center[:, :, None]
        precision = torch.exp(-2.0 * log_scale.float()).clamp_max(100.0)
        spatial = (difference.square() * precision[:, :, None]).sum(dim=-1)
        logits[:, :object_count] -= (
            F.softplus(self.spatial_strength) * spatial[:, :object_count]
        )
        motion = residual_motion.float().clamp(0.0, 2.0)
        cycle = cycle_error.float().clamp(0.0, 1.0)
        logits[:, :object_count] += (
            F.softplus(self.object_motion_strength) * motion[:, None]
        )
        logits[:, object_count] -= F.softplus(self.scene_motion_strength) * motion
        logits[:, object_count + 1] += F.softplus(self.transient_cycle_strength) * cycle
        logits = logits.masked_fill(~valid[:, None], -torch.finfo(logits.dtype).max)
        assignment = logits.softmax(dim=1) * valid[:, None]
        return assignment

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
            raise ValueError("tokenizer temporal inputs have incompatible shapes")
        batch, frames, patch_count = frozen_patches.shape[:3]
        if (
            observation_mask.shape != (batch, frames)
            or observation_mask.dtype != torch.bool
        ):
            raise ValueError("tokenizer observation mask must be boolean [B,T]")
        patches = self.patch_projection(frozen_patches)
        slots = (
            self.initial_slots[None].expand(batch, -1, -1) + self.role_embedding[None]
        )
        center = self.initial_centers[None].expand(batch, -1, -1).float()
        log_scale = self.initial_log_scales[None].expand(batch, -1, -1).float()
        presence = torch.full(
            (batch, self.config.total_slots), 0.5, device=patches.device
        )
        visibility = presence.clone()
        histories: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                "semantic",
                "dynamic",
                "center",
                "log_scale",
                "presence",
                "visibility",
                "assignment",
                "decoded_slots",
            )
        }
        for time in range(frames):
            delta = (
                frame_times[:, time]
                if time == 0
                else (frame_times[:, time] - frame_times[:, time - 1])
            )
            predicted = self._predict(
                slots, center, log_scale, presence, visibility, delta.float()
            )
            (
                predicted_slots,
                predicted_center,
                predicted_scale,
                predicted_presence,
                predicted_visibility,
            ) = predicted
            if time == 0:
                motion = torch.zeros(batch, patch_count, device=patches.device)
                cycle = torch.zeros_like(motion)
            else:
                motion = correspondence.residual_motion[:, time - 1]
                cycle = correspondence.cycle_error[:, time - 1]
            observed = observation_mask[:, time]
            causal_motion = motion * observed[:, None]
            causal_cycle = cycle * observed[:, None]
            assignment = self._assignment(
                predicted_slots,
                patches[:, time],
                coordinates[:, time],
                valid[:, time],
                predicted_center,
                predicted_scale,
                causal_motion,
                causal_cycle,
            )
            mass = assignment.sum(dim=-1).clamp_min(1e-6)
            pooled = (
                torch.einsum("bsn,bnd->bsd", assignment, self.value(patches[:, time]))
                / mass[..., None]
            )
            observed_center = (
                torch.einsum("bsn,bnd->bsd", assignment, coordinates[:, time].float())
                / mass[..., None]
            )
            offset = coordinates[:, time, None].float() - observed_center[:, :, None]
            observed_scale = (
                torch.log(
                    torch.einsum("bsn,bsnd->bsd", assignment, offset.square())
                    / mass[..., None]
                    + 1e-3
                )
                * 0.5
            )
            semantic_previous, dynamic_previous = predicted_slots.split(
                (self.config.semantic_dim, self.config.dynamic_dim), dim=-1
            )
            semantic_observation = F.normalize(
                self.semantic_observation(pooled).float(), dim=-1, eps=1e-6
            ).to(pooled.dtype)
            semantic = F.normalize(
                torch.lerp(
                    semantic_previous.float(),
                    semantic_observation.float(),
                    self.config.semantic_update_rate,
                ),
                dim=-1,
                eps=1e-6,
            ).to(pooled.dtype)
            motion_summary = torch.stack(
                (
                    (assignment * causal_motion[:, None]).sum(-1) / mass,
                    (assignment * causal_cycle[:, None]).sum(-1) / mass,
                ),
                dim=-1,
            )
            dynamic_input = self.dynamic_observation(
                torch.cat((pooled, motion_summary.to(pooled.dtype)), dim=-1)
            )
            dynamic = self.dynamic_update(
                dynamic_input.reshape(-1, self.config.dynamic_dim),
                dynamic_previous.reshape(-1, self.config.dynamic_dim),
            ).reshape(batch, self.config.total_slots, self.config.dynamic_dim)
            observed_presence = (
                (mass / valid[:, time].sum(-1, keepdim=True).clamp_min(1))
                .mul(self.config.total_slots)
                .clamp(0.0, 1.0)
            )
            gate = observed[:, None, None]
            slots = torch.cat((semantic, dynamic), dim=-1)
            slots = torch.where(gate, slots, predicted_slots)
            center = torch.where(gate, observed_center, predicted_center)
            log_scale = torch.where(gate, observed_scale, predicted_scale)
            presence = torch.where(
                observed[:, None], observed_presence, predicted_presence
            )
            visibility = torch.where(
                observed[:, None], observed_presence, predicted_visibility * 0.5
            )
            semantic, dynamic = slots.split(
                (self.config.semantic_dim, self.config.dynamic_dim), dim=-1
            )
            histories["semantic"].append(semantic)
            histories["dynamic"].append(dynamic)
            histories["center"].append(center)
            histories["log_scale"].append(log_scale)
            histories["presence"].append(presence)
            histories["visibility"].append(visibility)
            histories["assignment"].append(assignment)
            histories["decoded_slots"].append(self.semantic_decoder(semantic))
        output = {
            name: torch.stack(values, dim=1) for name, values in histories.items()
        }
        output["patch_embedding"] = patches
        return output
