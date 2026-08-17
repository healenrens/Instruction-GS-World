"""Causal object memory with disentangled identity, dynamics and lifecycle."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .recurrent_slot_state import _coordinate_basis, evidence_normalized_attention
from .stable_normalization import stable_rms_normalize, stable_unit_normalize


def normalize_support_shape(shape: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            shape[..., :1].float().clamp(0.0, 2.0),
            F.normalize(shape[..., 1:].float(), dim=-1, eps=1e-4),
        ),
        dim=-1,
    )


def support_shape_from_assignment(
    normalized: torch.Tensor,
    coordinates: torch.Tensor,
    center: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    offset = coordinates[:, None].float() - center[:, :, None].float()
    xx = torch.einsum("bkn,bkn->bk", normalized, offset[..., 0].square())
    yy = torch.einsum("bkn,bkn->bk", normalized, offset[..., 1].square())
    xy = torch.einsum("bkn,bkn->bk", normalized, offset[..., 0] * offset[..., 1])
    trace = (xx + yy).clamp_min(1e-4)
    anisotropy = torch.stack((xx - yy, 2.0 * xy), dim=-1)
    magnitude = anisotropy.norm(dim=-1).clamp_max(0.999 * trace)
    direction = F.normalize(anisotropy, dim=-1, eps=1e-4)
    fallback = torch.zeros_like(direction)
    fallback[..., 0] = 1.0
    direction = torch.where((magnitude >= 1e-4)[..., None], direction, fallback)
    major = 0.5 * (trace + magnitude).clamp_min(1e-4)
    minor = 0.5 * (trace - magnitude).clamp_min(1e-4)
    log_aspect = 0.5 * (major / minor).log().clamp(0.0, 4.0)
    support_shape = torch.cat((log_aspect[..., None], direction), dim=-1)
    log_scale = 0.25 * (major * minor).clamp_min(1e-8).log()
    return log_scale.clamp(-3.0, 0.7), support_shape


class PointTrackPatchAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.appearance = nn.Sequential(
            nn.LayerNorm(config.patch_dim),
            nn.Linear(config.patch_dim, config.identity_dim),
            nn.GELU(),
            nn.Linear(config.identity_dim, config.identity_dim),
        )
        self.dynamic = nn.Sequential(
            nn.LayerNorm(config.patch_dim + 3),
            nn.Linear(config.patch_dim + 3, config.dynamic_dim),
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

    def forward(self, patches, coordinates, temporal_residual, residual_flow, confidence):
        appearance = self.appearance(patches)
        dynamic = self.dynamic(
            torch.cat(
                (
                    temporal_residual,
                    residual_flow.to(temporal_residual.dtype),
                    confidence[..., None].to(temporal_residual.dtype),
                ),
                dim=-1,
            )
        )
        combined = torch.cat((appearance, dynamic), dim=-1)
        combined = combined + self.position(_coordinate_basis(coordinates))
        return appearance, dynamic, self.key(combined), self.value(combined)


class CausalObjectMemoryPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input = nn.Linear(config.state_token_dim, config.state_dim)
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
        update_dim = config.dynamic_dim + 2 + 1 + config.support_shape_dim + 1
        self.output = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, 2 * config.state_dim),
            nn.GELU(),
            nn.Linear(2 * config.state_dim, update_dim),
        )

    def forward(self, state: dict[str, torch.Tensor], delta_time: torch.Tensor) -> dict:
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
        token = torch.cat(
            (
                state["identity"].float(),
                state["dynamic"].float(),
                state["center"].float(),
                state["log_scale"].float()[..., None],
                state["support_shape"].float(),
                state["presence"].float()[..., None],
                state["visibility"].float()[..., None],
                state["unobserved_time"].float()[..., None],
            ),
            dim=-1,
        )
        hidden = self.input(token) + self.time(time)[:, None]
        attended, _ = self.attention(
            self.norm(hidden), self.norm(hidden), self.norm(hidden), need_weights=False
        )
        update = self.output(hidden + attended)
        sizes = (self.config.dynamic_dim, 2, 1, 3, 1)
        dynamic, center, scale, shape, presence = update.split(sizes, dim=-1)
        shape_direction = F.normalize(
            state["support_shape"][..., 1:].float() + 0.10 * torch.tanh(shape[..., 1:]),
            dim=-1,
            eps=1e-4,
        )
        prior_presence = state["presence"].float().clamp(1e-4, 1.0 - 1e-4)
        prior_logit = torch.logit(prior_presence)
        presence_rate = torch.tanh(presence.squeeze(-1))
        return {
            "identity": state["identity"],
            "dynamic": stable_rms_normalize(
                state["dynamic"].float() + 0.25 * torch.tanh(dynamic)
            ).to(state["dynamic"].dtype),
            "center": (
                state["center"].float() + 0.10 * torch.tanh(center) * dt[:, None, None]
            ).clamp(-1.25, 1.25),
            "log_scale": (
                state["log_scale"].float() + 0.10 * torch.tanh(scale.squeeze(-1))
            ).clamp(-3.0, 0.7),
            "support_shape": normalize_support_shape(torch.cat(
                (
                    (state["support_shape"][..., :1].float() + 0.10 * torch.tanh(shape[..., :1])).clamp(0.0, 2.0),
                    shape_direction,
                ),
                dim=-1,
            )),
            "presence": torch.sigmoid(prior_logit + presence_rate * dt[:, None]),
            "visibility": torch.zeros_like(state["visibility"].float()),
            "unobserved_time": state["unobserved_time"].float() + dt[:, None],
        }


class PointTrackObjectStateEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.initial_identity = nn.Parameter(torch.empty(1, config.object_slots, config.identity_dim))
        self.initial_dynamic = nn.Parameter(torch.empty(1, config.object_slots, config.dynamic_dim))
        nn.init.normal_(self.initial_identity, std=0.02)
        nn.init.normal_(self.initial_dynamic, std=0.02)
        side = math.ceil(config.object_slots**0.5)
        axis = torch.linspace(-0.75, 0.75, side)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.initial_center = nn.Parameter(
            torch.stack((x, y), dim=-1).reshape(-1, 2)[: config.object_slots][None]
        )
        self.initial_log_scale = nn.Parameter(torch.full((1, config.object_slots), -1.25))
        shape = torch.zeros(1, config.object_slots, 3)
        shape[..., 1] = 1.0
        self.initial_support_shape = nn.Parameter(shape)
        self.initial_presence = nn.Parameter(torch.full((1, config.object_slots), -1.5))
        self.scene_state = nn.Parameter(torch.empty(1, 1, config.state_dim))
        self.transient_state = nn.Parameter(torch.empty(1, 1, config.state_dim))
        nn.init.normal_(self.scene_state, std=0.02)
        nn.init.normal_(self.transient_state, std=0.02)
        self.adapter = PointTrackPatchAdapter(config)
        self.predictor = CausalObjectMemoryPredictor(config)
        self.object_query = nn.Linear(config.state_dim, config.state_dim, bias=False)
        self.nuisance_query = nn.Linear(config.state_dim, config.state_dim, bias=False)
        self.identity_update = nn.GRUCell(config.identity_dim, config.identity_dim)
        self.dynamic_update = nn.GRUCell(config.dynamic_dim, config.dynamic_dim)
        self.scene_update = nn.GRUCell(config.state_dim, config.state_dim)
        self.transient_update = nn.GRUCell(config.state_dim, config.state_dim)

    def _initial_state(self, batch: int) -> dict[str, torch.Tensor]:
        return {
            "identity": stable_unit_normalize(self.initial_identity.expand(batch, -1, -1)),
            "dynamic": stable_rms_normalize(self.initial_dynamic.expand(batch, -1, -1)),
            "center": self.initial_center.expand(batch, -1, -1).float(),
            "log_scale": self.initial_log_scale.expand(batch, -1).float(),
            "support_shape": normalize_support_shape(
                self.initial_support_shape.expand(batch, -1, -1)
            ),
            "presence": self.initial_presence.sigmoid().expand(batch, -1).float(),
            "visibility": torch.zeros(batch, self.config.object_slots, device=self.initial_presence.device),
            "unobserved_time": torch.zeros(
                batch, self.config.object_slots, device=self.initial_presence.device
            ),
            "scene": self.scene_state.expand(batch, -1, -1),
        }

    def _correct(
        self,
        state,
        patches,
        coordinates,
        valid,
        temporal_residual,
        residual_flow,
        confidence,
    ) -> dict[str, torch.Tensor]:
        appearance, dynamic_input, keys, values = self.adapter(
            patches,
            coordinates,
            temporal_residual,
            residual_flow,
            confidence,
        )
        object_state = torch.cat((state["identity"], state["dynamic"]), dim=-1)
        object_logits = torch.einsum(
            "bkd,bnd->bkn", self.object_query(object_state.float()), keys.float()
        ) / self.config.state_dim**0.5
        offset = coordinates[:, None].float() - state["center"][:, :, None].float()
        angle = 0.5 * torch.atan2(
            state["support_shape"][..., 2], state["support_shape"][..., 1]
        )
        cosine, sine = angle.cos()[:, :, None], angle.sin()[:, :, None]
        major = offset[..., 0] * cosine + offset[..., 1] * sine
        minor = -offset[..., 0] * sine + offset[..., 1] * cosine
        aspect = state["support_shape"][..., 0].exp()[:, :, None]
        variance = (2.0 * state["log_scale"]).exp()[:, :, None].clamp_min(1e-3)
        distance = major.square() / (variance * aspect) + minor.square() / (variance / aspect)
        object_logits = object_logits - 0.25 * distance
        object_logits = object_logits + state["presence"].clamp_min(0.05).log()[..., None]
        nuisance = torch.cat(
            (state["scene"], self.transient_state.expand(len(patches), -1, -1)), dim=1
        )
        nuisance_logits = torch.einsum(
            "bod,bnd->bon", self.nuisance_query(nuisance.float()), keys.float()
        ) / self.config.state_dim**0.5
        logits = torch.cat((object_logits, nuisance_logits), dim=1)
        owner = logits.masked_fill(~valid[:, None], -1e4).softmax(dim=1)
        owner = owner * valid[:, None].float()
        objects = owner[:, : self.config.object_slots]
        normalized, support = evidence_normalized_attention(objects)
        mass = objects.sum(dim=2)
        appearance_observation = torch.einsum("bkn,bnd->bkd", normalized, appearance.float())
        dynamic_observation = torch.einsum("bkn,bnd->bkd", normalized, dynamic_input.float())
        identity_candidate = self.identity_update(
            appearance_observation.flatten(0, 1), state["identity"].float().flatten(0, 1)
        ).reshape_as(state["identity"])
        dynamic_candidate = self.dynamic_update(
            dynamic_observation.flatten(0, 1), state["dynamic"].float().flatten(0, 1)
        ).reshape_as(state["dynamic"])
        visibility = 1.0 - torch.exp(-mass.float() / 2.0)
        identity = stable_unit_normalize(
            torch.lerp(
                state["identity"].float(),
                identity_candidate.float(),
                support.float() * self.config.identity_update_rate,
            )
        )
        dynamic = stable_rms_normalize(
            torch.lerp(
                state["dynamic"].float(), dynamic_candidate.float(), support.float()
            )
        )
        center_observation = torch.einsum("bkn,bnd->bkd", normalized, coordinates.float())
        log_scale, support_shape = support_shape_from_assignment(
            normalized, coordinates, center_observation
        )
        scalar_support = support.squeeze(-1)
        nuisance_normalized, _ = evidence_normalized_attention(owner[:, -2:])
        nuisance_observation = torch.einsum("bon,bnd->bod", nuisance_normalized, values.float())
        scene = self.scene_update(nuisance_observation[:, 0], state["scene"][:, 0].float())[:, None]
        transient = self.transient_update(
            nuisance_observation[:, 1], self.transient_state[:, 0].expand(len(patches), -1).float()
        )[:, None]
        blended_shape = normalize_support_shape(
            torch.lerp(
                state["support_shape"].float(),
                support_shape.float(),
                support.float(),
            )
        )
        return {
            "identity": identity.to(state["identity"].dtype),
            "dynamic": dynamic.to(state["dynamic"].dtype),
            "center": torch.lerp(
                state["center"].float(),
                center_observation.float(),
                support.float(),
            ),
            "log_scale": torch.lerp(
                state["log_scale"].float(),
                log_scale.float(),
                scalar_support.float(),
            ),
            "support_shape": blended_shape,
            "presence": torch.where(
                visibility >= self.config.lifecycle_visible_track_fraction,
                torch.maximum(state["presence"].float(), visibility),
                state["presence"].float(),
            ),
            "visibility": visibility,
            "unobserved_time": torch.where(
                visibility >= self.config.lifecycle_visible_track_fraction,
                torch.zeros_like(state["unobserved_time"].float()),
                state["unobserved_time"].float(),
            ),
            "scene": scene.to(state["scene"].dtype),
            "transient": transient.to(state["scene"].dtype),
            "assignment": owner.transpose(1, 2),
            "mass": mass,
        }

    def forward(
        self,
        patches,
        coordinates,
        valid,
        frame_times,
        temporal_residual,
        residual_flow,
        confidence,
    ) -> dict[str, torch.Tensor]:
        if patches.ndim != 4 or coordinates.shape != (*patches.shape[:3], 2):
            raise ValueError("v51 patch and coordinate shapes differ")
        if valid.shape != patches.shape[:3] or frame_times.shape != patches.shape[:2]:
            raise ValueError("v51 validity or time shape differs")
        if temporal_residual.shape != patches.shape:
            raise ValueError("v51 temporal feature residual shape differs")
        if residual_flow.shape != (*patches.shape[:3], 2):
            raise ValueError("v51 residual-flow shape differs")
        if confidence.shape != patches.shape[:3]:
            raise ValueError("v51 tracklet confidence shape differs")
        batch, frames = patches.shape[:2]
        state = self._initial_state(batch)
        history: dict[str, list[torch.Tensor]] = {}
        previous_time = frame_times[:, 0]
        for index in range(frames):
            if index and index % self.config.bptt_span == 0:
                state = {name: value.detach() for name, value in state.items()}
            if index:
                state = {
                    **self.predictor(state, frame_times[:, index] - previous_time),
                    "scene": state["scene"],
                }
            corrected = self._correct(
                state,
                patches[:, index],
                coordinates[:, index],
                valid[:, index],
                temporal_residual[:, index],
                residual_flow[:, index],
                confidence[:, index],
            )
            state = {
                name: corrected[name]
                for name in (
                    "identity", "dynamic", "center", "log_scale",
                    "support_shape", "presence", "visibility", "unobserved_time", "scene",
                )
            }
            for name, value in corrected.items():
                history.setdefault(name, []).append(value)
            previous_time = frame_times[:, index]
        return {name: torch.stack(values, dim=1) for name, values in history.items()}
