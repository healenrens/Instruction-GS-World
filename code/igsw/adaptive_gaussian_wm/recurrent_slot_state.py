"""Causal recurrent Slot Attention over frozen dense video features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .v48_config import SlotContrastConfig


def _coordinate_basis(coordinates: torch.Tensor) -> torch.Tensor:
    x, y = coordinates.unbind(-1)
    frequencies = (1.0, 2.0, 4.0, 8.0)
    values = [x, y, x.square(), y.square(), x * y]
    for frequency in frequencies:
        values.extend(
            (
                torch.sin(torch.pi * frequency * x),
                torch.cos(torch.pi * frequency * x),
                torch.sin(torch.pi * frequency * y),
                torch.cos(torch.pi * frequency * y),
            )
        )
    return torch.stack(values, dim=-1)


class FrozenFeatureAdapter(nn.Module):
    def __init__(self, config: SlotContrastConfig):
        super().__init__()
        self.feature = nn.Sequential(
            nn.LayerNorm(config.patch_dim),
            nn.Linear(config.patch_dim, config.slot_dim),
            nn.GELU(),
            nn.Linear(config.slot_dim, config.slot_dim),
            nn.LayerNorm(config.slot_dim),
        )
        self.position = nn.Sequential(
            nn.Linear(21, config.slot_dim),
            nn.GELU(),
            nn.Linear(config.slot_dim, config.slot_dim),
        )
        self.output = nn.LayerNorm(config.slot_dim)

    def forward(
        self, patches: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        return self.output(
            self.feature(patches) + self.position(_coordinate_basis(coordinates))
        )


class SlotTemporalPredictor(nn.Module):
    def __init__(self, config: SlotContrastConfig):
        super().__init__()
        self.time = nn.Sequential(
            nn.Linear(5, config.slot_dim),
            nn.SiLU(),
            nn.Linear(config.slot_dim, config.slot_dim),
        )
        self.norm_attention = nn.LayerNorm(config.slot_dim)
        self.attention = nn.MultiheadAttention(
            config.slot_dim,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(config.slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.slot_dim, 4 * config.slot_dim),
            nn.GELU(),
            nn.Linear(4 * config.slot_dim, config.slot_dim),
        )

    def forward(self, slots: torch.Tensor, delta_time: torch.Tensor) -> torch.Tensor:
        dt = delta_time.float().clamp_min(0.0)
        time_features = torch.stack(
            (
                torch.log1p(dt),
                torch.sin(dt),
                torch.cos(dt),
                torch.sin(0.25 * dt),
                torch.cos(0.25 * dt),
            ),
            dim=-1,
        ).to(slots.dtype)
        predicted = slots + self.time(time_features)[:, None]
        normalized = self.norm_attention(predicted)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        predicted = predicted + attended
        return predicted + self.mlp(self.norm_mlp(predicted))


class RecurrentSlotAttention(nn.Module):
    def __init__(self, config: SlotContrastConfig):
        super().__init__()
        self.config = config
        self.initial_slots = nn.Parameter(
            torch.empty(1, config.object_slots, config.slot_dim)
        )
        nn.init.normal_(self.initial_slots, std=0.02)
        self.adapter = FrozenFeatureAdapter(config)
        self.predictor = SlotTemporalPredictor(config)
        self.norm_slots = nn.LayerNorm(config.slot_dim)
        self.norm_inputs = nn.LayerNorm(config.slot_dim)
        self.query = nn.Linear(config.slot_dim, config.slot_dim, bias=False)
        self.key = nn.Linear(config.slot_dim, config.slot_dim, bias=False)
        self.value = nn.Linear(config.slot_dim, config.slot_dim, bias=False)
        self.update = nn.GRUCell(config.slot_dim, config.slot_dim)
        self.norm_update = nn.LayerNorm(config.slot_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(config.slot_dim, 2 * config.slot_dim),
            nn.GELU(),
            nn.Linear(2 * config.slot_dim, config.slot_dim),
        )

    def _correct(
        self,
        predicted: torch.Tensor,
        inputs: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys = self.key(self.norm_inputs(inputs))
        values = self.value(self.norm_inputs(inputs))
        slots = predicted
        attention = None
        for _ in range(self.config.slot_iterations):
            queries = self.query(self.norm_slots(slots))
            logits = torch.einsum("bkd,bnd->bkn", queries, keys)
            logits = logits / self.config.slot_dim**0.5
            logits = logits.float().masked_fill(~valid[:, None], -1e4)
            competition = logits.softmax(dim=1) * valid[:, None].float()
            normalized = competition / competition.sum(dim=2, keepdim=True).clamp_min(1e-6)
            updates = torch.einsum(
                "bkn,bnd->bkd", normalized.to(values.dtype), values
            )
            previous = slots
            slots = self.update(
                updates.flatten(0, 1), previous.flatten(0, 1)
            ).reshape_as(previous)
            slots = slots + self.update_mlp(self.norm_update(slots))
            attention = competition.transpose(1, 2)
        if attention is None:
            raise RuntimeError("slot correction produced no attention")
        return slots, attention

    def forward(
        self,
        patches: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        frame_times: torch.Tensor,
        observation_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if patches.ndim != 4 or coordinates.shape != (*patches.shape[:3], 2):
            raise ValueError("v48 patch and coordinate shapes differ")
        if valid.shape != patches.shape[:3]:
            raise ValueError("v48 valid mask shape differs")
        if frame_times.shape != patches.shape[:2]:
            raise ValueError("v48 frame time shape differs")
        if observation_mask.shape != patches.shape[:2]:
            raise ValueError("v48 observation mask shape differs")
        if not bool(valid.any(dim=2).all()):
            raise ValueError("v48 received a frame without valid patches")
        batch, frames = patches.shape[:2]
        slots = self.initial_slots.expand(batch, -1, -1)
        states, encoder_assignments = [], []
        previous_time = frame_times[:, 0]
        for index in range(frames):
            delta = frame_times[:, index] - previous_time if index else torch.zeros_like(previous_time)
            predicted = self.predictor(slots, delta)
            inputs = self.adapter(patches[:, index], coordinates[:, index])
            corrected, assignment = self._correct(predicted, inputs, valid[:, index])
            observed = observation_mask[:, index, None, None]
            slots = torch.where(observed, corrected, predicted)
            assignment = torch.where(
                observed,
                assignment,
                torch.zeros_like(assignment),
            )
            states.append(slots)
            encoder_assignments.append(assignment)
            previous_time = frame_times[:, index]
        return torch.stack(states, dim=1), torch.stack(encoder_assignments, dim=1)
