"""Position-conditioned SlotMixer decoder for dense frozen-DINO features."""

from __future__ import annotations

import torch
import torch.nn as nn

from .recurrent_slot_state import _coordinate_basis
from .stable_normalization import stable_unit_normalize
from .v48_config import SlotContrastConfig


class _MixerLayer(nn.Module):
    def __init__(self, config: SlotContrastConfig):
        super().__init__()
        self.norm_query = nn.LayerNorm(config.slot_dim)
        self.norm_slots = nn.LayerNorm(config.slot_dim)
        self.cross_attention = nn.MultiheadAttention(
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

    def forward(
        self,
        query: torch.Tensor,
        slots: torch.Tensor,
        slot_valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_slots = self.norm_slots(slots)
        mixed, attention = self.cross_attention(
            self.norm_query(query),
            normalized_slots,
            normalized_slots,
            key_padding_mask=None if slot_valid is None else ~slot_valid,
            need_weights=True,
            average_attn_weights=False,
        )
        query = query + mixed
        query = query + self.mlp(self.norm_mlp(query))
        return query, attention.mean(dim=1)


class PositionConditionedSlotMixer(nn.Module):
    def __init__(self, config: SlotContrastConfig):
        super().__init__()
        self.position = nn.Sequential(
            nn.Linear(21, config.slot_dim),
            nn.GELU(),
            nn.Linear(config.slot_dim, config.slot_dim),
        )
        self.layers = nn.ModuleList(
            [_MixerLayer(config) for _ in range(config.decoder_layers)]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.slot_dim),
            nn.Linear(config.slot_dim, 2 * config.slot_dim),
            nn.GELU(),
            nn.Linear(2 * config.slot_dim, config.patch_dim),
        )

    def forward(
        self,
        slots: torch.Tensor,
        coordinates: torch.Tensor,
        slot_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if slots.ndim != 3 or coordinates.ndim != 3:
            raise ValueError("SlotMixer expects [B,K,D] slots and [B,N,2] coordinates")
        if slots.shape[0] != coordinates.shape[0]:
            raise ValueError("SlotMixer batch dimensions differ")
        if slot_valid is not None and slot_valid.shape != slots.shape[:2]:
            raise ValueError("SlotMixer slot validity shape differs")
        if slot_valid is not None and not bool(slot_valid.any(dim=1).all()):
            raise ValueError("SlotMixer cannot delete every slot")
        query = self.position(_coordinate_basis(coordinates))
        assignment = None
        for layer in self.layers:
            query, assignment = layer(query, slots, slot_valid)
        if assignment is None:
            raise RuntimeError("SlotMixer produced no assignment")
        decoded = stable_unit_normalize(self.output(query))
        return decoded, assignment.float()
