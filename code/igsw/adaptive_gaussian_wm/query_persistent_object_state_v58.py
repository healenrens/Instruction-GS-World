"""Deployable v58 persistent query state with training-only external targets."""

from __future__ import annotations

import torch.nn as nn

from .query_persistent_objective_v58 import query_persistent_state_terms
from .query_persistent_state_v58 import QueryPersistentStateEncoder


class QueryPersistentObjectStateModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = QueryPersistentStateEncoder(config)
        self.motion_readout = nn.Sequential(
            nn.LayerNorm(config.dynamic_dim),
            nn.Linear(config.dynamic_dim, config.dynamic_dim // 2),
            nn.GELU(),
            nn.Linear(config.dynamic_dim // 2, 2),
        )

    def encode(self, patches, coordinates, valid, frame_times, query_coordinate):
        return self.encoder(patches, coordinates, valid, frame_times, query_coordinate)

    def forward(
        self,
        patches,
        coordinates,
        valid,
        frame_times,
        teacher,
        observed_evidence,
        grid_hw,
    ):
        primary = self.encode(
            patches, coordinates, valid, frame_times, teacher.query_coordinate
        )
        alternate = self.encode(
            patches, coordinates, valid, frame_times, teacher.alternate_coordinate
        )
        negative = self.encode(
            patches, coordinates, valid, frame_times, teacher.negative_coordinate
        )
        motion_prediction = self.motion_readout(primary.dynamic)
        features = type("Features", (), {"patches": patches, "valid": valid})()
        parts = query_persistent_state_terms(
            primary,
            alternate,
            negative,
            motion_prediction,
            features,
            observed_evidence,
            teacher,
            grid_hw,
            self.config,
        )
        return {
            "loss": parts["total"],
            "parts": parts,
            "primary": primary,
            "alternate": alternate,
            "negative": negative,
            "motion_prediction": motion_prediction,
        }
