"""Deployable v57 query encoder with training-only binding objectives."""

from __future__ import annotations

import torch.nn as nn

from .query_object_state_v57 import QueryConditionedObjectStateEncoder
from .query_objective_v57 import query_object_binding_terms


class QueryObjectBindingModel(nn.Module):
    """Run three independent single-query views through one shared student."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = QueryConditionedObjectStateEncoder(config)
        self.encoder.dynamic.requires_grad_(False)

    def encode(self, patches, coordinates, valid, frame_times, query_coordinate):
        return self.encoder(
            patches, coordinates, valid, frame_times, query_coordinate
        )

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
        parts = query_object_binding_terms(
            primary,
            alternate,
            negative,
            type("Features", (), {"patches": patches, "valid": valid})(),
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
        }
