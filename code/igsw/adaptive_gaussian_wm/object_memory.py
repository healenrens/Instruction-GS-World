"""Persistent predictor-corrector state for object-centric video encoding."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AdaptiveGaussianWMConfig
from .gpstoken import GPSTokenState
from .object_correspondence import (
    CausalObjectCorrespondence,
    align_object_observation,
)
from .object_slots import ObjectSlotState
from .relative_geometry import (
    RelativeGeometryEncoder,
    pairwise_relative_geometry,
    pool_object_geometry,
)


@dataclass
class ObjectMemoryState:
    slots: torch.Tensor
    tracking_slots: torch.Tensor
    assignment: torch.Tensor
    background_assignment: torch.Tensor
    potential_change: torch.Tensor
    potential_change_logits: torch.Tensor
    activity: torch.Tensor
    center: torch.Tensor
    feature: torch.Tensor
    decoded_center: torch.Tensor
    decoded_feature: torch.Tensor
    auxiliary_enabled: bool
    center_auxiliary_enabled: bool
    relative_scale: torch.Tensor
    relative_disparity: torch.Tensor
    relations: torch.Tensor
    existence: torch.Tensor
    in_frame: torch.Tensor
    visibility: torch.Tensor
    update_gate: torch.Tensor
    identity_key: torch.Tensor
    identity_similarity: torch.Tensor
    association_matrix: torch.Tensor
    association_match: torch.Tensor
    association_unmatched: torch.Tensor
    association_discovery: torch.Tensor
    association_entropy: torch.Tensor
    association_support_distance: torch.Tensor
    observation_confidence: torch.Tensor
    birth_evidence: torch.Tensor


@dataclass
class PredictedObjectMemory:
    slots: torch.Tensor
    tracking_slots: torch.Tensor
    center: torch.Tensor
    feature: torch.Tensor
    decoded_feature: torch.Tensor
    relative_scale: torch.Tensor
    relative_disparity: torch.Tensor
    relations: torch.Tensor
    existence: torch.Tensor
    in_frame: torch.Tensor
    identity_key: torch.Tensor


def _inside_frame(center: torch.Tensor) -> torch.Tensor:
    margin = 1.0 - center.abs().amax(dim=-1)
    return torch.sigmoid(10.0 * margin)


class ObjectMemoryTransition(nn.Module):
    """Predict hidden object state, then correct only from visible evidence."""

    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        dim = config.object_dim
        self.motion_scale = config.memory_motion_scale
        self.identity_update_rate = config.identity_memory_update_rate
        self.persistent_identity_key = config.persistent_identity_key
        self.correspondence = (
            CausalObjectCorrespondence(config)
            if config.causal_object_correspondence
            else None
        )
        self.geometry = RelativeGeometryEncoder(config)
        self.time_input = nn.Sequential(
            nn.Linear(3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.tracking_predictor = nn.GRUCell(dim, dim)
        self.semantic_predictor = nn.GRUCell(dim, dim)
        self.motion_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 4),
        )
        self.correction_gate = nn.Sequential(
            nn.LayerNorm(dim * 2 + 3),
            nn.Linear(dim * 2 + 3, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)

    @staticmethod
    def _time_features(delta_time: torch.Tensor) -> torch.Tensor:
        if delta_time.ndim != 1:
            raise ValueError("delta_time must have shape [B]")
        return torch.stack(
            (delta_time, delta_time.abs(), torch.tanh(delta_time)),
            dim=-1,
        )

    def initialize(
        self,
        observation: ObjectSlotState,
        tokens: GPSTokenState,
    ) -> ObjectMemoryState:
        visibility = observation.activity.clamp(0.0, 1.0)
        geometry = pool_object_geometry(
            tokens,
            observation.assignment,
            visibility,
        )
        in_frame = torch.maximum(visibility, _inside_frame(geometry.center))
        identity_key = F.normalize(observation.tracking_slots.float(), dim=-1).to(
            observation.tracking_slots.dtype
        )
        object_count = visibility.shape[-1]
        identity = torch.eye(
            object_count,
            device=visibility.device,
            dtype=visibility.dtype,
        )[None]
        association = identity * visibility[..., None]
        return ObjectMemoryState(
            slots=observation.slots,
            tracking_slots=observation.tracking_slots,
            assignment=observation.assignment,
            background_assignment=observation.background_assignment,
            potential_change=observation.potential_change,
            potential_change_logits=observation.potential_change_logits,
            activity=visibility,
            center=geometry.center,
            feature=observation.feature,
            decoded_center=geometry.center,
            decoded_feature=observation.decoded_feature,
            auxiliary_enabled=observation.auxiliary_enabled,
            center_auxiliary_enabled=observation.center_auxiliary_enabled,
            relative_scale=geometry.relative_scale,
            relative_disparity=geometry.relative_disparity,
            relations=geometry.relations,
            existence=visibility,
            in_frame=in_frame,
            visibility=visibility,
            update_gate=visibility,
            identity_key=identity_key,
            identity_similarity=torch.ones_like(visibility),
            association_matrix=association,
            association_match=visibility,
            association_unmatched=1.0 - visibility,
            association_discovery=1.0 - visibility,
            association_entropy=torch.zeros_like(visibility),
            association_support_distance=torch.zeros_like(visibility),
            observation_confidence=visibility,
            birth_evidence=visibility,
        )

    def predict(
        self,
        previous: ObjectMemoryState,
        delta_time: torch.Tensor,
    ) -> PredictedObjectMemory:
        relation_context = self.geometry(previous.relations)
        context = relation_context + self.time_input(
            self._time_features(delta_time)
        )[:, None]
        shape = previous.tracking_slots.shape
        tracking = self.tracking_predictor(
            context.reshape(-1, shape[-1]),
            previous.tracking_slots.reshape(-1, shape[-1]),
        ).reshape_as(previous.tracking_slots)
        slots = self.semantic_predictor(
            context.reshape(-1, shape[-1]),
            previous.slots.reshape(-1, shape[-1]),
        ).reshape_as(previous.slots)
        motion = self.motion_head(tracking)
        step = torch.tanh(delta_time)[:, None, None]
        transport = self.motion_scale * step * torch.tanh(motion[..., :2])
        center = (
            previous.center
            + previous.relative_scale[..., None] * transport
        )
        relative_scale = previous.relative_scale * torch.exp(
            self.motion_scale * step.squeeze(-1) * torch.tanh(motion[..., 2])
        )
        relative_disparity = (
            previous.relative_disparity
            + self.motion_scale
            * step.squeeze(-1)
            * torch.tanh(motion[..., 3])
        )
        relations = pairwise_relative_geometry(
            center,
            relative_scale,
            relative_disparity,
            previous.visibility,
        )
        return PredictedObjectMemory(
            slots=slots,
            tracking_slots=tracking,
            center=center,
            feature=previous.feature,
            decoded_feature=previous.decoded_feature,
            relative_scale=relative_scale,
            relative_disparity=relative_disparity,
            relations=relations,
            existence=previous.existence,
            in_frame=previous.in_frame,
            identity_key=previous.identity_key,
        )

    def correct(
        self,
        predicted: PredictedObjectMemory,
        observation: ObjectSlotState,
        tokens: GPSTokenState,
    ) -> ObjectMemoryState:
        observed_visibility = observation.activity.clamp(0.0, 1.0)
        observed_geometry = pool_object_geometry(
            tokens,
            observation.assignment,
            observed_visibility,
        )
        association = None
        if self.correspondence is not None:
            association = self.correspondence(
                predicted,
                observation,
                observed_geometry,
            )
            observation = align_object_observation(observation, association)
            observed_visibility = observation.activity.clamp(0.0, 1.0)
            observed_geometry = pool_object_geometry(
                tokens,
                observation.assignment,
                observed_visibility,
            )
        feature_similarity = F.cosine_similarity(
            predicted.decoded_feature,
            observation.decoded_feature,
            dim=-1,
        )
        observed_identity = F.normalize(
            observation.tracking_slots.float(), dim=-1
        ).to(observation.tracking_slots.dtype)
        identity_similarity = (
            association.identity_similarity.to(predicted.identity_key.dtype)
            if association is not None
            else F.cosine_similarity(
                predicted.identity_key,
                observed_identity,
                dim=-1,
            )
        )
        association_similarity = (
            identity_similarity if self.persistent_identity_key else feature_similarity
        )
        center_distance = torch.sqrt(
            (observed_geometry.center - predicted.center)
            .square()
            .sum(dim=-1)
            + 1e-6
        )
        if association is not None:
            center_distance = association.support_distance.to(center_distance.dtype)
        gate_features = torch.cat(
            (
                predicted.tracking_slots,
                observation.tracking_slots,
                observed_visibility[..., None],
                association_similarity[..., None],
                center_distance[..., None],
            ),
            dim=-1,
        )
        update_gate = torch.sigmoid(
            self.correction_gate(gate_features)
        ).squeeze(-1) * observed_visibility
        gate = update_gate[..., None]
        slots = predicted.slots + gate * (observation.slots - predicted.slots)
        tracking = predicted.tracking_slots + gate * (
            observation.tracking_slots - predicted.tracking_slots
        )
        center = predicted.center + gate * (
            observed_geometry.center - predicted.center
        )
        feature = predicted.feature + gate * (
            observation.feature - predicted.feature
        )
        decoded_feature = predicted.decoded_feature + gate * (
            observation.decoded_feature - predicted.decoded_feature
        )
        identity_gate = self.identity_update_rate * gate
        updated_identity = F.normalize(
            predicted.identity_key.float()
            + identity_gate.float() * (
                observed_identity.float() - predicted.identity_key.float()
            ),
            dim=-1,
        ).to(predicted.identity_key.dtype)
        identity_key = torch.where(
            (update_gate > 0.0)[..., None],
            updated_identity,
            predicted.identity_key,
        )
        relative_scale = predicted.relative_scale + update_gate * (
            observed_geometry.relative_scale - predicted.relative_scale
        )
        relative_disparity = predicted.relative_disparity + update_gate * (
            observed_geometry.relative_disparity - predicted.relative_disparity
        )
        inside = _inside_frame(center)
        visibility = observed_visibility * inside
        existence = predicted.existence + (
            1.0 - predicted.existence
        ) * visibility
        in_frame = torch.maximum(
            visibility,
            predicted.in_frame * inside,
        )
        relations = pairwise_relative_geometry(
            center,
            relative_scale,
            relative_disparity,
            visibility,
        )
        if association is None:
            object_count = visibility.shape[-1]
            identity = torch.eye(
                object_count,
                device=visibility.device,
                dtype=visibility.dtype,
            )[None]
            association_matrix = identity * visibility[..., None]
            association_match = visibility
            association_unmatched = 1.0 - visibility
            association_discovery = 1.0 - visibility
            association_entropy = torch.zeros_like(visibility)
            association_distance = center_distance
        else:
            association_matrix = association.transport.to(visibility.dtype)
            association_match = association.match_probability.to(visibility.dtype)
            association_unmatched = association.unmatched_probability.to(
                visibility.dtype
            )
            association_discovery = association.discovery_probability.to(
                visibility.dtype
            )
            association_entropy = association.entropy.to(visibility.dtype)
            association_distance = association.support_distance.to(visibility.dtype)
        birth_evidence = (1.0 - predicted.existence) * visibility
        return ObjectMemoryState(
            slots=slots,
            tracking_slots=tracking,
            assignment=observation.assignment,
            background_assignment=observation.background_assignment,
            potential_change=observation.potential_change,
            potential_change_logits=observation.potential_change_logits,
            activity=visibility,
            center=center,
            feature=feature,
            decoded_center=center,
            decoded_feature=decoded_feature,
            auxiliary_enabled=observation.auxiliary_enabled,
            center_auxiliary_enabled=observation.center_auxiliary_enabled,
            relative_scale=relative_scale,
            relative_disparity=relative_disparity,
            relations=relations,
            existence=existence,
            in_frame=in_frame,
            visibility=visibility,
            update_gate=update_gate,
            identity_key=identity_key,
            identity_similarity=identity_similarity,
            association_matrix=association_matrix,
            association_match=association_match,
            association_unmatched=association_unmatched,
            association_discovery=association_discovery,
            association_entropy=association_entropy,
            association_support_distance=association_distance,
            observation_confidence=visibility,
            birth_evidence=birth_evidence,
        )
