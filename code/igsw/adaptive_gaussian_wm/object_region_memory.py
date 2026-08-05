"""Persistent object, scene, and transient region memory for v43."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AdaptiveGaussianWMConfig
from .hierarchical_world_state import RegionMemoryState
from .region_correspondence import CausalRegionCorrespondence


class ObjectRegionMemory(nn.Module):
    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        self.config = config
        dim = config.region_dim
        self.transient_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1),
        )
        nn.init.constant_(self.transient_head[-1].bias, -2.0)
        self.identity_head = nn.Sequential(
            nn.LayerNorm(dim + 2),
            nn.Linear(dim + 2, config.region_identity_dim),
        )
        self.time_input = nn.Sequential(
            nn.Linear(3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.motion_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 2),
        )
        self.feature_predictor = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 2, dim),
        )
        self.correction_gate = nn.Sequential(
            nn.Linear(dim * 2 + 3, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)
        self.correspondence = CausalRegionCorrespondence(config)

    def environment_weight(
        self,
        contextual_feature: torch.Tensor,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        """Softly exclude transient visual regions from environment roots."""
        transient = torch.sigmoid(
            self.transient_head(contextual_feature).squeeze(-1)
        )
        active = activation > 0.5
        order = transient.argsort(dim=1, descending=True)
        rank = torch.empty_like(order)
        indices = torch.arange(transient.shape[1], device=transient.device)
        rank.scatter_(1, order, indices[None].expand_as(order))
        allowance = torch.floor(0.25 * active.sum(dim=1).float()).long()
        allowed = (rank < allowance[:, None]) & active & (transient > 0.5)
        return torch.where(allowed, 1.0 - transient, torch.ones_like(transient))

    @staticmethod
    def _identity_association(feature: torch.Tensor) -> torch.Tensor:
        batch, regions = feature.shape[:2]
        return torch.eye(
            regions, device=feature.device, dtype=feature.dtype
        )[None].expand(batch, -1, -1)

    def _hard_presence(self, presence: torch.Tensor) -> torch.Tensor:
        count = torch.round(presence.detach().sum(dim=1)).long().clamp(
            self.config.min_active_tokens,
            self.config.max_micro_tokens,
        )
        order = presence.argsort(dim=1, descending=True)
        rank = torch.empty_like(order)
        indices = torch.arange(presence.shape[1], device=presence.device)
        rank.scatter_(1, order, indices[None].expand_as(order))
        hard = (rank < count[:, None]).to(presence.dtype)
        return hard.detach() + presence - presence.detach()

    def _scene_quota(
        self,
        scene_logit: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        order = scene_logit.argsort(dim=1, descending=True)
        rank = torch.empty_like(order)
        index = torch.arange(scene_logit.shape[1], device=scene_logit.device)
        rank.scatter_(1, order, index[None].expand_as(order))
        active_count = active.sum(dim=1)
        allowance = torch.floor(
            active_count.float() * self.config.region_scene_fraction
        ).long().clamp_min(1)
        return (rank < allowance[:, None]) & active

    def _enforce_scene_quota(
        self,
        owner: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        allowed = self._scene_quota(owner[..., -2], active)
        scene = owner[..., -2] * allowed.to(owner.dtype)
        transient = owner[..., -1] + owner[..., -2] - scene
        return torch.cat((owner[..., :-2], scene[..., None], transient[..., None]), -1)

    def observe(
        self,
        tokens,
        contextual_feature: torch.Tensor,
        projected_feature: torch.Tensor,
        roots,
    ) -> RegionMemoryState:
        batch, regions, dim = contextual_feature.shape
        if projected_feature.shape != contextual_feature.shape:
            raise ValueError("projected region observations have an invalid shape")
        if roots.assignment.shape[:2] != (batch, regions):
            raise ValueError("root assignment and region count differ")
        feature = F.layer_norm(
            contextual_feature + projected_feature,
            (dim,),
        )
        active = tokens.activation.squeeze(-1) > 0.5
        active_weight = active.to(feature.dtype)
        feature = feature * active_weight[..., None]
        object_logits = roots.assignment.float().clamp_min(1e-6).log()
        scene_logit = roots.background_assignment.float().clamp_min(1e-6).log()
        scene_allowed = self._scene_quota(scene_logit, active)
        scene_logit = torch.where(
            scene_allowed,
            scene_logit,
            scene_logit.new_full((), -20.0),
        )
        transient_logit = self.transient_head(contextual_feature).squeeze(-1)
        owner = torch.softmax(
            torch.cat(
                (object_logits, scene_logit[..., None], transient_logit[..., None]),
                dim=-1,
            ),
            dim=-1,
        ).to(feature.dtype)
        owner = self._enforce_scene_quota(owner, active)
        inactive_owner = torch.zeros_like(owner)
        inactive_owner[..., -1] = 1.0
        owner = torch.where(active[..., None], owner, inactive_owner)
        object_owner = owner[..., : self.config.object_slots]
        object_mass = object_owner.sum(dim=-1, keepdim=True)
        object_center = torch.einsum(
            "brk,bkd->brd",
            object_owner / object_mass.clamp_min(1e-6),
            roots.center,
        )
        center = tokens.center * active_weight[..., None]
        relative_center = object_mass * (center - object_center)
        relative_center = relative_center + (1.0 - object_mass) * center
        identity = F.normalize(
            self.identity_head(torch.cat((feature, relative_center), dim=-1)).float(),
            dim=-1,
        ).to(feature.dtype)
        identity = identity * object_mass.to(identity.dtype)
        activation = tokens.activation.squeeze(-1)
        presence = activation
        visibility = activation
        covariance_eye = torch.eye(2, device=feature.device, dtype=feature.dtype)
        covariance = tokens.covariance * active_weight[..., None, None]
        covariance = covariance + (
            1.0 - active_weight
        )[..., None, None] * self.config.covariance_floor * covariance_eye
        association = self._identity_association(feature)
        state = RegionMemoryState(
            feature=feature,
            center=center,
            covariance=covariance,
            owner=owner,
            relative_center=relative_center,
            activation=activation,
            presence=presence,
            visibility=visibility,
            identity_key=identity,
            observed=visibility,
            update_gate=torch.ones_like(presence),
            association=association,
            association_confidence=presence,
        )
        state.validate(self.config.object_slots)
        return state

    def predict(
        self,
        previous: RegionMemoryState,
        delta_time: torch.Tensor,
        predicted_roots,
    ) -> RegionMemoryState:
        if delta_time.ndim != 1:
            raise ValueError("region delta_time must have shape [B]")
        time = torch.stack(
            (delta_time, delta_time.abs(), torch.tanh(delta_time)), dim=-1
        )
        time_feature = self.time_input(time)[:, None]
        feature = previous.feature + 0.05 * self.feature_predictor(
            previous.feature + time_feature
        )
        center = (
            previous.center
            + 0.25
            * torch.tanh(self.motion_head(previous.feature + time_feature))
            * delta_time[:, None, None]
        ).clamp(-1.25, 1.25)
        object_owner = previous.owner[..., : self.config.object_slots]
        object_mass = object_owner.sum(dim=-1, keepdim=True)
        root_center = torch.einsum(
            "brk,bkd->brd",
            object_owner / object_mass.clamp_min(1e-6),
            predicted_roots.center,
        )
        relative_center = object_mass * (center - root_center)
        relative_center = relative_center + (1.0 - object_mass) * center
        object_mass = previous.owner[..., : self.config.object_slots].sum(dim=-1)
        scene_mass = previous.owner[..., -2]
        transient_mass = previous.owner[..., -1]
        decay_rate = 0.01 * object_mass + 0.05 * scene_mass + 1.5 * transient_mass
        survival = torch.exp(-decay_rate * delta_time.abs()[:, None])
        presence = previous.presence * survival
        visibility = previous.visibility * torch.exp(
            -(0.5 + decay_rate) * delta_time.abs()[:, None]
        )
        return RegionMemoryState(
            feature=feature,
            center=center,
            covariance=previous.covariance,
            owner=previous.owner,
            relative_center=relative_center,
            activation=presence,
            presence=presence,
            visibility=visibility,
            identity_key=previous.identity_key,
            observed=torch.zeros_like(previous.observed),
            update_gate=torch.zeros_like(previous.update_gate),
            association=previous.association,
            association_confidence=torch.zeros_like(previous.association_confidence),
        )

    def correct(
        self,
        predicted: RegionMemoryState,
        observation: RegionMemoryState,
    ) -> RegionMemoryState:
        matched = self.correspondence(predicted, observation)
        gate_input = torch.cat(
            (
                predicted.feature,
                matched.aligned_feature,
                matched.confidence[..., None],
                predicted.presence[..., None],
                matched.aligned_visibility[..., None],
            ),
            dim=-1,
        )
        gate = torch.sigmoid(self.correction_gate(gate_input)).squeeze(-1)
        gate = gate * matched.aligned_visibility * matched.confidence
        birth = (
            (predicted.presence < 0.1).to(gate.dtype)
            * matched.aligned_presence
        )

        def blend(predicted_value, aligned_value):
            weight = gate
            while weight.ndim < predicted_value.ndim:
                weight = weight[..., None]
            return torch.lerp(predicted_value, aligned_value, weight)

        feature = blend(predicted.feature, matched.aligned_feature)
        center = blend(predicted.center, matched.aligned_center)
        covariance = blend(predicted.covariance, matched.aligned_covariance)
        owner = blend(predicted.owner, matched.aligned_owner)
        relative_center = blend(
            predicted.relative_center, matched.aligned_relative_center
        )
        birth_weight = birth
        while birth_weight.ndim < feature.ndim:
            birth_weight = birth_weight[..., None]
        feature = torch.lerp(feature, matched.aligned_feature, birth_weight)
        center = torch.lerp(center, matched.aligned_center, birth[..., None])
        covariance = torch.lerp(
            covariance, matched.aligned_covariance, birth[..., None, None]
        )
        owner = torch.lerp(owner, matched.aligned_owner, birth[..., None])
        relative_center = torch.lerp(
            relative_center, matched.aligned_relative_center, birth[..., None]
        )
        presence = torch.maximum(
            predicted.presence * (1.0 - gate) + matched.aligned_presence * gate,
            birth,
        ).clamp(0.0, 1.0)
        presence = self._hard_presence(presence)
        normalized_owner = owner / owner.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        owner = self._enforce_scene_quota(normalized_owner, presence > 0.5)
        visibility = torch.maximum(matched.aligned_visibility * gate, birth).clamp(
            0.0, 1.0
        ) * presence
        object_mass = owner[..., : self.config.object_slots].sum(dim=-1)
        identity_value = blend(predicted.identity_key, matched.aligned_identity)
        identity_value = torch.lerp(
            identity_value, matched.aligned_identity, birth[..., None]
        )
        identity = F.normalize(identity_value.float(), dim=-1).to(feature.dtype)
        identity = identity * object_mass[..., None]
        return RegionMemoryState(
            feature=feature,
            center=center,
            covariance=covariance,
            owner=owner,
            relative_center=relative_center,
            activation=presence,
            presence=presence,
            visibility=visibility,
            identity_key=identity,
            observed=visibility,
            update_gate=torch.maximum(gate, birth),
            association=matched.matrix,
            association_confidence=matched.confidence,
        )
