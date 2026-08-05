"""Persistent object, scene, and transient region memory for v43/v44."""
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
        self.video_projection = None
        self.video_gate = None
        self.video_decoder = None
        if config.video_vae_feature_dim > 0:
            self.video_projection = nn.Sequential(
                nn.LayerNorm(config.video_vae_feature_dim),
                nn.Linear(config.video_vae_feature_dim, dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(dim, dim),
            )
            self.video_gate = nn.Sequential(
                nn.Linear(dim * 2, dim // 2),
                nn.SiLU(),
                nn.Linear(dim // 2, 1),
            )
            nn.init.constant_(self.video_gate[-1].bias, -1.5)
            self.video_decoder = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(dim, config.video_vae_feature_dim),
            )

    def decode_video_feature(self, feature: torch.Tensor) -> torch.Tensor:
        if self.video_decoder is None:
            raise ValueError("region memory has no video-detail decoder")
        return self.video_decoder(feature)

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
        auxiliary_feature: torch.Tensor | None = None,
        auxiliary_valid: torch.Tensor | None = None,
    ) -> RegionMemoryState:
        batch, regions, dim = contextual_feature.shape
        if projected_feature.shape != contextual_feature.shape:
            raise ValueError("projected region observations have an invalid shape")
        if roots.assignment.shape[:2] != (batch, regions):
            raise ValueError("root assignment and region count differ")
        semantic_feature = F.layer_norm(
            contextual_feature + projected_feature,
            (dim,),
        )
        active = tokens.activation.squeeze(-1) > 0.5
        if self.config.video_vae_feature_dim > 0:
            expected = (batch, regions, self.config.video_vae_feature_dim)
            if auxiliary_feature is None or auxiliary_feature.shape != expected:
                raise ValueError(f"v44 auxiliary region feature must have {expected}")
            if auxiliary_valid is None or auxiliary_valid.shape != (batch,):
                raise ValueError("v44 auxiliary validity must have shape [B]")
            valid_weight = auxiliary_valid[:, None, None].to(
                semantic_feature.dtype
            )
            detail_latent = auxiliary_feature.to(semantic_feature.dtype) * valid_weight
            detail = self.video_projection(detail_latent)
            gate = torch.sigmoid(
                self.video_gate(torch.cat((semantic_feature, detail), dim=-1))
            ) * valid_weight
            feature = F.layer_norm(semantic_feature + gate * detail, (dim,))
            detail_valid = active.to(feature.dtype) * auxiliary_valid[:, None].to(
                feature.dtype
            )
            detail_gate = gate.squeeze(-1) * active.to(gate.dtype)
        else:
            if auxiliary_feature is not None or auxiliary_valid is not None:
                raise ValueError("v43 cannot receive video VAE features")
            feature = semantic_feature
            detail_latent = feature.new_zeros((batch, regions, 0))
            detail_valid = feature.new_zeros((batch, regions))
            detail_gate = feature.new_zeros((batch, regions))
        feature = torch.where(active[..., None], feature, torch.zeros_like(feature))
        detail_latent = torch.where(
            active[..., None], detail_latent, torch.zeros_like(detail_latent)
        )
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
        center = torch.where(
            active[..., None], tokens.center, torch.zeros_like(tokens.center)
        )
        relative_center = object_mass * (center - object_center)
        relative_center = relative_center + (1.0 - object_mass) * center
        identity = F.normalize(
            self.identity_head(
                torch.cat((semantic_feature, relative_center), dim=-1)
            ).float(),
            dim=-1,
        ).to(feature.dtype)
        identity = identity * object_mass.to(identity.dtype)
        activation = tokens.activation.squeeze(-1)
        presence = activation
        visibility = activation
        covariance_eye = torch.eye(2, device=feature.device, dtype=feature.dtype)
        inactive_covariance = self.config.covariance_floor * covariance_eye
        covariance = torch.where(
            active[..., None, None],
            tokens.covariance,
            inactive_covariance,
        )
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
            detail_latent=detail_latent,
            detail_valid=detail_valid,
            detail_gate=detail_gate,
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
            detail_latent=previous.detail_latent,
            detail_valid=previous.detail_valid * survival,
            detail_gate=previous.detail_gate * survival,
        )

    def correct(
        self,
        predicted: RegionMemoryState,
        observation: RegionMemoryState,
    ) -> RegionMemoryState:
        matched = self.correspondence(predicted, observation)
        aligned = {
            "feature": matched.aligned_feature.to(predicted.feature.dtype),
            "center": matched.aligned_center.to(predicted.center.dtype),
            "covariance": matched.aligned_covariance.to(
                predicted.covariance.dtype
            ),
            "owner": matched.aligned_owner.to(predicted.owner.dtype),
            "relative_center": matched.aligned_relative_center.to(
                predicted.relative_center.dtype
            ),
            "presence": matched.aligned_presence.to(predicted.presence.dtype),
            "visibility": matched.aligned_visibility.to(
                predicted.visibility.dtype
            ),
            "identity": matched.aligned_identity.to(predicted.identity_key.dtype),
            "detail_latent": matched.aligned_detail_latent.to(
                predicted.detail_latent.dtype
            ),
            "detail_valid": matched.aligned_detail_valid.to(
                predicted.detail_valid.dtype
            ),
            "detail_gate": matched.aligned_detail_gate.to(
                predicted.detail_gate.dtype
            ),
        }
        confidence = matched.confidence.to(predicted.presence.dtype)
        gate_input = torch.cat(
            (
                predicted.feature,
                aligned["feature"],
                confidence[..., None].to(predicted.feature.dtype),
                predicted.presence[..., None].to(predicted.feature.dtype),
                aligned["visibility"][..., None].to(predicted.feature.dtype),
            ),
            dim=-1,
        )
        gate = torch.sigmoid(self.correction_gate(gate_input)).squeeze(-1)
        gate = gate.to(predicted.presence.dtype)
        gate = gate * aligned["visibility"] * confidence
        birth = (
            (predicted.presence < 0.1).to(gate.dtype)
            * aligned["presence"]
        )

        def blend(predicted_value, aligned_value, blend_weight):
            aligned_value = aligned_value.to(predicted_value.dtype)
            blend_weight = blend_weight.to(predicted_value.dtype)
            while blend_weight.ndim < predicted_value.ndim:
                blend_weight = blend_weight[..., None]
            return torch.lerp(predicted_value, aligned_value, blend_weight)

        feature = blend(predicted.feature, aligned["feature"], gate)
        center = blend(predicted.center, aligned["center"], gate)
        covariance = blend(predicted.covariance, aligned["covariance"], gate)
        owner = blend(predicted.owner, aligned["owner"], gate)
        relative_center = blend(
            predicted.relative_center, aligned["relative_center"], gate
        )
        detail_latent = blend(
            predicted.detail_latent, aligned["detail_latent"], gate
        )
        detail_gate = blend(predicted.detail_gate, aligned["detail_gate"], gate)
        feature = blend(feature, aligned["feature"], birth)
        center = blend(center, aligned["center"], birth)
        covariance = blend(covariance, aligned["covariance"], birth)
        owner = blend(owner, aligned["owner"], birth)
        relative_center = blend(
            relative_center, aligned["relative_center"], birth
        )
        detail_latent = blend(
            detail_latent, aligned["detail_latent"], birth
        )
        detail_gate = blend(detail_gate, aligned["detail_gate"], birth)
        presence = torch.maximum(
            predicted.presence * (1.0 - gate) + aligned["presence"] * gate,
            birth,
        ).clamp(0.0, 1.0)
        presence = self._hard_presence(presence)
        normalized_owner = owner / owner.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        owner = self._enforce_scene_quota(normalized_owner, presence > 0.5)
        visibility = torch.maximum(aligned["visibility"] * gate, birth).clamp(
            0.0, 1.0
        ) * presence
        object_mass = owner[..., : self.config.object_slots].sum(dim=-1)
        identity_value = blend(predicted.identity_key, aligned["identity"], gate)
        identity_value = blend(identity_value, aligned["identity"], birth)
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
            detail_latent=detail_latent,
            detail_valid=torch.maximum(
                predicted.detail_valid * (1.0 - gate),
                aligned["detail_valid"] * torch.maximum(gate, birth),
            ),
            detail_gate=detail_gate * presence,
        )
