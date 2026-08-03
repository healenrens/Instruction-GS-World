"""Causal soft data association between persistent tracks and observed slots."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AdaptiveGaussianWMConfig
from .object_slots import ObjectSlotState
from .relative_geometry import ObjectGeometryState


PAIR_FEATURE_DIM = 9


@dataclass
class ObjectCorrespondenceState:
    transport: torch.Tensor
    normalized_transport: torch.Tensor
    match_probability: torch.Tensor
    unmatched_probability: torch.Tensor
    discovery_probability: torch.Tensor
    entropy: torch.Tensor
    identity_similarity: torch.Tensor
    support_distance: torch.Tensor


def _pairwise_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = F.normalize(left.float(), dim=-1)
    right = F.normalize(right.float(), dim=-1)
    return torch.einsum("bkd,bjd->bkj", left, right)


def _soft_clip(value: torch.Tensor, limit: float) -> torch.Tensor:
    return limit * torch.tanh(value.float() / limit)


def _log_sinkhorn(
    scores: torch.Tensor,
    log_row_mass: torch.Tensor,
    log_column_mass: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    row_dual = torch.zeros_like(log_row_mass)
    column_dual = torch.zeros_like(log_column_mass)
    for _ in range(iterations):
        row_dual = log_row_mass - torch.logsumexp(
            scores + column_dual[:, None], dim=2
        )
        column_dual = log_column_mass - torch.logsumexp(
            scores + row_dual[:, :, None], dim=1
        )
    return scores + row_dual[:, :, None] + column_dual[:, None]


def _augmented_optimal_transport(
    scores: torch.Tensor,
    dustbin: torch.Tensor,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, tracks, observations = scores.shape
    if tracks <= 0 or observations <= 0:
        raise ValueError("correspondence requires non-empty track and observation sets")
    bins_track = dustbin.expand(batch, tracks, 1)
    bins_observation = dustbin.expand(batch, 1, observations)
    corner = dustbin.expand(batch, 1, 1)
    augmented = torch.cat(
        (
            torch.cat((scores, bins_track), dim=2),
            torch.cat((bins_observation, corner), dim=2),
        ),
        dim=1,
    )
    normalizer = -torch.log(
        scores.new_tensor(float(tracks + observations))
    )
    log_row_mass = torch.cat(
        (
            normalizer.expand(batch, tracks),
            (torch.log(scores.new_tensor(float(observations))) + normalizer)
            .expand(batch, 1),
        ),
        dim=1,
    )
    log_column_mass = torch.cat(
        (
            normalizer.expand(batch, observations),
            (torch.log(scores.new_tensor(float(tracks))) + normalizer)
            .expand(batch, 1),
        ),
        dim=1,
    )
    log_transport = _log_sinkhorn(
        augmented,
        log_row_mass,
        log_column_mass,
        iterations,
    ) - normalizer
    transport = log_transport[:, :tracks, :observations].exp()
    unmatched = log_transport[:, :tracks, observations].exp()
    discovery = log_transport[:, tracks, :observations].exp()
    return transport, unmatched, discovery


class CausalObjectCorrespondence(nn.Module):
    """Associate current-only slot observations with persistent track order."""

    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        hidden = config.memory_relation_dim
        self.temperature = config.correspondence_temperature
        self.iterations = config.correspondence_sinkhorn_iterations
        self.residual_scale = config.correspondence_residual_scale
        self.logit_clip = config.correspondence_logit_clip
        self.dustbin_logit = nn.Parameter(
            torch.tensor(float(config.correspondence_dustbin_logit))
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(PAIR_FEATURE_DIM),
            nn.Linear(PAIR_FEATURE_DIM, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1, bias=False),
        )
        nn.init.zeros_(self.residual[-1].weight)

    def forward(
        self,
        predicted,
        observation: ObjectSlotState,
        observed_geometry: ObjectGeometryState,
    ) -> ObjectCorrespondenceState:
        identity = _pairwise_cosine(
            predicted.identity_key,
            observation.tracking_slots,
        )
        appearance = _pairwise_cosine(
            predicted.decoded_feature,
            observation.decoded_feature,
        )
        support = torch.sqrt(
            predicted.relative_scale[:, :, None].float().clamp_min(1e-6)
            * observed_geometry.relative_scale[:, None, :].float().clamp_min(1e-6)
        )
        displacement = (
            observed_geometry.center[:, None].float()
            - predicted.center[:, :, None].float()
        ) / support[..., None]
        displacement = displacement.clamp(-8.0, 8.0)
        distance = displacement.square().sum(dim=-1).add(1e-6).sqrt().clamp_max(8.0)
        scale_difference = (
            observed_geometry.relative_scale[:, None, :].float().clamp_min(1e-6).log()
            - predicted.relative_scale[:, :, None].float().clamp_min(1e-6).log()
        ).abs().clamp_max(8.0)
        disparity_difference = (
            observed_geometry.relative_disparity[:, None, :].float()
            - predicted.relative_disparity[:, :, None].float()
        ).abs().clamp_max(8.0)
        presence = predicted.existence[:, :, None].float().clamp(0.0, 1.0)
        observed = observation.activity[:, None, :].float().clamp(0.0, 1.0)
        pair_features = torch.stack(
            (
                identity,
                appearance,
                displacement[..., 0],
                displacement[..., 1],
                distance,
                scale_difference,
                disparity_difference,
                presence.expand_as(identity),
                observed.expand_as(identity),
            ),
            dim=-1,
        )
        tracked_score = (
            2.0 * identity
            + appearance
            - 0.5 * distance
            - 0.25 * scale_difference
            - 0.1 * disparity_difference
        )
        discovery_score = (
            0.5 * appearance - 0.25 * distance + observed
        )
        score = presence * tracked_score + (1.0 - presence) * discovery_score
        score = score + 0.25 * observed.clamp_min(1e-4).log()
        learned_residual = torch.tanh(self.residual(pair_features).squeeze(-1).float())
        score = score + self.residual_scale * learned_residual
        score = _soft_clip(score / self.temperature, self.logit_clip)
        dustbin = _soft_clip(
            self.dustbin_logit.float() / self.temperature,
            self.logit_clip,
        ).reshape(1, 1, 1)
        transport, unmatched, discovery = _augmented_optimal_transport(
            score,
            dustbin,
            self.iterations,
        )
        match = transport.sum(dim=-1).clamp(0.0, 1.0)
        normalized = transport / match[..., None].clamp_min(1e-6)
        entropy = -(
            normalized * normalized.clamp_min(1e-8).log()
        ).sum(dim=-1)
        if transport.shape[-1] > 1:
            entropy = entropy / torch.log(
                transport.new_tensor(float(transport.shape[-1]))
            )
        matched_identity = (normalized * identity).sum(dim=-1)
        matched_distance = (normalized * distance).sum(dim=-1)
        return ObjectCorrespondenceState(
            transport=transport,
            normalized_transport=normalized,
            match_probability=match,
            unmatched_probability=unmatched.clamp(0.0, 1.0),
            discovery_probability=discovery.clamp(0.0, 1.0),
            entropy=entropy,
            identity_similarity=matched_identity,
            support_distance=matched_distance,
        )


def _align(weights: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bko,bod->bkd", weights.to(value.dtype), value)


def align_object_observation(
    observation: ObjectSlotState,
    correspondence: ObjectCorrespondenceState,
) -> ObjectSlotState:
    weights = correspondence.normalized_transport
    activity = correspondence.match_probability * torch.einsum(
        "bko,bo->bk",
        weights.to(observation.activity.dtype),
        observation.activity,
    )
    assignment = torch.einsum(
        "bmo,bko->bmk",
        observation.assignment,
        weights.to(observation.assignment.dtype),
    )
    return ObjectSlotState(
        slots=_align(weights, observation.slots),
        tracking_slots=_align(weights, observation.tracking_slots),
        assignment=assignment,
        background_assignment=observation.background_assignment,
        potential_change=observation.potential_change,
        potential_change_logits=observation.potential_change_logits,
        activity=activity,
        center=_align(weights, observation.center),
        feature=_align(weights, observation.feature),
        decoded_center=_align(weights, observation.decoded_center),
        decoded_feature=_align(weights, observation.decoded_feature),
        auxiliary_enabled=observation.auxiliary_enabled,
        center_auxiliary_enabled=observation.center_auxiliary_enabled,
    )
