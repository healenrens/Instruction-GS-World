"""Learned soft association between persistent tracks and frame observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .v46_config import ObservationCompleteConfig


@dataclass
class SlotAssociation:
    transport: torch.Tensor
    normalized_transport: torch.Tensor
    match_probability: torch.Tensor
    unmatched_probability: torch.Tensor
    discovery_probability: torch.Tensor
    entropy: torch.Tensor
    appearance_similarity: torch.Tensor
    support_distance: torch.Tensor


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


def _augmented_transport(
    scores: torch.Tensor, dustbin: torch.Tensor, iterations: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, tracks, observations = scores.shape
    augmented = torch.cat(
        (
            torch.cat((scores, dustbin.expand(batch, tracks, 1)), dim=2),
            torch.cat(
                (
                    dustbin.expand(batch, 1, observations),
                    dustbin.expand(batch, 1, 1),
                ),
                dim=2,
            ),
        ),
        dim=1,
    )
    normalizer = -torch.log(scores.new_tensor(float(tracks + observations)))
    row_mass = torch.cat(
        (
            normalizer.expand(batch, tracks),
            (torch.log(scores.new_tensor(float(observations))) + normalizer)
            .expand(batch, 1),
        ),
        dim=1,
    )
    column_mass = torch.cat(
        (
            normalizer.expand(batch, observations),
            (torch.log(scores.new_tensor(float(tracks))) + normalizer)
            .expand(batch, 1),
        ),
        dim=1,
    )
    log_transport = _log_sinkhorn(
        augmented, row_mass, column_mass, iterations
    ) - normalizer
    return (
        log_transport[:, :tracks, :observations].exp(),
        log_transport[:, :tracks, observations].exp(),
        log_transport[:, tracks, :observations].exp(),
    )


class LearnedSlotAssociation(nn.Module):
    def __init__(self, config: ObservationCompleteConfig):
        super().__init__()
        self.temperature = config.association_temperature
        self.iterations = config.association_sinkhorn_iterations
        self.dustbin_logit = nn.Parameter(
            torch.tensor(config.association_dustbin_logit)
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, 64),
            nn.SiLU(),
            nn.Linear(64, 1, bias=False),
        )
        nn.init.zeros_(self.residual[-1].weight)

    def forward(
        self,
        track_feature: torch.Tensor,
        track_center: torch.Tensor,
        track_scale: torch.Tensor,
        track_presence: torch.Tensor,
        observed_feature: torch.Tensor,
        observed_center: torch.Tensor,
        observed_scale: torch.Tensor,
        observed_activity: torch.Tensor,
    ) -> SlotAssociation:
        track = F.normalize(track_feature.float(), dim=-1, eps=1e-6)
        observed = F.normalize(observed_feature.float(), dim=-1, eps=1e-6)
        appearance = torch.einsum("bkd,bjd->bkj", track, observed)
        support = torch.exp(
            0.5
            * (
                track_scale[:, :, None].float()
                + observed_scale[:, None].float()
            )
        ).mean(dim=-1).clamp_min(1e-3)
        displacement = (
            observed_center[:, None].float() - track_center[:, :, None].float()
        ) / support[..., None]
        displacement = displacement.clamp(-8.0, 8.0)
        distance = displacement.square().sum(dim=-1).add(1e-6).sqrt()
        scale_difference = (
            observed_scale[:, None].float() - track_scale[:, :, None].float()
        ).abs().mean(dim=-1).clamp_max(8.0)
        presence = track_presence[:, :, None].float().clamp(0.0, 1.0)
        activity = observed_activity[:, None].float().clamp(0.0, 1.0)
        pair = torch.stack(
            (
                appearance,
                displacement[..., 0],
                displacement[..., 1],
                distance,
                scale_difference,
                presence.expand_as(appearance),
                activity.expand_as(appearance),
                appearance * activity,
            ),
            dim=-1,
        )
        tracked = 2.0 * appearance - 0.5 * distance - 0.25 * scale_difference
        discovery = appearance - 0.25 * distance + activity
        score = presence * tracked + (1.0 - presence) * discovery
        score = score + 0.25 * activity.clamp_min(1e-4).log()
        score = score + torch.tanh(self.residual(pair).squeeze(-1).float())
        score = 20.0 * torch.tanh(score / self.temperature / 20.0)
        dustbin = 20.0 * torch.tanh(
            self.dustbin_logit.float() / self.temperature / 20.0
        )
        transport, unmatched, new_observation = _augmented_transport(
            score, dustbin.reshape(1, 1, 1), self.iterations
        )
        match = transport.sum(dim=-1).clamp(0.0, 1.0)
        normalized = transport / match[..., None].clamp_min(1e-6)
        entropy = -(normalized * normalized.clamp_min(1e-8).log()).sum(dim=-1)
        if transport.shape[-1] > 1:
            entropy = entropy / torch.log(
                transport.new_tensor(float(transport.shape[-1]))
            )
        return SlotAssociation(
            transport=transport,
            normalized_transport=normalized,
            match_probability=match,
            unmatched_probability=unmatched.clamp(0.0, 1.0),
            discovery_probability=new_observation.clamp(0.0, 1.0),
            entropy=entropy,
            appearance_similarity=(normalized * appearance).sum(dim=-1),
            support_distance=(normalized * distance).sum(dim=-1),
        )


def align_slots(weights: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bko,bod->bkd", weights.to(value.dtype), value)


def align_scalars(weights: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bko,bo->bk", weights.to(value.dtype), value)
