"""Persistent identity-key objectives for recurrent object memory."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def persistent_identity_loss(
    output: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    states = output["history_slot_states"]
    reference = output["predicted_future_slots"].sum() * 0.0
    if not states or not hasattr(states[0], "identity_key"):
        return reference, {
            "identity_visible_alignment": reference,
            "identity_temporal_drift": reference,
            "identity_occluded_drift": reference,
            "identity_association_entropy": reference,
        }
    visible_terms = []
    visible_weights = []
    temporal_terms = []
    temporal_weights = []
    occluded_terms = []
    occluded_weights = []
    separation_terms = []
    separation_weights = []
    association_terms = []
    association_weights = []
    for index, state in enumerate(states):
        visible_terms.append(
            1.0
            - F.cosine_similarity(
                state.identity_key.float(),
                state.tracking_slots.detach().float(),
                dim=-1,
            )
        )
        visible_weights.append(
            state.visibility.float() * state.association_match.float()
        )
        association_terms.append(state.association_entropy.float())
        association_weights.append(state.observation_confidence.float())
        normalized = F.normalize(state.identity_key.float(), dim=-1)
        similarity = torch.einsum("bkd,bjd->bkj", normalized, normalized)
        object_count = similarity.shape[-1]
        off_diagonal = 1.0 - torch.eye(
            object_count,
            device=similarity.device,
            dtype=similarity.dtype,
        )[None]
        separation_terms.append(F.relu(similarity - 0.5))
        separation_weights.append(
            state.existence.float().unsqueeze(-1)
            * state.existence.float().unsqueeze(-2)
            * off_diagonal
        )
        if index == 0:
            continue
        previous = states[index - 1]
        drift = 1.0 - F.cosine_similarity(
            state.identity_key.float(),
            previous.identity_key.detach().float(),
            dim=-1,
        )
        persistent = torch.minimum(
            state.existence.float(), previous.existence.float()
        )
        temporal_terms.append(drift)
        temporal_weights.append(persistent * state.association_match.float())
        occluded_terms.append(drift)
        occluded_weights.append(
            persistent * state.association_unmatched.float()
        )
    visible = _weighted_mean(
        torch.stack(visible_terms),
        torch.stack(visible_weights),
    )
    separation = _weighted_mean(
        torch.stack(separation_terms),
        torch.stack(separation_weights),
    )
    association = _weighted_mean(
        torch.stack(association_terms),
        torch.stack(association_weights),
    )
    if temporal_terms:
        temporal = _weighted_mean(
            torch.stack(temporal_terms),
            torch.stack(temporal_weights),
        )
        occluded = _weighted_mean(
            torch.stack(occluded_terms),
            torch.stack(occluded_weights),
        )
    else:
        temporal = reference
        occluded = reference
    total = (
        visible
        + 0.5 * temporal
        + occluded
        + 0.25 * separation
        + 0.1 * association
    )
    return total, {
        "identity_visible_alignment": visible,
        "identity_temporal_drift": temporal,
        "identity_occluded_drift": occluded,
        "identity_inter_object_separation": separation,
        "identity_association_entropy": association,
    }
