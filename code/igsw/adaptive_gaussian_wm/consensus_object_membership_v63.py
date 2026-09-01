"""Cross-teacher graph consensus for training-only object membership."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ConsensusAffinityV63:
    affinity: torch.Tensor
    persistence: torch.Tensor
    activity: torch.Tensor


@dataclass(frozen=True)
class ConsensusObjectMembershipV63:
    components: torch.Tensor
    valid: torch.Tensor
    seed_tracks: torch.Tensor
    selected: torch.Tensor
    selected_valid: torch.Tensor
    selected_seed: torch.Tensor
    effective_track_count: torch.Tensor


def _pooled_tracks(features, visibility):
    weight = visibility.float()
    pooled = (features.float() * weight[..., None]).sum(dim=1)
    pooled = pooled / weight.sum(dim=1).clamp_min(1.0)[..., None]
    return F.normalize(pooled, dim=-1, eps=1e-6)


def _pairwise_geometry(coordinates, visibility, config):
    relative = coordinates[:, :, :, None] - coordinates[:, :, None]
    distance = relative.norm(dim=-1)
    jointly_visible = visibility[:, :, :, None] * visibility[:, :, None]
    joint_count = jointly_visible.sum(dim=1)
    mean_distance = (distance * jointly_visible).sum(dim=1)
    mean_distance = mean_distance / joint_count.clamp_min(1.0)
    variance = ((distance - mean_distance[:, None]).square() * jointly_visible).sum(
        dim=1
    )
    variance = variance / joint_count.clamp_min(1.0)
    rigidity = torch.exp(-variance.sqrt() / config.group_distance_sigma)
    locality = torch.exp(
        -mean_distance.square() / (2.0 * config.group_locality_sigma**2)
    )
    visible_count = visibility.sum(dim=1)
    union = (
        visible_count[:, :, None] + visible_count[:, None] - joint_count
    )
    covisibility = joint_count / union.clamp_min(1.0)
    return rigidity, locality, covisibility


def _pairwise_motion(evidence, visibility, start, stop, config):
    pair_visible = evidence.visibility[:, start + 1 : stop]
    pair_visible = pair_visible & evidence.visibility[:, start : stop - 1]
    pair_joint = pair_visible[:, :, :, None] & pair_visible[:, :, None]
    flow = evidence.residual_flow[:, start : stop - 1].float()
    flow_delta = (flow[:, :, :, None] - flow[:, :, None]).norm(dim=-1)
    mean_delta = (flow_delta * pair_joint.float()).sum(dim=1)
    mean_delta = mean_delta / pair_joint.float().sum(dim=1).clamp_min(1.0)
    coherence = torch.exp(-mean_delta / config.relation_motion_sigma)
    activity = (
        evidence.motion_salience[:, start : stop - 1].float()
        * pair_visible.float()
    ).amax(dim=1)
    return coherence, activity


def consensus_affinity_v63(observation, evidence, start, stop, config):
    if stop - start < 2:
        raise ValueError("consensus membership needs at least two frames")
    visibility = evidence.visibility[:, start:stop].float()
    persistence = visibility.mean(dim=1)
    dino = _pooled_tracks(observation.dino[:, start:stop], visibility)
    siglip = _pooled_tracks(observation.siglip[:, start:stop], visibility)
    dino_affinity = (
        (torch.einsum("bpd,bqd->bpq", dino, dino) + 1.0) * 0.5
    ).clamp(0.0, 1.0)
    siglip_affinity = (
        (torch.einsum("bpd,bqd->bpq", siglip, siglip) + 1.0) * 0.5
    ).clamp(0.0, 1.0)
    semantic = (dino_affinity * siglip_affinity).clamp_min(0.0).sqrt()
    rigidity, locality, covisibility = _pairwise_geometry(
        evidence.coordinates[:, start:stop].float(), visibility, config
    )
    motion, activity = _pairwise_motion(
        evidence, visibility, start, stop, config
    )
    structure = torch.maximum(rigidity, motion)
    persistence_pair = (
        persistence[:, :, None] * persistence[:, None]
    ).clamp_min(0.0).sqrt()
    positive = semantic * locality.sqrt() * (0.5 + 0.5 * structure)
    positive = positive * covisibility.sqrt() * persistence_pair
    kinematic_separation = (1.0 - rigidity) * (1.0 - motion)
    semantic_separation = (1.0 - semantic) * (1.0 - locality)
    separation = torch.maximum(kinematic_separation, semantic_separation)
    affinity = positive * (1.0 - separation).clamp(0.0, 1.0)
    diagonal = torch.eye(
        affinity.shape[-1], device=affinity.device, dtype=torch.bool
    )[None]
    affinity = affinity.masked_fill(diagonal, 0.0)
    return ConsensusAffinityV63(
        affinity=affinity,
        persistence=persistence,
        activity=activity,
    )


def diffuse_seed_membership_v63(affinity, persistence, seeds):
    points = affinity.shape[-1]
    seed = F.one_hot(seeds, points).float()
    transition = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    one_hop = torch.bmm(seed[:, None], transition)[:, 0]
    two_hop = torch.bmm(one_hop[:, None], transition)[:, 0]
    membership = seed + one_hop + two_hop
    membership = membership * persistence
    return membership / membership.amax(dim=-1, keepdim=True).clamp_min(1e-6)


def _effective_tracks(membership):
    return membership.sum(dim=-1).square() / membership.square().sum(
        dim=-1
    ).clamp_min(1e-6)


def build_consensus_object_membership_v63(
    observation,
    evidence,
    sequence_index,
    config,
    start,
    stop,
):
    graph = consensus_affinity_v63(observation, evidence, start, stop, config)
    degree = graph.affinity.sum(dim=-1)
    degree = degree / degree.amax(dim=-1, keepdim=True).clamp_min(1e-6)
    score = graph.activity * graph.persistence * (0.5 + 0.5 * degree)
    batch, points = score.shape
    batch_index = torch.arange(batch, device=score.device)
    components, valid, seeds, effective = [], [], [], []
    remaining = score.clone()
    for _ in range(config.carrier_count):
        seed = remaining.argmax(dim=-1)
        seed_score = remaining[batch_index, seed]
        membership = diffuse_seed_membership_v63(
            graph.affinity, graph.persistence, seed
        )
        count = _effective_tracks(membership)
        component_valid = (seed_score > 0.0) & (count >= 2.0)
        membership = membership * component_valid[:, None].float()
        components.append(membership)
        valid.append(component_valid)
        seeds.append(seed)
        effective.append(count * component_valid.float())
        remaining = remaining * (1.0 - membership)
        remaining[batch_index, seed] = -1.0
    components = torch.stack(components, dim=1)
    valid = torch.stack(valid, dim=1)
    seeds = torch.stack(seeds, dim=1)
    effective = torch.stack(effective, dim=1)
    valid_count = valid.sum(dim=1).clamp_min(1)
    selection = sequence_index.long().remainder(valid_count)
    valid_rank = valid.long().cumsum(dim=1) - 1
    selected_mask = valid & (valid_rank == selection[:, None])
    selected = (components * selected_mask[..., None].float()).sum(dim=1)
    selected_seed = torch.where(selected_mask, seeds, 0).sum(dim=1)
    selected_effective = torch.where(selected_mask, effective, 0.0).sum(dim=1)
    return ConsensusObjectMembershipV63(
        components=components.detach(),
        valid=valid.detach(),
        seed_tracks=seeds.detach(),
        selected=selected.detach(),
        selected_valid=valid.any(dim=1).detach(),
        selected_seed=selected_seed.detach(),
        effective_track_count=selected_effective.detach(),
    )
