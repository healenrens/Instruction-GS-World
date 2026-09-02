"""Direct-seed object components constrained by a shared transition model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ObjectBoundEvidenceV64:
    direct_affinity: torch.Tensor
    persistence: torch.Tensor
    activity: torch.Tensor
    scene_score: torch.Tensor


@dataclass(frozen=True)
class ObjectBoundMembershipV64:
    components: torch.Tensor
    valid: torch.Tensor
    seed_tracks: torch.Tensor
    effective_track_count: torch.Tensor
    prefix_transition_residual: torch.Tensor
    selected: torch.Tensor
    selected_valid: torch.Tensor
    selected_seed: torch.Tensor
    selected_component: torch.Tensor
    selected_effective_track_count: torch.Tensor
    selected_prefix_transition_residual: torch.Tensor
    scene_membership: torch.Tensor
    unknown_membership: torch.Tensor


def _pooled_tracks(features, visibility):
    weight = visibility.float()
    pooled = (features.float() * weight[..., None]).sum(dim=1)
    pooled = pooled / weight.sum(dim=1).clamp_min(1.0)[..., None]
    return F.normalize(pooled, dim=-1, eps=1e-6)


def _pairwise_geometry(coordinates, visibility, config):
    relative = coordinates[:, :, :, None] - coordinates[:, :, None]
    distance = relative.norm(dim=-1)
    joint = visibility[:, :, :, None] * visibility[:, :, None]
    count = joint.sum(dim=1)
    mean = (distance * joint).sum(dim=1) / count.clamp_min(1.0)
    variance = ((distance - mean[:, None]).square() * joint).sum(dim=1)
    variance = variance / count.clamp_min(1.0)
    rigidity = torch.exp(-variance.sqrt() / config.group_distance_sigma)
    locality = torch.exp(-mean.square() / (2.0 * config.group_locality_sigma**2))
    visible_count = visibility.sum(dim=1)
    union = visible_count[:, :, None] + visible_count[:, None] - count
    covisibility = count / union.clamp_min(1.0)
    return rigidity, locality, covisibility


def _pairwise_motion(evidence, start, stop, config):
    pair_visible = evidence.visibility[:, start + 1 : stop]
    pair_visible = pair_visible & evidence.visibility[:, start : stop - 1]
    pair_joint = pair_visible[:, :, :, None] & pair_visible[:, :, None]
    flow = evidence.residual_flow[:, start : stop - 1].float()
    difference = (flow[:, :, :, None] - flow[:, :, None]).norm(dim=-1)
    weight = pair_joint.float()
    mean = (difference * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
    coherence = torch.exp(-mean / config.relation_motion_sigma)
    activity = (
        evidence.motion_salience[:, start : stop - 1].float() * pair_visible.float()
    ).amax(dim=1)
    return coherence, activity


def direct_object_evidence_v64(observation, evidence, start, stop, config):
    if stop - start < 2:
        raise ValueError("v64 object evidence needs at least two frames")
    visibility = evidence.visibility[:, start:stop].float()
    persistence = visibility.mean(dim=1)
    dino = _pooled_tracks(observation.dino[:, start:stop], visibility)
    siglip = _pooled_tracks(observation.siglip[:, start:stop], visibility)
    dino_affinity = ((torch.einsum("bpd,bqd->bpq", dino, dino) + 1.0) * 0.5).clamp(
        0.0, 1.0
    )
    siglip_affinity = (
        (torch.einsum("bpd,bqd->bpq", siglip, siglip) + 1.0) * 0.5
    ).clamp(0.0, 1.0)
    semantic = (dino_affinity * siglip_affinity).sqrt()
    rigidity, locality, covisibility = _pairwise_geometry(
        evidence.coordinates[:, start:stop].float(), visibility, config
    )
    motion, activity = _pairwise_motion(evidence, start, stop, config)
    kinematic = torch.minimum(rigidity, motion)
    persistent_pair = (persistence[:, :, None] * persistence[:, None]).sqrt()
    direct = semantic * locality.sqrt() * kinematic
    direct = direct * covisibility.sqrt() * persistent_pair
    diagonal = torch.eye(direct.shape[-1], device=direct.device, dtype=torch.bool)[None]
    direct = direct.masked_fill(diagonal, 0.0)
    scene_score = (1.0 - activity) * persistence
    return ObjectBoundEvidenceV64(
        direct_affinity=direct,
        persistence=persistence,
        activity=activity,
        scene_score=scene_score,
    )


def transition_horizon_tensors_v64(evidence, start, stop, horizon):
    source = evidence.coordinates[:, start : stop - horizon].float()
    target = evidence.coordinates[:, start + horizon : stop].float()
    visible = evidence.visibility[:, start + horizon : stop]
    visible = visible & evidence.visibility[:, start : stop - horizon]
    flow = target - source
    weight = visible.float()
    global_flow = (flow * weight[..., None]).sum(dim=2, keepdim=True)
    global_flow = (
        global_flow / weight.sum(dim=2, keepdim=True).clamp_min(1.0)[..., None]
    )
    flow = flow - global_flow
    ones = torch.ones_like(source[..., :1])
    design = torch.cat((source, ones), dim=-1)
    return design, flow, visible.float()


def _fit_transition_horizon(evidence, membership, start, stop, horizon, config):
    design, flow, visible = transition_horizon_tensors_v64(
        evidence, start, stop, horizon
    )
    weight = visible * membership[:, None].float()
    gram = torch.einsum("btpi,btp,btpj->bij", design, weight, design)
    rhs = torch.einsum("btpi,btp,btpd->bid", design, weight, flow)
    identity = torch.eye(3, device=gram.device, dtype=gram.dtype)[None]
    coefficients = torch.linalg.solve(gram + config.transition_ridge * identity, rhs)
    prediction = torch.einsum("btpi,bid->btpd", design, coefficients)
    residual = (prediction - flow).norm(dim=-1)
    track_count = visible.sum(dim=1)
    track_residual = (residual * visible).sum(dim=1)
    track_residual = track_residual / track_count.clamp_min(1.0)
    flow_scale = flow.norm(dim=-1).reshape(len(flow), -1)
    flow_scale = torch.quantile(flow_scale, 0.75, dim=1).clamp_min(0.005)
    inlier = torch.exp(-track_residual / flow_scale[:, None])
    inlier = inlier * (track_count > 0).float()
    total_weight = weight.sum(dim=(1, 2))
    rms = (residual.square() * weight).sum(dim=(1, 2))
    rms = (rms / total_weight.clamp_min(1e-6)).sqrt()
    visible_tracks = (
        ((visible > 0.0) & (membership[:, None] > 0.0)).any(dim=1).sum(dim=1)
    )
    valid = visible_tracks >= config.minimum_component_tracks
    return coefficients, inlier, rms / flow_scale, valid


def fit_shared_transition_v64(evidence, membership, start, stop, config):
    fits = [
        _fit_transition_horizon(evidence, membership, start, stop, horizon, config)
        for horizon in config.transition_horizons
        if stop - start > horizon
    ]
    if not fits:
        raise ValueError("v64 clip cannot support any transition horizon")
    coefficients = torch.stack([fit[0] for fit in fits], dim=1)
    inlier = torch.stack([fit[1] for fit in fits], dim=1).amin(dim=1)
    residual = torch.stack([fit[2] for fit in fits], dim=1).mean(dim=1)
    valid = torch.stack([fit[3] for fit in fits], dim=1).all(dim=1)
    return coefficients, inlier, residual, valid


def _effective_tracks(membership):
    return membership.sum(dim=-1).square() / membership.square().sum(dim=-1).clamp_min(
        1e-6
    )


def _refine_component(evidence, seed_affinity, seed, start, stop, config):
    points = seed_affinity.shape[-1]
    seed_mask = F.one_hot(seed, points).float()
    membership = torch.maximum(seed_affinity, seed_mask)
    transition_residual = torch.zeros(
        len(membership), device=membership.device, dtype=torch.float32
    )
    for _ in range(config.transition_refinement_steps):
        _, inlier, _, _ = fit_shared_transition_v64(
            evidence, membership, start, stop, config
        )
        membership = seed_affinity * inlier
        membership = torch.maximum(membership, seed_mask)
    _, _, transition_residual, _ = fit_shared_transition_v64(
        evidence, membership, start, stop, config
    )
    return membership, transition_residual


def build_seed_membership_v64(
    observation,
    evidence,
    seeds,
    config,
    start,
    stop,
):
    graph = direct_object_evidence_v64(observation, evidence, start, stop, config)
    batch = torch.arange(len(seeds), device=seeds.device)
    direct = graph.direct_affinity[batch, seeds] * graph.persistence
    membership, residual = _refine_component(
        evidence, direct, seeds, start, stop, config
    )
    claim = membership >= (
        config.component_claim_ratio
        * membership.amax(dim=-1, keepdim=True).clamp_min(1e-6)
    )
    membership = membership * claim.float()
    effective = _effective_tracks(membership)
    valid = claim.sum(dim=-1) >= config.minimum_component_tracks
    return (
        membership * valid[:, None].float(),
        valid,
        effective * valid.float(),
        residual * valid.float(),
    )


def _select_components(components, initial_valid, activity, scene_score, config):
    components = components * initial_valid[..., None].float()
    object_mass = components.amax(dim=1)
    object_wins = object_mass > scene_score
    components = components * object_wins[:, None].float()
    effective = _effective_tracks(components)
    track_count = (components > 0.0).sum(dim=-1)
    valid = initial_valid & (track_count >= config.minimum_component_tracks)
    components = components * valid[..., None].float()
    object_mass = components.amax(dim=1)
    scene = scene_score * (scene_score >= object_mass).float()
    explained = torch.maximum(object_mass, scene).clamp(0.0, 1.0)
    unknown = (1.0 - explained) * (0.5 + 0.5 * (1.0 - activity))
    return components, valid, effective, scene, unknown


def build_object_bound_membership_v64(
    observation,
    evidence,
    sequence_index,
    config,
    start,
    stop,
):
    graph = direct_object_evidence_v64(observation, evidence, start, stop, config)
    degree = graph.direct_affinity.sum(dim=-1)
    degree = degree / degree.amax(dim=-1, keepdim=True).clamp_min(1e-6)
    score = graph.activity * graph.persistence * (0.5 + 0.5 * degree)
    batch, points = score.shape
    batch_index = torch.arange(batch, device=score.device)
    components, valid, seeds, residuals = [], [], [], []
    remaining = score.clone()
    claimed = torch.zeros(batch, points, device=score.device, dtype=torch.bool)
    for _ in range(config.carrier_count):
        remaining = remaining.masked_fill(claimed, -1.0)
        seed = remaining.argmax(dim=-1)
        seed_score = remaining[batch_index, seed]
        direct = graph.direct_affinity[batch_index, seed]
        direct = direct * graph.persistence
        membership, transition_residual = _refine_component(
            evidence, direct, seed, start, stop, config
        )
        claim = membership >= (
            config.component_claim_ratio
            * membership.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        )
        claim = claim & (~claimed)
        membership = membership * claim.float()
        count = claim.sum(dim=-1)
        component_valid = (seed_score > 0.0) & (
            count >= config.minimum_component_tracks
        )
        membership = membership * component_valid[:, None].float()
        claimed = claimed | (claim & component_valid[:, None])
        components.append(membership)
        valid.append(component_valid)
        seeds.append(seed)
        residuals.append(transition_residual)
        remaining = remaining * (1.0 - membership)
        remaining[batch_index, seed] = -1.0
    components = torch.stack(components, dim=1)
    initial_valid = torch.stack(valid, dim=1)
    seeds = torch.stack(seeds, dim=1)
    residuals = torch.stack(residuals, dim=1)
    components, valid, effective, scene, unknown = _select_components(
        components, initial_valid, graph.activity, graph.scene_score, config
    )
    residuals = residuals * valid.float()
    valid_count = valid.sum(dim=1).clamp_min(1)
    selection = sequence_index.long().remainder(valid_count)
    valid_rank = valid.long().cumsum(dim=1) - 1
    selected_mask = valid & (valid_rank == selection[:, None])
    selected = (components * selected_mask[..., None].float()).sum(dim=1)
    selected_seed = torch.where(selected_mask, seeds, 0).sum(dim=1)
    selected_effective = torch.where(selected_mask, effective, 0.0).sum(dim=1)
    selected_residual = torch.where(selected_mask, residuals, 0.0).sum(dim=1)
    selected_component = torch.where(
        selected_mask,
        torch.arange(config.carrier_count, device=score.device)[None],
        0,
    ).sum(dim=1)
    tensors = (components, effective, residuals, selected, scene, unknown)
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise RuntimeError("v64 object-bound membership is non-finite")
    return ObjectBoundMembershipV64(
        components=components.detach(),
        valid=valid.detach(),
        seed_tracks=seeds.detach(),
        effective_track_count=effective.detach(),
        prefix_transition_residual=residuals.detach(),
        selected=selected.detach(),
        selected_valid=valid.any(dim=1).detach(),
        selected_seed=selected_seed.detach(),
        selected_component=selected_component.detach(),
        selected_effective_track_count=selected_effective.detach(),
        selected_prefix_transition_residual=selected_residual.detach(),
        scene_membership=scene.detach(),
        unknown_membership=unknown.detach(),
    )
