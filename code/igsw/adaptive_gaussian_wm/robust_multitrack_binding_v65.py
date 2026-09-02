"""Reliable multi-track cores and robust prefix transition fits for v65."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MultiTrackEvidenceV65:
    direct_affinity: torch.Tensor
    persistence: torch.Tensor
    activity: torch.Tensor
    scene_score: torch.Tensor
    reliability: torch.Tensor


@dataclass(frozen=True)
class RobustTransitionFitV65:
    coefficients: torch.Tensor
    inlier: torch.Tensor
    residual: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class ReliableCoreBindingV65:
    components: torch.Tensor
    cores: torch.Tensor
    holdouts: torch.Tensor
    valid: torch.Tensor
    coefficients: torch.Tensor
    effective_track_count: torch.Tensor
    prefix_transition_residual: torch.Tensor
    selected: torch.Tensor
    selected_core: torch.Tensor
    selected_holdout: torch.Tensor
    selected_coefficients: torch.Tensor
    selected_valid: torch.Tensor
    selected_component: torch.Tensor
    selected_effective_track_count: torch.Tensor
    selected_prefix_transition_residual: torch.Tensor
    scene_membership: torch.Tensor
    unknown_membership: torch.Tensor


def _pooled_tracks(features, visibility, reliability):
    weight = visibility.float() * reliability[:, None]
    pooled = (features.float() * weight[..., None]).sum(dim=1)
    pooled = pooled / weight.sum(dim=1).clamp_min(1.0)[..., None]
    return F.normalize(pooled, dim=-1, eps=1e-6)


def _pairwise_geometry(coordinates, visibility, reliability, config):
    relative = coordinates[:, :, :, None] - coordinates[:, :, None]
    distance = relative.norm(dim=-1)
    joint = visibility[:, :, :, None] & visibility[:, :, None]
    pair_reliability = (
        reliability[:, :, None] * reliability[:, None]
    ).sqrt()
    weight = joint.float() * pair_reliability[:, None]
    count = weight.sum(dim=1)
    mean = (distance * weight).sum(dim=1) / count.clamp_min(1e-6)
    variance = ((distance - mean[:, None]).square() * weight).sum(dim=1)
    variance = variance / count.clamp_min(1e-6)
    rigidity = torch.exp(-variance.sqrt() / config.group_distance_sigma)
    locality = torch.exp(-mean.square() / (2.0 * config.group_locality_sigma**2))
    visible_count = (visibility.float() * reliability[:, None]).sum(dim=1)
    union = visible_count[:, :, None] + visible_count[:, None] - count
    covisibility = count / union.clamp_min(1e-6)
    return rigidity, locality, covisibility


def _pairwise_motion(evidence, start, stop, config):
    pair_visible = evidence.visibility[:, start + 1 : stop]
    pair_visible = pair_visible & evidence.visibility[:, start : stop - 1]
    joint = pair_visible[:, :, :, None] & pair_visible[:, :, None]
    pair_reliability = (
        evidence.reliability[:, :, None] * evidence.reliability[:, None]
    ).sqrt()
    flow = evidence.residual_flow[:, start : stop - 1].float()
    difference = (flow[:, :, :, None] - flow[:, :, None]).norm(dim=-1)
    weight = joint.float() * pair_reliability[:, None]
    mean = (difference * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-6)
    coherence = torch.exp(-mean / config.relation_motion_sigma)
    activity = (
        evidence.motion_salience[:, start : stop - 1].float()
        * pair_visible.float()
    ).amax(dim=1)
    return coherence, activity


def direct_multitrack_evidence_v65(observation, evidence, start, stop, config):
    visibility = evidence.visibility[:, start:stop]
    persistence = visibility.float().mean(dim=1)
    dino = _pooled_tracks(
        observation.dino[:, start:stop], visibility, evidence.reliability
    )
    siglip = _pooled_tracks(
        observation.siglip[:, start:stop], visibility, evidence.reliability
    )
    dino_affinity = ((torch.einsum("bpd,bqd->bpq", dino, dino) + 1.0) * 0.5)
    siglip_affinity = (
        (torch.einsum("bpd,bqd->bpq", siglip, siglip) + 1.0) * 0.5
    )
    semantic = (dino_affinity.clamp(0.0, 1.0) * siglip_affinity.clamp(0.0, 1.0)).sqrt()
    rigidity, locality, covisibility = _pairwise_geometry(
        evidence.coordinates[:, start:stop].float(),
        visibility,
        evidence.reliability,
        config,
    )
    motion, activity = _pairwise_motion(evidence, start, stop, config)
    kinematic = torch.minimum(rigidity, motion)
    persistent_pair = (persistence[:, :, None] * persistence[:, None]).sqrt()
    reliable_pair = (
        evidence.reliability[:, :, None] * evidence.reliability[:, None]
    ).sqrt()
    direct = semantic * locality.sqrt() * kinematic
    direct = direct * covisibility.sqrt() * persistent_pair * reliable_pair
    diagonal = torch.eye(direct.shape[-1], device=direct.device, dtype=torch.bool)[None]
    direct = direct.masked_fill(diagonal, 0.0)
    scene_score = (1.0 - activity) * persistence * evidence.reliability
    return MultiTrackEvidenceV65(
        direct_affinity=direct,
        persistence=persistence,
        activity=activity,
        scene_score=scene_score,
        reliability=evidence.reliability,
    )


def _transition_pairs(evidence, start, stop, horizon):
    source = evidence.coordinates[:, start : stop - horizon].float()
    target = evidence.coordinates[:, start + horizon : stop].float()
    visible = evidence.visibility[:, start : stop - horizon]
    visible = visible & evidence.visibility[:, start + horizon : stop]
    design = torch.cat((source, torch.ones_like(source[..., :1])), dim=-1)
    return design, target, visible.float()


def _fit_horizon(evidence, membership, start, stop, horizon, config):
    design, target, visible = _transition_pairs(evidence, start, stop, horizon)
    base_weight = visible * membership[:, None].float()
    base_weight = base_weight * evidence.reliability[:, None]
    weight = base_weight
    identity = torch.eye(3, device=design.device, dtype=torch.float32)[None]
    coefficients = torch.zeros(
        len(design), 3, 2, device=design.device, dtype=torch.float32
    )
    for _ in range(config.transition_irls_steps):
        gram = torch.einsum("btpi,btp,btpj->bij", design, weight, design)
        rhs = torch.einsum("btpi,btp,btpd->bid", design, weight, target)
        coefficients = torch.linalg.solve(
            gram + config.transition_ridge * identity, rhs
        )
        prediction = torch.einsum("btpi,bid->btpd", design, coefficients)
        residual = (prediction - target).norm(dim=-1).clamp_min(1e-6)
        robust = (config.transition_huber_delta / residual).clamp(max=1.0)
        weight = base_weight * robust
    prediction = torch.einsum("btpi,bid->btpd", design, coefficients)
    residual = (prediction - target).norm(dim=-1)
    track_weight = visible * evidence.reliability[:, None]
    track_residual = (residual * track_weight).sum(dim=1)
    track_residual = track_residual / track_weight.sum(dim=1).clamp_min(1e-6)
    inlier = torch.exp(-track_residual / config.transition_huber_delta)
    inlier = inlier * (track_weight.sum(dim=1) > 0.0).float()
    total_weight = base_weight.sum(dim=(1, 2))
    rms = (residual.square() * base_weight).sum(dim=(1, 2))
    rms = (rms / total_weight.clamp_min(1e-6)).sqrt()
    visible_tracks = (
        ((visible > 0.0) & (membership[:, None] > 0.0)).any(dim=1).sum(dim=1)
    )
    valid = visible_tracks >= config.minimum_component_tracks
    return coefficients, inlier, rms, valid


def fit_robust_transitions_v65(evidence, membership, start, stop, config):
    fits = [
        _fit_horizon(evidence, membership, start, stop, horizon, config)
        for horizon in config.transition_horizons
        if stop - start > horizon
    ]
    coefficients = torch.stack([fit[0] for fit in fits], dim=1)
    inlier = torch.stack([fit[1] for fit in fits], dim=1).amin(dim=1)
    residual = torch.stack([fit[2] for fit in fits], dim=1).mean(dim=1)
    valid = torch.stack([fit[3] for fit in fits], dim=1).all(dim=1)
    return RobustTransitionFitV65(coefficients, inlier, residual, valid)


def _effective_tracks(membership):
    return membership.sum(dim=-1).square() / membership.square().sum(dim=-1).clamp_min(
        1e-6
    )


def _mutual_core(direct, score, proposal, available, config):
    batch, points = score.shape
    batch_index = torch.arange(batch, device=score.device)
    neighborhood_score = direct[batch_index, proposal] * (0.5 + 0.5 * score)
    neighborhood_score = neighborhood_score.masked_fill(~available, -1.0)
    candidate_count = min(config.core_candidate_tracks, points)
    candidate_index = neighborhood_score.topk(candidate_count, dim=-1).indices
    candidate_mask = F.one_hot(candidate_index, points).amax(dim=1).bool()
    candidate_pair = direct * candidate_mask[:, :, None] * candidate_mask[:, None]
    mutual_score = candidate_pair.sum(dim=-1)
    mutual_score = mutual_score / candidate_mask.sum(dim=-1, keepdim=True).clamp_min(2)
    mutual_score = mutual_score * (0.5 + 0.5 * score)
    mutual_score = mutual_score.masked_fill(~candidate_mask, -1.0)
    core_index = mutual_score.topk(config.core_tracks, dim=-1).indices
    return F.one_hot(core_index, points).amax(dim=1).float()


def _core_consensus_membership(direct, core):
    points = direct.shape[-1]
    diagonal = torch.eye(points, device=direct.device, dtype=torch.bool)[None]
    edge_mask = core[:, :, None] * (~diagonal).float()
    log_affinity = direct.clamp_min(1e-6).log() * edge_mask
    denominator = edge_mask.sum(dim=1).clamp_min(1.0)
    membership = torch.exp(log_affinity.sum(dim=1) / denominator)
    return membership * (denominator > 1.0).float()


def build_reliable_core_binding_v65(
    observation,
    evidence,
    sequence_index,
    config,
    start,
    stop,
):
    graph = direct_multitrack_evidence_v65(
        observation, evidence, start, stop, config
    )
    degree = graph.direct_affinity.sum(dim=-1)
    degree = degree / degree.amax(dim=-1, keepdim=True).clamp_min(1e-6)
    score = graph.activity * graph.persistence * graph.reliability
    score = score * (0.5 + 0.5 * degree)
    batch, points = score.shape
    batch_index = torch.arange(batch, device=score.device)
    claimed = torch.zeros(batch, points, device=score.device, dtype=torch.bool)
    components, cores, holdouts = [], [], []
    valid, coefficients, residuals = [], [], []
    for _ in range(config.carrier_count):
        available = ~claimed
        proposal_score = score.masked_fill(~available, -1.0)
        proposal = proposal_score.argmax(dim=-1)
        core = _mutual_core(
            graph.direct_affinity, score, proposal, available, config
        )
        core = core * available.float() * (graph.reliability > 0.0).float()
        membership = _core_consensus_membership(graph.direct_affinity, core)
        membership = membership * graph.persistence * graph.reliability
        fit = fit_robust_transitions_v65(
            evidence, core, start, stop, config
        )
        membership = membership * fit.inlier
        claim = membership >= (
            config.component_claim_ratio
            * membership.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        )
        claim = claim & (membership >= config.relation_confidence_floor)
        claim = claim & available
        component = membership * claim.float()
        core = core * claim.float()
        holdout = component * (1.0 - core)
        core_count = (core > 0.0).sum(dim=-1)
        holdout_count = (holdout > 0.0).sum(dim=-1)
        component_valid = fit.valid & (core_count >= config.core_tracks)
        component_valid = component_valid & (
            holdout_count >= config.minimum_holdout_tracks
        )
        component = component * component_valid[:, None].float()
        core = core * component_valid[:, None].float()
        holdout = holdout * component_valid[:, None].float()
        claimed = claimed | (claim & component_valid[:, None])
        components.append(component)
        cores.append(core)
        holdouts.append(holdout)
        valid.append(component_valid)
        coefficients.append(fit.coefficients)
        residuals.append(fit.residual)
        score[batch_index, proposal] = -1.0
    components = torch.stack(components, dim=1)
    cores = torch.stack(cores, dim=1)
    holdouts = torch.stack(holdouts, dim=1)
    valid = torch.stack(valid, dim=1)
    coefficients = torch.stack(coefficients, dim=1)
    residuals = torch.stack(residuals, dim=1) * valid.float()
    effective = _effective_tracks(components) * valid.float()
    object_mass = components.amax(dim=1)
    object_wins = object_mass > graph.scene_score
    components = components * object_wins[:, None].float()
    cores = cores * object_wins[:, None].float()
    holdouts = holdouts * object_wins[:, None].float()
    valid = valid & ((cores > 0.0).sum(dim=-1) >= config.core_tracks)
    valid = valid & (
        (holdouts > 0.0).sum(dim=-1) >= config.minimum_holdout_tracks
    )
    components = components * valid[..., None].float()
    cores = cores * valid[..., None].float()
    holdouts = holdouts * valid[..., None].float()
    coefficients = coefficients * valid[..., None, None, None].float()
    residuals = residuals * valid.float()
    effective = _effective_tracks(components) * valid.float()
    object_mass = components.amax(dim=1)
    scene = graph.scene_score * (graph.scene_score >= object_mass).float()
    explained = torch.maximum(object_mass, scene).clamp(0.0, 1.0)
    unknown = (1.0 - explained) * (0.5 + 0.5 * (1.0 - graph.activity))
    valid_count = valid.sum(dim=1).clamp_min(1)
    selection = sequence_index.long().remainder(valid_count)
    valid_rank = valid.long().cumsum(dim=1) - 1
    selected_mask = valid & (valid_rank == selection[:, None])
    selected = (components * selected_mask[..., None].float()).sum(dim=1)
    selected_core = (cores * selected_mask[..., None].float()).sum(dim=1)
    selected_holdout = (holdouts * selected_mask[..., None].float()).sum(dim=1)
    selected_coefficients = (
        coefficients * selected_mask[..., None, None, None].float()
    ).sum(dim=1)
    selected_component = torch.where(
        selected_mask,
        torch.arange(config.carrier_count, device=score.device)[None],
        0,
    ).sum(dim=1)
    selected_effective = torch.where(selected_mask, effective, 0.0).sum(dim=1)
    selected_residual = torch.where(selected_mask, residuals, 0.0).sum(dim=1)
    tensors = (
        components,
        cores,
        holdouts,
        coefficients,
        selected,
        scene,
        unknown,
    )
    if not all(bool(value.isfinite().all()) for value in tensors):
        raise RuntimeError("v65 reliable core binding is non-finite")
    return ReliableCoreBindingV65(
        components=components.detach(),
        cores=cores.detach(),
        holdouts=holdouts.detach(),
        valid=valid.detach(),
        coefficients=coefficients.detach(),
        effective_track_count=effective.detach(),
        prefix_transition_residual=residuals.detach(),
        selected=selected.detach(),
        selected_core=selected_core.detach(),
        selected_holdout=selected_holdout.detach(),
        selected_coefficients=selected_coefficients.detach(),
        selected_valid=valid.any(dim=1).detach(),
        selected_component=selected_component.detach(),
        selected_effective_track_count=selected_effective.detach(),
        selected_prefix_transition_residual=selected_residual.detach(),
        scene_membership=scene.detach(),
        unknown_membership=unknown.detach(),
    )
