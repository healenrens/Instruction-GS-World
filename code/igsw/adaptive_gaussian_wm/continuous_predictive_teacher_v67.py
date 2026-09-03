"""Training-only continuous correspondence and semantic evidence for v67."""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F

from .continuous_field_sampling_v67 import (
    context_coordinate_mask_v67,
    evenly_spaced_query_indices_v67,
    stratified_query_coordinates_v67,
)
from .native_local_feature_field_v65 import (
    NativeTiledPerceptionRuntimeV65,
    pool_native_local_features_v65,
)
from .reliable_point_tracker_v65 import relay_track_reliability_v65


@dataclass(frozen=True)
class ContinuousQueryTrackEvidenceV67:
    anchor_coordinates: torch.Tensor
    coordinates: torch.Tensor
    visibility: torch.Tensor
    tracker_reliability: torch.Tensor
    appearance_reliability: torch.Tensor
    reliability: torch.Tensor
    relay_error: torch.Tensor


@dataclass(frozen=True)
class ContinuousPredictiveTeacherBatchV67:
    anchor_coordinates: torch.Tensor
    track_coordinates: torch.Tensor
    scales: torch.Tensor
    dino: torch.Tensor
    siglip: torch.Tensor
    visibility: torch.Tensor
    reliability: torch.Tensor
    query_indices: torch.Tensor
    context_mask: torch.Tensor


def _normalize_tracks_v67(tracks: torch.Tensor, native_hw: torch.Tensor) -> torch.Tensor:
    normalized = tracks.float().clone()
    height = native_hw[:, 0].float().clamp_min(2.0)[:, None, None]
    width = native_hw[:, 1].float().clamp_min(2.0)[:, None, None]
    normalized[..., 0] = normalized[..., 0] / (width - 1.0) * 2.0 - 1.0
    normalized[..., 1] = normalized[..., 1] / (height - 1.0) * 2.0 - 1.0
    return normalized


class ContinuousQueryTrackerRuntimeV67:
    """Measure noisy correspondences for continuous, jittered source queries."""

    def __init__(self, config, device: torch.device, checkpoint_path: str):
        if device.type != "cuda":
            raise ValueError("v67 point tracking requires CUDA")
        checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise ValueError(f"v67 tracker checkpoint is missing: {checkpoint_path}")
        from cotracker.predictor import CoTrackerPredictor

        self.config = config
        self.device = device
        self.model = CoTrackerPredictor(
            checkpoint=checkpoint_path,
            v2=False,
            offline=True,
        ).to(device).eval()
        self.model.requires_grad_(False)

    def _queries(self, batch, native_hw):
        coordinates = stratified_query_coordinates_v67(
            batch["sequence_index"],
            self.config.tracker_grid_side,
            self.config.coordinate_jitter_fraction,
        )
        height = native_hw[:, 0].float().clamp_min(2.0)[:, None]
        width = native_hw[:, 1].float().clamp_min(2.0)[:, None]
        pixel = coordinates.clone()
        pixel[..., 0] = (pixel[..., 0] + 1.0) * 0.5 * (width - 1.0)
        pixel[..., 1] = (pixel[..., 1] + 1.0) * 0.5 * (height - 1.0)
        time = torch.full(
            (*pixel.shape[:-1], 1),
            float(self.config.source_frame),
            device=pixel.device,
        )
        return coordinates, torch.cat((time, pixel), dim=-1)

    @torch.no_grad()
    def _predict(self, rgb, queries, native_hw):
        tracks, visibility = [], []
        for index in range(len(rgb)):
            height, width = native_hw[index].tolist()
            predicted, visible = self.model(
                rgb[index : index + 1, :, :, :height, :width].float(),
                queries=queries[index : index + 1],
                backward_tracking=True,
            )
            tracks.append(predicted.float())
            visibility.append(visible.bool())
        return torch.cat(tracks), torch.cat(visibility)

    @torch.no_grad()
    def __call__(self, batch) -> ContinuousQueryTrackEvidenceV67:
        rgb = batch["video_rgb"]
        native_hw = batch["native_image_hw"]
        anchor_coordinates, queries = self._queries(batch, native_hw)
        primary_px, primary_visible = self._predict(rgb, queries, native_hw)
        midpoint = self.config.midpoint_frame
        relay_xy = primary_px[:, midpoint]
        relay_time = torch.full(
            (*relay_xy.shape[:-1], 1), float(midpoint), device=rgb.device
        )
        relay_queries = torch.cat((relay_time, relay_xy), dim=-1)
        relay_px, relay_visible = self._predict(rgb, relay_queries, native_hw)
        primary = _normalize_tracks_v67(primary_px, native_hw)
        relay = _normalize_tracks_v67(relay_px, native_hw)
        primary_visible = primary_visible & (primary.abs().amax(dim=-1) <= 1.0)
        relay_visible = relay_visible & (relay.abs().amax(dim=-1) <= 1.0)
        relay_error, joint_fraction, tracker_reliability = relay_track_reliability_v65(
            primary,
            primary_visible,
            relay,
            relay_visible,
            self.config.tracker_relay_sigma,
        )
        tracker_reliability = tracker_reliability * (
            joint_fraction >= self.config.tracker_min_joint_fraction
        ).float()
        return ContinuousQueryTrackEvidenceV67(
            anchor_coordinates=anchor_coordinates.detach(),
            coordinates=primary.detach(),
            visibility=primary_visible.detach(),
            tracker_reliability=tracker_reliability.detach(),
            appearance_reliability=torch.ones_like(tracker_reliability),
            reliability=tracker_reliability.detach(),
            relay_error=relay_error.detach(),
        )


def _appearance_reliability_v67(evidence, dino, siglip, config):
    visible = evidence.visibility & dino.valid & siglip.valid
    weight = visible.float()

    def temporal_error(features):
        centroid = (features.float() * weight[..., None]).sum(dim=1)
        centroid = F.normalize(centroid, dim=-1, eps=1e-6)
        error = 1.0 - F.cosine_similarity(features.float(), centroid[:, None], dim=-1)
        return (error * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    appearance_error = 0.5 * (temporal_error(dino.features) + temporal_error(siglip.features))
    appearance_reliability = torch.exp(
        -appearance_error / config.appearance_reliability_sigma
    )
    reliability = evidence.tracker_reliability * appearance_reliability
    reliability = reliability * (reliability >= config.tracker_reliability_floor).float()
    return visible, appearance_reliability, reliability


def _query_scales_v67(config, sequence_index: torch.Tensor) -> torch.Tensor:
    point = torch.arange(config.candidate_count, device=sequence_index.device)[None]
    phase = (point + sequence_index.long()[:, None]).remainder(4).float()
    scale = config.base_query_scale * torch.exp2((phase - 1.5) / 2.0)
    return scale.clamp(config.minimum_scale, config.maximum_scale)


class ContinuousPredictiveTeacherRuntimeV67:
    """Bundle semantic observables and noisy correspondence measurements."""

    def __init__(
        self,
        config,
        device,
        amp,
        dino_checkpoint,
        siglip_checkpoint,
        tracker_checkpoint,
        dino_frame_batch,
        siglip_frame_batch,
    ):
        self.config = config
        self.perception = NativeTiledPerceptionRuntimeV65(
            config,
            device,
            amp,
            dino_checkpoint,
            siglip_checkpoint,
            dino_frame_batch,
            siglip_frame_batch,
        )
        self.tracker = ContinuousQueryTrackerRuntimeV67(
            config, device, tracker_checkpoint
        )

    @torch.no_grad()
    def __call__(self, batch) -> ContinuousPredictiveTeacherBatchV67:
        evidence = self.tracker(batch)
        field = self.perception(batch)
        dino = pool_native_local_features_v65(
            field.dino,
            evidence.coordinates,
            self.config.local_radii_pixels,
            self.config.local_tokens_per_scale,
        )
        siglip = pool_native_local_features_v65(
            field.siglip,
            evidence.coordinates,
            self.config.local_radii_pixels,
            self.config.local_tokens_per_scale,
        )
        visibility, _, reliability = _appearance_reliability_v67(
            evidence, dino, siglip, self.config
        )
        query_indices = evenly_spaced_query_indices_v67(
            self.config.candidate_count,
            self.config.query_count,
            batch["video_rgb"].device,
        )
        return ContinuousPredictiveTeacherBatchV67(
            anchor_coordinates=evidence.anchor_coordinates,
            track_coordinates=evidence.coordinates,
            scales=_query_scales_v67(self.config, batch["sequence_index"]),
            dino=dino.features.detach(),
            siglip=siglip.features.detach(),
            visibility=visibility.detach(),
            reliability=reliability.detach(),
            query_indices=query_indices,
            context_mask=context_coordinate_mask_v67(
                self.config.candidate_count,
                self.config.context_fraction,
                batch["sequence_index"],
            ),
        )


def teacher_relation_evidence_v67(target, frame: int, config):
    """Soft evidence only; it is not promoted to a hard object label."""
    query = target.query_indices
    dino = F.normalize(target.dino[:, frame].float(), dim=-1, eps=1e-6)
    siglip = F.normalize(target.siglip[:, frame].float(), dim=-1, eps=1e-6)
    dino_affinity = (torch.einsum("bqd,bpd->bqp", dino.index_select(1, query), dino) + 1.0) * 0.5
    siglip_affinity = (torch.einsum("bqd,bpd->bqp", siglip.index_select(1, query), siglip) + 1.0) * 0.5
    semantic = (dino_affinity.clamp(0.0, 1.0) * siglip_affinity.clamp(0.0, 1.0)).sqrt()
    if frame <= config.source_frame:
        # Source-time grouping may only use motion that was already observed.
        prefix = target.track_coordinates[:, : config.source_frame + 1]
        prefix_visible = target.visibility[:, : config.source_frame + 1]
        flow = prefix[:, 1:] - prefix[:, :-1]
        flow_valid = prefix_visible[:, 1:] & prefix_visible[:, :-1]
        flow_weight = flow_valid.float()
        displacement = (flow * flow_weight[..., None]).sum(dim=1)
        displacement = displacement / flow_weight.sum(dim=1).clamp_min(1.0)[..., None]
    else:
        # Future relation targets describe the transition that actually occurred.
        displacement = (
            target.track_coordinates[:, frame]
            - target.track_coordinates[:, config.source_frame]
        )
    query_displacement = displacement.index_select(1, query)
    motion_difference = (query_displacement[:, :, None] - displacement[:, None]).norm(dim=-1)
    motion = torch.exp(-motion_difference / config.relation_motion_sigma)
    relation = (semantic * motion).sqrt().clamp(0.0, 1.0)
    visible = target.visibility[:, frame]
    pair_visible = visible.index_select(1, query)[:, :, None] & visible[:, None]
    reliability = target.reliability
    pair_reliability = (
        reliability.index_select(1, query)[:, :, None] * reliability[:, None]
    ).sqrt()
    weight = pair_visible.float() * pair_reliability
    diagonal = query[None, :, None] == torch.arange(
        relation.shape[-1], device=relation.device
    )[None, None]
    relation = torch.where(diagonal, torch.ones_like(relation), relation)
    weight = torch.where(diagonal, pair_reliability, weight)
    return relation.detach(), weight.detach()
