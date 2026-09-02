"""Native-resolution relay-consistent point-track measurements for v65."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ReliablePointTrackEvidenceV65:
    coordinates: torch.Tensor
    visibility: torch.Tensor
    residual_flow: torch.Tensor
    motion_salience: torch.Tensor
    query_times: torch.Tensor
    relay_coordinates: torch.Tensor
    relay_visibility: torch.Tensor
    relay_error: torch.Tensor
    joint_visibility_fraction: torch.Tensor
    tracker_reliability: torch.Tensor
    appearance_reliability: torch.Tensor
    reliability: torch.Tensor
    sampled_features: torch.Tensor | None = None


def _normalize_tracks(tracks, native_hw):
    normalized = tracks.float().clone()
    height = native_hw[:, 0].float().clamp_min(2.0)[:, None, None]
    width = native_hw[:, 1].float().clamp_min(2.0)[:, None, None]
    normalized[..., 0] = normalized[..., 0] / (width - 1.0) * 2.0 - 1.0
    normalized[..., 1] = normalized[..., 1] / (height - 1.0) * 2.0 - 1.0
    return normalized


def relay_track_reliability_v65(
    primary,
    primary_visible,
    relay,
    relay_visible,
    sigma: float,
):
    joint = primary_visible & relay_visible
    error = (primary.float() - relay.float()).norm(dim=-1)
    joint_fraction = joint.float().mean(dim=1)
    mean_error = (error * joint.float()).sum(dim=1)
    mean_error = mean_error / joint.float().sum(dim=1).clamp_min(1.0)
    reliability = torch.exp(-mean_error / sigma) * joint_fraction
    return mean_error, joint_fraction, reliability


def _robust_global_flow(flow, visible, reliability, iterations: int = 3):
    weight = visible.float() * reliability[:, None]
    center = (flow * weight[..., None]).sum(dim=2, keepdim=True)
    center = center / weight.sum(dim=2, keepdim=True).clamp_min(1.0)[..., None]
    for _ in range(iterations):
        residual = (flow - center).norm(dim=-1).clamp_min(1e-6)
        robust = (0.03 / residual).clamp(max=1.0)
        combined = weight * robust
        center = (flow * combined[..., None]).sum(dim=2, keepdim=True)
        center = center / combined.sum(dim=2, keepdim=True).clamp_min(1.0)[..., None]
    return center


class NativeRelayPointTrackerRuntimeV65:
    """Run CoTracker on the native tensor and re-query tracks at a relay frame."""

    def __init__(self, config, device, checkpoint_path: str):
        if device.type != "cuda":
            raise ValueError("v65 native point tracking requires CUDA")
        checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise ValueError(f"local CoTracker checkpoint is missing: {checkpoint_path}")
        from cotracker.predictor import CoTrackerPredictor

        self.config = config
        self.device = device
        self.model = CoTrackerPredictor(
            checkpoint=checkpoint_path,
            v2=False,
            offline=True,
        ).to(device).eval()
        self.model.requires_grad_(False)

    def _primary_queries(self, native_hw, frames: int):
        side = self.config.tracker_grid_side
        anchor_frames = torch.tensor(
            sorted(
                {
                    round(fraction * (frames - 1))
                    for fraction in self.config.tracker_anchor_fractions
                }
            ),
            device=self.device,
            dtype=torch.long,
        )
        queries = []
        for height, width in native_hw.tolist():
            x = (torch.arange(side, device=self.device).float() + 0.5) * (
                width / side
            )
            y = (torch.arange(side, device=self.device).float() + 0.5) * (
                height / side
            )
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            xy = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
            query_xy = xy.repeat(len(anchor_frames), 1)
            query_time = anchor_frames[:, None].expand(-1, len(xy)).reshape(-1)
            queries.append(
                torch.cat((query_time[:, None].float(), query_xy), dim=-1)
            )
        return torch.stack(queries), query_time

    @torch.no_grad()
    def _predict(self, video, queries, native_hw):
        tracks, visibility = [], []
        for index in range(len(video)):
            height, width = native_hw[index].tolist()
            predicted, visible = self.model(
                video[index : index + 1, :, :, :height, :width],
                queries=queries[index : index + 1],
                backward_tracking=True,
            )
            tracks.append(predicted.float())
            visibility.append(visible.bool())
        return torch.cat(tracks), torch.cat(visibility)

    @torch.no_grad()
    def __call__(self, batch) -> ReliablePointTrackEvidenceV65:
        rgb = batch["video_rgb"]
        if rgb.ndim != 5 or rgb.shape[2] != 3 or rgb.dtype != torch.uint8:
            raise ValueError("v65 tracker expects [B,T,3,H,W] uint8 native RGB")
        batch_size, frames, _, height, width = rgb.shape
        native_hw = batch.get("native_image_hw")
        if native_hw is None:
            native_hw = torch.tensor(
                (height, width), device=rgb.device, dtype=torch.long
            )[None].expand(batch_size, -1)
        queries, query_times = self._primary_queries(native_hw, frames)
        primary_px, primary_visible = self._predict(
            rgb.float(), queries, native_hw
        )
        relay_times = (query_times + frames - 1).div(2, rounding_mode="floor")
        batch_index = torch.arange(batch_size, device=rgb.device)[:, None]
        relay_xy = primary_px[batch_index, relay_times[None], torch.arange(len(query_times), device=rgb.device)[None]]
        relay_queries = torch.cat(
            (
                relay_times[None, :, None].expand(batch_size, -1, -1).float(),
                relay_xy,
            ),
            dim=-1,
        )
        relay_px, relay_visible = self._predict(
            rgb.float(), relay_queries, native_hw
        )
        primary = _normalize_tracks(primary_px, native_hw)
        relay = _normalize_tracks(relay_px, native_hw)
        primary_in_frame = primary.abs().amax(dim=-1) <= 1.0
        relay_in_frame = relay.abs().amax(dim=-1) <= 1.0
        primary_visible = primary_visible & primary_in_frame
        relay_visible = relay_visible & relay_in_frame
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
        flow = primary[:, 1:] - primary[:, :-1]
        pair_visible = primary_visible[:, 1:] & primary_visible[:, :-1]
        global_flow = _robust_global_flow(
            flow, pair_visible, tracker_reliability
        )
        residual = flow - global_flow
        speed = residual.norm(dim=-1) * pair_visible.float()
        scale = torch.quantile(speed, 0.75, dim=2, keepdim=True).clamp_min(0.01)
        salience = (speed / scale).clamp(0.0, 1.0) * pair_visible.float()
        reliability = tracker_reliability * (
            tracker_reliability >= self.config.tracker_reliability_floor
        ).float()
        tensors = (primary, relay, residual, salience, reliability)
        if not all(bool(value.isfinite().all()) for value in tensors):
            raise RuntimeError("v65 native relay tracker produced non-finite evidence")
        return ReliablePointTrackEvidenceV65(
            coordinates=primary.detach(),
            visibility=primary_visible.detach(),
            residual_flow=residual.detach(),
            motion_salience=salience.detach(),
            query_times=query_times.detach(),
            relay_coordinates=relay.detach(),
            relay_visibility=relay_visible.detach(),
            relay_error=relay_error.detach(),
            joint_visibility_fraction=joint_fraction.detach(),
            tracker_reliability=tracker_reliability.detach(),
            appearance_reliability=torch.ones_like(tracker_reliability),
            reliability=reliability.detach(),
        )


def add_appearance_reliability_v65(
    evidence: ReliablePointTrackEvidenceV65,
    dino,
    siglip,
    sigma: float,
    floor: float,
):
    visibility = evidence.visibility & dino.valid & siglip.valid
    weight = visibility.float()

    def consistency(features):
        centroid = (features.float() * weight[..., None]).sum(dim=1)
        centroid = F.normalize(centroid, dim=-1, eps=1e-6)
        error = 1.0 - F.cosine_similarity(
            features.float(), centroid[:, None], dim=-1
        )
        return (error * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    appearance_error = 0.5 * (
        consistency(dino.features) + consistency(siglip.features)
    )
    appearance_reliability = torch.exp(-appearance_error / sigma)
    reliability = evidence.tracker_reliability * appearance_reliability
    reliability = reliability * (reliability >= floor).float()
    return replace(
        evidence,
        visibility=visibility.detach(),
        appearance_reliability=appearance_reliability.detach(),
        reliability=reliability.detach(),
        sampled_features=dino.features.detach(),
    )
