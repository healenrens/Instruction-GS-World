"""Continuous training-only object observations from frozen video teachers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .carrier_teacher_v61 import select_soft_object_components
from .fixed_teacher_projection_v61 import fixed_group_projection_v61
from .point_track_teacher import sample_patch_field


@dataclass(frozen=True)
class SiglipVideoFeaturesV62:
    patches: torch.Tensor
    valid: torch.Tensor
    grid_hw: tuple[int, int]


@dataclass(frozen=True)
class ContinuousObjectObservationV62:
    coordinates: torch.Tensor
    dino: torch.Tensor
    siglip: torch.Tensor
    support: torch.Tensor
    visibility: torch.Tensor
    lifecycle: torch.Tensor
    membership: torch.Tensor
    object_valid: torch.Tensor
    seed_track: torch.Tensor


class FrozenSiglipVideoRuntimeV62:
    """Expose frozen SigLIP patch fields instead of crop-level pseudo labels."""

    def __init__(self, checkpoint: str, device: torch.device, frame_batch: int):
        from transformers import SiglipVisionModel

        self.model = (
            SiglipVisionModel.from_pretrained(checkpoint, local_files_only=True)
            .to(device)
            .eval()
        )
        self.model.requires_grad_(False)
        self.device = device
        self.frame_batch = int(frame_batch)
        self.feature_dim = int(self.model.config.hidden_size)
        self.image_size = int(self.model.config.image_size)
        self.patch_size = int(self.model.config.patch_size)

    @torch.no_grad()
    def __call__(self, batch: dict[str, torch.Tensor]) -> SiglipVideoFeaturesV62:
        rgb = batch["video_rgb"]
        pixel_valid = batch["video_pixel_valid"]
        batch_size, frames = rgb.shape[:2]
        flat = rgb.flatten(0, 1)
        flat_valid = pixel_valid.flatten(0, 1)
        patches, masks = [], []
        for start in range(0, len(flat), self.frame_batch):
            stop = min(start + self.frame_batch, len(flat))
            images = F.interpolate(
                flat[start:stop].float() / 255.0,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            validity = F.interpolate(
                flat_valid[start:stop, None].float(),
                size=(self.image_size, self.image_size),
                mode="nearest",
            )
            output = self.model(pixel_values=(images - 0.5) / 0.5)
            patches.append(F.normalize(output.last_hidden_state.float(), dim=-1))
            masks.append(
                F.avg_pool2d(validity, self.patch_size, self.patch_size).flatten(1)
                >= 0.5
            )
        grid = self.image_size // self.patch_size
        return SiglipVideoFeaturesV62(
            patches=torch.cat(patches).reshape(batch_size, frames, grid * grid, -1),
            valid=torch.cat(masks).reshape(batch_size, frames, grid * grid),
            grid_hw=(grid, grid),
        )


def _select_query_object(batch, relation, maximum_components: int):
    components, valid = select_soft_object_components(relation, maximum_components)
    valid_count = valid.sum(dim=1).clamp_min(1)
    sequence = batch["sequence_index"].long()
    selection = sequence.remainder(valid_count)
    batch_index = torch.arange(len(sequence), device=sequence.device)
    membership = components[batch_index, selection]
    object_valid = valid[batch_index, selection]
    seed_track = membership.argmax(dim=-1)
    return membership, object_valid, seed_track


def _stabilized_coordinates(evidence):
    zero = torch.zeros_like(evidence.coordinates[:, :1])
    residual = torch.cat((zero, evidence.residual_flow.float()), dim=1)
    return evidence.coordinates[:, :1].float() + residual.cumsum(dim=1)


def _object_lifecycle(relation, membership):
    weight = membership[:, None] * relation.lifecycle_known.float()
    normalizer = weight.sum(dim=-1).clamp_min(1e-6)
    visible = (weight * relation.visibility.float()).sum(dim=-1) / normalizer
    present = (weight * relation.presence.float()).sum(dim=-1) / normalizer
    known = weight.sum(dim=-1) / membership.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    occluded = (present - visible).clamp(0.0, 1.0)
    unknown = (1.0 - known).clamp(0.0, 1.0)
    lifecycle = torch.stack((visible, occluded, unknown), dim=-1)
    lifecycle = lifecycle / lifecycle.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return lifecycle


@torch.no_grad()
def build_continuous_object_observation_v62(
    batch,
    evidence,
    relation,
    siglip_features,
    config,
) -> ContinuousObjectObservationV62:
    membership, object_valid, seed_track = _select_query_object(
        batch, relation, config.carrier_count
    )
    siglip_points = sample_patch_field(
        siglip_features.patches,
        evidence.coordinates,
        siglip_features.grid_hw,
    )
    dino = fixed_group_projection_v61(evidence.sampled_features, config.semantic_dim)
    siglip = fixed_group_projection_v61(siglip_points, config.semantic_dim)
    visible = evidence.visibility.float()
    support = membership[:, None] * visible
    lifecycle = _object_lifecycle(relation, membership)
    tensors = (dino, siglip, support, lifecycle)
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise RuntimeError("v62 continuous teacher produced non-finite observation")
    return ContinuousObjectObservationV62(
        coordinates=_stabilized_coordinates(evidence).detach(),
        dino=dino.detach(),
        siglip=siglip.detach(),
        support=support.detach(),
        visibility=visible.detach(),
        lifecycle=lifecycle.detach(),
        membership=membership.detach(),
        object_valid=object_valid.detach(),
        seed_track=seed_track.detach(),
    )
