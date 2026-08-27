"""Training-only continuous-track and object-crop teachers for v61."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TeacherObjectComponentsV61:
    membership: torch.Tensor
    valid: torch.Tensor
    semantic: torch.Tensor | None
    semantic_valid: torch.Tensor | None
    frame_indices: torch.Tensor


def select_soft_object_components(relation, object_roots: int):
    same = relation.same_confidence.float()
    score = relation.object_confidence.float().clone()
    batch, points = score.shape
    indices, valid = [], []
    batch_index = torch.arange(batch, device=score.device)
    for _ in range(object_roots):
        seed = score.argmax(dim=-1)
        seed_score = score[batch_index, seed]
        is_valid = seed_score > 0.0
        indices.append(seed)
        valid.append(is_valid)
        suppress = same[batch_index, seed]
        score = score * (1.0 - suppress)
        score[batch_index, seed] = -1.0
    indices = torch.stack(indices, dim=1)
    valid = torch.stack(valid, dim=1)
    selected = same.gather(1, indices[:, :, None].expand(-1, -1, points))
    selected = torch.maximum(selected, F.one_hot(indices, points).float())
    membership = selected * valid[..., None].float()
    return membership, valid


def _component_boxes(evidence, membership, frame_indices):
    coordinates = evidence.coordinates.float().index_select(1, frame_indices)
    visibility = evidence.visibility.index_select(1, frame_indices)
    weight = visibility[:, :, None].float() * membership[:, None]
    supported = weight > 0.05
    x, y = coordinates[..., 0], coordinates[..., 1]
    large = torch.full_like(x[:, :, None], 2.0)
    x_values, y_values = x[:, :, None], y[:, :, None]
    minimum_x = torch.where(supported, x_values, large).amin(dim=-1)
    maximum_x = torch.where(supported, x_values, -large).amax(dim=-1)
    minimum_y = torch.where(supported, y_values, large).amin(dim=-1)
    maximum_y = torch.where(supported, y_values, -large).amax(dim=-1)
    valid = supported.any(dim=-1)
    center_x = (minimum_x + maximum_x) * 0.5
    center_y = (minimum_y + maximum_y) * 0.5
    half_x = ((maximum_x - minimum_x) * 0.65).clamp(0.15, 1.0)
    half_y = ((maximum_y - minimum_y) * 0.65).clamp(0.15, 1.0)
    boxes = torch.stack((center_x, center_y, half_x, half_y), dim=-1)
    return boxes, valid


def _crop_objects(rgb, boxes, valid, image_size: int):
    batch, anchors, roots = boxes.shape[:3]
    images = rgb.index_select(1, torch.arange(anchors, device=rgb.device))
    images = images[:, :, None].expand(-1, -1, roots, -1, -1, -1)
    images = images.reshape(batch * anchors * roots, *rgb.shape[2:]).float() / 255.0
    flat_boxes = boxes.reshape(-1, 4)
    theta = torch.zeros(len(flat_boxes), 2, 3, device=rgb.device)
    theta[:, 0, 0] = flat_boxes[:, 2]
    theta[:, 1, 1] = flat_boxes[:, 3]
    theta[:, 0, 2] = flat_boxes[:, 0]
    theta[:, 1, 2] = flat_boxes[:, 1]
    grid = F.affine_grid(
        theta,
        (len(flat_boxes), 3, image_size, image_size),
        align_corners=True,
    )
    crops = F.grid_sample(images, grid, mode="bilinear", align_corners=True)
    crops = crops * valid.reshape(-1, 1, 1, 1).float()
    return crops


class FrozenSiglip2ObjectTeacherV61:
    def __init__(self, checkpoint: str, device: torch.device, frame_batch: int):
        from transformers import Siglip2VisionModel

        self.model = (
            Siglip2VisionModel.from_pretrained(checkpoint, local_files_only=True)
            .to(device)
            .eval()
        )
        self.model.requires_grad_(False)
        self.device = device
        self.frame_batch = int(frame_batch)
        self.patch_size = int(self.model.config.patch_size)
        self.max_patches = int(self.model.config.num_patches)
        self.feature_dim = int(self.model.config.hidden_size)
        self.image_size = 224

    def _patchify(self, images):
        images = (images - 0.5) / 0.5
        grid = self.image_size // self.patch_size
        patches = images.reshape(
            len(images), 3, grid, self.patch_size, grid, self.patch_size
        )
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(
            len(images), grid * grid, -1
        )
        actual = patches.shape[1]
        if actual > self.max_patches:
            raise RuntimeError("v61 object crop exceeds SigLIP2 patch capacity")
        mask = torch.ones(len(images), actual, device=images.device, dtype=torch.bool)
        if actual < self.max_patches:
            patches = F.pad(patches, (0, 0, 0, self.max_patches - actual))
            mask = F.pad(mask, (0, self.max_patches - actual), value=False)
        shapes = torch.tensor((grid, grid), device=images.device)[None].expand(
            len(images), -1
        )
        return patches, mask, shapes

    @torch.no_grad()
    def __call__(self, crops: torch.Tensor) -> torch.Tensor:
        outputs = []
        for start in range(0, len(crops), self.frame_batch):
            stop = min(start + self.frame_batch, len(crops))
            patches, mask, shapes = self._patchify(crops[start:stop])
            output = self.model(
                pixel_values=patches,
                pixel_attention_mask=mask,
                spatial_shapes=shapes,
            )
            outputs.append(F.normalize(output.pooler_output.float(), dim=-1, eps=1e-6))
        return torch.cat(outputs)


@torch.no_grad()
def build_object_components_v61(
    batch,
    evidence,
    relation,
    object_roots: int,
    semantic_teacher: FrozenSiglip2ObjectTeacherV61 | None,
):
    membership, valid = select_soft_object_components(relation, object_roots)
    frame_indices = torch.tensor(
        (0, batch["video_rgb"].shape[1] - 1),
        device=batch["video_rgb"].device,
        dtype=torch.long,
    ).unique()
    semantic, semantic_valid = None, None
    if semantic_teacher is not None:
        boxes, semantic_valid = _component_boxes(evidence, membership, frame_indices)
        selected_rgb = batch["video_rgb"].index_select(1, frame_indices)
        crops = _crop_objects(
            selected_rgb, boxes, semantic_valid, semantic_teacher.image_size
        )
        flat_valid = semantic_valid.reshape(-1)
        semantic = crops.new_zeros(len(crops), semantic_teacher.feature_dim)
        if bool(flat_valid.any()):
            semantic[flat_valid] = semantic_teacher(crops[flat_valid])
        semantic = semantic.reshape(
            len(membership), len(frame_indices), object_roots, -1
        )
    return TeacherObjectComponentsV61(
        membership=membership.detach(),
        valid=valid.detach(),
        semantic=None if semantic is None else semantic.detach(),
        semantic_valid=None if semantic_valid is None else semantic_valid.detach(),
        frame_indices=frame_indices.detach(),
    )
