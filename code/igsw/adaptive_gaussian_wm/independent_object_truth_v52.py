"""Independent object-truth clips and metrics for v52 promotion."""

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_jpeg

from .rgb_episode_cache_contract import validate_manifest
from .sequence_contract import EPISODE_MANIFEST_NAME


INDEPENDENT_OBJECT_TRUTH_CONTRACT = "independent_object_truth_v1"


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _decode_frames(rgb: dict, indices: torch.Tensor) -> torch.Tensor:
    blob, offsets = rgb["jpeg_bytes"], rgb["jpeg_offsets"].long()
    return torch.stack(
        [
            decode_jpeg(
                blob[int(offsets[index]) : int(offsets[index + 1])],
                mode=ImageReadMode.RGB,
            )
            for index in indices.tolist()
        ]
    )


def _validate_annotation(payload: dict, entry: dict, episode: dict) -> None:
    if payload.get("contract") != INDEPENDENT_OBJECT_TRUTH_CONTRACT:
        raise ValueError("independent object annotation contract differs")
    if payload.get("episode_filename") != entry["episode_filename"]:
        raise ValueError("independent annotation episode differs from manifest")
    frame_indices = payload.get("frame_indices")
    masks = payload.get("instance_masks")
    object_ids = payload.get("object_ids")
    presence = payload.get("object_presence")
    if not torch.is_tensor(frame_indices) or frame_indices.dtype != torch.int64:
        raise ValueError("independent frame_indices must be int64")
    if frame_indices.ndim != 1 or not 3 <= len(frame_indices) <= 32:
        raise ValueError("independent clip length must stay within [3,32]")
    if not bool((frame_indices[1:] > frame_indices[:-1]).all()):
        raise ValueError("independent frame_indices must increase")
    if int(frame_indices[0]) < 0 or int(frame_indices[-1]) >= int(episode["frame_count"]):
        raise ValueError("independent frame_indices exceed the RGB episode")
    if not torch.is_tensor(masks) or masks.dtype not in (torch.int32, torch.int64):
        raise ValueError("independent instance_masks must use an integer dtype")
    if masks.ndim != 3 or masks.shape[0] != len(frame_indices):
        raise ValueError("independent instance_masks shape differs")
    if not torch.is_tensor(object_ids) or object_ids.dtype != torch.int64:
        raise ValueError("independent object_ids must be int64")
    if object_ids.ndim != 1 or len(object_ids) < 1:
        raise ValueError("independent annotation has no objects")
    if not bool((object_ids > 0).all()) or len(object_ids.unique()) != len(object_ids):
        raise ValueError("independent object_ids must be unique and positive")
    labels = masks.unique()
    allowed = torch.cat((torch.tensor([-1, 0], dtype=labels.dtype), object_ids.to(labels.dtype)))
    if not bool(torch.isin(labels, allowed).all()):
        raise ValueError("independent masks contain undeclared object IDs")
    if not torch.is_tensor(presence) or presence.dtype not in (torch.int8, torch.int16):
        raise ValueError("independent object_presence must use int8 or int16")
    if presence.shape != (len(frame_indices), len(object_ids)):
        raise ValueError("independent object_presence shape differs")
    if not bool(torch.isin(presence, torch.tensor([-1, 0, 1], dtype=presence.dtype)).all()):
        raise ValueError("independent object_presence must use unknown/absent/present")
    visible = torch.stack([(masks == object_id).flatten(1).any(dim=1) for object_id in object_ids], dim=1)
    if not bool((presence[visible] == 1).all()):
        raise ValueError("visible independent objects must be present")
    if not bool((visible.sum(dim=0) >= 2).all()):
        raise ValueError("each independent object needs at least two visible frames")
    if not bool((presence >= 0).any(dim=0).all()):
        raise ValueError("each independent object needs known lifecycle evidence")


class IndependentObjectTruthDataset(Dataset):
    """Decode RGB clips paired with evaluation-only stable object truth."""

    def __init__(self, data_root: str, truth_manifest: str, splits: tuple[str, ...]):
        self.data_root = os.path.abspath(data_root)
        self.truth_manifest_path = os.path.abspath(truth_manifest)
        self.truth_root = os.path.dirname(self.truth_manifest_path)
        data_manifest_path = os.path.join(self.data_root, EPISODE_MANIFEST_NAME)
        self.data_manifest = _read_json(data_manifest_path)
        self.control_hz = validate_manifest(self.data_manifest, data_manifest_path)
        episodes = {
            entry["filename"]: entry for entry in self.data_manifest["episodes"]
        }
        truth = _read_json(self.truth_manifest_path)
        if truth.get("contract") != INDEPENDENT_OBJECT_TRUTH_CONTRACT:
            raise ValueError("independent object-truth manifest contract differs")
        if truth.get("complete") is not True:
            raise ValueError("independent object-truth manifest is incomplete")
        provenance = truth.get("provenance", {})
        if provenance.get("kind") not in ("simulator_ground_truth", "human_annotation"):
            raise ValueError("independent object truth has unsupported provenance")
        if provenance.get("uses_training_tracker") is not False:
            raise ValueError("independent object truth cannot use the training tracker")
        entries = [entry for entry in truth.get("items", ()) if entry.get("split") in splits]
        if not entries:
            raise ValueError("independent object-truth manifest has no requested items")
        available_splits = {entry.get("split") for entry in entries}
        missing_splits = set(splits) - available_splits
        if missing_splits:
            raise ValueError(
                f"independent object-truth manifest is missing splits: {sorted(missing_splits)}"
            )
        for entry in entries:
            filename = entry.get("episode_filename")
            if filename not in episodes:
                raise ValueError(f"independent truth references unknown episode: {filename}")
            if episodes[filename]["split"] != entry.get("split"):
                raise ValueError("independent truth split differs from RGB cache")
            annotation = os.path.abspath(os.path.join(self.truth_root, entry["annotation"]))
            if not os.path.isfile(annotation):
                raise ValueError(f"independent annotation is missing: {annotation}")
            entry["annotation"] = annotation
        self.entries = entries
        self.episodes = episodes
        self.provenance = provenance

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        entry = self.entries[index]
        episode = self.episodes[entry["episode_filename"]]
        annotation = torch.load(entry["annotation"], map_location="cpu", weights_only=False)
        _validate_annotation(annotation, entry, episode)
        cache_path = os.path.join(self.data_root, entry["episode_filename"])
        cache = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
        frame_indices = annotation["frame_indices"]
        rgb = _decode_frames(cache["rgb"], frame_indices)
        masks = annotation["instance_masks"].long()
        if masks.shape[-2:] != rgb.shape[-2:]:
            raise ValueError("independent truth resolution differs from cached RGB")
        return {
            "video_rgb": rgb,
            "video_pixel_valid": torch.ones_like(masks, dtype=torch.bool),
            "frame_times": frame_indices.float() / self.control_hz,
            "control_indices": frame_indices,
            "instance_masks": masks,
            "object_ids": annotation["object_ids"],
            "object_presence": annotation["object_presence"].long(),
            "split": str(entry["split"]),
            "episode_filename": str(entry["episode_filename"]),
        }


def patch_truth_masks(instance_masks: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        instance_masks[:, None].float(), size=grid_hw, mode="nearest"
    )[:, 0].long().flatten(1)


def _mean(value: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    weight = mask.float()
    return float((value.float() * weight).sum()), float(weight.sum())


def _object_root_distributions(
    assignment: torch.Tensor,
    masks: torch.Tensor,
    object_ids: torch.Tensor,
    object_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    distributions, visible = [], []
    for object_id in object_ids:
        support = masks == object_id
        mass = torch.einsum(
            "tnk,tn->tk", assignment[..., :object_slots], support.float()
        )
        mass = mass / support.sum(dim=1, keepdim=True).clamp_min(1.0)
        distributions.append(mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-6))
        visible.append(support.any(dim=1))
    return torch.stack(distributions, dim=1), torch.stack(visible, dim=1)


def independent_object_metrics(model, features, state, decoder_assignment, truth) -> dict:
    assignment = state["assignment"][0].float()
    decoder = decoder_assignment[0].float()
    masks = patch_truth_masks(truth["instance_masks"], features.grid_hw)
    object_ids = truth["object_ids"]
    valid = masks >= 0
    foreground = masks > 0
    background = masks == 0
    object_probability = assignment[..., : model.config.object_slots].sum(dim=-1)
    scene_probability = assignment[..., model.config.object_slots]
    sums: dict[str, tuple[float, float]] = {}
    sums["foreground_object_routing"] = _mean(object_probability, foreground)
    sums["background_scene_routing"] = _mean(scene_probability, background)

    same_sum = same_count = different_sum = different_count = 0.0
    normalized = assignment[..., : model.config.object_slots]
    normalized = normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    for frame in range(len(masks)):
        selected = foreground[frame] & valid[frame]
        labels = masks[frame, selected]
        roots = normalized[frame, selected]
        if len(labels) < 2:
            continue
        similarity = roots @ roots.T
        diagonal = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
        same = (labels[:, None] == labels[None]) & ~diagonal
        different = (labels[:, None] != labels[None]) & ~diagonal
        value, count = _mean(similarity, same)
        same_sum, same_count = same_sum + value, same_count + count
        value, count = _mean(similarity, different)
        different_sum, different_count = different_sum + value, different_count + count
    sums["same_object_root_similarity"] = (same_sum, same_count)
    sums["different_object_root_similarity"] = (different_sum, different_count)

    decoder_object_probability = decoder[..., : model.config.object_slots].sum(dim=-1)
    decoder_scene_probability = decoder[..., model.config.object_slots]
    sums["decoder_foreground_object_routing"] = _mean(
        decoder_object_probability, foreground
    )
    sums["decoder_background_scene_routing"] = _mean(
        decoder_scene_probability, background
    )
    decoder_same_sum = decoder_same_count = 0.0
    decoder_different_sum = decoder_different_count = 0.0
    decoder_roots = decoder[..., : model.config.object_slots]
    decoder_roots = decoder_roots / decoder_roots.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    for frame in range(len(masks)):
        selected = foreground[frame] & valid[frame]
        labels = masks[frame, selected]
        roots = decoder_roots[frame, selected]
        if len(labels) < 2:
            continue
        similarity = roots @ roots.T
        diagonal = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
        same = (labels[:, None] == labels[None]) & ~diagonal
        different = (labels[:, None] != labels[None]) & ~diagonal
        value, count = _mean(similarity, same)
        decoder_same_sum, decoder_same_count = (
            decoder_same_sum + value,
            decoder_same_count + count,
        )
        value, count = _mean(similarity, different)
        decoder_different_sum, decoder_different_count = (
            decoder_different_sum + value,
            decoder_different_count + count,
        )
    sums["decoder_same_object_similarity"] = (
        decoder_same_sum,
        decoder_same_count,
    )
    sums["decoder_different_object_similarity"] = (
        decoder_different_sum,
        decoder_different_count,
    )

    root_distribution, visibility = _object_root_distributions(
        assignment, masks, object_ids, model.config.object_slots
    )
    if not bool((visibility.sum(dim=0) >= 2).all()):
        raise ValueError("independent object disappears at the DINO patch grid")
    canonical = root_distribution.sum(dim=0)
    canonical = canonical / visibility.sum(dim=0).clamp_min(1.0)[:, None]
    canonical_slot = canonical.argmax(dim=-1)
    root_entropy = -(
        canonical * canonical.clamp_min(1e-8).log()
    ).sum(dim=-1)
    sums["object_effective_roots"] = (
        float(root_entropy.exp().sum()),
        float(len(root_entropy)),
    )
    temporal_similarity = (root_distribution * canonical[None]).sum(dim=-1)
    sums["object_assignment_temporal_similarity"] = _mean(
        temporal_similarity, visibility
    )
    identity_values, identity_same_values, center_values = [], [], []
    reappearance_values, reappearance_other_values = [], []
    object_identities, object_visibility = [], []
    grid_coordinates = features.coordinates[0]
    for object_index, object_id in enumerate(object_ids):
        slot = int(canonical_slot[object_index])
        object_visible = visibility[:, object_index]
        identity = F.normalize(state["identity"][0, :, slot].float(), dim=-1, eps=1e-6)
        object_identities.append(identity)
        object_visibility.append(object_visible)
        mean_identity = F.normalize(
            identity[object_visible].mean(dim=0), dim=-1, eps=1e-6
        )
        identity_values.extend((identity[object_visible] * mean_identity).sum(dim=-1))
        visible_identity = identity[object_visible]
        if len(visible_identity) > 1:
            similarity = visible_identity @ visible_identity.T
            off_diagonal = ~torch.eye(
                len(visible_identity), device=similarity.device, dtype=torch.bool
            )
            identity_same_values.extend(similarity[off_diagonal])
        for frame in torch.where(object_visible)[0].tolist():
            support = masks[frame] == object_id
            target_center = grid_coordinates[frame, support].float().mean(dim=0)
            center_values.append(
                (state["center"][0, frame, slot].float() - target_center).norm()
            )
        visible_indices = torch.where(object_visible)[0]
        for left, right in zip(visible_indices[:-1], visible_indices[1:]):
            occluded_between = truth["object_presence"][left + 1 : right, object_index]
            if int(right - left) > 1 and bool((occluded_between == 1).all()):
                other_values = []
                for other_index in range(len(object_ids)):
                    if other_index == object_index:
                        continue
                    other_support = masks[right] == object_ids[other_index]
                    if bool(other_support.any()):
                        other_slot = int(canonical_slot[other_index])
                        other_identity = F.normalize(
                            state["identity"][0, right, other_slot].float(),
                            dim=-1,
                            eps=1e-6,
                        )
                        other_values.append((identity[left] * other_identity).sum())
                if other_values:
                    reappearance_values.append((identity[left] * identity[right]).sum())
                    reappearance_other_values.extend(other_values)
    identity_different_values = []
    for left in range(len(object_ids)):
        for right in range(left + 1, len(object_ids)):
            co_visible = object_visibility[left] & object_visibility[right]
            identity_different_values.extend(
                (object_identities[left][co_visible] * object_identities[right][co_visible])
                .sum(dim=-1)
            )
    identity_tensor = torch.stack(identity_values) if identity_values else assignment.new_zeros(0)
    identity_same_tensor = (
        torch.stack(identity_same_values)
        if identity_same_values else assignment.new_zeros(0)
    )
    identity_different_tensor = (
        torch.stack(identity_different_values)
        if identity_different_values else assignment.new_zeros(0)
    )
    center_tensor = torch.stack(center_values) if center_values else assignment.new_zeros(0)
    reappearance_tensor = (
        torch.stack(reappearance_values) if reappearance_values else assignment.new_zeros(0)
    )
    reappearance_other_tensor = (
        torch.stack(reappearance_other_values)
        if reappearance_other_values else assignment.new_zeros(0)
    )
    sums["identity_temporal_cosine"] = (float(identity_tensor.sum()), float(len(identity_tensor)))
    sums["identity_same_object_cosine"] = (
        float(identity_same_tensor.sum()),
        float(len(identity_same_tensor)),
    )
    sums["identity_different_object_cosine"] = (
        float(identity_different_tensor.sum()),
        float(len(identity_different_tensor)),
    )
    sums["relative_center_error"] = (float(center_tensor.sum()), float(len(center_tensor)))
    sums["reappearance_identity_cosine"] = (
        float(reappearance_tensor.sum()), float(len(reappearance_tensor))
    )
    sums["reappearance_other_identity_cosine"] = (
        float(reappearance_other_tensor.sum()),
        float(len(reappearance_other_tensor)),
    )
    sums["identity_different_pairs"] = (float(len(identity_different_tensor)), 1.0)

    visibility_scores, presence_scores = [], []
    visibility_targets, presence_targets = [], []
    occluded_presence_scores, absent_presence_scores = [], []
    truth_presence = truth["object_presence"]
    for object_index in range(len(object_ids)):
        slot = int(canonical_slot[object_index])
        visibility_score = state["visibility"][0, :, slot].float()
        presence_score = state["presence"][0, :, slot].float()
        visibility_scores.append(visibility_score)
        visibility_targets.append(visibility[:, object_index].float())
        known = truth_presence[:, object_index] >= 0
        presence_scores.append(presence_score[known])
        presence_targets.append(truth_presence[known, object_index].float())
        occluded = (truth_presence[:, object_index] == 1) & ~visibility[:, object_index]
        absent = truth_presence[:, object_index] == 0
        occluded_presence_scores.extend(presence_score[occluded])
        absent_presence_scores.extend(presence_score[absent])
    visibility_score = torch.cat(visibility_scores)
    visibility_target = torch.cat(visibility_targets)
    presence_score = torch.cat(presence_scores)
    presence_target = torch.cat(presence_targets)
    sums["visibility_accuracy"] = _mean(
        ((visibility_score >= 0.5) == visibility_target.bool()).float(),
        torch.ones_like(visibility_target, dtype=torch.bool),
    )
    sums["presence_accuracy"] = _mean(
        ((presence_score >= 0.5) == presence_target.bool()).float(),
        torch.ones_like(presence_target, dtype=torch.bool),
    )
    visible = visibility_target.bool()
    sums["visible_visibility_recall"] = _mean(
        (visibility_score >= 0.5).float(), visible
    )
    sums["invisible_visibility_rejection"] = _mean(
        (visibility_score < 0.5).float(), ~visible
    )
    occluded_presence = (
        torch.stack(occluded_presence_scores)
        if occluded_presence_scores else assignment.new_zeros(0)
    )
    absent_presence = (
        torch.stack(absent_presence_scores)
        if absent_presence_scores else assignment.new_zeros(0)
    )
    sums["occluded_presence_recall"] = (
        float((occluded_presence >= 0.5).float().sum()),
        float(len(occluded_presence)),
    )
    sums["absent_presence_rejection"] = (
        float((absent_presence < 0.5).float().sum()),
        float(len(absent_presence)),
    )
    occluded = (truth_presence == 1) & ~visibility
    absent = truth_presence == 0
    sums["independent_objects"] = (float(len(object_ids)), 1.0)
    sums["reappearance_cases"] = (float(len(reappearance_tensor)), 1.0)
    sums["occluded_cases"] = (float(occluded.sum()), 1.0)
    sums["absent_cases"] = (float(absent.sum()), 1.0)
    sums["annotated_frames"] = (float(len(masks)), 1.0)
    sums["valid_patch_fraction"] = _mean(valid.float(), torch.ones_like(valid))
    return sums


@torch.no_grad()
def independent_deletion_metrics(model, features, state, truth, amp_context) -> dict:
    masks = patch_truth_masks(truth["instance_masks"], features.grid_hw)
    assignment = state["assignment"][0].float()
    distribution, visibility = _object_root_distributions(
        assignment, masks, truth["object_ids"], model.config.object_slots
    )
    areas = torch.stack([(masks == object_id).sum(dim=1) for object_id in truth["object_ids"]], dim=1)
    totals = {"deletion_inside": [0.0, 0.0], "deletion_outside": [0.0, 0.0]}
    for object_index, object_id in enumerate(truth["object_ids"]):
        frame = int(areas[:, object_index].argmax())
        if not bool(visibility[frame, object_index]):
            continue
        slot = int(distribution[:, object_index].sum(dim=0).argmax())
        frame_state = {
            name: value[0:1, frame]
            for name, value in state.items()
            if name not in ("assignment", "mass")
        }
        coordinates = features.coordinates[0:1, frame]
        valid = features.valid[0:1, frame]
        with amp_context():
            reference, _ = model.decoder(frame_state, coordinates, valid)
            object_valid = torch.ones(
                1, model.config.object_slots, device=reference.device, dtype=torch.bool
            )
            object_valid[:, slot] = False
            deleted, _ = model.decoder(frame_state, coordinates, valid, object_valid)
        change = 1.0 - F.cosine_similarity(reference.float(), deleted.float(), dim=-1)
        inside = (masks[frame] == object_id) & valid[0]
        outside = (masks[frame] >= 0) & ~inside & valid[0]
        for name, mask in (("deletion_inside", inside), ("deletion_outside", outside)):
            value_sum, value_count = _mean(change[0], mask)
            totals[name][0] += value_sum
            totals[name][1] += value_count
    return {name: tuple(value) for name, value in totals.items()}
