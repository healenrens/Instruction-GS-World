"""Held-teacher representation sufficiency metrics for V61 Object State."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .representation_probes_v61 import representation_probe_metrics_v61


@dataclass(frozen=True)
class SufficiencyConditionV61:
    identity: torch.Tensor
    teacher_identity: torch.Tensor
    dynamic: torch.Tensor
    root_assignment: torch.Tensor
    point_owner: torch.Tensor
    root_owner: torch.Tensor
    carrier_presence: torch.Tensor
    root_presence: torch.Tensor
    visibility: torch.Tensor
    lifecycle_known: torch.Tensor
    presence: torch.Tensor
    coordinates: torch.Tensor
    motion: torch.Tensor
    motion_valid: torch.Tensor
    same: torch.Tensor
    different: torch.Tensor
    object_confidence: torch.Tensor
    scene_confidence: torch.Tensor
    source_index: int
    group_index: int
    chunk_length: int
    temporal_step_seconds: float
    decode_replaced: float
    parts: dict[str, float]


def condition_from_v61_output(output, evidence, relation, batch):
    state = output["state"]
    assignment = output["carrier_assignment"].float()
    if assignment.shape[0] != 1:
        raise ValueError("V61 sufficiency conditions require batch size one")
    identity = torch.einsum(
        "btpq,btqd->btpd", assignment, state.carriers.identity.float()
    )
    identity = F.normalize(identity, dim=-1, eps=1e-6)
    dynamic = torch.einsum(
        "btpq,btqd->btpd", assignment, state.carriers.dynamic.float()
    )
    point_owner = torch.einsum("btpq,btqo->btpo", assignment, state.roots.owner.float())
    return SufficiencyConditionV61(
        identity=identity[0].detach().cpu(),
        teacher_identity=F.normalize(
            evidence.sampled_features[0].float(), dim=-1, eps=1e-6
        )
        .detach()
        .cpu(),
        dynamic=dynamic[0].detach().cpu(),
        root_assignment=output["root_assignment"][0].float().detach().cpu(),
        point_owner=point_owner[0].detach().cpu(),
        root_owner=state.roots.owner[0].float().detach().cpu(),
        carrier_presence=state.carriers.presence[0].float().detach().cpu(),
        root_presence=state.roots.presence[0].float().detach().cpu(),
        visibility=evidence.visibility[0].bool().detach().cpu(),
        lifecycle_known=relation.lifecycle_known[0].bool().detach().cpu(),
        presence=relation.presence[0].float().detach().cpu(),
        coordinates=evidence.coordinates[0].float().detach().cpu(),
        motion=relation.motion[0].float().detach().cpu(),
        motion_valid=relation.motion_valid[0].bool().detach().cpu(),
        same=relation.same_confidence[0].float().detach().cpu(),
        different=relation.different_confidence[0].float().detach().cpu(),
        object_confidence=relation.object_confidence[0].float().detach().cpu(),
        scene_confidence=relation.scene_confidence[0].float().detach().cpu(),
        source_index=int(batch["source_index"][0]),
        group_index=int(batch["task_group_index"][0]),
        chunk_length=int(batch["chunk_length"][0]),
        temporal_step_seconds=float(batch["temporal_step_seconds"][0]),
        decode_replaced=float(batch["decode_replaced"][0]),
        parts={name: float(value) for name, value in output["parts"].items()},
    )


def _weighted_mean(value, weight):
    return float(
        (value.float() * weight.float()).sum() / weight.float().sum().clamp_min(1.0)
    )


def _rank_metrics(vectors, prefix, variance_floor):
    centered = vectors.float() - vectors.float().mean(dim=0, keepdim=True)
    variance = centered.square().mean(dim=0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    effective = eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-12)
    usable = float(min(max(len(centered) - 1, 1), centered.shape[-1]))
    sample = F.normalize(vectors.float()[: min(len(vectors), 2048)], dim=-1, eps=1e-6)
    gram = sample @ sample.T
    off_diagonal = (gram.sum() - gram.diagonal().sum()) / max(
        len(sample) ** 2 - len(sample), 1
    )
    return {
        f"capacity/{prefix}_active_units": float((variance >= variance_floor).sum()),
        f"capacity/{prefix}_active_fraction": float(
            (variance >= variance_floor).float().mean()
        ),
        f"capacity/{prefix}_effective_rank": float(effective),
        f"capacity/{prefix}_effective_rank_fraction": float(effective / usable),
        f"capacity/{prefix}_off_diagonal_cosine": float(off_diagonal),
    }


def _topk_retrieval(records, prefix, reappearance=False, feature_name="identity"):
    hits = {1: 0.0, 5: 0.0, 10: 0.0}
    chance = {1: 0.0, 5: 0.0, 10: 0.0}
    queries = 0.0
    for record in records:
        features = getattr(record, feature_name)
        if reappearance:
            last = features[0].clone()
            seen = record.visibility[0].clone()
            gap = torch.zeros_like(seen)
            frame_pairs = []
            for frame in range(1, len(features)):
                current = record.visibility[frame] & record.lifecycle_known[frame]
                reappeared = current & seen & gap
                if bool(reappeared.any()):
                    frame_pairs.append(
                        (last[reappeared], features[frame], reappeared, current)
                    )
                last = torch.where(current[:, None], features[frame], last)
                gap = torch.where(current, torch.zeros_like(gap), gap | seen)
                seen = seen | current
        else:
            valid = record.visibility[0] & record.visibility[-1]
            odd = torch.arange(len(valid)) % 2 == 1
            valid = valid & odd
            frame_pairs = [(features[0][valid], features[-1], valid, valid)]
        for query, gallery_all, labels_mask, gallery_mask in frame_pairs:
            gallery_indices = gallery_mask.nonzero(as_tuple=False)[:, 0]
            label_indices = labels_mask.nonzero(as_tuple=False)[:, 0]
            if len(query) == 0 or len(gallery_indices) == 0:
                continue
            gallery = gallery_all[gallery_indices]
            similarity = F.normalize(query, dim=-1) @ F.normalize(gallery, dim=-1).T
            labels = (
                (gallery_indices[None] == label_indices[:, None]).float().argmax(dim=-1)
            )
            ranking = similarity.argsort(dim=-1, descending=True)
            queries += len(query)
            for k in hits:
                width = min(k, len(gallery))
                hits[k] += float(
                    (ranking[:, :width] == labels[:, None]).any(dim=1).sum()
                )
                chance[k] += len(query) * width / len(gallery)
    metrics = {f"retrieval/{prefix}_queries": queries}
    for k in hits:
        metrics[f"retrieval/{prefix}_recall_at_{k}"] = hits[k] / max(queries, 1.0)
        metrics[f"retrieval/{prefix}_chance_at_{k}"] = chance[k] / max(queries, 1.0)
    return metrics


def _owner_relation_metrics(records):
    owner_mass = torch.cat(
        [
            record.root_owner.reshape(-1, record.root_owner.shape[-1])
            for record in records
        ]
    )
    distribution = owner_mass.mean(dim=0)
    entropy = -(distribution * distribution.clamp_min(1e-8).log()).sum()
    effective = torch.exp(entropy)
    object_scene_sum = scene_object_sum = object_weight = scene_weight = 0.0
    same_sum = different_sum = same_weight = different_weight = 0.0
    inside_sum = outside_sum = deletion_count = 0.0
    for record in records:
        object_weight += float(record.object_confidence.sum()) * len(record.identity)
        scene_weight += float(record.scene_confidence.sum()) * len(record.identity)
        object_scene_sum += float(
            (record.point_owner[..., -1] * record.object_confidence[None]).sum()
        )
        scene_object_sum += float(
            (
                record.point_owner[..., :-1].sum(dim=-1) * record.scene_confidence[None]
            ).sum()
        )
        visible = record.visibility.float()
        track_identity = (record.identity * visible[..., None]).sum(dim=0)
        track_identity = F.normalize(
            track_identity / visible.sum(dim=0)[:, None].clamp_min(1.0), dim=-1
        )
        similarity = track_identity @ track_identity.T
        diagonal = torch.eye(len(similarity), dtype=torch.bool)
        same = record.same.masked_fill(diagonal, 0.0)
        different = record.different.masked_fill(diagonal, 0.0)
        same_sum += float((similarity * same).sum())
        different_sum += float((similarity * different).sum())
        same_weight += float(same.sum())
        different_weight += float(different.sum())
        root_average = (record.root_assignment * visible[..., None]).sum(dim=0)
        root_average = root_average / visible.sum(dim=0)[:, None].clamp_min(1.0)
        anchors = record.object_confidence.topk(
            min(16, len(record.object_confidence))
        ).indices
        for anchor in anchors.tolist():
            inside, outside = record.same[anchor].clone(), record.different[anchor]
            inside[anchor] = 1.0
            if float(inside.sum()) <= 0.0 or float(outside.sum()) <= 0.0:
                continue
            root = int(root_average[anchor].argmax())
            impact = root_average[:, root]
            inside_sum += _weighted_mean(impact, inside)
            outside_sum += _weighted_mean(impact, outside)
            deletion_count += 1.0
    inside = inside_sum / max(deletion_count, 1.0)
    outside = outside_sum / max(deletion_count, 1.0)
    return {
        "owner/effective_categories": float(effective),
        "owner/entropy": float(entropy),
        "owner/maximum_fraction": float(distribution.max()),
        "owner/scene_fraction": float(distribution[-1]),
        "owner/object_evidence_to_scene": object_scene_sum / max(object_weight, 1.0),
        "owner/scene_evidence_to_object": scene_object_sum / max(scene_weight, 1.0),
        "relation/identity_same_similarity": same_sum / max(same_weight, 1.0),
        "relation/identity_different_similarity": different_sum
        / max(different_weight, 1.0),
        "relation/identity_similarity_margin": same_sum / max(same_weight, 1.0)
        - different_sum / max(different_weight, 1.0),
        "deletion/external_track_inside_mass": inside,
        "deletion/external_track_outside_mass": outside,
        "deletion/external_track_locality_ratio": inside / max(outside, 1e-8),
        "deletion/external_track_events": deletion_count,
    }


def evaluate_representation_sufficiency_v61(
    records,
    variance_floor=1e-4,
    sample_limit=16384,
    mlp_steps=100,
    include_probes=True,
):
    if not records:
        raise ValueError("V61 sufficiency evaluation has no conditions")
    visible_identity = torch.cat(
        [record.identity[record.visibility] for record in records]
    )
    visible_teacher_identity = torch.cat(
        [record.teacher_identity[record.visibility] for record in records]
    )
    visible_dynamic = torch.cat(
        [record.dynamic[record.visibility] for record in records]
    )
    metrics = {}
    metrics.update(_rank_metrics(visible_identity, "identity", variance_floor))
    metrics.update(
        _rank_metrics(visible_teacher_identity, "teacher_dino", variance_floor)
    )
    metrics.update(_rank_metrics(visible_dynamic, "dynamic", variance_floor))
    metrics.update(_topk_retrieval(records, "first_last"))
    metrics.update(_topk_retrieval(records, "reappearance", reappearance=True))
    metrics.update(
        _topk_retrieval(
            records, "teacher_dino_first_last", feature_name="teacher_identity"
        )
    )
    metrics.update(
        _topk_retrieval(
            records,
            "teacher_dino_reappearance",
            reappearance=True,
            feature_name="teacher_identity",
        )
    )
    metrics.update(_owner_relation_metrics(records))
    if include_probes:
        metrics.update(
            representation_probe_metrics_v61(records, sample_limit, mlp_steps)
        )
    metrics["data/conditions"] = float(len(records))
    metrics["data/decode_replacement_rate"] = sum(
        record.decode_replaced for record in records
    ) / len(records)
    metrics["capacity/carrier_presence_mean"] = float(
        torch.cat([record.carrier_presence.flatten() for record in records]).mean()
    )
    metrics["capacity/effective_carriers"] = float(
        torch.stack(
            [record.carrier_presence.sum(dim=-1).mean() for record in records]
        ).mean()
    )
    metrics["capacity/effective_roots"] = float(
        torch.stack(
            [
                (
                    record.root_presence.sum(dim=-1).square()
                    / record.root_presence.square().sum(dim=-1).clamp_min(1e-8)
                ).mean()
                for record in records
            ]
        ).mean()
    )
    part_names = sorted({name for record in records for name in record.parts})
    for name in part_names:
        values = [record.parts[name] for record in records if name in record.parts]
        metrics[f"distortion/{name}"] = sum(values) / len(values)
    return metrics
