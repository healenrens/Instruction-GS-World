"""Real-data admission gates for v56 teacher evidence and target geometry."""

from __future__ import annotations

import torch

from .verified_relation_objective_v56 import balanced_relation_weights


def _batch_weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    dimensions = tuple(range(1, value.ndim))
    numerator = (value * weight).sum(dim=dimensions)
    denominator = weight.sum(dim=dimensions)
    usable = (denominator > 0.0).float()
    per_sample = numerator / denominator.clamp_min(1e-6)
    return (per_sample * usable).sum() / usable.sum().clamp_min(1.0)


def _graph_loss(assignment: torch.Tensor, teacher) -> torch.Tensor:
    similarity = torch.einsum("bpk,bqk->bpq", assignment.float(), assignment.float())
    same_weight, different_weight = balanced_relation_weights(teacher)
    weight = same_weight + different_weight
    target = same_weight / weight.clamp_min(1e-6)
    probability = similarity.clamp(1e-6, 1.0 - 1e-6)
    bce = -(target * probability.log() + (1.0 - target) * torch.log1p(-probability))
    return _batch_weighted_mean(bce, weight)


def _root_statistics(assignment: torch.Tensor, teacher):
    degree = (teacher.same_confidence + teacher.different_confidence).amax(dim=-1)
    weight = teacher.object_confidence * degree
    root_mass = (assignment * weight[..., None]).sum(dim=1)
    share = root_mass / root_mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    entropy = -(share * share.clamp_min(1e-6).log()).sum(dim=-1)
    effective = entropy.exp()
    negative_supported = teacher.different_confidence.amax(dim=(-2, -1)) > 0.0
    usable = negative_supported.float()
    conditional_effective = (effective * usable).sum() / usable.sum().clamp_min(1.0)
    maximum_share = share.amax(dim=-1)
    conditional_maximum = (maximum_share * usable).sum() / usable.sum().clamp_min(1.0)
    return effective.mean(), conditional_effective, conditional_maximum, usable.mean()


def factorization_probe(teacher, object_slots: int, steps: int = 80) -> dict:
    batch, points = teacher.object_confidence.shape
    generator = torch.Generator(device=teacher.object_confidence.device)
    generator.manual_seed(56017)
    logits = (
        torch.randn(
            batch,
            points,
            object_slots,
            generator=generator,
            device=teacher.object_confidence.device,
            dtype=torch.float32,
        )
        * 0.05
    )
    logits.requires_grad_(True)
    optimizer = torch.optim.Adam((logits,), lr=0.20)
    initial = None
    for _ in range(steps):
        assignment = logits.softmax(dim=-1)
        loss = _graph_loss(assignment, teacher)
        if initial is None:
            initial = loss.detach()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assignment = logits.detach().softmax(dim=-1)
    optimized = _graph_loss(assignment, teacher)
    collapsed = torch.zeros_like(assignment)
    collapsed[..., 0] = 1.0
    collapse_loss = _graph_loss(collapsed, teacher)
    effective, conditional, maximum, negative_fraction = _root_statistics(
        assignment, teacher
    )
    return {
        "factorization_initial_loss": float(initial),
        "factorization_optimized_loss": float(optimized),
        "factorization_collapse_loss": float(collapse_loss),
        "factorization_optimization_gain": float(initial - optimized),
        "factorization_collapse_margin": float(collapse_loss - optimized),
        "factorization_effective_roots": float(effective),
        "factorization_negative_supported_effective_roots": float(conditional),
        "factorization_negative_supported_maximum_root_share": float(maximum),
        "factorization_negative_supported_sample_fraction": float(negative_fraction),
    }


def teacher_batch_metrics(teacher, source_names, config) -> list[dict]:
    points = teacher.same_confidence.shape[-1]
    off_diagonal = ~torch.eye(
        points, device=teacher.same_confidence.device, dtype=torch.bool
    )[None]
    same = (teacher.same_confidence > 0.0) & off_diagonal
    different = (teacher.different_confidence > 0.0) & off_diagonal
    overlap = same & different
    rows = []
    for index, source in enumerate(source_names):
        rows.append(
            {
                "source": source,
                "same_edge_fraction": float(same[index].float().mean()),
                "different_edge_fraction": float(different[index].float().mean()),
                "known_relation_fraction": float(
                    (same[index] | different[index]).float().mean()
                ),
                "negative_track_fraction": float(
                    different[index].any(dim=-1).float().mean()
                ),
                "object_support_fraction": float(
                    (
                        teacher.object_confidence[index]
                        >= config.minimum_object_support_fraction
                    )
                    .float()
                    .mean()
                ),
                "object_support_mean": float(teacher.object_confidence[index].mean()),
                "lifecycle_known_fraction": float(
                    teacher.lifecycle_known[index].float().mean()
                ),
                "relation_overlap_fraction": float(overlap[index].float().mean()),
                "scene_target_maximum": float(teacher.scene_confidence[index].amax()),
                "transient_target_maximum": float(
                    teacher.transient_confidence[index].amax()
                ),
            }
        )
    return rows


def summarize_real_target_audit(rows: list[dict], probes: list[dict], config):
    sources = sorted({row["source"] for row in rows})
    per_source = {}
    for source in sources:
        selected = [row for row in rows if row["source"] == source]
        per_source[source] = {
            name: sum(row[name] for row in selected) / len(selected)
            for name in selected[0]
            if name != "source"
        }
    aggregate = {
        name: sum(row[name] for row in rows) / len(rows)
        for name in rows[0]
        if name != "source"
    }
    probe_mean = {
        name: sum(probe[name] for probe in probes) / len(probes) for name in probes[0]
    }
    checks = {
        "every_source_has_same_relation_evidence": all(
            value["same_edge_fraction"] >= config.minimum_same_edge_fraction
            for value in per_source.values()
        ),
        "every_source_has_different_relation_evidence": all(
            value["different_edge_fraction"] >= config.minimum_different_edge_fraction
            for value in per_source.values()
        ),
        "same_relation_evidence_is_dense_enough": (
            aggregate["same_edge_fraction"] >= config.minimum_same_edge_fraction
        ),
        "different_relation_evidence_is_dense_enough": (
            aggregate["different_edge_fraction"]
            >= config.minimum_different_edge_fraction
        ),
        "negative_tracks_are_supported": (
            aggregate["negative_track_fraction"]
            >= config.minimum_negative_track_fraction
        ),
        "positive_object_support_is_present": (
            aggregate["object_support_fraction"]
            >= config.minimum_object_support_fraction
        ),
        "object_support_is_not_all_tracks": (
            aggregate["object_support_fraction"] < 0.95
        ),
        "owner_targets_do_not_label_unknown_as_scene": (
            aggregate["scene_target_maximum"] == 0.0
        ),
        "owner_targets_do_not_invent_transient_labels": (
            aggregate["transient_target_maximum"] == 0.0
        ),
        "signed_relations_do_not_overlap": (
            aggregate["relation_overlap_fraction"] == 0.0
        ),
        "factorization_improves_the_real_target": (
            probe_mean["factorization_optimization_gain"] > 0.0
        ),
        "real_target_rejects_single_root_collapse": (
            min(probe["factorization_collapse_margin"] for probe in probes)
            >= config.minimum_real_target_collapse_margin
        ),
        "real_target_supports_multiple_roots_when_negatives_exist": (
            probe_mean["factorization_negative_supported_effective_roots"]
            >= config.minimum_factorized_effective_roots
        ),
        "real_target_avoids_dominant_root_when_negatives_exist": (
            probe_mean["factorization_negative_supported_maximum_root_share"]
            <= config.maximum_factorized_root_share
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "aggregate": aggregate,
        "per_source": per_source,
        "factorization": probe_mean,
        "probe_count": len(rows),
        "source_count": len(sources),
    }
