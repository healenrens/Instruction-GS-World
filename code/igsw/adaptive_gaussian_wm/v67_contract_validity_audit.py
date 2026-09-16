"""Evidence utilities for auditing v67 names, proxies, and teacher contracts."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


CLAIM_REGISTRY_V67 = {
    "relation_probability_is_object_membership": {
        "implementation": "sigmoid of a query-candidate pair MLP trained against the teacher relation proxy",
        "required_evidence": "independent same-object labels and localized deletion/intervention",
        "currently_allowed": "student relation-proxy probability",
        "not_established_by": ["teacher BCE", "symmetry", "transitivity"],
    },
    "gaussian_code_is_object_state": {
        "implementation": "320-D Gaussian code produced from query, relation-weighted features, response, and support mass",
        "required_evidence": "independent object correspondence and localized component intervention",
        "currently_allowed": "stochastic query-conditioned compressed code",
        "not_established_by": ["KL rate", "code standard deviation", "feature reconstruction"],
    },
    "identity_partition_has_object_identity_semantics": {
        "implementation": "L2-normalized first 128 dimensions of the 320-D code",
        "required_evidence": "same physical object retrieval against independently labelled positives and negatives",
        "currently_allowed": "identity-named latent partition",
        "not_established_by": ["same query cosine across time", "fixed query-index retrieval"],
    },
    "dynamic_partition_has_motion_semantics": {
        "implementation": "remaining 192 dimensions of the same 320-D code",
        "required_evidence": "held-group motion probe where this partition carries transition information distinctly better than the identity partition",
        "currently_allowed": "dynamic-named latent partition",
        "not_established_by": ["nonzero variance", "decoder response variance"],
    },
    "response_feature_has_dynamic_semantics": {
        "implementation": "192-D relation-pair MLP feature plus a projected candidate feature; the decoder is trained to reproduce its stop-gradient value",
        "required_evidence": "held transition probe and intervention showing information beyond position, query index, and appearance",
        "currently_allowed": "learned relation-conditioned response feature",
        "not_established_by": ["response reconstruction loss", "response standard deviation"],
    },
    "teacher_relation_means_same_object": {
        "implementation": "geometric mean of DINO affinity, SigLIP affinity, and motion coherence",
        "required_evidence": "independent same-object labels on query-candidate pairs",
        "currently_allowed": "teacher relation proxy",
        "not_established_by": ["query-shuffle margin", "agreement with its student"],
    },
    "teacher_visibility_means_visual_visibility": {
        "implementation": "CoTracker visibility intersected with in-bounds and local DINO/SigLIP validity",
        "required_evidence": "human or independent visual visibility labels",
        "currently_allowed": "teacher visibility proxy",
        "not_established_by": ["relay agreement", "in-bounds status"],
    },
    "student_visibility_logit_means_visual_visibility": {
        "implementation": "decoder and pair-field logits trained against the composed teacher visibility proxy",
        "required_evidence": "independent visible/occluded/out-of-frame labels",
        "currently_allowed": "student teacher-visibility prediction",
        "not_established_by": ["teacher visibility BCE"],
    },
    "teacher_reliability_means_correctness": {
        "implementation": "relay consistency multiplied by temporal appearance stability and thresholded",
        "required_evidence": "calibration against independent correspondence correctness labels",
        "currently_allowed": "teacher confidence heuristic",
        "not_established_by": ["low relay error", "high appearance cosine"],
    },
    "shared_code_cost_saving_proves_object_compression": {
        "implementation": "teacher-relation-weighted semantic distortion plus 0.005 times KL; separate rate is a weighted sum of per-point KL normalized by maximum support weight",
        "required_evidence": "per-query distortion-rate frontier under matched fidelity plus independent object grouping evidence",
        "currently_allowed": "objective-specific shared-versus-separate cost proxy",
        "not_established_by": ["mean positive saving", "fraction of queries with positive proxy saving"],
    },
    "decoded_dino_siglip_features_are_object_semantics": {
        "implementation": "continuous decoder outputs aligned to frozen local DINO and SigLIP projections",
        "required_evidence": "independent object/property probes and object-local intervention",
        "currently_allowed": "teacher-feature reconstruction at queried coordinates",
        "not_established_by": ["low DINO/SigLIP cosine error"],
    },
}


def distribution_summary_v67(values: torch.Tensor) -> dict[str, float | int | None]:
    flat = values.detach().float().flatten().cpu()
    flat = flat[torch.isfinite(flat)]
    if not len(flat):
        return {"count": 0, "mean": None, "std": None}
    quantile_levels = torch.tensor(
        (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99), dtype=torch.float32
    )
    quantiles = torch.quantile(flat, quantile_levels)
    result: dict[str, float | int | None] = {
        "count": int(len(flat)),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }
    for name, value in zip(
        ("q01", "q05", "q25", "q50", "q75", "q95", "q99"), quantiles
    ):
        result[name] = float(value)
    return result


def effective_rank_v67(features: torch.Tensor, maximum_vectors: int = 8192) -> dict:
    flat = features.detach().float().flatten(0, -2).cpu()
    if len(flat) > maximum_vectors:
        indices = torch.linspace(0, len(flat) - 1, maximum_vectors).long()
        flat = flat.index_select(0, indices)
    centered = flat - flat.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    total_energy = energy.sum()
    probability = energy / total_energy.clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    effective_rank = torch.where(
        total_energy > 1e-12, entropy.exp(), torch.zeros_like(entropy)
    )
    dimension_std = flat.std(dim=0, unbiased=False)
    available_rank = min(max(len(flat) - 1, 1), flat.shape[1])
    return {
        "vectors_used": int(len(flat)),
        "dimension": int(flat.shape[1]),
        "effective_rank": float(effective_rank),
        "effective_rank_fraction": float(effective_rank / available_rank),
        "dimension_std": distribution_summary_v67(dimension_std),
        "fraction_dimension_std_below_0_05": float((dimension_std < 0.05).float().mean()),
        "interpretation": "numerical anti-collapse health only; it does not establish semantics",
    }


def _split_by_group_v67(group_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    group_ids = group_ids.detach().long().flatten().cpu()
    test = group_ids.remainder(5) == 0
    return ~test, test


def cross_fitted_ridge_probe_v67(
    features: torch.Tensor,
    target: torch.Tensor,
    group_ids: torch.Tensor,
    ridge_values: tuple[float, ...] = (1e-4, 1e-2, 1.0),
) -> dict:
    features = features.detach().float().flatten(0, -2).cpu()
    target = target.detach().float().flatten(0, -2).cpu()
    train, test = _split_by_group_v67(group_ids)
    if len(group_ids) != len(features):
        raise ValueError("probe group count differs from feature count")
    train_count, test_count = int(train.sum()), int(test.sum())
    if train_count < features.shape[1] + 1 or test_count < 2:
        return {
            "status": "insufficient_cross_fit_samples",
            "train_count": train_count,
            "test_count": test_count,
        }
    x_train, x_test = features[train], features[test]
    y_train, y_test = target[train], target[test]
    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    x_train = torch.cat((x_train, torch.ones(train_count, 1)), dim=-1)
    x_test = torch.cat((x_test, torch.ones(test_count, 1)), dim=-1)
    target_mean = y_train.mean(dim=0, keepdim=True)
    baseline_mse = (y_test - target_mean).square().mean()
    gram = x_train.T @ x_train
    rhs = x_train.T @ y_train
    identity = torch.eye(gram.shape[0])
    identity[-1, -1] = 0.0
    results = {}
    for ridge in ridge_values:
        weights = torch.linalg.solve(gram + ridge * identity, rhs)
        prediction = x_test @ weights
        error = prediction - y_test
        mse = error.square().mean()
        mae = error.abs().mean()
        results[f"ridge_{ridge:g}"] = {
            "mse": float(mse),
            "mae": float(mae),
            "baseline_mse": float(baseline_mse),
            "r2_against_train_mean": float(1.0 - mse / baseline_mse.clamp_min(1e-12)),
        }
    return {
        "status": "completed",
        "split": "task_group_id_mod_5; fold_0 held out",
        "train_count": train_count,
        "test_count": test_count,
        "results": results,
    }


def query_index_centroid_probe_v67(
    features: torch.Tensor,
    query_indices: torch.Tensor,
    group_ids: torch.Tensor,
) -> dict:
    features = F.normalize(features.detach().float().flatten(0, -2).cpu(), dim=-1)
    query_indices = query_indices.detach().long().flatten().cpu()
    train, test = _split_by_group_v67(group_ids)
    classes = int(query_indices.max()) + 1
    centroids = []
    class_valid = []
    for index in range(classes):
        mask = train & (query_indices == index)
        class_valid.append(bool(mask.any()))
        centroid = features[mask].mean(dim=0) if mask.any() else torch.zeros(features.shape[1])
        centroids.append(centroid)
    centroids = F.normalize(torch.stack(centroids), dim=-1)
    valid_classes = torch.tensor(class_valid)
    logits = features[test] @ centroids.T
    logits[:, ~valid_classes] = -1e9
    prediction = logits.argmax(dim=-1)
    labels = query_indices[test]
    return {
        "status": "completed",
        "test_count": int(test.sum()),
        "class_count": classes,
        "top1_accuracy": float((prediction == labels).float().mean()),
        "chance_accuracy": 1.0 / classes,
        "interpretation": "high accuracy indicates fixed query/location leakage, not object identity",
    }


def temporal_query_retrieval_v67(
    source: torch.Tensor,
    future: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    source = F.normalize(source.float(), dim=-1)
    future = F.normalize(future.float(), dim=-1)
    similarity = torch.einsum("bqd,bkd->bqk", source, future)
    pair_valid = valid[:, :, None] & valid[:, None, :]
    similarity = similarity.masked_fill(~pair_valid, -2.0)
    query_count = similarity.shape[1]
    diagonal = similarity.diagonal(dim1=1, dim2=2)
    eye = torch.eye(query_count, device=similarity.device, dtype=torch.bool)[None]
    hardest_negative = similarity.masked_fill(eye, -2.0).amax(dim=-1)
    order = similarity.argsort(dim=-1, descending=True)
    labels = torch.arange(query_count, device=similarity.device)[None, :, None]
    rank = (order == labels).float().argmax(dim=-1) + 1
    top1 = order[..., 0] == labels[..., 0]
    valid_float = valid.float()
    return {
        "same_query_cosine": diagonal,
        "same_query_cosine_error": 1.0 - diagonal,
        "hardest_negative_cosine": hardest_negative,
        "retrieval_margin": diagonal - hardest_negative,
        "top1": top1.float(),
        "reciprocal_rank": rank.float().reciprocal(),
        "valid": valid_float,
    }


def binary_annotation_metrics_v67(
    probability: torch.Tensor,
    label: torch.Tensor,
) -> dict:
    probability = probability.detach().float().flatten().cpu().clamp(0.0, 1.0)
    label = label.detach().bool().flatten().cpu()
    prediction = probability >= 0.5
    true_positive = (prediction & label).sum().float()
    false_positive = (prediction & ~label).sum().float()
    false_negative = (~prediction & label).sum().float()
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
    positives = probability[label]
    negatives = probability[~label]
    if len(positives) and len(negatives):
        comparisons = positives[:, None] - negatives[None]
        auroc = (comparisons.gt(0).float() + 0.5 * comparisons.eq(0).float()).mean()
        auroc_value: float | None = float(auroc)
    else:
        auroc_value = None
    return {
        "count": int(len(label)),
        "positive_count": int(label.sum()),
        "negative_count": int((~label).sum()),
        "accuracy_at_0_5": float((prediction == label).float().mean()),
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(2.0 * precision * recall / (precision + recall).clamp_min(1e-12)),
        "brier": float((probability - label.float()).square().mean()),
        "auroc": auroc_value,
    }


def relation_entropy_v67(relation: torch.Tensor) -> torch.Tensor:
    probability = relation.float().clamp(1e-6, 1.0 - 1e-6)
    return -(probability * probability.log() + (1.0 - probability) * (1.0 - probability).log())


def weighted_concentration_v67(weight: torch.Tensor, fraction: float = 0.01) -> torch.Tensor:
    flat = weight.float().flatten(1)
    count = max(1, math.ceil(flat.shape[1] * fraction))
    top = flat.topk(count, dim=-1).values.sum(dim=-1)
    return top / flat.sum(dim=-1).clamp_min(1e-12)
