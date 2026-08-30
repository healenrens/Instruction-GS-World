"""Frozen low-complexity probes for V61 Object State representations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_split(groups):
    unique = groups.unique(sorted=True)
    test_groups = unique[::5] if len(unique) >= 5 else unique[::2]
    test = (groups[:, None] == test_groups[None]).any(dim=1)
    return ~test, test


def _limit_split(mask, limit):
    indices = mask.nonzero(as_tuple=False)[:, 0]
    if len(indices) <= limit:
        return indices
    positions = torch.linspace(0, len(indices) - 1, limit).long()
    return indices[positions]


def _standardized_features(features, train):
    mean = features[train].mean(dim=0)
    scale = features[train].std(dim=0, unbiased=False).clamp_min(1e-4)
    normalized = (features - mean) / scale
    return torch.cat((normalized, torch.ones(len(normalized), 1)), dim=1)


def _ridge_regression_gain(features, target, weight, groups, sample_limit):
    keep = weight > 0.5
    train, test = _group_split(groups)
    train = _limit_split(keep & train, sample_limit)
    test = _limit_split(keep & test, sample_limit)
    normalized = _standardized_features(features.float(), train)
    gram = normalized[train].T @ normalized[train]
    coefficient = torch.linalg.solve(
        gram + 1e-2 * torch.eye(len(gram)), normalized[train].T @ target[train].float()
    )
    prediction = normalized[test] @ coefficient
    mse = (prediction - target[test].float()).square().mean()
    baseline = (
        (target[test].float() - target[train].float().mean(dim=0))
        .square()
        .mean()
        .clamp_min(1e-8)
    )
    return float(1.0 - mse / baseline)


def _balanced_accuracy(prediction, target):
    values = []
    for label in target.unique(sorted=True):
        selected = target == label
        values.append((prediction[selected] == label).float().mean())
    return float(torch.stack(values).mean())


def _ridge_classification(features, target, weight, groups, sample_limit):
    keep = weight > 0.5
    train, test = _group_split(groups)
    train = _limit_split(keep & train, sample_limit)
    test = _limit_split(keep & test, sample_limit)
    normalized = _standardized_features(features.float(), train)
    classes = int(target.max()) + 1
    labels = F.one_hot(target.long(), classes).float()
    gram = normalized[train].T @ normalized[train]
    coefficient = torch.linalg.solve(
        gram + 1e-2 * torch.eye(len(gram)), normalized[train].T @ labels[train]
    )
    return _balanced_accuracy(
        (normalized[test] @ coefficient).argmax(dim=-1), target[test]
    )


def _mlp_probe(features, target, weight, groups, sample_limit, steps, classification):
    keep = weight > 0.5
    train_mask, test_mask = _group_split(groups)
    train = _limit_split(keep & train_mask, sample_limit)
    test = _limit_split(keep & test_mask, sample_limit)
    x = _standardized_features(features.float(), train)[:, :-1]
    output_dim = int(target.max()) + 1 if classification else target.shape[-1]
    torch.manual_seed(17)
    model = nn.Sequential(
        nn.Linear(x.shape[-1], 128), nn.GELU(), nn.Linear(128, output_dim)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    if classification:
        y = target.long()
    else:
        mean = target[train].float().mean(dim=0)
        scale = target[train].float().std(dim=0, unbiased=False).clamp_min(1e-4)
        y = (target.float() - mean) / scale
    for step in range(steps):
        start = (step * 1024) % len(train)
        batch = train[torch.arange(start, start + min(1024, len(train))) % len(train)]
        prediction = model(x[batch])
        loss = (
            F.cross_entropy(prediction, y[batch])
            if classification
            else F.mse_loss(prediction, y[batch])
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = model(x[test])
    if classification:
        return _balanced_accuracy(prediction.argmax(dim=-1), y[test])
    mse = (prediction - y[test]).square().mean()
    baseline = y[test].square().mean().clamp_min(1e-8)
    return float(1.0 - mse / baseline)


def representation_probe_metrics_v61(records, sample_limit, mlp_steps):
    identity = torch.cat(
        [record.identity.reshape(-1, record.identity.shape[-1]) for record in records]
    )
    dynamic = torch.cat(
        [record.dynamic.reshape(-1, record.dynamic.shape[-1]) for record in records]
    )
    coordinates = torch.cat([record.coordinates.reshape(-1, 2) for record in records])
    visibility = torch.cat(
        [record.visibility.reshape(-1) for record in records]
    ).float()
    known = torch.cat(
        [record.lifecycle_known.reshape(-1) for record in records]
    ).float()
    groups = torch.cat(
        [
            torch.full((record.visibility.numel(),), record.group_index)
            for record in records
        ]
    )
    sources = torch.cat(
        [
            torch.full((record.visibility.numel(),), record.source_index)
            for record in records
        ]
    )
    motion_features = torch.cat(
        [
            record.dynamic[:, :, None]
            .expand(-1, -1, record.motion.shape[2], -1)
            .reshape(-1, record.dynamic.shape[-1])
            for record in records
        ]
    )
    motion = torch.cat([record.motion.reshape(-1, 2) for record in records])
    motion_weight = torch.cat(
        [record.motion_valid.reshape(-1) for record in records]
    ).float()
    motion_groups = torch.cat(
        [
            torch.full((record.motion_valid.numel(),), record.group_index)
            for record in records
        ]
    )
    current = torch.cat(
        [record.dynamic[1:].reshape(-1, record.dynamic.shape[-1]) for record in records]
    )
    history = torch.cat(
        [
            torch.cat((record.dynamic[:-1], record.dynamic[1:]), dim=-1).reshape(
                -1, record.dynamic.shape[-1] * 2
            )
            for record in records
        ]
    )
    markov_target = torch.cat(
        [record.motion[1:, :, 0].reshape(-1, 2) for record in records]
    )
    markov_weight = torch.cat(
        [record.motion_valid[1:, :, 0].reshape(-1) for record in records]
    ).float()
    markov_groups = torch.cat(
        [
            torch.full((record.motion_valid[1:, :, 0].numel(),), record.group_index)
            for record in records
        ]
    )
    current_gain = _ridge_regression_gain(
        current, markov_target, markov_weight, markov_groups, sample_limit
    )
    history_gain = _ridge_regression_gain(
        history, markov_target, markov_weight, markov_groups, sample_limit
    )
    metrics = {
        "probe/coordinate_identity_linear_gain": _ridge_regression_gain(
            identity, coordinates, visibility, groups, sample_limit
        ),
        "probe/motion_dynamic_linear_gain": _ridge_regression_gain(
            motion_features, motion, motion_weight, motion_groups, sample_limit
        ),
        "probe/motion_dynamic_mlp_gain": _mlp_probe(
            motion_features,
            motion,
            motion_weight,
            motion_groups,
            sample_limit,
            mlp_steps,
            False,
        ),
        "probe/visibility_dynamic_linear_balanced_accuracy": _ridge_classification(
            dynamic, visibility.long(), known, groups, sample_limit
        ),
        "probe/visibility_dynamic_mlp_balanced_accuracy": _mlp_probe(
            dynamic, visibility.long(), known, groups, sample_limit, mlp_steps, True
        ),
        "nuisance/source_identity_linear_balanced_accuracy": _ridge_classification(
            identity, sources.long(), visibility, groups, sample_limit
        ),
        "nuisance/source_identity_mlp_balanced_accuracy": _mlp_probe(
            identity, sources.long(), visibility, groups, sample_limit, mlp_steps, True
        ),
        "markov/current_dynamic_motion_gain": current_gain,
        "markov/history_dynamic_motion_gain": history_gain,
        "markov/history_incremental_gain": history_gain - current_gain,
    }
    for fraction in (0.1, 0.25, 0.5):
        limit = max(256, round(sample_limit * fraction))
        metrics[f"probe/motion_dynamic_linear_gain_train_{int(fraction * 100)}pct"] = (
            _ridge_regression_gain(
                motion_features, motion, motion_weight, motion_groups, limit
            )
        )
    return metrics
