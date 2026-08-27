"""Observation-grounded reconstruction diagnostics for compact V60 object states."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


ROUTES = (
    "actual_support_oracle",
    "teacher_state_oracle",
    "correct",
    "base",
    "shuffled",
    "persistence",
)


@dataclass(frozen=True)
class ObjectObservationTarget:
    source_patches: torch.Tensor
    future_patches: torch.Tensor
    future_valid: torch.Tensor
    relative_patch_coordinates: torch.Tensor
    future_track_features: torch.Tensor
    future_track_weights: torch.Tensor
    actual_support: torch.Tensor
    pair_valid: torch.Tensor


@dataclass(frozen=True)
class ObservationRoute:
    semantic: torch.Tensor
    geometry: torch.Tensor
    visibility: torch.Tensor
    support: torch.Tensor


def _query_membership(evidence, relation, binding) -> torch.Tensor:
    batch = torch.arange(len(binding.query_index), device=binding.query_index.device)
    same = relation.same_confidence.float()[batch, binding.query_index]
    query = F.one_hot(binding.query_index, evidence.coordinates.shape[2]).float()
    return torch.maximum(same, query) * binding.query_valid[:, None].float()


def _scene_centers(evidence) -> torch.Tensor:
    visible = evidence.visibility.float()
    weight = visible / visible.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.einsum("btp,btpd->btd", weight, evidence.coordinates.float())


def _track_support(
    tracks: torch.Tensor,
    weights: torch.Tensor,
    patch_coordinates: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    difference = tracks[:, :, :, None] - patch_coordinates[:, :, None]
    kernel = torch.exp(-difference.square().sum(dim=-1) / (2.0 * sigma**2))
    return (kernel * weights[..., None]).amax(dim=2)


def build_object_observation_target_v60(
    features,
    evidence,
    relation,
    binding,
    transition_target,
    observed_frames: int,
    dynamic_horizons: tuple[int, ...],
    support_sigma: float,
) -> ObjectObservationTarget:
    source_index = observed_frames - 1
    horizons = torch.tensor(
        dynamic_horizons, device=features.patches.device, dtype=torch.long
    )
    future_indices = source_index + horizons
    membership = _query_membership(evidence, relation, binding)
    future_visibility = evidence.visibility[:, future_indices].float()
    track_weights = future_visibility * membership[:, None]
    scene_center = _scene_centers(evidence)[:, future_indices]
    patch_coordinates = features.coordinates[:, future_indices].float()
    relative_patch_coordinates = patch_coordinates - scene_center[:, :, None]
    relative_tracks = (
        evidence.coordinates[:, future_indices].float() - scene_center[:, :, None]
    )
    actual_support = _track_support(
        relative_tracks,
        track_weights,
        relative_patch_coordinates,
        support_sigma,
    )
    return ObjectObservationTarget(
        source_patches=features.patches[:, source_index].float(),
        future_patches=features.patches[:, future_indices].float(),
        future_valid=features.valid[:, future_indices].bool(),
        relative_patch_coordinates=relative_patch_coordinates,
        future_track_features=evidence.sampled_features[:, future_indices].float(),
        future_track_weights=track_weights,
        actual_support=actual_support,
        pair_valid=transition_target.pair_valid.bool(),
    )


def _moment_support(
    geometry: torch.Tensor,
    visibility: torch.Tensor,
    coordinates: torch.Tensor,
    support_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    center = geometry[..., :2].float()
    xx_raw, yy_raw, xy_raw = (geometry[..., index].float() for index in (2, 3, 4))
    psd = (xx_raw > 0.0) & (yy_raw > 0.0) & (xx_raw * yy_raw - xy_raw.square() > 0.0)
    xx = xx_raw.clamp_min(1e-5) + support_sigma**2
    yy = yy_raw.clamp_min(1e-5) + support_sigma**2
    xy_limit = 0.99 * (xx * yy).sqrt()
    xy = torch.maximum(torch.minimum(xy_raw, xy_limit), -xy_limit)
    determinant = (xx * yy - xy.square()).clamp_min(1e-8)
    offset = coordinates - center[:, :, None]
    mahalanobis = (
        yy[:, :, None] * offset[..., 0].square()
        - 2.0 * xy[:, :, None] * offset[..., 0] * offset[..., 1]
        + xx[:, :, None] * offset[..., 1].square()
    ) / determinant[:, :, None]
    support = torch.exp(-0.5 * mahalanobis.clamp_min(0.0))
    return support * visibility.float()[:, :, None], psd


def _route(
    semantic: torch.Tensor,
    geometry: torch.Tensor,
    visibility: torch.Tensor,
    observation: ObjectObservationTarget,
    support_sigma: float,
) -> tuple[ObservationRoute, torch.Tensor]:
    support, psd = _moment_support(
        geometry,
        visibility,
        observation.relative_patch_coordinates,
        support_sigma,
    )
    return ObservationRoute(
        semantic=F.normalize(semantic.float(), dim=-1, eps=1e-6),
        geometry=geometry.float(),
        visibility=visibility.float(),
        support=support,
    ), psd


def observation_routes_v60(
    output,
    target,
    observation: ObjectObservationTarget,
    support_sigma: float,
) -> tuple[dict[str, ObservationRoute], dict[str, torch.Tensor]]:
    horizons = target.future_semantic.shape[1]
    persistence_semantic = target.source_semantic[:, None].expand_as(
        target.future_semantic
    )
    persistence_geometry = target.source_geometry[:, None].expand_as(
        target.future_geometry
    )
    persistence_visibility = target.source_visibility[:, None].expand_as(
        target.future_visibility
    )
    values = {
        "teacher_state_oracle": (
            target.future_semantic,
            target.future_geometry,
            target.future_visibility,
        ),
        "correct": (
            output["correct"].semantic,
            output["correct"].geometry,
            output["correct"].visibility_logits.sigmoid(),
        ),
        "base": (
            output["base"].semantic,
            output["base"].geometry,
            output["base"].visibility_logits.sigmoid(),
        ),
        "shuffled": (
            output["shuffled"].semantic,
            output["shuffled"].geometry,
            output["shuffled"].visibility_logits.sigmoid(),
        ),
        "persistence": (
            persistence_semantic,
            persistence_geometry,
            persistence_visibility,
        ),
    }
    routes, psd = {}, {}
    for name, value in values.items():
        routes[name], psd[name] = _route(*value, observation, support_sigma)
    routes["actual_support_oracle"] = ObservationRoute(
        semantic=F.normalize(target.future_semantic.float(), dim=-1, eps=1e-6),
        geometry=target.future_geometry.float(),
        visibility=target.future_visibility.float(),
        support=observation.actual_support,
    )
    psd["actual_support_oracle"] = torch.ones(
        target.pair_valid.shape, device=target.pair_valid.device, dtype=torch.bool
    )
    if any(
        route.semantic.shape[:2] != (len(target.pair_valid), horizons)
        for route in routes.values()
    ):
        raise ValueError("v60 observation route shape differs from transition target")
    return routes, psd


def _sum_and_weight(value: torch.Tensor, weight: torch.Tensor):
    return (value * weight).sum(), weight.sum()


def _composite(route: ObservationRoute, observation: ObjectObservationTarget):
    source = observation.source_patches[:, None].expand_as(observation.future_patches)
    support = route.support[..., None].clamp(0.0, 1.0)
    semantic = route.semantic[:, :, None].expand_as(observation.future_patches)
    return F.normalize(source * (1.0 - support) + semantic * support, dim=-1, eps=1e-6)


def observation_reconstruction_statistics_v60(
    output,
    target,
    observation: ObjectObservationTarget,
    support_sigma: float,
    sample_mask: torch.Tensor | None = None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    routes, psd = observation_routes_v60(output, target, observation, support_sigma)
    pair = observation.pair_valid
    if sample_mask is not None:
        pair = pair & sample_mask.bool()[:, None]
    pair = pair.float()
    valid = observation.future_valid.float() * pair[..., None]
    object_weight = valid * observation.actual_support
    background_weight = valid * (1.0 - observation.actual_support)
    track_weight = observation.future_track_weights * pair[..., None]
    statistics = {}
    for name in ROUTES:
        route = routes[name]
        semantic_patch_error = 1.0 - F.cosine_similarity(
            route.semantic[:, :, None], observation.future_patches, dim=-1
        )
        semantic_track_error = 1.0 - F.cosine_similarity(
            route.semantic[:, :, None], observation.future_track_features, dim=-1
        )
        reconstruction = _composite(route, observation)
        reconstruction_error = 1.0 - F.cosine_similarity(
            reconstruction, observation.future_patches, dim=-1
        )
        intersection = (
            torch.minimum(route.support, observation.actual_support) * valid
        ).sum(dim=-1)
        union = (torch.maximum(route.support, observation.actual_support) * valid).sum(
            dim=-1
        )
        support_iou = intersection / union.clamp_min(1e-6)
        prefix = f"{name}_"
        statistics[prefix + "dense_object_semantic_error"] = _sum_and_weight(
            semantic_patch_error, object_weight
        )
        statistics[prefix + "track_semantic_error"] = _sum_and_weight(
            semantic_track_error, track_weight
        )
        statistics[prefix + "composite_full_error"] = _sum_and_weight(
            reconstruction_error, valid
        )
        statistics[prefix + "composite_object_error"] = _sum_and_weight(
            reconstruction_error, object_weight
        )
        statistics[prefix + "composite_background_error"] = _sum_and_weight(
            reconstruction_error, background_weight
        )
        statistics[prefix + "support_iou"] = _sum_and_weight(support_iou, pair)
        statistics[prefix + "covariance_psd_fraction"] = _sum_and_weight(
            psd[name].float(), pair
        )
    return statistics


class ObjectObservationAccumulator:
    def __init__(self):
        self.sums: dict[str, float] = {}
        self.weights: dict[str, float] = {}

    def update(self, statistics):
        for name, (value, weight) in statistics.items():
            self.sums[name] = self.sums.get(name, 0.0) + float(value)
            self.weights[name] = self.weights.get(name, 0.0) + float(weight)

    def finalize(self) -> dict[str, float]:
        metrics = {
            name: total / max(self.weights[name], 1e-6)
            for name, total in self.sums.items()
        }
        direct_oracle = metrics["actual_support_oracle_composite_object_error"]
        oracle_object = metrics["teacher_state_oracle_composite_object_error"]
        correct_object = metrics["correct_composite_object_error"]
        persistence_object = metrics["persistence_composite_object_error"]
        metrics.update(
            {
                "semantic_compression_floor": metrics[
                    "actual_support_oracle_dense_object_semantic_error"
                ],
                "teacher_moment_support_iou": metrics[
                    "teacher_state_oracle_support_iou"
                ],
                "direct_observation_floor": direct_oracle,
                "teacher_state_observation_error": oracle_object,
                "geometry_state_penalty": oracle_object - direct_oracle,
                "dynamics_observation_gap": correct_object - oracle_object,
                "prediction_gain_over_persistence": (
                    persistence_object - correct_object
                )
                / max(persistence_object, 1e-6),
                "teacher_state_gain_over_persistence": (
                    persistence_object - oracle_object
                )
                / max(persistence_object, 1e-6),
            }
        )
        return metrics


def macro_observation_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: sum(record[name] for record in records) / len(records)
        for name in records[0]
    }
