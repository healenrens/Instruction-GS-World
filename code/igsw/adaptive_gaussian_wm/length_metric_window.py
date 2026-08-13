"""Windowed metrics that preserve dynamic-history length comparisons."""

from __future__ import annotations


_PER_LENGTH_METRICS = (
    "loss_state",
    "loss_observation_complete",
    "loss_masked_observation",
    "loss_masked_state",
    "loss_track_observation_retrieval",
    "track_observation_retrieval_top1",
    "loss_lifecycle_prediction",
    "loss_presence_prediction",
    "loss_visibility_prediction",
    "loss_object_grounding",
    "diagnostic_object_gain",
    "diagnostic_motion_object_gain",
    "object_effective_count",
    "object_supported_count",
    "object_owner_fraction",
    "object_utility",
    "presence_mean",
    "visibility_mean",
    "presence_visibility_gap",
)


class LengthMetricWindow:
    def __init__(self) -> None:
        self._global_sums: dict[str, float] = {}
        self._length_sums: dict[int, dict[str, float]] = {}
        self._length_counts: dict[int, int] = {}
        self._chunk_sum = 0.0
        self._stride_sum = 0.0
        self._observation_sum = 0.0
        self._count = 0

    def add(
        self,
        metrics: dict[str, float],
        chunk_length: float,
        temporal_stride: float,
        observation_fraction: float,
        weight: int = 1,
    ) -> None:
        if weight < 1:
            raise ValueError("metric-window weight must be positive")
        length = int(round(chunk_length))
        for name, value in metrics.items():
            self._global_sums[name] = (
                self._global_sums.get(name, 0.0) + float(value) * weight
            )
        length_sums = self._length_sums.setdefault(length, {})
        for name in _PER_LENGTH_METRICS:
            if name in metrics:
                length_sums[name] = (
                    length_sums.get(name, 0.0) + float(metrics[name]) * weight
                )
        self._length_counts[length] = self._length_counts.get(length, 0) + weight
        self._chunk_sum += float(chunk_length) * weight
        self._stride_sum += float(temporal_stride) * weight
        self._observation_sum += float(observation_fraction) * weight
        self._count += weight

    def summarize_and_reset(self) -> dict[str, float]:
        if self._count == 0:
            raise ValueError("cannot summarize an empty metric window")
        summary = {
            name: total / self._count for name, total in self._global_sums.items()
        }
        summary.update({
            "chunk_length": self._chunk_sum / self._count,
            "temporal_stride": self._stride_sum / self._count,
            "observation_fraction": self._observation_sum / self._count,
            "metric_window_microbatches": float(self._count),
            "history_length_coverage": float(len(self._length_counts)),
        })
        for length, count in sorted(self._length_counts.items()):
            summary[f"history_h{length}_updates"] = float(count)
            for name, total in self._length_sums[length].items():
                summary[f"history_h{length}_{name}"] = total / count
        self.__init__()
        return summary
