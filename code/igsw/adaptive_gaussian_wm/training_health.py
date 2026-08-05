"""Runtime contracts that stop training on algorithmic invariant failures."""

from __future__ import annotations

import math


CORRESPONDENCE_MASS_METRICS = frozenset(
    {
        "correspondence_row_mass_max_error",
        "correspondence_column_mass_max_error",
    }
)


def enforce_object_memory_training_health(config, metrics: dict[str, float]) -> None:
    if config.architecture not in (
        "object_memory_v3",
        "object_region_memory_v1",
        "object_region_dual_encoder_v1",
    ):
        return
    tolerance = float(config.correspondence_mass_tolerance)
    for name in CORRESPONDENCE_MASS_METRICS:
        if name not in metrics:
            continue
        value = float(metrics[name])
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite correspondence invariant: {name}")
        if value > tolerance:
            raise RuntimeError(
                f"{name}={value:.6g} exceeds configured tolerance {tolerance:.6g}"
            )
