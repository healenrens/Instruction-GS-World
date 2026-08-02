"""Semantic parameter transforms for object_memory_v1 to v2 warm starts."""

from __future__ import annotations

import torch


def transform_v40_parameter(
    name: str,
    source: torch.Tensor,
    target: torch.Tensor,
    source_config: dict,
    target_config,
) -> tuple[torch.Tensor, str] | None:
    if (
        source_config.get("architecture") != "object_memory_v1"
        or target_config.architecture
        not in ("object_memory_v2", "object_memory_v3")
    ):
        return None
    if name in (
        "dynamics.base_lifecycle_output.weight",
        "dynamics.base_lifecycle_output.bias",
    ):
        return target.clone(), "reinitialize_factorized_lifecycle_semantics"
    if name == "dynamics.base_geometry_output.weight":
        if source.shape != target.shape or source.shape[0] != 4:
            return None
        transformed = target.clone()
        transformed[2:] = source[2:]
        return transformed, "reinitialize_xy_keep_scale_disparity"
    if name == "dynamics.base_geometry_output.bias":
        if source.shape != target.shape or source.shape[0] != 4:
            return None
        transformed = target.clone()
        transformed[2:] = source[2:]
        return transformed, "reinitialize_xy_keep_scale_disparity"
    if name in (
        "object_memory.motion_head.3.weight",
        "object_memory.motion_head.3.bias",
        "target_object_memory.motion_head.3.weight",
        "target_object_memory.motion_head.3.bias",
    ):
        if source.shape != target.shape or source.shape[0] != 4:
            return None
        transformed = target.clone()
        transformed[2:] = source[2:]
        return transformed, "reinitialize_xy_for_support_relative_transport"
    return None


def record_v40_transform(
    name: str,
    source: torch.Tensor,
    target: torch.Tensor,
    source_config: dict,
    target_config,
    compatible: dict,
    transformed: dict,
) -> bool:
    result = transform_v40_parameter(
        name,
        source,
        target,
        source_config,
        target_config,
    )
    if result is None:
        return False
    compatible[name], transform = result
    transformed[name] = {
        "source": list(source.shape),
        "target": list(target.shape),
        "transform": transform,
    }
    return True
