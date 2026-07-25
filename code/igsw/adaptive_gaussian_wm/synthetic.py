"""Controlled feature-video scenes for structural feasibility experiments."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _coordinate_grid(
    grid_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, grid_size, device=device, dtype=dtype)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def _render_scene(
    positions: torch.Tensor,
    object_features: torch.Tensor,
    object_count: torch.Tensor,
    coordinates: torch.Tensor,
    texture_scale: torch.Tensor,
    texture_frequency: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, count, _ = positions.shape
    grid_count = coordinates.shape[0]
    feature_dim = object_features.shape[-1]
    features = positions.new_zeros(batch, grid_count, feature_dim)
    labels = torch.full(
        (batch, grid_count),
        -1,
        device=positions.device,
        dtype=torch.long,
    )
    strongest = positions.new_zeros(batch, grid_count)
    coordinate_x = coordinates[:, 0][None]
    coordinate_y = coordinates[:, 1][None]
    texture = (
        torch.sin((2.0 + texture_frequency[:, None]) * coordinate_x)
        * torch.cos((3.0 + texture_frequency[:, None]) * coordinate_y)
        * texture_scale[:, None]
    )
    features[..., 0] = 0.1 * texture
    for index in range(count):
        valid_object = (index < object_count)[:, None]
        difference = coordinates[None] - positions[:, index, None]
        width = 0.15 + 0.03 * (index % 3)
        weight = torch.exp(-difference.square().sum(dim=-1) / (2.0 * width**2))
        weight = weight * valid_object
        features = features + weight[..., None] * object_features[:, index, None]
        replace = (weight > strongest) & (weight > 0.12)
        labels = torch.where(replace, index, labels)
        strongest = torch.maximum(strongest, weight)
    return features, labels


def make_oracle_mode_actions(
    batch: dict[str, torch.Tensor],
    action_tokens: int,
    action_dim: int,
) -> torch.Tensor:
    actions = batch["future_times"].new_zeros(
        batch["future_times"].shape[0],
        batch["future_times"].shape[1],
        action_tokens,
        action_dim,
    )
    actions[:, :, 0, 0] = batch["branch_mode"][:, None]
    return actions


def make_synthetic_batch(
    feature_dim: int,
    batch_size: int,
    history_frames: int,
    future_steps: int,
    grid_size: int,
    device: torch.device,
    paired_futures: bool = True,
    irregular_gaps: bool = True,
    mode_count: int = 2,
    max_objects: int = 3,
    ambiguous_fraction: float = 1.0,
    balanced_ambiguity: bool = False,
    semantic_branch_strength: float = 0.0,
) -> dict[str, torch.Tensor]:
    if history_frames not in (1, 3):
        raise ValueError("history_frames must be 1 or 3")
    if future_steps <= 0 or grid_size <= 1 or batch_size <= 0:
        raise ValueError("future_steps, grid_size, and batch_size must be positive")
    if feature_dim < 2:
        raise ValueError("feature_dim must be at least two")
    if mode_count <= 0 or max_objects <= 0:
        raise ValueError("mode_count and max_objects must be positive")
    if not 0.0 <= ambiguous_fraction <= 1.0:
        raise ValueError("ambiguous_fraction must be in [0, 1]")
    if semantic_branch_strength < 0.0:
        raise ValueError("semantic_branch_strength must be non-negative")
    repeat = mode_count if paired_futures else 1
    base_batch = (batch_size + repeat - 1) // repeat
    dtype = torch.float32
    coordinates = _coordinate_grid(grid_size, device, dtype)
    object_count_base = torch.randint(
        1,
        max_objects + 1,
        (base_batch,),
        device=device,
    )
    current_base = torch.rand(base_batch, max_objects, 2, device=device) - 0.5
    history_velocity_base = 0.08 * torch.randn(
        base_batch,
        max_objects,
        2,
        device=device,
    )
    object_features_base = F.normalize(
        torch.randn(base_batch, max_objects, feature_dim, device=device),
        dim=-1,
    )
    branch_direction = (
        current_base + 0.35 * object_features_base[..., :2]
    )
    branch_velocity_base = 0.30 * F.normalize(
        branch_direction,
        dim=-1,
    )
    texture_base = 0.2 + 0.8 * torch.rand(base_batch, device=device)
    texture_frequency_base = torch.randint(
        1,
        5,
        (base_batch,),
        device=device,
    ).to(dtype)
    if balanced_ambiguity:
        ambiguity_base = (
            torch.arange(base_batch, device=device) % 2 == 0
        ).to(dtype)
    else:
        ambiguity_base = (
            torch.rand(base_batch, device=device) < ambiguous_fraction
        ).to(dtype)
    gap_factor_base = (
        0.65 + 0.7 * torch.rand(base_batch, device=device)
        if irregular_gaps
        else torch.ones(base_batch, device=device)
    )

    object_count = object_count_base.repeat_interleave(repeat)[:batch_size]
    current = current_base.repeat_interleave(repeat, dim=0)[:batch_size]
    history_velocity = history_velocity_base.repeat_interleave(
        repeat,
        dim=0,
    )[:batch_size]
    branch_velocity = branch_velocity_base.repeat_interleave(
        repeat,
        dim=0,
    )[:batch_size]
    object_features = object_features_base.repeat_interleave(
        repeat,
        dim=0,
    )[:batch_size]
    semantic_direction = torch.roll(object_features, shifts=1, dims=-1)
    semantic_direction = semantic_direction - (
        semantic_direction * object_features
    ).sum(dim=-1, keepdim=True) * object_features
    semantic_direction = F.normalize(semantic_direction, dim=-1)
    texture_scale = texture_base.repeat_interleave(repeat)[:batch_size]
    texture_frequency = texture_frequency_base.repeat_interleave(repeat)[
        :batch_size
    ]
    ambiguity = ambiguity_base.repeat_interleave(repeat)[:batch_size]
    gap_factor = gap_factor_base.repeat_interleave(repeat)[:batch_size]
    if paired_futures:
        mode_values = torch.linspace(
            -1.0,
            1.0,
            mode_count,
            device=device,
        )
        mode = mode_values[
            torch.arange(batch_size, device=device) % mode_count
        ]
        mode = mode * ambiguity
    else:
        mode_values = torch.linspace(
            -1.0,
            1.0,
            mode_count,
            device=device,
        )
        mode = mode_values[
            torch.randint(mode_count, (batch_size,), device=device)
        ]
        mode = mode * ambiguity

    if history_frames == 1:
        history_times = torch.zeros(batch_size, 1, device=device)
    else:
        history_times = torch.tensor(
            [-2.0, -1.0, 0.0],
            device=device,
        )[None] * gap_factor[:, None]
    future_base = torch.linspace(
        0.5,
        1.5,
        future_steps,
        device=device,
    )
    future_times = future_base[None] * gap_factor[:, None]

    history_features = []
    history_labels = []
    for index in range(history_frames):
        time = history_times[:, index, None, None]
        position = current + time * history_velocity
        feature, label = _render_scene(
            position,
            object_features,
            object_count,
            coordinates,
            texture_scale,
            texture_frequency,
        )
        feature[..., -1] = (
            feature[..., -1] + 0.15 * (2.0 * ambiguity[:, None] - 1.0)
        )
        history_features.append(feature)
        history_labels.append(label)

    future_features = []
    future_labels = []
    future_velocity = history_velocity + mode[:, None, None] * branch_velocity
    for index in range(future_steps):
        time = future_times[:, index, None, None]
        position = current + time * future_velocity
        position = position.clamp(-0.9, 0.9)
        semantic_scale = (
            semantic_branch_strength
            * float(future_base[index] / future_base[-1])
            * mode[:, None, None]
        )
        future_object_features = F.normalize(
            object_features + semantic_scale * semantic_direction,
            dim=-1,
        )
        feature, label = _render_scene(
            position,
            future_object_features,
            object_count,
            coordinates,
            texture_scale,
            texture_frequency,
        )
        feature[..., -1] = (
            feature[..., -1] + 0.15 * (2.0 * ambiguity[:, None] - 1.0)
        )
        future_features.append(feature)
        future_labels.append(label)

    history_features_tensor = torch.stack(history_features, dim=1)
    future_features_tensor = torch.stack(future_features, dim=1)
    history_coordinates = coordinates[None, None].expand(
        batch_size,
        history_frames,
        -1,
        -1,
    )
    future_coordinates = coordinates[None, None].expand(
        batch_size,
        future_steps,
        -1,
        -1,
    )
    return {
        "history_features": history_features_tensor,
        "history_coordinates": history_coordinates,
        "history_valid": torch.ones(
            history_features_tensor.shape[:3],
            device=device,
            dtype=torch.bool,
        ),
        "history_times": history_times,
        "future_features": future_features_tensor,
        "future_coordinates": future_coordinates,
        "future_valid": torch.ones(
            future_features_tensor.shape[:3],
            device=device,
            dtype=torch.bool,
        ),
        "future_times": future_times,
        "history_labels": torch.stack(history_labels, dim=1),
        "future_labels": torch.stack(future_labels, dim=1),
        "complexity": (
            object_count.to(dtype)
            + texture_scale * texture_frequency
        ),
        "object_count": object_count,
        "texture_frequency": texture_frequency,
        "ambiguity": ambiguity,
        "group_id": torch.arange(base_batch, device=device).repeat_interleave(
            repeat
        )[:batch_size],
        "branch_mode": mode,
    }
