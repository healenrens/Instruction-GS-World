"""CPU tensor contract for V61 representation sufficiency evaluation."""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.representation_sufficiency_v61 import (  # noqa: E402
    SufficiencyConditionV61,
    evaluate_representation_sufficiency_v61,
)


def condition(group_index: int):
    frames, points, identity_dim = 4, 12, 8
    roots, carriers = 4, 8
    point = torch.arange(points)
    identity_base = F.normalize(
        F.one_hot(point % identity_dim, identity_dim).float()
        + 0.1 * F.one_hot((point + 1) % identity_dim, identity_dim).float(),
        dim=-1,
    )
    identity = F.normalize(
        identity_base[None].expand(frames, -1, -1)
        + torch.arange(frames)[:, None, None] * 1e-3,
        dim=-1,
    )
    time = torch.arange(frames).float()[:, None]
    x = (point.float()[None] / points) + time * (0.01 + point[None] * 0.001)
    y = (point.float()[None] % 3) / 3.0 + time * 0.002
    coordinates = torch.stack((x, y), dim=-1)
    velocity = coordinates[1:] - coordinates[:-1]
    velocity = torch.cat((velocity, velocity[-1:]), dim=0)
    dynamic = torch.cat(
        (
            velocity,
            coordinates,
            torch.sin(coordinates[..., :1]),
            torch.cos(coordinates[..., 1:2]),
        ),
        dim=-1,
    )
    root_index = point // 3
    root_assignment = F.one_hot(root_index, roots).float()[None].expand(frames, -1, -1)
    point_owner = torch.cat(
        (root_assignment * 0.95, torch.full((frames, points, 1), 0.05)), dim=-1
    )
    root_owner = torch.zeros(frames, carriers, roots + 1)
    root_owner[..., -1] = 0.05
    for carrier in range(carriers):
        root_owner[:, carrier, carrier % roots] = 0.95
    visibility = torch.ones(frames, points, dtype=torch.bool)
    visibility[1, 0] = False
    same = (root_index[:, None] == root_index[None]).float()
    different = 1.0 - same
    motion = velocity[:, :, None].expand(-1, -1, 3, -1)
    object_confidence = torch.ones(points)
    object_confidence[-3:] = 0.0
    scene_confidence = 1.0 - object_confidence
    return SufficiencyConditionV61(
        identity=identity,
        teacher_identity=identity.clone(),
        dynamic=dynamic,
        root_assignment=root_assignment,
        point_owner=point_owner,
        root_owner=root_owner,
        carrier_presence=torch.full((frames, carriers), 0.8),
        root_presence=torch.full((frames, roots), 0.75),
        visibility=visibility,
        lifecycle_known=torch.ones_like(visibility),
        presence=torch.ones(frames, points),
        coordinates=coordinates,
        motion=motion,
        motion_valid=torch.ones(frames, points, 3, dtype=torch.bool),
        same=same,
        different=different,
        object_confidence=object_confidence,
        scene_confidence=scene_confidence,
        source_index=group_index % 3,
        group_index=group_index,
        chunk_length=4,
        temporal_step_seconds=0.1,
        decode_replaced=0.0,
        parts={"track_coordinate_error": 0.01 + group_index * 1e-4},
    )


def main():
    torch.manual_seed(17)
    records = [condition(group) for group in range(10)]
    metrics = evaluate_representation_sufficiency_v61(
        records,
        variance_floor=1e-5,
        sample_limit=256,
        mlp_steps=4,
    )
    values = torch.tensor(list(metrics.values()))
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("V61 sufficiency metrics are non-finite")
    if (
        metrics["retrieval/first_last_recall_at_1"]
        <= metrics["retrieval/first_last_chance_at_1"]
    ):
        raise RuntimeError("V61 retrieval contract did not recover stable identities")
    if metrics["retrieval/reappearance_queries"] < 1.0:
        raise RuntimeError("V61 reappearance contract produced no event")
    if metrics["deletion/external_track_events"] < 1.0:
        raise RuntimeError("V61 deletion contract produced no event")
    if metrics["capacity/identity_effective_rank"] <= 1.0:
        raise RuntimeError("V61 identity rank contract collapsed")
    required = (
        "probe/motion_dynamic_linear_gain",
        "probe/motion_dynamic_mlp_gain",
        "nuisance/source_identity_linear_balanced_accuracy",
        "markov/history_incremental_gain",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        raise RuntimeError(f"V61 sufficiency metrics are missing: {missing}")
    print(
        {
            "status": "passed",
            "metric_count": len(metrics),
            "first_last_recall_at_1": metrics["retrieval/first_last_recall_at_1"],
            "reappearance_queries": metrics["retrieval/reappearance_queries"],
            "identity_effective_rank": metrics["capacity/identity_effective_rank"],
        }
    )


if __name__ == "__main__":
    main()
