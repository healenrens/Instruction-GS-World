"""Training-only query selection and held-out track targets for v57."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .trajectory_relation_teacher import TrajectoryRelationTeacher


@dataclass(frozen=True)
class QueryObjectTeacher:
    query_index: torch.Tensor
    alternate_index: torch.Tensor
    negative_index: torch.Tensor
    query_coordinate: torch.Tensor
    alternate_coordinate: torch.Tensor
    negative_coordinate: torch.Tensor
    query_valid: torch.Tensor
    alternate_valid: torch.Tensor
    negative_valid: torch.Tensor
    prompt_track_mask: torch.Tensor
    heldout_track_mask: torch.Tensor
    same_target: torch.Tensor
    different_target: torch.Tensor
    track_visibility: torch.Tensor


def _batch_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(len(values), device=values.device)
    return values[batch, indices]


def _masked_argmax(score: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    selected = score.masked_fill(~valid, torch.finfo(score.dtype).min).argmax(dim=-1)
    return selected, valid.any(dim=-1)


def build_query_object_teacher_v57(
    evidence,
    relation: TrajectoryRelationTeacher,
    config,
    current_index: int = -1,
) -> QueryObjectTeacher:
    """Select one current-frame query and reserve related tracks for supervision.

    Full-video tracks may choose which current point is useful during training. The
    deployable student receives only the selected current coordinate and observed
    RGB features. Related tracks are withheld from the prompt and used as targets.
    """

    coordinates = evidence.coordinates.float()
    visibility = evidence.visibility.bool()
    batch, frames, points = visibility.shape
    current = current_index % frames
    current_visible = visibility[:, current]
    relation_degree = relation.same_confidence.amax(dim=-1)
    query_score = relation.object_confidence.float() * relation_degree
    query_candidate = current_visible & (
        relation.object_confidence >= config.minimum_query_object_confidence
    ) & (relation_degree >= config.minimum_query_relation_confidence)
    query_index, query_valid = _masked_argmax(query_score, query_candidate)

    query_same = _batch_gather(relation.same_confidence.float(), query_index)
    query_different = _batch_gather(
        relation.different_confidence.float(), query_index
    )
    point_axis = torch.arange(points, device=coordinates.device)[None]
    not_query = point_axis != query_index[:, None]

    alternate_candidate = current_visible & not_query & (
        query_same >= config.minimum_query_relation_confidence
    )
    alternate_index, alternate_valid = _masked_argmax(
        query_same, alternate_candidate
    )
    negative_candidate = current_visible & not_query & (
        query_different >= config.minimum_query_relation_confidence
    )
    negative_index, negative_valid = _masked_argmax(
        query_different, negative_candidate
    )
    alternate_valid = alternate_valid & query_valid
    negative_valid = negative_valid & query_valid

    prompt = point_axis == query_index[:, None]
    prompt = prompt | (
        (point_axis == alternate_index[:, None]) & alternate_valid[:, None]
    )
    known_relation = (query_same > 0.0) | (query_different > 0.0)
    heldout = known_relation & ~prompt
    heldout = heldout & query_valid[:, None]

    query_coordinate = _batch_gather(coordinates[:, current], query_index)
    alternate_coordinate = _batch_gather(
        coordinates[:, current], alternate_index
    )
    negative_coordinate = _batch_gather(coordinates[:, current], negative_index)
    return QueryObjectTeacher(
        query_index=query_index,
        alternate_index=alternate_index,
        negative_index=negative_index,
        query_coordinate=query_coordinate.detach(),
        alternate_coordinate=alternate_coordinate.detach(),
        negative_coordinate=negative_coordinate.detach(),
        query_valid=query_valid.detach(),
        alternate_valid=alternate_valid.detach(),
        negative_valid=negative_valid.detach(),
        prompt_track_mask=prompt.detach(),
        heldout_track_mask=heldout.detach(),
        same_target=query_same.detach(),
        different_target=query_different.detach(),
        track_visibility=visibility.detach(),
    )


def query_teacher_contract_metrics(teacher: QueryObjectTeacher) -> dict[str, torch.Tensor]:
    heldout_positive = teacher.heldout_track_mask & (teacher.same_target > 0.0)
    heldout_negative = teacher.heldout_track_mask & (teacher.different_target > 0.0)
    return {
        "query_valid_fraction": teacher.query_valid.float().mean(),
        "alternate_valid_fraction": teacher.alternate_valid.float().mean(),
        "negative_valid_fraction": teacher.negative_valid.float().mean(),
        "heldout_positive_fraction": heldout_positive.float().mean(),
        "heldout_negative_fraction": heldout_negative.float().mean(),
        "prompt_heldout_overlap": (
            teacher.prompt_track_mask & teacher.heldout_track_mask
        ).float().sum(),
    }
