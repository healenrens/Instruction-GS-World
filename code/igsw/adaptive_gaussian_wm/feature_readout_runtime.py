"""Build Gaussian diagnostics and the selected dense feature readout."""

from __future__ import annotations

import torch

from .decoder import feature_loss_coverage
from .readout_runtime import (
    current_background_feature,
    decode_gaussian_readout,
    residual_future_features,
)


def _dense_state(
    model,
    batch: dict[str, torch.Tensor],
    current_tokens,
    current_slots,
    predicted_slots: torch.Tensor,
    predicted_centers: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    *,
    predicted_visibility: torch.Tensor | None,
    predicted_scale: torch.Tensor | None,
):
    if model.dense_readout is None:
        raise ValueError("dense readout is disabled")
    return model.dense_readout(
        current_tokens,
        current_slots.assignment,
        current_slots.slots,
        predicted_slots,
        coordinates,
        valid,
        current_background_feature(batch),
        current_centers=current_slots.center,
        predicted_centers=predicted_centers,
        current_activity=current_slots.activity,
        predicted_visibility=predicted_visibility,
        current_scale=getattr(current_slots, "relative_scale", None),
        predicted_scale=predicted_scale,
    )


def decode_feature_readouts(
    model,
    batch: dict[str, torch.Tensor],
    current_tokens,
    current_slots,
    future_output,
    predicted_future_centers: torch.Tensor,
) -> tuple[dict, object]:
    """Return feature predictions while retaining Gaussian diagnostic states."""
    readout, context = decode_gaussian_readout(
        model,
        batch,
        current_tokens,
        current_slots,
        future_output.future_slots,
        predicted_future_centers,
        predicted_relative_scale=getattr(future_output, "future_relative_scale", None),
        predicted_relative_disparity=getattr(
            future_output, "future_relative_disparity", None
        ),
    )
    query_count = future_output.future_slots.shape[1]
    current_slots_expanded = current_slots.slots[:, None].expand(
        -1, query_count, -1, -1
    )
    current_centers_expanded = current_slots.center[:, None].expand(
        -1, query_count, -1, -1
    )
    current_readout, _ = decode_gaussian_readout(
        model,
        batch,
        current_tokens,
        current_slots,
        current_slots_expanded,
        current_centers_expanded,
        context.micro_rgb,
    )
    direct, coverage = model.gaussian_readout.splat_features(
        readout, batch["future_coordinates"]
    )
    reference, reference_coverage = model.gaussian_readout.splat_features(
        current_readout, batch["future_coordinates"]
    )
    fields = {
        "gaussian_readout": readout,
        "current_gaussian_readout": current_readout,
        "rendered_future_features": residual_future_features(direct, reference, batch),
        "residual_reference_features": reference,
        "residual_reference_coverage": reference_coverage,
        "render_coverage": coverage,
        "feature_loss_coverage": feature_loss_coverage(readout, coverage),
        "dense_future_readout": None,
        "dense_reference_readout": None,
        "current_dense_readout": None,
    }
    if model.dense_readout is None:
        return fields, context

    future_dense = _dense_state(
        model,
        batch,
        current_tokens,
        current_slots,
        future_output.future_slots,
        predicted_future_centers,
        batch["future_coordinates"],
        batch["future_valid"],
        predicted_visibility=getattr(future_output, "future_visibility", None),
        predicted_scale=getattr(future_output, "future_relative_scale", None),
    )
    reference_dense = _dense_state(
        model,
        batch,
        current_tokens,
        current_slots,
        current_slots_expanded,
        current_centers_expanded,
        batch["future_coordinates"],
        batch["future_valid"],
        predicted_visibility=current_slots.activity[:, None].expand(
            -1, query_count, -1
        ),
        predicted_scale=(
            getattr(current_slots, "relative_scale", None)[:, None].expand(
                -1, query_count, -1
            )
            if getattr(current_slots, "relative_scale", None) is not None
            else None
        ),
    )
    current_dense = _dense_state(
        model,
        batch,
        current_tokens,
        current_slots,
        current_slots.slots[:, None],
        current_slots.center[:, None],
        batch["history_coordinates"][:, -1:],
        batch["history_valid"][:, -1:],
        predicted_visibility=current_slots.activity[:, None],
        predicted_scale=(
            getattr(current_slots, "relative_scale", None)[:, None]
            if getattr(current_slots, "relative_scale", None) is not None
            else None
        ),
    )
    fields.update(
        rendered_future_features=residual_future_features(
            future_dense.feature, reference_dense.feature, batch
        ),
        residual_reference_features=reference_dense.feature,
        residual_reference_coverage=reference_dense.coverage,
        render_coverage=future_dense.coverage,
        feature_loss_coverage=torch.ones_like(future_dense.coverage),
        dense_future_readout=future_dense,
        dense_reference_readout=reference_dense,
        current_dense_readout=current_dense,
    )
    return fields, context


def decode_current_dense_readout(model, batch: dict[str, torch.Tensor], history: dict):
    """Decode the last observed frame without reading any future field."""
    if model.dense_readout is None:
        raise ValueError("current dense readout requires dense_object_readout")
    tokens = history["token_states"][-1]
    slots = history["slot_states"][-1]
    return _dense_state(
        model,
        batch,
        tokens,
        slots,
        slots.slots[:, None],
        slots.center[:, None],
        batch["history_coordinates"][:, -1:],
        batch["history_valid"][:, -1:],
        predicted_visibility=slots.activity[:, None],
        predicted_scale=(
            slots.relative_scale[:, None] if hasattr(slots, "relative_scale") else None
        ),
    )


def decode_inference_features(
    model,
    batch: dict[str, torch.Tensor],
    current_tokens,
    current_slots,
    prediction,
) -> torch.Tensor:
    predicted_centers = (
        prediction.future_centers
        if prediction.future_centers is not None
        else model.object_aggregator.decode_center(prediction.future_slots)
    )
    if model.dense_readout is not None:
        future = _dense_state(
            model,
            batch,
            current_tokens,
            current_slots,
            prediction.future_slots,
            predicted_centers,
            batch["future_coordinates"],
            batch["future_valid"],
            predicted_visibility=getattr(prediction, "future_visibility", None),
            predicted_scale=getattr(prediction, "future_relative_scale", None),
        )
        query_count = prediction.future_slots.shape[1]
        reference = _dense_state(
            model,
            batch,
            current_tokens,
            current_slots,
            current_slots.slots[:, None].expand(-1, query_count, -1, -1),
            current_slots.center[:, None].expand(-1, query_count, -1, -1),
            batch["future_coordinates"],
            batch["future_valid"],
            predicted_visibility=current_slots.activity[:, None].expand(
                -1, query_count, -1
            ),
            predicted_scale=(
                current_slots.relative_scale[:, None].expand(-1, query_count, -1)
                if hasattr(current_slots, "relative_scale")
                else None
            ),
        )
        return residual_future_features(future.feature, reference.feature, batch)
    readout, _ = decode_gaussian_readout(
        model,
        batch,
        current_tokens,
        current_slots,
        prediction.future_slots,
        predicted_centers,
        predicted_relative_scale=getattr(prediction, "future_relative_scale", None),
        predicted_relative_disparity=getattr(
            prediction, "future_relative_disparity", None
        ),
    )
    return model.gaussian_readout.splat_features(readout, batch["future_coordinates"])[
        0
    ]
