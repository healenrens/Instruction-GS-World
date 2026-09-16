#!/usr/bin/env python3
"""Held-data audit of v67 latent names and teacher proxy contracts."""

from __future__ import annotations

import argparse
import json
import os
import sys

from PIL import Image, ImageDraw
import torch
import torch.distributed as dist
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.continuous_predictive_object_field_v67 import (  # noqa: E402
    ContinuousPredictiveObjectFieldV67,
)
from igsw.adaptive_gaussian_wm.continuous_predictive_teacher_v67 import (  # noqa: E402
    ContinuousPredictiveTeacherRuntimeV67,
    teacher_relation_components_v67,
)
from igsw.adaptive_gaussian_wm.distributed_audit_v62 import (  # noqa: E402
    finish_distributed_audit_v62,
    initialize_distributed_audit_v62,
    shard_indices_v62,
)
from igsw.adaptive_gaussian_wm.multisource_point_track_dataset import (  # noqa: E402
    MultiSourcePointTrackObjectVideoDataset,
)
from igsw.adaptive_gaussian_wm.native_video_batch_v65 import (  # noqa: E402
    collate_native_video_batch_v65,
)
from igsw.adaptive_gaussian_wm.train_runtime import move_to_device  # noqa: E402
from igsw.adaptive_gaussian_wm.v67_config import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINT_VERSION,
    STATE_STAGE,
    ContinuousPredictiveObjectFieldConfigV67,
)
from igsw.adaptive_gaussian_wm.v67_contract_validity_audit import (  # noqa: E402
    CLAIM_REGISTRY_V67,
    binary_annotation_metrics_v67,
    cross_fitted_ridge_probe_v67,
    distribution_summary_v67,
    effective_rank_v67,
    query_index_centroid_probe_v67,
    relation_entropy_v67,
    temporal_query_retrieval_v67,
    weighted_concentration_v67,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_index", required=True)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--siglip_checkpoint", required=True)
    parser.add_argument("--tracker_checkpoint", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--annotation_jsonl", default="")
    parser.add_argument("--held_group_stride", type=int, default=20)
    parser.add_argument("--items_per_source", type=int, default=256)
    parser.add_argument("--review_cases_per_source", type=int, default=12)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--expected_world_size", type=int, default=8)
    parser.add_argument("--dino_frame_batch", type=int, default=96)
    parser.add_argument("--siglip_frame_batch", type=int, default=96)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--motion_sigmas", default="0.02,0.04,0.08,0.16,0.32")
    parser.add_argument(
        "--wandb_mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", required=True)
    parser.add_argument(
        "--wandb_group", default="continuous-predictive-object-field-v67-contract-audit"
    )
    parser.add_argument("--wandb_dir", required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_model(args, config, device):
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    expected = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "stage": STATE_STAGE,
        "config": config.to_dict(),
        "historical_checkpoint_used": False,
    }
    differences = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    require(not differences, f"v67 contract audit checkpoint differs: {differences}")
    model = ContinuousPredictiveObjectFieldV67(config, STATE_STAGE)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def weighted_mean_per_case(value, weight):
    axes = tuple(range(1, value.ndim))
    return (value.float() * weight.float()).sum(dim=axes) / weight.float().sum(
        dim=axes
    ).clamp_min(1e-8)


def quantile_per_query(value, quantile):
    return torch.quantile(value.float(), quantile, dim=-1)


def masked_quantile_per_case(value, valid, quantile):
    return torch.stack(
        [
            torch.quantile(row[mask].float(), quantile)
            if mask.any()
            else torch.zeros((), device=value.device)
            for row, mask in zip(value, valid)
        ]
    )


def reappearance_fraction(visibility):
    seen_before = visibility[:, 0]
    events = []
    for frame in range(1, visibility.shape[1]):
        events.append(visibility[:, frame] & ~visibility[:, frame - 1] & seen_before)
        seen_before = seen_before | visibility[:, frame]
    return torch.stack(events, dim=1).any(dim=1).float().mean(dim=-1)


def relation_metrics(prefix, components, sigmas, query_indices):
    relation = components.relation
    weight = components.weight
    pair_count = relation.shape[1] * relation.shape[2]
    diagonal = query_indices[None, :, None] == torch.arange(
        relation.shape[-1], device=relation.device
    )[None, None]
    off_diagonal = (~diagonal).expand(relation.shape[0], -1, -1)
    off_diagonal_weight = weight * off_diagonal.float()
    result = {
        f"{prefix}_relation_unweighted_mean": relation.mean(dim=(1, 2)),
        f"{prefix}_relation_weighted_mean": weighted_mean_per_case(relation, weight),
        f"{prefix}_relation_offdiagonal_mean": weighted_mean_per_case(
            relation, off_diagonal
        ),
        f"{prefix}_relation_offdiagonal_weighted_mean": weighted_mean_per_case(
            relation, off_diagonal_weight
        ),
        f"{prefix}_relation_q50": torch.quantile(relation.flatten(1), 0.50, dim=1),
        f"{prefix}_relation_q90": torch.quantile(relation.flatten(1), 0.90, dim=1),
        f"{prefix}_relation_density_0_25": (relation >= 0.25).float().mean(dim=(1, 2)),
        f"{prefix}_relation_density_0_50": (relation >= 0.50).float().mean(dim=(1, 2)),
        f"{prefix}_relation_density_0_75": (relation >= 0.75).float().mean(dim=(1, 2)),
        f"{prefix}_relation_offdiagonal_density_0_50": weighted_mean_per_case(
            (relation >= 0.50).float(), off_diagonal
        ),
        f"{prefix}_relation_entropy": relation_entropy_v67(relation).mean(dim=(1, 2)),
        f"{prefix}_dino_siglip_disagreement": (
            components.dino_affinity - components.siglip_affinity
        ).abs().mean(dim=(1, 2)),
        f"{prefix}_semantic_motion_disagreement": (
            components.semantic_affinity - components.motion_affinity
        ).abs().mean(dim=(1, 2)),
        f"{prefix}_weighted_pair_fraction": (weight > 0).float().sum(dim=(1, 2))
        / pair_count,
        f"{prefix}_weight_top_1pct_concentration": weighted_concentration_v67(weight),
        f"{prefix}_motion_difference_q50": torch.quantile(
            components.motion_difference.flatten(1), 0.50, dim=1
        ),
        f"{prefix}_motion_difference_q90": torch.quantile(
            components.motion_difference.flatten(1), 0.90, dim=1
        ),
    }
    for sigma in sigmas:
        motion = torch.exp(-components.motion_difference / sigma)
        candidate = (components.semantic_affinity * motion).sqrt().clamp(0.0, 1.0)
        candidate = torch.where(diagonal, torch.ones_like(candidate), candidate)
        label = str(sigma).replace(".", "p")
        result[f"{prefix}_sigma_{label}_relation_mean"] = candidate.mean(dim=(1, 2))
        result[f"{prefix}_sigma_{label}_density_0_50"] = (
            candidate >= 0.50
        ).float().mean(dim=(1, 2))
    return result


def visibility_metrics(target):
    final = target.visibility
    raw = target.tracker_visibility
    in_bounds = target.in_bounds
    dino = target.dino_valid
    siglip = target.siglip_valid
    joint = final & target.relay_visibility
    return {
        "visibility_raw_tracker_fraction": raw.float().mean(dim=(1, 2)),
        "visibility_in_bounds_fraction": in_bounds.float().mean(dim=(1, 2)),
        "visibility_dino_valid_fraction": dino.float().mean(dim=(1, 2)),
        "visibility_siglip_valid_fraction": siglip.float().mean(dim=(1, 2)),
        "visibility_final_fraction": final.float().mean(dim=(1, 2)),
        "visibility_raw_visible_out_of_bounds_fraction": (
            raw & ~in_bounds
        ).float().mean(dim=(1, 2)),
        "visibility_removed_by_dino_fraction": (
            raw & in_bounds & ~dino
        ).float().mean(dim=(1, 2)),
        "visibility_removed_by_siglip_fraction": (
            raw & in_bounds & ~siglip
        ).float().mean(dim=(1, 2)),
        "visibility_primary_relay_disagreement_fraction": (
            final != target.relay_visibility
        ).float().mean(dim=(1, 2)),
        "visibility_primary_relay_joint_fraction": joint.float().mean(dim=(1, 2)),
        "visibility_reappearance_candidate_fraction": reappearance_fraction(final),
        "reliability_nonzero_fraction": (target.reliability > 0).float().mean(dim=1),
        "reliability_q10": torch.quantile(target.reliability.float(), 0.10, dim=1),
        "reliability_q50": torch.quantile(target.reliability.float(), 0.50, dim=1),
        "reliability_q90": torch.quantile(target.reliability.float(), 0.90, dim=1),
        "tracker_reliability_q50": torch.quantile(
            target.tracker_reliability.float(), 0.50, dim=1
        ),
        "appearance_reliability_q50": torch.quantile(
            target.appearance_reliability.float(), 0.50, dim=1
        ),
        "relay_error_q50": torch.quantile(target.relay_error.float(), 0.50, dim=1),
        "relay_error_q90": torch.quantile(target.relay_error.float(), 0.90, dim=1),
        "relay_joint_visibility_q50": torch.quantile(
            target.joint_visibility_fraction.float(), 0.50, dim=1
        ),
    }


def code_and_response_metrics(output, target, config):
    source = output["source_code"]
    future = output["target_code"]
    query_visible = target.visibility[:, config.source_frame].index_select(
        1, target.query_indices
    ) & target.visibility[:, config.target_frame].index_select(1, target.query_indices)
    retrieval = temporal_query_retrieval_v67(
        source.identity, future.identity, query_visible
    )
    relation = output["source_relation"]
    response_norm = relation.response.float().norm(dim=-1)
    response_centered = relation.response.float() - relation.response.float().mean(
        dim=2, keepdim=True
    )
    metrics = {
        "code_rate_q50": torch.quantile(source.rate.float(), 0.50, dim=1),
        "code_rate_q90": torch.quantile(source.rate.float(), 0.90, dim=1),
        "code_mean_norm_q50": torch.quantile(source.mean.float().norm(dim=-1), 0.50, dim=1),
        "code_log_variance_q50": torch.quantile(
            source.log_variance.float().flatten(1), 0.50, dim=1
        ),
        "temporal_same_query_cosine_error": weighted_mean_per_case(
            retrieval["same_query_cosine_error"], retrieval["valid"]
        ),
        "temporal_query_retrieval_margin": weighted_mean_per_case(
            retrieval["retrieval_margin"], retrieval["valid"]
        ),
        "temporal_query_retrieval_top1": weighted_mean_per_case(
            retrieval["top1"], retrieval["valid"]
        ),
        "temporal_query_reciprocal_rank": weighted_mean_per_case(
            retrieval["reciprocal_rank"], retrieval["valid"]
        ),
        "response_norm_q10": quantile_per_query(response_norm, 0.10).mean(dim=1),
        "response_norm_q50": quantile_per_query(response_norm, 0.50).mean(dim=1),
        "response_norm_q90": quantile_per_query(response_norm, 0.90).mean(dim=1),
        "response_spatial_rms": response_centered.square().mean(dim=(2, 3)).sqrt().mean(dim=1),
    }
    return metrics, retrieval, response_norm, response_centered


def aggregated_response_feature(model, output, target):
    relation = output["source_relation"]
    current = output["source_history"].current
    projected = model.object_code.response_projection(current)
    response = relation.response + projected[:, None]
    valid = output["source_history"].valid[:, -1]
    weight = relation.probability.float() * valid[:, None].float()
    weight = weight * target.context_mask[:, None].float()
    normalized = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.einsum("bqp,bqpd->bqd", normalized, response.float())


def _decoded_proxy_errors(field, target, components, config):
    held = (~target.context_mask)[:, None].float()
    semantic_weight = components.weight * components.relation * held
    dino = 1.0 - F.cosine_similarity(
        field.dino.float(), target.dino[:, config.source_frame, None].float(), dim=-1
    )
    siglip = 1.0 - F.cosine_similarity(
        field.siglip.float(), target.siglip[:, config.source_frame, None].float(), dim=-1
    )
    semantic = weighted_mean_per_case(0.5 * (dino + siglip), semantic_weight)
    support_error = F.binary_cross_entropy_with_logits(
        field.support_logits.float(), components.relation.float(), reduction="none"
    )
    support = weighted_mean_per_case(
        support_error, components.weight * held
    )
    return semantic, support


def partition_intervention_metrics(model, output, target, components, config):
    source = output["source_code"].sample.float()
    anchor = target.anchor_coordinates.index_select(1, target.query_indices)
    variants = {
        "first_128_zero": torch.cat(
            (torch.zeros_like(source[..., : config.identity_dim]), source[..., config.identity_dim :]),
            dim=-1,
        ),
        "last_192_zero": torch.cat(
            (source[..., : config.identity_dim], torch.zeros_like(source[..., config.identity_dim :])),
            dim=-1,
        ),
        "first_128_query_roll": torch.cat(
            (source[..., : config.identity_dim].roll(1, dims=1), source[..., config.identity_dim :]),
            dim=-1,
        ),
        "last_192_query_roll": torch.cat(
            (source[..., : config.identity_dim], source[..., config.identity_dim :].roll(1, dims=1)),
            dim=-1,
        ),
    }
    baseline = output["source_decoded"]
    baseline_semantic, baseline_support = _decoded_proxy_errors(
        baseline, target, components, config
    )
    held = (~target.context_mask)[:, None].float()
    result = {
        "partition_baseline_semantic_proxy_error": baseline_semantic,
        "partition_baseline_support_proxy_bce": baseline_support,
    }
    for name, code in variants.items():
        decoded = model.operator.decoder(
            code,
            anchor,
            target.anchor_coordinates,
            target.scales,
        )
        semantic, support = _decoded_proxy_errors(decoded, target, components, config)
        output_weight = held.expand_as(decoded.support_logits)
        semantic_change = 0.5 * (
            1.0 - F.cosine_similarity(decoded.dino, baseline.dino, dim=-1)
            + 1.0 - F.cosine_similarity(decoded.siglip, baseline.siglip, dim=-1)
        )
        response_change = (
            decoded.response.float() - baseline.response.float()
        ).square().mean(dim=-1).sqrt()
        result[f"partition_{name}_semantic_proxy_error_delta"] = (
            semantic - baseline_semantic
        )
        result[f"partition_{name}_support_proxy_bce_delta"] = support - baseline_support
        result[f"partition_{name}_decoded_semantic_change"] = weighted_mean_per_case(
            semantic_change, output_weight
        )
        result[f"partition_{name}_decoded_response_change"] = weighted_mean_per_case(
            response_change, output_weight
        )
    return result


def query_rate_distortion_terms(output, target, components, config):
    held = (~target.context_mask)[:, None].float()
    semantic_weight = components.relation * components.weight * held
    shared_semantic = 0.5 * (
        1.0
        - F.cosine_similarity(
            output["source_decoded"].dino.float(),
            target.dino[:, config.source_frame, None].float(),
            dim=-1,
        )
        + 1.0
        - F.cosine_similarity(
            output["source_decoded"].siglip.float(),
            target.siglip[:, config.source_frame, None].float(),
            dim=-1,
        )
    )
    denominator = semantic_weight.sum(dim=-1)
    shared_distortion = (shared_semantic * semantic_weight).sum(dim=-1)
    shared_distortion = shared_distortion / denominator.clamp_min(1e-8)
    point_distortion = 0.5 * (
        1.0
        - F.cosine_similarity(
            output["point_dino"].float(),
            target.dino[:, config.source_frame].float(),
            dim=-1,
        )
        + 1.0
        - F.cosine_similarity(
            output["point_siglip"].float(),
            target.siglip[:, config.source_frame].float(),
            dim=-1,
        )
    )
    separate_distortion = torch.einsum(
        "bqp,bp->bq", semantic_weight, point_distortion
    ) / denominator.clamp_min(1e-8)
    maximum_weight = semantic_weight.amax(dim=-1)
    separate_rate = torch.einsum(
        "bqp,bp->bq", semantic_weight, output["source_code"].point_rate.float()
    ) / maximum_weight.clamp_min(1e-8)
    effective_point_count = denominator / maximum_weight.clamp_min(1e-8)
    shared_rate = output["source_code"].rate.float()
    shared_cost = shared_distortion + config.rate_weight * shared_rate
    separate_cost = separate_distortion + config.rate_weight * separate_rate
    valid = denominator > 0
    return {
        "valid": valid,
        "shared_distortion": shared_distortion,
        "separate_distortion": separate_distortion,
        "shared_rate": shared_rate,
        "separate_rate": separate_rate,
        "effective_point_count": effective_point_count,
        "shared_cost": shared_cost,
        "separate_cost": separate_cost,
        "saving": separate_cost - shared_cost,
    }


def _point_xy(coordinate, width, height):
    x = float((coordinate[0] + 1.0) * 0.5 * (width - 1))
    y = float((coordinate[1] + 1.0) * 0.5 * (height - 1))
    return x, y


def _frame_image(frame, height, width, panel_width=320):
    array = frame[:, :height, :width].permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(array)
    panel_height = max(1, round(height * panel_width / width))
    return image.resize((panel_width, panel_height), Image.Resampling.BILINEAR)


def _mosaic(panels, columns=4):
    rows = (len(panels) + columns - 1) // columns
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (columns * width, rows * height), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % columns) * width, (index // columns) * height))
    return canvas


def render_visibility_review(path, video, native_hw, target, batch_index, candidates):
    height, width = (int(value) for value in native_hw[batch_index].tolist())
    panels = []
    colors = ("#00ff66", "#ffcc00", "#ff3355", "#33ccff")
    for frame_index in range(video.shape[1]):
        panel = _frame_image(video[batch_index, frame_index], height, width)
        draw = ImageDraw.Draw(panel)
        x_scale, y_scale = panel.width / width, panel.height / height
        for order, candidate in enumerate(candidates):
            coordinate = target.track_coordinates[batch_index, frame_index, candidate]
            x, y = _point_xy(coordinate, width, height)
            x, y = x * x_scale, y * y_scale
            out_of_bounds = not bool(target.in_bounds[batch_index, frame_index, candidate])
            x = min(max(x, 7.0), panel.width - 7.0)
            y = min(max(y, 7.0), panel.height - 7.0)
            raw_visible = bool(
                target.tracker_visibility[batch_index, frame_index, candidate]
            )
            final_visible = bool(target.visibility[batch_index, frame_index, candidate])
            if out_of_bounds:
                status_color = "#ff33cc"
            elif final_visible:
                status_color = "#00ff66"
            elif raw_visible:
                status_color = "#ffcc00"
            else:
                status_color = "#ff3355"
            radius = 6
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=colors[order % len(colors)],
                width=3,
            )
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=status_color)
            suffix = " OOB" if out_of_bounds else ""
            draw.text(
                (x + 7, y - 7),
                f"{candidate}{suffix}",
                fill=colors[order % len(colors)],
            )
        draw.text((6, 6), f"frame={frame_index}", fill="white", stroke_width=2, stroke_fill="black")
        panels.append(panel)
    image = _mosaic(panels)
    image.save(path)


def render_relation_review(path, video, native_hw, target, batch_index, query, candidate):
    height, width = (int(value) for value in native_hw[batch_index].tolist())
    query_candidate = int(target.query_indices[query])
    panels = []
    for frame_index in range(video.shape[1]):
        panel = _frame_image(video[batch_index, frame_index], height, width)
        draw = ImageDraw.Draw(panel)
        x_scale, y_scale = panel.width / width, panel.height / height
        points = (
            (query_candidate, "#33ccff", "query"),
            (candidate, "#ff33cc", "candidate"),
        )
        pixel_points = []
        for point_index, color, label in points:
            coordinate = target.track_coordinates[batch_index, frame_index, point_index]
            x, y = _point_xy(coordinate, width, height)
            x, y = x * x_scale, y * y_scale
            out_of_bounds = not bool(
                target.in_bounds[batch_index, frame_index, point_index]
            )
            x = min(max(x, 8.0), panel.width - 8.0)
            y = min(max(y, 8.0), panel.height - 8.0)
            pixel_points.append((x, y))
            radius = 7
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=color,
                width=4,
            )
            suffix = " OOB" if out_of_bounds else ""
            draw.text((x + 8, y - 8), f"{label}{suffix}", fill=color)
        draw.line((*pixel_points[0], *pixel_points[1]), fill="#ffffff", width=2)
        draw.text((6, 6), f"frame={frame_index}", fill="white", stroke_width=2, stroke_fill="black")
        panels.append(panel)
    image = _mosaic(panels)
    image.save(path)


def select_visibility_candidates(target, batch_index, count=4):
    relay_error = target.relay_error[batch_index].float()
    reliability = target.reliability[batch_index].float()
    removed = (
        target.tracker_visibility[batch_index]
        & ~target.visibility[batch_index]
    ).any(dim=0)
    choices = [
        int(relay_error.argmax()),
        int(reliability.argmax()),
        int(reliability.argmin()),
    ]
    if removed.any():
        choices.append(int(removed.float().argmax()))
    unique = []
    for candidate in choices:
        if candidate not in unique:
            unique.append(candidate)
    for candidate in relay_error.argsort(descending=True).tolist():
        if candidate not in unique:
            unique.append(candidate)
        if len(unique) == count:
            break
    return unique[:count]


def select_relation_pairs(components, batch_index, query_indices, count=2):
    relation = components.relation[batch_index]
    weight = components.weight[batch_index]
    disagreement = (
        components.dino_affinity[batch_index] - components.siglip_affinity[batch_index]
    ).abs()
    diagonal = query_indices[:, None] == torch.arange(
        relation.shape[-1], device=relation.device
    )[None]
    usable = (weight > 0) & ~diagonal
    uncertain_score = -(relation - 0.5).abs()
    uncertain_score = uncertain_score.masked_fill(~usable, -1e9)
    disagreement_score = disagreement.masked_fill(~usable, -1e9)
    flat_indices = [
        int(uncertain_score.flatten().argmax()),
        int(disagreement_score.flatten().argmax()),
    ]
    pairs = []
    candidate_count = relation.shape[-1]
    for flat in flat_indices:
        pair = (flat // candidate_count, flat % candidate_count)
        if pair not in pairs:
            pairs.append(pair)
    return pairs[:count]


def build_review_artifacts(
    args,
    batch,
    target,
    source_components,
    student_relation,
    selected,
    source_name,
    local_review_limit,
    rank,
):
    manifest, images = [], []
    review_count = min(len(selected), local_review_limit)
    image_dir = os.path.join(args.artifact_dir, "review_images")
    os.makedirs(image_dir, exist_ok=True)
    for batch_index in range(review_count):
        dataset_index = int(selected[batch_index])
        stem = f"{source_name}_{dataset_index}_r{rank}"
        candidates = select_visibility_candidates(target, batch_index)
        visibility_path = os.path.join(image_dir, f"{stem}_visibility.png")
        render_visibility_review(
            visibility_path,
            batch["video_rgb"].cpu(),
            batch["native_image_hw"].cpu(),
            target,
            batch_index,
            candidates,
        )
        images.append(
            {
                "type": "visibility",
                "source": source_name,
                "dataset_index": dataset_index,
                "image_path": visibility_path,
                "description": (
                    "ring color identifies candidate ID; center dot is green=final visible, "
                    "yellow=tracker visible but feature-invalid, red=tracker invisible, "
                    "magenta=out of bounds; all eight frames are shown"
                ),
            }
        )
        for candidate in candidates:
            for frame_index in range(target.visibility.shape[1]):
                manifest.append(
                    {
                        "type": "visibility",
                        "source": source_name,
                        "dataset_index": dataset_index,
                        "candidate_index": candidate,
                        "frame_index": frame_index,
                        "image_path": visibility_path,
                        "raw_tracker_visible": bool(
                            target.tracker_visibility[batch_index, frame_index, candidate]
                        ),
                        "in_bounds": bool(
                            target.in_bounds[batch_index, frame_index, candidate]
                        ),
                        "dino_valid": bool(
                            target.dino_valid[batch_index, frame_index, candidate]
                        ),
                        "siglip_valid": bool(
                            target.siglip_valid[batch_index, frame_index, candidate]
                        ),
                        "teacher_visible": bool(
                            target.visibility[batch_index, frame_index, candidate]
                        ),
                        "label": None,
                    }
                )
        for pair_index, (query, candidate) in enumerate(
            select_relation_pairs(
                source_components, batch_index, target.query_indices
            )
        ):
            relation_path = os.path.join(
                image_dir, f"{stem}_relation_{pair_index}.png"
            )
            render_relation_review(
                relation_path,
                batch["video_rgb"].cpu(),
                batch["native_image_hw"].cpu(),
                target,
                batch_index,
                query,
                candidate,
            )
            probability = float(source_components.relation[batch_index, query, candidate])
            manifest.append(
                {
                    "type": "relation",
                    "source": source_name,
                    "dataset_index": dataset_index,
                    "query_ordinal": query,
                    "query_candidate_index": int(target.query_indices[query]),
                    "candidate_index": candidate,
                    "image_path": relation_path,
                    "teacher_probability": probability,
                    "student_probability": float(
                        student_relation[batch_index, query, candidate]
                    ),
                    "dino_affinity": float(
                        source_components.dino_affinity[batch_index, query, candidate]
                    ),
                    "siglip_affinity": float(
                        source_components.siglip_affinity[batch_index, query, candidate]
                    ),
                    "motion_affinity": float(
                        source_components.motion_affinity[batch_index, query, candidate]
                    ),
                    "label": None,
                }
            )
            images.append(
                {
                    "type": "relation",
                    "source": source_name,
                    "dataset_index": dataset_index,
                    "image_path": relation_path,
                    "description": (
                        f"query={int(target.query_indices[query])} candidate={candidate} "
                        f"teacher_relation={probability:.3f} "
                        f"student_relation={float(student_relation[batch_index, query, candidate]):.3f}"
                    ),
                }
            )
    return manifest, images


@torch.no_grad()
def evaluate_source(args, dataset, source_index, teacher, model, context, config, sigmas):
    indices = dataset.balanced_source_evaluation_indices(
        source_index, args.items_per_source
    )
    indices = shard_indices_v62(indices, context)
    cases, queries, review_manifest, review_images = [], [], [], []
    probe_chunks = {
        "full_code_320": [],
        "first_128_normalized": [],
        "last_192": [],
        "aggregated_response_192": [],
        "future_displacement": [],
        "anchor_coordinate": [],
        "query_ordinal": [],
        "task_group": [],
        "source_index": [],
        "temporal_query_valid": [],
    }
    local_review_limit = args.review_cases_per_source // context.world_size
    local_review_limit += int(
        context.rank < args.review_cases_per_source % context.world_size
    )
    reviewed = 0
    for start in range(0, len(indices), args.batch):
        selected = indices[start : start + args.batch]
        samples = [dataset[(index, config.clip_frames)] for index in selected]
        cpu_batch = collate_native_video_batch_v65(samples)
        batch = move_to_device(cpu_batch, context.device)
        target = teacher(batch)
        output = model(batch, target)
        source_components = teacher_relation_components_v67(
            target, config.source_frame, config
        )
        future_components = teacher_relation_components_v67(
            target, config.target_frame, config
        )
        metrics = {}
        metrics.update(visibility_metrics(target))
        metrics.update(
            relation_metrics(
                "source", source_components, sigmas, target.query_indices
            )
        )
        metrics.update(
            relation_metrics(
                "future", future_components, sigmas, target.query_indices
            )
        )
        code_metrics, retrieval, response_norm, response_centered = (
            code_and_response_metrics(output, target, config)
        )
        metrics.update(code_metrics)
        rate_distortion = query_rate_distortion_terms(
            output, target, source_components, config
        )
        rate_valid = rate_distortion["valid"]
        metrics.update(
            {
                "rate_proxy_valid_query_fraction": rate_valid.float().mean(dim=1),
                "rate_proxy_saving_q10": masked_quantile_per_case(
                    rate_distortion["saving"], rate_valid, 0.10
                ),
                "rate_proxy_saving_q50": masked_quantile_per_case(
                    rate_distortion["saving"], rate_valid, 0.50
                ),
                "rate_proxy_saving_q90": masked_quantile_per_case(
                    rate_distortion["saving"], rate_valid, 0.90
                ),
                "rate_proxy_positive_fraction": (
                    (rate_distortion["saving"] > 0) & rate_valid
                ).float().sum(dim=1)
                / rate_valid.float().sum(dim=1).clamp_min(1.0),
                "rate_proxy_effective_point_count_q50": masked_quantile_per_case(
                    rate_distortion["effective_point_count"], rate_valid, 0.50
                ),
            }
        )
        metrics.update(
            partition_intervention_metrics(
                model, output, target, source_components, config
            )
        )
        metrics["teacher_relation_temporal_absolute_change"] = (
            source_components.relation - future_components.relation
        ).abs().mean(dim=(1, 2))
        future_displacement = (
            target.track_coordinates[:, config.target_frame]
            - target.track_coordinates[:, config.source_frame]
        )
        query_displacement = future_displacement.index_select(1, target.query_indices)
        query_anchor = target.anchor_coordinates.index_select(1, target.query_indices)
        source_code = output["source_code"]
        aggregate_response = aggregated_response_feature(
            model, output, target
        )
        query_count = config.query_count
        for batch_index, dataset_index in enumerate(selected):
            row = {
                "dataset_index": int(dataset_index),
                "source": dataset.source_names[source_index],
                "source_index": source_index,
                "task_group_index": int(batch["task_group_index"][batch_index]),
                "temporal_step_seconds": float(
                    batch["temporal_step_seconds"][batch_index]
                ),
                "native_height": int(batch["native_image_hw"][batch_index, 0]),
                "native_width": int(batch["native_image_hw"][batch_index, 1]),
                "decode_replaced": bool(batch["decode_replaced"][batch_index]),
            }
            row.update(
                {
                    name: float(value[batch_index])
                    for name, value in metrics.items()
                }
            )
            for sigma in sigmas:
                label = str(sigma).replace(".", "p")
                row[f"motion_sigma_{label}_pixels_x"] = sigma * (
                    row["native_width"] - 1
                ) / 2.0
                row[f"motion_sigma_{label}_pixels_y"] = sigma * (
                    row["native_height"] - 1
                ) / 2.0
            cases.append(row)
            for query in range(query_count):
                query_valid = bool(retrieval["valid"][batch_index, query])
                rate_query_valid = bool(rate_valid[batch_index, query])
                response = response_norm[batch_index, query]
                relation_weight = source_components.weight[batch_index, query]
                relation_target = source_components.relation[batch_index, query]
                positive = (relation_target >= 0.5) & (relation_weight > 0)
                negative = (relation_target < 0.25) & (relation_weight > 0)
                positive_norm = response[positive].mean() if positive.any() else torch.tensor(0.0, device=response.device)
                negative_norm = response[negative].mean() if negative.any() else torch.tensor(0.0, device=response.device)
                queries.append(
                    {
                        "dataset_index": int(dataset_index),
                        "source": dataset.source_names[source_index],
                        "source_index": source_index,
                        "task_group_index": int(batch["task_group_index"][batch_index]),
                        "query_ordinal": query,
                        "query_candidate_index": int(target.query_indices[query]),
                        "anchor_x": float(query_anchor[batch_index, query, 0]),
                        "anchor_y": float(query_anchor[batch_index, query, 1]),
                        "future_displacement_x": float(query_displacement[batch_index, query, 0]),
                        "future_displacement_y": float(query_displacement[batch_index, query, 1]),
                        "future_displacement_norm": float(query_displacement[batch_index, query].norm()),
                        "code_rate": float(source_code.rate[batch_index, query]),
                        "support_mass": float(source_code.support_mass[batch_index, query]),
                        "rate_proxy_valid": rate_query_valid,
                        "shared_distortion_proxy": float(
                            rate_distortion["shared_distortion"][batch_index, query]
                        )
                        if rate_query_valid
                        else None,
                        "separate_distortion_proxy": float(
                            rate_distortion["separate_distortion"][batch_index, query]
                        )
                        if rate_query_valid
                        else None,
                        "shared_rate_proxy": float(
                            rate_distortion["shared_rate"][batch_index, query]
                        )
                        if rate_query_valid
                        else None,
                        "separate_rate_proxy": float(
                            rate_distortion["separate_rate"][batch_index, query]
                        )
                        if rate_query_valid
                        else None,
                        "effective_point_count_proxy": float(
                            rate_distortion["effective_point_count"][batch_index, query]
                        )
                        if rate_query_valid
                        else None,
                        "shared_cost_proxy": float(
                            rate_distortion["shared_cost"][batch_index, query]
                        )
                        if rate_query_valid
                        else None,
                        "separate_cost_proxy": float(
                            rate_distortion["separate_cost"][batch_index, query]
                        )
                        if rate_query_valid
                        else None,
                        "rate_saving_proxy": float(
                            rate_distortion["saving"][batch_index, query]
                        )
                        if rate_query_valid
                        else None,
                        "temporal_query_valid": query_valid,
                        "same_query_cosine_error": float(retrieval["same_query_cosine_error"][batch_index, query])
                        if query_valid
                        else None,
                        "query_retrieval_margin": float(retrieval["retrieval_margin"][batch_index, query])
                        if query_valid
                        else None,
                        "query_retrieval_top1": float(retrieval["top1"][batch_index, query])
                        if query_valid
                        else None,
                        "query_reciprocal_rank": float(retrieval["reciprocal_rank"][batch_index, query])
                        if query_valid
                        else None,
                        "response_norm_q10": float(torch.quantile(response, 0.10)),
                        "response_norm_q50": float(torch.quantile(response, 0.50)),
                        "response_norm_q90": float(torch.quantile(response, 0.90)),
                        "response_spatial_rms": float(
                            response_centered[batch_index, query].square().mean().sqrt()
                        ),
                        "response_norm_proxy_contrast": float(positive_norm - negative_norm),
                        "source_relation_q50": float(torch.quantile(relation_target, 0.50)),
                        "source_relation_q90": float(torch.quantile(relation_target, 0.90)),
                        "source_relation_weighted_candidate_fraction": float((relation_weight > 0).float().mean()),
                        "source_dino_siglip_disagreement": float(
                            (
                                source_components.dino_affinity[batch_index, query]
                                - source_components.siglip_affinity[batch_index, query]
                            ).abs().mean()
                        ),
                    }
                )
        repeated_groups = batch["task_group_index"][:, None].expand(-1, query_count)
        repeated_sources = batch["source_index"][:, None].expand(-1, query_count)
        query_ordinals = torch.arange(query_count, device=batch["video_rgb"].device)
        query_ordinals = query_ordinals[None].expand(len(selected), -1)
        probe_chunks["full_code_320"].append(source_code.mean.cpu())
        probe_chunks["first_128_normalized"].append(source_code.identity.cpu())
        probe_chunks["last_192"].append(source_code.dynamic.cpu())
        probe_chunks["aggregated_response_192"].append(aggregate_response.cpu())
        probe_chunks["future_displacement"].append(query_displacement.cpu())
        probe_chunks["anchor_coordinate"].append(query_anchor.cpu())
        probe_chunks["query_ordinal"].append(query_ordinals.cpu())
        probe_chunks["task_group"].append(repeated_groups.cpu())
        probe_chunks["source_index"].append(repeated_sources.cpu())
        probe_chunks["temporal_query_valid"].append(
            retrieval["valid"].bool().cpu()
        )
        remaining = max(0, local_review_limit - reviewed)
        if remaining:
            manifest, images = build_review_artifacts(
                args,
                cpu_batch,
                target,
                source_components,
                output["source_relation"].probability,
                selected,
                dataset.source_names[source_index],
                remaining,
                context.rank,
            )
            review_manifest.extend(manifest)
            review_images.extend(images)
            reviewed += min(len(selected), remaining)
    probes = {name: torch.cat(chunks) for name, chunks in probe_chunks.items()}
    return {
        "cases": cases,
        "queries": queries,
        "review_manifest": review_manifest,
        "review_images": review_images,
        "probes": probes,
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def summarize_rows(rows):
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    return {
        key: distribution_summary_v67(
            torch.tensor(
                [
                    float(row[key])
                    for row in rows
                    if isinstance(row.get(key), (int, float))
                    and not isinstance(row.get(key), bool)
                ]
            )
        )
        for key in keys
    }


def summarize_sources(cases, queries, source_names):
    return {
        source: {
            "case_count": sum(row["source"] == source for row in cases),
            "query_count": sum(row["source"] == source for row in queries),
            "case_distributions": summarize_rows(
                [row for row in cases if row["source"] == source]
            ),
            "query_distributions": summarize_rows(
                [row for row in queries if row["source"] == source]
            ),
        }
        for source in source_names
    }


def read_annotations(path):
    if not path:
        return []
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    require(all(row.get("label") is not None for row in rows), "annotation labels contain null")
    return rows


def annotation_key(row):
    if row["type"] == "visibility":
        return (
            "visibility",
            int(row["dataset_index"]),
            int(row["candidate_index"]),
            int(row["frame_index"]),
        )
    return (
        "relation",
        int(row["dataset_index"]),
        int(row["query_candidate_index"]),
        int(row["candidate_index"]),
    )


def score_annotations(annotations, manifest):
    if not annotations:
        return {
            "status": "unverified_no_independent_annotations",
            "visibility": None,
            "teacher_relation": None,
            "student_relation": None,
        }
    predictions = {annotation_key(row): row for row in manifest}
    visibility_probability, visibility_label = [], []
    teacher_relation_probability, student_relation_probability = [], []
    relation_label = []
    for annotation in annotations:
        key = annotation_key(annotation)
        require(key in predictions, f"annotation does not match review manifest: {key}")
        prediction = predictions[key]
        if annotation["type"] == "visibility":
            visibility_probability.append(float(prediction["teacher_visible"]))
            visibility_label.append(bool(annotation["label"]))
        else:
            teacher_relation_probability.append(float(prediction["teacher_probability"]))
            student_relation_probability.append(float(prediction["student_probability"]))
            relation_label.append(bool(annotation["label"]))

    def metrics(probability, label):
        if not probability:
            return None
        return binary_annotation_metrics_v67(
            torch.tensor(probability), torch.tensor(label, dtype=torch.bool)
        )

    return {
        "status": "measured_against_independent_annotations",
        "visibility": metrics(visibility_probability, visibility_label),
        "teacher_relation": metrics(
            teacher_relation_probability, relation_label
        ),
        "student_relation": metrics(
            student_relation_probability, relation_label
        ),
    }


def probe_report(probes):
    group = probes["task_group"].flatten()
    query = probes["query_ordinal"].flatten()
    motion = probes["future_displacement"]
    anchor = probes["anchor_coordinate"]
    temporal_valid = probes["temporal_query_valid"].bool()
    valid_group = probes["task_group"][temporal_valid]
    valid_motion = motion[temporal_valid]
    partitions = (
        "full_code_320",
        "first_128_normalized",
        "last_192",
        "aggregated_response_192",
    )
    result = {
        "code_health": {
            name: effective_rank_v67(probes[name])
            for name in partitions
        },
        "future_displacement_cross_fit": {
            name: cross_fitted_ridge_probe_v67(
                probes[name][temporal_valid], valid_motion, valid_group
            )
            for name in partitions
        },
        "anchor_coordinate_cross_fit": {
            name: cross_fitted_ridge_probe_v67(probes[name], anchor, group)
            for name in partitions
        },
        "fixed_query_index_probe": {
            name: query_index_centroid_probe_v67(probes[name], query, group)
            for name in partitions
        },
    }
    result["interpretation"] = {
        "code_health": "anti-collapse only",
        "future_displacement_cross_fit": "tests linear predictability of the teacher track displacement on source/target-visible queries and held task groups; it does not establish physical motion semantics or causal dynamics",
        "anchor_coordinate_cross_fit": "measures spatial leakage into each named partition",
        "fixed_query_index_probe": "measures query-index leakage; high accuracy is not identity",
    }
    return result


def flatten_numeric(prefix, value, output):
    if isinstance(value, dict):
        for name, child in value.items():
            flatten_numeric(f"{prefix}/{name}" if prefix else name, child, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = value


def defined_metric_values(rows, name):
    values = []
    for row in rows:
        value = row.get(name)
        if value is not None:
            values.append(float(value))
    return values


def write_wandb(args, report, cases, queries, review_images, checkpoint_step):
    if args.wandb_mode == "disabled":
        return ""
    import wandb

    os.makedirs(args.wandb_dir, exist_ok=True)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=("v67", "contract-validity", "held", "no-promotion-decision"),
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        config={
            "checkpoint": os.path.abspath(args.checkpoint),
            "source_revision": args.source_revision,
            "items_per_source": args.items_per_source,
            "review_cases_per_source": args.review_cases_per_source,
            "evaluation_partition": f"held:{args.held_group_stride}",
            "motion_sigmas": args.motion_sigmas,
            "annotation_jsonl": os.path.abspath(args.annotation_jsonl)
            if args.annotation_jsonl
            else "",
        },
    )
    scalars = {}
    flatten_numeric("audit", report["case_distributions"], scalars)
    flatten_numeric("audit/query", report["query_distributions"], scalars)
    flatten_numeric("audit/probe", report["probes"], scalars)
    flatten_numeric("audit/annotations", report["independent_annotations"], scalars)
    source_metrics = (
        "visibility_final_fraction",
        "reliability_q50",
        "source_relation_offdiagonal_weighted_mean",
        "source_dino_siglip_disagreement",
        "source_semantic_motion_disagreement",
        "temporal_same_query_cosine_error",
        "temporal_query_retrieval_top1",
        "response_spatial_rms",
        "rate_proxy_saving_q50",
    )
    for source, summary in report["per_source"].items():
        for metric in source_metrics:
            distribution = summary["case_distributions"].get(metric)
            if distribution:
                for statistic in ("q05", "q50", "q95"):
                    value = distribution.get(statistic)
                    if value is not None:
                        scalars[f"audit/source/{source}/{metric}/{statistic}"] = value
    selected_histograms = (
        "visibility_final_fraction",
        "reliability_q50",
        "source_relation_weighted_mean",
        "source_dino_siglip_disagreement",
        "future_displacement_norm",
        "same_query_cosine_error",
        "response_spatial_rms",
        "rate_saving_proxy",
        "effective_point_count_proxy",
    )
    for name in selected_histograms:
        rows = cases if any(name in row for row in cases) else queries
        values = defined_metric_values(rows, name)
        scalars[f"audit/histogram_support/{name}/defined_count"] = len(values)
        scalars[f"audit/histogram_support/{name}/undefined_count"] = (
            len(rows) - len(values)
        )
        if values:
            scalars[f"histogram/{name}"] = wandb.Histogram(values)
    case_columns = (
        "source",
        "dataset_index",
        "task_group_index",
        "temporal_step_seconds",
        "visibility_final_fraction",
        "reliability_q50",
        "source_relation_weighted_mean",
        "source_dino_siglip_disagreement",
        "source_semantic_motion_disagreement",
        "temporal_same_query_cosine_error",
        "temporal_query_retrieval_top1",
        "response_spatial_rms",
        "rate_proxy_saving_q50",
        "rate_proxy_positive_fraction",
        "rate_proxy_effective_point_count_q50",
    )
    case_table = wandb.Table(columns=list(case_columns))
    for row in cases[: min(len(cases), 512)]:
        case_table.add_data(*(row.get(name) for name in case_columns))
    review_table = wandb.Table(
        columns=("type", "source", "dataset_index", "description", "image")
    )
    for item in review_images:
        review_table.add_data(
            item["type"],
            item["source"],
            item["dataset_index"],
            item["description"],
            wandb.Image(item["image_path"]),
        )
    scalars["audit/cases"] = case_table
    scalars["audit/review_images"] = review_table
    run.log(scalars, step=checkpoint_step)
    artifact = wandb.Artifact(
        f"{args.wandb_name}-evidence", type="contract-validity-audit"
    )
    artifact.add_file(args.output)
    artifact.add_dir(args.artifact_dir)
    run.log_artifact(artifact)
    run.summary.update(
        {
            "audit/status": report["status"],
            "audit/promotion_decision": report["promotion_decision"],
            "audit/annotation_status": report["independent_annotations"]["status"],
            "audit/case_count": report["case_count"],
            "audit/query_count": report["query_count"],
        }
    )
    run_id = run.id
    run.finish()
    return run_id


def main():
    args = parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)
    context = initialize_distributed_audit_v62()
    require(
        context.world_size == args.expected_world_size,
        "v67 contract audit world size differs from explicit contract",
    )
    config = ContinuousPredictiveObjectFieldConfigV67()
    config.validate()
    sigmas = tuple(float(value) for value in args.motion_sigmas.split(","))
    dataset = MultiSourcePointTrackObjectVideoDataset(
        args.data_index,
        "train",
        str(config.clip_frames),
        "100,200,400",
        0,
        17,
        group_partition="held",
        held_group_stride=args.held_group_stride,
        preserve_native_rgb=True,
    )
    model, checkpoint = load_model(args, config, context.device)
    teacher = ContinuousPredictiveTeacherRuntimeV67(
        config,
        context.device,
        args.amp,
        args.dino_checkpoint,
        args.siglip_checkpoint,
        args.tracker_checkpoint,
        args.dino_frame_batch,
        args.siglip_frame_batch,
    )
    local = {
        name: evaluate_source(
            args,
            dataset,
            source_index,
            teacher,
            model,
            context,
            config,
            sigmas,
        )
        for source_index, name in enumerate(dataset.source_names)
    }
    rank_path = os.path.join(args.artifact_dir, f"rank_{context.rank:02d}.pt")
    torch.save(local, rank_path)
    dist.barrier()
    if not context.is_main:
        finish_distributed_audit_v62()
        return
    cases, queries, manifest, review_images = [], [], [], []
    probe_parts = {}
    for rank in range(context.world_size):
        payload = torch.load(
            os.path.join(args.artifact_dir, f"rank_{rank:02d}.pt"),
            map_location="cpu",
            weights_only=False,
        )
        for source_payload in payload.values():
            cases.extend(source_payload["cases"])
            queries.extend(source_payload["queries"])
            manifest.extend(source_payload["review_manifest"])
            review_images.extend(source_payload["review_images"])
            for name, value in source_payload["probes"].items():
                probe_parts.setdefault(name, []).append(value)
    probes = {name: torch.cat(parts) for name, parts in probe_parts.items()}
    cases.sort(key=lambda row: (row["source_index"], row["dataset_index"]))
    queries.sort(
        key=lambda row: (
            row["source_index"],
            row["dataset_index"],
            row["query_ordinal"],
        )
    )
    manifest.sort(
        key=lambda row: (
            row["source"],
            row["dataset_index"],
            row["type"],
            row.get("candidate_index", -1),
            row.get("frame_index", -1),
        )
    )
    case_path = os.path.join(args.artifact_dir, "case_metrics.jsonl")
    query_path = os.path.join(args.artifact_dir, "query_metrics.jsonl")
    manifest_path = os.path.join(args.artifact_dir, "human_review_manifest.jsonl")
    write_jsonl(case_path, cases)
    write_jsonl(query_path, queries)
    write_jsonl(manifest_path, manifest)
    annotations = read_annotations(args.annotation_jsonl)
    report = {
        "status": "completed",
        "promotion_decision": "not_computed_by_design",
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_step": int(checkpoint["global_step"]),
        "checkpoint_git_commit": checkpoint["git_commit"],
        "evaluation_source_revision": args.source_revision,
        "data": os.path.abspath(args.data_index),
        "partition": f"held:{args.held_group_stride}",
        "world_size": context.world_size,
        "items_per_source": args.items_per_source,
        "source_names": list(dataset.source_names),
        "case_count": len(cases),
        "query_count": len(queries),
        "motion_sigmas": sigmas,
        "claim_registry": CLAIM_REGISTRY_V67,
        "contract_formulas": {
            "source_displacement": (
                "For each candidate, average adjacent normalized CoTracker coordinate "
                "differences over frames 0..source_frame, using only adjacent pairs "
                "marked visible by the composed teacher visibility."
            ),
            "future_displacement": (
                "Normalized CoTracker coordinate at target_frame minus its coordinate "
                "at source_frame; this is a tracker measurement, not physical ground truth."
            ),
            "motion_affinity": (
                "exp(-L2(query_displacement - candidate_displacement) / motion_sigma)."
            ),
            "semantic_affinity": (
                "sqrt(remapped_DINO_cosine * remapped_SigLIP_cosine), with each "
                "cosine remapped from [-1,1] to [0,1]."
            ),
            "teacher_relation": "sqrt(semantic_affinity * motion_affinity).",
            "teacher_pair_weight": (
                "pair_visibility * sqrt(query_reliability * candidate_reliability)."
            ),
            "teacher_visibility": (
                "CoTracker model-visible AND normalized coordinate in bounds AND "
                "DINO local feature valid AND SigLIP local feature valid."
            ),
            "tracker_reliability": (
                "exp(-mean primary-vs-relay coordinate error / 0.04) * joint-visible "
                "fraction, then zeroed below joint-visible fraction 0.35."
            ),
            "appearance_reliability": (
                "exp(-mean temporal DINO/SigLIP cosine error / 0.25)."
            ),
            "final_reliability": (
                "tracker_reliability * appearance_reliability, zeroed below 0.10."
            ),
            "gaussian_code": (
                "MLP([query feature, relation-weighted feature, relation-weighted "
                "response, support mass]) -> 320-D mean and log-variance."
            ),
            "identity_named_partition": "L2-normalized first 128 sampled code dimensions.",
            "dynamic_named_partition": "Remaining 192 sampled code dimensions.",
            "aggregated_response_192": (
                "Model relation probability and context-validity weighted aggregate of "
                "the 192-D relation response plus projected candidate feature; its "
                "decoder reconstruction target is its own stop-gradient value."
            ),
            "shared_code_cost_proxy": (
                "teacher-relation-weighted DINO/SigLIP cosine distortion + 0.005 * "
                "query-code KL. The separate comparator uses per-point distortion and "
                "a weighted sum of per-point KL divided by the maximum support weight; "
                "therefore its saving is objective-specific and is not a measured bitrate."
            ),
        },
        "metric_classes": {
            "numerical_health": ["effective rank", "dimension std", "response spatial RMS"],
            "model_internal": ["temporal same-query retrieval", "held-group linear probes"],
            "teacher_proxy": ["relation", "visibility", "reliability", "motion coherence"],
            "independent": "only metrics computed from supplied human annotation JSONL",
        },
        "case_distributions": summarize_rows(cases),
        "query_distributions": summarize_rows(queries),
        "per_source": summarize_sources(cases, queries, dataset.source_names),
        "probes": probe_report(probes),
        "independent_annotations": score_annotations(annotations, manifest),
        "evidence_files": {
            "case_metrics": case_path,
            "query_metrics": query_path,
            "human_review_manifest": manifest_path,
            "review_image_directory": os.path.join(args.artifact_dir, "review_images"),
        },
        "conclusion_boundary": (
            "No latent name or teacher proxy is accepted as semantic correctness without "
            "the required independent evidence listed in claim_registry."
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    run_id = write_wandb(
        args,
        report,
        cases,
        queries,
        review_images,
        int(checkpoint["global_step"]),
    )
    report["wandb_run_id"] = run_id
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    finish_distributed_audit_v62()


if __name__ == "__main__":
    main()
