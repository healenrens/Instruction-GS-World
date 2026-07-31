"""CLI and construction helpers for pair and visual-sequence datasets."""

from __future__ import annotations

from .dynamic_dual_horizon_dataset import (
    DYNAMIC_DUAL_HORIZON_CONTRACT,
    DynamicDualHorizonEpisodeDataset,
)
from .pair_dataset import CausalPairFeatureDataset
from .sequence_dataset import CausalVisualSequenceDataset


def add_dataset_arguments(parser) -> None:
    parser.add_argument(
        "--data_format",
        choices=("pair", "sequence"),
        default="pair",
    )
    parser.add_argument("--history_frames", type=int, default=4)
    parser.add_argument("--future_frames", type=int, default=4)
    parser.add_argument("--sequence_anchors", default="3,5,8")
    parser.add_argument(
        "--temporal_contract",
        choices=("legacy_window_v1", DYNAMIC_DUAL_HORIZON_CONTRACT),
        default="legacy_window_v1",
    )
    parser.add_argument("--history_frames_min", type=int, default=1)
    parser.add_argument("--history_frames_max", type=int, default=4)
    parser.add_argument("--history_span_frames", default="15,30,45")
    parser.add_argument("--short_horizon_frames", type=int, default=30)
    parser.add_argument("--goal_query_seconds", type=float, default=6.0)
    parser.add_argument("--goal_tail_guard_frames", type=int, default=0)
    parser.add_argument("--goal_probe_frames", type=int, default=3)
    parser.add_argument("--goal_stability_threshold", type=float, default=0.05)
    parser.add_argument("--teacher_sidecar", default="")
    parser.add_argument("--feature_source", choices=("cached", "jit"), default="cached")
    parser.add_argument("--jit_dino_batch", type=int, default=4)


def build_training_dataset(
    args,
    split: str,
    language_enabled: bool,
    rgb_enabled: bool,
):
    if args.teacher_sidecar and args.data_format != "sequence":
        raise ValueError("teacher sidecars require sequence data")
    if args.data_format == "pair":
        if args.feature_source != "cached":
            raise ValueError("pair data only supports cached features")
        if not args.dino:
            raise ValueError("pair data requires --dino")
        return CausalPairFeatureDataset(
            args.data,
            args.dino,
            split,
            max_items=args.max_train_items,
            condition_cache=args.condition_cache if language_enabled else "",
            load_rgb=rgb_enabled,
            rgb_short_side=args.rgb_short_side,
            rgb_pad_multiple=args.rgb_pad_multiple,
        )
    if language_enabled or args.condition_cache:
        raise ValueError(
            "visual sequence core forbids condition caches and language inputs"
        )
    if args.temporal_contract == DYNAMIC_DUAL_HORIZON_CONTRACT:
        if rgb_enabled:
            raise ValueError("dynamic dual-horizon core forbids RGB supervision")
        return DynamicDualHorizonEpisodeDataset(
            args.data,
            split,
            history_frames_min=args.history_frames_min,
            history_frames_max=args.history_frames_max,
            history_span_frames=args.history_span_frames,
            short_horizon_frames=args.short_horizon_frames,
            goal_query_seconds=args.goal_query_seconds,
            goal_tail_guard_frames=args.goal_tail_guard_frames,
            goal_probe_frames=args.goal_probe_frames,
            max_items=args.max_train_items,
            teacher_sidecar=args.teacher_sidecar,
            feature_source=args.feature_source,
        )
    return CausalVisualSequenceDataset(
        args.data,
        split,
        history_frames=args.history_frames,
        future_frames=args.future_frames,
        anchors=args.sequence_anchors,
        max_items=args.max_train_items,
        load_rgb=rgb_enabled,
        rgb_short_side=args.rgb_short_side,
        rgb_pad_multiple=args.rgb_pad_multiple,
        teacher_sidecar=args.teacher_sidecar,
        feature_source=args.feature_source,
    )
