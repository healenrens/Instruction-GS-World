"""CLI and construction helpers for pair and visual-sequence datasets."""

from __future__ import annotations

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
