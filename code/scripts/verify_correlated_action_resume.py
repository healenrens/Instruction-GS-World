"""Verify that an interrupted DDP run reproduces an uninterrupted checkpoint."""
from __future__ import annotations

import argparse
import json
import os

import torch


def compare_values(left, right, path: str, differences: list[dict]) -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            differences.append({"path": path, "reason": "tensor type mismatch"})
            return
        if left.shape != right.shape or left.dtype != right.dtype:
            differences.append(
                {
                    "path": path,
                    "reason": "tensor metadata mismatch",
                    "left_shape": list(left.shape),
                    "right_shape": list(right.shape),
                    "left_dtype": str(left.dtype),
                    "right_dtype": str(right.dtype),
                }
            )
            return
        if not torch.equal(left, right):
            maximum = (
                float((left.float() - right.float()).abs().max())
                if left.numel()
                else 0.0
            )
            differences.append(
                {
                    "path": path,
                    "reason": "tensor values differ",
                    "max_abs_difference": maximum,
                }
            )
        return

    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            differences.append({"path": path, "reason": "mapping type mismatch"})
            return
        if left.keys() != right.keys():
            differences.append(
                {
                    "path": path,
                    "reason": "mapping keys differ",
                    "left_only": sorted(set(left) - set(right)),
                    "right_only": sorted(set(right) - set(left)),
                }
            )
            return
        for key in left:
            compare_values(left[key], right[key], f"{path}.{key}", differences)
        return

    left_is_sequence = isinstance(left, (list, tuple))
    right_is_sequence = isinstance(right, (list, tuple))
    if left_is_sequence or right_is_sequence:
        left_length = len(left) if left_is_sequence else None
        right_length = len(right) if right_is_sequence else None
        if type(left) is not type(right) or left_length != right_length:
            differences.append(
                {
                    "path": path,
                    "reason": "sequence metadata mismatch",
                    "left_type": type(left).__name__,
                    "right_type": type(right).__name__,
                    "left_length": left_length,
                    "right_length": right_length,
                }
            )
            return
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            compare_values(left_item, right_item, f"{path}[{index}]", differences)
        return

    if left != right:
        differences.append(
            {
                "path": path,
                "reason": "values differ",
                "left": left,
                "right": right,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uninterrupted", required=True)
    parser.add_argument("--resumed", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    uninterrupted = torch.load(
        args.uninterrupted,
        map_location="cpu",
        weights_only=False,
    )
    resumed = torch.load(args.resumed, map_location="cpu", weights_only=False)
    differences: list[dict] = []
    for field in (
        "checkpoint_version",
        "config",
        "phase",
        "posterior_step",
        "prior_step",
        "world_size",
        "model",
        "optimizer",
        "scheduler",
        "rng_states",
    ):
        compare_values(uninterrupted[field], resumed[field], field, differences)

    result = {
        "status": "pass" if not differences else "fail",
        "uninterrupted": os.path.abspath(args.uninterrupted),
        "resumed": os.path.abspath(args.resumed),
        "checked_fields": [
            "model",
            "optimizer",
            "scheduler",
            "rng_states",
            "steps",
            "config",
            "world_size",
        ],
        "difference_count": len(differences),
        "differences": differences[:100],
    }
    output = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = f"{output}.tmp.{os.getpid()}"
    with open(temporary, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    os.replace(temporary, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
