"""Verify that a derived checkpoint preserves every model tensor exactly."""
from __future__ import annotations

import argparse
import json
import os

import torch


def _load(path: str) -> dict:
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--derived", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = _load(args.source)
    derived = _load(args.derived)
    source_model = source["model"]
    derived_model = derived["model"]
    source_keys = set(source_model)
    derived_keys = set(derived_model)
    missing = sorted(source_keys - derived_keys)
    unexpected = sorted(derived_keys - source_keys)
    changed = []
    for name in sorted(source_keys & derived_keys):
        left = source_model[name]
        right = derived_model[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            changed.append(
                {
                    "name": name,
                    "source_shape": list(left.shape),
                    "derived_shape": list(right.shape),
                    "source_dtype": str(left.dtype),
                    "derived_dtype": str(right.dtype),
                }
            )
        elif not torch.equal(left, right):
            changed.append(
                {
                    "name": name,
                    "max_abs_difference": float(
                        (left.float() - right.float()).abs().max()
                    ),
                }
            )

    report = {
        "status": (
            "ok"
            if not missing and not unexpected and not changed
            else "mismatch"
        ),
        "source": os.path.abspath(args.source),
        "derived": os.path.abspath(args.derived),
        "source_checkpoint_version": source.get("checkpoint_version"),
        "derived_checkpoint_version": derived.get("checkpoint_version"),
        "model_tensor_count": len(source_model),
        "missing_model_tensors": missing,
        "unexpected_model_tensors": unexpected,
        "changed_model_tensors": changed,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "ok":
        raise AssertionError("derived checkpoint changed model tensors")


if __name__ == "__main__":
    main()
