"""Verify that Prior-only training changes only declared active parameters."""
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
    parser.add_argument("--trained", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = _load(args.source)
    trained = _load(args.trained)
    active = set(trained.get("active_model_parameters", ()))
    if not active:
        raise ValueError("trained checkpoint does not declare active parameters")
    source_model = source["model"]
    trained_model = trained["model"]
    source_keys = set(source_model)
    trained_keys = set(trained_model)
    missing = sorted(source_keys - trained_keys)
    unexpected = sorted(trained_keys - source_keys)
    changed = []
    changed_outside_scope = []
    for name in sorted(source_keys & trained_keys):
        left = source_model[name]
        right = trained_model[name]
        differs = (
            left.shape != right.shape
            or left.dtype != right.dtype
            or not torch.equal(left, right)
        )
        if differs:
            changed.append(name)
            if name not in active:
                changed_outside_scope.append(name)

    report = {
        "status": (
            "ok"
            if (
                not missing
                and not unexpected
                and changed
                and not changed_outside_scope
            )
            else "mismatch"
        ),
        "source": os.path.abspath(args.source),
        "trained": os.path.abspath(args.trained),
        "declared_active_parameter_count": len(active),
        "changed_parameter_count": len(changed),
        "changed_parameters": changed,
        "changed_outside_scope": changed_outside_scope,
        "missing_model_tensors": missing,
        "unexpected_model_tensors": unexpected,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "ok":
        raise AssertionError("Prior update scope contract failed")


if __name__ == "__main__":
    main()
