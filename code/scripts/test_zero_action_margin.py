"""Remote CPU contract for detached zero-action relative ranking."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.zero_action_margin import (  # noqa: E402
    relative_margin_ranking,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    matched = torch.tensor([0.8, 1.1], requires_grad=True)
    reference = torch.tensor([1.0, 1.0], requires_grad=True)
    ranking = relative_margin_ranking(matched, reference, 0.05)
    expected = torch.tensor([0.0, 0.15])
    if not torch.allclose(ranking, expected):
        raise AssertionError("relative zero-action ranking is incorrect")
    ranking.sum().backward()
    if matched.grad is None or float(matched.grad[1]) <= 0.0:
        raise AssertionError("matched prediction did not receive a gradient")
    if reference.grad is not None:
        raise AssertionError("zero-action reference retained a gradient")
    report = {
        "status": "ok",
        "ranking": ranking.detach().tolist(),
        "matched_gradient": matched.grad.tolist(),
        "reference_gradient": None,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
