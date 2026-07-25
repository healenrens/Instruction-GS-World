"""Remote CPU contract for the action-transfer evidence gate."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile


GATE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "verify_action_transfer_gate.py",
)


def _paired() -> dict:
    return {
        "positive_ci95_lower": True,
        "relative_improvement": 0.10,
    }


def _report(checkpoint: str, data: str, split: str) -> dict:
    primary = {
        metric: {
            "matched_effect_over_shuffled": _paired(),
            "matched_residual_over_random": _paired(),
        }
        for metric in ("feature_mse", "latent_mse", "rgb_distance")
    }
    changed = {
        metric: {
            "matched_effect_over_shuffled": _paired(),
            "matched_residual_over_random": _paired(),
        }
        for metric in (
            "change_weighted_feature_mse",
            "change_weighted_rgb_charbonnier",
        )
    }
    return {
        "status": "ok",
        "checkpoint": checkpoint,
        "checkpoint_global_step": 4000,
        "checkpoint_phase": "joint",
        "data": data,
        "data_sha256": "b" * 64,
        "split": split,
        "requested_max_items": 2,
        "available_samples": 2,
        "action_dim": 14,
        "canonical_action_dim": 6,
        "action_residual_dim": 8,
        "evaluation": {
            "samples": 2,
            "clusters": 2,
            "transfer_diagnostics": {
                "matched_same_episode_collisions": 0,
                "shuffled_same_episode_collisions": 0,
                "matched_distance_ratio": 0.5,
            },
            "comparison": primary,
            "change_weighted_comparison": changed,
        },
    }


def _write(path: str, value: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle)
        handle.write("\n")


def _command(seed: str, task: str, output: str) -> list[str]:
    return [
        sys.executable,
        GATE,
        "--heldseed",
        seed,
        "--heldtask",
        task,
        "--minimum_heldseed_samples",
        "1",
        "--minimum_heldtask_samples",
        "1",
        "--minimum_clusters",
        "1",
        "--output",
        output,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = os.path.join(temporary, "joint_0004000.pt")
        data = os.path.join(temporary, "data")
        os.makedirs(data)
        open(checkpoint, "wb").close()
        seed = os.path.join(temporary, "seed.json")
        task = os.path.join(temporary, "task.json")
        passing = os.path.join(temporary, "passing.json")
        failing = os.path.join(temporary, "failing.json")
        _write(seed, _report(checkpoint, data, "heldseed"))
        _write(task, _report(checkpoint, data, "heldtask"))
        subprocess.run(
            _command(seed, task, passing),
            check=True,
            stdout=subprocess.DEVNULL,
        )
        if json.load(open(passing, encoding="utf-8"))["status"] != "pass":
            raise AssertionError("valid transfer fixture did not pass")

        incomplete = _report(checkpoint, data, "heldtask")
        incomplete["evaluation"]["samples"] = 1
        _write(task, incomplete)
        failed = subprocess.run(
            _command(seed, task, failing),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        failing_report = json.load(open(failing, encoding="utf-8"))
        if failed.returncode == 0 or failing_report["status"] != "fail":
            raise AssertionError("incomplete transfer report did not fail closed")
        if failing_report["checks"]["heldtask_sample_coverage"]:
            raise AssertionError("incomplete transfer report passed coverage")

    report = {
        "status": "ok",
        "valid_contract_passes": True,
        "incomplete_sample_coverage_fails_closed": True,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    _write(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
