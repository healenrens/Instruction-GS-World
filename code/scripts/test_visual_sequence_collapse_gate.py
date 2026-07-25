"""Remote CPU contract for the longitudinal collapse-gate report schema."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

from evaluate_visual_sequence_temporal_regions import _available_samples
from evaluate_visual_sequence_representation import _sequence_cluster_ids


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GATE_SCRIPT = os.path.join(SCRIPT_DIR, "verify_visual_sequence_collapse_gate.py")


def _evaluation_report(
    checkpoint: str,
    data: str,
    split: str,
    checkpoint_step: int,
) -> dict:
    return {
        "status": "ok",
        "checkpoint": checkpoint,
        "checkpoint_global_step": checkpoint_step,
        "checkpoint_phase": "joint",
        "checkpoint_version": 4,
        "data": data,
        "data_sha256": "a" * 64,
        "split": split,
        "amp": "bf16",
        "requested_max_items": 2,
        "available_samples": 2,
        "anchors": [3, 5, 8],
        "evaluation": {
            "samples": 2,
            "action_source": "future_conditioned_posterior_oracle",
            "deployable_prediction": False,
            "headline": {
                "change_posterior_vs_shuffled_relative": 0.10,
                "change_posterior_vs_zero_relative": 0.12,
                "static_posterior_vs_zero_relative": 0.0,
                "posterior_change_topk_iou": 0.30,
            },
            "localization": {
                "clusters": 2,
                "mean": {
                    "change_magnitude": {"posterior": 0.02},
                    "static_magnitude": {"posterior": 0.005},
                    "change_static_ratio": {"posterior": 4.0},
                },
            },
            "regions": {
                "change": {
                    "comparison": {
                        "charbonnier": {
                            "posterior_vs_shuffled": {
                                "positive_ci95_lower": True,
                            },
                            "posterior_vs_zero": {
                                "positive_ci95_lower": True,
                            },
                        },
                    },
                },
            },
        },
    }


def _write_json(path: str, value: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle)
        handle.write("\n")


def _training_log(path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for step in range(3620, 4001, 20):
            record = {
                "global_step": step,
                "rgb_future_delta_change_energy_ratio": 0.25,
                "rgb_future_delta_change_gain_over_copy": 0.04,
                "rgb_future_delta_predicted_change_rms": 0.1,
                "rgb_future_delta_target_change_rms": 0.3,
                "grad_norm": 1.0,
            }
            handle.write(json.dumps(record) + "\n")


def _gate_command(
    heldseed: str,
    heldtask: str,
    train_log: str,
    output: str,
) -> list[str]:
    return [
        sys.executable,
        GATE_SCRIPT,
        "--heldseed",
        heldseed,
        "--heldtask",
        heldtask,
        "--train_log",
        train_log,
        "--output",
        output,
        "--minimum_heldseed_samples",
        "1",
        "--minimum_heldtask_samples",
        "1",
        "--minimum_clusters",
        "1",
        "--maximum_step",
        "4000",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    class DenseDataset:
        _full_length = 17

        def __len__(self) -> int:
            return 2

    class WindowDataset:
        def __len__(self) -> int:
            return 3

    class EpisodeRecord:
        def __init__(self, episode_index: int):
            self.episode_index = episode_index

    class ClusterDataset:
        def __len__(self) -> int:
            return 3

        def _locate(self, index: int):
            return EpisodeRecord(index // 2), 0, 3

    if _available_samples(DenseDataset()) != 17:
        raise AssertionError("dense episode availability ignored full length")
    if _available_samples(WindowDataset()) != 3:
        raise AssertionError("window availability did not fall back to length")
    if _sequence_cluster_ids(ClusterDataset()).tolist() != [0, 0, 1]:
        raise AssertionError("sequence representation clusters are not episode based")

    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = os.path.join(temporary, "joint_0004000.pt")
        data = os.path.join(temporary, "data")
        os.makedirs(data)
        open(checkpoint, "wb").close()
        heldseed = os.path.join(temporary, "heldseed.json")
        heldtask = os.path.join(temporary, "heldtask.json")
        train_log = os.path.join(temporary, "train.jsonl")
        passing = os.path.join(temporary, "passing.json")
        failing = os.path.join(temporary, "failing.json")
        _write_json(
            heldseed,
            _evaluation_report(checkpoint, data, "heldseed", 4000),
        )
        _write_json(
            heldtask,
            _evaluation_report(checkpoint, data, "heldtask", 4000),
        )
        _training_log(train_log)
        subprocess.run(
            _gate_command(heldseed, heldtask, train_log, passing),
            check=True,
            stdout=subprocess.DEVNULL,
        )
        passing_report = json.load(open(passing, encoding="utf-8"))
        if passing_report["status"] != "pass":
            raise AssertionError("valid collapse-gate fixture did not pass")

        mismatched = _evaluation_report(checkpoint, data, "heldtask", 3999)
        _write_json(heldtask, mismatched)
        failed = subprocess.run(
            _gate_command(heldseed, heldtask, train_log, failing),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        failing_report = json.load(open(failing, encoding="utf-8"))
        if failed.returncode == 0 or failing_report["status"] != "fail":
            raise AssertionError("checkpoint mismatch did not fail closed")
        if failing_report["checks"]["shared_checkpoint_identity"]:
            raise AssertionError("checkpoint mismatch passed identity check")

    report = {
        "status": "ok",
        "nested_localization_clusters": True,
        "valid_contract_passes": True,
        "checkpoint_mismatch_fails_closed": True,
        "episode_backend_availability": True,
        "sequence_episode_clusters": True,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    _write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
