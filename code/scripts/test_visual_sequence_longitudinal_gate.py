"""CPU contract tests for the 4k/8k/12k stability gate."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(SCRIPT_DIR, "verify_visual_sequence_longitudinal_gate.py")
STEPS = (4000, 8000, 12000)
SPLITS = ("heldseed", "heldtask")


def write_json(path: str, value: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle)
        handle.write("\n")


def collapse_report(
    checkpoint: str,
    data: str,
    step: int,
    signal: float,
    data_sha256: str = "a" * 64,
) -> dict:
    return {
        "status": "pass",
        "thresholds": {"maximum_step": step},
        "training": {"first_step": max(step - 380, 1), "last_step": step},
        "evaluation": {
            split: {
                "checkpoint": checkpoint,
                "data": data,
                "data_sha256": data_sha256,
                "change_vs_shuffled": signal,
                "change_vs_zero": signal * 1.1,
            }
            for split in SPLITS
        },
    }


def run_gate(reports: dict[int, str], output: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            GATE,
            "--step4000",
            reports[4000],
            "--step8000",
            reports[8000],
            "--step12000",
            reports[12000],
            "--output",
            output,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def assert_fails(reports: dict[int, str], output: str, check: str) -> None:
    result = run_gate(reports, output)
    report = json.load(open(output, encoding="utf-8"))
    if result.returncode == 0 or report["status"] != "fail":
        raise AssertionError(f"invalid longitudinal fixture passed: {check}")
    if report["checks"].get(check, True):
        raise AssertionError(f"expected failed longitudinal check: {check}")


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        data = os.path.join(temporary, "data")
        os.makedirs(data)
        checkpoints = {}
        reports = {}
        signals = {4000: 0.10, 8000: 0.12, 12000: 0.09}
        for step in STEPS:
            checkpoint = os.path.join(temporary, f"joint_{step:07d}.pt")
            open(checkpoint, "wb").close()
            path = os.path.join(temporary, f"step_{step}.json")
            write_json(path, collapse_report(checkpoint, data, step, signals[step]))
            checkpoints[step] = checkpoint
            reports[step] = path

        passing = os.path.join(temporary, "passing.json")
        if run_gate(reports, passing).returncode != 0:
            raise AssertionError("valid longitudinal fixture did not pass")

        write_json(
            reports[12000],
            collapse_report(checkpoints[12000], data, 12000, 0.05),
        )
        assert_fails(
            reports,
            os.path.join(temporary, "collapse.json"),
            "heldseed_change_vs_shuffled_retained",
        )
        write_json(
            reports[12000],
            collapse_report(checkpoints[12000], data, 12000, signals[12000]),
        )

        wrong_checkpoint = os.path.join(temporary, "wrong_8000.pt")
        open(wrong_checkpoint, "wb").close()
        write_json(
            reports[8000],
            collapse_report(wrong_checkpoint, data, 8000, signals[8000]),
        )
        assert_fails(
            reports,
            os.path.join(temporary, "checkpoint.json"),
            "step8000_checkpoint_identity",
        )
        write_json(
            reports[8000],
            collapse_report(checkpoints[8000], data, 8000, signals[8000]),
        )

        write_json(
            reports[8000],
            collapse_report(
                checkpoints[8000], data, 8000, signals[8000], "b" * 64
            ),
        )
        assert_fails(
            reports,
            os.path.join(temporary, "data.json"),
            "shared_data_identity",
        )

    print(json.dumps({
        "status": "ok",
        "valid_contract_passes": True,
        "final_collapse_fails": True,
        "checkpoint_identity_fails": True,
        "data_identity_fails": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
