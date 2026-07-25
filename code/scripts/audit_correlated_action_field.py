"""Audit causal, effect, coverage, rendering, and checkpoint gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.latent_particle_wm.action_field import (  # noqa: E402
    ActionFieldConfig,
    CorrelatedActionField,
)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_check(
    checks: list[dict],
    name: str,
    passed: bool,
    actual,
    requirement: str,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "actual": actual,
            "requirement": requirement,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--heldseed", required=True)
    parser.add_argument("--heldtask", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--all_flow_tolerance_px", type=float, default=0.025)
    parser.add_argument("--all_xyz_tolerance_cm", type=float, default=0.005)
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ActionFieldConfig(**checkpoint["config"])
    model = CorrelatedActionField(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    calibration = checkpoint.get("inference_calibration", {})
    checks: list[dict] = []
    add_check(
        checks,
        "checkpoint.posterior_xy_scale",
        config.posterior_xy_scale == 0.25,
        config.posterior_xy_scale,
        "equal to 0.25",
    )
    add_check(
        checks,
        "checkpoint.posterior_depth_scale",
        config.posterior_depth_scale == 0.0,
        config.posterior_depth_scale,
        "equal to 0.0",
    )
    add_check(
        checks,
        "checkpoint.prior_scale",
        calibration.get("prior_xy_scale") == 1.0
        and calibration.get("prior_depth_scale") == 1.0,
        {
            "xy": calibration.get("prior_xy_scale"),
            "depth": calibration.get("prior_depth_scale"),
        },
        "both equal to 1.0",
    )
    source_path = calibration.get("source_checkpoint", "")
    source_hash = file_sha256(source_path)
    add_check(
        checks,
        "checkpoint.source_sha256",
        source_hash == calibration.get("source_sha256"),
        source_hash,
        "match inference_calibration.source_sha256",
    )
    add_check(
        checks,
        "checkpoint.completed_steps",
        int(checkpoint["posterior_step"]) == 1600
        and int(checkpoint["prior_step"]) == 800,
        {
            "posterior": int(checkpoint["posterior_step"]),
            "prior": int(checkpoint["prior_step"]),
        },
        "posterior=1600 and prior=800",
    )

    reports = {
        "heldseed": json.load(open(args.heldseed)),
        "heldtask": json.load(open(args.heldtask)),
    }
    for split, report in reports.items():
        metrics = report["metrics"]
        prefix = f"{split}."
        add_check(checks, prefix + "status", report["status"] == "ok", report["status"], "ok")
        add_check(checks, prefix + "count", report["count"] > 0, report["count"], "> 0")
        add_check(checks, prefix + "samples", report["samples"] >= 16, report["samples"], ">= 16")
        add_check(
            checks,
            prefix + "checkpoint",
            os.path.abspath(report["checkpoint"]) == checkpoint_path,
            os.path.abspath(report["checkpoint"]),
            checkpoint_path,
        )
        add_check(
            checks,
            prefix + "causal_prior",
            report["causal"]["prior_context_future_swap_max_abs_difference"] == 0.0,
            report["causal"]["prior_context_future_swap_max_abs_difference"],
            "equal to 0.0",
        )
        add_check(
            checks,
            prefix + "posterior_future_sensitive",
            report["causal"]["posterior_future_swap_max_abs_difference"] > 1e-6,
            report["causal"]["posterior_future_swap_max_abs_difference"],
            "> 1e-6",
        )
        add_check(
            checks,
            prefix + "coverage_monotonic",
            report["coverage_monotonic_nonincreasing"],
            report["coverage_monotonic_nonincreasing"],
            "true",
        )

        comparisons = (
            ("posterior/flow_epe_mover_px_point", "deterministic/flow_epe_mover_px_point"),
            ("posterior/xyz_epe_mover_cm_point", "deterministic/xyz_epe_mover_cm_point"),
            ("prior_best_16/flow_epe_mover_px_point", "deterministic/flow_epe_mover_px_point"),
            ("prior_best_16/xyz_epe_mover_cm_point", "deterministic/xyz_epe_mover_cm_point"),
            ("prior_best_16/flow_epe_mover_px_point", "zero/flow_epe_mover_px_point"),
            ("prior_best_16/xyz_epe_mover_cm_point", "zero/xyz_epe_mover_cm_point"),
        )
        for left, right in comparisons:
            add_check(
                checks,
                prefix + left + "<" + right,
                metrics[left] < metrics[right],
                {"left": metrics[left], "right": metrics[right]},
                "left < right",
            )
        add_check(
            checks,
            prefix + "all_flow_nondegradation",
            metrics["posterior/flow_epe_px_point"]
            <= metrics["deterministic/flow_epe_px_point"] + args.all_flow_tolerance_px,
            {
                "posterior": metrics["posterior/flow_epe_px_point"],
                "deterministic": metrics["deterministic/flow_epe_px_point"],
            },
            f"posterior <= deterministic + {args.all_flow_tolerance_px}",
        )
        add_check(
            checks,
            prefix + "all_xyz_nondegradation",
            metrics["posterior/xyz_epe_cm_point"]
            <= metrics["deterministic/xyz_epe_cm_point"] + args.all_xyz_tolerance_cm,
            {
                "posterior": metrics["posterior/xyz_epe_cm_point"],
                "deterministic": metrics["deterministic/xyz_epe_cm_point"],
            },
            f"posterior <= deterministic + {args.all_xyz_tolerance_cm}",
        )
        add_check(
            checks,
            prefix + "prior_diversity",
            metrics["prior/diversity_mover_px_point"] > 0.1,
            metrics["prior/diversity_mover_px_point"],
            "> 0.1 px",
        )
        add_check(
            checks,
            prefix + "rgb_psnr",
            metrics["posterior/rgb_psnr_db"] > metrics["deterministic/rgb_psnr_db"],
            {
                "posterior": metrics["posterior/rgb_psnr_db"],
                "deterministic": metrics["deterministic/rgb_psnr_db"],
            },
            "posterior > deterministic",
        )
        add_check(
            checks,
            prefix + "dino_cosine",
            metrics["posterior/dino_cosine"] > metrics["deterministic/dino_cosine"],
            {
                "posterior": metrics["posterior/dino_cosine"],
                "deterministic": metrics["deterministic/dino_cosine"],
            },
            "posterior > deterministic",
        )

    result = {
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "calibration": calibration,
        "checks": checks,
        "failed": [item["name"] for item in checks if not item["passed"]],
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
