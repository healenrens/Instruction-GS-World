"""Create one compact truth surface for the latent particle probe experiments."""
from __future__ import annotations

import argparse
import json
import os


RUNS = {
    "deterministic": "deterministic",
    "independent_local_two_stage": "local_twostage",
    "global_mixture": "global_mixture_twostage",
    "global_mixture_aligned": "global_mixture_aligned_twostage",
    "global_flow_aligned": "global_flow_aligned_twostage",
    "global_mixture_32d": "global_mixture32_twostage",
}

METRICS = (
    "zero_motion/flow_epe_mover_px",
    "posterior/flow_epe_mover_px",
    "prior_mean/flow_epe_mover_px",
    "prior_best/flow_epe_mover_px",
    "posterior/flow_dcos_mover",
    "prior_mean/flow_dcos_mover",
    "posterior/magnitude_ratio",
    "prior_mean/magnitude_ratio",
    "posterior/xyz_epe_mover_cm",
    "prior_mean/xyz_epe_mover_cm",
    "prior_best/xyz_epe_mover_cm",
    "prior_samples/diversity_mover_px",
    "prior_samples/uncertainty_error_corr",
    "posterior/local_pair_cm",
    "prior_mean/local_pair_cm",
)


def selected_metrics(metrics: dict) -> dict:
    return {key: metrics.get(key) for key in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = {
        "status": "ok",
        "root": os.path.abspath(args.root),
        "cache_audit": json.load(open(os.path.join(args.root, "cache_audit.json"))),
        "runs": {},
        "coverage": {},
        "latent_transfer": {},
        "conclusions": {
            "joint_cvae": "posterior collapse; latent samples do not add meaningful coverage",
            "independent_particle_latent": "best posterior reconstruction but incoherent/unfit prior",
            "mixture_prior": "alignment helps transfer but discrete components remain under-used",
            "conditional_flow_prior": "best unseen-task best-of-N coverage; depth and calibration remain open",
        },
    }
    for label, directory in RUNS.items():
        run = json.load(open(os.path.join(args.root, directory, "metrics.json")))
        result["runs"][label] = {
            "kind": run["kind"],
            "config": run["config"],
            "parameter_count": run["parameter_count"],
            "elapsed_seconds": run["elapsed_seconds"],
            "splits": {
                split: {
                    "all": selected_metrics(run["evaluation"][split]["all"]),
                    "invariants": run["evaluation"][split]["invariants"],
                    "horizons": {
                        horizon: selected_metrics(values)
                        for horizon, values in run["evaluation"][split].items()
                        if horizon.startswith("horizon_")
                    },
                }
                for split in ("train", "heldseed", "heldtask")
            },
        }
    for label, directory in (
        ("global_mixture_aligned", "global_mixture_aligned_twostage"),
        ("global_flow_aligned", "global_flow_aligned_twostage"),
    ):
        result["coverage"][label] = json.load(
            open(os.path.join(args.root, directory, "prior_coverage.json"))
        )
    for label, directory in (
        ("global_mixture", "global_mixture_twostage"),
        ("global_mixture_aligned", "global_mixture_aligned_twostage"),
        ("global_flow_aligned", "global_flow_aligned_twostage"),
    ):
        result["latent_transfer"][label] = json.load(
            open(os.path.join(args.root, directory, "latent_analysis.json"))
        )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps({"status": "ok", "out": os.path.abspath(args.out)}, indent=2))


if __name__ == "__main__":
    main()
