"""Weights & Biases tracking for the v48 held-object-state evaluation."""

from __future__ import annotations

import argparse
import os

from .v48_config import ARCHITECTURE, CHECKPOINT_VERSION


def add_evaluation_wandb_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wandb_mode",
        choices=("disabled", "online", "offline"),
        default="disabled",
    )
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_tags", default="")
    parser.add_argument("--wandb_run_id", default="")
    parser.add_argument("--wandb_dir", default="")
    parser.add_argument("--wandb_source_run", default="")


def validate_evaluation_wandb_arguments(args: argparse.Namespace) -> None:
    if args.wandb_mode == "disabled":
        return
    if not args.wandb_project:
        raise ValueError("v48 W&B evaluation requires --wandb_project")
    if not args.wandb_dir:
        raise ValueError("v48 W&B evaluation requires --wandb_dir")


def _read_run_id(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as handle:
        return handle.read().strip()


def _write_run_id(path: str, run_id: str) -> None:
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(run_id + "\n")
    os.replace(temporary, path)


def _numeric_fields(prefix: str, values: dict) -> dict[str, float | int]:
    return {
        f"{prefix}/{name}": value
        for name, value in values.items()
        if isinstance(value, (bool, int, float))
    }


class V48EvaluationTracker:
    def __init__(self, run, wandb_module) -> None:
        self.run = run
        self.wandb = wandb_module

    def log_length(self, split: str, chunk_length: int, result: dict) -> None:
        payload: dict[str, float | int] = {
            "evaluation/history_length": chunk_length,
            "evaluation/items": result["items"],
            "evaluation/episodes": result["episodes"],
            "evaluation/check_pass_fraction": sum(result["checks"].values())
            / len(result["checks"]),
        }
        for name, summary in result["metrics"].items():
            payload[f"metrics/{name}"] = summary["episode_balanced_mean"]
            payload[f"metrics_ci95_lower/{name}"] = summary["ci95_lower"]
            payload[f"metrics_ci95_upper/{name}"] = summary["ci95_upper"]
        for name, comparison in result["paired_comparisons"].items():
            payload.update(_numeric_fields(f"comparisons/{name}", comparison))
        for name, passed in result["checks"].items():
            payload[f"checks/{name}"] = int(passed)
        cross_episode = result["cross_episode_proxy"]
        margin = cross_episode["same_group_margin_over_different"]
        if margin is not None:
            payload["cross_episode/same_group_margin_over_different"] = margin
        for name, category in cross_episode["categories"].items():
            if category["available"]:
                payload.update(_numeric_fields(f"cross_episode/{name}", category))
        self.run.log(payload, step=chunk_length)

    def finish(self, report: dict, output_path: str) -> None:
        metrics = self.wandb.Table(
            columns=[
                "split",
                "history_length",
                "metric",
                "samples",
                "episodes",
                "sample_mean",
                "episode_balanced_mean",
                "episode_standard_error",
                "ci95_lower",
                "ci95_upper",
            ]
        )
        comparisons = self.wandb.Table(
            columns=[
                "split",
                "history_length",
                "comparison",
                "clusters",
                "absolute_improvement",
                "relative_improvement",
                "cluster_win_fraction",
                "cluster_standard_error",
                "ci95_lower",
                "ci95_upper",
                "passed",
            ]
        )
        checks = self.wandb.Table(
            columns=["split", "history_length", "check", "passed"]
        )
        cross_episode = self.wandb.Table(
            columns=[
                "split",
                "history_length",
                "category",
                "available",
                "pairs",
                "mean_set_similarity",
                "standard_error",
            ]
        )
        visualizations = self.wandb.Table(
            columns=["split", "history_length", "path", "visualization"]
        )
        summary_payload: dict[str, str | bool | float | int | None] = {
            "evaluation/status": report["status"],
            "evaluation/all_checks_passed": report["all_checks_passed"],
            "evaluation/semantic_object_correspondence_verified": report[
                "semantic_object_correspondence_verified"
            ],
            "evaluation/natural_occlusion_ground_truth_used": report[
                "natural_occlusion_ground_truth_used"
            ],
            "checkpoint/global_step": report["checkpoint_global_step"],
            "checkpoint/git_commit": report["checkpoint_git_commit"],
            "evaluation/report_path": output_path,
            "evaluation/result_transport": "wandb_summary_and_tables",
        }
        for length, result in report["evaluation_by_chunk_length"].items():
            prefix = f"held/H{length}"
            summary_payload[f"{prefix}/items"] = result["items"]
            summary_payload[f"{prefix}/episodes"] = result["episodes"]
            for name, summary in result["metrics"].items():
                metrics.add_data(
                    report["split"],
                    int(length),
                    name,
                    summary["samples"],
                    summary["episodes"],
                    summary["sample_mean"],
                    summary["episode_balanced_mean"],
                    summary["episode_standard_error"],
                    summary["ci95_lower"],
                    summary["ci95_upper"],
                )
                summary_payload.update(
                    _numeric_fields(f"{prefix}/metrics/{name}", summary)
                )
            for name, comparison in result["paired_comparisons"].items():
                comparisons.add_data(
                    report["split"],
                    int(length),
                    name,
                    comparison["clusters"],
                    comparison["absolute_improvement"],
                    comparison["relative_improvement"],
                    comparison["cluster_win_fraction"],
                    comparison["cluster_standard_error"],
                    comparison["ci95_lower"],
                    comparison["ci95_upper"],
                    comparison["positive_ci95_lower"],
                )
                summary_payload.update(
                    _numeric_fields(f"{prefix}/comparisons/{name}", comparison)
                )
            for name, passed in result["checks"].items():
                checks.add_data(report["split"], int(length), name, passed)
                summary_payload[f"{prefix}/checks/{name}"] = passed
            cross = result["cross_episode_proxy"]
            summary_payload[
                f"{prefix}/cross_episode/same_group_margin_over_different"
            ] = cross["same_group_margin_over_different"]
            for name, category in cross["categories"].items():
                cross_episode.add_data(
                    report["split"],
                    int(length),
                    name,
                    category["available"],
                    category["pairs"],
                    category.get("mean_set_similarity"),
                    category.get("standard_error"),
                )
                summary_payload.update(
                    _numeric_fields(f"{prefix}/cross_episode/{name}", category)
                )
            for path in result["visualizations"]:
                visualizations.add_data(
                    report["split"],
                    int(length),
                    path,
                    self.wandb.Image(path),
                )
        self.run.log(
            {
                "tables/metrics": metrics,
                "tables/paired_comparisons": comparisons,
                "tables/checks": checks,
                "tables/cross_episode_proxy": cross_episode,
                "tables/qualitative_visualizations": visualizations,
            }
        )
        self.run.summary.update(summary_payload)
        artifact = self.wandb.Artifact(
            name=f"v48-held-object-state-{report['split']}-{self.run.id}",
            type="evaluation",
            metadata={
                "contract": report["contract"],
                "checkpoint_version": report["checkpoint_version"],
                "checkpoint_global_step": report["checkpoint_global_step"],
                "split": report["split"],
            },
        )
        artifact.add_file(output_path, name=f"{report['split']}.json")
        self.run.log_artifact(artifact)
        self.run.finish()


def init_evaluation_tracker(
    args: argparse.Namespace,
    checkpoint: dict,
    chunk_lengths: tuple[int, ...],
) -> V48EvaluationTracker | None:
    validate_evaluation_wandb_arguments(args)
    if args.wandb_mode == "disabled":
        return None

    import wandb

    os.makedirs(args.wandb_dir, exist_ok=True)
    run_id_path = os.path.abspath(args.output) + ".wandb_run_id.txt"
    saved_run_id = _read_run_id(run_id_path)
    requested_run_id = args.wandb_run_id.strip()
    if saved_run_id and requested_run_id and saved_run_id != requested_run_id:
        raise ValueError("v48 requested W&B run id differs from saved evaluation id")
    run_id = requested_run_id or saved_run_id or wandb.util.generate_id()
    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name or None,
        group=args.wandb_group or None,
        job_type="evaluation",
        tags=tags or None,
        id=run_id,
        resume="allow" if saved_run_id else "never",
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        config={
            "contract": "v48_held_object_state_v1",
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_global_step": int(checkpoint["global_step"]),
            "checkpoint_git_commit": checkpoint.get("git_commit"),
            "source_training_run": args.wandb_source_run or None,
            "data": os.path.abspath(args.data),
            "split": args.split,
            "history_lengths": list(chunk_lengths),
            "temporal_strides": args.temporal_strides,
            "max_items": args.max_items,
            "batch": args.batch,
            "dino_frame_batch": args.dino_frame_batch,
            "amp": args.amp,
            "seed": args.seed,
            "cross_episode_pairs": args.cross_episode_pairs,
            "semantic_object_correspondence_verified": False,
            "natural_occlusion_ground_truth_used": False,
        },
    )
    if run is None:
        raise RuntimeError("wandb.init returned no v48 evaluation run")
    _write_run_id(run_id_path, run.id)
    run.define_metric("evaluation/history_length")
    run.define_metric("metrics/*", step_metric="evaluation/history_length")
    run.define_metric("metrics_ci95_lower/*", step_metric="evaluation/history_length")
    run.define_metric("metrics_ci95_upper/*", step_metric="evaluation/history_length")
    run.define_metric("comparisons/*", step_metric="evaluation/history_length")
    run.define_metric("checks/*", step_metric="evaluation/history_length")
    run.define_metric("cross_episode/*", step_metric="evaluation/history_length")
    return V48EvaluationTracker(run, wandb)
