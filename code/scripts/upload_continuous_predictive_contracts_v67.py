#!/usr/bin/env python3
"""Upload an already-computed v67 contract audit to W&B."""

from __future__ import annotations

import argparse
import json
import os

from audit_continuous_predictive_contracts_v67 import write_wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--wandb_mode", choices=("online", "offline"), default="online")
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", required=True)
    parser.add_argument(
        "--wandb_group", default="continuous-predictive-object-field-v67-contract-audit"
    )
    parser.add_argument("--wandb_dir", required=True)
    return parser.parse_args()


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def review_images_from_manifest(rows):
    images = []
    seen = set()
    for row in rows:
        image_path = row["image_path"]
        if image_path in seen:
            continue
        seen.add(image_path)
        if row["type"] == "visibility":
            description = (
                "ring color identifies candidate ID; center dot is green=final visible, "
                "yellow=tracker visible but feature-invalid, red=tracker invisible, "
                "magenta=out of bounds; all eight frames are shown"
            )
        else:
            description = (
                f"query={row['query_candidate_index']} candidate={row['candidate_index']} "
                f"teacher_relation={row['teacher_probability']:.3f} "
                f"student_relation={row['student_probability']:.3f}"
            )
        images.append(
            {
                "type": row["type"],
                "source": row["source"],
                "dataset_index": row["dataset_index"],
                "image_path": image_path,
                "description": description,
            }
        )
    return images


def main():
    args = parse_args()
    args.report = os.path.abspath(args.report)
    args.output = args.report
    args.artifact_dir = os.path.abspath(args.artifact_dir)
    args.wandb_dir = os.path.abspath(args.wandb_dir)
    with open(args.report, encoding="utf-8") as handle:
        report = json.load(handle)

    evidence = report["evidence_files"]
    cases = read_jsonl(evidence["case_metrics"])
    queries = read_jsonl(evidence["query_metrics"])
    manifest = read_jsonl(evidence["human_review_manifest"])
    review_images = review_images_from_manifest(manifest)

    args.checkpoint = report["checkpoint"]
    args.source_revision = report["evaluation_source_revision"]
    args.items_per_source = report["items_per_source"]
    review_case_count = len(
        {(row["source"], row["dataset_index"]) for row in manifest}
    )
    args.review_cases_per_source = review_case_count // len(report["source_names"])
    args.held_group_stride = int(report["partition"].split(":", 1)[1])
    args.motion_sigmas = ",".join(str(value) for value in report["motion_sigmas"])
    args.annotation_jsonl = ""

    run_id = write_wandb(
        args,
        report,
        cases,
        queries,
        review_images,
        int(report["checkpoint_step"]),
    )
    report["wandb_run_id"] = run_id
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "report": args.report,
                "case_count": len(cases),
                "query_count": len(queries),
                "review_image_count": len(review_images),
                "wandb_run_id": run_id,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
