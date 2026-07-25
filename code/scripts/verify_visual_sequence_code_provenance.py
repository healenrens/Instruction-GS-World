"""Verify a launch snapshot with explicit evaluation-only code exceptions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXCEPTION_KEYS = {
    "schema_version",
    "scope",
    "root",
    "training_manifest",
    "runtime_manifest",
    "evaluation_manifest",
    "exclusions",
}
EXCLUSION_KEYS = {
    "path",
    "role",
    "launch_sha256",
    "current_sha256",
    "reason",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2 or SHA256.fullmatch(parts[0]) is None:
            raise ValueError(f"invalid manifest record {path}:{line_number}")
        target = parts[1].lstrip(" *")
        if os.path.isabs(target) or target in records:
            raise ValueError(f"invalid manifest target {path}:{line_number}: {target}")
        records[target] = parts[0]
    if not records:
        raise ValueError(f"empty manifest: {path}")
    return records


def target_path(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if os.path.commonpath((str(root), str(target))) != str(root):
        raise ValueError(f"manifest target escapes root: {relative}")
    return target


def live_state(root: Path, records: dict[str, str]) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    mismatched: list[str] = []
    for relative, expected in records.items():
        target = target_path(root, relative)
        if not target.is_file():
            missing.append(relative)
        elif file_sha256(target) != expected:
            mismatched.append(relative)
    return sorted(missing), sorted(mismatched)


def absolute_file(path: str) -> Path:
    result = Path(path).resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def verify(args: argparse.Namespace) -> dict:
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    training_path = absolute_file(args.training_manifest)
    runtime_path = absolute_file(args.runtime_manifest)
    evaluation_path = absolute_file(args.evaluation_manifest)
    exception_path = absolute_file(args.exception)
    exception = json.loads(exception_path.read_text())
    if not isinstance(exception, dict) or set(exception) != EXCEPTION_KEYS:
        raise ValueError("code exception has an unexpected schema")
    if exception["schema_version"] != 1:
        raise ValueError("unsupported code exception schema")
    if exception["scope"] != "post_launch_evaluation_only_change":
        raise ValueError("invalid code exception scope")
    expected_paths = {
        "root": str(root),
        "training_manifest": str(training_path),
        "runtime_manifest": str(runtime_path),
        "evaluation_manifest": str(evaluation_path),
    }
    for name, expected in expected_paths.items():
        if os.path.abspath(exception[name]) != expected:
            raise ValueError(f"code exception {name} differs from invocation")

    training = load_manifest(training_path)
    runtime = load_manifest(runtime_path)
    evaluation = load_manifest(evaluation_path)
    exclusions = exception["exclusions"]
    if not isinstance(exclusions, list) or not exclusions:
        raise ValueError("at least one explicit exclusion is required")
    exclusion_records: dict[str, dict] = {}
    for item in exclusions:
        if not isinstance(item, dict) or set(item) != EXCLUSION_KEYS:
            raise ValueError("code exclusion has an unexpected schema")
        relative = item["path"]
        if relative in exclusion_records or os.path.isabs(relative):
            raise ValueError(f"invalid duplicate exclusion: {relative}")
        if item["role"] != "evaluation_only" or not item["reason"].strip():
            raise ValueError(f"invalid exclusion role or reason: {relative}")
        exclusion_records[relative] = item

    expected_runtime = {
        relative: digest
        for relative, digest in training.items()
        if relative not in exclusion_records
    }
    if runtime != expected_runtime:
        raise ValueError("runtime manifest is not training manifest minus exclusions")

    runtime_missing, runtime_mismatched = live_state(root, runtime)
    evaluation_missing, evaluation_mismatched = live_state(root, evaluation)
    training_missing, training_mismatched = live_state(root, training)
    if runtime_missing or runtime_mismatched:
        raise ValueError("runtime-compatible training files differ from launch snapshot")
    if evaluation_missing or evaluation_mismatched:
        raise ValueError("evaluation files differ from evaluation manifest")
    if training_missing or set(training_mismatched) != set(exclusion_records):
        raise ValueError("training snapshot drift does not equal explicit exclusions")

    for relative, item in exclusion_records.items():
        if relative not in training or relative not in evaluation:
            raise ValueError(f"excluded file lacks dual-manifest coverage: {relative}")
        current = file_sha256(target_path(root, relative))
        if item["launch_sha256"] != training[relative]:
            raise ValueError(f"excluded launch digest differs: {relative}")
        if item["current_sha256"] != evaluation[relative] or current != evaluation[relative]:
            raise ValueError(f"excluded current digest differs: {relative}")
        if item["launch_sha256"] == item["current_sha256"]:
            raise ValueError(f"excluded file did not change: {relative}")

    return {
        "status": "pass",
        "scope": exception["scope"],
        "root": str(root),
        "training_manifest": str(training_path),
        "training_records": len(training),
        "runtime_manifest": str(runtime_path),
        "runtime_records": len(runtime),
        "evaluation_manifest": str(evaluation_path),
        "evaluation_records": len(evaluation),
        "excluded_paths": sorted(exclusion_records),
        "training_live_mismatches": training_mismatched,
    }


def write_report(path: str, report: dict) -> None:
    output = Path(path).resolve()
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output.exists():
        if output.read_text() != serialized:
            raise FileExistsError(f"refusing to replace differing report: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    temporary.write_text(serialized)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--training_manifest", required=True)
    parser.add_argument("--runtime_manifest", required=True)
    parser.add_argument("--evaluation_manifest", required=True)
    parser.add_argument("--exception", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = verify(args)
    if args.output:
        write_report(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
