#!/usr/bin/env python3
"""Audit every V69 entry/teacher and original language mapping without modifying data."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--stage2_manifest", "--manifest", dest="manifest")
    inputs.add_argument("--stage2_run", help="Stage 2 output directory or run.json; reads args.manifest")
    parser.add_argument("--output", required=True, help="Audit JSON; sibling CSV is written by default")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--sidecar_template", default=None)
    parser.add_argument("--language_sidecar", default=None)
    parser.add_argument("--verified_trace", default=None, help="Authoritative server JSONL; report trace coverage without reloading teachers")
    parser.add_argument("--source_root", action="append", default=[], help="SOURCE=/absolute/native/root; repeatable")
    args = parser.parse_args()
    from igsw.adaptive_gaussian_wm.language_manifest_v70 import (
        inspect_language_sources_v70, resolve_stage2_manifest_v70, write_language_audit_v70,
    )
    roots = {}
    for value in args.source_root:
        source, root = value.split("=", 1)
        roots.setdefault(source, []).append(root)
    manifest = resolve_stage2_manifest_v70(args.manifest, args.stage2_run)
    print(json.dumps({"resolved_manifest": manifest, "stage2_run": args.stage2_run}), flush=True)
    def progress(done, total):
        print(f"[language-audit-v70] entries={done}/{total}", file=sys.stderr, flush=True)
    report = inspect_language_sources_v70(manifest, roots, args.language_sidecar, progress, args.verified_trace)
    report["stage2_run"] = args.stage2_run
    write_language_audit_v70(report, args.output, args.csv, args.sidecar_template)
    print(json.dumps({key: report[key] for key in ("stage2_manifest", "entries_audited", "unique_episodes",
                                                "partition_counts", "teacher_files_read", "totals", "per_source")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
