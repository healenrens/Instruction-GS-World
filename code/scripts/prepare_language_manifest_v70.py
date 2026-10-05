#!/usr/bin/env python3
"""Prepare V70 language windows after a complete read-only V69 language audit."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--stage2_manifest", "--manifest", dest="manifest")
    inputs.add_argument("--stage2_run", help="Output directory or run.json; reads args.manifest")
    parser.add_argument("--teacher_checkpoint", required=True, help="Fixed teacher snapshot, not a moving last.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", default=None, help="Previously completed full-entry audit JSON")
    parser.add_argument("--verified_trace", default=None, help="Authoritative full server audit JSONL; no native fallback or teacher reload")
    parser.add_argument("--language_sidecar", default=None)
    parser.add_argument("--source_root", action="append", default=[], help="SOURCE=/absolute/native/root; repeatable")
    parser.add_argument("--max_windows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17, help="Fixed held episode diagnostic/test split seed")
    parser.add_argument("--tokenizer_path", default=None, help="Local Qwen tokenizer; never downloaded")
    parser.add_argument("--instruction_tokens", type=int, default=512, help="Exclude over-budget instructions, never truncate")
    args = parser.parse_args()
    from igsw.adaptive_gaussian_wm.language_manifest_v70 import (
        inspect_language_sources_v70, prepare_language_manifest_v70,
        resolve_stage2_manifest_v70, write_language_audit_v70,
    )
    roots = {}
    for value in args.source_root:
        source, root = value.split("=", 1)
        roots.setdefault(source, []).append(root)
    manifest = resolve_stage2_manifest_v70(args.manifest, args.stage2_run)
    audit = args.audit
    if audit is None and args.verified_trace is None:
        report = inspect_language_sources_v70(manifest, roots, args.language_sidecar)
        audit = str(Path(args.output).with_suffix(".audit.json"))
        write_language_audit_v70(report, audit)
    result = prepare_language_manifest_v70(manifest, args.teacher_checkpoint, args.output,
                                          roots, args.language_sidecar, audit, args.max_windows, args.seed,
                                          args.tokenizer_path, args.instruction_tokens, verified_trace=args.verified_trace)
    print(json.dumps({"manifest": str(Path(args.output).resolve()), "stage2_manifest": manifest,
                      "input_clips": result["report"]["input_clips"], "windows": len(result["entries"]),
                      "excluded": len(result["report"]["excluded"]), "audit": audit}), flush=True)


if __name__ == "__main__":
    main()
