"""Audited cross-commit resume exceptions for Object-Region JEPA v43."""
from __future__ import annotations


_APPROVED_SOURCE_COMMITS = {
    "cea05ac01178cb650cfa4668f3c9b48e46ca007a": (
        "curriculum_ddp_identity_stability_v2"
    ),
}
_MIGRATABLE_ARGUMENTS = frozenset(("gate_report", "gate_report_sha256"))


def validate_v43_resume_commit(checkpoint: dict, args) -> frozenset[str]:
    source = str(checkpoint.get("git_commit", ""))
    current = str(getattr(args, "git_commit", ""))
    requested = str(getattr(args, "resume_compatible_git_commit", ""))
    if source == current:
        if requested:
            raise ValueError(
                "resume compatibility must be empty for a same-commit checkpoint"
            )
        args.resume_compatibility_reason = ""
        return frozenset()
    architecture = checkpoint.get("config", {}).get("architecture")
    if architecture != "object_region_memory_v1":
        raise ValueError("cross-commit resume is restricted to Object-Region JEPA v43")
    if requested != source:
        raise ValueError("resume checkpoint git commit differs")
    reason = _APPROVED_SOURCE_COMMITS.get(source)
    if reason is None:
        raise ValueError("resume checkpoint commit has no audited compatibility rule")
    args.resume_compatibility_reason = reason
    return _MIGRATABLE_ARGUMENTS
