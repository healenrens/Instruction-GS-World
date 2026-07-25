"""Verified data-split evidence for dense visual episode manifests."""
from __future__ import annotations

import hashlib
import json
import os


EXPECTED_SPLITS = ("train", "heldseed", "heldtask")
SHA256_HEX = frozenset("0123456789abcdef")


def _set_digest(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def manifest_split_summary(manifest: dict) -> dict:
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("episode manifest has no episodes")
    split_counts = {split: 0 for split in EXPECTED_SPLITS}
    split_groups = {split: set() for split in EXPECTED_SPLITS}
    filenames = []
    cache_hashes = []
    cache_bytes = []
    filename_split_consistent = True
    for episode in episodes:
        split = str(episode["split"])
        if split not in split_counts:
            raise ValueError(f"unexpected episode split: {split}")
        filename = str(episode["filename"])
        split_counts[split] += 1
        split_groups[split].add(str(episode["sampling_group"]))
        filenames.append(filename)
        cache_hashes.append(episode.get("cache_sha256"))
        cache_bytes.append(episode.get("cache_bytes"))
        filename_split_consistent &= (
            os.path.basename(filename) == filename
            and filename.endswith(f"_{split}.pt")
        )

    contract = manifest.get("split_contract")
    if not isinstance(contract, dict):
        raise ValueError("episode manifest has no split contract")
    heldtasks = contract.get("heldtasks")
    if not isinstance(heldtasks, list) or not heldtasks:
        raise ValueError("episode split contract has no held tasks")
    heldtask_names = [str(name) for name in heldtasks]
    heldtask_contract_groups = {
        hashlib.sha256(name.encode()).hexdigest()[:16]
        for name in heldtask_names
    }
    intersections = {
        "train_heldseed": len(split_groups["train"] & split_groups["heldseed"]),
        "train_heldtask": len(split_groups["train"] & split_groups["heldtask"]),
        "heldseed_heldtask": len(
            split_groups["heldseed"] & split_groups["heldtask"]
        ),
    }
    checks = {
        "all_splits_nonempty": all(split_counts.values()),
        "train_and_heldseed_share_task_set": (
            split_groups["train"] == split_groups["heldseed"]
        ),
        "heldtask_disjoint_from_train": intersections["train_heldtask"] == 0,
        "heldtask_disjoint_from_heldseed": (
            intersections["heldseed_heldtask"] == 0
        ),
        "heldtask_groups_match_contract": (
            split_groups["heldtask"] == heldtask_contract_groups
        ),
        "split_contract_version": (
            contract.get("name") == "rt2_task_md5_v1"
            and float(contract.get("heldseed_fraction", -1.0)) == 0.2
        ),
        "unique_episode_filenames": len(filenames) == len(set(filenames)),
        "filename_split_consistent": filename_split_consistent,
        "valid_cache_sha256": all(
            isinstance(value, str)
            and len(value) == 64
            and set(value) <= SHA256_HEX
            for value in cache_hashes
        ),
        "positive_cache_bytes": all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
            for value in cache_bytes
        ),
        "unique_cache_sha256": (
            len(cache_hashes) == len(set(cache_hashes))
        ),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "split_counts": split_counts,
        "group_counts": {
            split: len(groups) for split, groups in split_groups.items()
        },
        "group_set_sha256": {
            split: _set_digest(groups) for split, groups in split_groups.items()
        },
        "group_intersections": intersections,
        "heldtask_contract_group_count": len(heldtask_contract_groups),
        "declared_cache_bytes": sum(
            value for value in cache_bytes if isinstance(value, int)
        ),
    }


def episode_file_inventory(root: str, manifest: dict) -> dict:
    expected = {
        str(episode["filename"]): int(episode["cache_bytes"])
        for episode in manifest["episodes"]
    }
    actual = {}
    nonregular = []
    with os.scandir(root) as entries:
        for entry in entries:
            if not entry.name.endswith(".pt"):
                continue
            if not entry.is_file(follow_symlinks=False):
                nonregular.append(entry.name)
                continue
            actual[entry.name] = entry.stat(follow_symlinks=False).st_size
    expected_names = set(expected)
    actual_names = set(actual)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    size_mismatches = sorted(
        name
        for name in expected_names & actual_names
        if expected[name] != actual[name]
    )
    checks = {
        "episode_file_set_matches_manifest": not missing and not extra,
        "episode_files_are_regular": not nonregular,
        "episode_file_sizes_match_manifest": not size_mismatches,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "files": len(actual),
        "bytes": sum(actual.values()),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "nonregular_count": len(nonregular),
        "size_mismatch_count": len(size_mismatches),
        "examples": {
            "missing": missing[:10],
            "extra": extra[:10],
            "nonregular": sorted(nonregular)[:10],
            "size_mismatch": size_mismatches[:10],
        },
    }


def sequence_manifest_summary(root: str) -> dict:
    manifest_path = os.path.join(root, "episode_manifest.json")
    verified_path = os.path.join(root, "episode_manifest.verified.sha256")
    with open(verified_path, encoding="utf-8") as handle:
        expected = handle.read().split()[0]
    digest = hashlib.sha256()
    with open(manifest_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError("sequence manifest differs from its verified digest")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    return {
        "root": root,
        "manifest_sha256": actual,
        "complete": bool(manifest["complete"]),
        "episodes": len(manifest["episodes"]),
        "projection_sha256": manifest["projection_sha256"],
        "split_isolation": manifest_split_summary(manifest),
        "file_inventory": episode_file_inventory(root, manifest),
    }
