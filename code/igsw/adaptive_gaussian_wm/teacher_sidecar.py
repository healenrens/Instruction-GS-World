"""Versioned, supervision-only sidecars aligned to visual episode caches."""
from __future__ import annotations

import hashlib
import json
import os
import string

import torch


TEACHER_SIDECAR_VERSION = "teacher_sidecar_v1"
TEACHER_SIDECAR_MANIFEST = "teacher_sidecar_manifest.json"
TEACHER_FIELDS = (
    "relative_disparity",
    "correspondence",
    "visibility",
    "confidence",
)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
    )


class TeacherSidecarStore:
    """Load teacher tensors without exposing them to deployment inputs."""

    def __init__(
        self,
        root: str,
        source_manifest: dict,
        source_manifest_sha256: str,
        grid_height: int,
        grid_width: int,
    ):
        manifest_path = os.path.join(root, TEACHER_SIDECAR_MANIFEST)
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("teacher_sidecar_version") != TEACHER_SIDECAR_VERSION:
            raise ValueError("teacher sidecar version mismatch")
        if manifest.get("complete") is not True:
            raise ValueError("teacher sidecar manifest is incomplete")
        if manifest.get("source_manifest_sha256") != source_manifest_sha256:
            raise ValueError("teacher sidecar source manifest differs")
        if manifest.get("fields") != list(TEACHER_FIELDS):
            raise ValueError("teacher sidecar fields differ from v1 contract")
        if manifest.get("correspondence_semantics") != "stable_track_id":
            raise ValueError("teacher sidecar correspondence semantics differ")
        if (
            int(manifest.get("grid_height", 0)) != grid_height
            or int(manifest.get("grid_width", 0)) != grid_width
        ):
            raise ValueError("teacher sidecar and DINO grids differ")
        source_entries = {
            entry["filename"]: entry for entry in source_manifest["episodes"]
        }
        entries = {}
        paths = [manifest_path]
        for entry in manifest.get("episodes", []):
            source_name = entry.get("source_filename")
            source_entry = source_entries.get(source_name)
            if source_entry is None:
                raise ValueError(f"unknown sidecar source episode: {source_name}")
            if entry.get("source_cache_sha256") != source_entry.get("cache_sha256"):
                raise ValueError(f"source episode checksum differs: {source_name}")
            if int(entry.get("frame_count", -1)) != int(source_entry["frame_count"]):
                raise ValueError(f"source episode frame count differs: {source_name}")
            if not _valid_sha256(entry.get("sidecar_sha256")):
                raise ValueError(f"invalid sidecar checksum: {source_name}")
            path = os.path.abspath(os.path.join(root, entry["filename"]))
            if os.path.commonpath((os.path.abspath(root), path)) != os.path.abspath(root):
                raise ValueError(f"sidecar path escapes its root: {path}")
            if not os.path.isfile(path):
                raise ValueError(f"teacher sidecar is missing: {path}")
            if source_name in entries:
                raise ValueError(f"duplicate teacher sidecar: {source_name}")
            entries[source_name] = (path, entry)
            paths.append(path)
        missing = sorted(set(source_entries) - set(entries))
        if missing:
            raise ValueError(f"teacher sidecar misses {len(missing)} episodes")
        self.root = os.path.abspath(root)
        self.manifest_path = manifest_path
        self.manifest_sha256 = file_sha256(manifest_path)
        self.entries = entries
        self.paths = paths
        self.grid_height = grid_height
        self.grid_width = grid_width

    def _load(self, source_path: str) -> dict:
        basename = os.path.basename(source_path)
        matches = [
            name for name in self.entries if os.path.basename(name) == basename
        ]
        if len(matches) != 1:
            raise ValueError(f"sidecar source filename is ambiguous: {basename}")
        source_name = matches[0]
        path, entry = self.entries[source_name]
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if payload.get("teacher_sidecar_version") != TEACHER_SIDECAR_VERSION:
            raise ValueError(f"sidecar payload version mismatch: {path}")
        if payload.get("source_filename") != source_name:
            raise ValueError(f"sidecar payload source mismatch: {path}")
        if payload.get("source_cache_sha256") != entry["source_cache_sha256"]:
            raise ValueError(f"sidecar payload checksum contract differs: {path}")
        controls = payload.get("frame_control_indices")
        frame_count = int(entry["frame_count"])
        if (
            not torch.is_tensor(controls)
            or controls.dtype != torch.int64
            or controls.shape != (frame_count,)
            or not torch.equal(controls, torch.arange(frame_count, dtype=controls.dtype))
        ):
            raise ValueError(f"invalid sidecar frame indices: {path}")
        expected = (frame_count, self.grid_height, self.grid_width)
        for name in TEACHER_FIELDS:
            value = payload.get(name)
            if not torch.is_tensor(value) or value.shape != expected:
                raise ValueError(f"invalid sidecar tensor {name}: {path}")
        if payload["correspondence"].dtype != torch.int64:
            raise ValueError(f"sidecar correspondence must be int64: {path}")
        if bool((payload["correspondence"] < -1).any()):
            raise ValueError(f"sidecar correspondence ids must be >= -1: {path}")
        for name in ("relative_disparity", "visibility", "confidence"):
            if not payload[name].is_floating_point():
                raise ValueError(f"sidecar tensor {name} must be floating point: {path}")
            if not bool(torch.isfinite(payload[name].float()).all()):
                raise ValueError(f"non-finite sidecar tensor {name}: {path}")
        for name in ("visibility", "confidence"):
            value = payload[name]
            if bool(((value < 0) | (value > 1)).any()):
                raise ValueError(f"sidecar tensor {name} must be in [0,1]: {path}")
        return payload

    def sample(
        self,
        source_path: str,
        sampled_controls: torch.Tensor,
        history_index: torch.Tensor,
        future_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        payload = self._load(source_path)
        result = {"teacher_sidecar_present": torch.tensor(True)}
        for name in TEACHER_FIELDS:
            sampled = payload[name][sampled_controls]
            result[f"teacher_history_{name}"] = sampled[history_index].flatten(1, 2)
            result[f"teacher_future_{name}"] = sampled[future_index].flatten(1, 2)
        return result

    def verify_hashes(self) -> None:
        for path, entry in self.entries.values():
            if file_sha256(path) != entry["sidecar_sha256"]:
                raise ValueError(f"teacher sidecar checksum mismatch: {path}")
