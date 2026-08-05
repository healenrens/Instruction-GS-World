"""Pinned external artifact contract for the frozen Wan2.2 video VAE."""
from __future__ import annotations

import hashlib
import json
import os


VIDEO_VAE_CONTRACT = "wan2_2_ti2v_5b_diffusers_vae_v1"
VIDEO_VAE_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
VIDEO_VAE_DIFFUSERS_VERSION = "0.35.2"


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_video_vae_contract(path: str) -> dict:
    if not os.path.isabs(path) or not os.path.isfile(path):
        raise ValueError("video VAE contract must be an existing absolute file")
    with open(path, encoding="utf-8") as handle:
        contract = json.load(handle)
    expected = {
        "contract": VIDEO_VAE_CONTRACT,
        "model_id": VIDEO_VAE_MODEL_ID,
        "subfolder": "vae",
        "class_name": "AutoencoderKLWan",
        "latent_dim": 48,
        "temporal_compression": 4,
        "spatial_compression": 16,
        "patch_size": 2,
        "probe_clip_frames": 5,
        "probe_status": "passed",
        "diffusers_version": VIDEO_VAE_DIFFUSERS_VERSION,
    }
    mismatch = {
        name: {"contract": contract.get(name), "required": value}
        for name, value in expected.items()
        if contract.get(name) != value
    }
    if mismatch:
        raise ValueError(f"video VAE contract differs: {mismatch}")
    revision = contract.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("video VAE contract requires a pinned 40-char revision")
    files = contract.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("video VAE contract has no file hashes")
    for relative, digest in files.items():
        if os.path.isabs(relative) or relative.startswith("../"):
            raise ValueError("video VAE contract file path escapes the model root")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("video VAE contract contains an invalid SHA256")
    probe_shapes = {
        "probe_latent_shape": (1, 48, 2),
        "probe_decoded_shape": (1, 3, 5),
    }
    for name, prefix in probe_shapes.items():
        shape = contract.get(name)
        if not isinstance(shape, list) or tuple(shape[:3]) != prefix:
            raise ValueError(f"video VAE contract has an invalid {name}")
    sensitivity = contract.get("probe_input_sensitivity_l1")
    peak_memory = contract.get("probe_peak_memory_gb")
    if not isinstance(sensitivity, (int, float)) or sensitivity <= 1e-6:
        raise ValueError("video VAE probe did not establish input sensitivity")
    if not isinstance(peak_memory, (int, float)) or peak_memory <= 0.0:
        raise ValueError("video VAE probe has no peak-memory measurement")
    return contract


def validate_video_vae_artifact(
    model_root: str,
    contract_path: str,
    verify_hashes: bool,
) -> dict:
    if not os.path.isabs(model_root) or not os.path.isdir(model_root):
        raise ValueError("video VAE model root must be an existing absolute directory")
    contract = load_video_vae_contract(contract_path)
    for relative, expected in contract["files"].items():
        path = os.path.join(model_root, relative)
        if not os.path.isfile(path):
            raise ValueError(f"video VAE artifact is missing: {path}")
        if verify_hashes:
            actual = file_sha256(path)
            if actual != expected:
                raise ValueError(f"video VAE artifact SHA256 differs: {path}")
    config_path = os.path.join(model_root, "vae", "config.json")
    if config_path not in [
        os.path.join(model_root, relative) for relative in contract["files"]
    ]:
        raise ValueError("video VAE config is not covered by the hash manifest")
    return contract
