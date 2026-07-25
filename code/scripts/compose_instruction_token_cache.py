"""Combine checkpoint-compatible pooled features with exact token features."""
from __future__ import annotations

import argparse
import hashlib
import os

import torch


def _load(path: str) -> dict:
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooled_cache", required=True)
    parser.add_argument("--token_cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pooled_path = os.path.abspath(args.pooled_cache)
    token_path = os.path.abspath(args.token_cache)
    pooled = _load(pooled_path)
    token = _load(token_path)
    if pooled.get("instructions") != token.get("instructions"):
        raise ValueError("pooled and token cache instruction indices differ")
    pooled_features = pooled.get("features")
    token_features = token.get("token_features")
    token_valid = token.get("token_valid")
    if not torch.is_tensor(pooled_features) or pooled_features.ndim != 2:
        raise ValueError("pooled cache features must have shape [U,D]")
    if (
        not torch.is_tensor(token_features)
        or token_features.ndim != 3
        or token_features.shape[0] != pooled_features.shape[0]
        or token_features.shape[2] != pooled_features.shape[1]
    ):
        raise ValueError("token cache features must have shape [U,L,D]")
    if (
        not torch.is_tensor(token_valid)
        or token_valid.shape != token_features.shape[:2]
        or token_valid.dtype != torch.bool
    ):
        raise ValueError("token cache mask must have shape [U,L]")

    payload = dict(token)
    payload["features"] = pooled_features.float().contiguous()
    payload["feature_dim"] = int(pooled_features.shape[1])
    payload["feature_sha256"] = hashlib.sha256(
        payload["features"].numpy().tobytes()
    ).hexdigest()
    payload["pooled_source_cache"] = pooled_path
    payload["token_source_cache"] = token_path
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = f"{output}.tmp.{os.getpid()}"
    torch.save(payload, temporary)
    os.replace(temporary, output)
    print(
        {
            "status": "ok",
            "output": output,
            "features": tuple(payload["features"].shape),
            "tokens": tuple(token_features.shape),
            "feature_sha256": payload["feature_sha256"],
            "token_feature_sha256": payload["token_feature_sha256"],
        }
    )


if __name__ == "__main__":
    main()
