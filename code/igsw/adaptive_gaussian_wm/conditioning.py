"""Cached instruction features and the trainable semantic projection."""
from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn

from .config import AdaptiveGaussianWMConfig


CONDITION_CACHE_VERSION = "qwen_instruction_span_pool_v2"
TOKEN_CONDITION_CACHE_VERSION = "qwen_instruction_tokens_v3"
LEGACY_CONDITION_CACHE_VERSIONS = frozenset(
    ("qwen_text_pool_v1", TOKEN_CONDITION_CACHE_VERSION)
)


def normalize_instruction(instruction: str) -> str:
    if not isinstance(instruction, str):
        raise ValueError("instruction must be a string")
    normalized = " ".join(instruction.split())
    if not normalized:
        raise ValueError("instruction must not be empty")
    return normalized


class InstructionConditionStore:
    """Read a compact cache indexed by normalized instruction text."""

    def __init__(self, path: str):
        if not path or not os.path.isfile(path):
            raise ValueError(f"condition cache not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        version = payload.get("version")
        if version not in {
            CONDITION_CACHE_VERSION,
            *LEGACY_CONDITION_CACHE_VERSIONS,
        }:
            raise ValueError("instruction condition cache version mismatch")
        features = payload.get("features")
        instructions = payload.get("instructions")
        if not torch.is_tensor(features) or features.ndim != 2:
            raise ValueError("condition cache features must have shape [U,D]")
        if not isinstance(instructions, list) or len(instructions) != len(features):
            raise ValueError("condition cache instruction index is malformed")
        normalized = [normalize_instruction(value) for value in instructions]
        if len(set(normalized)) != len(normalized):
            raise ValueError("condition cache contains duplicate instructions")
        self.path = os.path.abspath(path)
        self.version = str(version)
        self.features = features.float().contiguous()
        self.index = {value: index for index, value in enumerate(normalized)}
        self.feature_dim = int(features.shape[1])
        self.model = str(payload.get("model", ""))
        digest = hashlib.sha256(self.features.numpy().tobytes()).hexdigest()
        expected_digest = payload.get("feature_sha256")
        if expected_digest and digest != expected_digest:
            raise ValueError("condition cache feature digest mismatch")
        self.feature_sha256 = digest
        token_features = payload.get("token_features")
        token_valid = payload.get("token_valid")
        if version == TOKEN_CONDITION_CACHE_VERSION:
            if (
                not torch.is_tensor(token_features)
                or token_features.ndim != 3
                or token_features.shape[0] != len(features)
                or token_features.shape[2] != features.shape[1]
            ):
                raise ValueError(
                    "token condition features must have shape [U,L,D]"
                )
            if (
                not torch.is_tensor(token_valid)
                or token_valid.shape != token_features.shape[:2]
                or token_valid.dtype != torch.bool
                or not bool(token_valid.any(dim=1).all())
            ):
                raise ValueError(
                    "token condition valid mask must have shape [U,L]"
                )
            self.token_features = token_features.float().contiguous()
            self.token_valid = token_valid.contiguous()
            token_hasher = hashlib.sha256()
            token_hasher.update(self.token_features.numpy().tobytes())
            token_hasher.update(self.token_valid.numpy().tobytes())
            token_digest = token_hasher.hexdigest()
            if payload.get("token_feature_sha256") != token_digest:
                raise ValueError("token condition cache digest mismatch")
            self.token_feature_sha256 = token_digest
        else:
            self.token_features = None
            self.token_valid = None
            self.token_feature_sha256 = ""

    def lookup(self, instruction: str) -> torch.Tensor:
        return self.features[self.lookup_index(instruction)]

    def lookup_tokens(
        self,
        instruction: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.token_features is None or self.token_valid is None:
            raise ValueError("condition cache has no instruction token features")
        index = self.lookup_index(instruction)
        return self.token_features[index], self.token_valid[index]

    def lookup_index(self, instruction: str) -> int:
        normalized = normalize_instruction(instruction)
        if normalized not in self.index:
            raise ValueError(f"instruction missing from condition cache: {normalized!r}")
        return self.index[normalized]


class LanguageConditionProjector(nn.Module):
    """Map frozen Qwen instruction features into the world-model width."""

    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        if config.condition_dim <= 0:
            raise ValueError("language projector requires condition_dim > 0")
        self.input_dim = config.condition_dim
        self.output_dim = config.model_dim
        self.projection = nn.Sequential(
            nn.LayerNorm(config.condition_dim),
            nn.Linear(config.condition_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
            nn.LayerNorm(config.model_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"condition_feature must have shape [B,{self.input_dim}]"
            )
        return self.projection(features.float())
