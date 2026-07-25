"""Adapter for the existing strict-causal DINO pair artifacts."""
from __future__ import annotations

import glob
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .conditioning import InstructionConditionStore
from .instruction_groups import stable_task_index
from .rgb_supervision import resize_and_pad_rgb
from igsw.latent_particle_wm.pair_targets import CAUSAL_PAIR_VERSION


class CausalPairFeatureDataset(Dataset):
    """Expose causal DINO, RGB targets, and cached deployment-time intent."""

    def __init__(
        self,
        pair_root: str,
        dino_root: str,
        split: str,
        max_items: int = 0,
        condition_cache: str = "",
        load_rgb: bool = False,
        rgb_short_side: int = 256,
        rgb_pad_multiple: int = 16,
    ):
        paths = sorted(glob.glob(os.path.join(pair_root, "*.pt")))
        self.all_paths = [
            path for path in paths if f"_{split}_t" in os.path.basename(path)
        ]
        self.paths = self.all_paths
        if max_items > 0:
            self.paths = self.paths[:max_items]
        if not self.paths:
            raise ValueError(f"no {split} strict-causal pairs in {pair_root}")
        self.dino_root = dino_root
        self.condition_store = (
            InstructionConditionStore(condition_cache)
            if condition_cache
            else None
        )
        self.condition_dim = (
            self.condition_store.feature_dim
            if self.condition_store is not None
            else 0
        )
        self.load_rgb = load_rgb
        self.contract_label = "strict-causal training pairs"
        self.rgb_short_side = rgb_short_side
        self.rgb_pad_multiple = rgb_pad_multiple
        pair, dino = self._load_pair(0)
        if pair.get("pair_version") != CAUSAL_PAIR_VERSION:
            raise ValueError("strict-causal pair version mismatch")
        self.feature_dim = int(dino["feature_dim"])
        dino0 = dino["dino0"]
        if dino0.ndim != 3 or dino0.shape[-1] != self.feature_dim:
            raise ValueError("DINO sidecar must have shape [H,W,C]")
        self.grid_height = int(dino0.shape[0])
        self.grid_width = int(dino0.shape[1])
        if dino["dino1"].shape != dino0.shape:
            raise ValueError("current/future DINO grid shapes differ")
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.grid_height),
            torch.linspace(-1.0, 1.0, self.grid_width),
            indexing="ij",
        )
        self.coordinates = torch.stack((x, y), dim=-1).reshape(-1, 2)
        self.rgb_height = 0
        self.rgb_width = 0
        if self.condition_store is not None:
            self.condition_store.lookup(pair.get("instruction", ""))
            if self.condition_store.token_features is not None:
                self.condition_store.lookup_tokens(
                    pair.get("instruction", "")
                )
        if self.load_rgb:
            rgb, _ = self._rgb_pair(pair)
            self.rgb_height, self.rgb_width = rgb.shape[-2:]

    def _load_pair(self, index: int) -> tuple[dict, dict]:
        pair_path = self.paths[index]
        pair = torch.load(pair_path, map_location="cpu", weights_only=False)
        dino_path = os.path.join(self.dino_root, os.path.basename(pair_path))
        if not os.path.isfile(dino_path):
            raise ValueError(f"missing DINO sidecar: {dino_path}")
        dino = torch.load(dino_path, map_location="cpu", weights_only=False)
        identity = ("source_name", "start", "end")
        mismatched = [
            name for name in identity if dino.get(name) != pair.get(name)
        ]
        if mismatched:
            raise ValueError(
                f"DINO sidecar identity mismatch for {pair_path}: {mismatched}"
            )
        return pair, dino

    def __len__(self) -> int:
        return len(self.paths)

    def _rgb_pair(self, pair: dict) -> tuple[torch.Tensor, torch.Tensor]:
        path = pair.get("rgb_path")
        if not torch.is_tensor(path) or path.ndim != 4 or len(path) < 2:
            raise ValueError("strict-causal pair must contain at least two RGB frames")
        endpoints = torch.stack((path[0], path[-1]))
        return resize_and_pad_rgb(
            endpoints,
            self.rgb_short_side,
            self.rgb_pad_multiple,
        )

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(features.float(), (features.shape[-1],))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pair, dino = self._load_pair(index)
        if pair.get("pair_version") != CAUSAL_PAIR_VERSION:
            raise ValueError(f"strict-causal pair version mismatch: {self.paths[index]}")
        current = self._normalize(dino["dino0"]).reshape(-1, self.feature_dim)
        future = self._normalize(dino["dino1"]).reshape(-1, self.feature_dim)
        if current.shape[0] != len(self.coordinates):
            raise ValueError("DINO grid changed within the dataset")
        valid = torch.ones(len(self.coordinates), dtype=torch.bool)
        result = {
            "history_features": current[None],
            "history_coordinates": self.coordinates[None],
            "history_valid": valid[None],
            "history_times": torch.zeros(1),
            "future_features": future[None],
            "future_coordinates": self.coordinates[None],
            "future_valid": valid[None],
            "future_times": torch.tensor([float(pair["horizon"])]),
            "feature_grid_hw": torch.tensor(
                [self.grid_height, self.grid_width],
                dtype=torch.long,
            ),
            "task_index": torch.tensor(
                stable_task_index(pair.get("task", "")),
                dtype=torch.long,
            ),
        }
        if self.condition_store is not None:
            instruction = pair.get("instruction", "")
            result["condition_feature"] = self.condition_store.lookup(instruction)
            result["condition_index"] = torch.tensor(
                self.condition_store.lookup_index(instruction),
                dtype=torch.long,
            )
            if self.condition_store.token_features is not None:
                (
                    result["condition_tokens"],
                    result["condition_token_valid"],
                ) = self.condition_store.lookup_tokens(instruction)
        if self.load_rgb:
            rgb, rgb_valid = self._rgb_pair(pair)
            if rgb.shape[-2:] != (self.rgb_height, self.rgb_width):
                raise ValueError("RGB padded shape changed within the dataset")
            result.update(
                {
                    "history_rgb": rgb[0:1],
                    "history_rgb_valid": rgb_valid[0:1],
                    "future_rgb": rgb[1:2],
                    "future_rgb_valid": rgb_valid[1:2],
                }
            )
        return result
