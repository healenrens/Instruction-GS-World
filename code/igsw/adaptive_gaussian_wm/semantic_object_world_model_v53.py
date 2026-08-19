"""V53 semantic object tokenizer followed by continuous latent object dynamics."""

from __future__ import annotations

import torch
from torch import nn

from .object_latent_dynamics_v53 import (
    ContinuousObjectEffectPosterior,
    ObjectLevelDynamics,
    latent_dynamics_objective,
)
from .semantic_object_tokenizer_v53 import (
    SemanticObjectTokenizer,
    tokenizer_objective,
)
from .v53_config import STAGES, SemanticObjectWorldModelConfig


class SemanticObjectLatentWorldModel(nn.Module):
    def __init__(self, config: SemanticObjectWorldModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.tokenizer = SemanticObjectTokenizer(config)
        self.posterior = ContinuousObjectEffectPosterior(config)
        self.dynamics = ObjectLevelDynamics(config)
        self.stage = ""

    def configure_stage(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"unsupported v53 stage: {stage}")
        tokenizer_trainable = stage == "tokenizer"
        self.tokenizer.requires_grad_(tokenizer_trainable)
        self.posterior.requires_grad_(not tokenizer_trainable)
        self.dynamics.requires_grad_(not tokenizer_trainable)
        self.stage = stage

    def forward(
        self,
        patches: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        frame_times: torch.Tensor,
    ) -> dict[str, object]:
        if self.stage not in STAGES:
            raise RuntimeError("v53 model stage was not configured")
        if patches.shape[1] < 2:
            raise ValueError("v53 training requires at least two video observations")
        if frame_times.shape != patches.shape[:2]:
            raise ValueError("v53 frame times differ from the selected video frames")
        if self.stage == "tokenizer":
            encoding = self.tokenizer(patches, coordinates, valid)
            loss, parts = tokenizer_objective(self.config, encoding, patches, valid)
            return {"loss": loss, "parts": parts, "encoding": encoding}
        with torch.no_grad():
            encoding = self.tokenizer(patches, coordinates, valid)
        source = encoding.slots[:, 0, : self.config.object_slots]
        target = encoding.slots[:, -1, : self.config.object_slots]
        scene = encoding.slots[:, 0, self.config.object_slots :]
        delta_seconds = frame_times[:, -1] - frame_times[:, 0]
        loss, parts, outputs = latent_dynamics_objective(
            self.config,
            self.posterior,
            self.dynamics,
            source,
            target,
            scene,
            delta_seconds,
        )
        return {
            "loss": loss,
            "parts": parts,
            "encoding": encoding,
            **outputs,
        }
