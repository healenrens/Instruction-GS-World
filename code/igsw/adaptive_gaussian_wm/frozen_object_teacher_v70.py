"""Fixed V69 interfaces: history-only training runtime and offline label teacher."""

from pathlib import Path

import torch
from torch import nn

from .object_sequence_dynamics_v69 import ObjectSequencePosteriorV69
from .pretrained_visual_encoder_v69 import PretrainedVisualEncoderV69
from .query_object_video_encoder_v69 import (
    ObjectQueriesV69, QueryObjectVideoEncoderV69, history_queries_v69,
)
from .v69_config import ObjectVideoConfigV69


def module_state(checkpoint, prefix):
    return {key[len(prefix) + 1:]: value for key, value in checkpoint["model"].items()
            if key.startswith(prefix + ".")}


def perception_from_checkpoint(checkpoint, config, frame_batch):
    args = checkpoint["args"]
    return PretrainedVisualEncoderV69(
        config.encoder, args["encoder_repository"], args["encoder_weights"],
        frame_batch=frame_batch, history_seconds=config.history_seconds,
        saved_backbone=checkpoint["perception"],
    )


def query_from_batch(batch):
    return ObjectQueriesV69(batch["query_xy"], batch["query_frame_index"],
                           batch["query_features"], batch["query_valid"])


def history_context(states):
    return {"tokens": torch.stack([state.tokens for state in states], 1),
            "centers": torch.stack([state.centers for state in states], 1),
            "times": torch.stack([state.time for state in states], 1),
            "query_valid": states[-1].query_valid}


class FrozenHistoryStateV70(nn.Module):
    """Only perception and online State are retained in Stage 3 GPU memory."""

    def __init__(self, checkpoint, device, frame_batch=8):
        super().__init__()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.config = ObjectVideoConfigV69(**saved["config"])
        self.perception = perception_from_checkpoint(saved, self.config, frame_batch)
        self.encoder = QueryObjectVideoEncoderV69(self.config)
        self.encoder.load_state_dict(module_state(saved, "encoder"))
        self.teacher = {"path": str(Path(checkpoint).resolve()),
                        "global_step": saved["step"],
                        "source_revision": saved["args"].get("source_revision"),
                        "config": saved["config"]}
        self.requires_grad_(False).to(device).eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, batch):
        perception = self.perception(batch["rgb"], batch["pixel_valid"],
                                     batch["times"], batch["native_hw"])
        states = self.encoder(perception, query_from_batch(batch))
        return history_context(states)


class FixedEffectTeacherV70(FrozenHistoryStateV70):
    """Future access exists only here, in offline target preparation."""

    def __init__(self, checkpoint, device, frame_batch=8):
        super().__init__(checkpoint, device, frame_batch)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.target_encoder = QueryObjectVideoEncoderV69(self.config)
        self.target_encoder.load_state_dict(module_state(saved, "target_encoder"))
        self.posterior = ObjectSequencePosteriorV69(self.config)
        self.posterior.load_state_dict(module_state(saved, "posterior"))
        self.requires_grad_(False).to(device).eval()

    @torch.no_grad()
    def forward(self, batch):
        perception = self.perception(batch["rgb"], batch["pixel_valid"],
                                     batch["times"], batch["native_hw"])
        th = self.config.history_frames
        past = perception.prefix(th)
        queries = history_queries_v69(past, self.config.object_queries)
        history = self.encoder(past, queries)
        state, targets = history[-1].detach(), []
        for frame in range(th, perception.features.shape[1]):
            state = self.target_encoder.observe(
                state, perception.features[:, frame], perception.coordinates,
                perception.valid[:, frame], perception.times[:, frame])
            targets.append(state)
        effect = self.posterior(history[-1], targets, True, batch["frame_valid"][:, th:])
        return {"mean": effect["mean"], "logvar": effect["logvar"],
                "query_xy": queries.xy, "query_frame_index": queries.frame_index,
                "query_features": queries.features, "query_valid": queries.valid}
