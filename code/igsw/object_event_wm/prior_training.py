"""Train only the prior against a fixed, externally stored world-model snapshot."""

from dataclasses import asdict
from pathlib import Path
import os

import torch
from torch import nn
import torch.distributed as dist

from .codec import letterbox_video
from .config import ModelConfig
from .model import ObjectEventWorldModel
from .prior import ObjectEffectPrior, prior_objective


@torch.no_grad()
def infer_effect_targets(teacher, codec, batch):
    images, valid = letterbox_video(batch, codec.image_size)
    features = codec.encode(images)
    observed = teacher.observe(features, codec.patch_valid(valid), batch["times"])
    th = batch["history_frames"]
    source = observed[th-1]
    previous = features.new_zeros((len(features), teacher.config.slots, teacher.config.effect_dim))
    states, effects = [], []
    for index in range(th, len(observed)):
        effect = teacher.posterior(observed[index-1], observed[index], previous,
                                   first=index == th, deterministic=True)
        valid_frame = batch["frame_valid"][:, index]
        previous = torch.where(valid_frame[:, None, None], effect["value"], previous)
        effect["value"] = previous
        effect["boundary"] = effect["boundary"]*valid_frame[:, None]
        source = teacher.dynamics(source, previous, batch["times"][:, index])
        states.append(source)
        effects.append(effect)
    return observed[:th], states, effects


class PriorTrainingModel(nn.Module):
    def __init__(self, config, text_width=2560, layers=4, language_dropout=.1):
        super().__init__()
        self.config = config
        self.prior_config = {"text_width": text_width, "layers": layers, "language_dropout": language_dropout}
        self.prior = ObjectEffectPrior(config, text_width, layers)
        self.language_dropout = language_dropout

    def forward(self, batch, codec, teacher, rollout_weight=1.):
        history, states, effects = infer_effect_targets(teacher, codec, batch)
        tokens, valid = batch.get("text_tokens"), batch.get("text_valid")
        if valid is not None and self.training:
            keep = torch.rand((len(valid), 1), device=valid.device) >= self.language_dropout
            valid = valid & keep
        th = batch["history_frames"]
        loss, parts = prior_objective(self.prior, history, states, effects,
                                     batch["times"][:, th:], batch["frame_valid"][:, th:], tokens, valid)
        parts["language_condition_fraction"] = (valid.any(-1).float().mean() if valid is not None else loss.detach()*0)
        return {"loss": loss, "parts": parts}


def prepare_prior(args, checkpoint, rank, world, device):
    from .text import TextFeatureStore
    from .training import checkpoint_path, read_checkpoint

    if checkpoint is None:
        source = checkpoint_path(args.base_checkpoint)
        snapshot = Path(args.out).resolve() / "teacher.pt"
        if rank == 0:
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, snapshot)
        if world > 1:
            dist.barrier()
        args.base_checkpoint = str(snapshot)
    saved = read_checkpoint(args.base_checkpoint)
    config = ModelConfig(**saved["model_config"])
    teacher = ObjectEventWorldModel(config).to(device).requires_grad_(False).eval()
    teacher.load_state_dict(saved["model"])
    width = TextFeatureStore(args.text_cache).width if args.text_cache else 2560
    options = checkpoint["prior_config"] if checkpoint else {
        "text_width": width, "layers": args.prior_layers, "language_dropout": args.language_dropout}
    model = PriorTrainingModel(config, **options).to(device)
    args.effect_mode = config.effect_mode
    metadata = {"format": "object_event_prior_v71", "prior_config": model.prior_config,
                "teacher_checkpoint": args.base_checkpoint, "teacher_step": saved["step"],
                "teacher_model_config": asdict(config), "teacher_source_revision": saved["args"].get("source_revision", "")}
    return model, teacher, metadata


def load_prior_model(path, device):
    from .training import read_checkpoint

    saved = read_checkpoint(path)
    base = read_checkpoint(saved["teacher_checkpoint"])
    config = ModelConfig(**base["model_config"])
    teacher = ObjectEventWorldModel(config).to(device).requires_grad_(False).eval()
    teacher.load_state_dict(base["model"])
    wrapper = PriorTrainingModel(config, **saved["prior_config"]).to(device).eval()
    wrapper.load_state_dict(saved["model"])
    return teacher, wrapper.prior, saved
