"""History-conditioned effect generation, separate from future-reading inference."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model import weighted_mean
from .state import AttentionBlock, position_features


@dataclass
class PriorContext:
    history: torch.Tensor
    language: torch.Tensor
    language_valid: torch.Tensor
    anchor_time: torch.Tensor


class ObjectEffectPrior(nn.Module):
    """Generate object effects; language inputs are external frozen text features."""

    def __init__(self, config, text_width, layers=4):
        super().__init__()
        self.config = config
        self.text_width = text_width
        self.layers = layers
        d = config.width
        self.history_input = nn.Linear(d+19, d)
        self.current_input = nn.Linear(d+18, d)
        self.previous_effect = nn.Linear(config.effect_dim, d)
        self.time_input = nn.Sequential(nn.Linear(2, d), nn.SiLU(), nn.Linear(d, d))
        self.text_input = nn.Linear(text_width, d)
        self.null_text = nn.Parameter(torch.zeros(1, 1, d))
        self.history_attention = nn.ModuleList(AttentionBlock(d, config.heads) for _ in range(layers))
        self.language_attention = nn.ModuleList(AttentionBlock(d, config.heads) for _ in range(layers))
        self.interaction = nn.ModuleList(AttentionBlock(d, config.heads) for _ in range(layers))
        self.distribution = nn.Linear(d, config.effect_dim*2)
        self.boundary = nn.Linear(d, 1) if config.effect_mode == "object_segment" else None

    def condition(self, observed_history, text_tokens=None, text_valid=None):
        final_time = observed_history[-1].time
        frames = []
        for state in observed_history:
            elapsed = (state.time-final_time)[:, None, None, None].expand(*state.tokens.shape[:-1], 1)
            frames.append(self.history_input(torch.cat((state.tokens.float(),
                          position_features(state.centers.float()), elapsed), -1)))
        history = torch.stack(frames, 2).flatten(2, 3)
        b = len(history)
        null = self.null_text.expand(b, -1, -1)
        valid = torch.ones((b, 1), dtype=torch.bool, device=history.device)
        if text_tokens is None:
            text_tokens = history.new_empty((b, 0, self.text_width))
            text_valid = valid[:, :0]
        # The null token also makes an empty instruction a valid attention context.
        language = torch.cat((null, self.text_input(text_tokens.float())), 1)
        valid = torch.cat((valid, text_valid.bool()), 1)
        return PriorContext(history, language, valid, final_time)

    def forward(self, context, state, previous, time):
        result = self.forward_sequence(context, state.tokens[:, None], state.centers[:, None],
                                       state.time[:, None], previous[:, None], time[:, None])
        return {key: value[:, 0] if value is not None else None for key, value in result.items()}

    def forward_sequence(self, context, tokens, centers, source_times, previous, times):
        b, t, k, _, d = tokens.shape
        value = self.current_input(torch.cat((tokens[:, :, :, 0].float(),
                                              position_features(centers[:, :, :, 0].float())), -1))
        value = value + self.previous_effect(previous.float())
        time_features = torch.stack((times-source_times, times-context.anchor_time[:, None]), -1)
        value = value + self.time_input(time_features)[:, :, None]
        for history, language, interaction in zip(self.history_attention, self.language_attention, self.interaction):
            if self.config.effect_mode == "global_step":
                value = history(value.flatten(1, 2), context.history.flatten(1, 2)).reshape(b, t, k, d)
            else:
                queries = value.permute(0, 2, 1, 3).reshape(b*k, t, d)
                value = history(queries, context.history.reshape(b*k, -1, d)).reshape(b, k, t, d).transpose(1, 2)
            # Cross-attention shares fixed history/text keys; no future query is another query's key.
            value = language(value.reshape(b, t*k, d), context.language, context.language_valid).reshape(b, t, k, d)
            value = interaction(value.reshape(b*t, k, d)).reshape(b, t, k, d)
        mean, logvar = self.distribution(value).float().chunk(2, -1)
        return {"mean": mean, "logvar": logvar.clamp(-6, 2),
                "boundary_logits": self.boundary(value).float().squeeze(-1) if self.boundary is not None else None}

    def sample_effect(self, distribution, previous, first, generator=None, deterministic=False):
        mean, logvar = distribution["mean"], distribution["logvar"]
        candidate = mean
        if not deterministic:
            noise = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
            candidate = mean + (.5*logvar).exp()*noise
        logits = distribution["boundary_logits"]
        if logits is None or first:
            update = torch.ones_like(mean[..., 0])
        else:
            threshold = .5 if deterministic else torch.rand(logits.shape, device=logits.device, generator=generator)
            update = (logits.sigmoid() > threshold).float()
        return previous + update[..., None]*(candidate-previous), update

    def generate(self, observed_history, times, dynamics, text_tokens=None, text_valid=None,
                 generator=None, deterministic=False):
        context = self.condition(observed_history, text_tokens, text_valid)
        state = observed_history[-1]
        previous = state.tokens.new_zeros((len(state.tokens), self.config.slots, self.config.effect_dim))
        states, effects, updates = [], [], []
        for index, time in enumerate(times.unbind(1)):
            distribution = self(context, state, previous, time)
            previous, update = self.sample_effect(distribution, previous, index == 0, generator, deterministic)
            state = dynamics(state, previous, time)
            states.append(state)
            effects.append(previous)
            updates.append(update)
        return {"states": states, "effects": torch.stack(effects, 1), "updates": torch.stack(updates, 1)}


def prior_objective(prior, observed_history, rollout_states, posterior_effects, times, frame_valid,
                    text_tokens=None, text_valid=None):
    """Teacher-force earlier inferred effects, never the effect being predicted."""
    context = prior.condition([state.detach() for state in observed_history], text_tokens, text_valid)
    sources = [observed_history[-1], *rollout_states[:-1]]
    values = torch.stack([row["value"].detach() for row in posterior_effects], 1)
    previous = torch.cat((torch.zeros_like(values[:, :1]), values[:, :-1]), 1)
    prediction = prior.forward_sequence(
        context, torch.stack([s.tokens.detach() for s in sources], 1),
        torch.stack([s.centers.detach() for s in sources], 1),
        torch.stack([s.time for s in sources], 1), previous, times)
    q_mean = torch.stack([row["mean"].detach() for row in posterior_effects], 1)
    q_logvar = torch.stack([row["logvar"].detach() for row in posterior_effects], 1)
    p_mean, p_logvar = prediction["mean"], prediction["logvar"]
    kl = .5*(p_logvar-q_logvar + (q_logvar.exp()+(q_mean-p_mean).square())/p_logvar.exp()-1)
    used = torch.stack([row["boundary"].detach() for row in posterior_effects], 1)
    innovation = weighted_mean(kl.mean(-1), frame_valid[..., None]*used)
    logits = prediction["boundary_logits"]
    if logits is not None and len(posterior_effects) > 1:
        probability = torch.stack([row["boundary_probability"].detach() for row in posterior_effects], 1)
        boundary = weighted_mean(F.binary_cross_entropy_with_logits(
            logits[:, 1:], probability[:, 1:], reduction="none"), frame_valid[:, 1:, None])
    else:
        boundary = innovation*0
    return innovation+boundary, {"innovation_kl": innovation, "boundary_cross_entropy": boundary}
