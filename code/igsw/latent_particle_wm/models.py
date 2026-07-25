"""Compact latent particle world models for structural validation.

These models deliberately keep perception frozen outside the probe. They test
the causal and probabilistic structure before scaling to the full GPSToken DiT.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow_prior import ConditionalFlowPrior

@dataclass
class WorldModelConfig:
    kind: str = "hierarchical"
    state_dim: int = 14
    target_dim: int = 6
    hidden_dim: int = 128
    layers: int = 3
    heads: int = 4
    local_latent_dim: int = 8
    global_latent_dim: int = 12
    mixture_components: int = 4
    flow_steps: int = 16
    dropout: float = 0.0
    max_horizon: int = 12

    def to_dict(self) -> dict:
        return asdict(self)

def _normal_params(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu, log_std = raw.chunk(2, dim=-1)
    return mu, log_std.clamp(-5.0, 2.0)

def _sample_normal(mu: torch.Tensor, log_std: torch.Tensor, sample: bool) -> torch.Tensor:
    if not sample:
        return mu
    return mu + torch.randn_like(mu) * log_std.exp()

def _normal_kl(
    q_mu: torch.Tensor,
    q_log_std: torch.Tensor,
    p_mu: torch.Tensor,
    p_log_std: torch.Tensor,
) -> torch.Tensor:
    variance_ratio = ((q_log_std.exp() / p_log_std.exp()) ** 2)
    mean_term = ((q_mu - p_mu) / p_log_std.exp()) ** 2
    return 0.5 * (variance_ratio + mean_term - 1.0 + 2.0 * (p_log_std - q_log_std))

def _normal_log_prob(value: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    normalized = (value - mu) / log_std.exp()
    return -0.5 * (normalized.square() + 2.0 * log_std + math.log(2.0 * math.pi))

class HorizonEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, max_horizon: int):
        super().__init__()
        self.max_horizon = max_horizon
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, horizon: torch.Tensor) -> torch.Tensor:
        value = horizon.float() / self.max_horizon
        features = torch.stack(
            (
                value,
                value.square(),
                torch.sin(math.pi * value),
                torch.cos(math.pi * value),
                torch.sin(2.0 * math.pi * value),
            ),
            dim=-1,
        )
        return self.mlp(features)

class ParticleWorldModel(nn.Module):
    """Probe variants sharing one interaction-aware particle dynamics core.

    kind:
      deterministic: no latent action.
      global: one correlated scene-action Gaussian.
      global_mixture: one correlated scene-action with a mixture prior.
      local: independent per-particle Gaussian posterior/prior.
      hierarchical: global Gaussian plus local residual Gaussians.
      hierarchical_mixture: Gaussian posterior, mixture global prior, local residuals.
    """

    VALID_KINDS = {
        "deterministic",
        "global",
        "global_flow",
        "global_mixture",
        "local",
        "hierarchical",
        "hierarchical_mixture",
    }

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        if config.kind not in self.VALID_KINDS:
            raise ValueError(f"unknown model kind: {config.kind}")
        self.config = config
        self.has_local = config.kind in {"local", "hierarchical", "hierarchical_mixture"}
        self.has_global = config.kind in {
            "global",
            "global_flow",
            "global_mixture",
            "hierarchical",
            "hierarchical_mixture",
        }
        self.mixture_prior = config.kind in {"global_mixture", "hierarchical_mixture"}
        self.flow_prior = config.kind == "global_flow"
        dim = config.hidden_dim
        block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.heads,
            dim_feedforward=dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.state_input = nn.Sequential(
            nn.Linear(config.state_dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.state_encoder = nn.TransformerEncoder(block, num_layers=config.layers)
        self.horizon = HorizonEmbedding(dim, config.max_horizon)
        self.effect_input = nn.Sequential(
            nn.Linear(config.target_dim + 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        local_dim = config.local_latent_dim
        global_dim = config.global_latent_dim
        if self.has_local:
            local_context = dim * 2 + (global_dim if self.has_global else 0)
            local_prior_context = dim + (global_dim if self.has_global else 0)
            self.local_posterior = nn.Sequential(
                nn.Linear(local_context, dim),
                nn.SiLU(),
                nn.Linear(dim, local_dim * 2),
            )
            self.local_prior = nn.Sequential(
                nn.Linear(local_prior_context, dim),
                nn.SiLU(),
                nn.Linear(dim, local_dim * 2),
            )
            self.local_effect_head = nn.Sequential(
                nn.Linear(local_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, 3),
            )
        if self.has_global:
            self.global_posterior = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.SiLU(),
                nn.Linear(dim, global_dim * 2),
            )
            if self.flow_prior:
                self.global_prior = ConditionalFlowPrior(
                    global_dim,
                    dim,
                    dim,
                    steps=config.flow_steps,
                )
            elif self.mixture_prior:
                count = config.mixture_components
                self.global_prior = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.SiLU(),
                    nn.Linear(dim, count * global_dim * 2 + count),
                )
            else:
                self.global_prior = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.SiLU(),
                    nn.Linear(dim, global_dim * 2),
                )
            self.effect_head = nn.Sequential(
                nn.Linear(global_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, 8),
            )

        latent_total = local_dim if self.has_local else 0
        latent_total += global_dim if self.has_global else 0
        self.dynamics_input = nn.Sequential(
            nn.Linear(dim * 2 + latent_total, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        dynamics_block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.heads,
            dim_feedforward=dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.dynamics = nn.TransformerEncoder(dynamics_block, num_layers=config.layers)
        self.target_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, config.target_dim))
        self.visibility_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.float()[..., None]
        return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def encode_state(self, state: torch.Tensor, horizon: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        horizon_embedding = self.horizon(horizon)
        hidden = self.state_encoder(self.state_input(state) + horizon_embedding[:, None])
        return hidden, horizon_embedding

    def _effect_hidden(self, batch: dict, hidden: torch.Tensor) -> torch.Tensor:
        valid = batch["motion_valid"].float()[..., None]
        visible = batch["visible"].float()[..., None]
        effect = torch.cat((batch["target"] * valid, valid, visible), dim=-1)
        return self.effect_input(effect) + hidden

    def _global_prior_params(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        config = self.config
        if self.flow_prior:
            return {"global_p_context": pooled}
        raw = self.global_prior(pooled)
        if not self.mixture_prior:
            mu, log_std = _normal_params(raw)
            return {"global_p_mu": mu, "global_p_log_std": log_std}
        count = config.mixture_components
        logits = raw[:, :count]
        params = raw[:, count:].reshape(len(raw), count, config.global_latent_dim * 2)
        mu, log_std = _normal_params(params)
        return {
            "global_p_logits": logits,
            "global_p_mu": mu,
            "global_p_log_std": log_std,
        }

    def prior_parameters(self, batch: dict) -> dict[str, torch.Tensor]:
        hidden, horizon_embedding = self.encode_state(batch["state"], batch["horizon"])
        pooled = hidden.mean(dim=1) + horizon_embedding
        result = {"hidden": hidden, "horizon_embedding": horizon_embedding}
        if not self.has_local and not self.has_global:
            return result
        if self.has_global:
            result.update(self._global_prior_params(pooled))
        if self.has_local and not self.has_global:
            local_mu, local_log_std = _normal_params(self.local_prior(hidden))
            result.update({"local_p_mu": local_mu, "local_p_log_std": local_log_std})
        return result

    def _sample_global_prior(self, params: dict[str, torch.Tensor], sample: bool) -> torch.Tensor:
        if self.flow_prior:
            return self.global_prior.sample(params["global_p_context"], sample)
        if not self.mixture_prior:
            return _sample_normal(params["global_p_mu"], params["global_p_log_std"], sample)
        logits = params["global_p_logits"]
        if sample:
            component = torch.distributions.Categorical(logits=logits).sample()
            row = torch.arange(len(component), device=component.device)
            mu = params["global_p_mu"][row, component]
            log_std = params["global_p_log_std"][row, component]
            return _sample_normal(mu, log_std, True)
        weight = logits.softmax(dim=-1)[..., None]
        return (weight * params["global_p_mu"]).sum(dim=1)

    def _posterior_latents(
        self,
        batch: dict,
        params: dict[str, torch.Tensor],
        sample: bool,
    ) -> dict[str, torch.Tensor]:
        hidden = params["hidden"]
        effect_hidden = self._effect_hidden(batch, hidden)
        result: dict[str, torch.Tensor] = {}
        if self.has_global:
            pooled_effect = self._masked_mean(effect_hidden, batch["motion_valid"])
            pooled_state = hidden.mean(dim=1)
            q_mu, q_log_std = _normal_params(
                self.global_posterior(torch.cat((pooled_state, pooled_effect), dim=-1))
            )
            global_z = _sample_normal(q_mu, q_log_std, sample)
            result.update(
                {
                    "global_q_mu": q_mu,
                    "global_q_log_std": q_log_std,
                    "global_z": global_z,
                }
            )
        if self.has_local and self.has_global:
            global_particles = result["global_z"][:, None].expand(-1, hidden.shape[1], -1)
            q_input = torch.cat((hidden, effect_hidden, global_particles), dim=-1)
            p_input = torch.cat((hidden, global_particles), dim=-1)
        elif self.has_local:
            q_input = torch.cat((hidden, effect_hidden), dim=-1)
            p_input = hidden
        if self.has_local:
            local_q_mu, local_q_log_std = _normal_params(self.local_posterior(q_input))
            local_p_mu, local_p_log_std = _normal_params(self.local_prior(p_input))
            result.update(
                {
                    "local_q_mu": local_q_mu,
                    "local_q_log_std": local_q_log_std,
                    "local_p_mu": local_p_mu,
                    "local_p_log_std": local_p_log_std,
                    "local_z": _sample_normal(local_q_mu, local_q_log_std, sample),
                }
            )
        return result

    def _prior_latents(
        self,
        params: dict[str, torch.Tensor],
        sample: bool,
    ) -> dict[str, torch.Tensor]:
        hidden = params["hidden"]
        if not self.has_local and not self.has_global:
            return {}
        if self.has_local and not self.has_global:
            return {
                "local_z": _sample_normal(params["local_p_mu"], params["local_p_log_std"], sample)
            }
        result = {"global_z": self._sample_global_prior(params, sample)}
        if self.has_local:
            global_particles = result["global_z"][:, None].expand(-1, hidden.shape[1], -1)
            local_mu, local_log_std = _normal_params(
                self.local_prior(torch.cat((hidden, global_particles), dim=-1))
            )
            result.update(
                {
                    "local_z": _sample_normal(local_mu, local_log_std, sample),
                    "local_p_mu": local_mu,
                    "local_p_log_std": local_log_std,
                }
            )
        return result

    def _decode(
        self,
        params: dict[str, torch.Tensor],
        latents: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = params["hidden"]
        horizon = params["horizon_embedding"][:, None].expand_as(hidden)
        values = [hidden, horizon]
        if "global_z" in latents:
            values.append(latents["global_z"][:, None].expand(-1, hidden.shape[1], -1))
        if "local_z" in latents:
            values.append(latents["local_z"])
        dynamics = self.dynamics(self.dynamics_input(torch.cat(values, dim=-1)))
        return self.target_head(dynamics), self.visibility_head(dynamics).squeeze(-1)

    def forward(self, batch: dict, sample_posterior: bool = True) -> dict[str, torch.Tensor]:
        params = self.prior_parameters(batch)
        if not self.has_local and not self.has_global:
            target, visibility = self._decode(params, {})
            return {**params, "prediction": target, "visibility_logits": visibility}
        latents = self._posterior_latents(batch, params, sample_posterior)
        target, visibility = self._decode(params, latents)
        result = {**params, **latents, "prediction": target, "visibility_logits": visibility}
        if "local_z" in latents:
            result["local_effect_prediction"] = self.local_effect_head(latents["local_z"])
        if "global_z" in latents:
            result["effect_prediction"] = self.effect_head(latents["global_z"])
        return result

    def decode_latents(
        self,
        output: dict[str, torch.Tensor],
        local_z: torch.Tensor | None = None,
        global_z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = {}
        if local_z is not None:
            latents["local_z"] = local_z
        if global_z is not None:
            latents["global_z"] = global_z
        return self._decode(output, latents)

    def set_training_phase(self, phase: str) -> None:
        """Select posterior/dynamics learning or frozen-code prior fitting."""
        if phase not in {"posterior", "prior", "all"}:
            raise ValueError(f"unknown phase: {phase}")
        for parameter in self.parameters():
            parameter.requires_grad_(phase == "all")
        if phase == "posterior":
            for parameter in self.parameters():
                parameter.requires_grad_(True)
            for module in (getattr(self, "local_prior", None), getattr(self, "global_prior", None)):
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
        elif phase == "prior":
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for module in (getattr(self, "local_prior", None), getattr(self, "global_prior", None)):
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)

    @torch.no_grad()
    def predict_prior(
        self,
        batch: dict,
        samples: int,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.prior_parameters(batch)
        if not self.has_local and not self.has_global:
            target, visible = self._decode(params, {})
            return (
                target[None].expand(samples, -1, -1, -1),
                visible[None].expand(samples, -1, -1),
            )
        predictions, visibility = [], []
        for _ in range(samples):
            latent = self._prior_latents(params, sample)
            target, visible = self._decode(params, latent)
            predictions.append(target)
            visibility.append(visible)
        return torch.stack(predictions), torch.stack(visibility)

    def kl_loss(
        self,
        output: dict[str, torch.Tensor],
        valid: torch.Tensor,
        free_bits: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.has_local and not self.has_global:
            zero = output["prediction"].sum() * 0.0
            return zero, {"kl_global": zero.detach(), "kl_local": zero.detach()}
        local_kl = output["prediction"].sum() * 0.0
        if self.has_local:
            local = _normal_kl(
                output["local_q_mu"],
                output["local_q_log_std"],
                output["local_p_mu"],
                output["local_p_log_std"],
            ).sum(dim=-1)
            local = local.clamp_min(free_bits)
            local_kl = (local * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
        if not self.has_global:
            zero = local_kl.detach() * 0.0
            return local_kl, {"kl_global": zero, "kl_local": local_kl.detach()}
        if self.flow_prior:
            global_kl = output["prediction"].sum() * 0.0
        elif not self.mixture_prior:
            global_kl = _normal_kl(
                output["global_q_mu"],
                output["global_q_log_std"],
                output["global_p_mu"],
                output["global_p_log_std"],
            ).sum(dim=-1)
        else:
            value = output["global_z"]
            log_q = _normal_log_prob(
                value,
                output["global_q_mu"],
                output["global_q_log_std"],
            ).sum(dim=-1)
            value = value[:, None]
            component_log_prob = _normal_log_prob(
                value,
                output["global_p_mu"],
                output["global_p_log_std"],
            ).sum(dim=-1)
            log_p = torch.logsumexp(
                output["global_p_logits"].log_softmax(dim=-1) + component_log_prob,
                dim=-1,
            )
            global_kl = log_q - log_p
        global_kl = global_kl.clamp_min(free_bits).mean()
        total = global_kl + local_kl
        return total, {"kl_global": global_kl.detach(), "kl_local": local_kl.detach()}

    def prior_fitting_loss(
        self,
        output: dict[str, torch.Tensor],
        valid: torch.Tensor,
        free_bits: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.flow_prior:
            return self.kl_loss(output, valid, free_bits)
        loss = self.global_prior.loss(
            output["global_q_mu"].detach(),
            output["global_p_context"],
        )
        zero = loss.detach() * 0.0
        return loss, {"kl_global": loss.detach(), "kl_local": zero}
