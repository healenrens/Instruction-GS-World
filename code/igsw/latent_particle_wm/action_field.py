"""Effect-aligned correlated action tokens and a current-only flow prior."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow_prior import ConditionalFlowPrior
from .pair_data import PAIR_STATE_DIM


@dataclass
class ActionFieldConfig:
    state_dim: int = PAIR_STATE_DIM
    hidden_dim: int = 128
    layers: int = 2
    heads: int = 4
    action_count: int = 4
    action_dim: int = 16
    dino_dim: int = 32
    control_rows: int = 16
    control_cols: int = 16
    dense_grid: int = 48
    max_horizon: int = 12
    flow_steps: int = 16
    dropout: float = 0.0
    posterior_xy_scale: float = 1.0
    posterior_depth_scale: float = 1.0

    @property
    def field_dim(self) -> int:
        return 10 + self.dino_dim

    def to_dict(self) -> dict:
        return asdict(self)


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


class CorrelatedActionField(nn.Module):
    def __init__(self, config: ActionFieldConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.state_input = nn.Sequential(
            nn.Linear(config.state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        state_block = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=hidden * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.state_encoder = nn.TransformerEncoder(state_block, config.layers)
        self.horizon = HorizonEmbedding(hidden, config.max_horizon)

        self.effect_input = nn.Sequential(
            nn.Linear(8, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.posterior_slots = nn.Parameter(torch.randn(config.action_count, hidden) * 0.02)
        self.posterior_attention = nn.MultiheadAttention(
            hidden,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.posterior_norm = nn.LayerNorm(hidden)
        self.posterior_ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, config.action_dim),
        )

        latent_dim = config.action_count * config.action_dim
        self.flow_prior = ConditionalFlowPrior(
            latent_dim,
            hidden,
            hidden,
            steps=config.flow_steps,
        )
        self.action_input = nn.Sequential(
            nn.Linear(config.action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.action_context = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.action_slots = nn.Parameter(
            torch.randn(config.action_count, hidden) * 0.02
        )
        self.action_cross_attention = nn.MultiheadAttention(
            hidden,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dynamics_fusion = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        dynamics_block = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=hidden * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.dynamics = nn.TransformerEncoder(dynamics_block, config.layers)
        self.dynamics_norm = nn.LayerNorm(hidden)
        self.field_head = nn.Linear(hidden, config.field_dim)
        self.visibility_head = nn.Linear(hidden, 1)
        deterministic_block = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=hidden * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.deterministic_dynamics = nn.TransformerEncoder(
            deterministic_block,
            config.layers,
        )
        self.deterministic_norm = nn.LayerNorm(hidden)
        self.deterministic_field_head = nn.Linear(hidden, config.field_dim)
        self.effect_head = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 8),
        )
        nn.init.zeros_(self.field_head.weight)
        nn.init.zeros_(self.field_head.bias)
        nn.init.zeros_(self.deterministic_field_head.weight)
        nn.init.zeros_(self.deterministic_field_head.bias)

    def encode_current(self, batch: dict) -> dict[str, torch.Tensor]:
        horizon = self.horizon(batch["horizon"])
        hidden = self.state_encoder(self.state_input(batch["state"]) + horizon[:, None])
        context = hidden.mean(dim=1) + horizon
        return {"hidden": hidden, "horizon_embedding": horizon, "prior_context": context}

    def posterior_actions(
        self,
        batch: dict,
        encoded: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = batch["motion_valid"]
        if not bool(valid.any(dim=1).all()):
            raise ValueError("posterior requires at least one valid motion target per sample")
        effect = torch.cat(
            (
                batch["target"] * valid[..., None],
                valid.float()[..., None],
                batch["visible"].float()[..., None],
            ),
            dim=-1,
        )
        effect_hidden = self.effect_input(effect) + encoded["hidden"]
        query = self.posterior_slots[None].expand(len(effect), -1, -1)
        query = query + encoded["prior_context"][:, None]
        action_hidden, attention = self.posterior_attention(
            query,
            effect_hidden,
            effect_hidden,
            key_padding_mask=~valid,
            need_weights=True,
        )
        actions = torch.tanh(self.posterior_ffn(self.posterior_norm(action_hidden + query)))
        return actions, attention

    def decode_actions(
        self,
        encoded: dict[str, torch.Tensor],
        actions: torch.Tensor,
        deterministic_field: torch.Tensor | None = None,
        residual_xy_scale: float = 1.0,
        residual_depth_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        action_hidden = self.action_input(actions) + self.action_slots[None]
        if self.config.action_count != 4:
            raise ValueError("spatial action binding requires a 2x2 action-token grid")
        spatial_action = F.interpolate(
            action_hidden.reshape(
                len(actions),
                2,
                2,
                self.config.hidden_dim,
            ).permute(0, 3, 1, 2),
            size=(self.config.control_rows, self.config.control_cols),
            mode="bilinear",
            align_corners=True,
        ).permute(0, 2, 3, 1).flatten(1, 2)
        global_action = self.action_context(actions.flatten(1))[:, None]
        global_action = global_action.expand(-1, encoded["hidden"].shape[1], -1)
        correlated, attention = self.action_cross_attention(
            encoded["hidden"],
            action_hidden,
            action_hidden,
            need_weights=True,
        )
        dynamics_input = (
            self.dynamics_fusion(
                torch.cat(
                    (
                        encoded["hidden"],
                        spatial_action,
                        global_action,
                        correlated,
                    ),
                    dim=-1,
                )
            )
            + encoded["horizon_embedding"][:, None]
        )
        dynamics = self.dynamics(self.dynamics_norm(dynamics_input))
        if deterministic_field is None:
            deterministic_field = self.decode_deterministic(encoded)["control_field"]
        raw_residual = self.field_head(dynamics)
        motion_scale = raw_residual.new_tensor(
            [residual_xy_scale, residual_xy_scale, residual_depth_scale]
        )
        motion_residual = raw_residual[..., :3] * motion_scale
        action_residual = torch.cat((motion_residual, raw_residual[..., 3:]), dim=-1)
        control_field = deterministic_field.detach() + action_residual
        control_visibility = self.visibility_head(dynamics).squeeze(-1)
        return {
            "control_field": control_field,
            "action_residual": action_residual,
            "control_visibility_logits": control_visibility,
            "action_attention": attention,
            "effect_prediction": self.effect_head(actions.flatten(1)),
        }

    def decode_deterministic(
        self,
        encoded: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        hidden = encoded["hidden"] + encoded["horizon_embedding"][:, None]
        dynamics = self.deterministic_dynamics(self.deterministic_norm(hidden))
        return {"control_field": self.deterministic_field_head(dynamics)}

    def dense_field(self, control_field: torch.Tensor) -> torch.Tensor:
        config = self.config
        expected = config.control_rows * config.control_cols
        if control_field.shape[1] != expected:
            raise ValueError(f"expected {expected} control particles")
        grid = control_field.reshape(
            len(control_field),
            config.control_rows,
            config.control_cols,
            control_field.shape[-1],
        ).permute(0, 3, 1, 2)
        dense = F.interpolate(
            grid,
            size=(config.dense_grid, config.dense_grid),
            mode="bilinear",
            align_corners=True,
        )
        return dense.permute(0, 2, 3, 1).flatten(1, 2)

    def predict_deterministic(self, batch: dict) -> dict[str, torch.Tensor]:
        encoded = self.encode_current(batch)
        output = self.decode_deterministic(encoded)
        output.update(
            {
                **encoded,
                "dense_field": self.dense_field(output["control_field"]),
            }
        )
        return output

    def forward_posterior(self, batch: dict) -> dict[str, torch.Tensor]:
        encoded = self.encode_current(batch)
        actions, posterior_attention = self.posterior_actions(batch, encoded)
        deterministic = self.decode_deterministic(encoded)
        output = self.decode_actions(
            encoded,
            actions,
            deterministic["control_field"],
            self.config.posterior_xy_scale,
            self.config.posterior_depth_scale,
        )
        output.update(
            {
                **encoded,
                "actions": actions,
                "posterior_attention": posterior_attention,
                "dense_field": self.dense_field(output["control_field"]),
                "deterministic_control_field": deterministic["control_field"],
                "deterministic_dense_field": self.dense_field(
                    deterministic["control_field"]
                ),
            }
        )
        return output

    def forward(self, batch: dict, phase: str = "posterior") -> dict[str, torch.Tensor]:
        if phase == "posterior":
            return self.forward_posterior(batch)
        if phase == "prior":
            return {"prior_loss": self.prior_matching_loss(batch)}
        raise ValueError(f"unknown forward phase: {phase}")

    def prior_matching_loss(self, batch: dict) -> torch.Tensor:
        encoded = self.encode_current(batch)
        actions, _ = self.posterior_actions(batch, encoded)
        return self.flow_prior.loss(
            actions.detach().flatten(1),
            encoded["prior_context"],
        )

    @torch.no_grad()
    def sample_prior(
        self,
        batch: dict,
        samples: int,
        stochastic: bool = True,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode_current(batch)
        deterministic_field = self.decode_deterministic(encoded)["control_field"]
        fields = []
        control_fields = []
        visibility = []
        actions = []
        for _ in range(samples):
            flat = self.flow_prior.sample(encoded["prior_context"], stochastic)
            action = flat.reshape(
                len(flat),
                self.config.action_count,
                self.config.action_dim,
            )
            decoded = self.decode_actions(encoded, action, deterministic_field)
            actions.append(action)
            control_fields.append(decoded["control_field"])
            fields.append(self.dense_field(decoded["control_field"]))
            visibility.append(decoded["control_visibility_logits"])
        return {
            "actions": torch.stack(actions),
            "control_field": torch.stack(control_fields),
            "dense_field": torch.stack(fields),
            "control_visibility_logits": torch.stack(visibility),
        }

    def set_training_phase(self, phase: str) -> None:
        if phase not in {"posterior", "prior", "all"}:
            raise ValueError(f"unknown training phase: {phase}")
        for parameter in self.parameters():
            parameter.requires_grad_(phase == "all")
        if phase == "posterior":
            for parameter in self.parameters():
                parameter.requires_grad_(True)
            for parameter in self.flow_prior.parameters():
                parameter.requires_grad_(False)
        elif phase == "prior":
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for parameter in self.flow_prior.parameters():
                parameter.requires_grad_(True)
