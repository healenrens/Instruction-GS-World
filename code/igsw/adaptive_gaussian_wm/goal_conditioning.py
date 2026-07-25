"""Object-aligned image-goal conditioning for the deployable action Prior."""
from __future__ import annotations

import torch
import torch.nn as nn

from .readout_runtime import object_rgb_from_micro
from .rgb_supervision import current_micro_rgb, micro_rgb_from_assignment
from .scale import signed_gap_scale


GOAL_CONDITION_VERSION = 2


def _require_shape(
    value: torch.Tensor,
    shape: tuple[int | None, ...],
    name: str,
) -> None:
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape, strict=True)
    ):
        expected = ["*" if item is None else str(item) for item in shape]
        raise ValueError(f"{name} must have shape [{','.join(expected)}]")


@torch.no_grad()
def encode_explicit_goal(
    model,
    batch: dict[str, torch.Tensor],
    online_history: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Encode history plus the explicitly supplied goal without reading futures."""
    required = (
        "history_features",
        "history_coordinates",
        "history_valid",
        "goal_features",
        "goal_coordinates",
        "goal_valid",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"explicit image goal requires fields {missing}")
    target_history = model._encode_sequence(
        batch["history_features"],
        batch["history_coordinates"],
        batch["history_valid"],
        model.target_allocator,
        model.target_object_aggregator,
    )
    anchor_state = target_history["slot_states"][0]
    goal_tokens = model.target_allocator(
        batch["goal_features"],
        batch["goal_coordinates"],
        batch["goal_valid"],
    )
    goal_state = model.target_object_aggregator(
        goal_tokens,
        anchor_state.tracking_slots,
        anchor_state.center,
    )
    result = {
        "slots": goal_state.slots,
        "tracking_slots": goal_state.tracking_slots,
        "activity": goal_state.activity,
        "center": goal_state.center,
        "feature": goal_state.feature,
    }
    if model.config.rgb_supervision:
        required_rgb = ("goal_rgb", "goal_rgb_valid", "feature_grid_hw")
        missing_rgb = [name for name in required_rgb if name not in batch]
        if missing_rgb:
            raise ValueError(f"RGB image goal requires fields {missing_rgb}")
        grid_hw = batch["feature_grid_hw"]
        if grid_hw.ndim != 2 or grid_hw.shape[1] != 2:
            raise ValueError("feature_grid_hw must have shape [B,2]")
        if not bool((grid_hw == grid_hw[:1]).all()):
            raise ValueError("feature grid dimensions differ within the batch")
        rgb_history = (
            target_history if online_history is None else online_history
        )
        if "token_states" not in rgb_history or "slot_states" not in rgb_history:
            raise ValueError("online history is missing RGB assignment states")
        current_tokens = rgb_history["token_states"][-1]
        current_state = rgb_history["slot_states"][-1]
        current_micro = current_micro_rgb(current_tokens, batch)
        goal_micro = micro_rgb_from_assignment(
            goal_tokens,
            batch["goal_rgb"],
            batch["goal_rgb_valid"],
            int(grid_hw[0, 0]),
            int(grid_hw[0, 1]),
        )
        result["current_rgb"] = object_rgb_from_micro(
            current_micro,
            current_state.assignment,
            current_tokens.activation,
        )
        result["rgb"] = object_rgb_from_micro(
            goal_micro,
            goal_state.assignment,
            goal_tokens.activation,
        )
    return result


class ObjectGoalConditioner(nn.Module):
    """Map current-to-goal object deltas and goal time to Prior query tokens."""

    def __init__(self, config):
        super().__init__()
        object_dim = config.object_dim
        model_dim = config.model_dim
        self.object_dim = object_dim
        self.model_dim = model_dim
        self.use_rgb = config.rgb_supervision
        self.current_norm = nn.LayerNorm(object_dim)
        self.goal_norm = nn.LayerNorm(object_dim)
        observed_dim = 9 + (9 if self.use_rgb else 0)
        self.object_projection = nn.Linear(
            object_dim * 3 + observed_dim,
            model_dim,
        )
        self.goal_time_projection = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.refinement = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.gate_logit = nn.Parameter(torch.tensor(-1.38629436112))

    def forward(
        self,
        current_slots: torch.Tensor,
        current_centers: torch.Tensor,
        current_activity: torch.Tensor,
        goal_slots: torch.Tensor,
        goal_centers: torch.Tensor,
        goal_activity: torch.Tensor,
        goal_scale: torch.Tensor,
        current_rgb: torch.Tensor | None = None,
        goal_rgb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, objects, object_dim = current_slots.shape
        _require_shape(
            current_slots,
            (batch, objects, self.object_dim),
            "current_slots",
        )
        _require_shape(goal_slots, current_slots.shape, "goal_slots")
        _require_shape(current_centers, (batch, objects, 2), "current_centers")
        _require_shape(goal_centers, current_centers.shape, "goal_centers")
        _require_shape(current_activity, (batch, objects), "current_activity")
        _require_shape(goal_activity, current_activity.shape, "goal_activity")
        _require_shape(goal_scale, (batch,), "goal_scale")
        if object_dim != self.object_dim:
            raise ValueError("goal conditioner object width differs from config")

        current = self.current_norm(current_slots.detach())
        goal = self.goal_norm(goal_slots.detach())
        center_delta = goal_centers.detach() - current_centers.detach()
        canonical_center = torch.tanh(
            torch.cat(
                (
                    center_delta,
                    center_delta.norm(dim=-1, keepdim=True),
                ),
                dim=-1,
            )
            / 0.25
        )
        observed = [
            current_centers.detach(),
            goal_centers.detach(),
            canonical_center,
            current_activity.detach()[..., None],
            goal_activity.detach()[..., None],
        ]
        if self.use_rgb:
            if current_rgb is None or goal_rgb is None:
                raise ValueError("RGB goal conditioner requires object RGB")
            _require_shape(current_rgb, (batch, objects, 3), "current_rgb")
            _require_shape(goal_rgb, current_rgb.shape, "goal_rgb")
            current_rgb = current_rgb.detach().float().clamp(1e-4, 1.0 - 1e-4)
            goal_rgb = goal_rgb.detach().float().clamp(1e-4, 1.0 - 1e-4)
            rgb_delta = torch.tanh(
                torch.logit(goal_rgb) - torch.logit(current_rgb)
            )
            observed.extend((current_rgb, goal_rgb, rgb_delta))
        elif current_rgb is not None or goal_rgb is not None:
            raise ValueError("RGB values supplied to a non-RGB goal conditioner")
        geometry = torch.cat(
            (
                *observed,
            ),
            dim=-1,
        ).to(current.dtype)
        object_input = torch.cat((current, goal, goal - current, geometry), dim=-1)
        condition = self.object_projection(object_input)
        condition = condition + self.goal_time_projection(
            goal_scale.detach()[..., None]
        )[:, None]
        condition = self.output_norm(condition + self.refinement(condition))
        return torch.sigmoid(self.gate_logit) * condition


def goal_scale_from_batch(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if "goal_time" not in batch:
        raise ValueError("goal-conditioned Prior requires goal_time")
    return signed_gap_scale(batch["goal_time"], model.config.gap_reference)


def build_goal_prior_context(
    model,
    conditioner: ObjectGoalConditioner,
    history: dict[str, torch.Tensor],
    goal: dict[str, torch.Tensor],
    history_scale: torch.Tensor,
    future_scale: torch.Tensor,
    goal_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build context from history, explicit physical times, and image goal only."""
    if not model.config.object_aligned_actions:
        raise ValueError("image-goal context requires object-aligned actions")
    base = model.prior_context(
        history,
        future_scale,
        history_scale,
        condition=None,
    )
    if base.ndim != 4:
        raise ValueError("object-aligned Prior context must have shape [B,Q,K,D]")
    goal_tokens = conditioner(
        history["slots"][:, -1],
        history["center"][:, -1],
        history["activity"][:, -1],
        goal["slots"],
        goal["center"],
        goal["activity"],
        goal_scale,
        goal.get("current_rgb"),
        goal.get("rgb"),
    )
    if goal_tokens.shape != (base.shape[0], base.shape[2], base.shape[3]):
        raise ValueError("goal tokens do not align with Prior context")
    return base + goal_tokens[:, None], goal_tokens
