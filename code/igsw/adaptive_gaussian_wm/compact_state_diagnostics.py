"""Low-overhead diagnostics for compact object-region states."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _effective_rank(feature: torch.Tensor) -> tuple[torch.Tensor, int]:
    matrix = feature.float().reshape(-1, feature.shape[-1])
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    sampled = matrix[:, ::4]
    singular = torch.linalg.svdvals(sampled)
    probability = singular / singular.sum().clamp_min(1e-8)
    entropy = -(probability * probability.clamp_min(1e-8).log()).sum()
    return entropy.exp(), min(sampled.shape)


def _ema_distance(online, target) -> torch.Tensor:
    total = None
    count = 0
    for online_parameter, target_parameter in zip(
        online.parameters(), target.parameters(), strict=True
    ):
        if not online_parameter.requires_grad:
            continue
        online_sample = online_parameter.float().flatten()[::4096]
        target_sample = target_parameter.float().flatten()[::4096]
        value = (online_sample - target_sample).square().mean()
        total = value if total is None else total + value
        count += 1
        if count == 8:
            break
    if total is None:
        raise ValueError("EMA distance has no parameters")
    return (total / count).sqrt()


@torch.no_grad()
def compact_state_diagnostics(model, batch: dict, output: dict) -> dict:
    online = output["online"]
    regions = online["regions"]
    owner = regions["owner"][:, -1]
    presence = regions["presence"][:, -1]
    active = presence > 0.5
    active_weight = active.float()
    denominator = active_weight.sum().clamp_min(1.0)
    scene = owner[..., -2] * active_weight
    transient = owner[..., -1] * active_weight
    object_owner = owner[..., :-2].sum(dim=-1) * active_weight
    effective_rank, usable_rank = _effective_rank(
        regions["feature"][:, -1][active]
    )
    diagnostics = {
        "region_active_count": active_weight.sum(dim=1).mean(),
        "region_active_count_std": active_weight.sum(dim=1).std(unbiased=False),
        "region_min_budget_fraction": (
            active_weight.sum(dim=1) == 64
        ).float().mean(),
        "region_max_budget_fraction": (
            active_weight.sum(dim=1) == 256
        ).float().mean(),
        "region_scene_fraction": scene.sum() / denominator,
        "region_transient_fraction": transient.sum() / denominator,
        "region_object_fraction": object_owner.sum() / denominator,
        "root_environment_weight_mean": (
            online["root_environment_weight"][:, -1] * active_weight
        ).sum()
        / denominator,
        "region_presence_mean": presence.mean(),
        "region_visibility_mean": regions["visibility"][:, -1].mean(),
        "region_association_confidence": regions[
            "association_confidence"
        ][:, -1].mean(),
        "region_effective_rank": effective_rank,
        "dino_valid_patch_fraction": target_valid_fraction(
            output["target"]["valid"], batch["history_length"]
        ),
        "dino_ema_parameter_distance": _ema_distance(
            model.online_dino, model.target_dino
        ),
        "region_transformer_ema_distance": _ema_distance(
            model.region_transformer, model.target_region_transformer
        ),
    }
    diagnostics["region_effective_rank_fraction"] = diagnostics[
        "region_effective_rank"
    ] / max(1, usable_rank)
    diagnostics["goal_valid_fraction"] = output["future_horizon_valid"][
        :, 1
    ].float().mean()
    roots = online["roots"]
    root_association = roots["association_matrix"][:, -1]
    row_mass = root_association.sum(dim=-1) + roots[
        "association_unmatched"
    ][:, -1]
    column_mass = root_association.sum(dim=-2) + roots[
        "association_discovery"
    ][:, -1]
    diagnostics.update(
        root_presence_mean=roots["existence"][:, -1].mean(),
        root_visibility_mean=roots["visibility"][:, -1].mean(),
        root_correspondence_confidence=roots["association_match"][:, -1].mean(),
        root_correspondence_entropy=roots["association_entropy"][:, -1].mean(),
        correspondence_row_mass_max_error=(row_mass - 1.0).abs().max(),
        correspondence_column_mass_max_error=(column_mass - 1.0).abs().max(),
    )
    history_length = int(batch["history_length"][0].item())
    diagnostics[f"history_h{history_length}_region_presence"] = presence.mean()
    if history_length > 1:
        inputs = online["transformer_inputs"]
        activation = torch.stack(
            [state.activation.squeeze(-1) for state in online["token_states"]], dim=1
        )
        permutation = torch.cat(
            (
                torch.arange(
                    history_length - 2,
                    -1,
                    -1,
                    device=inputs.device,
                ),
                torch.tensor([history_length - 1], device=inputs.device),
            )
        )
        reversed_context = model.region_transformer._run(
            inputs[:, permutation], (activation[:, permutation] > 0.5)
        )[:, -1]
        target_current = output["target"]["contextual_regions"][:, history_length - 1]
        target_previous = output["target"]["contextual_regions"][
            :, history_length - 2
        ]
        motion = (
            1.0
            - F.cosine_similarity(
                target_current.float(), target_previous.float(), dim=-1
            )
        ).clamp_min(0.0)
        motion = motion * output["target"]["regions"]["presence"][
            :, history_length - 1
        ]
        ordered = 1.0 - F.cosine_similarity(
            online["contextual_regions"][:, -1].float(),
            target_current.float(),
            dim=-1,
        )
        reversed_value = 1.0 - F.cosine_similarity(
            reversed_context.float(), target_current.float(), dim=-1
        )
        ordered_error = (ordered * motion).sum() / motion.sum().clamp_min(1e-6)
        reversed_error = (reversed_value * motion).sum() / motion.sum().clamp_min(
            1e-6
        )
        diagnostics.update(
            temporal_ordered_error=ordered_error,
            temporal_reversed_error=reversed_error,
            temporal_order_relative_gain=(reversed_error - ordered_error)
            / reversed_error.clamp_min(1e-6),
        )
        hidden = (
            regions["presence"][:, -2] > 0.5
        ) & (regions["visibility"][:, -1] < 0.1)
        identity_similarity = F.cosine_similarity(
            regions["identity_key"][:, -2].float(),
            regions["identity_key"][:, -1].float(),
            dim=-1,
        )
        diagnostics["region_unobserved_identity_persistence"] = (
            identity_similarity * hidden.float()
        ).sum() / hidden.float().sum().clamp_min(1.0)
    if output["root_prediction"] is not None:
        current_root = online["roots"]["slots"][:, -1]
        current_region = regions["feature"][:, -1]
        diagnostics.update(
            short_predicted_root_delta_rms=(
                output["root_prediction"].future_slots[:, 0] - current_root
            ).float().square().mean().sqrt(),
            short_target_root_delta_rms=(
                output["target_short_root"].slots
                - output["target"]["roots"]["slots"][:, history_length - 1]
            ).float().square().mean().sqrt(),
            short_predicted_region_delta_rms=(
                output["region_prediction"].future_feature[:, 0] - current_region
            ).float().square().mean().sqrt(),
            short_target_region_delta_rms=(
                output["target_short_region"].feature
                - output["target"]["regions"]["feature"][:, history_length - 1]
            ).float().square().mean().sqrt(),
            posterior_effect_rms=output["short_action"].float().square().mean().sqrt(),
            posterior_tail_effect_rms=output["tail_action"]
            .float()
            .square()
            .mean()
            .sqrt(),
            posterior_composed_effect_rms=output["composed_action"]
            .float()
            .square()
            .mean()
            .sqrt(),
        )
    return diagnostics


def target_valid_fraction(valid: torch.Tensor, history_length: torch.Tensor):
    index = int(history_length[0].item()) - 1
    return valid[:, index].float().mean()
