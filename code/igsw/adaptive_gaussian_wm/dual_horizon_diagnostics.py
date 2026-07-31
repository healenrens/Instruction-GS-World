"""Distributed diagnostics for hierarchical continuous latent effects."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _gather_batch(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().float()
    if not dist.is_available() or not dist.is_initialized():
        return value
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    return torch.cat(gathered, dim=0)


def _masked_codes(actions: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    selected = actions[valid]
    if selected.numel() == 0:
        return actions.new_zeros((0, actions.shape[-2] * actions.shape[-1]))
    return selected.flatten(1)


def _code_health(prefix: str, codes: torch.Tensor) -> dict[str, torch.Tensor]:
    if codes.shape[0] == 0:
        zero = codes.new_zeros(())
        return {
            f"{prefix}_rms": zero,
            f"{prefix}_sample_norm": zero,
            f"{prefix}_variance_mean": zero,
            f"{prefix}_active_dimension_fraction": zero,
            f"{prefix}_effective_rank_fraction": zero,
        }
    centered = codes - codes.mean(dim=0, keepdim=True)
    variance = centered.square().mean(dim=0)
    singular = torch.linalg.svdvals(centered)
    spectrum = singular.square()
    energy = spectrum.sum()
    probability = spectrum / energy.clamp_min(1e-12)
    effective_rank = torch.where(
        energy > 1e-12,
        torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum()),
        energy.new_zeros(()),
    )
    maximum_rank = float(min(codes.shape))
    return {
        f"{prefix}_rms": codes.square().mean().sqrt(),
        f"{prefix}_sample_norm": codes.norm(dim=-1).mean(),
        f"{prefix}_variance_mean": variance.mean(),
        f"{prefix}_active_dimension_fraction": (variance > 1e-4).float().mean(),
        f"{prefix}_effective_rank_fraction": effective_rank / maximum_rank,
    }


@torch.no_grad()
def dual_horizon_effect_diagnostics(
    batch: dict[str, torch.Tensor],
    output: dict,
) -> dict[str, torch.Tensor]:
    """Measure posterior code use over the complete distributed microbatch."""
    if not output.get("dual_horizon", False):
        return {}
    short = _gather_batch(output["posterior_short_effect"])
    tail = _gather_batch(output["posterior_tail_effect"])
    composed = _gather_batch(output["posterior_composed_goal_effect"])
    valid = _gather_batch(batch["future_horizon_valid"].float()) > 0.5
    short_codes = _masked_codes(short, valid[:, 0])
    tail_codes = _masked_codes(tail, valid[:, 1])
    composed_codes = _masked_codes(composed, valid[:, 1])
    result = {}
    result.update(_code_health("dual_horizon_short_effect", short_codes))
    result.update(_code_health("dual_horizon_tail_effect", tail_codes))
    result.update(_code_health("dual_horizon_composed_effect", composed_codes))

    paired = valid.all(dim=1)
    paired_short = short[paired].flatten(1)
    paired_tail = tail[paired].flatten(1)
    paired_composed = composed[paired].flatten(1)
    if paired_short.shape[0] == 0:
        zero = short.new_zeros(())
        result.update(
            dual_horizon_effect_valid_samples=zero,
            dual_horizon_short_tail_cosine=zero,
            dual_horizon_composition_delta_ratio=zero,
        )
        return result
    composition_delta = paired_composed - paired_short
    result.update(
        dual_horizon_effect_valid_samples=short.new_tensor(
            float(paired_short.shape[0])
        ),
        dual_horizon_short_tail_cosine=F.cosine_similarity(
            paired_short, paired_tail, dim=-1
        ).mean(),
        dual_horizon_composition_delta_ratio=(
            composition_delta.norm(dim=-1)
            / paired_tail.norm(dim=-1).clamp_min(1e-6)
        ).mean(),
    )
    return result
