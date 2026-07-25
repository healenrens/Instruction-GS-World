"""Optional multi-positive language-to-effect alignment fallback."""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .config import AdaptiveGaussianWMConfig
from .distributed_statistics import gather_batch_with_grad


def _gather_ids(value: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return value
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    return torch.cat(gathered, dim=0)


class LanguageEffectAlignment(nn.Module):
    """CLIP-style loss with every equal-instruction row treated as positive."""

    def __init__(self, config: AdaptiveGaussianWMConfig):
        super().__init__()
        projection_dim = min(256, config.model_dim)
        self.temperature = 0.07
        self.language_projection = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, projection_dim),
        )
        self.effect_projection = nn.Sequential(
            nn.LayerNorm(config.object_dim),
            nn.Linear(config.object_dim, projection_dim),
        )

    @staticmethod
    def _multi_positive_loss(
        logits: torch.Tensor,
        positive: torch.Tensor,
    ) -> torch.Tensor:
        negative_infinity = torch.finfo(logits.dtype).min
        numerator = torch.logsumexp(
            logits.masked_fill(~positive, negative_infinity),
            dim=-1,
        )
        denominator = torch.logsumexp(logits, dim=-1)
        return (denominator - numerator).mean()

    def forward(
        self,
        condition: torch.Tensor,
        effect: torch.Tensor,
        condition_index: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if effect.ndim != 3:
            raise ValueError("effect must have shape [B,Q,D]")
        if condition.shape[0] != effect.shape[0]:
            raise ValueError("condition and effect batch sizes differ")
        if condition_index.shape != (effect.shape[0],):
            raise ValueError("condition_index must have shape [B]")
        query_count = effect.shape[1]
        language = F.normalize(
            self.language_projection(condition),
            dim=-1,
        )[:, None].expand(-1, query_count, -1).flatten(0, 1)
        effect_embedding = F.normalize(
            self.effect_projection(effect),
            dim=-1,
        ).flatten(0, 1)
        ids = condition_index[:, None].expand(-1, query_count).flatten()
        language = gather_batch_with_grad(language)
        effect_embedding = gather_batch_with_grad(effect_embedding)
        ids = _gather_ids(ids)
        logits = effect_embedding.float() @ language.float().transpose(0, 1)
        logits = logits / self.temperature
        positive = ids[:, None] == ids[None]
        loss = 0.5 * (
            self._multi_positive_loss(logits, positive)
            + self._multi_positive_loss(logits.transpose(0, 1), positive.transpose(0, 1))
        )
        retrieval = (
            ids[logits.argmax(dim=-1)] == ids
        ).float().mean()
        return loss, {
            "language_effect_retrieval": retrieval,
            "language_effect_samples": logits.new_tensor(float(len(ids))),
        }
