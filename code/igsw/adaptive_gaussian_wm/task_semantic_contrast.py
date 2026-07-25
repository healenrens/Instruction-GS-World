"""Multi-positive task semantics for exact instruction-token features."""
from __future__ import annotations

import torch


def task_semantic_contrast(
    token_conditioner,
    token_features: torch.Tensor,
    token_valid: torch.Tensor,
    task_index: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if task_index.ndim != 1 or len(task_index) != len(token_features):
        raise ValueError("task index must align with instruction token bank")
    observed = task_index >= 0
    if int(observed.sum()) < 2:
        raise ValueError("task contrast requires observed instructions")
    embedding = token_conditioner.pooled_semantic(
        token_features[observed],
        token_valid[observed],
    )
    embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
    ids = task_index[observed]
    logits = embedding @ embedding.transpose(0, 1) / temperature
    diagonal = torch.eye(
        len(logits),
        device=logits.device,
        dtype=torch.bool,
    )
    positive = (ids[:, None] == ids[None]) & ~diagonal
    anchor = positive.any(dim=1)
    if not bool(anchor.any()):
        raise ValueError("task contrast requires a paraphrase pair")
    masked_logits = logits.masked_fill(diagonal, -torch.inf)
    numerator = torch.logsumexp(
        logits.masked_fill(~positive, -torch.inf),
        dim=-1,
    )
    denominator = torch.logsumexp(masked_logits, dim=-1)
    loss = (denominator[anchor] - numerator[anchor]).mean()
    nearest = masked_logits.argmax(dim=-1)
    retrieval = (ids[nearest[anchor]] == ids[anchor]).float().mean()
    return loss, {
        "task_semantic_retrieval": retrieval,
        "task_semantic_anchors": anchor.float().sum(),
    }
