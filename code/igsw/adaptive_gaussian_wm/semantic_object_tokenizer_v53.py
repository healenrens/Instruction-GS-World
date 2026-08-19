"""Object tokens learned from frozen semantic patches and temporal affinity."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .v53_config import SemanticObjectWorldModelConfig


@dataclass
class SemanticObjectEncoding:
    slots: torch.Tensor
    assignments: torch.Tensor
    reconstruction: torch.Tensor


class CompetitiveSlotAttention(nn.Module):
    def __init__(self, config: SemanticObjectWorldModelConfig):
        super().__init__()
        width = config.slot_dim
        self.iterations = config.slot_iterations
        self.scale = width**-0.5
        self.temperature = config.assignment_temperature
        self.patch_norm = nn.LayerNorm(width)
        self.slot_norm = nn.LayerNorm(width)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.query = nn.Linear(width, width, bias=False)
        self.update = nn.GRUCell(width, width)
        self.mlp = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
        )

    def forward(
        self,
        patches: torch.Tensor,
        slots: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys = self.key(self.patch_norm(patches))
        values = self.value(self.patch_norm(patches))
        mask = valid[:, None]
        for _ in range(self.iterations):
            previous = slots
            queries = self.query(self.slot_norm(slots))
            logits = torch.einsum("bkd,bpd->bkp", queries, keys) * self.scale
            logits = logits / self.temperature
            logits = logits.masked_fill(~mask, -1e4)
            competition = logits.softmax(dim=1) * mask
            normalized = competition / competition.sum(dim=2, keepdim=True).clamp_min(
                1e-6
            )
            updates = torch.einsum("bkp,bpd->bkd", normalized, values)
            slots = self.update(
                updates.flatten(0, 1), previous.flatten(0, 1)
            ).reshape_as(previous)
            slots = slots + self.mlp(slots)
        queries = self.query(self.slot_norm(slots))
        logits = torch.einsum("bkd,bpd->bkp", queries, keys) * self.scale
        logits = (logits / self.temperature).masked_fill(~mask, -1e4)
        assignments = logits.softmax(dim=1) * mask
        return slots, assignments


class LowRankCompositionalDecoder(nn.Module):
    def __init__(self, config: SemanticObjectWorldModelConfig):
        super().__init__()
        self.rank = config.decoder_rank
        self.patch_dim = config.patch_dim
        self.slot_base = nn.Linear(config.slot_dim, config.patch_dim)
        self.slot_coefficients = nn.Linear(config.slot_dim, config.decoder_rank)
        self.coordinate_basis = nn.Sequential(
            nn.Linear(2, config.slot_dim),
            nn.GELU(),
            nn.Linear(config.slot_dim, config.decoder_rank * config.patch_dim),
        )

    def forward(
        self,
        slots: torch.Tensor,
        assignments: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        base = self.slot_base(slots)
        coefficients = self.slot_coefficients(slots)
        basis = self.coordinate_basis(coordinates).reshape(
            *coordinates.shape[:-1], self.rank, self.patch_dim
        )
        reconstructed = torch.einsum("bkp,bkd->bpd", assignments, base)
        reconstructed = reconstructed + torch.einsum(
            "bkp,bkr,bprd->bpd", assignments, coefficients, basis
        )
        return F.normalize(reconstructed.float(), dim=-1, eps=1e-6)


class SemanticObjectTokenizer(nn.Module):
    def __init__(self, config: SemanticObjectWorldModelConfig):
        super().__init__()
        self.config = config
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(config.patch_dim),
            nn.Linear(config.patch_dim, config.slot_dim),
        )
        self.coordinate_projection = nn.Sequential(
            nn.Linear(2, config.slot_dim),
            nn.GELU(),
            nn.Linear(config.slot_dim, config.slot_dim),
        )
        self.object_initial = nn.Parameter(
            torch.randn(1, config.object_slots, config.slot_dim) * 0.02
        )
        self.scene_initial = nn.Parameter(
            torch.randn(1, config.scene_slots, config.slot_dim) * 0.02
        )
        self.object_type = nn.Parameter(
            torch.zeros(1, config.object_slots, config.slot_dim)
        )
        self.scene_type = nn.Parameter(
            torch.zeros(1, config.scene_slots, config.slot_dim)
        )
        self.attention = CompetitiveSlotAttention(config)
        self.decoder = LowRankCompositionalDecoder(config)
        self.output_norm = nn.LayerNorm(config.slot_dim)

    def initial_slots(self, batch_size: int) -> torch.Tensor:
        objects = self.object_initial.expand(batch_size, -1, -1) + self.object_type
        scene = self.scene_initial.expand(batch_size, -1, -1) + self.scene_type
        return torch.cat((objects, scene), dim=1)

    def forward(
        self,
        patches: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
    ) -> SemanticObjectEncoding:
        if patches.ndim != 4 or coordinates.shape[:3] != patches.shape[:3]:
            raise ValueError("v53 tokenizer expects [B,T,P,D] patches and coordinates")
        if valid.shape != patches.shape[:3]:
            raise ValueError("v53 tokenizer validity shape differs from patches")
        projected = self.patch_projection(patches.float()) + self.coordinate_projection(
            coordinates.float()
        )
        slots = self.initial_slots(patches.shape[0])
        states, assignments, reconstructions = [], [], []
        for frame in range(patches.shape[1]):
            slots, assignment = self.attention(
                projected[:, frame], slots, valid[:, frame]
            )
            slots = self.output_norm(slots)
            reconstruction = self.decoder(
                slots, assignment, coordinates[:, frame].float()
            )
            states.append(slots)
            assignments.append(assignment)
            reconstructions.append(reconstruction)
        return SemanticObjectEncoding(
            slots=torch.stack(states, dim=1),
            assignments=torch.stack(assignments, dim=1),
            reconstruction=torch.stack(reconstructions, dim=1),
        )


def temporal_affinity_loss(
    assignments: torch.Tensor,
    patches: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses, agreements = [], []
    for frame in range(patches.shape[1] - 1):
        source = F.normalize(patches[:, frame].float(), dim=-1, eps=1e-6)
        target = F.normalize(patches[:, frame + 1].float(), dim=-1, eps=1e-6)
        target_logits = torch.einsum("bpd,bqd->bpq", source, target) / temperature
        pair_valid = valid[:, frame, :, None] & valid[:, frame + 1, None, :]
        target_logits = target_logits.masked_fill(~pair_valid, -1e4)
        target_distribution = target_logits.softmax(dim=-1)
        predicted_affinity = torch.einsum(
            "bkp,bkq->bpq", assignments[:, frame], assignments[:, frame + 1]
        )
        predicted_distribution = predicted_affinity / predicted_affinity.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        predicted_log = predicted_distribution.clamp_min(1e-8).log()
        row_loss = -(target_distribution * predicted_log).sum(dim=-1)
        source_valid = valid[:, frame]
        losses.append((row_loss * source_valid).sum() / source_valid.sum().clamp_min(1))
        predicted_index = predicted_affinity.argmax(dim=-1)
        target_index = target_distribution.argmax(dim=-1)
        agreements.append(
            ((predicted_index == target_index) * source_valid).sum()
            / source_valid.sum().clamp_min(1)
        )
    return torch.stack(losses).mean(), {
        "temporal_affinity_top1": torch.stack(agreements).mean()
    }


def tokenizer_objective(
    config: SemanticObjectWorldModelConfig,
    encoding: SemanticObjectEncoding,
    patches: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cosine = 1.0 - (encoding.reconstruction * patches.float()).sum(dim=-1)
    reconstruction = (cosine * valid).sum() / valid.sum().clamp_min(1)
    affinity, affinity_metrics = temporal_affinity_loss(
        encoding.assignments, patches, valid, config.temporal_temperature
    )
    object_slots = encoding.slots[:, :, : config.object_slots]
    normalized = F.normalize(object_slots.float(), dim=-1, eps=1e-6)
    gram = torch.einsum("btkd,btjd->btkj", normalized, normalized)
    eye = torch.eye(config.object_slots, device=gram.device)[None, None]
    diversity = ((gram - eye) * (1.0 - eye)).square().mean()
    loss = (
        config.feature_reconstruction_weight * reconstruction
        + config.temporal_affinity_weight * affinity
        + config.slot_diversity_weight * diversity
    )
    mass = encoding.assignments[:, :, : config.object_slots].mean(dim=-1)
    normalized_mass = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    entropy = -(normalized_mass.clamp_min(1e-8).log() * normalized_mass).sum(dim=-1)
    effective = entropy.exp()
    parts = {
        "loss": loss.detach(),
        "object_feature_reconstruction_error": reconstruction.detach(),
        "object_temporal_affinity_loss": affinity.detach(),
        "object_slot_diversity_loss": diversity.detach(),
        "object_effective_slot_count": effective.mean().detach(),
        "object_max_slot_mass": normalized_mass.max(dim=-1).values.mean().detach(),
        "scene_assignment_fraction": encoding.assignments[:, :, -1].mean().detach(),
        **{name: value.detach() for name, value in affinity_metrics.items()},
    }
    return loss, parts
