"""Shape, gradient, stochasticity, and causal-interface tests for probe models."""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.latent_particle_wm.models import ParticleWorldModel, WorldModelConfig  # noqa: E402
from igsw.latent_particle_wm.objectives import particle_world_model_loss  # noqa: E402


def fake_batch() -> dict[str, torch.Tensor]:
    batch, particles = 3, 16
    target = torch.randn(batch, particles, 6) * 0.05
    valid = torch.rand(batch, particles) > 0.2
    visible = (torch.rand(batch, particles) > 0.3) & valid
    motion_valid = visible & (torch.rand(batch, particles) > 0.1)
    return {
        "state": torch.randn(batch, particles, 14),
        "target": target,
        "valid": valid,
        "visible": visible,
        "motion_valid": motion_valid,
        "horizon": torch.tensor([1, 6, 12]),
    }


def main() -> None:
    torch.manual_seed(3)
    batch = fake_batch()
    for kind in sorted(ParticleWorldModel.VALID_KINDS):
        config = WorldModelConfig(
            kind=kind,
            hidden_dim=32,
            layers=1,
            heads=4,
            local_latent_dim=4,
            global_latent_dim=6,
            mixture_components=3,
        )
        model = ParticleWorldModel(config)
        output = model(batch)
        assert output["prediction"].shape == (3, 16, 6)
        assert output["visibility_logits"].shape == (3, 16)
        loss, parts = particle_world_model_loss(
            model,
            batch,
            output,
            kl_weight=1e-3,
            free_bits=0.01,
            move_weight=2.0,
            appearance_weight=0.1,
            visibility_weight=0.1,
            effect_weight=0.1,
            usage_weight=0.1,
            alignment_weight=0.1,
        )
        loss.backward()
        assert all(
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        if kind == "global_flow":
            model.zero_grad(set_to_none=True)
            flow_output = model(batch)
            prior_loss, _ = model.prior_fitting_loss(flow_output, batch["valid"], 0.0)
            prior_loss.backward()
            assert any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.global_prior.parameters()
            )
        samples, visibility = model.predict_prior(batch, samples=4, sample=True)
        assert samples.shape == (4, 3, 16, 6)
        assert visibility.shape == (4, 3, 16)

        swapped = dict(batch)
        for key in ("target", "valid", "visible", "motion_valid"):
            swapped[key] = batch[key].flip(0)
        prior_a = model.prior_parameters(batch)
        prior_b = model.prior_parameters(swapped)
        difference = max(
            float((prior_a[key] - prior_b[key]).detach().abs().max()) for key in prior_a
        )
        assert difference == 0.0
        print(
            f"[latent-particle-test] {kind} loss={float(loss.detach()):.5f} "
            f"kl={float(parts['kl'].detach()):.5f} prior_future_diff={difference:.1f}"
        )
    print("[OK] all latent particle world-model structural tests passed")


if __name__ == "__main__":
    main()
