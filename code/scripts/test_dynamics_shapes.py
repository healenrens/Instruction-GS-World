"""Shape / identity / backward test for GaussianDynamics (no real data needed).

Verifies:
  - forward step produces correctly-shaped next state,
  - at init the model is the IDENTITY (zero head + AdaLN-Zero => G_{t+1}==G_t),
  - autoregressive rollout runs,
  - a render-style loss backprops and produces finite grads.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.dynamics import GaussianDynamics, DynamicsConfig, GaussianState  # noqa: E402


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    B, N, L = 2, 16384, 24
    cfg = DynamicsConfig(d_model=512, n_heads=8, n_layers=12, lang_dim=2048)
    model = GaussianDynamics(cfg).to(dev)
    print(f"[model] params = {model.num_params()/1e6:.2f}M  d_model={cfg.d_model} layers={cfg.n_layers}")

    state = GaussianState(
        means=torch.randn(B, N, 3, device=dev),
        quats=torch.tensor([1.0, 0, 0, 0], device=dev).repeat(B, N, 1),
        scales=torch.rand(B, N, 3, device=dev) * 0.01 + 0.001,
        opacities=torch.rand(B, N, device=dev) * 0.5 + 0.25,
        colors=torch.rand(B, N, 3, device=dev),
    )
    lang = torch.randn(B, L, cfg.lang_dim, device=dev)
    lang_mask = torch.ones(B, L, dtype=torch.bool, device=dev)
    step = torch.zeros(B, dtype=torch.long, device=dev)

    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else torch.autocast("cpu")

    # --- identity at init ---
    with torch.no_grad(), amp:
        nxt = model.step(state, lang, lang_mask, step)
    dmean = (nxt.means - state.means).abs().max().item()
    dquat = (nxt.quats - state.quats).abs().max().item()
    dscale = (nxt.scales - state.scales).abs().max().item()
    dop = (nxt.opacities - state.opacities).abs().max().item()
    print(f"[identity@init] max|Δμ|={dmean:.2e} max|Δq|={dquat:.2e} "
          f"max|Δs|={dscale:.2e} max|Δσ|={dop:.2e}  (all should be ~0)")
    assert dmean < 1e-5 and dquat < 1e-5 and dscale < 1e-5 and dop < 1e-5, "not identity at init!"

    # --- shapes ---
    assert nxt.means.shape == (B, N, 3)
    assert nxt.quats.shape == (B, N, 4)
    print(f"[shapes] next means {tuple(nxt.means.shape)} quats {tuple(nxt.quats.shape)} OK")

    # --- rollout + backward through rollout (TBPTT-style) ---
    target = torch.randn(B, N, 3, device=dev)
    with amp:
        states = model.rollout(state, lang, lang_mask, n_steps=4)
        print(f"[rollout] produced {len(states)} states")
        loss = sum(((s.means - target) ** 2).mean() for s in states)
    loss.backward()
    gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None))
    finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(f"[backward] loss={loss.item():.4f} grad_norm={gnorm.item():.4f} all_finite={finite}")
    assert finite, "non-finite grads"

    # --- memory ---
    if dev == "cuda":
        print(f"[mem] peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB for B={B} N={N} L={L}")
    print("[OK] dynamics shape/identity/backward test passed.")


if __name__ == "__main__":
    main()
