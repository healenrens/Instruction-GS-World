"""Verify (1) gsplat gradients reach GaussianSet params, (2) Qwen3-VL encoder works."""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.gaussians import GaussianSet, render_gaussianset  # noqa: E402


def test_gsplat_grad():
    dev = "cuda"
    n = 2000
    means = (torch.randn(n, 3, device=dev) * 0.3 + torch.tensor([0, 0, 2.0], device=dev)).requires_grad_(True)
    quats = torch.tensor([1.0, 0, 0, 0], device=dev).repeat(n, 1).requires_grad_(True)
    scales = (torch.rand(n, 3, device=dev) * 0.02 + 0.01).requires_grad_(True)
    opac = (torch.rand(n, device=dev) * 0.5 + 0.4).requires_grad_(True)
    cols = torch.rand(n, 3, device=dev).requires_grad_(True)
    gs = GaussianSet(means, quats, scales, opac, cols)

    H = W = 96
    K = torch.tensor([[80.0, 0, W / 2], [0, 80.0, H / 2], [0, 0, 1]], device=dev)
    viewmat = torch.eye(4, device=dev)
    colors, alphas, _ = render_gaussianset(gs, viewmat, K, W, H)
    target = torch.rand(1, H, W, 3, device=dev)
    loss = ((colors - target) ** 2).mean()
    loss.backward()
    grads = {"means": means.grad, "scales": scales.grad, "opac": opac.grad, "cols": cols.grad}
    for k, g in grads.items():
        finite = torch.isfinite(g).all().item()
        nz = (g.abs().sum() > 0).item()
        print(f"  [gsplat grad] {k:6s} finite={finite} nonzero={nz} |g|={g.abs().mean().item():.3e}")
        assert finite and nz, f"bad grad for {k}"
    print("  [gsplat grad] OK — render is differentiable wrt all Gaussian params\n")


def test_qwen():
    import numpy as np
    from igsw.dynamics.conditioning import QwenVLEncoder

    enc = QwenVLEncoder(device="cuda")
    print(f"  [qwen] loaded; hidden_size={enc.hidden_size}")
    txt = "Pickup items in the supermarket. Pick up the apple from the fruit stand."
    h, m = enc.encode(txt, image=None)
    print(f"  [qwen] text-only  hidden={tuple(h.shape)} dtype={h.dtype} mask_sum={int(m.sum())}")
    assert h.shape[1] == enc.hidden_size
    img = (np.random.rand(434, 574, 3) * 255).astype("uint8")
    h2, m2 = enc.encode(txt, image=img)
    print(f"  [qwen] image+text hidden={tuple(h2.shape)} mask_sum={int(m2.sum())} (longer => image tokens added)")
    assert h2.shape[0] > h.shape[0], "image tokens should extend the sequence"
    print("  [qwen] OK\n")


if __name__ == "__main__":
    print("== test gsplat gradient flow ==")
    test_gsplat_grad()
    print("== test Qwen3-VL encoder ==")
    test_qwen()
    print("[ALL OK]")
