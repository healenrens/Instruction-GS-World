"""SIGReg — Sketch Isotropic Gaussian Regularizer (ported from LeJEPA / le-wm, the reference project's
sigreg.py). Anti-collapse WITHOUT EMA / frozen target / predictor asymmetry: draw num_proj random unit
1-D projections of the embeddings, minimize an Epps-Pulley (characteristic-function) normality statistic
vs N(0,1). Every 1-D projection -> N(0,1) ⇒ (Cramér-Wold) the whole distribution is isotropic Gaussian,
which cannot collapse (collapse = zero variance). The reference found EMA causes TEMPORAL collapse here,
so this is the chosen anti-collapse term (PLAN §3, §8)."""
import torch
from torch import nn


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb (..., D) -> scalar (lower = more isotropic-Gaussian)."""
        x = emb.reshape(-1, emb.shape[-1]).float()
        x = x - x.mean(0, keepdim=True)
        A = torch.randn(x.shape[-1], self.num_proj, device=x.device, dtype=x.dtype)
        A = A.div_(A.norm(p=2, dim=0).clamp_min(1e-8))
        proj = x @ A
        proj = proj / proj.std(0, keepdim=True).clamp_min(1e-6)
        x_t = proj.unsqueeze(-1) * self.t
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()
        statistic = (err @ self.weights) * proj.shape[0]
        return statistic.mean()
