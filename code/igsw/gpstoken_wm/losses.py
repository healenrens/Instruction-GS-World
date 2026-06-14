"""Losses for the GPSToken-JEPA world model (PLAN §3.4).
  L = L_geom (per-token 3D translation, LOAD-BEARING)
    + α·L_jepa (future-feature smooth-L1, auxiliary feature shaping)
    + β·SIGReg (anti-collapse, in sigreg.py)
    + γ·(L_ground InfoNCE + L_cf counterfactual)   (instruction grounding; treats motion≠relevance, §54)
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def geom_loss(xyz1_pred: torch.Tensor, xyz1_gt: torch.Tensor, weight: torch.Tensor | None = None,
              beta: float = 0.02):
    """Per-token 3D position smooth-L1 (clean per-Gaussian GT from sim traj). weight [M] optional
    (relevance up-weighting). Returns (loss, err_pred[m], err_static[m]) for skill logging."""
    per = F.smooth_l1_loss(xyz1_pred, xyz1_gt, beta=beta, reduction="none").mean(-1)   # [M]
    if weight is not None:
        loss = (per * weight).sum() / weight.sum().clamp_min(1e-6)
    else:
        loss = per.mean()
    return loss


def mover_magnitude(xyz1_pred: torch.Tensor, xyz0: torch.Tensor, xyz1_gt: torch.Tensor,
                    disp_tok: torch.Tensor, thresh: float = 0.01, eps: float = 1e-3):
    """RELATIVE displacement-magnitude loss on the movers — fights smooth-L1's median-seeking
    UNDER-prediction (the mag-ratio 0.32 collapse). |‖pred_disp‖−‖gt_disp‖|/‖gt_disp‖ on movers."""
    mv = disp_tok > thresh
    if mv.sum() == 0:
        return xyz1_pred.new_zeros(())
    pd = (xyz1_pred - xyz0)[mv].norm(dim=-1)
    gd = (xyz1_gt - xyz0)[mv].norm(dim=-1)
    return ((pd - gd).abs() / gd.clamp_min(eps)).mean()


def jepa_loss(feat_pred: torch.Tensor, feat_target: torch.Tensor, weight: torch.Tensor | None = None,
              beta: float = 0.5):
    """Future-feature JEPA: predicted vs frozen-encoded-future (target is stop-grad in the trainer)."""
    per = F.smooth_l1_loss(feat_pred, feat_target, beta=beta, reduction="none").mean(-1)
    if weight is not None:
        return (per * weight).sum() / weight.sum().clamp_min(1e-6)
    return per.mean()


def relevance_infonce(logits: torch.Tensor, pos_mask: torch.Tensor,
                      arm_mask: torch.Tensor | None = None, arm_weight: float = 1.0):
    """Multi-positive InfoNCE (ref temporal_wm): pull the instruction's target tokens above all others;
    arm tokens optionally up-weighted as harder negatives. logits [M], pos_mask [M] bool."""
    if pos_mask.sum() == 0 or pos_mask.all():
        return logits.new_zeros(())
    den = logits
    if arm_weight > 1.0 and arm_mask is not None:
        den = den + math.log(arm_weight) * arm_mask.float()
    return torch.logsumexp(den, 0) - torch.logsumexp(logits[pos_mask], 0)


def counterfactual_push(logits_wrong: torch.Tensor, pos_mask: torch.Tensor):
    """§54 counterfactual: under a WRONG instruction, the true-target tokens must NOT stay selected.
    Push their relevance logit down (same image + different text must flip => vision-only unsatisfiable)."""
    if pos_mask.sum() == 0:
        return logits_wrong.new_zeros(())
    return F.softplus(logits_wrong[pos_mask]).mean()


def kabsch_R(P: torch.Tensor, Q: torch.Tensor):
    """Best-fit rotation P->Q (both [n,3]) via SVD. EVAL-ONLY readout of emergent rotation (never trained)."""
    Pm, Qm = P.mean(0), Q.mean(0)
    Hmat = (P - Pm).T @ (Q - Qm)
    U, S, Vt = torch.linalg.svd(Hmat)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=P.device, dtype=P.dtype))
    return Vt.T @ D @ U.T
