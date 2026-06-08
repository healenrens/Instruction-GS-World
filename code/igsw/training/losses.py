"""Training losses for Instruct-GS-World.

- Photometric = 0.8 * L1 + 0.2 * (1 - SSIM)   (3DGS / Feature-3DGS convention)
- SSIM uses an 11x11 Gaussian window (Wang et al. 2004), implemented faithfully.
- Delta / motion regularizers for stable autoregressive rollout (brief D).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    w2d = (g[:, None] * g[None, :])               # [ws,ws]
    return w2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """img1,img2: [B,3,H,W] in [0,1]. Returns mean SSIM (scalar)."""
    c = img1.shape[1]
    w = _gaussian_window(window_size, sigma, c, img1.device, img1.dtype)
    pad = window_size // 2
    mu1 = F.conv2d(img1, w, padding=pad, groups=c)
    mu2 = F.conv2d(img2, w, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, w, padding=pad, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, w, padding=pad, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, w, padding=pad, groups=c) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def photometric_loss(pred_hw3: torch.Tensor, gt_hw3: torch.Tensor, l1_w: float = 0.8, ssim_w: float = 0.2):
    """pred,gt: [H,W,3] (or [B,H,W,3]) in [0,1]. Returns (loss, l1, dssim)."""
    if pred_hw3.ndim == 3:
        pred_hw3 = pred_hw3[None]
        gt_hw3 = gt_hw3[None]
    l1 = (pred_hw3 - gt_hw3).abs().mean()
    p = pred_hw3.permute(0, 3, 1, 2).clamp(0, 1)
    g = gt_hw3.permute(0, 3, 1, 2).clamp(0, 1)
    dssim = 1.0 - ssim(p, g)
    return l1_w * l1 + ssim_w * dssim, l1.detach(), dssim.detach()


def delta_reg(v: torch.Tensor, omega: torch.Tensor, dlog_s: torch.Tensor,
              w_v: float = 1.0, w_w: float = 1.0, w_s: float = 1.0) -> torch.Tensor:
    """L2 penalty on per-step deltas (keeps motion minimal -> anti-drift)."""
    return w_v * (v ** 2).mean() + w_w * (omega ** 2).mean() + w_s * (dlog_s ** 2).mean()


def trajectory_loss(pred, gt, vis, init, relevance=None, obj_focus: float = 0.0, eps: float = 1e-6):
    """Direct 3D supervision of per-Gaussian motion (the primary loss).

    pred, gt: [K,M,3] predicted/ground-truth control positions over K steps
    vis: [K,M] bool visibility (occluded/out-of-frame points are not supervised)
    init: [M,3] shared initial positions (G0 = frame 0).
    relevance: optional [M] task-relevance in [0,1]; with obj_focus>0 the per-Gaussian
      weight becomes vis·(1 + obj_focus·relevance) → concentrates supervision on the
      task-relevant subset (Module-E L_obj_focus) without dropping the rest.
    Returns (pos_loss, vel_loss): masked L1 on position AND on per-step velocity.
    """
    visf = vis.float()                                          # [K,M]
    w = visf
    if relevance is not None and obj_focus > 0:
        w = visf * (1.0 + obj_focus * relevance.clamp(0, 1)[None])   # fg boosted, bg stays 1×
    w = w[..., None]                                             # [K,M,1]
    # normalize by VISIBLE count (not the boosted weight-sum) so obj_focus boosts foreground
    # ABOVE uniform WITHOUT diluting background supervision (fixes background drift).
    denom = visf.sum().clamp_min(eps) * 3.0
    pos_loss = (((pred - gt).abs()) * w).sum() / denom

    pred_full = torch.cat([init[None], pred], dim=0)             # [K+1,M,3]
    gt_full = torch.cat([init[None], gt], dim=0)
    pred_vel = pred_full[1:] - pred_full[:-1]                    # [K,M,3]
    gt_vel = gt_full[1:] - gt_full[:-1]
    vel_loss = (((pred_vel - gt_vel).abs()) * w).sum() / denom
    return pos_loss, vel_loss


def background_static_loss(traj, init, relevance, eps: float = 1e-6):
    """Module-E anti-drift: penalize ACCUMULATED displacement of LOW-relevance (background)
    control Gaussians from their start position (a real anti-drift signal, unlike the tiny
    per-step ‖v‖² which is redundant with the trajectory GT).
    traj: [K,M,3] predicted control positions; init: [M,3] start; relevance: [M] task relevance.
    L = Σ_i (1-rel_i)·‖pos_i(t) - init_i‖²  / (Σ bg · K)."""
    bg = (1.0 - relevance.clamp(0, 1))                        # [M]  (1=background)
    disp = (traj - init[None]).pow(2).sum(-1)                 # [K,M] squared drift from start
    return (bg[None] * disp).sum() / (bg.sum().clamp_min(eps) * traj.shape[0])


def kabsch_rotation(P0: torch.Tensor, P1: torch.Tensor) -> torch.Tensor:
    """Per-point local rigid rotation mapping neighbourhood P0->P1 (Kabsch/SVD).
    P0,P1: [...,k,3] -> R [...,3,3] with R @ (P0-c0) ≈ (P1-c1)."""
    c0 = P0.mean(dim=-2, keepdim=True); c1 = P1.mean(dim=-2, keepdim=True)
    A = P0 - c0; B = P1 - c1
    H = A.transpose(-1, -2) @ B                                   # [...,3,3]
    U, _, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    d = torch.det(V @ U.transpose(-1, -2))                        # [...]
    D = torch.eye(3, device=P0.device, dtype=P0.dtype).expand(*H.shape[:-2], 3, 3).clone()
    D[..., 2, 2] = torch.sign(d)
    return V @ D @ U.transpose(-1, -2)


def rotation_loss(pred_omega, gt_traj_full, knn_idx, vis_full, min_vis_frac: float = 0.5, eps: float = 1e-6):
    """Direct per-Gaussian ROTATION supervision (completes the 'transformation' target).

    pred_omega: [K,M,3] predicted per-step axis-angle (from the dynamics deltas).
    gt_traj_full: [K+1,M,3] GT control trajectory incl. init at index 0.
    knn_idx: [M,k] local neighbours (control indices) for the Kabsch fit.
    vis_full: [K+1,M] visibility. Returns masked chordal (Frobenius) rotation loss.
    """
    from ..dynamics.manifold import axis_angle_to_quat, quat_to_rotmat
    K = pred_omega.shape[0]
    Rp = quat_to_rotmat(axis_angle_to_quat(pred_omega))          # [K,M,3,3]
    tot = pred_omega.new_zeros(()); cnt = pred_omega.new_zeros(())
    for t in range(K):
        P0 = gt_traj_full[t][knn_idx]                            # [M,k,3]
        P1 = gt_traj_full[t + 1][knn_idx]
        Rg = kabsch_rotation(P0, P1).to(Rp.dtype)               # [M,3,3]
        v0 = vis_full[t][knn_idx].float().mean(-1) > min_vis_frac
        v1 = vis_full[t + 1][knn_idx].float().mean(-1) > min_vis_frac
        m = (v0 & v1).float()                                    # [M]
        diff = ((Rp[t] - Rg) ** 2).sum(dim=(-1, -2))            # [M] Frobenius^2
        tot = tot + (diff * m).sum(); cnt = cnt + m.sum()
    return tot / cnt.clamp_min(eps)


def contrastive_lang_loss(correct_v, wrong_v, gt_v, vis, margin: float = 0.003, eps: float = 1e-6):
    """Force the model to USE the instruction: the step-0 motion predicted under the CORRECT
    instruction must fit the GT motion at least `margin` better than under a WRONG instruction.
    correct_v, wrong_v, gt_v: [M,3]; vis: [M] bool. Hinge, masked by visibility."""
    e_correct = ((correct_v - gt_v) ** 2).sum(-1)         # [M]
    e_wrong = ((wrong_v - gt_v) ** 2).sum(-1)
    w = vis.float()
    hinge = torch.relu(margin + e_correct - e_wrong) * w
    return hinge.sum() / w.sum().clamp_min(eps)


def scale_anchor_loss(scales_traj, init_scales, eps: float = 1e-8):
    """Anti-drift for LONG-HORIZON rollout. The dynamics applies s←s·exp(δs) each step; scales are
    otherwise UNSUPERVISED (trajectory_loss only constrains position), so the model learns a
    consistent positive δs (bigger Gaussians fill render gaps within the K-horizon) → scales compound
    EXPONENTIALLY at extrapolation (measured 0.013→383575 over 40 steps) → giant Gaussians → render
    collapse ≈5s. Real objects barely change apparent size over a few seconds, so we anchor the
    rolled-out scales to G0 in LOG space (symmetric for grow/shrink, scale-invariant). A modest weight
    lets genuine GT-driven changes through while killing the runaway growth.
    scales_traj: [K,N,3] dense scales over the rollout; init_scales: [N,3] G0 scales."""
    ls = torch.log(scales_traj.clamp_min(eps))
    l0 = torch.log(init_scales.clamp_min(eps))[None]            # [1,N,3]
    return ((ls - l0) ** 2).mean()


def contrastive_infonce(g, t, q_g, q_t, tau: float = 0.07):
    """Symmetric CLIP-style InfoNCE forcing PREDICTED motion to be instruction-specific
    (non-saturating MI lower bound, research_F #1 — replaces the saturating hinge above).
    g,t: [proj] current clip's (predicted-motion, instruction) unit-norm embeddings (with grad).
    q_g,q_t: [Q,proj] DETACHED queues of OTHER clips' (motion, instruction) embeddings = negatives.
    ~0 until the queue fills."""
    if q_t is None or q_t.shape[0] == 0:
        return g.new_zeros(())
    pos = (g * t).sum()                                       # motion ↔ its own instruction
    logits_g = torch.cat([pos[None], (g[None] * q_t).sum(-1)], 0) / tau   # motion-anchor
    logits_t = torch.cat([pos[None], (t[None] * q_g).sum(-1)], 0) / tau   # language-anchor
    return 0.5 * (-torch.log_softmax(logits_g, 0)[0] - torch.log_softmax(logits_t, 0)[0])


def velocity_smoothness(traj: list[torch.Tensor]) -> torch.Tensor:
    """traj = list of means [N,3] over steps. Penalize acceleration ||x_{t+1}-2x_t+x_{t-1}||."""
    if len(traj) < 3:
        return traj[0].new_zeros(())
    acc = 0.0
    for t in range(1, len(traj) - 1):
        acc = acc + ((traj[t + 1] - 2 * traj[t] + traj[t - 1]) ** 2).mean()
    return acc / (len(traj) - 2)
