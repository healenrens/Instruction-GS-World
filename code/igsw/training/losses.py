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


def mover_magnitude_loss(pred, gt, init, vis, mover_thresh: float = 0.01, eps: float = 1e-3,
                         steps=None):
    """RELATIVE-magnitude loss on the movers' total displacement — fights the L1 heavy-tailed
    UNDER-prediction (the stuck top-mover ratio ~0.46). plain L1 on positions is minimized by the
    MEDIAN, so a few large movers (the cube) get systematically under-predicted no matter how much
    obj_focus up-weights them (the weight scales the L1, but L1's argmin is still the median).
    Here, for each control whose GT displacement (frame0->K) exceeds `mover_thresh`, we penalize the
    FRACTIONAL magnitude error  |‖pred_disp‖ - ‖gt_disp‖| / (‖gt_disp‖ + eps)  — so a 50% under-shoot
    of a LARGE mover costs as much as of a small one, directly pulling ‖pred_disp‖ -> ‖gt_disp‖ (the
    ratio metric). Symmetric (over- and under-shoot both penalized), magnitude-only (direction stays
    supervised by the L1 trajectory loss). pred,gt: [K,M,3]; init: [M,3]; vis: [K,M]."""
    K = pred.shape[0]
    # §54: evaluate the relative-magnitude error at MULTIPLE steps (default {K//2, K-1}) so the MID
    # trajectory is constrained too — endpoint-only let the mover lag mid-rollout then snap at the end.
    steps = steps if steps is not None else sorted({max(1, K // 2), K})
    tot = pred.new_zeros(()); cnt = 0
    for k in steps:
        if k < 1 or k > K:
            continue
        pred_disp = (pred[k - 1] - init).norm(dim=-1)          # [M] predicted disp magnitude @ step k
        gt_disp = (gt[k - 1] - init).norm(dim=-1)              # [M] GT disp magnitude @ step k
        mover = (gt_disp > mover_thresh) & vis[k - 1].bool()
        if int(mover.sum()) < 1:
            continue
        tot = tot + ((pred_disp[mover] - gt_disp[mover]).abs() / (gt_disp[mover] + eps)).mean(); cnt += 1
    return tot / max(cnt, 1)


def background_static_loss(traj, init, relevance, eps: float = 1e-6):
    """Module-E anti-drift: penalize ACCUMULATED displacement of LOW-relevance (background)
    control Gaussians from their start position (a real anti-drift signal, unlike the tiny
    per-step ‖v‖² which is redundant with the trajectory GT).
    traj: [K,M,3] predicted control positions; init: [M,3] start; relevance: [M] task relevance.
    L = Σ_i (1-rel_i)·‖pos_i(t) - init_i‖²  / (Σ bg · K)."""
    bg = (1.0 - relevance.clamp(0, 1))                        # [M]  (1=background)
    disp = (traj - init[None]).pow(2).sum(-1)                 # [K,M] squared drift from start
    return (bg[None] * disp).sum() / (bg.sum().clamp_min(eps) * traj.shape[0])


def mover_bce_loss(p_dyn_logit, mover_label, vis=None, eps: float = 1e-6):
    """Exp-1: supervise the per-control DYNAMICS GATE with the FREE sim mover label.

    p_dyn_logit: [M] raw logit from the dyn-gate head (out["p_dyn"]).
    mover_label: [M] float/bool — 1 where the control's GT clip displacement > thresh (a mover),
      0 where it is static. Computed at TRAIN time from `traj` only; the head predicts it from the
      Qwen patch feature, so at inference NO GT is needed (the whole point of the gate).
    vis: optional [M] bool to mask controls that are never observed (sim = all visible -> None).
    Returns the (masked, mean) binary cross-entropy. This is the explicit move/stay signal the
    regression L1 never provided (notes Exp-1 §2.1/§2.2) and a classification target that does NOT
    suffer the heavy-tailed-displacement median-collapse (so it generalizes; FlowBot3D/GAMMA recipe)."""
    target = mover_label.float().reshape(-1)
    logit = p_dyn_logit.reshape(-1)
    per = F.binary_cross_entropy_with_logits(logit, target, reduction="none")   # [M]
    if vis is not None:
        w = vis.float().reshape(-1)
        return (per * w).sum() / w.sum().clamp_min(eps)
    return per.mean()


def semantic_id_loss(e_sem, seg_ids, knn_idx, sem_proto=None, n_proto: int = 64, tau: float = 0.1,
                     w_nn: float = 0.1, eps: float = 1e-6):
    """Exp-1 #3 (OPTIONAL): per-control object-identity loss from the free `seg_per_g` entity ids.

    e_sem: [M,S] per-control object embedding (out["e_sem"]).
    seg_ids: [M] long entity id per control (from clip["seg_per_g"][ctrl_idx]).
    knn_idx: [M,k] local-neighbour control indices (reuse the trainer's knn_idx).
    sem_proto: [n_classes,S] LEARNABLE class-prototype bank (out["sem_proto"]); row `id` is entity
      `id`'s prototype (indexed by the RAW sparse entity id, so a clip with ids {1,3,16} uses rows
      1/3/16). This replaces the old DETACHED batch-mean prototypes.

    Two terms (Gaussian-Grouping arXiv:2312.00732 + OpenGaussian):
      (a) prototype CE: each control's normalized embedding is classified (cosine logits / tau) against
          the LEARNABLE prototypes; CE target = the control's raw entity id. Both e_sem AND the
          prototypes get gradient -> a real, stable identity classifier that pulls each control to its
          entity and pushes from others (cannot collapse).
      (b) 3D-NN consistency: pull k-NN controls' embeddings together (cosine), so spatially-adjacent
          Gaussians of the same object share identity (the unsupervised regularizer).
    Returns the summed loss (scalar). No external model — `seg_per_g` is the free sim label.

    BUG HISTORY: prototypes used to be the batch-mean of `e_sem` then `.detach()`. With the head
    zero-init -> e_sem=0 -> protos=0 -> logits=0 (uniform softmax, CE=log#entities) AND ∂logits/∂z =
    protos.t() = 0, so the gradient to the head was EXACTLY zero — a dead saddle that pinned the loss
    at ~2.7 forever. Learnable prototypes + a non-zero head init fix it."""
    z = F.normalize(e_sem.float(), dim=-1)                       # [M,S]
    seg_ids = seg_ids.long().reshape(-1)
    # §54: fold the per-clip arm SUB-PART ids (50+c, assigned by per-clip k-means order) back to the
    # arm id 8 so the prototype bank gets a CONSISTENT "arm" target across clips (id 51 is a different
    # physical part in clip A vs B -> contradictory CE that kept seg loss high in v7).
    seg_ids = torch.where(seg_ids >= 50, torch.full_like(seg_ids, 8), seg_ids)
    ce = z.new_zeros(())
    if sem_proto is not None:
        # (a) CE against the LEARNABLE prototype bank, indexed by the RAW entity id (stable across steps).
        protos = F.normalize(sem_proto.float(), dim=-1)          # [n_classes,S] (WITH grad)
        logits = (z @ protos.t()) / tau                          # [M,n_classes]
        ce = F.cross_entropy(logits, seg_ids.clamp(max=protos.shape[0] - 1))
    else:
        # Fallback (legacy, no learnable bank passed): cosine-logit CE against the per-batch entities,
        # targets = dense column index. Kept so older callers still run; the trainer passes sem_proto.
        uniq = torch.unique(seg_ids)
        remap = {int(s): i for i, s in enumerate(uniq.tolist())}
        tgt = torch.tensor([remap[int(s)] for s in seg_ids.tolist()], device=z.device)
        # one normalized embedding per entity acts as the (with-grad) anchor row
        anchors = F.normalize(torch.stack([z[seg_ids == s].mean(0) for s in uniq.tolist()], 0), dim=-1)
        logits = (z @ anchors.t()) / tau
        ce = F.cross_entropy(logits, tgt)
    # (b) 3D-NN consistency (pull neighbours together in cosine space)
    nn = z[knn_idx]                                               # [M,k,S]
    nn_cos = (z[:, None, :] * nn).sum(-1)                         # [M,k]
    nn_loss = (1.0 - nn_cos).mean()
    return ce + w_nn * nn_loss


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


def entity_rigidity_loss(pred_ctrl, init, seg_ctrl, vis=None, min_pts: int = 4,
                         max_entities: int = 24, eps: float = 1e-6):
    """§49 RIGID-CONSENSUS prior — the learning-method fix for "the Gaussians explode/diffuse
    instead of MOVING" (user review of libero_v4).

    The per-control velocity head regresses each control INDEPENDENTLY; the L1 trajectory loss is
    satisfiable by moving a fraction of an object's controls fully while the rest lag — which
    renders as the object smearing/exploding along the path. Our GT is BY CONSTRUCTION rigid per
    entity (per-object rigid PnP / sim rigid bodies), so we add the matching structural prior:
    for every entity, the predicted control endpoints must agree with the BEST-FIT rigid motion
    of that entity's own points (differentiable Kabsch on the predictions themselves). The loss
    is the residual to the fitted SE(3) — invariant to WHAT the rigid motion is (does not fight
    pos/magnitude supervision), penalizing only intra-entity inconsistency (the spread).

    pred_ctrl [K,M,3] predicted control positions; init [M,3] frame-0 positions;
    seg_ctrl [M] long entity ids; vis [K,M] optional. Evaluated at mid + final steps.
    """
    Kk = pred_ctrl.shape[0]
    steps = sorted({Kk // 2, Kk - 1})
    ids, counts = torch.unique(seg_ctrl, return_counts=True)
    ids = ids[counts >= min_pts]
    if ids.numel() > max_entities:                       # cap cost; keep the largest entities
        order = torch.argsort(counts[counts >= min_pts], descending=True)[:max_entities]
        ids = ids[order]
    if ids.numel() == 0:
        return pred_ctrl.new_zeros(())
    tot = pred_ctrl.new_zeros(()); cnt = 0
    for e in ids.tolist():
        m = seg_ctrl == e
        X = init[m].float()                              # [P,3] frame-0
        if X.shape[0] < min_pts:
            continue
        for k in steps:
            if vis is not None:
                mv = vis[k][m].bool()
                if int(mv.sum()) < min_pts:
                    continue
                Xk = X[mv]; Yk = pred_ctrl[k][m].float()[mv]
            else:
                Xk = X; Yk = pred_ctrl[k][m].float()
            R = kabsch_rotation(Xk[None], Yk[None])[0]   # differentiable best-fit rotation
            t = Yk.mean(0) - Xk.mean(0) @ R.T
            res = (Yk - (Xk @ R.T + t)).norm(dim=-1)     # residual to the entity's OWN rigid fit
            tot = tot + res.mean(); cnt += 1
    return tot / max(cnt, 1)


def relevance_bce_loss(p_rel, is_obj_ctrl, objmask, eps: float = 1e-6):
    """§54 the language-causal supervision: push the per-control relevance logit r toward `is_obj`
    (the instruction-named object's controls = 1, the OTHER objects = 0), ONLY over object-class
    controls (objmask = 1 for seg ids 1..7). Arm/gripper/basket/background get NO relevance signal —
    language selects among the manipulable OBJECTS, not whether the robot acts.
    p_rel / is_obj_ctrl / objmask: [M]."""
    w = objmask.float().reshape(-1)
    if float(w.sum()) < 1:
        return p_rel.new_zeros(())
    tgt = is_obj_ctrl.float().reshape(-1)
    per = F.binary_cross_entropy_with_logits(p_rel.reshape(-1), tgt, reduction="none")  # [M]
    return (per * w).sum() / w.sum().clamp_min(eps)


def counterfactual_gate_loss(p_dyn_wrong, p_rel_wrong, is_obj_ctrl, eps: float = 1e-6):
    """§54 the LOAD-BEARING loss (breaks the gripper-proximity shortcut): under a WRONG instruction
    the TRUE-named object must NOT be selected. Drive BOTH the pooled gate logit (visual+language)
    AND the raw relevance toward 0 on the named object's controls (is_obj=1). Because the patch
    features are identical to the true pass and only the text K/V differ, satisfying this is
    impossible for a vision-only head — it forces r = f(patch, TEXT). p_*_wrong / is_obj_ctrl: [M]."""
    m = is_obj_ctrl.float().reshape(-1)
    if float(m.sum()) < 1:
        return p_dyn_wrong.new_zeros(())
    zeros = torch.zeros_like(m)
    lg = F.binary_cross_entropy_with_logits(p_dyn_wrong.reshape(-1), zeros, reduction="none")
    lr = F.binary_cross_entropy_with_logits(p_rel_wrong.reshape(-1), zeros, reduction="none")
    return ((lg + lr) * m).sum() / m.sum().clamp_min(eps)
