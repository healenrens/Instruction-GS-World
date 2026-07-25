"""VLA trainer — JOINT (mixed) training of the 3D-flow head + π0-style action expert sharing the 1.6B DiT
trunk (agent.md §VLA). NO JEPA. Builds on the world-model trainer but:
  * adds the flow-matching ACTION EXPERT (predicts Δqpos[50,14]) on the SAME trunk (both losses backprop it),
  * reads ONLY frame0 (gt_rgb[0]) — no future frame, no JEPA target, no vlmK,
  * warm-starts the trunk + geom_head from checkpoints/wm_rt2/wm_002500.pt (expert trains from scratch),
  * loads data/rt2_joint/ clips (each has dq[50,14] + anchor[14] on top of the world-model keys),
  * action normalization from data/rt2_act/norm_stats.pt (arm dims standardized; gripper dims -> [-1,1]).

  torchrun --nproc_per_node=4 code/scripts/train_vla.py \
      --data data/rt2_joint --out checkpoints/vla_rt2 \
      --init_from checkpoints/wm_rt2/wm_002500.pt \
      --norm_stats data/rt2_act/norm_stats.pt \
      --geom_mode xyz --img_loss 1 --L 512 --steps 4000 \
      --w_flow 1.0 --w_act 1.0
  (geom_mode=xyz + img_loss=1 MATCH the warm-start ckpt wm_rt2/wm_002500.pt — keep them aligned.)

CPU smoke-test (tiny dims, fake clip, no GPU):
  python code/scripts/train_vla.py --smoke
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.gpstoken_wm import GPSTokenWM, place_tokens  # noqa: E402
from igsw.gaussians.gpstoken import mover_saliency  # noqa: E402


def mv_in(inputs, dev):
    return {k: (v.to(dev, dtype=torch.bfloat16) if (torch.is_tensor(v) and v.is_floating_point())
                else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in inputs.items()}


def build_batch(c, dev, args, enc):
    """rt2_joint clip dict -> prepared VLA batch (frame0-only; adds dq). Mirrors the world-model prep."""
    means = c["means"].to(dev).float(); uv = c["uv"].to(dev).float()
    traj = c["traj"].to(dev).float(); N = means.shape[0]
    H, W = int(c["H"]), int(c["W"]); K = int(c["Kf"])
    instr = c.get("instruction", ""); n_keep = N - int(c.get("n_fill", 0))
    disp = (traj[K] - traj[0]).norm(dim=-1)
    sal = mover_saliency(uv, disp, n_keep, H, W) if args.beta > 0 else None
    rgb0 = c["gt_rgb"][0].cpu().numpy().astype(np.uint8)               # frame0 ONLY (no future frame)
    cen, sig, idx = place_tokens(rgb0, uv, n_keep, args.L, dev, sal=sal, beta=args.beta)
    if idx.shape[0] < 16:
        raise ValueError("too few tokens")
    center = means[:n_keep].mean(0, keepdim=True)
    return {
        "vlm0": mv_in(enc.build_inputs(instr, rgb0), dev),
        "cen": cen, "sig_n": (sig / float(max(H, W))).clamp(0, 1),
        "tok_xyz0": means[idx], "xyz1_gt": traj[K][idx], "disp_tok": disp[idx],
        "traj_gt": (traj[1:K + 1][:, idx] if args.traj_pred else None),
        "center": center, "radius": (means[:n_keep] - center).norm(dim=-1).amax().clamp_min(1e-6),
        "K_intr": c["K_intr"].to(dev).float(), "viewmat": c["viewmat"].to(dev).float(),
        "H": H, "W": W, "rgb0_np": rgb0,
        "dq": c["dq"].to(dev).float(),                                 # [A,14] action target
    }


def build_clip_single(c, dev, args, enc):
    """Prepare ONE clip into the per-sample fields needed for the BATCHED path (no [None]-batching, no
    padding yet). Returns a dict of M-length tensors + scalars. build_batch_padded stacks/pads these."""
    means = c["means"].to(dev).float(); uv = c["uv"].to(dev).float()
    traj = c["traj"].to(dev).float(); N = means.shape[0]
    H, W = int(c["H"]), int(c["W"]); K = int(c["Kf"])
    instr = c.get("instruction", ""); n_keep = N - int(c.get("n_fill", 0))
    disp = (traj[K] - traj[0]).norm(dim=-1)
    sal = mover_saliency(uv, disp, n_keep, H, W) if args.beta > 0 else None
    rgb0 = c["gt_rgb"][0].cpu().numpy().astype(np.uint8)
    cen, sig, idx = place_tokens(rgb0, uv, n_keep, args.L, dev, sal=sal, beta=args.beta)
    M = idx.shape[0]
    if M < 16:
        raise ValueError("too few tokens")
    center = means[:n_keep].mean(0, keepdim=True)
    radius = (means[:n_keep] - center).norm(dim=-1).amax().clamp_min(1e-6)
    return {
        "vlm0": mv_in(enc.build_inputs(instr, rgb0), dev),
        "cen": cen, "sig_n": (sig / float(max(H, W))).clamp(0, 1),
        "tok_xyz0": means[idx], "xyz1_gt": traj[K][idx], "disp_tok": disp[idx],
        "center": center, "radius": radius, "M": M,
        "K_intr": c["K_intr"].to(dev).float(), "viewmat": c["viewmat"].to(dev).float(),
        "H": H, "W": W, "rgb0_np": rgb0,
        "dq": c["dq"].to(dev).float(), "anchor": c["anchor"].to(dev).float(),
    }


def build_batch_padded(clips, dev, args, enc, L=None):
    """Stack a list of prepared single clips into a BATCHED, padded batch (token axis padded to L with a
    boolean tok_mask). The action fields (dq[50,14], anchor[14]) are fixed-size -> stack directly.
    Per-clip K_intr/viewmat/H/W/grid kept as lists (the geom projection is per-clip). Padded token slots
    carry 0 and tok_mask=False -> the trunk self-attn masks them and the geom loss ignores them."""
    L = L or args.L
    B = len(clips)
    fdim_xyz = 3
    tok_xyz0 = torch.zeros(B, L, fdim_xyz, device=dev)
    cen = torch.zeros(B, L, 2, device=dev)
    sig_n = torch.zeros(B, L, 2, device=dev)
    xyz1_gt = torch.zeros(B, L, 3, device=dev)
    disp_tok = torch.zeros(B, L, device=dev)
    tok_mask = torch.zeros(B, L, dtype=torch.bool, device=dev)
    center = torch.zeros(B, 1, 3, device=dev)
    radius = torch.ones(B, 1, device=dev)
    dq = torch.zeros(B, args.action_steps, args.action_dim, device=dev)
    anchor = torch.zeros(B, args.action_dim, device=dev)
    vlm_list, grid0, K_list, vm_list, H_list, W_list = [], [], [], [], [], []
    for i, s in enumerate(clips):
        M = min(s["M"], L)
        tok_xyz0[i, :M] = s["tok_xyz0"][:M]; cen[i, :M] = s["cen"][:M]
        sig_n[i, :M] = s["sig_n"][:M]; xyz1_gt[i, :M] = s["xyz1_gt"][:M]
        disp_tok[i, :M] = s["disp_tok"][:M]; tok_mask[i, :M] = True
        center[i] = s["center"]; radius[i, 0] = s["radius"]
        dq[i] = s["dq"]; anchor[i] = s["anchor"]
        vlm_list.append(s["vlm0"])
        # per-clip frozen visual grid (computed once here, reused in the trunk)
        g0, ghw0 = enc.image_grid_features(s["vlm0"])
        grid0.append((g0, ghw0)); K_list.append(s["K_intr"]); vm_list.append(s["viewmat"])
        H_list.append(s["H"]); W_list.append(s["W"])
    return {
        "vlm_list": vlm_list, "grid0": grid0, "tok_xyz0": tok_xyz0, "cen": cen, "sig_n": sig_n,
        "xyz1_gt": xyz1_gt, "disp_tok": disp_tok, "tok_mask": tok_mask, "center": center, "radius": radius,
        "K_intr": K_list, "viewmat": vm_list, "H_list": H_list, "W_list": W_list,
        "dq": dq, "anchor": anchor,
    }


def accum_for_log(args):
    return max(1, args.accum)


def make_lr_lambda(warmup_steps, total_steps, peak_lr, floor_lr):
    """Linear warmup (0 -> peak over warmup_steps) then cosine decay (peak -> floor over the remainder).
    Returned multiplier is RELATIVE to peak_lr (so set the optimizer base lr = peak_lr)."""
    import math
    floor_frac = floor_lr / peak_lr

    def fn(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        prog = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        prog = min(1.0, prog)
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return floor_frac + (1.0 - floor_frac) * cos
    return fn


def make_model(args, dev, Kf):
    model = GPSTokenWM(geom_mode=args.geom_mode, fdim=args.fdim, feat_source=args.feat_source,
                       dino_imgsize=args.dino_imgsize, traj_pred=bool(args.traj_pred), Kf=Kf).to(dev)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    model.img_loss = bool(args.img_loss)
    model.w_depth = args.w_depth
    model.cam_cond = bool(args.cam_cond)
    # FROM SCRATCH by default (--init_from empty): DiT trunk + 3D-flow head + action expert all random-init,
    # only the Cosmos-Reason2-2B encoder is frozen. (--init_from kept available but unused per the spec.)
    if args.init_from:
        sd = torch.load(args.init_from, map_location=dev, weights_only=False)["model"]
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"[vla] warm-start {args.init_from}: loaded {len(sd)} tensors, {len(miss)} new", flush=True)
    else:
        print("[vla] FROM SCRATCH: no warm-start (trunk + heads + expert random-init; encoder frozen)", flush=True)
    # attach the action expert (with proprioception state cross-attn)
    model.attach_action_expert(action_dim=args.action_dim, action_steps=args.action_steps,
                               d_act=args.d_act, n_heads_act=args.n_heads_act, mlp_ratio=args.mlp_ratio,
                               norm_stats_path=(args.norm_stats or None), n_state_tokens=args.n_state_tokens)
    model.w_flow_vla, model.w_act_vla = args.w_flow, args.w_act
    model = model.to(dev)                       # move the newly-attached action expert + act_norm onto dev
    return model


def run_smoke():
    """CPU smoke-test: tiny dims, fake clip, no GPU. Validates shapes/forward/loss/backward/sampler/norm."""
    import torch.nn as nn
    from igsw.gpstoken_wm.action_expert import ActionExpert, ActionNormalizer
    torch.manual_seed(0)
    dev = "cpu"
    A, AD = 50, 14
    d_act, n_l, d_trunk, M, Q = 64, 3, 96, 8, 5
    print("=" * 70)
    print("CPU SMOKE-TEST: action expert (tiny dims)")
    print(f"  d_act={d_act} n_layers={n_l} d_trunk={d_trunk} M={M} tokens Q={Q} A={A} steps action_dim={AD}")
    print("=" * 70)

    # ---- 1. normalizer round-trip ----
    mean = torch.randn(AD) * 0.01
    std = torch.rand(AD) * 0.02 + 0.005
    norm = ActionNormalizer(mean, std, gripper_dims=(6, 13), dim=AD)
    dq = torch.randn(A, AD) * 0.05
    dq[:, 6] = (torch.rand(A) > 0.5).float()      # gripper near-binary {0,1}
    dq[:, 13] = (torch.rand(A) > 0.5).float()
    z = norm.normalize(dq)
    dq_rt = norm.denormalize(z)
    rt_err = (dq - dq_rt).abs().max().item()
    grip_ok = (z[:, [6, 13]].abs() <= 1.0 + 1e-5).all().item()
    print(f"[1] normalize/denormalize round-trip max-err={rt_err:.2e}  gripper in [-1,1]: {grip_ok}")
    assert rt_err < 1e-5, "round-trip failed"
    assert grip_ok, "gripper not mapped to [-1,1]"

    # ---- 2. expert forward: per-layer KV consumed ----
    expert = ActionExpert(n_layers=n_l, action_dim=AD, action_steps=A, d=d_act, n_heads=4,
                          trunk_dim=d_trunk, vlm_dim=d_trunk, mlp_ratio=4.0)
    # NOTE: at init the model is gradient-blocked from the trunk on TWO fronts (both intentional, both
    # warm-start-safe): (1) the zero-init OUTPUT HEAD makes ∂v_pred/∂upstream = 0; (2) the AdaLN-Zero
    # cross-attn GATES are tanh(0)=0 so trunk/vlm KV contribute nothing. The model learns to open both.
    # To VERIFY the wiring actually routes the per-layer KV, un-zero the output head AND open the
    # cross-attn gate biases (ct_g, cv_g groups) for this consumption check, like a trained block.
    with torch.no_grad():
        nn.init.normal_(expert.out_head.weight, std=0.02)
        for blk in expert.blocks:
            b_ada = blk.ada[-1].bias                       # 15 groups: sa(0-2) ct(3-5) cv(6-8) cs(9-11) mlp(12-14)
            b_ada[5 * d_act:6 * d_act].fill_(1.0)          # ct_g (trunk cross-attn gate)
            b_ada[8 * d_act:9 * d_act].fill_(1.0)          # cv_g (vlm cross-attn gate)
            b_ada[11 * d_act:12 * d_act].fill_(1.0)        # cs_g (state/proprioception cross-attn gate)
    trunk_kv = [torch.randn(1, M, d_trunk, requires_grad=True) for _ in range(n_l)]
    vlm_kv = [torch.randn(1, Q, d_trunk, requires_grad=True) for _ in range(n_l)]
    state_in = torch.randn(1, AD)                          # current qpos (anchor) -> state condition
    state_kv = expert.embed_state(state_in)
    state_kv.retain_grad()
    xt = torch.randn(1, A, AD)
    t = torch.rand(1)
    v_pred = expert(xt, t, trunk_kv, vlm_kv, state_kv)
    print(f"[2] forward: v_pred shape={tuple(v_pred.shape)} (expect (1,{A},{AD}))")
    assert v_pred.shape == (1, A, AD), "velocity shape wrong"

    # confirm per-layer KV + the proprioception state are actually consumed (grad flows with gates open)
    v_pred.sum().backward()
    kv_grad_layers = sum(1 for g in trunk_kv if g.grad is not None and g.grad.abs().sum() > 0)
    vlm_grad_layers = sum(1 for g in vlm_kv if g.grad is not None and g.grad.abs().sum() > 0)
    state_consumed = state_kv.grad is not None and state_kv.grad.abs().sum() > 0
    print(f"[2b] per-layer KV consumed (gates opened): trunk {kv_grad_layers}/{n_l}, vlm {vlm_grad_layers}/{n_l}; "
          f"proprioception state consumed: {state_consumed}")
    assert kv_grad_layers == n_l and vlm_grad_layers == n_l, "not all layer KV consumed"
    assert state_consumed, "proprioception state cross-attn not consumed"

    # ---- 3. flow-matching loss + backward (gates still open -> full param propagation) ----
    expert.zero_grad()
    x1 = norm.normalize(dq)[None]
    state_kv = expert.embed_state(state_in)
    l_flow, flogs = expert.flow_loss(x1, trunk_kv, vlm_kv, state_kv)
    l_flow.backward()
    n_grad = sum(1 for p in expert.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_tot = sum(1 for p in expert.parameters() if p.requires_grad)
    print(f"[3] flow_loss={l_flow.item():.4f} v_norm={flogs['v_norm'].item():.3f}  "
          f"expert params with grad: {n_grad}/{n_tot}")
    assert torch.isfinite(l_flow), "flow loss not finite"

    # ---- 4. ODE sampler ----
    with torch.no_grad():
        z_samp = expert.sample(trunk_kv, vlm_kv, expert.embed_state(state_in),
                               n_steps=10, device=dev, dtype=torch.float32)
    dq_samp = norm.denormalize(z_samp[0])
    print(f"[4] ODE sample (10 Euler steps): normalized {tuple(z_samp.shape)} -> denorm {tuple(dq_samp.shape)} "
          f"(expect ({A},{AD}))")
    assert dq_samp.shape == (A, AD), "sample shape wrong"

    # ---- 5. trunk per-layer exposure + joint forward (mini GPSTokenWM, no Qwen) ----
    print("[5] trunk per-layer exposure (predict return_layers=True), mini-trunk no-Qwen...")
    from igsw.dynamics.transformer import DiTBlock
    from igsw.dynamics.pe import FourierPE3D

    class MiniWM(nn.Module):
        """Stand-in for GPSTokenWM's trunk path to validate per-layer exposure + joint loss on CPU
        WITHOUT loading the 2B Qwen encoder. Same predict() layer-exposure contract."""
        def __init__(self, d, n_l, fdim):
            super().__init__()
            self.d, self.fdim = d, fdim
            self.pe = FourierPE3D(num_freqs=4)
            self.feat_in = nn.Linear(fdim, fdim)
            self.tok_embed = nn.Sequential(nn.Linear(self.pe.out_dim + fdim + 2, d), nn.SiLU(), nn.Linear(d, d))
            self.blocks = nn.ModuleList([DiTBlock(d, 4, ctx_dim=d) for _ in range(n_l)])
            self.final_norm = nn.LayerNorm(d, eps=1e-6)
            self.geom_head = nn.Linear(d, 3)
        predict = GPSTokenWM.predict
        attach_action_expert = GPSTokenWM.attach_action_expert

    mwm = MiniWM(d_trunk, n_l, fdim=16)
    mwm.attach_action_expert(action_dim=AD, action_steps=A, d_act=d_act, n_heads_act=4, mlp_ratio=4.0)
    mwm.act_norm = norm
    # open the expert cross-attn gates + un-zero the output head so the flow loss genuinely routes
    # gradient THROUGH the expert INTO the trunk KV (proving shared-trunk joint training, not just the
    # direct geom-head path). Both are zero at the real init (warm-start-safe) and learned open.
    with torch.no_grad():
        nn.init.normal_(mwm.action_expert.out_head.weight, std=0.02)
        for blk in mwm.action_expert.blocks:
            blk.ada[-1].bias[5 * d_act:6 * d_act].fill_(1.0)
            blk.ada[-1].bias[8 * d_act:9 * d_act].fill_(1.0)
    tok_xyz0 = torch.randn(M, 3)
    tok_feat = mwm.feat_in(torch.randn(M, 16)).float()
    sig = torch.rand(M, 2)
    center = tok_xyz0.mean(0, keepdim=True); radius = torch.tensor(1.0)
    cond = torch.randn(1, d_trunk)
    ctx = torch.randn(1, n_l, Q, d_trunk)
    ctxm = torch.ones(1, Q, dtype=torch.bool)
    x, layers = mwm.predict(tok_xyz0, tok_feat, sig, center, radius, ctx, ctxm, cond, return_layers=True)
    print(f"    predict -> final x={tuple(x.shape)}, per-layer features: {len(layers)} x {tuple(layers[0].shape)} "
          f"(expect {n_l} x (1,{M},{d_trunk}))")
    assert len(layers) == n_l and layers[0].shape == (1, M, d_trunk), "per-layer exposure wrong"
    vlm_ctx = [ctx[:, j] for j in range(n_l)]
    state_kv_m = mwm.action_expert.embed_state(state_in)
    # (5c) the action FLOW loss ALONE must route gradient into the trunk (the shared-trunk claim)
    mwm.zero_grad()
    l_flow_only, _ = mwm.action_expert.flow_loss(mwm.act_norm.normalize(dq)[None], layers, vlm_ctx,
                                                 state_kv_m, vlm_mask=ctxm)
    l_flow_only.backward(retain_graph=True)
    trunk_from_flow = any(p.grad is not None and p.grad.abs().sum() > 0 for p in mwm.blocks.parameters())
    print(f"[5c] action flow loss ALONE -> trunk grad: {trunk_from_flow} (shared-trunk path verified)")
    assert trunk_from_flow, "flow loss does not backprop the trunk"
    # joint loss: flow + a fake geom (both must backprop the trunk)
    mwm.zero_grad()
    xyz1_pred = tok_xyz0 + mwm.geom_head(x[0])
    l_geom = torch.nn.functional.smooth_l1_loss(xyz1_pred, tok_xyz0 + 0.01)
    l_flow2, _ = mwm.action_expert.flow_loss(mwm.act_norm.normalize(dq)[None], layers, vlm_ctx,
                                             state_kv_m, vlm_mask=ctxm)
    loss = 1.0 * l_geom + 1.0 * l_flow2
    loss.backward()
    trunk_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in mwm.blocks.parameters())
    expert_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in mwm.action_expert.parameters())
    print(f"[5b] JOINT loss={loss.item():.4f} (geom={l_geom.item():.4f} flow={l_flow2.item():.4f}) "
          f"-> trunk grad: {trunk_has_grad}, expert grad: {expert_has_grad}")
    assert trunk_has_grad and expert_has_grad, "joint loss did not backprop both"

    # ---- 6. param counts ----
    # verify the analytic estimator against a REAL instantiation at a mid config (trunk_dim=vlm_dim=d,
    # the production setting where both ctx_dim == d_act... but production uses trunk_dim=1536 != d_act=768,
    # so also instantiate at the production cross-attn ctx_dim to get the EXACT full-config count).
    mid = ActionExpert(n_layers=4, action_dim=AD, action_steps=A, d=256, n_heads=4,
                       trunk_dim=256, vlm_dim=256, mlp_ratio=4.0)
    mid_est = ActionExpert.estimate_params(n_layers=4, action_dim=AD, action_steps=A, d=256, mlp_ratio=4.0)
    print("=" * 70)
    print(f"[6] PARAM COUNTS")
    print(f"    estimator check @ (n_l=4,d=256,ctx=256): actual={mid.num_params()/1e6:.3f}M "
          f"estimate={mid_est/1e6:.3f}M (match when ctx_dim==d)")
    # EXACT full-config count (production default: d_act=704, n_heads=11, trunk=vlm=1536 cross-attn KV).
    # Build 1- and 2-layer instances at full width to read the EXACT per-layer cost, then extrapolate to 28.
    D_FULL, NH_FULL = 704, 11
    full2 = ActionExpert(n_layers=2, action_dim=AD, action_steps=A, d=D_FULL, n_heads=NH_FULL,
                         trunk_dim=1536, vlm_dim=1536, mlp_ratio=4.0)
    full1 = ActionExpert(n_layers=1, action_dim=AD, action_steps=A, d=D_FULL, n_heads=NH_FULL,
                         trunk_dim=1536, vlm_dim=1536, mlp_ratio=4.0)
    per_layer = full2.num_params() - full1.num_params()
    non_layer = full1.num_params() - per_layer          # in_proj + step_emb + t_embed + out head/norm
    full28 = non_layer + 28 * per_layer
    print(f"    tiny expert (this test): {expert.num_params()/1e6:.3f}M")
    print(f"    FULL config EXACT (n_l=28, d_act={D_FULL}, n_heads={NH_FULL}, trunk/vlm ctx=1536, mlp=4): "
          f"{full28/1e6:.0f}M (per-layer {per_layer/1e6:.1f}M; +state cross-attn vs the 511M pre-proprio expert)")
    assert 400e6 <= full28 <= 650e6, f"full expert {full28/1e6:.0f}M outside 400-650M target"
    print("=" * 70)
    print("ALL SMOKE-TEST CHECKS PASSED")


def _load_clips_for_test(args, dev, enc, n):
    """Load + prepare n valid single clips (for verify/probe)."""
    clips = sorted(glob.glob(f"{args.data}/*_train.pt"))
    singles, ci = [], 0
    while len(singles) < n and ci < len(clips):
        try:
            c = torch.load(clips[ci], map_location=dev, weights_only=False)
            singles.append(build_clip_single(c, dev, args, enc))
        except Exception:
            pass
        ci += 1
    return singles


@torch.no_grad()
def run_verify_batch(args):
    """Assert the BATCHED forward == the per-sample (B=1) path within fp tolerance, and that padded tokens
    do NOT leak through the trunk self-attn / cross-attn / geom loss. Deterministic comparison: we feed the
    SAME noised action xt + the SAME t in both paths (flow_loss samples noise internally, so we compare the
    raw velocity field instead), and the SAME initial noise x0 to the ODE sampler."""
    import torch.nn.functional as _F
    dev = "cuda"
    torch.manual_seed(0)
    model = make_model(args, dev, Kf=12).eval()
    enc = model.encoder
    N = 4
    singles = _load_clips_for_test(args, dev, enc, N)
    assert len(singles) == N, f"need {N} clips, got {len(singles)}"
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    print(f"[verify] {N} clips, token counts M = {[s['M'] for s in singles]} (padded to L={args.L})", flush=True)

    # fixed action-noise per clip so the expert forward is deterministic across the two paths
    A, AD = args.action_steps, args.action_dim
    xt = [torch.randn(1, A, AD, device=dev) for _ in range(N)]
    tt = [torch.rand(1, device=dev) for _ in range(N)]
    x0_ode = [torch.randn(1, A, AD, device=dev) for _ in range(N)]

    # ---------- per-sample path ----------
    persamp_v, persamp_xyz, persamp_act = [], [], []
    for i, s in enumerate(singles):
        with amp:
            x, tok_feat, layers, vlm_ctx, ctxm = model._trunk_features(s)
            xyz1, _ = model.heads(x, s["tok_xyz0"], s["K_intr"], s["viewmat"])      # [M,3]
            state_kv = model.action_expert.embed_state(s["anchor"].float()[None])
            v = model.action_expert.forward(xt[i], tt[i].clone(), layers, vlm_ctx, state_kv, vlm_mask=ctxm)  # [1,A,14]
            act = model.act_norm.denormalize(model.action_expert.sample(
                layers, vlm_ctx, state_kv, vlm_mask=ctxm, n_steps=10, device=dev, dtype=torch.float32, x0=x0_ode[i]))
        persamp_v.append(v[0].float()); persamp_xyz.append(xyz1.float()); persamp_act.append(act[0].float())

    # ---------- batched path ----------
    b = build_batch_padded(singles, dev, args, enc)
    with amp:
        x, tok_feat, layers, vlm_ctx, ctxm = model._trunk_features_batch(b)
        g = model.geom_head(x).float()
        xyz1_b = b["tok_xyz0"].float() + g                                          # geom_mode xyz -> [B,L,3]
        state_kv = model.action_expert.embed_state(b["anchor"].float())
        xt_b = torch.cat(xt, 0); t_b = torch.cat(tt, 0)
        v_b = model.action_expert.forward(xt_b, t_b, layers, vlm_ctx, state_kv,
                                          trunk_mask=b["tok_mask"], vlm_mask=ctxm)   # [B,A,14]
        x0_b = torch.cat(x0_ode, 0)
        act_b = model.act_norm.denormalize(model.action_expert.sample(
            layers, vlm_ctx, state_kv, trunk_mask=b["tok_mask"], vlm_mask=ctxm,
            n_steps=10, device=dev, dtype=torch.float32, x0=x0_b))

    # ---------- compare ----------
    print("[verify] max-abs-diff batched vs per-sample (bf16 trunk -> tolerance ~1e-2):", flush=True)
    ok = True
    for i in range(N):
        M = singles[i]["M"]
        dv = (v_b[i] - persamp_v[i]).abs().max().item()
        dxyz = (xyz1_b[i, :M] - persamp_xyz[i][:M]).abs().max().item()
        dact = (act_b[i] - persamp_act[i]).abs().max().item()
        vsc = persamp_v[i].abs().mean().item()
        print(f"  clip{i} (M={M}): d(velocity)={dv:.2e} (|v|~{vsc:.2f})  d(xyz1)={dxyz:.2e}  d(action)={dact:.2e}", flush=True)
        ok = ok and dv < 5e-2 and dact < 5e-2

    # ---------- mask-leak test: perturb PADDED token slots, outputs must NOT change ----------
    b2 = build_batch_padded(singles, dev, args, enc)
    pad = ~b2["tok_mask"]
    for key in ("tok_xyz0", "cen", "sig_n"):
        noise = torch.randn_like(b2[key]) * 5.0
        b2[key] = torch.where(pad.unsqueeze(-1).expand_as(b2[key]), b2[key] + noise, b2[key])
    with amp:
        x2, _, layers2, vlm2, ctxm2 = model._trunk_features_batch(b2)
        state2 = model.action_expert.embed_state(b2["anchor"].float())
        v2 = model.action_expert.forward(xt_b, t_b, layers2, vlm2, state2,
                                         trunk_mask=b2["tok_mask"], vlm_mask=ctxm2)
    leak = (v2 - v_b).abs().max().item()
    print(f"[verify] mask-leak: perturbed PADDED slots by N(0,5); change in real-token velocity = {leak:.2e} "
          f"(must be ~0 -> padded tokens do not leak)", flush=True)
    ok = ok and leak < 5e-2

    # ---------- LR schedule sanity ----------
    fn = make_lr_lambda(args.warmup_steps, args.steps, args.lr_peak, args.lr_floor)
    pts = [0, args.warmup_steps // 2, args.warmup_steps, args.steps // 2, args.steps - 1]
    print("[verify] LR schedule (warmup->peak->cosine->floor):", flush=True)
    for s in pts:
        print(f"    step {s:6d}: lr = {fn(s) * args.lr_peak:.3e}", flush=True)
    assert abs(fn(args.warmup_steps) * args.lr_peak - args.lr_peak) < 1e-9, "peak not hit at end of warmup"
    assert fn(0) * args.lr_peak < args.lr_peak, "warmup should start below peak"

    print(f"\n[verify] RESULT: {'PASS' if ok else 'FAIL'} "
          f"(batched matches per-sample within tol; masks correct)", flush=True)
    if not ok:
        raise SystemExit(1)


def run_batch_probe(args):
    """Ramp B_per_gpu over --probe_batches until OOM on ONE GPU. For each B: build a real padded batch,
    run --probe_iters fwd+bwd+opt.step (full model, bf16, AdamW), report peak mem + clips/s. Reports the
    max B that fits + projected 40k-step ETA at effective-batch 256 on 4 GPUs."""
    dev = "cuda"
    torch.manual_seed(0)
    model = make_model(args, dev, Kf=12)
    model._ddp_touch = False
    enc = model.encoder
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr_peak, weight_decay=0.0)
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    Bs = [int(x) for x in args.probe_batches.split(",")]
    maxB = max(Bs)
    pool = _load_clips_for_test(args, dev, enc, maxB)
    if len(pool) < maxB:
        pool = (pool * (maxB // max(1, len(pool)) + 1))[:maxB]   # recycle clips to fill the largest B
    tot_gpu = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[probe] device={torch.cuda.get_device_name(0)} total={tot_gpu:.0f}GB  L={args.L} "
          f"expert={model.action_expert.num_params()/1e6:.0f}M trainable={model.num_trainable()/1e9:.2f}B", flush=True)
    results = []
    for B in Bs:
        singles = pool[:B]
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            batch = build_batch_padded(singles, dev, args, enc)
            # warmup (build CUDA graphs / allocator) then timed iters
            for it in range(2 + args.probe_iters):
                if it == 2:
                    torch.cuda.synchronize(); t0 = time.time()
                opt.zero_grad(set_to_none=True)
                with amp:
                    loss, logs = model.forward_vla_batch(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / args.probe_iters
            peak = torch.cuda.max_memory_allocated() / 1e9
            reserved = torch.cuda.max_memory_reserved() / 1e9
            cps = B / dt
            results.append((B, peak, reserved, dt, cps))
            print(f"[probe] B={B:3d}  OK  peak_alloc={peak:5.1f}GB  reserved={reserved:5.1f}GB  "
                  f"{dt*1000:6.0f}ms/it  {cps:6.1f} clips/s", flush=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[probe] B={B:3d}  OOM  -> stop", flush=True)
                torch.cuda.empty_cache()
                break
            raise
    if not results:
        print("[probe] no batch size fit (even the smallest OOM'd)", flush=True)
        return
    maxfit, peak, reserved, dt, cps = results[-1]
    world = 4
    print("\n" + "=" * 70, flush=True)
    print(f"[probe] MAX B_per_gpu that fits = {maxfit}  (peak_alloc {peak:.1f}GB / reserved {reserved:.1f}GB "
          f"on {tot_gpu:.0f}GB -> headroom {tot_gpu - reserved:.1f}GB)", flush=True)
    print(f"[probe] throughput at B={maxfit}: {cps:.1f} clips/s/GPU  ({dt*1000:.0f} ms/iter)", flush=True)
    # projection at effective batch 256 on 4 GPUs
    eff = 256
    for B in sorted({maxfit, max(8, maxfit // 2)}, reverse=True):
        accum = max(1, round(eff / (B * world)))
        eff_real = B * world * accum
        # one optimizer step = accum micro-batches; throughput per GPU ~ cps clips/s -> time per micro = B/cps
        sec_per_micro = B / cps
        sec_per_optstep = sec_per_micro * accum
        eta_h = sec_per_optstep * args.steps / 3600.0
        print(f"[probe] @B={B}/gpu accum={accum} -> eff_batch={eff_real} (target {eff}); "
              f"{sec_per_optstep:.2f}s/optstep -> 40k steps ~ {eta_h:.1f}h on {world} GPUs", flush=True)
    print("=" * 70, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU smoke-test (tiny dims, fake clip, no GPU)")
    ap.add_argument("--batch_probe", action="store_true",
                    help="GPU: ramp B_per_gpu until OOM, report max-fit + mem + throughput, then exit (no training)")
    ap.add_argument("--probe_batches", default="4,8,16,24,32,48,64",
                    help="comma list of B_per_gpu to try in --batch_probe")
    ap.add_argument("--probe_iters", type=int, default=6, help="fwd+bwd iters per B in the probe (for throughput)")
    ap.add_argument("--verify_batch", action="store_true",
                    help="GPU: assert the BATCHED forward matches the per-sample (B=1) path within tolerance, then exit")
    ap.add_argument("--data", default="data/rt2_joint")
    ap.add_argument("--out", default="checkpoints/vla_rt2")
    ap.add_argument("--init_from", default="", help="empty = FROM SCRATCH (the spec); a path warm-starts the trunk")
    ap.add_argument("--norm_stats", default="data/rt2_act/norm_stats.pt")
    ap.add_argument("--geom_mode", default="xyz", choices=["xyz", "flowd"],
                    help="from-scratch run uses xyz + img_loss 1 (the spec)")
    ap.add_argument("--feat_source", default="qwen", choices=["qwen", "dino"])
    ap.add_argument("--dino_imgsize", type=int, default=518)
    ap.add_argument("--img_loss", type=int, default=1)
    ap.add_argument("--w_depth", type=float, default=0.5)
    ap.add_argument("--traj_pred", type=int, default=0)
    ap.add_argument("--cam_cond", type=int, default=0)
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--fdim", type=int, default=128)
    ap.add_argument("--beta", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=40000, help="OPTIMIZER steps (the real run = 40000)")
    ap.add_argument("--batch", type=int, default=8, help="B_per_gpu (clips per GPU per micro-step) — REAL batching")
    ap.add_argument("--accum", type=int, default=1, help="grad-accum micro-steps (effective = batch*world*accum)")
    # LR schedule: linear warmup -> peak -> cosine decay -> floor (the spec)
    ap.add_argument("--lr", type=float, default=5e-5, help="(legacy) base lr; superseded by --lr_peak for the schedule")
    ap.add_argument("--lr_peak", type=float, default=5e-5, help="peak LR after warmup")
    ap.add_argument("--lr_floor", type=float, default=1e-5, help="cosine decay floor")
    ap.add_argument("--warmup_steps", type=int, default=1500, help="linear warmup steps (~3-5%% of 40k)")
    # action expert
    ap.add_argument("--action_dim", type=int, default=14)
    ap.add_argument("--action_steps", type=int, default=50)
    ap.add_argument("--d_act", type=int, default=704, help="action-expert width (704=64*11 -> ~590M at n_l=28 incl. state cross-attn)")
    ap.add_argument("--n_heads_act", type=int, default=11, help="11 heads * 64 head_dim = 704")
    ap.add_argument("--n_state_tokens", type=int, default=1, help="# proprioception (current-qpos) condition tokens")
    ap.add_argument("--mlp_ratio", type=float, default=4.0)
    ap.add_argument("--w_flow", type=float, default=1.0, help="weight on the 3D-flow img_loss")
    ap.add_argument("--w_act", type=float, default=1.0, help="weight on the action flow-matching loss")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--lr_min_frac", type=float, default=1.0)
    ap.add_argument("--eval_ckpt", default="", help="if set: load ckpt, eval --eval_split (action metrics via predict_action), exit")
    ap.add_argument("--eval_split", default="heldseed")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return
    if args.batch_probe:
        run_batch_probe(args)
        return
    if args.verify_batch:
        run_verify_batch(args)
        return

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local); dev = f"cuda:{local}"
    else:
        rank, world, local, dev = 0, 1, 0, "cuda"
    is_main = rank == 0
    if is_main:
        os.makedirs(args.out, exist_ok=True)

    _probe = sorted(glob.glob(f"{args.data}/*_train.pt"))
    _Kf = int(torch.load(_probe[0], map_location="cpu", weights_only=False)["Kf"]) if _probe else 12
    # n_l for the expert comes from the trunk's block count inside attach_action_expert (= Qwen layers)
    model = make_model(args, dev, Kf=_Kf)
    enc = model.encoder
    model._ddp_touch = ddp
    mdl = DDP(model, device_ids=[local], find_unused_parameters=False, broadcast_buffers=False) if ddp else model
    # AdamW with base lr = peak; LambdaLR applies linear warmup -> cosine decay -> floor (the spec).
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr_peak, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, make_lr_lambda(args.warmup_steps, args.steps, args.lr_peak, args.lr_floor))
    if is_main:
        exp_p = model.action_expert.num_params()
        eff = args.batch * world * accum_for_log(args)
        print(f"[vla] trainable={model.num_trainable()/1e9:.3f}B  action_expert={exp_p/1e6:.0f}M  "
              f"geom_mode={args.geom_mode} L={args.L} world={world} B/gpu={args.batch} accum={args.accum} "
              f"eff_batch={eff} lr_peak={args.lr_peak} lr_floor={args.lr_floor} warmup={args.warmup_steps}", flush=True)

    clips = sorted(glob.glob(f"{args.data}/*_train.pt"))
    shard = clips[rank::world]
    if is_main:
        print(f"[vla] {len(clips)} train clips ({len(shard)}/rank)", flush=True)
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    step, t0, ci = 0, time.time(), 0
    accum = max(1, args.accum)
    if args.eval_ckpt:                                                    # HELD EVAL: action quality, then exit
        import numpy as _np, torch.nn.functional as _F
        sd = torch.load(args.eval_ckpt, map_location=dev, weights_only=False)["model"]
        miss, unexp = model.load_state_dict(sd, strict=False)             # ckpt omits frozen encoder.* by design
        miss = [m for m in miss if not m.startswith("encoder.")]
        if is_main:
            print(f"[vla-eval] loaded {len(sd)} tensors; non-encoder missing={len(miss)} unexpected={len(unexp)}", flush=True)
        model.eval()
        evf = sorted(glob.glob(f"{args.data}/*{args.eval_split}*.pt"))
        arm = [i for i in range(14) if i not in (6, 13)]
        ers, dcs, gacc = [], [], []
        for cp in evf:
            try:
                c = torch.load(cp, map_location=dev, weights_only=False); b = build_batch(c, dev, args, enc)
            except Exception:
                continue
            with torch.no_grad(), amp:
                pred = model.predict_action(b).float()                    # [A,14] raw Δqpos
            gt = b["dq"].float()
            ers.append(float((pred[:, arm] - gt[:, arm]).norm(dim=-1).mean()))
            dcs.append(float(_F.cosine_similarity(pred[:, arm].reshape(-1), gt[:, arm].reshape(-1), dim=0)))
            gacc.append(float(((pred[:, [6, 13]] > 0.5) == (gt[:, [6, 13]] > 0.5)).float().mean()))
        if is_main:
            print(f"[vla-eval] {os.path.basename(args.eval_ckpt)} {args.eval_split}: n={len(ers)} "
                  f"arm_dΔqpos_err={_np.median(ers):.4f} arm_dcos={_np.median(dcs):+.3f} "
                  f"grip_acc={_np.mean(gacc)*100:.0f}%", flush=True)
        return
    def next_padded_batch():
        """Pull args.batch valid clips from this rank's shard -> one padded batch (skips bad clips)."""
        nonlocal ci
        singles, tries = [], 0
        while len(singles) < args.batch and tries < args.batch * 8:
            cp = shard[ci % len(shard)]; ci += 1; tries += 1
            try:
                c = torch.load(cp, map_location=dev, weights_only=False)
                singles.append(build_clip_single(c, dev, args, enc))
            except Exception as e:
                if is_main and step < 3:
                    print(f"[skip] prep {os.path.basename(cp)}: {type(e).__name__}: {e}", flush=True)
        if len(singles) < args.batch:
            return None
        return build_batch_padded(singles, dev, args, enc)

    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        agg, n_ok = None, 0
        for _m in range(accum):                                  # accum MICRO-batches (each = args.batch clips)
            batch = next_padded_batch()
            ok = batch is not None
            if ddp:
                flag = torch.tensor([1.0 if ok else 0.0], device=dev)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                ok = flag.item() > 0.5
            if not ok:
                continue
            with amp:
                loss, logs = (mdl.module.forward_vla_batch(batch) if ddp else mdl.forward_vla_batch(batch))
            (loss / accum).backward()
            n_ok += 1
            agg = ({k: v.detach() for k, v in logs.items()} if agg is None
                   else {k: agg[k] + logs[k].detach() for k in agg})
        if n_ok == 0:
            step += 1
            continue
        gnorm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        finite = torch.tensor([1.0 if torch.isfinite(gnorm) else 0.0], device=dev)
        if ddp:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() > 0.5:
            opt.step()
        elif is_main:
            print(f"[skip] non-finite grad @s{step}", flush=True)
        if sched is not None:
            sched.step()
        logs = {k: v / n_ok for k, v in agg.items()}
        if is_main and step % args.log_every == 0:
            print(f"s{step} loss{logs['loss'].item():.3f} geom{logs['geom'].item():.4f} "
                  f"flow{logs['flow'].item():.4f} vnorm{logs['v_norm'].item():.2f} "
                  f"errp{logs['errp'].item()*100:.1f}cm dcos{logs['dcos'].item():.2f} "
                  f"{(step+1)/(time.time()-t0):.2f}it/s", flush=True)
        step += 1
        if is_main and (step % args.save_every == 0 or step == args.steps):
            sd = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
            torch.save({"model": sd, "args": vars(args), "step": step}, f"{args.out}/vla_{step:06d}.pt")
            print(f"[vla] saved vla_{step:06d}.pt ({len(sd)} tensors)", flush=True)
    if is_main:
        print("[vla] DONE", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
