"""VLA trainer for the joint geometry head and pi0-style action expert sharing the DiT trunk.

The causal-v1 run uses a strict observation contract:
  * means/uv/K/center/radius come from the current RGB frame through single-frame VGGT,
  * the fixed 48x48 candidate grid is never filtered by future visibility,
  * full-video SpaTracker output supplies only xyz1_gt/geom_valid supervision,
  * clips without a tracker target still train the action expert.

The trainer also:
  * adds the flow-matching ACTION EXPERT (predicts Δqpos[50,14]) on the SAME trunk (both losses backprop it),
  * reads ONLY frame0 (gt_rgb[0]) — no future frame, no JEPA target, no vlmK,
  * action normalization from data/rt2_act/norm_stats.pt (arm dims standardized; gripper dims -> [-1,1]).

  torchrun --nproc_per_node=4 code/scripts/train_vla.py \
      --data data/rt2_causal_v1 --prep_cache data/rt2_causal_v1_prepcache_wrist \
      --out checkpoints/vla_causal_v1 --init_from "" \
      --norm_stats data/rt2_act/norm_stats.pt \
      --geom_mode xyz --img_loss 1 --L 512 --wrist 1 --placement entropy \
      --causal_geometry_version vggt_t1_grid48_v1 --deepspeed 1 --steps 50000

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
from igsw.distributed import assert_same_paths, init_torchrun, shard_for_rank  # noqa: E402
from igsw.gpstoken_wm import GPSTokenWM, place_tokens  # noqa: E402
from igsw.gaussians.gpstoken import mover_saliency, relevance_saliency  # noqa: E402


def _to_dev(o, dev):
    """Recursively move tensors in a (possibly nested dict/list) prep-single to a device."""
    if torch.is_tensor(o):
        return o.to(dev)
    if isinstance(o, dict):
        return {k: _to_dev(v, dev) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_to_dev(v, dev) for v in o)
    return o


def mv_in(inputs, dev):
    return {k: (v.to(dev, dtype=torch.bfloat16) if (torch.is_tensor(v) and v.is_floating_point())
                else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in inputs.items()}


import contextlib


def _nullctx():
    return contextlib.nullcontext()


def save_ckpt(mdl, model, args, step, use_fsdp, is_main):
    """Save a regular (non-FSDP) state_dict so --eval_ckpt (which loads into a fresh make_model with
    strict=False) round-trips. Under FSDP we gather the FULL_STATE_DICT to rank0 (offloaded to CPU)
    via the FSDP state_dict_type context — this is a COLLECTIVE, so ALL ranks must enter it. Then rank0
    filters out the frozen encoder.* and writes. The saved keys match a plain GPSTokenWM (encoder + the
    attached action_expert/act_norm), exactly like the DDP path."""
    if use_fsdp:
        from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, StateDictType,
                                            FullStateDictConfig)
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(mdl, StateDictType.FULL_STATE_DICT, cfg):
            full_sd = mdl.state_dict()          # collective: every rank participates; only rank0 gets tensors
        if is_main:
            sd = {k: v for k, v in full_sd.items() if not k.startswith("encoder.")}
            torch.save({"model": sd, "args": vars(args), "step": step}, f"{args.out}/vla_{step:06d}.pt")
            print(f"[vla] saved vla_{step:06d}.pt ({len(sd)} tensors, FSDP full-state-dict)", flush=True)
    elif is_main:
        sd = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
        torch.save({"model": sd, "args": vars(args), "step": step}, f"{args.out}/vla_{step:06d}.pt")
        print(f"[vla] saved vla_{step:06d}.pt ({len(sd)} tensors)", flush=True)


def placement_saliency(args, enc, vlm0, uv, disp, n_keep, H, W):
    """Token-placement saliency (where the sparse GPSTokens go). 'relevance' (default, DEPLOYABLE) uses the
    frozen-Qwen instruction<->image-patch relevance grid (inference-available, NO GT). 'oracle' (ablation
    only) uses the GT future-motion mover_saliency (the old train+eval leak). beta<=0 -> None (pure entropy).
    NB: we NEVER silently fall back to the GT oracle at inference — if relevance is unavailable, sal=None."""
    plc = getattr(args, "placement", "entropy")
    if args.beta <= 0 or plc == "entropy":
        return None                                       # pure image-complexity (entropy) partition — DEPLOYABLE
    if plc == "oracle":
        return mover_saliency(uv, disp, n_keep, H, W)     # GT future-motion (ablation only — train/eval LEAK)
    rel, ghw = enc.relevance_grid(vlm0)                    # 'relevance': frozen-Qwen grounding (weak on RT2)
    return relevance_saliency(rel, ghw, H, W) if rel is not None else None


def _imgs_with_wrist(args, c, rgb0):
    """Head frame0 alone, OR [head, left, right] when --wrist and the clip carries wrist frames. Head is
    ALWAYS first so the per-token 3D grid (image #0) stays head-only; wrist views only enrich VLM context."""
    if not getattr(args, "wrist", 0) or "left_rgb" not in c or "right_rgb" not in c:
        return rgb0
    def _np(x):
        return (x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)).astype(np.uint8)
    return [rgb0, _np(c["left_rgb"]), _np(c["right_rgb"])]


def build_batch(c, dev, args, enc):
    """rt2_joint clip dict -> prepared VLA batch (frame0-only; adds dq). Mirrors the world-model prep."""
    means = c["means"].to(dev).float(); uv = c["uv"].to(dev).float()
    traj = c["traj"].to(dev).float(); N = means.shape[0]
    H, W = int(c["H"]), int(c["W"]); K = int(c["Kf"])
    instr = c.get("instruction", ""); n_keep = N - int(c.get("n_fill", 0))
    geom_valid = c.get("geom_valid", torch.ones(N, dtype=torch.bool)).to(dev).bool()
    disp = (traj[K] - traj[0]).norm(dim=-1)                            # GT future: TARGET ONLY (xyz1_gt/disp_tok)
    rgb0 = c["gt_rgb"][0].cpu().numpy().astype(np.uint8)               # head frame0 (drives 3D/placement)
    vlm0 = mv_in(enc.build_inputs(instr, _imgs_with_wrist(args, c, rgb0)), dev)   # +wrist views -> context only
    placement_target = disp if getattr(args, "placement", "entropy") == "oracle" else None
    sal = placement_saliency(args, enc, vlm0, uv, placement_target, n_keep, H, W)
    cen, sig, idx = place_tokens(rgb0, uv, n_keep, args.L, dev, sal=sal, beta=args.beta)
    if idx.shape[0] < 16:
        raise ValueError("too few tokens")
    center = means[:n_keep].mean(0, keepdim=True)
    return {
        "vlm0": vlm0,
        "cen": cen, "sig_n": (sig / float(max(H, W))).clamp(0, 1),
        "tok_xyz0": means[idx], "xyz1_gt": traj[K][idx], "disp_tok": disp[idx],
        "geom_valid": geom_valid[idx],
        "traj_gt": (traj[1:K + 1][:, idx] if args.traj_pred else None),
        "center": center, "radius": (means[:n_keep] - center).norm(dim=-1).amax().clamp_min(1e-6),
        "K_intr": c["K_intr"].to(dev).float(), "viewmat": c["viewmat"].to(dev).float(),
        "H": H, "W": W, "rgb0_np": rgb0,
        "dq": c["dq"].to(dev).float(),                                 # [A,14] action target
        "anchor": c["anchor"].to(dev).float(),                        # [14] current qpos (proprioception);
        #          REQUIRED by predict_action -> embed_state. Was missing -> eval KeyErrored on the 1st clip.
    }


def build_clip_single(c, dev, args, enc):
    """Prepare ONE clip into the per-sample fields needed for the BATCHED path (no [None]-batching, no
    padding yet). Returns a dict of M-length tensors + scalars. build_batch_padded stacks/pads these."""
    means = c["means"].to(dev).float(); uv = c["uv"].to(dev).float()
    traj = c["traj"].to(dev).float(); N = means.shape[0]
    H, W = int(c["H"]), int(c["W"]); K = int(c["Kf"])
    instr = c.get("instruction", ""); n_keep = N - int(c.get("n_fill", 0))
    causal_version = getattr(args, "causal_geometry_version", "")
    if causal_version and c.get("causal_geometry_version") != causal_version:
        raise ValueError(f"causal geometry mismatch: clip={c.get('causal_geometry_version')!r} "
                         f"run={causal_version!r}")
    geom_valid = c.get("geom_valid", torch.ones(N, dtype=torch.bool)).to(dev).bool()
    disp = (traj[K] - traj[0]).norm(dim=-1)                            # GT future: TARGET ONLY (xyz1_gt/disp_tok)
    rgb0 = c["gt_rgb"][0].cpu().numpy().astype(np.uint8)
    vlm0 = mv_in(enc.build_inputs(instr, _imgs_with_wrist(args, c, rgb0)), dev)   # +wrist views -> context only
    placement_target = disp if getattr(args, "placement", "entropy") == "oracle" else None
    sal = placement_saliency(args, enc, vlm0, uv, placement_target, n_keep, H, W)
    cen, sig, idx = place_tokens(rgb0, uv, n_keep, args.L, dev, sal=sal, beta=args.beta)
    M = idx.shape[0]
    if M < 16:
        raise ValueError("too few tokens")
    center = means[:n_keep].mean(0, keepdim=True)
    radius = (means[:n_keep] - center).norm(dim=-1).amax().clamp_min(1e-6)
    return {
        "vlm0": vlm0,
        "cen": cen, "sig_n": (sig / float(max(H, W))).clamp(0, 1),
        "tok_xyz0": means[idx], "xyz1_gt": traj[K][idx], "disp_tok": disp[idx],
        "geom_valid": geom_valid[idx],
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
    geom_valid = torch.zeros(B, L, dtype=torch.bool, device=dev)
    center = torch.zeros(B, 1, 3, device=dev)
    radius = torch.ones(B, 1, device=dev)
    dq = torch.zeros(B, args.action_steps, args.action_dim, device=dev)
    anchor = torch.zeros(B, args.action_dim, device=dev)
    vlm_list, K_list, vm_list, H_list, W_list = [], [], [], [], []
    for i, s in enumerate(clips):
        M = min(s["M"], L)
        tok_xyz0[i, :M] = s["tok_xyz0"][:M]; cen[i, :M] = s["cen"][:M]
        sig_n[i, :M] = s["sig_n"][:M]; xyz1_gt[i, :M] = s["xyz1_gt"][:M]
        disp_tok[i, :M] = s["disp_tok"][:M]; tok_mask[i, :M] = True
        geom_valid[i, :M] = s.get("geom_valid", torch.ones(M, dtype=torch.bool, device=dev))[:M]
        center[i] = s["center"]; radius[i, 0] = s["radius"]
        dq[i] = s["dq"]; anchor[i] = s["anchor"]
        vlm_list.append(s["vlm0"])
        # NOTE: the per-clip visual grid is NO LONGER computed here. It now comes FREE from
        # encode_cond_batch's forward (it used to be a redundant 2nd Qwen forward per clip).
        K_list.append(s["K_intr"]); vm_list.append(s["viewmat"])
        H_list.append(s["H"]); W_list.append(s["W"])
    return {
        "vlm_list": vlm_list, "tok_xyz0": tok_xyz0, "cen": cen, "sig_n": sig_n,
        "xyz1_gt": xyz1_gt, "disp_tok": disp_tok, "tok_mask": tok_mask, "geom_valid": geom_valid,
        "center": center, "radius": radius,
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


def adamw_param_groups(params, weight_decay):
    """Decoupled AdamW groups: weight decay on weight MATRICES (ndim>=2) only; norms/biases (ndim<2) get 0
    (the standard recipe — decaying LayerNorm/bias hurts)."""
    params = [p for p in params if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_fsdp(model, local_rank, reduce_dtype="fp32"):
    """Wrap GPSTokenWM in FSDP with ShardingStrategy.SHARD_GRAD_OP (ZeRO-2: shard grads + optimizer
    states across ranks, params REPLICATED). Returns the FSDP-wrapped model.

    Key choices (spec §FSDP):
      * sharding_strategy = SHARD_GRAD_OP  -> the AdamW states (~25GB for 2.28B params) + grads are
        sharded /world; the fp32 master params are sharded but PARAMS ARE NOT RESHARDED between fwd and
        bwd (the ZeRO-2 win: one all-gather per fwd, kept full through bwd; only grads/opt-states shrink).
      * use_orig_params=True -> the original Parameters remain on the module (flat-param is a view),
        so the existing param filtering (requires_grad), AdamW build, and save code keep working.
      * ignored_modules=[model.encoder] -> the FROZEN Cosmos-Reason2-2B encoder is excluded from FSDP
        (no grads -> nothing to shard); it stays replicated + frozen + on-device.
      * transformer_auto_wrap_policy on {DiTBlock, ActionExpertBlock} -> each trunk block & each expert
        block is its own FSDP unit (proper per-block grad/opt sharding).
      * MixedPrecision(param=bf16, reduce=fp32, buffer=fp32): bf16 compute (matches the autocast the
        trainer already uses), fp32 gradient reduce-scatter for numerical safety. buffer_dtype=fp32 so
        the ActionNormalizer stats (mean/std/grip-bool) are NOT corrupted by a bf16 buffer cast.
    """
    import functools
    from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, ShardingStrategy,
                                        MixedPrecision)
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from igsw.dynamics.transformer import DiTBlock
    from igsw.gpstoken_wm.action_expert import ActionExpertBlock

    # route the top-level call through forward() so the ROOT FSDP unit's hooks fire (restore 2-D orig
    # params + reduce-scatter). Calling forward_vla_batch on the inner module would skip the root hooks.
    model._fsdp_vla = True
    wrap_policy = functools.partial(transformer_auto_wrap_policy,
                                    transformer_layer_cls={DiTBlock, ActionExpertBlock})
    rd = torch.float32 if reduce_dtype == "fp32" else torch.bfloat16
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=rd, buffer_dtype=torch.float32)
    fsdp = FSDP(
        model,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp,
        ignored_modules=[model.encoder],
        device_id=local_rank,
        use_orig_params=True,
        sync_module_states=True,        # broadcast rank0 init -> identical params on all ranks
        limit_all_gathers=True,
    )
    return fsdp


def make_model(args, dev, Kf):
    model = GPSTokenWM(geom_mode=args.geom_mode, fdim=args.fdim, feat_source=args.feat_source,
                       dino_imgsize=args.dino_imgsize, traj_pred=bool(args.traj_pred), Kf=Kf).to(dev)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    model.img_loss = bool(args.img_loss)
    model.w_depth = args.w_depth
    model.cam_cond = bool(args.cam_cond)
    # Attach before loading so a VLA checkpoint restores the expert and action-normalizer buffers too.
    model.attach_action_expert(action_dim=args.action_dim, action_steps=args.action_steps,
                               d_act=args.d_act, n_heads_act=args.n_heads_act, mlp_ratio=args.mlp_ratio,
                               norm_stats_path=(args.norm_stats or None), n_state_tokens=args.n_state_tokens)
    model.w_flow_vla, model.w_act_vla = args.w_flow, args.w_act
    model = model.to(dev)

    # FROM SCRATCH by default. A VLA warm-start must restore every non-encoder tensor; the frozen encoder is
    # intentionally omitted from VLA checkpoints and comes from the configured base model.
    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu", weights_only=False)["model"]
        miss, unexp = model.load_state_dict(sd, strict=False)
        missing_non_encoder = [key for key in miss if not key.startswith("encoder.")]
        is_vla_checkpoint = any(key.startswith("action_expert.") for key in sd)
        if is_vla_checkpoint and (missing_non_encoder or unexp):
            raise RuntimeError(
                "incompatible VLA checkpoint: "
                f"missing_non_encoder={missing_non_encoder[:20]} unexpected={unexp[:20]}"
            )
        print(f"[vla] warm-start {args.init_from}: loaded {len(sd)} tensors, "
              f"missing_non_encoder={len(missing_non_encoder)} unexpected={len(unexp)}", flush=True)
    else:
        print("[vla] FROM SCRATCH: no warm-start (trunk + heads + expert random-init; encoder frozen)", flush=True)
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
    dq[:, 6] = torch.randint(-1, 2, (A,)).float()   # gripper PER-STEP Δ in {-1,0,+1} (NOT absolute {0,1})
    dq[:, 13] = torch.randint(-1, 2, (A,)).float()
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
    # The from-scratch model is zero-init on BOTH heads (geom_head, expert.out_head) and AdaLN-Zero gates,
    # so every output is identically 0 -> a 0==0 comparison would be vacuous. Randomize the heads and OPEN
    # the expert cross-attn gates (sa/ct/cv/cs) so outputs become NON-ZERO and INPUT-DEPENDENT -> the
    # batched-vs-per-sample diff genuinely exercises the padding/masking math (like the CPU smoke test does).
    import torch.nn as _nn
    d_act = model.action_expert.d
    with torch.no_grad():
        _nn.init.normal_(model.geom_head.weight, std=0.02); _nn.init.normal_(model.geom_head.bias, std=0.02)
        _nn.init.normal_(model.action_expert.out_head.weight, std=0.02)
        _nn.init.normal_(model.action_expert.out_head.bias, std=0.02)
        for blk in model.action_expert.blocks:
            b_ada = blk.ada[-1].bias                    # 15 groups: sa(0-2) ct(3-5) cv(6-8) cs(9-11) mlp(12-14)
            b_ada[2 * d_act:3 * d_act].fill_(1.0)       # sa_g
            b_ada[5 * d_act:6 * d_act].fill_(1.0)       # ct_g (trunk cross-attn)
            b_ada[8 * d_act:9 * d_act].fill_(1.0)       # cv_g (vlm cross-attn)
            b_ada[11 * d_act:12 * d_act].fill_(1.0)     # cs_g (state/proprioception cross-attn)
    N = 4
    singles = _load_clips_for_test(args, dev, enc, N)
    assert len(singles) == N, f"need {N} clips, got {len(singles)}"
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    print(f"[verify] {N} clips, token counts M = {[s['M'] for s in singles]} (padded to L={args.L}); "
          f"heads randomized + cross-attn gates OPENED so outputs are non-trivial", flush=True)

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


def run_fsdp_probe(args):
    """DISTRIBUTED probe under FSDP SHARD_GRAD_OP (run with torchrun --nproc_per_node=4 ... --fsdp_probe).
    Wraps the model in FSDP, then for each B in --probe_batches runs a few REAL optimizer steps (fwd +
    bwd + clip_grad_norm_ + opt.step, AdamW, no_sync accum window of 1) and reports the per-rank peak mem
    + throughput. Reports the max B that fits with headroom + the 50k ETA at effective-batch 256.

    Why distributed (not the single-GPU run_batch_probe): the WHOLE point of FSDP is that grads + Adam
    states are sharded /world, so the memory only drops with real ranks present. Measuring on 1 GPU would
    show the un-sharded footprint."""
    import torch.distributed as dist
    assert "RANK" in os.environ, "run under torchrun"
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local); dev = f"cuda:{local}"
    is_main = rank == 0
    torch.manual_seed(0)
    _probe = sorted(glob.glob(f"{args.data}/*_train.pt"))
    _Kf = int(torch.load(_probe[0], map_location="cpu", weights_only=False)["Kf"]) if _probe else 12
    model = make_model(args, dev, Kf=_Kf)
    model._ddp_touch = False
    enc = model.encoder
    _g_trainable = model.num_trainable(); _g_expert = model.action_expert.num_params()  # BEFORE wrap (global)
    mdl = build_fsdp(model, local, reduce_dtype=args.fsdp_reduce)
    opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad], lr=args.lr_peak, weight_decay=0.0)
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    Bs = [int(x) for x in args.probe_batches.split(",")]
    maxB = max(Bs)
    pool = _load_clips_for_test(args, dev, enc, maxB)
    if len(pool) < maxB:
        pool = (pool * (maxB // max(1, len(pool)) + 1))[:maxB]
    tot_gpu = torch.cuda.get_device_properties(local).total_memory / 1e9
    if is_main:
        print(f"[fsdp-probe] world={world} device={torch.cuda.get_device_name(local)} total={tot_gpu:.0f}GB "
              f"L={args.L} expert={_g_expert/1e6:.0f}M trainable={_g_trainable/1e9:.2f}B "
              f"(sharded {_g_trainable/world/1e9:.2f}B/rank) reduce={args.fsdp_reduce}", flush=True)
    results = []
    for B in Bs:
        singles = pool[:B]
        oom = torch.tensor([0.0], device=dev)
        peak = reserved = dt = cps = 0.0
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            batch = build_batch_padded(singles, dev, args, enc)
            for it in range(2 + args.probe_iters):           # 2 warmup (allocator/graphs) then timed
                if it == 2:
                    torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
                opt.zero_grad(set_to_none=True)
                with amp:
                    loss, logs = mdl(batch)            # FSDP.forward -> root hooks -> forward_vla_batch
                loss.backward()
                mdl.clip_grad_norm_(1.0)
                opt.step()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / args.probe_iters
            peak = torch.cuda.max_memory_allocated() / 1e9
            reserved = torch.cuda.max_memory_reserved() / 1e9
            cps = B / dt
            lval = float(loss.detach().float())
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom = torch.tensor([1.0], device=dev); torch.cuda.empty_cache(); lval = float("nan")
            else:
                raise
        # any rank OOM => treat B as not-fitting on all ranks
        dist.all_reduce(oom, op=dist.ReduceOp.MAX)
        if oom.item() > 0.5:
            if is_main:
                print(f"[fsdp-probe] B={B:3d}  OOM (>=1 rank)  -> stop", flush=True)
            break
        # gather worst-case peak/reserved across ranks
        stat = torch.tensor([peak, reserved, cps, float(torch.isfinite(torch.tensor(lval)))], device=dev)
        statmax = stat.clone(); dist.all_reduce(statmax, op=dist.ReduceOp.MAX)
        statmin = stat.clone(); dist.all_reduce(statmin, op=dist.ReduceOp.MIN)
        peak_w, res_w = statmax[0].item(), statmax[1].item()
        cps_min, finite_ok = statmin[2].item(), statmin[3].item() > 0.5
        results.append((B, peak_w, res_w, dt, cps))
        if is_main:
            print(f"[fsdp-probe] B={B:3d}  OK  peak_alloc(max-rank)={peak_w:5.1f}GB  reserved={res_w:5.1f}GB  "
                  f"{dt*1000:6.0f}ms/it  {cps:6.1f} clips/s/gpu  loss_finite={finite_ok}", flush=True)
    if is_main:
        if not results:
            print("[fsdp-probe] no batch size fit", flush=True)
        else:
            maxfit, peak, reserved, dt, cps = results[-1]
            print("\n" + "=" * 74, flush=True)
            print(f"[fsdp-probe] MAX B_per_gpu under FSDP = {maxfit}  (peak_alloc {peak:.1f}GB / reserved "
                  f"{reserved:.1f}GB on {tot_gpu:.0f}GB -> headroom {tot_gpu - reserved:.1f}GB)", flush=True)
            print(f"[fsdp-probe] throughput at B={maxfit}: {cps:.1f} clips/s/GPU ({dt*1000:.0f} ms/iter); "
                  f"aggregate {cps*world:.1f} clips/s", flush=True)
            eff = 256
            for B, *_r in [(maxfit,)] + ([(max(8, maxfit // 2),)] if maxfit > 8 else []):
                # find this B's measured cps (or use maxfit's)
                cB = next((r[4] for r in results if r[0] == B), cps)
                accum = max(1, round(eff / (B * world)))
                eff_real = B * world * accum
                sec_per_optstep = (B / cB) * accum
                eta_h = sec_per_optstep * args.steps / 3600.0
                print(f"[fsdp-probe] @B={B}/gpu accum={accum} -> eff_batch={eff_real} (target {eff}); "
                      f"{sec_per_optstep:.2f}s/optstep -> {args.steps//1000}k steps ~ {eta_h:.1f}h on {world} GPUs",
                      flush=True)
            print("=" * 74, flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU smoke-test (tiny dims, fake clip, no GPU)")
    ap.add_argument("--batch_probe", action="store_true",
                    help="GPU: ramp B_per_gpu until OOM, report max-fit + mem + throughput, then exit (no training)")
    ap.add_argument("--fsdp_probe", action="store_true",
                    help="DISTRIBUTED (torchrun): ramp B_per_gpu under FSDP SHARD_GRAD_OP, report max-fit + sharded mem + throughput, then exit")
    ap.add_argument("--probe_batches", default="4,8,16,24,32,48,64",
                    help="comma list of B_per_gpu to try in --batch_probe / --fsdp_probe")
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
    ap.add_argument("--placement", default="entropy", choices=["entropy", "relevance", "oracle"],
                    help="token placement. 'entropy' (default, DEPLOYABLE) = pure image-complexity partition "
                         "(no GT, no grounding; the textured object gets tokens via its edges). 'relevance' = "
                         "frozen-Qwen instruction grounding — but on RoboTwin2 frames it peaks on BACKGROUND not "
                         "the object (verified, _check_relevance.py), so weak. 'oracle' = GT future-motion "
                         "mover_saliency (the old train+eval LEAK; ablation only). prep_cache bakes this in.")
    ap.add_argument("--wrist", type=int, default=0, help="1 = feed left+right wrist frame0 as EXTRA Qwen images "
                    "(head stays image #0 -> 3D/GPSToken unchanged; wrist only enriches VLM context). Requires "
                    "clips patched with left_rgb/right_rgb (rt2_add_wrist.py). Changes the encode -> rebuild prep_cache.")
    ap.add_argument("--causal_geometry_version", default="", help="non-empty enforces a matching causal input "
                    "contract in every clip and records it in the checkpoint")
    ap.add_argument("--steps", type=int, default=40000, help="OPTIMIZER steps (the real run = 40000)")
    ap.add_argument("--batch", type=int, default=8, help="B_per_gpu (clips per GPU per micro-step) — REAL batching")
    ap.add_argument("--accum", type=int, default=1, help="grad-accum micro-steps (effective = batch*world*accum)")
    ap.add_argument("--prefetch", type=int, default=1, help="1 = background thread prepares batches (overlaps load+OpenCV with compute)")
    ap.add_argument("--prefetch_depth", type=int, default=2, help="prefetch queue depth (batches buffered ahead)")
    ap.add_argument("--prefetch_workers", type=int, default=4, help="# parallel producer threads")
    ap.add_argument("--prep_cache", default="", help="dir of PRECOMPUTED build_clip_single singles. When set, the producer just torch.loads tensors (I/O releases the GIL -> overlaps compute) instead of running the frozen processor+place_tokens every step. empty = compute on the fly")
    ap.add_argument("--cache_prep", action="store_true", help="PRECOMPUTE the prep cache (sharded over ranks) into --prep_cache, then exit")
    # FSDP (ZeRO-2 equivalent): shard grads + optimizer states across ranks, params replicated.
    ap.add_argument("--fsdp", type=int, default=0, help="1 = wrap in FSDP ShardingStrategy.SHARD_GRAD_OP (shard grads+opt states; default launch uses this)")
    ap.add_argument("--fsdp_reduce", default="fp32", choices=["fp32", "bf16"], help="FSDP gradient reduce dtype (fp32 = numerically safe; bf16 = lighter comms)")
    # DeepSpeed ZeRO-2: the engine manages grad-accum + the collectives internally (no hand-rolled no_sync),
    # which sidesteps the FSDP step-1 deadlock. Overrides --fsdp when set.
    ap.add_argument("--deepspeed", type=int, default=0, help="1 = DeepSpeed ZeRO-2 engine (stage2, bf16, client AdamW+LambdaLR); overrides --fsdp")
    # LR schedule: linear warmup -> peak -> cosine decay -> floor (the spec)
    ap.add_argument("--lr", type=float, default=5e-5, help="(legacy) base lr; superseded by --lr_peak for the schedule")
    ap.add_argument("--lr_peak", type=float, default=5e-5, help="peak LR after warmup")
    ap.add_argument("--lr_floor", type=float, default=1e-5, help="cosine decay floor")
    ap.add_argument("--warmup_steps", type=int, default=1500, help="linear warmup steps (~3-5%% of 40k)")
    ap.add_argument("--seed", type=int, default=42, help="shared model-init seed; training RNG is offset by global rank")
    ap.add_argument("--weight_decay", type=float, default=0.01, help="decoupled AdamW weight decay (on weight matrices only, not norms/biases). Mild default — the held-task failure is data-diversity/underfitting, NOT overfitting, so do NOT crank this")
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
    ap.add_argument("--eval_max", type=int, default=600, help="cap # eval clips (2 passes: real + instruction-shuffle). 0 = all")
    args = ap.parse_args()

    if args.causal_geometry_version and args.placement == "oracle":
        raise ValueError("causal training forbids future-motion oracle token placement")

    if args.smoke:
        run_smoke()
        return
    if args.batch_probe:
        run_batch_probe(args)
        return
    if args.fsdp_probe:
        run_fsdp_probe(args)
        return
    if args.verify_batch:
        run_verify_batch(args)
        return

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    context = init_torchrun()
    ddp = context.distributed
    rank, world, local, dev = (
        context.rank,
        context.world_size,
        context.local_rank,
        context.device,
    )
    is_main = context.is_main
    if is_main:
        os.makedirs(args.out, exist_ok=True)
    if ddp:
        dist.barrier()

    # Every rank constructs identical parameters. Training randomness is rank-offset after wrapping.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    use_ds = bool(args.deepspeed) and ddp
    use_fsdp = bool(args.fsdp) and ddp and not use_ds       # DeepSpeed overrides FSDP
    _probe = sorted(glob.glob(f"{args.data}/*_train.pt"))
    assert_same_paths(_probe, context, "training clips")
    probe_clip = torch.load(_probe[0], map_location="cpu", weights_only=False) if _probe else None
    _Kf = int(probe_clip["Kf"]) if probe_clip is not None else 12
    if args.causal_geometry_version:
        if probe_clip is None:
            raise ValueError("causal training data is empty")
        if probe_clip.get("causal_geometry_version") != args.causal_geometry_version:
            raise ValueError("training data does not satisfy the requested causal geometry contract")
        if "geom_valid" not in probe_clip or int(probe_clip["means"].shape[0]) != 48 * 48:
            raise ValueError("causal training clip must contain geom_valid and the full 48x48 input grid")
        if is_main:
            print(f"[vla] causal input contract={args.causal_geometry_version} grid=48x48; future labels "
                  "mask geometry loss only", flush=True)
    if args.cache_prep:                                                  # PRECOMPUTE the prep cache, then exit
        assert args.prep_cache, "--cache_prep needs --prep_cache <dir>"
        if args.placement == "entropy":
            from igsw.dynamics.conditioning import QwenInputProcessor
            enc = QwenInputProcessor()
        else:
            enc = make_model(args, dev, Kf=_Kf).encoder
        if is_main:
            os.makedirs(args.prep_cache, exist_ok=True)
        if ddp:
            dist.barrier()
        allclips = sorted(glob.glob(f"{args.data}/*.pt"))                # ALL splits (train + held)
        assert_same_paths(allclips, context, "cache input clips")
        myshard = shard_for_rank(allclips, rank, world, "cache input clips")
        done, skip = 0, 0
        for cp in myshard:
            out = os.path.join(args.prep_cache, os.path.basename(cp))
            if os.path.exists(out):
                done += 1; continue
            try:
                c = torch.load(cp, map_location=dev, weights_only=False)
                s = build_clip_single(c, dev, args, enc)
                s.pop("rgb0_np", None)                                   # unused in the batch path
                torch.save(_to_dev(s, "cpu"), out)
                done += 1
            except Exception as e:
                skip += 1
                if is_main and skip <= 5:
                    print(f"[cache] skip {os.path.basename(cp)}: {type(e).__name__}: {e}", flush=True)
            if is_main and done % 200 == 0:
                print(f"[cache] rank0 {done}/{len(myshard)} (skip {skip})", flush=True)
        if is_main:
            print(f"[cache] rank0 DONE {done}/{len(myshard)} (skip {skip}) -> {args.prep_cache}", flush=True)
        if ddp:
            dist.barrier(); dist.destroy_process_group()
        return
    # n_l for the expert comes from the trunk's block count inside attach_action_expert (= Qwen layers)
    model = make_model(args, dev, Kf=_Kf)
    enc = model.encoder
    # capture TRUE global param counts BEFORE the FSDP wrap (after wrap the orig params are 1-D flat shards
    # /world, so num_trainable()/num_params() would report the per-rank shard, not the global model).
    _global_trainable = model.num_trainable()
    _global_expert = model.action_expert.num_params()
    # FSDP (SHARD_GRAD_OP / ZeRO-2) shards grads + optimizer states; with use_orig_params=True params stay
    # replicated so the existing methods/save code work. The DDP-only zero-weight "touch" (forces every
    # param to participate under find_unused_parameters=False) is NOT needed under FSDP (use_orig_params
    # tolerates unused params) and would only add wasteful work -> keep it OFF for FSDP.
    model._ddp_touch = ddp and not use_fsdp and not use_ds
    if use_ds:
        import deepspeed
        # client AdamW over the TRAINABLE params only (frozen 2B encoder excluded -> not sharded, stays
        # replicated). client LambdaLR = the SAME warmup->peak->cosine->floor schedule; DeepSpeed steps it
        # on each optimizer step. ZeRO-2 shards this optimizer's states + grads /world.
        opt = torch.optim.AdamW(adamw_param_groups(model.parameters(), args.weight_decay), lr=args.lr_peak)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, make_lr_lambda(args.warmup_steps, args.steps, args.lr_peak, args.lr_floor))
        ds_config = {
            "train_micro_batch_size_per_gpu": args.batch,
            "gradient_accumulation_steps": max(1, args.accum),
            "bf16": {"enabled": True},
            "zero_optimization": {"stage": 2, "overlap_comm": True, "contiguous_gradients": True,
                                  "reduce_bucket_size": int(2e8), "allgather_bucket_size": int(2e8)},
            "gradient_clipping": 1.0,
            "steps_per_print": int(1e9),
            "wall_clock_breakdown": False,
        }
        # the engine uses the already-initialized torch.distributed group (from torchrun above).
        mdl, opt, _, sched = deepspeed.initialize(model=model, optimizer=opt, lr_scheduler=sched, config=ds_config)
        if is_main:
            print(f"[vla] DeepSpeed ZeRO-2 ready (stage2 bf16 grad_clip1.0; client AdamW + warmup-cosine LambdaLR); "
                  f"encoder frozen/replicated, engine handles accum+collectives", flush=True)
    elif use_fsdp:
        mdl = build_fsdp(model, local, reduce_dtype=args.fsdp_reduce)
        # optimizer AFTER the FSDP wrap, on the (orig) trainable params now owned by FSDP.
        opt = torch.optim.AdamW(adamw_param_groups(mdl.parameters(), args.weight_decay), lr=args.lr_peak)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, make_lr_lambda(args.warmup_steps, args.steps, args.lr_peak, args.lr_floor))
        if is_main:
            print(f"[vla] FSDP SHARD_GRAD_OP wrap done (reduce={args.fsdp_reduce}); encoder ignored (frozen, replicated)", flush=True)
    else:
        mdl = DDP(model, device_ids=[local], find_unused_parameters=False, broadcast_buffers=False) if ddp else model
        # AdamW with base lr = peak; LambdaLR applies linear warmup -> cosine decay -> floor (the spec).
        opt = torch.optim.AdamW(adamw_param_groups(model.parameters(), args.weight_decay), lr=args.lr_peak)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, make_lr_lambda(args.warmup_steps, args.steps, args.lr_peak, args.lr_floor))
    if is_main:
        eff = args.batch * world * accum_for_log(args)
        shard_note = f" (sharded {_global_trainable/world/1e9:.2f}B/rank)" if (use_fsdp or use_ds) else ""
        print(f"[vla] trainable={_global_trainable/1e9:.3f}B{shard_note}  action_expert={_global_expert/1e6:.0f}M  "
              f"geom_mode={args.geom_mode} L={args.L} world={world} B/gpu={args.batch} accum={args.accum} "
              f"eff_batch={eff} lr_peak={args.lr_peak} lr_floor={args.lr_floor} warmup={args.warmup_steps} "
              f"fsdp={int(use_fsdp)} ds={int(use_ds)}", flush=True)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    clips = _probe
    shard = shard_for_rank(clips, rank, world, "training clips")
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
        if args.eval_max > 0:
            evf = evf[:args.eval_max]
        arm = [i for i in range(14) if i not in (6, 13)]; grip = [6, 13]

        def _eval_pass(instr_override=None):
            """One eval pass over evf. instr_override[ci] (aligned to evf) replaces each clip's instruction
            (the language-shuffle test); None = real. Returns (metrics, instrs_used aligned to evf).
            NB: predict_action is INSIDE the try, so a bad clip is skipped (was: aborted the whole eval)."""
            early, late, dmean, ers, gstate, gtrans, used = [], [], [], [], [], [], []
            for ci, cp in enumerate(evf):
                instr_i = ""
                try:
                    c = torch.load(cp, map_location=dev, weights_only=False)
                    instr_i = instr_override[ci] if instr_override is not None else c.get("instruction", "")
                    c["instruction"] = instr_i
                    b = build_batch(c, dev, args, enc)
                    with torch.no_grad(), amp:
                        pred = model.predict_action(b).float()                 # [A,14] raw Δqpos
                    gt = b["dq"].float(); anch = b["anchor"].float()
                    ps = _F.cosine_similarity(pred[:, arm], gt[:, arm], dim=1)  # [A] PER-STEP arm dir-cos
                    A = ps.shape[0]; q = max(1, A // 5)
                    early.append(float(ps[:q].mean())); late.append(float(ps[-q:].mean())); dmean.append(float(ps.mean()))
                    ers.append(float((pred[:, arm] - gt[:, arm]).norm(dim=-1).mean()))
                    # gripper on the ABSOLUTE reconstructed state (anchor + cumsum of Δ), not the raw per-step Δ
                    a0 = anch[grip].round()                                     # frame0 open/close state
                    gt_st = (a0[None] + torch.cumsum(gt[:, grip], 0)).round().clamp(0, 1)
                    pr_st = (a0[None] + torch.cumsum(pred[:, grip], 0)).round().clamp(0, 1)
                    gstate.append(float((gt_st == pr_st).float().mean()))
                    m = gt[:, grip].abs() > 0.5                                 # the open/close transition steps only
                    if m.any():
                        gtrans.append(float((torch.sign(pred[:, grip]) == torch.sign(gt[:, grip]))[m].float().mean()))
                except Exception:
                    pass
                used.append(instr_i)
            med = lambda v: float(_np.median(v)) if v else float("nan")
            men = lambda v: float(_np.mean(v)) if v else float("nan")
            return ({"early": med(early), "late": med(late), "mean": med(dmean), "err": med(ers),
                     "gstate": men(gstate), "gtrans": men(gtrans), "n": len(dmean)}, used)

        real, instrs = _eval_pass(None)
        n = len(instrs); shuf = (instrs[n // 2:] + instrs[:n // 2]) if n > 1 else instrs   # derangement (roll n/2)
        shf, _ = _eval_pass(shuf)
        if is_main:
            _plc = getattr(args, "placement", "oracle")
            print(f"[vla-eval] {os.path.basename(args.eval_ckpt)} {args.eval_split} n={real['n']} (placement={_plc})", flush=True)
            print(f"  arm dir-cos: early={real['early']:+.3f} late={real['late']:+.3f} mean={real['mean']:+.3f}  "
                  f"drift(late-early)={real['late'] - real['early']:+.3f}", flush=True)
            print(f"  arm Δqpos err (median)={real['err']:.4f}", flush=True)
            print(f"  gripper: abs-state acc={real['gstate'] * 100:.0f}%  transition-sign acc={real['gtrans'] * 100:.0f}%", flush=True)
            print(f"  LANGUAGE shuffle: dir-cos real={real['mean']:+.3f} shuf={shf['mean']:+.3f}  "
                  f"delta={real['mean'] - shf['mean']:+.3f}  (delta~0 => instruction ignored)", flush=True)
        return
    def make_single(cp):
        if args.prep_cache:                                              # I/O-only producer (overlaps compute)
            cpath = os.path.join(args.prep_cache, os.path.basename(cp))
            if os.path.exists(cpath):
                return _to_dev(torch.load(cpath, map_location="cpu", weights_only=False), dev)
        c = torch.load(cp, map_location=dev, weights_only=False)
        return build_clip_single(c, dev, args, enc)

    # ---- data prep: the per-clip torch.load + OpenCV place_tokens (~3.5s/16-clip batch) is LARGER than the
    # GPU forward (~2.2s) and is the throughput bottleneck. A background prefetch thread overlaps the next
    # batch's prep with the current batch's compute. (PyTorch CUDA ops from the worker share the default
    # stream, so they serialize correctly w.r.t. the training step — the win is hiding the CPU/load latency.)
    if args.prefetch:
        import threading, queue
        n_workers = max(1, args.prefetch_workers)
        q: "queue.Queue" = queue.Queue(maxsize=max(args.prefetch_depth, n_workers + 1))
        stop = threading.Event()
        shard_cursor = 0
        shard_cursor_lock = threading.Lock()

        def take_clip_path():
            nonlocal shard_cursor
            with shard_cursor_lock:
                cp = shard[shard_cursor % len(shard)]
                shard_cursor += 1
            return cp

        def producer(tid):
            torch.cuda.set_device(local)                          # threads use this rank's device
            while not stop.is_set():
                singles, tries = [], 0
                # BOUND the fill loop. Without the tries cap, a run of clips that fail make_single makes
                # this spin forever -> q never fills -> the consumer's q.get() blocks on THIS rank -> the
                # other ranks hang at the ok-sync all_reduce (exactly the step-1 deadlock we hit). With the
                # cap we always q.put() within args.batch*8 tries; a short batch -> consumer returns None ->
                # the all_reduce(MIN) makes ALL ranks skip that micro-step together (no deadlock).
                while len(singles) < args.batch and tries < args.batch * 8 and not stop.is_set():
                    cp = take_clip_path()
                    tries += 1
                    try:
                        singles.append(make_single(cp))
                    except Exception:
                        pass
                if stop.is_set():
                    break
                try:
                    q.put(singles, timeout=1.0)
                except queue.Full:
                    continue
        ths = [threading.Thread(target=producer, args=(t,), daemon=True) for t in range(n_workers)]
        for th in ths:
            th.start()

        def next_padded_batch():
            singles = q.get()
            return build_batch_padded(singles, dev, args, enc) if singles and len(singles) >= args.batch else None
    else:
        def next_padded_batch():
            nonlocal ci
            singles, tries = [], 0
            while len(singles) < args.batch and tries < args.batch * 8:
                cp = shard[ci % len(shard)]; ci += 1; tries += 1
                try:
                    singles.append(make_single(cp))
                except Exception as e:
                    if is_main and step < 3:
                        print(f"[skip] prep {os.path.basename(cp)}: {type(e).__name__}: {e}", flush=True)
            if len(singles) < args.batch:
                return None
            return build_batch_padded(singles, dev, args, enc)

    def fwd(batch):
        # DeepSpeed: engine.module is the original model; ZeRO-2 keeps params replicated so calling
        # forward_vla_batch directly is correct (no param-gather hooks needed, unlike ZeRO-3).
        if use_ds:
            return mdl.module.forward_vla_batch(batch)
        # FSDP: call mdl(batch) -> FSDP.forward -> root hooks -> module.forward -> forward_vla_batch
        # (the _fsdp_vla dispatch). DDP: through .module. Single-GPU: direct.
        if use_fsdp:
            return mdl(batch)
        return mdl.module.forward_vla_batch(batch) if ddp else mdl.forward_vla_batch(batch)

    # ===== DeepSpeed ZeRO-2 training loop =====
    # The engine owns grad-accum, the gradient reduce, gradient clipping, optimizer.step, zero_grad and the
    # LR-scheduler step. We just feed micro-batches: backward()+step() each; DeepSpeed updates the optimizer
    # every gradient_accumulation_steps micro-batches (global_steps increments then). The bad-clip ok-sync
    # (all_reduce MIN) keeps the micro counter aligned across ranks. No hand-rolled no_sync -> the FSDP
    # step-1 collective deadlock cannot recur.
    if use_ds:
        agg, n_micro = None, 0
        while mdl.global_steps < args.steps:
            batch = next_padded_batch()
            ok = batch is not None
            flag = torch.tensor([1.0 if ok else 0.0], device=dev)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if flag.item() <= 0.5:
                continue                                  # all ranks skip this micro together
            gs_before = mdl.global_steps
            with amp:
                loss, logs = fwd(batch)
            mdl.backward(loss)                            # DeepSpeed scales by 1/accum + reduces on boundary
            mdl.step()                                    # opt.step + zero_grad + sched.step on the boundary
            agg = ({k: v.detach() for k, v in logs.items()} if agg is None
                   else {k: agg[k] + logs[k].detach() for k in agg})
            n_micro += 1
            if mdl.global_steps > gs_before:              # an optimizer step just happened this micro
                gs = mdl.global_steps
                if is_main and gs % args.log_every == 0:
                    l = {k: v / n_micro for k, v in agg.items()}
                    try:
                        cur_lr = mdl.get_lr()[0]
                    except Exception:
                        cur_lr = opt.param_groups[0]["lr"]
                    print(f"s{gs} loss{l['loss'].item():.3f} geom{l['geom'].item():.4f} "
                          f"flow{l['flow'].item():.4f} vnorm{l['v_norm'].item():.2f} "
                          f"errp{l['errp'].item()*100:.1f}cm dcos{l['dcos'].item():.2f} "
                          f"lr{cur_lr:.1e} {gs/(time.time()-t0):.2f}it/s", flush=True)
                agg, n_micro = None, 0
                if gs % args.save_every == 0 or gs == args.steps:
                    if is_main:
                        sd = {k: v for k, v in mdl.module.state_dict().items() if not k.startswith("encoder.")}
                        torch.save({"model": sd, "args": vars(args), "step": gs}, f"{args.out}/vla_{gs:06d}.pt")
                        print(f"[vla] saved vla_{gs:06d}.pt ({len(sd)} tensors, deepspeed zero2)", flush=True)
                    dist.barrier()
        if args.prefetch:
            stop.set()
        if is_main:
            print("[vla] DONE", flush=True)
        dist.destroy_process_group()
        return

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
            # FSDP grad-accum: every micro reduce-scatters (no no_sync). The old "no_sync all but the
            # positional last micro" was BUGGY: if the last micro got skipped by the ok-sync (bad clip),
            # NO micro ran outside no_sync -> the reduce-scatter never fired -> opt.step() ran on per-rank
            # UN-synced grads (silent divergence). Reducing every micro is correct (FSDP accumulates the
            # sharded grads across backwards) and only costs extra comms on this LEGACY path — the supported
            # fast path is DeepSpeed (--deepspeed 1), whose engine handles accum correctly.
            use_nosync = False
            ctx = mdl.no_sync() if use_nosync else _nullctx()
            with ctx:
                with amp:
                    loss, logs = fwd(batch)
                (loss / accum).backward()
            n_ok += 1
            agg = ({k: v.detach() for k, v in logs.items()} if agg is None
                   else {k: agg[k] + logs[k].detach() for k in agg})
        if n_ok == 0:
            step += 1
            continue
        # FSDP shards grads -> use FSDP.clip_grad_norm_ (handles the sharded-grad all-reduce of the norm).
        if use_fsdp:
            gnorm = mdl.clip_grad_norm_(1.0)
        else:
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
        if step % args.save_every == 0 or step == args.steps:
            save_ckpt(mdl, model, args, step, use_fsdp, is_main)
    if args.prefetch:
        stop.set()
    if is_main:
        print("[vla] DONE", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
