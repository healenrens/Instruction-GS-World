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


def make_model(args, dev, Kf):
    model = GPSTokenWM(geom_mode=args.geom_mode, fdim=args.fdim, feat_source=args.feat_source,
                       dino_imgsize=args.dino_imgsize, traj_pred=bool(args.traj_pred), Kf=Kf).to(dev)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    model.img_loss = bool(args.img_loss)
    model.w_depth = args.w_depth
    model.cam_cond = bool(args.cam_cond)
    # warm-start trunk + geom_head (strict=False so the expert/act_norm stay fresh)
    if args.init_from:
        sd = torch.load(args.init_from, map_location=dev, weights_only=False)["model"]
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"[vla] warm-start {args.init_from}: loaded {len(sd)} tensors, {len(miss)} new", flush=True)
    # attach the action expert AFTER warm-start (so it is not overwritten)
    model.attach_action_expert(action_dim=args.action_dim, action_steps=args.action_steps,
                               d_act=args.d_act, n_heads_act=args.n_heads_act, mlp_ratio=args.mlp_ratio,
                               norm_stats_path=(args.norm_stats or None))
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
            b_ada = blk.ada[-1].bias                       # 12 groups: sa(0-2) ct(3-5) cv(6-8) mlp(9-11)
            b_ada[5 * d_act:6 * d_act].fill_(1.0)          # ct_g (trunk cross-attn gate)
            b_ada[8 * d_act:9 * d_act].fill_(1.0)          # cv_g (vlm cross-attn gate)
    trunk_kv = [torch.randn(1, M, d_trunk, requires_grad=True) for _ in range(n_l)]
    vlm_kv = [torch.randn(1, Q, d_trunk, requires_grad=True) for _ in range(n_l)]
    xt = torch.randn(1, A, AD)
    t = torch.rand(1)
    v_pred = expert(xt, t, trunk_kv, vlm_kv)
    print(f"[2] forward: v_pred shape={tuple(v_pred.shape)} (expect (1,{A},{AD}))")
    assert v_pred.shape == (1, A, AD), "velocity shape wrong"

    # confirm per-layer KV is actually consumed (grad flows to every layer's KV with gates open)
    v_pred.sum().backward()
    kv_grad_layers = sum(1 for g in trunk_kv if g.grad is not None and g.grad.abs().sum() > 0)
    vlm_grad_layers = sum(1 for g in vlm_kv if g.grad is not None and g.grad.abs().sum() > 0)
    print(f"[2b] per-layer KV consumed (gates opened): trunk {kv_grad_layers}/{n_l}, vlm {vlm_grad_layers}/{n_l}")
    assert kv_grad_layers == n_l and vlm_grad_layers == n_l, "not all layer KV consumed"

    # ---- 3. flow-matching loss + backward (gates still open -> full param propagation) ----
    expert.zero_grad()
    x1 = norm.normalize(dq)[None]
    l_flow, flogs = expert.flow_loss(x1, trunk_kv, vlm_kv)
    l_flow.backward()
    n_grad = sum(1 for p in expert.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_tot = sum(1 for p in expert.parameters() if p.requires_grad)
    print(f"[3] flow_loss={l_flow.item():.4f} v_norm={flogs['v_norm'].item():.3f}  "
          f"expert params with grad: {n_grad}/{n_tot}")
    assert torch.isfinite(l_flow), "flow loss not finite"

    # ---- 4. ODE sampler ----
    with torch.no_grad():
        z_samp = expert.sample(trunk_kv, vlm_kv, n_steps=10, device=dev, dtype=torch.float32)
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
    # (5c) the action FLOW loss ALONE must route gradient into the trunk (the shared-trunk claim)
    mwm.zero_grad()
    l_flow_only, _ = mwm.action_expert.flow_loss(mwm.act_norm.normalize(dq)[None], layers, vlm_ctx, vlm_mask=ctxm)
    l_flow_only.backward(retain_graph=True)
    trunk_from_flow = any(p.grad is not None and p.grad.abs().sum() > 0 for p in mwm.blocks.parameters())
    print(f"[5c] action flow loss ALONE -> trunk grad: {trunk_from_flow} (shared-trunk path verified)")
    assert trunk_from_flow, "flow loss does not backprop the trunk"
    # joint loss: flow + a fake geom (both must backprop the trunk)
    mwm.zero_grad()
    xyz1_pred = tok_xyz0 + mwm.geom_head(x[0])
    l_geom = torch.nn.functional.smooth_l1_loss(xyz1_pred, tok_xyz0 + 0.01)
    l_flow2, _ = mwm.action_expert.flow_loss(mwm.act_norm.normalize(dq)[None], layers, vlm_ctx, vlm_mask=ctxm)
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
          f"{full28/1e6:.0f}M (per-layer {per_layer/1e6:.1f}M, target 400-600M)")
    assert 400e6 <= full28 <= 600e6, f"full expert {full28/1e6:.0f}M outside 400-600M target"
    print("=" * 70)
    print("ALL SMOKE-TEST CHECKS PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU smoke-test (tiny dims, fake clip, no GPU)")
    ap.add_argument("--data", default="data/rt2_joint")
    ap.add_argument("--out", default="checkpoints/vla_rt2")
    ap.add_argument("--init_from", default="checkpoints/wm_rt2/wm_002500.pt")
    ap.add_argument("--norm_stats", default="data/rt2_act/norm_stats.pt")
    ap.add_argument("--geom_mode", default="xyz", choices=["xyz", "flowd"],
                    help="MUST match the warm-start ckpt (wm_rt2/wm_002500.pt was trained geom_mode=xyz)")
    ap.add_argument("--feat_source", default="qwen", choices=["qwen", "dino"])
    ap.add_argument("--dino_imgsize", type=int, default=518)
    ap.add_argument("--img_loss", type=int, default=1)
    ap.add_argument("--w_depth", type=float, default=0.5)
    ap.add_argument("--traj_pred", type=int, default=0)
    ap.add_argument("--cam_cond", type=int, default=0)
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--fdim", type=int, default=128)
    ap.add_argument("--beta", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--accum", type=int, default=1)
    # action expert
    ap.add_argument("--action_dim", type=int, default=14)
    ap.add_argument("--action_steps", type=int, default=50)
    ap.add_argument("--d_act", type=int, default=704, help="action-expert width (704=64*11 -> ~511M at n_l=28, mid of 400-600M)")
    ap.add_argument("--n_heads_act", type=int, default=11, help="11 heads * 64 head_dim = 704")
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
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * args.lr_min_frac)
             if args.lr_min_frac < 1.0 else None)
    if is_main:
        exp_p = model.action_expert.num_params()
        print(f"[vla] trainable={model.num_trainable()/1e9:.3f}B  action_expert={exp_p/1e6:.0f}M  "
              f"geom_mode={args.geom_mode} L={args.L} world={world}", flush=True)

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
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        agg, n_ok = None, 0
        for _m in range(accum):
            cp = shard[ci % len(shard)]; ci += 1
            ok = True
            try:
                c = torch.load(cp, map_location=dev, weights_only=False)
                batch = build_batch(c, dev, args, enc)
            except Exception as e:
                ok = False
                if is_main:
                    print(f"[skip] prep {os.path.basename(cp)}: {type(e).__name__}: {e}", flush=True)
            if ddp:
                flag = torch.tensor([1.0 if ok else 0.0], device=dev)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                ok = flag.item() > 0.5
            if not ok:
                continue
            with amp:
                loss, logs = (mdl.module.forward_vla(batch) if ddp else mdl.forward_vla(batch))
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
