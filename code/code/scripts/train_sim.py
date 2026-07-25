"""4-GPU DDP trainer on CLEAN ManiSkill sim clips — the GENERALIZATION phase (agent.md §39).

The overfit (§39) PROVED the 1.76B model localizes motion on ONE clean clip (corr 0.95) with
per-control SPATIAL-grounding + clean analytic GT. This trainer SCALES that: it trains on MANY clean
sim clips (DistributedSampler shards the train split across 4 GPUs) and we check GENERALIZATION on a
held-out split. No simplifications — the same spatial-grounding model call as the overfit, the direct
3D trajectory loss vs the EXACT `traj`, the InfoNCE language loss, the NaN-guard, checkpointing.

Reuses train_stream.py's scaffolding (DDP static_graph, AdamW, the finite-grad-norm guard, InfoNCE +
MoCo queue, ckpt/tb) but the data path is the sim clips (no Pi3 lift / no CoTracker track at train
time — the clip already IS the clean supervised target). Resumes dynamics from stream11c (strict=False;
the spatial layers + InfoNCE heads reinit). Control sampling is MOVER-BIASED (as in overfit_motion_sim)
so the M controls actually contain movers -> the localization metrics are meaningful.

  torchrun --nproc_per_node=4 code/scripts/train_sim.py --data data/maniskill --out checkpoints/sim_gen \
      --resume checkpoints/stream11c_infonce/ckpt_0006000.pt --spatial_ground 1 --vlm_image 1
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.data.sim_clips import SimClipDataset, sim_collate  # noqa: E402
from igsw.gaussians import GaussianSet, render_gaussianset, psnr  # noqa: E402
from igsw.dynamics.model import DynamicsConfig  # noqa: E402
from igsw.model_full import InstructGSWorldModel  # noqa: E402
from igsw.training import photometric_loss, delta_reg, velocity_smoothness  # noqa: E402
from igsw.training.losses import (trajectory_loss, rotation_loss, contrastive_infonce,  # noqa: E402
                                  scale_anchor_loss, mover_bce_loss, semantic_id_loss,
                                  mover_magnitude_loss)


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank(); world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(local)
        return True, rank, world, local
    return False, 0, 1, 0


def move_vlm_inputs(inputs, dev, dtype):
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            out[k] = v.to(dev, dtype=dtype) if v.is_floating_point() else v.to(dev)
        else:
            out[k] = v
    return out


def sample_controls(g0_means, disp_all, M, gen, mover_thresh=0.01):
    """Mover-biased control sampling (as in overfit_motion_sim): up to half the M controls are
    drawn from Gaussians that actually move (disp>thresh), the rest static, then shuffled. Ensures
    the control set exercises localization (a uniform sample of 2048 from ~200k -> ~10 movers)."""
    N = g0_means.shape[0]
    M = min(M, N)
    mover_g = (disp_all > mover_thresh).nonzero(as_tuple=True)[0]
    static_g = (disp_all <= mover_thresh).nonzero(as_tuple=True)[0]
    n_mover = int(min(mover_g.numel(), M // 2))
    n_static = M - n_mover
    n_static = int(min(n_static, static_g.numel()))
    n_mover = M - n_static
    n_mover = int(min(n_mover, mover_g.numel()))
    sel_m = mover_g[torch.randperm(mover_g.numel(), device=g0_means.device, generator=gen)[:n_mover]]
    sel_s = static_g[torch.randperm(static_g.numel(), device=g0_means.device, generator=gen)[:n_static]]
    idx = torch.cat([sel_m, sel_s])
    idx = idx[torch.randperm(idx.numel(), device=g0_means.device, generator=gen)]
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/pfs/public/xuhaoming/instruct_gs_world/data/maniskill")
    ap.add_argument("--out", default="/mnt/pfs/public/xuhaoming/instruct_gs_world/checkpoints/sim_gen")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--n_query", type=int, default=16)
    ap.add_argument("--cond_mode", default="aggregator", choices=["aggregator", "metaquery"])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_sg", type=float, default=1e-3, help="LR for freshly-init spatial-grounding params (as in the overfit)")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--total_steps", type=int, default=60000)
    ap.add_argument("--w_traj_pos", type=float, default=1.0)
    ap.add_argument("--w_traj_vel", type=float, default=1.0)
    ap.add_argument("--w_traj_rot", type=float, default=0.2)
    ap.add_argument("--rot_knn", type=int, default=8)
    ap.add_argument("--w_lang_contrast", type=float, default=0.5)
    ap.add_argument("--lang_tau", type=float, default=0.07)
    ap.add_argument("--queue_size", type=int, default=256)
    ap.add_argument("--vlm_image", type=int, default=1)   # spatial-grounding NEEDS the image grid
    ap.add_argument("--w_render", type=float, default=0.1)
    ap.add_argument("--w_reg", type=float, default=1e-3)
    ap.add_argument("--w_scale_anchor", type=float, default=0.2)  # anti-drift; modest (don't crush motion)
    ap.add_argument("--w_vel", type=float, default=1e-3)
    ap.add_argument("--obj_focus", type=float, default=0.0,
                    help="weight traj loss toward GT-movers (0 = plain L1 = the overfit's clean test)")
    ap.add_argument("--spatial_ground", type=int, default=1)
    # ---- Exp-1: per-control mover/static GATE (default OFF -> A/B). Needs --spatial_ground 1.
    ap.add_argument("--dyn_gate", type=int, default=0,
                    help="Exp-1: per-control p_dyn gate (sigmoid(p_dyn)*v); BCE-supervised by the free sim mover label")
    ap.add_argument("--w_dyn", type=float, default=1.0, help="weight of the mover-BCE loss on p_dyn")
    ap.add_argument("--sem_dim", type=int, default=0,
                    help="Exp-1 #3 (optional): per-control object-semantic embedding dim (0=off)")
    ap.add_argument("--w_seg", type=float, default=0.2, help="weight of the object-semantic (seg_per_g) loss")
    ap.add_argument("--w_mag", type=float, default=0.0,
                    help="weight of the relative mover-MAGNITUDE loss (fights L1 heavy-tailed under-prediction; raises top-mover ratio)")
    ap.add_argument("--gate_uses_sem", type=int, default=1,
                    help="§44h: feed the occlusion-robust 3D identity e_sem INTO the dyn-gate (concat with "
                         "the 2D Qwen patch). Only active when --dyn_gate 1 --sem_dim>0; 0 = gate sees only "
                         "the 2D patch (A/B). Default 1.")
    ap.add_argument("--feature_dim", type=int, default=0)
    ap.add_argument("--render_steps", type=int, default=2)
    ap.add_argument("--checkpoint_every", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--mover_thresh", type=float, default=0.01)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--resume", default="")
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--detect_anomaly", type=int, default=0)
    args = ap.parse_args()
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    ddp, rank, world, local = setup_ddp()
    dev = torch.device(f"cuda:{local}")
    is_main = rank == 0
    if is_main:
        os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(1234 + rank)

    ds = SimClipDataset(args.data, splits=("train",))
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True) if ddp else None
    loader = DataLoader(ds, batch_size=1, sampler=sampler, shuffle=(sampler is None),
                        num_workers=args.workers, collate_fn=sim_collate,
                        persistent_workers=(args.workers > 0), prefetch_factor=(args.prefetch if args.workers > 0 else None),
                        pin_memory=False, drop_last=True)
    if is_main:
        print(f"[sim] {len(ds)} TRAIN clips | world={world} | {len(ds)//max(1,world)} clips/rank/epoch", flush=True)

    cfg = DynamicsConfig(d_model=args.dim, n_layers=args.layers, n_heads=args.heads,
                         lang_dim=2048, use_checkpoint=True, checkpoint_every=args.checkpoint_every,
                         feature_dim=args.feature_dim)
    model = InstructGSWorldModel(cfg, n_control=args.M, n_query=args.n_query,
                                 cond_mode=args.cond_mode, spatial_ground=bool(args.spatial_ground),
                                 dyn_gate=bool(args.dyn_gate), sem_dim=args.sem_dim,
                                 gate_uses_sem=bool(args.gate_uses_sem)).to(dev)
    if is_main and args.spatial_ground and not args.vlm_image:
        print("[WARN] --spatial_ground 1 needs --vlm_image 1 (per-control Qwen-image features).", flush=True)
    if is_main:
        print(f"[model] {model.param_report()}", flush=True)
    enc_ref = model.encoder

    # resume dynamics weights (stream11c) BEFORE DDP wrap; strict=False (spatial + InfoNCE heads reinit)
    start_step = 0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        cur = model.state_dict()
        compatible = {k: v for k, v in ck["model"].items() if k in cur and cur[k].shape == v.shape}
        missing = [k for k in cur if k not in compatible]
        model.load_state_dict(compatible, strict=False)
        if is_main:
            print(f"[resume] {args.resume}: loaded {len(compatible)} tensors, "
                  f"reinit {len(missing)} (spatial/InfoNCE heads e.g. {[m for m in missing if 'vis_' in m][:3]})", flush=True)
        # do NOT load optimizer / step: this is a NEW task (sim) with a changed param set + fresh LR schedule.

    if ddp:
        model = DDP(model, device_ids=[local], static_graph=True,
                    gradient_as_bucket_view=True, broadcast_buffers=False)

    # two param groups: base (lr) and freshly-init spatial-grounding (lr_sg, higher) as in the overfit.
    # Exp-1's dyn_head/sem_head are also freshly-init -> train them at lr_sg too.
    sg_names = ("vis_tok", "vis_film", "vis_norm", "vis_vhead", "dyn_head", "sem_head", "sem_proto")
    sg_params = [p for n, p in model.named_parameters() if p.requires_grad and any(s in n for s in sg_names)]
    base_params = [p for n, p in model.named_parameters() if p.requires_grad and not any(s in n for s in sg_names)]
    if args.spatial_ground and sg_params:
        opt = torch.optim.AdamW([{"params": base_params, "lr": args.lr},
                                 {"params": sg_params, "lr": args.lr_sg}],
                                weight_decay=1e-4, betas=(0.9, 0.95))
        lr_base = [args.lr, args.lr_sg]
    else:
        opt = torch.optim.AdamW(base_params, lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))
        lr_base = [args.lr]

    def lr_scale(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, args.total_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    writer = None
    if is_main:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(args.out, "tb"))
        except Exception:
            pass

    gen = torch.Generator(device=dev); gen.manual_seed(42 + rank)
    proj_dim = (model.module if ddp else model).proj_dim
    q_g = torch.nn.functional.normalize(torch.randn(args.queue_size, proj_dim, device=dev), dim=-1)
    q_t = torch.nn.functional.normalize(torch.randn(args.queue_size, proj_dim, device=dev), dim=-1)
    amp = torch.autocast("cuda", dtype=torch.bfloat16)

    step = 0; t0 = time.time(); model.train()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for clip in loader:
            if step >= args.total_steps or (args.max_steps and step >= args.max_steps):
                break
            sc = lr_scale(step)
            for gi, g in enumerate(opt.param_groups):
                g["lr"] = lr_base[gi] * sc
            opt.zero_grad(set_to_none=True)

            # ---- build the clip on GPU (no lift/track; clip IS the clean GT) ----
            ok = True
            try:
                g0 = GaussianSet(
                    clip["means"].to(dev), clip["quats"].to(dev), clip["scales"].to(dev),
                    clip["opacities"].to(dev), clip["colors"].to(dev), None).to(dev)
                N = len(g0)
                uv = clip["uv"].to(dev)                              # [N,2] frame-0 pixels
                traj_full = clip["traj"].to(dev)                    # [Kf+1,N,3] EXACT GT
                Kf = int(clip["Kf"]); K = min(args.K, Kf)
                H, W = int(clip["H"]), int(clip["W"])
                K_intr = clip["K_intr"].to(dev).float()             # [3,3]
                viewmat = clip["viewmat"].to(dev).float()           # [4,4] world->cam (static)
                gt_rgb = clip["gt_rgb"].to(dev).float() / 255.0     # [Kf+1,H,W,3]
                instruction = clip["instruction"]

                disp_all = (traj_full[K] - traj_full[0]).norm(dim=-1)   # [N]
                ctrl_idx = sample_controls(g0.means, disp_all, args.M, gen, args.mover_thresh)
                M = ctrl_idx.numel()
                control_uv = uv[ctrl_idx]                           # [M,2]
                gt_pos = traj_full[:, ctrl_idx, :]                  # [K+1,M,3]
                init = g0.means[ctrl_idx]                           # [M,3]
                gt_traj = gt_pos[1:K + 1]                           # [K,M,3]
                vis = torch.ones(K + 1, M, dtype=torch.bool, device=dev)   # SIM = fully observed
                vis_traj = vis[1:K + 1]
                # per-control task relevance from GT motion (for obj_focus if enabled)
                gtm = (gt_pos - gt_pos[:1]).norm(dim=-1).amax(0)
                rel = (gtm / gtm.quantile(0.9).clamp_min(1e-6)).clamp(0, 1)
                knn_idx = torch.cdist(init, init).topk(args.rot_knn + 1, largest=False).indices[:, 1:]
                # Exp-1: FREE per-control MOVER LABEL from the exact GT trajectory (no extra Qwen call,
                # no stored field): a control is a mover iff its frame0->K displacement exceeds the
                # threshold. This is exactly disp_all (already used for control sampling) sliced to the
                # control set -> the BCE target for the dyn-gate. Object-semantic uses seg_per_g.
                mover_label = (disp_all[ctrl_idx] > args.mover_thresh).float()      # [M]
                seg_ctrl = clip["seg_per_g"].to(dev)[ctrl_idx] if args.sem_dim > 0 else None  # [M]
                vlm_img = (gt_rgb[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy() if args.vlm_image else None
                vlm_inputs = move_vlm_inputs(enc_ref.build_inputs(instruction, vlm_img), dev, torch.bfloat16)
                if not (torch.isfinite(traj_full).all() and torch.isfinite(g0.means).all()
                        and torch.isfinite(g0.scales).all() and torch.isfinite(K_intr).all()
                        and torch.isfinite(viewmat).all()):
                    ok = False
            except Exception as e:
                ok = False
                if is_main:
                    print(f"[skip] prep failed: {type(e).__name__}: {e}", flush=True)

            if ddp:
                flag = torch.tensor([1.0 if ok else 0.0], device=dev)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                ok = flag.item() > 0.5
            if not ok:
                step += 1
                continue

            # ---- forward / loss / backward ----
            with amp:
                out = model(vlm_inputs, g0, K, ctrl_idx=ctrl_idx,
                            control_uv=(control_uv if args.spatial_ground else None),
                            control_uv_hw=((H, W) if args.spatial_ground else None))
            pos_l, vel_l = trajectory_loss(out["ctrl"].float(), gt_traj.float(), vis_traj, init.float(),
                                           relevance=rel, obj_focus=args.obj_focus)
            mag_l = (mover_magnitude_loss(out["ctrl"].float(), gt_traj.float(), init.float(), vis_traj,
                                          mover_thresh=args.mover_thresh)
                     if args.w_mag > 0 else out["v"].new_zeros(()))
            gt_full = torch.cat([init[None], gt_traj], 0)
            rot_l = rotation_loss(out["om"].float(), gt_full.float(), knn_idx,
                                  torch.cat([vis[:1], vis_traj], 0))
            lang_c = contrastive_infonce(out["motion_emb"], out["lang_emb"], q_g, q_t, tau=args.lang_tau)
            with torch.no_grad():
                if torch.isfinite(out["motion_emb"]).all() and torch.isfinite(out["lang_emb"]).all():
                    q_g = torch.cat([q_g, out["motion_emb"].detach().unsqueeze(0)], 0)[-args.queue_size:]
                    q_t = torch.cat([q_t, out["lang_emb"].detach().unsqueeze(0)], 0)[-args.queue_size:]
            # render aux on a few steps from the clip's STATIC camera (same K_intr/viewmat each frame)
            rsteps = sorted(set(np.linspace(0, K - 1, min(args.render_steps, K)).astype(int).tolist()))
            rloss = out["ctrl"].new_zeros(()); ps = []
            for k in rsteps:
                tt = k + 1
                s = GaussianSet(out["means"][k].float(), out["quats"][k].float(), out["scales"][k].float(),
                                out["opacities"][k].float(), out["colors"][k].float(), None)
                colors, _, _ = render_gaussianset(s, viewmat[None], K_intr[None], W, H)
                pl, _, _ = photometric_loss(colors[0], gt_rgb[tt]); rloss = rloss + pl
                ps.append(psnr(colors[0].clamp(0, 1).detach(), gt_rgb[tt]))
            rloss = rloss / max(1, len(rsteps))
            reg = delta_reg(out["v"], out["om"], out["dls"])
            vel = velocity_smoothness([out["ctrl"][i] for i in range(out["ctrl"].shape[0])])
            scale_a = scale_anchor_loss(out["scales"].float(), g0.scales.float())
            # Exp-1: mover-BCE on the dyn-gate + (optional) object-semantic loss. Zero when off so the
            # A/B baseline (--dyn_gate 0) is byte-for-byte the old total.
            dyn_l = out["v"].new_zeros(())
            seg_l = out["v"].new_zeros(())
            if args.dyn_gate and "p_dyn" in out:
                dyn_l = mover_bce_loss(out["p_dyn"].float(), mover_label, vis[K - 1])
                if args.sem_dim > 0 and "e_sem" in out:
                    seg_l = semantic_id_loss(out["e_sem"], seg_ctrl, knn_idx,
                                             sem_proto=out.get("sem_proto"))
            total = (args.w_traj_pos * pos_l + args.w_traj_vel * vel_l + args.w_traj_rot * rot_l
                     + args.w_lang_contrast * lang_c + args.w_render * rloss
                     + args.w_reg * reg + args.w_vel * vel + args.w_scale_anchor * scale_a
                     + args.w_dyn * dyn_l + args.w_seg * seg_l + args.w_mag * mag_l)
            # NaN/Inf guard (DDP-safe): always backward (lockstep), skip opt.step iff global gnorm non-finite.
            total.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(gnorm):
                opt.step()
            elif is_main:
                mf = bool(torch.isfinite(out["means"]).all())
                print(f"[skip] non-finite grad @s{step}: pos{pos_l.item():.3f} vel{vel_l.item():.3f} "
                      f"rot{rot_l.item():.3f} lang{lang_c.item():.3f} render{rloss.item():.3f} "
                      f"meansOK={mf} env={clip.get('env')} seed={clip.get('seed')} -> step skipped", flush=True)

            if is_main and step % args.log_every == 0:
                # localization metric on THIS train clip (corr + top-mover ratio), like the overfit
                with torch.no_grad():
                    radius = (init - init.mean(0)).norm(dim=-1).quantile(0.9).clamp_min(1e-6)
                    gt_disp = (gt_pos[K] - gt_pos[0]).norm(dim=-1) / radius
                    pd = (out["ctrl"][K - 1].float() - init).norm(dim=-1) / radius
                    corr = torch.corrcoef(torch.stack([gt_disp.float(), pd.float()]))[0, 1]
                    topk = gt_disp.topk(max(1, M // 20)).indices
                    ratio = (pd[topk].mean() / gt_disp[topk].mean().clamp_min(1e-6)).item()
                    # Exp-1 metrics: static-leakage (mean PRED disp of GT-static controls -> should ->0)
                    # and mover precision/recall of sigmoid(p_dyn)>0.5 vs the free mover label.
                    stat_m = mover_label < 0.5                              # GT-static controls
                    leak = pd[stat_m].mean().item() if stat_m.any() else float("nan")
                    mp_prec = mp_rec = float("nan")
                    if args.dyn_gate and "p_dyn" in out:
                        pred_mv = torch.sigmoid(out["p_dyn"].float()) > 0.5
                        gt_mv = mover_label > 0.5
                        tp = (pred_mv & gt_mv).sum().float()
                        mp_prec = (tp / pred_mv.sum().clamp_min(1)).item()
                        mp_rec = (tp / gt_mv.sum().clamp_min(1)).item()
                mp = float(np.mean(ps)); rate = (step + 1) / (time.time() - t0 + 1e-6)
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"e{epoch} s{step} lr{lr_base[0]*sc:.2e} pos{pos_l.item():.4f} vel{vel_l.item():.4f} "
                      f"rot{rot_l.item():.4f} lang{lang_c.item():.4f} scl{scale_a.item():.3f} "
                      f"dyn{dyn_l.item():.4f} seg{seg_l.item():.4f} rPSNR{mp:.1f} "
                      f"corr{corr.item():.3f} ratio{ratio:.2f} leak{leak:.4f} "
                      f"mP{mp_prec:.2f} mR{mp_rec:.2f} {rate:.2f}it/s peakGB{mem:.1f}", flush=True)
                if writer:
                    writer.add_scalar("loss/mover_bce", dyn_l.item(), step)
                    writer.add_scalar("loss/semantic", seg_l.item(), step)
                    writer.add_scalar("metric/static_leakage", leak, step)
                    if mp_prec == mp_prec:
                        writer.add_scalar("metric/mover_precision", mp_prec, step)
                        writer.add_scalar("metric/mover_recall", mp_rec, step)
                    writer.add_scalar("loss/traj_pos", pos_l.item(), step)
                    writer.add_scalar("loss/traj_vel", vel_l.item(), step)
                    writer.add_scalar("loss/traj_rot", rot_l.item(), step)
                    writer.add_scalar("loss/lang_contrast", lang_c.item(), step)
                    writer.add_scalar("metric/train_corr", corr.item(), step)
                    writer.add_scalar("metric/train_ratio", ratio, step)
                    writer.add_scalar("metric/psnr_aux", mp, step)
            if is_main and step > 0 and step % args.ckpt_every == 0:
                ckpt = {"model": (model.module if ddp else model).state_dict(),
                        "opt": opt.state_dict(), "step": step, "cfg": cfg.__dict__,
                        "n_query": args.n_query, "M": args.M, "spatial_ground": args.spatial_ground,
                        "cond_mode": args.cond_mode, "dyn_gate": args.dyn_gate, "sem_dim": args.sem_dim,
                        "gate_uses_sem": args.gate_uses_sem}
                torch.save(ckpt, os.path.join(args.out, f"ckpt_{step:07d}.pt"))
                torch.save(ckpt, os.path.join(args.out, "ckpt_last.pt"))
                print(f"[ckpt] @step {step}", flush=True)
            step += 1
        if step >= args.total_steps or (args.max_steps and step >= args.max_steps):
            break

    if is_main:
        ckpt = {"model": (model.module if ddp else model).state_dict(),
                "opt": opt.state_dict(), "step": step, "cfg": cfg.__dict__,
                "n_query": args.n_query, "M": args.M, "spatial_ground": args.spatial_ground,
                "cond_mode": args.cond_mode, "dyn_gate": args.dyn_gate, "sem_dim": args.sem_dim}
        torch.save(ckpt, os.path.join(args.out, "ckpt_last.pt"))
        print(f"[done] step {step}", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
