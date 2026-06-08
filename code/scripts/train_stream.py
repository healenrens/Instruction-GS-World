"""Streaming trainer (scale-up) — train over the FULL corpus with zero clip storage.

JIT pipeline: CPU DataLoader workers seek-decode random clips across ALL tasks; the
trainer lifts them with Pi3 (frozen, no-grad) inline, then trains the ≥1B model
(frozen Qwen3-VL + 1.66B dynamics) with SC-GS render-supervised rollout. No
pre-cached clips -> bounded storage, unbounded data.

  torchrun --nproc_per_node=4 code/scripts/train_stream.py --out ./checkpoints/stream1
"""

import argparse
import contextlib
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.data.streaming import StreamingClipDataset  # noqa: E402
from igsw.data.clip_dataset import identity_collate  # noqa: E402
from igsw.lifting import Pi3Lifter, points_to_gaussians  # noqa: E402
from igsw.lifting.tracking import CoTrackerTracker, sample_pointmaps_at  # noqa: E402
from igsw.gaussians import (GaussianSet, render_gaussianset, psnr,  # noqa: E402
                            intrinsics_from_local_points, viewmat_from_pose)
from igsw.dynamics.model import DynamicsConfig  # noqa: E402
from igsw.model_full import InstructGSWorldModel  # noqa: E402
from igsw.training import photometric_loss, delta_reg, velocity_smoothness  # noqa: E402
from igsw.training.losses import trajectory_loss, rotation_loss, contrastive_lang_loss, background_static_loss, contrastive_infonce, scale_anchor_loss  # noqa: E402
from igsw.grounding import RoleGrounder  # noqa: E402
from igsw.grounding.relevance import role_phrases_for_clip, sample_mask_at_uv  # noqa: E402
import random as _random
from collections import deque as _deque


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank(); world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(local)
        return True, rank, world, local
    return False, 0, 1, 0


def noise_augment(g0, frac, gen):
    if frac <= 0:
        return g0
    center = g0.means.mean(0, keepdim=True)
    radius = (g0.means - center).norm(dim=-1).quantile(0.9).clamp_min(1e-6)
    nm = g0.means + torch.randn(g0.means.shape, device=g0.device, generator=gen) * (frac * radius)
    nc = (g0.colors + torch.randn(g0.colors.shape, device=g0.device, generator=gen) * (0.5 * frac)).clamp(0, 1)
    feats = g0.features.clone() if g0.features is not None else None   # relevance is static -> not noised
    return GaussianSet(nm, g0.quats.clone(), g0.scales.clone(), g0.opacities.clone(), nc, feats)


def move_vlm_inputs(inputs, dev, dtype):
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            out[k] = v.to(dev, dtype=dtype) if v.is_floating_point() else v.to(dev)
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/pfs/public/xuhaoming/instruct_gs_world/checkpoints/stream1")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--n_query", type=int, default=16)
    ap.add_argument("--cond_mode", default="aggregator", choices=["aggregator", "metaquery"],
                    help="conditioning: 'aggregator' (default baseline) | 'metaquery' (append "
                         "learnable queries into frozen Qwen; needs --vlm_image 1)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--total_steps", type=int, default=300000)
    ap.add_argument("--noise_frac", type=float, default=0.0)   # direct-loss regime: off by default
    ap.add_argument("--w_traj_pos", type=float, default=1.0)   # PRIMARY: direct 3D position loss
    ap.add_argument("--w_traj_vel", type=float, default=1.0)   # PRIMARY: direct 3D velocity (motion) loss
    ap.add_argument("--w_traj_rot", type=float, default=0.2)   # direct per-Gaussian rotation (Kabsch GT)
    ap.add_argument("--rot_knn", type=int, default=8)          # neighbours for local Kabsch rotation
    ap.add_argument("--w_lang_contrast", type=float, default=0.5)  # weight on the InfoNCE language loss
    ap.add_argument("--lang_margin", type=float, default=0.003)    # (legacy hinge; unused with InfoNCE)
    ap.add_argument("--lang_tau", type=float, default=0.07)        # InfoNCE temperature
    ap.add_argument("--queue_size", type=int, default=256)         # MoCo-style negative queue size
    ap.add_argument("--vlm_image", type=int, default=0)   # 0=TEXT-ONLY VLM cond (text can't be bypassed via image), 1=image+text
    ap.add_argument("--action_dim", type=int, default=8)  # >0 = action-conditioned (per-step EEF Δ); 0=off
    ap.add_argument("--w_render", type=float, default=0.1)     # AUXILIARY: 2D render/PSNR
    ap.add_argument("--w_reg", type=float, default=1e-3)
    ap.add_argument("--w_scale_anchor", type=float, default=0.0)  # ANTI-DRIFT: anchor rolled-out scales to G0 (stops s·exp(δs) blowup -> long-horizon)
    ap.add_argument("--w_vel", type=float, default=1e-3)       # rollout accel smoothness
    ap.add_argument("--grad_accum", type=int, default=1)       # effective batch = grad_accum * world
    ap.add_argument("--render_steps", type=int, default=2)     # # rollout steps rendered for the aux loss
    ap.add_argument("--checkpoint_every", type=int, default=2)  # 1=all(safe), 2=half(faster), 0=none
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--min_gaussians", type=int, default=512)    # skip degenerate clips below this
    ap.add_argument("--max_gaussians", type=int, default=150000) # cap dense set to bound memory
    ap.add_argument("--boundary_frac", type=float, default=0.35)  # onset window = first frac of each sub-task
    ap.add_argument("--boundary_ratio", type=float, default=4.0)  # sampling odds boundary:middle (try up to 10)
    ap.add_argument("--middle_weight", type=float, default=0.3)   # loss weight for mid-action clips (auxiliary)
    ap.add_argument("--use_grounding", type=int, default=0)      # v7-min: GroundingDINO+SAM2 role masks
    ap.add_argument("--w_bg_static", type=float, default=0.5)    # freeze background Gaussians
    ap.add_argument("--obj_focus", type=float, default=2.0)      # weight traj loss toward task-relevant Gaussians
    ap.add_argument("--feature_dim", type=int, default=0)        # v7-min step2: feed relevance as model INPUT token (1)
    ap.add_argument("--spatial_ground", type=int, default=0)     # §37: per-control Qwen-image feature at frame-0 uv (1=on)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--resume", default="")
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--detect_anomaly", type=int, default=0,
                    help="debug: torch.autograd.detect_anomaly -> crash at the EXACT op producing a non-finite grad")
    args = ap.parse_args()
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    ddp, rank, world, local = setup_ddp()
    dev = torch.device(f"cuda:{local}")
    is_main = rank == 0
    if is_main:
        os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(1234 + rank)

    lifter = Pi3Lifter(device=f"cuda:{local}")
    tracker = CoTrackerTracker(device=f"cuda:{local}")   # frozen, for GT correspondence
    grounder = RoleGrounder(device=f"cuda:{local}") if args.use_grounding else None  # v7-min role masks

    ds = StreamingClipDataset(K=args.K, stride=args.stride, seed=1234 + 17 * rank,
                              boundary_frac=args.boundary_frac, boundary_ratio=args.boundary_ratio,
                              middle_weight=args.middle_weight)
    loader = DataLoader(ds, batch_size=1, num_workers=args.workers, collate_fn=identity_collate,
                        persistent_workers=True, prefetch_factor=args.prefetch, pin_memory=False)
    if is_main:
        print(f"[stream] {ds.n_episodes} episodes across {len(ds.task_roots)} tasks | world={world}", flush=True)

    cfg = DynamicsConfig(d_model=args.dim, n_layers=args.layers, n_heads=args.heads,
                         lang_dim=2048, use_checkpoint=True, checkpoint_every=args.checkpoint_every,
                         feature_dim=args.feature_dim)
    model = InstructGSWorldModel(cfg, n_control=args.M, n_query=args.n_query, action_dim=args.action_dim,
                                 cond_mode=args.cond_mode, spatial_ground=bool(args.spatial_ground)).to(dev)
    if is_main and args.spatial_ground and not args.vlm_image:
        print("[WARN] --spatial_ground 1 needs the image (--vlm_image 1) to sample per-control "
              "Qwen-image features; with text-only there is no image grid -> grounding is a no-op.",
              flush=True)
    if is_main and args.cond_mode == "metaquery" and not args.vlm_image:
        print("[WARN] cond_mode='metaquery' requires --vlm_image 1 (it needs the image); "
              "set --vlm_image 1.", flush=True)
    if is_main:
        print(f"[model] {model.param_report()}", flush=True)
    enc_ref = model.encoder
    if ddp:
        model = DDP(model, device_ids=[local], static_graph=True,
                    gradient_as_bucket_view=True, broadcast_buffers=False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, args.total_steps - args.warmup)
        return 0.5 * args.lr * (1 + math.cos(math.pi * min(1.0, p)))

    writer = None
    if is_main:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(args.out, "tb"))
        except Exception:
            pass

    start_step = 0
    if args.resume and os.path.isfile(args.resume):
        # load to CPU: map_location=dev would pin the whole ckpt (incl. ~28GB optimizer
        # state) on the GPU and keep it referenced -> wastes VRAM (caused 45.8->71.8GB).
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        tgt = (model.module if ddp else model)
        # shape-filtered load: skip params whose shape changed (e.g. tokenizer.proj when
        # feature_dim grows for v7-min step2) so warm-start still loads everything compatible.
        cur = tgt.state_dict()
        compatible = {k: v for k, v in ck["model"].items() if k in cur and cur[k].shape == v.shape}
        skipped = [k for k, v in ck["model"].items() if k in cur and cur[k].shape != v.shape]
        tgt.load_state_dict(compatible, strict=False)
        if is_main and skipped:
            print(f"[resume] shape-skipped {len(skipped)} params (e.g. {skipped[:2]})", flush=True)
        # Load optimizer state ONLY if the architecture is unchanged. If any param shape
        # changed (shape-skipped above), the saved Adam m/v have stale shapes -> the error
        # surfaces at .step(), not at load -> so gate on `skipped` and use a fresh optimizer.
        if "opt" in ck and not skipped:
            try:
                opt.load_state_dict(ck["opt"])
            except Exception as e:
                if is_main:
                    print(f"[resume] optimizer state skipped: {type(e).__name__}", flush=True)
        elif is_main and skipped:
            print("[resume] fresh optimizer (arch changed -> stale opt state not loaded)", flush=True)
        start_step = ck.get("step", 0)
        if is_main:
            print(f"[resume] {args.resume} @step {start_step}", flush=True)

    gen = torch.Generator(device=dev); gen.manual_seed(42 + rank)
    rng = _random.Random(7 + rank)
    instr_buffer = _deque(maxlen=128)   # recent instructions -> sample negatives for the contrastive loss
    proj_dim = (model.module if ddp else model).proj_dim
    # MoCo-style negative queues, PRE-FILLED to constant size (random unit vecs, rotated out by real
    # clips) so the graph shape is constant for DDP static_graph.
    q_g = torch.nn.functional.normalize(torch.randn(args.queue_size, proj_dim, device=dev), dim=-1)
    q_t = torch.nn.functional.normalize(torch.randn(args.queue_size, proj_dim, device=dev), dim=-1)
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    step = start_step; t0 = time.time(); model.train()
    for clip in loader:
        if step >= args.total_steps or (args.max_steps and step >= start_step + args.max_steps):
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        if step % args.grad_accum == 0:
            opt.zero_grad(set_to_none=True)

        # ---- sample prep: lift + correspondence + GT trajectory (NO collectives) ----
        ok = True
        try:
            frames = clip["frames"]                       # uint8 [K+1,H,W,3]
            res = lifter.lift(frames, conf_thr=0.1, edge_rtol=0.03)   # no_grad inside
            pts_all = res["points"].to(dev); imgs = res["images"].to(dev); mask = res["mask"].to(dev)
            local = res["local_points"].to(dev); poses = res["camera_poses"].to(dev)
            Kf1, _, H, W = imgs.shape
            K = min(args.K, Kf1 - 1)
            g0, uv = points_to_gaussians(pts_all[:1], imgs[:1], mask[:1], opacity_init=0.9,
                                         scale_factor=1.0, return_uv=True)
            N = len(g0)
            if N < args.min_gaussians:
                ok = False
            else:
                if N > args.max_gaussians:                # cap dense set to bound memory
                    keep = torch.randperm(N, device=dev)[:args.max_gaussians]
                    g0 = GaussianSet(g0.means[keep], g0.quats[keep], g0.scales[keep],
                                     g0.opacities[keep], g0.colors[keep], None)
                    uv = uv[keep]; N = args.max_gaussians
                Ks = torch.stack([intrinsics_from_local_points(local[i]) for i in range(Kf1)], 0)
                viewmats = torch.stack([viewmat_from_pose(poses[i]) for i in range(Kf1)], 0)
                gt_img = imgs.permute(0, 2, 3, 1)
                M = min(args.M, N)
                ctrl_idx = torch.randperm(N, device=dev)[:M]
                control_uv = uv[ctrl_idx]
                # v7-min: language-conditioned role masks -> per-control task relevance / bg weight
                rel_control = None; bg_control = None
                if grounder is not None:
                    f0_np = (gt_img[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                    _, fg_roles = role_phrases_for_clip(clip["instruction"])     # manipulator + instruction objects
                    fg_m = grounder.ground_roles(f0_np, fg_roles)
                    fg2d = np.maximum.reduce(list(fg_m.values())) if fg_m else np.zeros((H, W), np.float32)
                    # dilate foreground a little so grasped/adjacent movers are included
                    rel2d = torch.from_numpy(fg2d).to(dev).float()
                    rel_control = sample_mask_at_uv(rel2d, control_uv)          # [M] foreground/task relevance
                    bg_control = (1.0 - rel_control)                            # background = complement (robust)
                    if args.feature_dim > 0:                                    # step2: relevance as MODEL INPUT
                        rel_dense = sample_mask_at_uv(rel2d, uv)[:, None]       # [N,1]
                        g0 = GaussianSet(g0.means, g0.quats, g0.scales, g0.opacities, g0.colors, rel_dense)
                    if is_main and step < 80:
                        print(f"[ground] step{step} fg_cov={float(fg2d.mean()):.3f} "
                              f"rel_ctrl={float(rel_control.mean()):.3f}", flush=True)
                g0n = noise_augment(g0, args.noise_frac, gen)         # noised rollout start (anti-drift)
                tracks, vis = tracker.track(imgs, control_uv)         # [K+1,M,2],[K+1,M]
                gt_pos = sample_pointmaps_at(pts_all, tracks)         # [K+1,M,3]
                init = g0n.means[ctrl_idx]   # loss start == rollout start -> model learns to denoise toward clean GT
                gt_traj = gt_pos[1:K + 1]; vis_traj = vis[1:K + 1]
                if rel_control is None:
                    # SELF-SUPERVISED task-relevance from GT MOTION (no grounding): the per-control
                    # motion is heavy-tailed (most points ~static, a few move a lot = arm/object). An
                    # L1 loss normalized over ALL controls collapses to the static median (the model
                    # predicted ~1% of real motion + a global drift). Weighting by GT motion makes the
                    # MOVERS dominate the traj loss (obj_focus) while the static rest is held in place
                    # (bg_static) -> the model learns LOCALIZED articulated motion instead of "static".
                    gtm = (gt_pos - gt_pos[:1]).norm(dim=-1).amax(0)               # [M] max displacement / control
                    rel_control = (gtm / gtm.quantile(0.9).clamp_min(1e-6)).clamp(0, 1)   # movers~1, static~0
                if args.feature_dim > 0:
                    # CONFIRMATION EXPERIMENT: feed per-control task-relevance as a MODEL INPUT (a
                    # stand-in for grounding). If the dynamics then LOCALIZES motion to the movers
                    # (vs the uniform global drift it produces blind), it proves the architecture is
                    # capable and the missing piece is GROUNDING (deriving relevance from lang+image).
                    feat = g0n.means.new_zeros(len(g0n), args.feature_dim)
                    feat[ctrl_idx, 0] = rel_control
                    g0n = GaussianSet(g0n.means, g0n.quats, g0n.scales, g0n.opacities, g0n.colors, feat)
                # local k-NN among control points (clean positions) for Kabsch rotation GT
                clean_ctrl = g0.means[ctrl_idx]
                knn_idx = torch.cdist(clean_ctrl, clean_ctrl).topk(args.rot_knn + 1, largest=False).indices[:, 1:]
                vlm_img = (gt_img[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy() if args.vlm_image else None
                vlm_inputs = move_vlm_inputs(enc_ref.build_inputs(clip["instruction"], vlm_img), dev, torch.bfloat16)
                # InfoNCE uses a queue of OTHER clips' embeddings as negatives (no wrong-instruction
                # forward needed) -> simpler + saves a Qwen pass.
                vlm_inputs_wrong = None
                actions = None
                if args.action_dim > 0 and "actions" in clip:
                    actions = torch.as_tensor(clip["actions"], dtype=torch.float32, device=dev)[:K]
                # finite-check ALL render/loss inputs (not just gt_traj+means): Pi3 can emit a
                # degenerate CAMERA (Ks/viewmats) or scale for a rare frame -> NaN render -> NaN
                # loss/grad. These were previously unchecked = a real NaN entry path.
                if not (torch.isfinite(gt_traj).all() and torch.isfinite(g0.means).all()
                        and torch.isfinite(g0.scales).all()
                        and torch.isfinite(Ks).all() and torch.isfinite(viewmats).all()):
                    ok = False
        except Exception as e:
            ok = False
            if is_main:
                print(f"[skip] prep failed: {type(e).__name__}: {e}", flush=True)

        # ---- coordinate skip across ranks (collective every rank reaches) ----
        if ddp:
            flag = torch.tensor([1.0 if ok else 0.0], device=dev)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            ok = flag.item() > 0.5
        if not ok:
            step += 1
            continue

        # ---- forward / loss / backward (all ranks have valid data) ----
        with amp:
            out = model(vlm_inputs, g0n, K, ctrl_idx=ctrl_idx, vlm_inputs_wrong=vlm_inputs_wrong,
                        actions=actions,
                        control_uv=(control_uv if args.spatial_ground else None),
                        control_uv_hw=((H, W) if args.spatial_ground else None))
        pos_l, vel_l = trajectory_loss(out["ctrl"].float(), gt_traj.float(), vis_traj, init.float(),
                                       relevance=rel_control,
                                       obj_focus=(args.obj_focus if rel_control is not None else 0.0))
        rot_l = rotation_loss(out["om"].float(), gt_pos.float(), knn_idx, vis)
        bg_static_l = out["v"].float().new_zeros(())
        if rel_control is not None:                     # v7-min: hold background at its start (anti-drift)
            bg_static_l = background_static_loss(out["ctrl"].float(), init.float(), rel_control)
        # InfoNCE: force the PREDICTED motion to be instruction-specific (non-saturating; research_F).
        lang_c = contrastive_infonce(out["motion_emb"], out["lang_emb"], q_g, q_t, tau=args.lang_tau)
        with torch.no_grad():                                # update MoCo-style negative queues
            if torch.isfinite(out["motion_emb"]).all() and torch.isfinite(out["lang_emb"]).all():
                q_g = torch.cat([q_g, out["motion_emb"].detach().unsqueeze(0)], 0)[-args.queue_size:]
                q_t = torch.cat([q_t, out["lang_emb"].detach().unsqueeze(0)], 0)[-args.queue_size:]
        rsteps = sorted(set(np.linspace(0, K - 1, min(args.render_steps, K)).astype(int).tolist()))
        rloss = out["ctrl"].new_zeros(()); ps = []
        for k in rsteps:
            tt = k + 1
            s = GaussianSet(out["means"][k].float(), out["quats"][k].float(), out["scales"][k].float(),
                            out["opacities"][k].float(), out["colors"][k].float(), None)
            colors, _, _ = render_gaussianset(s, viewmats[tt], Ks[tt], W, H)
            pl, _, _ = photometric_loss(colors[0], gt_img[tt]); rloss = rloss + pl
            ps.append(psnr(colors[0].clamp(0, 1).detach(), gt_img[tt]))
        rloss = rloss / max(1, len(rsteps))
        reg = delta_reg(out["v"], out["om"], out["dls"])
        vel = velocity_smoothness([out["ctrl"][i] for i in range(out["ctrl"].shape[0])])
        scale_a = scale_anchor_loss(out["scales"].float(), g0.scales.float())   # anti-drift: stop scale blowup
        total = (args.w_traj_pos * pos_l + args.w_traj_vel * vel_l + args.w_traj_rot * rot_l
                 + args.w_lang_contrast * lang_c + args.w_bg_static * bg_static_l
                 + args.w_render * rloss + args.w_reg * reg + args.w_vel * vel
                 + args.w_scale_anchor * scale_a) / args.grad_accum
        # down-weight mid-action clips (boundary/onset clips, where language matters, keep full weight)
        total = total * float(clip.get("boundary_weight", 1.0))
        # NaN/Inf guard (DDP-safe). ALWAYS backward so DDP's grad all-reduce + static_graph stay
        # in lockstep across ranks (conditionally skipping backward on some ranks desyncs the
        # collectives). After backward the grads are all-reduced => identical on every rank, so
        # clip_grad_norm_'s norm is identical and the skip decision is consistent. If that GLOBAL
        # norm is non-finite, SKIP opt.step: otherwise clip_grad_norm_'s coef (max/total_norm)
        # would scale every grad to NaN and the optimizer would poison all 1.76B params at once
        # (root cause of the s6000 corruption). A rare degenerate clip/render then costs one
        # wasted step, not the whole run; the NaN grads are cleared by the next window's zero_grad.
        total.backward()
        if (step + 1) % args.grad_accum == 0:
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(gnorm):
                opt.step()
            elif is_main:
                # pinpoint the ORIGIN: which loss term is non-finite (forward NaN) + whether the
                # forward outputs / cameras blew up (render path) vs all-finite-but-grad-NaN
                # (a backward-only NaN, e.g. the rotation exp-map). + the exact clip for offline repro.
                mf = bool(torch.isfinite(out["means"]).all()); cf = bool(torch.isfinite(Ks).all() and torch.isfinite(viewmats).all())
                print(f"[skip] non-finite grad-norm @s{step}: pos{pos_l.item():.3f} vel{vel_l.item():.3f} "
                      f"rot{rot_l.item():.3f} lang{lang_c.item():.3f} render{rloss.item():.3f} bg{float(bg_static_l):.3f} "
                      f"| meansOK={mf} camOK={cf} task={clip.get('task')} ep={clip.get('ep')} f0={clip.get('f0')} "
                      f"-> opt.step skipped", flush=True)

        if is_main and step % args.log_every == 0:
            mp = float(np.mean(ps)); rate = (step - start_step + 1) / (time.time() - t0 + 1e-6)
            mem = torch.cuda.max_memory_allocated() / 1e9
            vfrac = float(vis_traj.float().mean())
            print(f"s{step} lr{lr_at(step):.2e} pos{pos_l.item():.4f} vel{vel_l.item():.4f} "
                  f"rot{rot_l.item():.4f} lang{lang_c.item():.4f} bg{float(bg_static_l):.2e} scl{scale_a.item():.3f} "
                  f"rPSNR{mp:.2f} vis{vfrac:.2f} {rate:.2f}it/s peakGB{mem:.1f}", flush=True)
            if writer:
                writer.add_scalar("loss/traj_pos", pos_l.item(), step)
                writer.add_scalar("loss/traj_vel", vel_l.item(), step)
                writer.add_scalar("loss/traj_rot", rot_l.item(), step)
                writer.add_scalar("loss/lang_contrast", lang_c.item(), step)
                writer.add_scalar("metric/psnr_aux", mp, step)
        if is_main and step > 0 and step % args.ckpt_every == 0:
            ckpt = {"model": (model.module if ddp else model).state_dict(),
                    "opt": opt.state_dict(), "step": step, "cfg": cfg.__dict__,
                    "n_query": args.n_query, "M": args.M}
            torch.save(ckpt, os.path.join(args.out, f"ckpt_{step:07d}.pt"))
            torch.save(ckpt, os.path.join(args.out, "ckpt_last.pt"))
            print(f"[ckpt] @step {step}", flush=True)
        step += 1

    if is_main:
        ckpt = {"model": (model.module if ddp else model).state_dict(),
                "opt": opt.state_dict(), "step": step, "cfg": cfg.__dict__,
                "n_query": args.n_query, "M": args.M}
        torch.save(ckpt, os.path.join(args.out, "ckpt_last.pt"))
        print(f"[done] step {step}", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
