"""Small multi-clip Stage-B DynamicGS training/eval gate."""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.dynamics.model import DynamicsConfig  # noqa: E402
from igsw.gaussians import GaussianSet  # noqa: E402
from igsw.grounding.multiview_runtime import build_multiview_inputs  # noqa: E402
from igsw.model_full import InstructGSWorldModel  # noqa: E402
from igsw.training.losses import trajectory_loss  # noqa: E402


WRONG_INSTRS = [
    "Pick up the red cube and move it to the goal position.",
    "Push the cube to the goal region.",
    "Pick up the red cube and stack it on top of the green cube.",
]


def list_split(root, split):
    return sorted(glob.glob(os.path.join(root, f"*_{split}.pt")))


def sample_controls(disp_all, m_count, gen, mover_thresh):
    m_count = min(int(m_count), int(disp_all.shape[0]))
    movers = (disp_all > mover_thresh).nonzero(as_tuple=True)[0]
    static = (disp_all <= mover_thresh).nonzero(as_tuple=True)[0]
    n_mover = min(int(movers.numel()), m_count // 2)
    n_static = min(m_count - n_mover, int(static.numel()))
    n_mover = min(m_count - n_static, int(movers.numel()))
    sel_m = movers[torch.randperm(movers.numel(), device=disp_all.device, generator=gen)[:n_mover]]
    sel_s = static[torch.randperm(static.numel(), device=disp_all.device, generator=gen)[:n_static]]
    idx = torch.cat([sel_m, sel_s])
    return idx[torch.randperm(idx.numel(), device=disp_all.device, generator=gen)]


def load_clip(path, dev):
    c = torch.load(path, map_location="cpu", weights_only=False)
    if c.get("contract_version") != "dynamic_gs_stage_b_v1":
        raise ValueError(f"{path} is not a Stage-B clip")
    return c


def prepare_item(model, clip, dev, k_steps, m_count, seed, mover_thresh):
    g0 = GaussianSet(
        clip["means"].to(dev), clip["quats"].to(dev), clip["scales"].to(dev),
        clip["opacities"].to(dev), clip["colors"].to(dev), None).to(dev)
    traj_full = clip["traj"].to(dev)
    k_steps = min(int(k_steps), int(clip["Kf"]))
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    disp_all = (traj_full[k_steps] - traj_full[0]).norm(dim=-1)
    ctrl_idx = sample_controls(disp_all, m_count, gen, mover_thresh)
    init = g0.means[ctrl_idx]
    gt = traj_full[1:k_steps + 1, ctrl_idx]
    vis = torch.ones(k_steps, ctrl_idx.numel(), dtype=torch.bool, device=dev)
    uvv = clip["uv_by_view"].to(dev)[ctrl_idx]
    valid = clip["uv_valid_by_view"].to(dev)[ctrl_idx]
    vi, vi_views = build_multiview_inputs(
        model.encoder, clip["instruction"], clip["scan_rgb"], int(clip.get("policy_view", 0)),
        dev, torch.bfloat16)
    return (
        g0, ctrl_idx, init, gt, vis, uvv, valid, vi, vi_views, k_steps,
        clip["scan_rgb"], clip["instruction"], int(clip.get("policy_view", 0)),
    )


def metrics(pred, gt, init):
    radius = (init - init.mean(0)).norm(dim=-1).quantile(0.9).clamp_min(1e-6)
    gt_disp = (gt[-1] - init).norm(dim=-1) / radius
    pred_disp = (pred[-1] - init).norm(dim=-1) / radius
    corr = torch.corrcoef(torch.stack([gt_disp.float(), pred_disp.float()]))[0, 1]
    top = gt_disp.topk(max(1, gt_disp.numel() // 20)).indices
    ratio = pred_disp[top].mean() / gt_disp[top].mean().clamp_min(1e-6)
    return corr.item(), ratio.item(), (pred_disp > 0.02).float().mean().item(), (gt_disp > 0.02).float().mean().item()


def scheduled_weight(base, start, ramp, step):
    if base <= 0 or step < start:
        return 0.0
    if ramp <= 0:
        return float(base)
    return float(base) * min(1.0, float(step - start + 1) / float(ramp))


def forward_loss(model, item, hw, amp, args, step):
    g0, ctrl_idx, init, gt, vis, uvv, valid, vi, vi_views, k_steps, scan_rgb, instr, policy_view = item
    with amp:
        out = model(
            vi, g0, k_steps, ctrl_idx=ctrl_idx, control_uv_hw=hw,
            vlm_inputs_by_view=vi_views, control_uv_by_view=uvv, control_uv_valid_by_view=valid,
            freeze_appearance=bool(args.freeze_appearance),
            freeze_rotation=bool(args.freeze_rotation))
        pos, vel = trajectory_loss(out["ctrl"].float(), gt.float(), vis, init.float())
        motion = pos + vel
        lang = motion.new_zeros(())
        if args.w_lang_margin > 0:
            wrong = next(w for w in WRONG_INSTRS if w.strip() != str(instr).strip())
            vi_w, vi_views_w = build_multiview_inputs(
                model.encoder, wrong, scan_rgb, policy_view, g0.device, torch.bfloat16)
            out_w = model(
                vi_w, g0, k_steps, ctrl_idx=ctrl_idx, control_uv_hw=hw,
                vlm_inputs_by_view=vi_views_w, control_uv_by_view=uvv,
                control_uv_valid_by_view=valid, freeze_appearance=bool(args.freeze_appearance),
                freeze_rotation=bool(args.freeze_rotation))
            pos_w, vel_w = trajectory_loss(out_w["ctrl"].float(), gt.float(), vis, init.float())
            lang = lang + float(args.w_lang_margin) * torch.relu(float(args.lang_margin) + motion - (pos_w + vel_w))
        w_null_static = scheduled_weight(
            args.w_null_static, args.null_static_start, args.null_static_ramp, step)
        if args.w_null_margin > 0 or w_null_static > 0:
            vi_n, vi_views_n = build_multiview_inputs(
                model.encoder, "", scan_rgb, policy_view, g0.device, torch.bfloat16)
            out_n = model(
                vi_n, g0, k_steps, ctrl_idx=ctrl_idx, control_uv_hw=hw,
                vlm_inputs_by_view=vi_views_n, control_uv_by_view=uvv,
                control_uv_valid_by_view=valid, freeze_appearance=bool(args.freeze_appearance),
                freeze_rotation=bool(args.freeze_rotation))
            if args.w_null_margin > 0:
                pos_n, vel_n = trajectory_loss(out_n["ctrl"].float(), gt.float(), vis, init.float())
                lang = lang + float(args.w_null_margin) * torch.relu(float(args.lang_margin) + motion - (pos_n + vel_n))
            if w_null_static > 0:
                static_gt = init[None].expand(k_steps, -1, -1)
                n_pos, n_vel = trajectory_loss(out_n["ctrl"].float(), static_gt.float(), vis, init.float())
                lang = lang + w_null_static * (n_pos + n_vel)
        loss = motion + lang
    return loss, pos, vel, lang, out


def forward_item(model, item, hw, amp, args):
    g0, ctrl_idx, init, gt, vis, uvv, valid, vi, vi_views, k_steps = item[:10]
    with amp:
        out = model(
            vi, g0, k_steps, ctrl_idx=ctrl_idx, control_uv_hw=hw,
            vlm_inputs_by_view=vi_views, control_uv_by_view=uvv, control_uv_valid_by_view=valid,
            freeze_appearance=bool(args.freeze_appearance),
            freeze_rotation=bool(args.freeze_rotation))
        pos, vel = trajectory_loss(out["ctrl"].float(), gt.float(), vis, init.float())
    corr, ratio, fp, fg = metrics(out["ctrl"].detach().float(), gt.float(), init.float())
    return float((pos + vel).item()), corr, ratio, fp, fg


def evaluate(model, paths, dev, args, amp, label):
    model.eval()
    rows = []
    with torch.no_grad():
        for i, p in enumerate(paths):
            c = load_clip(p, dev)
            item = prepare_item(model, c, dev, args.K, args.M, 1000 + i, args.mover_thresh)
            rows.append(forward_item(model, item, (int(c["H"]), int(c["W"])), amp, args))
    arr = np.array(rows, dtype=np.float64)
    print(f"[eval {label}] n={len(rows)} loss={arr[:,0].mean():.5f} corr={np.nanmean(arr[:,1]):.3f} "
          f"ratio={np.nanmean(arr[:,2]):.2f} frac_pred={arr[:,3].mean():.3f} frac_gt={arr[:,4].mean():.3f}",
          flush=True)
    model.train()
    return arr


def save_checkpoint(path, model, cfg, args, step):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "cfg": cfg.__dict__,
        "M": args.M,
        "n_query": 16,
        "spatial_ground": 1,
        "cond_mode": "aggregator",
        "freeze_appearance": int(args.freeze_appearance),
        "freeze_rotation": int(args.freeze_rotation),
        "w_lang_margin": float(args.w_lang_margin),
        "w_null_margin": float(args.w_null_margin),
        "w_null_static": float(args.w_null_static),
        "null_static_start": int(args.null_static_start),
        "null_static_ramp": int(args.null_static_ramp),
        "lang_margin": float(args.lang_margin),
        "seed": int(args.seed),
        "step": int(step),
    }, path)
    print(f"[stage-b-small] saved {path}", flush=True)


def checkpoint_path(base, step):
    stem, ext = os.path.splitext(base)
    return f"{stem}_s{int(step)}{ext or '.pt'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--mover_thresh", type=float, default=0.01)
    ap.add_argument("--freeze_appearance", type=int, default=1)
    ap.add_argument("--freeze_rotation", type=int, default=0)
    ap.add_argument("--w_lang_margin", type=float, default=0.0)
    ap.add_argument("--w_null_margin", type=float, default=0.0)
    ap.add_argument("--w_null_static", type=float, default=0.0)
    ap.add_argument("--null_static_start", type=int, default=0)
    ap.add_argument("--null_static_ramp", type=int, default=0)
    ap.add_argument("--lang_margin", type=float, default=0.002)
    ap.add_argument("--init_ckpt", default="")
    ap.add_argument("--ckpt_out", default="")
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    dev = torch.device("cuda")
    train_paths = list_split(args.data, "train")
    held_paths = list_split(args.data, "heldseed")
    heldtask_paths = list_split(args.data, "heldtask")
    if not train_paths or (not held_paths and not heldtask_paths):
        raise RuntimeError(f"need train and held-out clips under {args.data}")
    cfg = DynamicsConfig(d_model=args.dim, n_layers=28, n_heads=args.heads,
                         lang_dim=2048, use_checkpoint=True, checkpoint_every=1)
    model = InstructGSWorldModel(cfg, n_control=args.M, n_query=16, spatial_ground=True).to(dev)
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        miss, unexp = model.load_state_dict(ck["model"], strict=False)
        print(f"[stage-b-small] init={args.init_ckpt} missing={len(miss)} unexpected={len(unexp)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    print(f"[stage-b-small] train={len(train_paths)} heldseed={len(held_paths)} "
          f"heldtask={len(heldtask_paths)} M={args.M} K={args.K} "
          f"freeze_app={int(args.freeze_appearance)} freeze_rot={int(args.freeze_rotation)} "
          f"w_lang={args.w_lang_margin} w_null={args.w_null_margin} "
          f"w_null_static={args.w_null_static} null_start={args.null_static_start} "
          f"null_ramp={args.null_static_ramp} seed={args.seed}", flush=True)
    evaluate(model, train_paths, dev, args, amp, "train0")
    if held_paths:
        evaluate(model, held_paths, dev, args, amp, "held0")
    if heldtask_paths:
        evaluate(model, heldtask_paths, dev, args, amp, "heldtask0")

    t0 = time.time()
    for step in range(int(args.steps)):
        p = train_paths[step % len(train_paths)]
        c = load_clip(p, dev)
        item = prepare_item(model, c, dev, args.K, args.M, step + 17, args.mover_thresh)
        hw = (int(c["H"]), int(c["W"]))
        loss, pos, vel, lang, out = forward_loss(model, item, hw, amp, args, step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            raise RuntimeError(f"non-finite grad at step {step}: {gnorm}")
        opt.step()
        corr, ratio, fp, _fg = metrics(out["ctrl"].detach().float(), item[3].float(), item[2].float())
        print(f"[train] s{step} loss{loss.item():.5f} pos{pos.item():.5f} vel{vel.item():.5f} "
              f"lang{lang.item():.5f} "
              f"corr{corr:.3f} ratio{ratio:.2f} frac{fp:.3f} gnorm{float(gnorm):.3f}", flush=True)
        if (step + 1) % int(args.eval_every) == 0:
            evaluate(model, train_paths, dev, args, amp, f"train_s{step+1}")
            if held_paths:
                evaluate(model, held_paths, dev, args, amp, f"held_s{step+1}")
            if heldtask_paths:
                evaluate(model, heldtask_paths, dev, args, amp, f"heldtask_s{step+1}")
            if args.ckpt_out and int(args.save_every) > 0 and (step + 1) % int(args.save_every) == 0:
                save_checkpoint(checkpoint_path(args.ckpt_out, step + 1), model, cfg, args, step + 1)
    if args.ckpt_out:
        save_checkpoint(args.ckpt_out, model, cfg, args, args.steps)
    print(f"[stage-b-small] done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
