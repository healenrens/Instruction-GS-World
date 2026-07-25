"""Tiny Stage-B DynamicGS training gate.

Runs a few optimization steps on one multiview fixed-ID clip to verify the new
data contract and multiview grounding path are trainable. This is not the full
production trainer.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.dynamics.model import DynamicsConfig  # noqa: E402
from igsw.gaussians import GaussianSet  # noqa: E402
from igsw.grounding.multiview_runtime import build_multiview_inputs  # noqa: E402
from igsw.model_full import InstructGSWorldModel  # noqa: E402
from igsw.training.losses import trajectory_loss  # noqa: E402


def sample_controls(disp_all, m_count, gen, mover_thresh):
    n = int(disp_all.shape[0])
    m_count = min(int(m_count), n)
    movers = (disp_all > mover_thresh).nonzero(as_tuple=True)[0]
    static = (disp_all <= mover_thresh).nonzero(as_tuple=True)[0]
    n_mover = min(int(movers.numel()), m_count // 2)
    n_static = min(m_count - n_mover, int(static.numel()))
    n_mover = min(m_count - n_static, int(movers.numel()))
    sel_m = movers[torch.randperm(movers.numel(), device=disp_all.device, generator=gen)[:n_mover]]
    sel_s = static[torch.randperm(static.numel(), device=disp_all.device, generator=gen)[:n_static]]
    idx = torch.cat([sel_m, sel_s])
    return idx[torch.randperm(idx.numel(), device=disp_all.device, generator=gen)]


def corr_and_ratio(pred, gt, init):
    radius = (init - init.mean(0)).norm(dim=-1).quantile(0.9).clamp_min(1e-6)
    gt_disp = (gt[-1] - init).norm(dim=-1) / radius
    pred_disp = (pred[-1] - init).norm(dim=-1) / radius
    corr = torch.corrcoef(torch.stack([gt_disp.float(), pred_disp.float()]))[0, 1]
    top = gt_disp.topk(max(1, gt_disp.numel() // 20)).indices
    ratio = pred_disp[top].mean() / gt_disp[top].mean().clamp_min(1e-6)
    return corr.item(), ratio.item(), (pred_disp > 0.02).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mover_thresh", type=float, default=0.01)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        raise RuntimeError("Stage-B gate needs CUDA because Qwen/gs dynamics are GPU-sized")
    clip = torch.load(args.clip, map_location="cpu", weights_only=False)
    if clip.get("contract_version") != "dynamic_gs_stage_b_v1":
        raise ValueError(f"not a Stage-B clip: {clip.get('contract_version')}")

    g0 = GaussianSet(
        clip["means"].to(dev), clip["quats"].to(dev), clip["scales"].to(dev),
        clip["opacities"].to(dev), clip["colors"].to(dev), None).to(dev)
    traj_full = clip["traj"].to(dev)
    k_steps = min(int(args.K), int(clip["Kf"]))
    gen = torch.Generator(device=dev).manual_seed(7)
    disp_all = (traj_full[k_steps] - traj_full[0]).norm(dim=-1)
    ctrl_idx = sample_controls(disp_all, args.M, gen, args.mover_thresh)
    init = g0.means[ctrl_idx]
    gt_traj = traj_full[1:k_steps + 1, ctrl_idx]
    vis = torch.ones(k_steps, ctrl_idx.numel(), dtype=torch.bool, device=dev)
    control_uv_by_view = clip["uv_by_view"].to(dev)[ctrl_idx]
    control_uv_valid_by_view = clip["uv_valid_by_view"].to(dev)[ctrl_idx]
    valid_mean = control_uv_valid_by_view.float().sum(1).mean().item()
    zero_valid = (control_uv_valid_by_view.sum(1) == 0).float().mean().item()

    cfg = DynamicsConfig(d_model=args.dim, n_layers=28, n_heads=args.heads,
                         lang_dim=2048, use_checkpoint=True, checkpoint_every=1)
    model = InstructGSWorldModel(cfg, n_control=args.M, n_query=16, spatial_ground=True).to(dev)
    model.train()
    enc = model.encoder
    vi, vi_by_view = build_multiview_inputs(
        enc, clip["instruction"], clip["scan_rgb"], int(clip.get("policy_view", 0)),
        dev, torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    print(f"[gate] clip={args.clip} N={len(g0)} M={ctrl_idx.numel()} K={k_steps} "
          f"valid_views_mean={valid_mean:.2f} zero_valid={zero_valid:.3f}", flush=True)

    t0 = time.time()
    for step in range(int(args.steps)):
        opt.zero_grad(set_to_none=True)
        with amp:
            out = model(
                vi, g0, k_steps, ctrl_idx=ctrl_idx, control_uv_hw=(int(clip["H"]), int(clip["W"])),
                vlm_inputs_by_view=vi_by_view, control_uv_by_view=control_uv_by_view,
                control_uv_valid_by_view=control_uv_valid_by_view)
            pos_l, vel_l = trajectory_loss(out["ctrl"].float(), gt_traj.float(), vis, init.float())
            loss = pos_l + vel_l
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            raise RuntimeError(f"non-finite grad norm at step {step}: {gnorm}")
        opt.step()
        corr, ratio, frac = corr_and_ratio(out["ctrl"].detach().float(), gt_traj.float(), init.float())
        print(f"[gate] s{step} loss{loss.item():.5f} pos{pos_l.item():.5f} "
              f"vel{vel_l.item():.5f} corr{corr:.3f} ratio{ratio:.2f} "
              f"frac{frac:.3f} gnorm{float(gnorm):.3f}", flush=True)
    print(f"[gate] done {args.steps} steps in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
