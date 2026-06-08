"""Large-scale trainer (v2) — ≥1B language-conditioned Gaussian world model.

InstructGSWorldModel (Qwen3-VL+LoRA in-loop, per-layer interaction, ~1.7B dynamics)
trained per clip with SC-GS free-running rollout + render supervision. One DDP
forward/step (dynamics called K× inside it). Reuses the cached clips (g0+gt+cameras
+instruction); the VLM image is the cached frame0.

  torchrun --nproc_per_node=4 code/scripts/train.py --data ./data/clips_v1 --out ./checkpoints/run2
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
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.data.clip_dataset import ClipDataset, identity_collate  # noqa: E402
from igsw.gaussians import GaussianSet, render_gaussianset, psnr  # noqa: E402
from igsw.dynamics.model import DynamicsConfig  # noqa: E402
from igsw.model_full import InstructGSWorldModel  # noqa: E402
from igsw.training import photometric_loss, delta_reg, velocity_smoothness  # noqa: E402


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank(); world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(local)
        return True, rank, world, local
    return False, 0, 1, 0


def noise_augment(g0: GaussianSet, frac: float, gen) -> GaussianSet:
    if frac <= 0:
        return g0
    center = g0.means.mean(0, keepdim=True)
    radius = (g0.means - center).norm(dim=-1).quantile(0.9).clamp_min(1e-6)
    nm = g0.means + torch.randn(g0.means.shape, device=g0.device, generator=gen) * (frac * radius)
    nc = (g0.colors + torch.randn(g0.colors.shape, device=g0.device, generator=gen) * (0.5 * frac)).clamp(0, 1)
    return GaussianSet(nm, g0.quats.clone(), g0.scales.clone(), g0.opacities.clone(), nc, None)


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
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="/mnt/pfs/public/xuhaoming/instruct_gs_world/checkpoints/run2")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--noise_frac", type=float, default=0.02)
    ap.add_argument("--w_reg", type=float, default=1e-3)
    ap.add_argument("--w_vel", type=float, default=1e-2)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--resume", default="")
    ap.add_argument("--max_steps", type=int, default=0)
    args = ap.parse_args()

    ddp, rank, world, local = setup_ddp()
    dev = torch.device(f"cuda:{local}")
    is_main = rank == 0
    if is_main:
        os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(1234 + rank)

    ds = ClipDataset(args.data, require_lang=False)   # we re-encode with the VLM in-loop
    sampler = (DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
               if ddp else None)
    loader = DataLoader(ds, batch_size=1, sampler=sampler, shuffle=(sampler is None),
                        num_workers=3, collate_fn=identity_collate, persistent_workers=True, pin_memory=False)

    cfg = DynamicsConfig(d_model=args.dim, n_layers=args.layers, n_heads=args.heads,
                         lang_dim=2048, use_checkpoint=True)
    model = InstructGSWorldModel(cfg, n_control=args.M).to(dev)
    if is_main:
        print(f"[data] {len(ds)} clips | world={world} | steps/epoch≈{len(ds)//world}", flush=True)
        print(f"[model] {model.param_report()}", flush=True)

    enc_ref = model.encoder
    if ddp:
        # Qwen3-VL is frozen (no trainable params there) -> only the 3D modules sync.
        model = DDP(model, device_ids=[local], static_graph=True,
                    gradient_as_bucket_view=True, broadcast_buffers=False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))

    total_steps = args.max_steps or (args.epochs * (len(ds) // world))

    def lr_scale(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    writer = None
    if is_main:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(args.out, "tb"))
        except Exception:
            pass

    start_step = 0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        (model.module if ddp else model).load_state_dict(ck["model"], strict=False)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_step = ck.get("step", 0)
        if is_main:
            print(f"[resume] {args.resume} @step {start_step}", flush=True)

    gen = torch.Generator(device=dev); gen.manual_seed(42 + rank)
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    step = start_step; t0 = time.time(); model.train()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for clip in loader:
            if args.max_steps and step >= args.max_steps:
                break
            sc = lr_scale(step)
            for g in opt.param_groups:
                g["lr"] = args.lr * sc

            g0 = clip["g0"].to(dev)
            gt = clip["gt"].to(dev)
            Ks = clip["Ks"].to(dev); viewmats = clip["viewmats"].to(dev)
            H, W, K = clip["H"], clip["W"], min(args.K, clip["K"])
            frame0 = (gt[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            vlm_inputs = move_vlm_inputs(enc_ref.build_inputs(clip["instruction"], frame0), dev, torch.bfloat16)

            g0n = noise_augment(g0, args.noise_frac, gen)
            opt.zero_grad(set_to_none=True)
            with amp:
                out = model(vlm_inputs, g0n, K)
            loss = 0.0; ps = []
            for k in range(K):
                tt = k + 1
                s = GaussianSet(out["means"][k].float(), out["quats"][k].float(), out["scales"][k].float(),
                                out["opacities"][k].float(), out["colors"][k].float(), None)
                colors, _, _ = render_gaussianset(s, viewmats[tt], Ks[tt], W, H)
                pl, _, _ = photometric_loss(colors[0], gt[tt]); loss = loss + pl
                ps.append(psnr(colors[0].clamp(0, 1).detach(), gt[tt]))
            loss = loss / K
            reg = delta_reg(out["v"], out["om"], out["dls"])
            vel = velocity_smoothness([out["ctrl"][i] for i in range(out["ctrl"].shape[0])])
            total = loss + args.w_reg * reg + args.w_vel * vel
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if is_main and step % args.log_every == 0:
                mp = float(np.mean(ps)); rate = (step - start_step + 1) / (time.time() - t0 + 1e-6)
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"e{epoch} s{step} lr{opt.param_groups[0]['lr']:.2e} loss{loss.item():.4f} "
                      f"reg{reg.item():.2e} vel{vel.item():.2e} PSNR{mp:.2f} {rate:.2f}it/s peakGB{mem:.1f}", flush=True)
                if writer:
                    writer.add_scalar("loss/photometric", loss.item(), step)
                    writer.add_scalar("metric/psnr", mp, step)
            if is_main and step > 0 and step % args.ckpt_every == 0:
                ckpt = {"model": (model.module if ddp else model).state_dict(),
                        "opt": opt.state_dict(), "step": step, "cfg": cfg.__dict__}
                torch.save(ckpt, os.path.join(args.out, f"ckpt_{step:07d}.pt"))
                torch.save(ckpt, os.path.join(args.out, "ckpt_last.pt"))
                print(f"[ckpt] saved @step {step}", flush=True)
            step += 1
        if args.max_steps and step >= args.max_steps:
            break

    if is_main:
        ckpt = {"model": (model.module if ddp else model).state_dict(),
                "opt": opt.state_dict(), "step": step, "cfg": cfg.__dict__}
        torch.save(ckpt, os.path.join(args.out, "ckpt_last.pt"))
        print(f"[done] trained to step {step}", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
