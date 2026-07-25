"""FSDP correctness round-trip (run under torchrun --nproc_per_node=4).
Verifies, on the REAL model (2B frozen + 1.6B trunk + 609M expert), that:
  (1) FSDP SHARD_GRAD_OP wraps; encoder is ignored (frozen, replicated, on-device);
  (2) optimizer states + grads are SHARDED (flat-param numel per rank ~ total/world);
  (3) a few real fwd+bwd+clip+opt.step run with FINITE loss (geom + action flow), incl. no_sync accum;
  (4) FULL_STATE_DICT save -> reload into a FRESH non-FSDP make_model (strict=False) round-trips
      (only encoder.* missing), and the reloaded model produces a finite action prediction.
"""
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import train_vla as T  # noqa: E402


class A:  # minimal args
    data = "data/rt2_joint"; out = "checkpoints/_fsdp_test"; init_from = ""
    norm_stats = "data/rt2_act/norm_stats.pt"; geom_mode = "xyz"; feat_source = "qwen"
    dino_imgsize = 518; img_loss = 1; w_depth = 0.5; traj_pred = 0; cam_cond = 0
    L = 512; fdim = 128; beta = 30.0; steps = 50000; batch = 4; accum = 2
    action_dim = 14; action_steps = 50; d_act = 704; n_heads_act = 11; n_state_tokens = 1
    mlp_ratio = 4.0; w_flow = 1.0; w_act = 1.0; lr_peak = 5e-5; lr_floor = 1e-5
    warmup_steps = 1500; fsdp = 1; fsdp_reduce = "fp32"


def main():
    import torch.distributed as dist
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local); dev = f"cuda:{local}"
    is_main = rank == 0
    args = A()
    if is_main:
        os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    probe = sorted(glob.glob(f"{args.data}/*_train.pt"))
    Kf = int(torch.load(probe[0], map_location="cpu", weights_only=False)["Kf"])
    model = T.make_model(args, dev, Kf=Kf)
    model._ddp_touch = False
    enc = model.encoder

    # encoder frozen + on-device BEFORE wrap
    enc_frozen = all(not p.requires_grad for p in enc.parameters())
    enc_dev = next(enc.parameters()).device
    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    mdl = T.build_fsdp(model, local, reduce_dtype=args.fsdp_reduce)
    opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad], lr=args.lr_peak, weight_decay=0.0)

    # ---- sharding check: sum FSDP flat-param shard numel across managed units, per rank ----
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    shard_numel = 0
    n_units = 0
    for m in FSDP.fsdp_modules(mdl):
        fp = getattr(m, "_flat_param", None)
        if fp is not None:
            shard_numel += fp.numel()
            n_units += 1
    # param dtype actually stored (MixedPrecision param_dtype)
    pdtypes = {p.dtype for p in mdl.parameters() if p.requires_grad}
    enc_still_on_dev = next(enc.parameters()).device.type == "cuda"
    enc_still_frozen = all(not p.requires_grad for p in enc.parameters())

    if is_main:
        print(f"[chk] encoder pre-wrap frozen={enc_frozen} dev={enc_dev}", flush=True)
        print(f"[chk] trainable total params = {trainable_total/1e9:.3f}B", flush=True)
        print(f"[chk] FSDP units (flat-params) = {n_units}; per-rank shard numel = {shard_numel/1e9:.3f}B "
              f"(total/{world} = {trainable_total/world/1e9:.3f}B expected)", flush=True)
        print(f"[chk] managed-param stored dtype = {pdtypes} (MixedPrecision param_dtype)", flush=True)
        print(f"[chk] encoder POST-wrap on_cuda={enc_still_on_dev} frozen={enc_still_frozen}", flush=True)

    # ---- a few real steps with no_sync accum ----
    pool = T._load_clips_for_test(args, dev, enc, args.batch)
    if len(pool) < args.batch:
        pool = (pool * (args.batch // max(1, len(pool)) + 1))[:args.batch]
    batch = T.build_batch_padded(pool[:args.batch], dev, args, enc)
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    losses = []
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        accum = args.accum
        for _m in range(accum):
            is_last = (_m == accum - 1)
            import contextlib
            ctx = mdl.no_sync() if (accum > 1 and not is_last) else contextlib.nullcontext()
            with ctx:
                with amp:
                    loss, logs = mdl(batch)            # FSDP.forward -> root hooks -> forward_vla_batch
                (loss / accum).backward()
        gnorm = mdl.clip_grad_norm_(1.0)
        opt.step()
        if is_main:
            print(f"[step {step}] loss={float(loss):.4f} geom={float(logs['geom']):.4f} "
                  f"flow={float(logs['flow']):.4f} vnorm={float(logs['v_norm']):.3f} "
                  f"gnorm={float(gnorm):.3f} dcos={float(logs['dcos']):.3f} finite={torch.isfinite(loss).item()}",
                  flush=True)
        losses.append(float(loss))

    # ---- save FULL_STATE_DICT (collective) then reload into a fresh non-FSDP model ----
    T.save_ckpt(mdl, model, args, 3, use_fsdp=True, is_main=is_main)
    dist.barrier()
    if is_main:
        ckpt = f"{args.out}/vla_{3:06d}.pt"
        sd = torch.load(ckpt, map_location=dev, weights_only=False)["model"]
        has_enc = any(k.startswith("encoder.") for k in sd)
        has_expert = any(k.startswith("action_expert.") for k in sd)
        has_actnorm = any(k.startswith("act_norm.") for k in sd)
        # fresh non-FSDP model
        fresh = T.make_model(args, dev, Kf=Kf)
        miss, unexp = fresh.load_state_dict(sd, strict=False)
        miss_nonenc = [m for m in miss if not m.startswith("encoder.")]
        print(f"[save] ckpt tensors={len(sd)} encoder_in_ckpt={has_enc} expert_in_ckpt={has_expert} "
              f"act_norm_in_ckpt={has_actnorm}", flush=True)
        print(f"[reload] into fresh non-FSDP make_model: non-encoder missing={len(miss_nonenc)} "
              f"unexpected={len(unexp)}", flush=True)
        if miss_nonenc:
            print(f"[reload] NON-ENCODER MISSING (BAD): {miss_nonenc[:8]}", flush=True)
        if unexp:
            print(f"[reload] UNEXPECTED (BAD): {unexp[:8]}", flush=True)
        # reloaded model produces a finite action prediction (use build_clip_single -> has 'anchor')
        fresh.eval()
        c0 = torch.load(probe[0], map_location=dev, weights_only=False)
        b0 = T.build_clip_single(c0, dev, args, enc)
        with torch.no_grad(), amp:
            pred = fresh.predict_action(b0)
        print(f"[reload] predict_action shape={tuple(pred.shape)} finite={torch.isfinite(pred).all().item()} "
              f"|pred|={pred.float().abs().mean().item():.4f}", flush=True)
        ok = (not has_enc) and has_expert and has_actnorm and len(miss_nonenc) == 0 and len(unexp) == 0 \
            and torch.isfinite(pred).all().item() and all(torch.isfinite(torch.tensor(losses)).tolist())
        print(f"\n[RESULT] {'PASS' if ok else 'FAIL'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
