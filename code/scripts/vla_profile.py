"""Per-phase timing harness for the VLA training step (single GPU). Pinpoints WHERE the ~11s/microbatch
goes and whether the GPU is compute-bound or launch/serialization-bound. Reuses the REAL build + forward.

  PROF_B=24 .venv/bin/python code/scripts/vla_profile.py
"""
import os, sys, time, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))            # code/scripts
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # code
import torch
from types import SimpleNamespace
from train_vla import make_model, build_clip_single, build_batch_padded
from igsw.gpstoken_wm.tokens import sample_grid_feat


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tm(fn, n=6, warmup=2):
    for _ in range(warmup):
        fn()
    sync(); t = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - t) / n


def main():
    dev = "cuda"
    args = SimpleNamespace(geom_mode="xyz", fdim=128, feat_source="qwen", dino_imgsize=518,
                           traj_pred=0, img_loss=1, w_depth=0.5, cam_cond=0, L=512, beta=30.0,
                           init_from="", norm_stats="data/rt2_act/norm_stats.pt", action_dim=14,
                           action_steps=50, d_act=704, n_heads_act=11, n_state_tokens=1,
                           mlp_ratio=4.0, w_flow=1.0, w_act=1.0)
    B = int(os.environ.get("PROF_B", "8"))   # single GPU, no sharding -> keep B small (timing is per-clip)
    probe = sorted(glob.glob("data/rt2_joint/*_train.pt"))
    Kf = int(torch.load(probe[0], map_location="cpu", weights_only=False)["Kf"])
    print(f"[prof] building model (frozen 2B encoder + 1.6B trunk + action expert)...", flush=True)
    model = make_model(args, dev, Kf=Kf); model.train()
    enc = model.encoder

    # ---- load B clips through the REAL per-clip prep ----
    t = time.perf_counter(); singles = []
    for cp in probe:
        try:
            c = torch.load(cp, map_location=dev, weights_only=False)
            singles.append(build_clip_single(c, dev, args, enc))
        except Exception:
            pass
        if len(singles) >= B:
            break
    sync(); t_load = time.perf_counter() - t
    B = len(singles)
    print(f"[prof] loaded {B} clips (torch.load + build_clip_single) in {t_load:.2f}s "
          f"= {t_load/B*1000:.0f}ms/clip\n", flush=True)

    # ---- phase timers (forward-only phases under no_grad so no graph is retained) ----
    def grid_only():
        for s in singles:
            enc.image_grid_features(s["vlm0"])

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):   # MATCH real training (bf16 autocast)
        t_pad = tm(lambda: build_batch_padded(singles, dev, args, enc))
        b = build_batch_padded(singles, dev, args, enc)
        t_grid = tm(grid_only)
        t_enc = tm(lambda: model.encode_cond_batch(b["vlm_list"]))
        t_enc1 = tm(lambda: model.encode_cond(singles[0]["vlm0"]), n=8)
        ctx, ctxm, cond, grids = model.encode_cond_batch(b["vlm_list"])
        # CORRECTNESS: the unified grid (from encode_cond_batch's forward) must equal the old
        # image_grid_features (a separate 2nd forward we eliminated).
        g_new, ghw_new = grids[0]
        g_old, ghw_old = enc.image_grid_features(b["vlm_list"][0])
        gdiff = (g_new.float() - g_old.float()).abs().max().item() if (g_new is not None and g_old is not None) else -1.0
        print(f"  [verify] batched-grid vs image_grid_features: max|diff|={gdiff:.2e}  ghw {ghw_new} vs {ghw_old}")
        # CORRECTNESS: the BATCHED encode_cond_batch must match the per-clip reference (encode_cond_batch_seq).
        ctx_s, _, cond_s, grids_s = model.encode_cond_batch_seq(b["vlm_list"])
        cdiff = (ctx.float() - ctx_s.float()).abs().max().item()
        conddiff = (cond.float() - cond_s.float()).abs().max().item()
        gpairs = [(gb[0], gs[0]) for gb, gs in zip(grids, grids_s) if gb[0] is not None and gs[0] is not None]
        gbdiff = max((a.float() - b2.float()).abs().max().item() for a, b2 in gpairs) if gpairs else -1.0
        print(f"  [verify] BATCHED vs per-clip-seq: ctx max|diff|={cdiff:.2e}  cond={conddiff:.2e}  grid={gbdiff:.2e}")
        # B=1 sanity: batched-of-ONE vs per-clip. If this matches (~bf16), the collation/extraction LOGIC is
        # correct and the B=8 diff above is just batch-size attention-kernel noise (fine for training).
        ctx1, _, cond1, grids1 = model.encode_cond_batch([b["vlm_list"][0]])
        ctxr, _, condr, gridr = model.encode_cond_batch_seq([b["vlm_list"][0]])
        g1d = (grids1[0][0].float() - gridr[0][0].float()).abs().max().item()
        gmag = gridr[0][0].float().abs().max().item()
        print(f"  [verify B=1] ctx={ (ctx1.float()-ctxr.float()).abs().max().item():.2e}  "
              f"grid={g1d:.2e} (grid|max|={gmag:.1f} -> rel {g1d/max(gmag,1e-6)*100:.1f}%)")
        feats = []
        for i in range(B):
            g0, ghw0 = grids[i]
            feats.append(model.feat_in(sample_grid_feat(g0, ghw0, b["cen"][i], b["H_list"][i], b["W_list"][i])))
        tok_feat = torch.stack(feats, 0).float()
        t_trunk = tm(lambda: model.predict_batch(b["tok_xyz0"], tok_feat, b["sig_n"], b["center"],
                                                 b["radius"], ctx, ctxm, cond, b["tok_mask"]))
        t_fwd = tm(lambda: model.forward_vla_batch(b))
    del ctx, tok_feat, feats
    torch.cuda.empty_cache()

    def fwd_bwd():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model.forward_vla_batch(b)
        loss.backward()
        model.zero_grad(set_to_none=True)
    t_step = tm(fwd_bwd, n=4, warmup=2)
    t_bwd = max(0.0, t_step - t_fwd)

    step = t_pad + t_fwd + t_bwd
    print(f"===== PER-PHASE  (B={B}, seconds, cuda-synced) =====")
    print(f"  build_batch_padded (consumer side)   : {t_pad:6.3f}   ({t_pad/step*100:4.0f}% of step)")
    print(f"     of which image_grid_features ×B   : {t_grid:6.3f}   (vision tower, {B}x sequential)")
    print(f"  encode_cond_batch (Qwen×B + aggr)    : {t_enc:6.3f}   (1-clip={t_enc1:.3f}  ideal-batched-floor≈{t_enc1:.3f}, seq-sum={t_enc1*B:.3f})")
    print(f"  predict_batch (DiT trunk, batched)   : {t_trunk:6.3f}")
    print(f"  forward_vla_batch (FULL fwd)         : {t_fwd:6.3f}   (= encode {t_enc:.2f} + trunk {t_trunk:.2f} + featloop+loss {max(0,t_fwd-t_enc-t_trunk):.2f})")
    print(f"  backward                             : {t_bwd:6.3f}")
    print(f"  ----------------------------------------------------")
    print(f"  STEP (pad+fwd+bwd)                   : {step:6.3f}   (~{1.0/step:.3f} microbatch/s, ~{step*3:.1f}s/opt-step @accum3)")
    print(f"\n  read: if encode_cond_batch ≈ seq-sum ({t_enc1*B:.2f}s) >> ideal-batched ({t_enc1:.2f}s),")
    print(f"        the B× sequential Qwen forwards are the cost and batching collapses them to ~1×.")


if __name__ == "__main__":
    main()
