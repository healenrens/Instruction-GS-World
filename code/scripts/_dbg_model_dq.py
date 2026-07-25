"""Debug #1b: does the MODEL over-predict Δqpos magnitude? Run vla_040000 on TRAIN clips (where GT dq is
known) and compare predicted |dq| vs GT |dq|. If pred ~5x GT -> confirms the slow-flailing is the model
(underfit), not the data. Run on a free GPU."""
import os, sys, glob, torch, numpy as np
from types import SimpleNamespace
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_vla import make_model, build_batch

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/vla_50k_v2/vla_040000.pt"
DEV = "cuda"
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
args = SimpleNamespace(geom_mode="xyz", fdim=128, feat_source="qwen", dino_imgsize=518, traj_pred=0,
                       img_loss=1, w_depth=0.5, cam_cond=0, L=512, beta=30.0, init_from="",
                       norm_stats="data/rt2_act/norm_stats.pt", action_dim=14, action_steps=50, d_act=704,
                       n_heads_act=11, n_state_tokens=1, mlp_ratio=4.0, w_flow=1.0, w_act=1.0, placement="entropy")
TASKS = sys.argv[2].split(",") if len(sys.argv) > 2 else None
if TASKS:
    probe = sorted(sum([sorted(glob.glob(f"data/rt2_joint/{t}_*_train.pt"))[:5] for t in TASKS], []))
else:
    probe = sorted(glob.glob("data/rt2_joint/*_train.pt"))
Kf = int(torch.load(probe[0], map_location="cpu", weights_only=False)["Kf"])
model = make_model(args, DEV, Kf=Kf)
sd = torch.load(CKPT, map_location=DEV, weights_only=False)["model"]
model.load_state_dict(sd, strict=False); model.eval()
enc = model.encoder
print(f"loaded {CKPT}, Kf={Kf}\n")

pa, ga, cos = [], [], []
print(f"{'clip':<26}{'pred|dq|p95':>12}{'GT|dq|p95':>11}{'pred/GT':>9}{'dir-cos':>9}")
for f in (probe if TASKS else probe[:10]):
    c = torch.load(f, map_location="cpu", weights_only=False)
    try:
        b = build_batch(c, DEV, args, enc)
    except Exception as e:
        print(f"{os.path.basename(f)[:24]:<26} skip ({type(e).__name__})"); continue
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        dq_pred, _ = model.rollout_predict(b)
    dp = dq_pred.float().cpu().numpy()[:, ARM]                          # [A,12]
    gt = c["dq"].numpy()[:, ARM]
    p95p, p95g = np.percentile(np.abs(dp), 95), np.percentile(np.abs(gt), 95)
    # per-step direction cosine (arm), averaged over the chunk
    dn = dp / (np.linalg.norm(dp, axis=1, keepdims=True) + 1e-8)
    gn = gt / (np.linalg.norm(gt, axis=1, keepdims=True) + 1e-8)
    dc = float((dn * gn).sum(1).mean())
    pa.append(p95p); ga.append(p95g); cos.append(dc)
    print(f"{os.path.basename(f)[:24]:<26}{p95p:>12.4f}{p95g:>11.4f}{p95p/(p95g+1e-9):>9.2f}{dc:>9.3f}")

print(f"\nMEAN pred|dq|p95={np.mean(pa):.4f}  GT|dq|p95={np.mean(ga):.4f}  "
      f"ratio={np.mean(pa)/(np.mean(ga)+1e-9):.2f}x  dir-cos={np.mean(cos):.3f}")
print("ratio>>1 => model over-predicts magnitude (slow TOPP flailing). dir-cos low => also wrong direction.")
