"""Smoke-test multi-image conditioning plumbing (wrist camera): encode a clip with 1 image vs 3 images and
check the per-token grid stays the HEAD (image #0) — so 3D/geom is unchanged while wrist enriches context."""
import os, sys, glob, numpy as np, torch
from types import SimpleNamespace
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_vla import make_model, mv_in

DEV = "cuda"
args = SimpleNamespace(geom_mode="xyz", fdim=128, feat_source="qwen", dino_imgsize=518, traj_pred=0,
                       img_loss=1, w_depth=0.5, cam_cond=0, L=512, beta=30.0, init_from="",
                       norm_stats="data/rt2_act/norm_stats.pt", action_dim=14, action_steps=50, d_act=704,
                       n_heads_act=11, n_state_tokens=1, mlp_ratio=4.0, w_flow=1.0, w_act=1.0, placement="entropy")
probe = sorted(glob.glob("data/rt2_joint/adjust_bottle_*_train.pt"))
Kf = int(torch.load(probe[0], map_location="cpu", weights_only=False)["Kf"])
model = make_model(args, DEV, Kf=Kf)
sd = torch.load("checkpoints/vla_50k_v2/vla_040000.pt", map_location=DEV, weights_only=False)["model"]
model.load_state_dict(sd, strict=False); model.eval()
enc = model.encoder
c = torch.load(probe[0], map_location="cpu", weights_only=False)
head = c["gt_rgb"][0].numpy().astype(np.uint8)
instr = c.get("instruction", "do the task")
print(f"head rgb {head.shape}, instr={instr!r}\n")


def encode(images):
    vlm = mv_in(enc.build_inputs(instr, images), DEV)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ctx, ctxm, cond, grids = model.encode_cond_batch([vlm])
    g, ghw = grids[0]
    nimg = int((vlm["input_ids"][0] == enc.image_token_id).sum())
    return ctx, cond, g, ghw, nimg


ctx1, cond1, g1, ghw1, n1 = encode(head)
ctx3, cond3, g3, ghw3, n3 = encode([head, head, head])
black = np.zeros_like(head)
ctxb, condb, gb, ghwb, nb = encode([head, black, black])
print(f"1-img : grid {tuple(g1.shape)} ghw={ghw1} ctx={tuple(ctx1.shape)} img_tokens={n1}")
print(f"3-img : grid {tuple(g3.shape)} ghw={ghw3} ctx={tuple(ctx3.shape)} img_tokens={n3}")
print(f"hbb   : grid {tuple(gb.shape)} ghw={ghwb} img_tokens={nb}")
cos = torch.nn.functional.cosine_similarity(g1.flatten().float(), gb.flatten().float(), dim=0).item()
print(f"\nCHECKS:")
print(f"  grid shape stays head (1 vs 3): {tuple(g1.shape)==tuple(g3.shape)}  ghw {ghw1}=={ghw3}: {ghw1==ghw3}")
print(f"  ctx shape stable (Q queries):   {tuple(ctx1.shape)==tuple(ctx3.shape)}")
print(f"  img_tokens scale ~3x:           {n1} -> {n3} ({n3/max(n1,1):.1f}x)")
print(f"  grids finite:                   {torch.isfinite(g3).all().item()} {torch.isfinite(gb).all().item()}")
print(f"  [head,black,black] grid vs head-only cos = {cos:.3f}  (HIGH => head-select extracts the head)")
print("\nPASS" if (tuple(g1.shape)==tuple(g3.shape) and cos > 0.9) else "\nCHECK FAILED")
