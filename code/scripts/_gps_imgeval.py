"""Image-space held eval (for the --img_loss idea): does the model predict the NORMALIZED 2D image
displacement (Δu/W, Δv/H) with the RIGHT MAGNITUDE? Compares any ckpt's mover image-flow vs GT:
image dir-cos + image-flow magnitude ratio (median |pred|/|GT|). The decisive test of whether
supervising in normalized image space cures the 3D magnitude under-prediction (3D magR ~0.5).
Usage: _gps_imgeval.py --ckpt checkpoints/gpswm_img/wm_000750.pt --data data/trans_v1 --split held
"""
import argparse, glob, os, sys
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.gpstoken_wm import GPSTokenWM, place_tokens, sample_grid_feat  # noqa: E402
from igsw.gpstoken_wm.tokens import project_to_uv, to_cam  # noqa: E402
from igsw.gaussians.gpstoken import mover_saliency  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="held"); ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--beta", type=float, default=30.0)
    ap.add_argument("--mov_pct", type=float, default=0.0, help="if >0, define movers as the top fraction by displacement (gauge-agnostic) instead of the abs disp>0.01 (sim-meters) threshold")
    args = ap.parse_args(); dev = "cuda"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False); cargs = ck.get("args", {})
    model = GPSTokenWM(geom_mode=cargs.get("geom_mode", "xyz"), fdim=cargs.get("fdim", 128),
                       feat_source=cargs.get("feat_source", "qwen"), dino_imgsize=cargs.get("dino_imgsize", 518)).to(dev)
    model.load_state_dict(ck["model"], strict=False); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    enc = model.encoder

    def mv_in(inp):
        return {k: (v.to(dev, dtype=torch.bfloat16) if (torch.is_tensor(v) and v.is_floating_point())
                    else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in inp.items()}

    clips = sorted(glob.glob(f"{args.data}/*{args.split}*.pt"))
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    rows = []
    for cp in clips:
        c = torch.load(cp, map_location=dev, weights_only=False)
        means = c["means"].to(dev).float(); uv = c["uv"].to(dev).float(); traj = c["traj"].to(dev).float()
        N = means.shape[0]; H, W = int(c["H"]), int(c["W"]); K = int(c["Kf"])
        Ki = c["K_intr"].to(dev).float(); vm = c["viewmat"].to(dev).float()
        instr = c.get("instruction", ""); n_keep = N - int(c.get("n_fill", 0))
        disp = (traj[K] - traj[0]).norm(dim=-1)
        sal = mover_saliency(uv, disp, n_keep, H, W) if args.beta > 0 else None
        rgb0 = c["gt_rgb"][0].cpu().numpy().astype(np.uint8)
        cen, sig, idx = place_tokens(rgb0, uv, n_keep, args.L, dev, sal=sal, beta=args.beta)
        tok_xyz0 = means[idx]; xyz1_gt = traj[K][idx]; sig_n = (sig / float(max(H, W))).clamp(0, 1)
        center = means[:n_keep].mean(0, keepdim=True); radius = (means[:n_keep] - center).norm(dim=-1).amax().clamp_min(1e-6)
        vlm0 = mv_in(enc.build_inputs(instr, rgb0))
        with torch.no_grad(), amp:
            ctxc, ctxm, cond, text_feats = model.encode_cond(vlm0)
            grid0, ghw0 = (model.dino.grid(rgb0) if model.dino is not None else enc.image_grid_features(vlm0))
            tok_feat = model.feat_in(sample_grid_feat(grid0, ghw0, cen, H, W)).float()
            cam_tok = None
            if cargs.get("cam_cond", False):                                 # B: inject camera pose
                cg, cam_tok = model.cam_cond_signals(tok_xyz0.float(), center, radius, Ki.float(), vm.float())
                cond = cond + cg
            x = model.predict(tok_xyz0, tok_feat, sig_n, center, radius, ctxc, ctxm, cond, cam_tok=cam_tok)
            if cargs.get("fuse", False):                                     # v2: grounding-gate modulates motion
                gate = torch.sigmoid(model.relevance(tok_feat, text_feats.mean(0)))
                g = gate[:, None] * model.geom_head(x[0]).float()
                xyz1_pred = model.geom_to_xyz(g, tok_xyz0.float(), Ki.float(), vm.float())
            else:
                xyz1_pred, _ = model.heads(x, tok_xyz0, Ki, vm)
        xyz1_pred = xyz1_pred.float()
        Wn = xyz1_pred.new_tensor([float(W), float(H)])
        uv0 = project_to_uv(tok_xyz0, Ki, vm); uv1p = project_to_uv(xyz1_pred, Ki, vm); uv1g = project_to_uv(xyz1_gt, Ki, vm)
        fp = (uv1p - uv0) / Wn; fg = (uv1g - uv0) / Wn                       # normalized image flow
        di = disp[idx]
        if args.mov_pct > 0:                                                 # relative mover threshold (gauge-agnostic)
            thr = float(di.quantile(1.0 - args.mov_pct)) if (di > 1e-6).any() else 1e9
            mv = di > max(thr, 1e-6)
        else:
            mv = di > 0.01
        if mv.sum() < 5:
            continue
        dcos = float(F.cosine_similarity(fp[mv], fg[mv], dim=-1).mean())
        d3p = xyz1_pred - tok_xyz0; d3g = xyz1_gt - tok_xyz0                  # 3D motion vectors
        dcos3 = float(F.cosine_similarity(d3p[mv], d3g[mv], dim=-1).mean())   # 3D dir-cos (vs image dcos)
        magr3 = float(d3p[mv].norm(dim=-1).median() / d3g[mv].norm(dim=-1).median().clamp_min(1e-9))
        magr = float(fp[mv].norm(dim=-1).median() / fg[mv].norm(dim=-1).median().clamp_min(1e-9))
        gpx = float(fg[mv].norm(dim=-1).median()) * 100                      # GT image flow as % of image size
        ppx = float(fp[mv].norm(dim=-1).median()) * 100
        # depth-change (Δlog z): magnitude ratio + sign agreement (is depth motion learned?)
        z0 = to_cam(tok_xyz0, vm)[:, 2].clamp_min(1e-3)
        ldp = torch.log(to_cam(xyz1_pred, vm)[:, 2].clamp_min(1e-3) / z0)
        ldg = torch.log(to_cam(xyz1_gt, vm)[:, 2].clamp_min(1e-3) / z0)
        dmagr = float(ldp[mv].abs().median() / ldg[mv].abs().median().clamp_min(1e-9))
        dsign = float((torch.sign(ldp[mv]) == torch.sign(ldg[mv])).float().mean())
        rows.append((dcos, magr, gpx, ppx, dmagr, dsign, dcos3, magr3))
    A = np.array(rows)
    def md(x): return float(np.median(x))
    print(f"[{os.path.basename(args.ckpt)} on {args.split}]  n_clip={len(A)}  IMAGE-space:")
    print(f"  3D    dir-cos: {md(A[:,6]):+.2f}   3D    mag-ratio: {md(A[:,7]):.2f}   <== TRUE 3D motion direction/scale")
    print(f"  image dir-cos: {md(A[:,0]):+.2f}")
    print(f"  image-flow MAG-RATIO (pred/GT): {md(A[:,1]):.2f}   <== ~1.0 = magnitude SOLVED (vs 3D magR ~0.5)")
    print(f"  GT image flow {md(A[:,2]):.1f}% of img  vs PRED {md(A[:,3]):.1f}%")
    print(f"  DEPTH Δlogz mag-ratio: {md(A[:,4]):.2f}   sign-agreement: {md(A[:,5])*100:.0f}%   <== is depth motion learned?")


if __name__ == "__main__":
    main()
