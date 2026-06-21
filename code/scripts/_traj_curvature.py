"""PREMISE VALIDATION (agent.md curve-task): measure how CURVED the GT trajectories actually are.

For mover tokens (GT image-flow frame0->frameK > gt_flow_thr of image size), measure the deviation of
the intermediate trajectory points from the straight chord traj[0]->traj[Kf]:
  * 3D curvature ratio  = max perpendicular dist of intermediate 3D points to the 3D chord / chord length
  * IMAGE curvature ratio = same but in projected pixel (uv) space / image-chord length
  * IMAGE deviation in px = max perpendicular dist in pixels (absolute, normalized by max(H,W) for a %)
A near-zero deviation => trajectories are straight => curve-fitting cannot help. Aggregate over train clips.

Usage: _traj_curvature.py --data data/rtvid_multi_v2 --split train --gt_flow_thr 0.05
"""
import argparse, glob, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.gpstoken_wm.tokens import project_to_uv  # noqa: E402


def perp_dev(pts, p0, pK):
    """Max perpendicular distance of intermediate points `pts` [T,D] to the chord p0->pK [D]. Returns
    (max_perp_dist, chord_len). pts should be the intermediate points (t=1..K-1)."""
    chord = pK - p0
    clen = np.linalg.norm(chord)
    if clen < 1e-9:
        return 0.0, 0.0
    u = chord / clen
    rel = pts - p0[None]                      # [T,D]
    proj = (rel @ u)[:, None] * u[None]       # projection onto chord
    perp = rel - proj                          # perpendicular component
    d = np.linalg.norm(perp, axis=-1)          # [T]
    return float(d.max()) if len(d) else 0.0, float(clen)


def smooth_vs_noise(full_uv):
    """full_uv [Kf+1,2] = the FULL projected image trajectory of ONE token. Distinguish SMOOTH curvature
    from high-frequency jitter:
      arc_ratio = total polyline length / chord length  (1.0=straight; >1 = path is longer => curved OR wiggly)
      quad_frac = fraction of the perpendicular-deviation VARIANCE explained by a smooth per-axis quadratic
                  fit in t (high => deviation is a smooth bend a curve model can fit; low => it's jitter)
    Returns (arc_ratio, quad_frac, max_perp_px, chord_px)."""
    Kp1 = full_uv.shape[0]
    p0, pK = full_uv[0], full_uv[-1]
    chord = pK - p0
    clen = float(np.linalg.norm(chord))
    seglen = float(np.linalg.norm(np.diff(full_uv, axis=0), axis=-1).sum())
    arc_ratio = seglen / clen if clen > 1e-6 else 1.0
    if clen < 1e-6:
        return arc_ratio, 0.0, 0.0, clen
    u = chord / clen
    rel = full_uv - p0[None]
    proj = (rel @ u)[:, None] * u[None]
    perp = rel - proj                                       # [Kp1,2] perpendicular component (the deviation)
    # signed perp coord (2D so take the 2D perp magnitude with sign via cross product wrt chord dir)
    perp_signed = rel[:, 0] * (-u[1]) + rel[:, 1] * u[0]    # signed scalar deviation
    t = np.linspace(0.0, 1.0, Kp1)
    # smooth model: quadratic through endpoints (deviation=0 at t=0 and t=1) -> single param: a*t*(1-t)
    basis = (t * (1.0 - t))
    denom = float((basis * basis).sum())
    a = float((perp_signed * basis).sum() / denom) if denom > 1e-9 else 0.0
    fit = a * basis
    ss_tot = float(((perp_signed - perp_signed.mean()) ** 2).sum())
    # variance of deviation explained by the smooth single-arc model (vs leftover = jitter)
    ss_res = float(((perp_signed - fit) ** 2).sum())
    quad_frac = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    max_perp_px = float(np.linalg.norm(perp, axis=-1).max())
    return arc_ratio, quad_frac, max_perp_px, clen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--gt_flow_thr", type=float, default=0.05,
                    help="mover = GT image-flow frame0->frameK magnitude > this fraction of image size")
    ap.add_argument("--max_clips", type=int, default=0, help="0=all")
    args = ap.parse_args()
    dev = "cpu"

    clips = sorted(glob.glob(f"{args.data}/*{args.split}*.pt"))
    if args.max_clips > 0:
        clips = clips[:args.max_clips]

    # per-clip medians, then aggregate
    clip_curv3d, clip_curvimg, clip_devpx = [], [], []
    clip_curv3d_p90, clip_curvimg_p90, clip_devpx_p90 = [], [], []
    all_curvimg = []   # token-level pooled
    all_devpx = []
    n_mover_total = 0
    clip_chordpx = []
    clip_arc, clip_quad, all_arc, all_quad = [], [], [], []

    for cp in clips:
        c = torch.load(cp, map_location=dev, weights_only=False)
        traj = c["traj"].float().numpy()                       # [Kf+1, N, 3]
        Kf1, N, _ = traj.shape
        K = int(c["Kf"])
        H, W = int(c["H"]), int(c["W"])
        Ki = c["K_intr"].float(); vm = c["viewmat"].float()
        n_keep = N - int(c.get("n_fill", 0))

        # project the WHOLE trajectory to uv -> [Kf+1, N, 2]
        uv_traj = np.zeros((Kf1, N, 2), dtype=np.float32)
        for t in range(Kf1):
            uv_traj[t] = project_to_uv(torch.from_numpy(traj[t]), Ki, vm).numpy()

        maxhw = float(max(H, W))
        # mover def: image flow frame0->frameK as fraction of image
        img_flow = np.linalg.norm((uv_traj[K] - uv_traj[0]), axis=-1) / maxhw   # [N]
        mover = np.zeros(N, dtype=bool)
        mover[:n_keep] = img_flow[:n_keep] > args.gt_flow_thr
        idx = np.where(mover)[0]
        if len(idx) < 5:
            continue
        n_mover_total += len(idx)

        c3, cimg, dpx = [], [], []
        arcs, quads = [], []
        for n in idx:
            # 3D curvature
            p0, pK = traj[0, n], traj[K, n]
            inter3d = traj[1:K, n]                              # t=1..K-1
            d3, clen3 = perp_dev(inter3d, p0, pK)
            if clen3 > 1e-9:
                c3.append(d3 / clen3)
            # image curvature
            u0, uK = uv_traj[0, n], uv_traj[K, n]
            interimg = uv_traj[1:K, n]
            dimg, clenimg = perp_dev(interimg, u0, uK)
            if clenimg > 1e-6:
                cimg.append(dimg / clenimg)
            dpx.append(dimg)                                    # absolute px deviation
            # smooth-vs-noise on the FULL image trajectory
            ar, qf, _, _ = smooth_vs_noise(uv_traj[:K + 1, n])
            arcs.append(ar); quads.append(qf)
        if not cimg:
            continue
        clip_arc.append(np.median(arcs)); clip_quad.append(np.median(quads))
        all_arc.extend(arcs); all_quad.extend(quads)
        clip_curv3d.append(np.median(c3) if c3 else 0.0)
        clip_curvimg.append(np.median(cimg))
        clip_devpx.append(np.median(dpx))
        clip_curv3d_p90.append(np.percentile(c3, 90) if c3 else 0.0)
        clip_curvimg_p90.append(np.percentile(cimg, 90))
        clip_devpx_p90.append(np.percentile(dpx, 90))
        clip_chordpx.append(np.median(np.linalg.norm(uv_traj[K, idx] - uv_traj[0, idx], axis=-1)))
        all_curvimg.extend(cimg)
        all_devpx.extend(dpx)

    def md(x): return float(np.median(x)) if len(x) else float("nan")
    def mn(x): return float(np.mean(x)) if len(x) else float("nan")

    print(f"=== GT TRAJECTORY CURVATURE [{args.data} split={args.split}] ===")
    print(f"clips used (>=5 movers): {len(clip_curvimg)} / {len(clips)}   total mover tokens: {n_mover_total}")
    print(f"mover def: GT image-flow frame0->frameK > {args.gt_flow_thr*100:.0f}% of image (max(H,W))")
    print()
    print("CURVATURE RATIO = max perpendicular deviation of intermediate points / chord length")
    print("  (0.0 = perfectly straight; 0.1 = bulges 10% of chord length sideways)")
    print(f"  3D    curv-ratio:  median-of-clip-medians {md(clip_curv3d):.3f}   mean {mn(clip_curv3d):.3f}   "
          f"clip-median p90 {md(clip_curv3d_p90):.3f}")
    print(f"  IMAGE curv-ratio:  median-of-clip-medians {md(clip_curvimg):.3f}   mean {mn(clip_curvimg):.3f}   "
          f"clip-median p90 {md(clip_curvimg_p90):.3f}")
    print(f"  token-level IMAGE curv-ratio (pooled): median {md(all_curvimg):.3f}  p75 "
          f"{float(np.percentile(all_curvimg,75)):.3f}  p90 {float(np.percentile(all_curvimg,90)):.3f}  "
          f"p99 {float(np.percentile(all_curvimg,99)):.3f}")
    print()
    print("ABSOLUTE IMAGE DEVIATION (max perpendicular dist in pixels)")
    print(f"  median-of-clip-medians {md(clip_devpx):.2f} px   (p90 across clips {md(clip_devpx_p90):.2f} px)")
    print(f"  token-level pooled: median {md(all_devpx):.2f} px   p90 {float(np.percentile(all_devpx,90)):.2f} px")
    print(f"  for reference: median mover image CHORD length = {md(clip_chordpx):.1f} px   (image ~{W}x{H})")
    print()
    print("SMOOTH vs NOISE (is the deviation a fittable smooth bend, or unfittable jitter?)")
    print("  arc-ratio = polyline length / chord (1.0=straight; high can mean curved OR wiggly)")
    print(f"    median-of-clip-medians {md(clip_arc):.3f}   token-pooled median {md(all_arc):.3f}  "
          f"p90 {float(np.percentile(all_arc,90)):.3f}")
    print("  quad-frac = variance of deviation explained by a SINGLE smooth arc a*t(1-t) (1.0=pure smooth bend; ~0=jitter)")
    print(f"    median-of-clip-medians {md(clip_quad):.3f}   token-pooled median {md(all_quad):.3f}  "
          f"p25 {float(np.percentile(all_quad,25)):.3f}  p75 {float(np.percentile(all_quad,75)):.3f}")
    print()
    # verdict heuristic
    cr = md(clip_curvimg); qf = md(all_quad)
    print("VERDICT HEURISTIC:")
    if cr < 0.03:
        print(f"  IMAGE curv-ratio {cr:.3f} < 0.03 => trajectories ESSENTIALLY STRAIGHT. Curve-fitting CANNOT help.")
    elif qf < 0.4:
        print(f"  IMAGE curv-ratio {cr:.3f} but quad-frac {qf:.3f} < 0.4 => deviation is mostly JITTER, not a "
              f"smooth fittable bend. Curve-fitting likely fits noise (HARMFUL/useless).")
    elif cr < 0.07:
        print(f"  IMAGE curv-ratio {cr:.3f} in [0.03,0.07), quad-frac {qf:.3f} => MILD smooth curvature. MARGINAL.")
    else:
        print(f"  IMAGE curv-ratio {cr:.3f}>=0.07, quad-frac {qf:.3f}>=0.4 => MEANINGFUL SMOOTH curvature. WORTH testing.")


if __name__ == "__main__":
    main()
