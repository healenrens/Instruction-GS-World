"""Merge SpaTracker flow clips with rt2_act action entries -> compact frame0-only JOINT samples.

For each flow clip in --flow_dir (produced by robotwin_spatrack_clip.py over the full plan; windows dropped
by --min_movers simply have no flow clip), find the SAME-named rt2_act entry and emit a compact joint .pt to
--out:
  means[N,3], uv[N,2], traj[13,N,3], K_intr[3,3], viewmat=I[4,4], H, W, Kf, instruction,
  gt_rgb=[1,H,W,3] (FRAME0 ONLY), vis[13,N],         <- flow GT (JEPA removed -> drop the 12 extra frames)
  dq[50,14], anchor[14], control_hz                  <- action chunk, pulled from the SAME window's rt2_act

Filenames are identical across flow_dir / rt2_act / out: "{name}_{split}.pt" -> exact window alignment.
The JOINT set = windows that have BOTH a flow clip AND an action entry.

Usage: rt2_joint_merge.py --flow_dir data/rt2_joint_flow --act_dir data/rt2_act --out data/rt2_joint \
         [--shard 0 --nshard 4]
"""
import argparse, glob, os, torch


def merge_one(flow_path, act_dir, out_dir, overwrite=False):
    base = os.path.basename(flow_path)
    out = os.path.join(out_dir, base)
    if os.path.exists(out) and not overwrite:
        return "skip"
    act_path = os.path.join(act_dir, base)
    if not os.path.exists(act_path):
        return "no_act"
    c = torch.load(flow_path, weights_only=False)
    a = torch.load(act_path, weights_only=False)
    # .clone() (not just [:1].contiguous()) so torch.save serializes ONLY frame0's storage, not the
    # full [13,H,W,3] underlying storage that a view would still reference.
    gt0 = c["gt_rgb"][:1].clone()                       # [1,H,W,3] uint8 (frame0 only)
    joint = {
        # ---- flow GT ---- (.clone() everywhere so no saved tensor references oversized parent storage)
        "means": c["means"].clone(),
        "uv": c["uv"].clone(),
        "traj": c["traj"].clone(),
        "K_intr": c["K_intr"].clone(),
        "viewmat": c["viewmat"].clone(),
        "H": int(c["H"]), "W": int(c["W"]), "Kf": int(c["Kf"]),
        "instruction": c["instruction"],
        "gt_rgb": gt0,
        "vis": c["vis"].clone(),
        # ---- action chunk (same window) ----
        "dq": a["dq"].clone(),
        "anchor": a["anchor"].clone(),
        "control_hz": float(a["control_hz"]),
        # ---- bookkeeping ----
        "task": a["task"], "ep": int(a["ep"]), "s": int(a["s"]), "win": int(a["win"]),
        "split": a["split"], "backend": c.get("backend", "spatrack_p1"),
    }
    os.makedirs(out_dir, exist_ok=True)
    torch.save(joint, out)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_dir", required=True)
    ap.add_argument("--act_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    flows = sorted(glob.glob(os.path.join(args.flow_dir, "*.pt")))
    flows = flows[args.shard::args.nshard]
    ok = sk = na = 0
    for i, fp in enumerate(flows):
        r = merge_one(fp, args.act_dir, args.out, args.overwrite)
        ok += (r == "ok"); sk += (r == "skip"); na += (r == "no_act")
        if (i + 1) % 500 == 0:
            print(f"[jmerge] shard{args.shard}: {i+1}/{len(flows)} ok={ok} skip={sk} no_act={na}", flush=True)
    print(f"[jmerge] shard{args.shard} DONE: ok={ok} skip={sk} no_act={na} (of {len(flows)})", flush=True)


if __name__ == "__main__":
    main()
