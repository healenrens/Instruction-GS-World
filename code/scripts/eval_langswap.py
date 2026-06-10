"""§54 LANGUAGE-SELECTION eval — the v8 success metric. For each held-out clip, run the TRUE
instruction and several SWAP instructions (other LIBERO-object nouns) on the SAME scene/g0, with
PER-ENTITY UNIFORM control sampling (no GT-mover bias — the eval-leak fix), and measure whether the
model's MOTION follows the instruction.

Selection success for a (clip, swap) pair requires ALL of:
  (a) FLOOR        true-mover entity moves >= 0.25x its GT displacement under the TRUE instruction
                   (so 'predict nothing' cannot win by suppression),
  (b) SUPPRESSION  the true-mover moves <= 0.5x as much under the SWAP instruction,
  (c) QUIET        other OBJECT-class entities stay < 2cm under the TRUE instruction.
Reports selection accuracy + mean swap/true ratio, and an all-static baseline row.

  python code/scripts/eval_langswap.py --ckpt checkpoints/libero_v8lang/ckpt_last.pt --data data/libero_pi3 --split heldtask
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn.functional as F
from igsw.gaussians import GaussianSet
from igsw.dynamics.model import DynamicsConfig
from igsw.model_full import InstructGSWorldModel
from scripts.eval_sim_generalization import _to_dev

NOUNS = ["alphabet soup", "bbq sauce", "butter", "chocolate pudding", "cream cheese",
         "ketchup", "milk", "orange juice", "salad dressing", "tomato sauce"]
INSTR = lambda n: f"pick up the {n} and place it in the basket"


def build_model(ck):
    cfg = DynamicsConfig(**ck["cfg"])
    m = InstructGSWorldModel(cfg, n_control=ck.get("M", 2048), n_query=ck.get("n_query", 16),
                             cond_mode="aggregator", spatial_ground=True, dyn_gate=bool(ck.get("dyn_gate", 0)),
                             sem_dim=ck.get("sem_dim", 0), gate_uses_sem=bool(ck.get("gate_uses_sem", 1)),
                             gate_entity_pool=bool(ck.get("gate_entity_pool", 0)), entity_lbs=bool(ck.get("entity_lbs", 0)),
                             rel_head=bool(ck.get("rel_head", 0)), entity_head=bool(ck.get("entity_head", 0))).cuda().eval()
    m.load_state_dict(ck["model"], strict=False)
    return m


def uniform_controls(seg, n_keep, M, per=256, seed=0):
    """Per-entity uniform sampling over the first n_keep (non-fill) Gaussians; no GT-mover bias."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    parts = []
    for e in torch.unique(seg[:n_keep]).tolist():
        ii = (seg[:n_keep] == e).nonzero(as_tuple=True)[0]
        parts.append(ii[torch.randperm(ii.numel(), device="cuda", generator=g)[:min(per, ii.numel())]])
    ci = torch.cat(parts)
    if ci.numel() > M:
        ci = ci[torch.randperm(ci.numel(), device="cuda", generator=g)[:M]]
    return ci


@torch.no_grad()
def run(mdl, c, ci, seg_g, instr, K):
    g0 = GaussianSet(c["means"].cuda(), c["quats"].cuda(), c["scales"].cuda(), c["opacities"].cuda(), c["colors"].cuda(), None)
    img0 = c["gt_rgb"][0].cpu().numpy()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        vi = _to_dev(mdl.encoder.build_inputs(instr, img0), "cuda")
        out = mdl(vi, g0, K, ctrl_idx=ci, control_uv=c["uv"].cuda()[ci],
                  control_uv_hw=(int(c["H"]), int(c["W"])), seg_per_g=seg_g)
    return out["ctrl"][K - 1].float(), g0.means[ci].float()           # endpoints, init  [M,3] each


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/libero_pi3")
    ap.add_argument("--split", default="heldtask")
    ap.add_argument("--n_swap", type=int, default=4)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mdl = build_model(ck)
    clips = sorted(glob.glob(os.path.join(args.data, f"*_{args.split}.pt")))
    print(f"ckpt={args.ckpt} rel_head={ck.get('rel_head',0)} | {len(clips)} {args.split} clips")
    M = ck.get("M", 2048)
    succ, ratios, base_succ, dcoss, eerrs = [], [], [], [], []
    for cp in clips:
        c = torch.load(cp, map_location="cuda", weights_only=False)
        seg = c["seg_per_g"].cuda().long(); N = len(seg); nkeep = N - int(c.get("n_fill", 0))
        K = int(c["Kf"]); tr = c["traj"].cuda().float()
        ci = uniform_controls(seg, nkeep, M)
        seg_c = seg[ci]
        gt_disp = (tr[K] - tr[0]).norm(dim=-1)[ci]
        # true-mover entity = the object-class (1..7) entity with the largest GT displacement
        obj_es = [e for e in torch.unique(seg_c).tolist() if 1 <= e <= 7]
        if not obj_es:
            continue
        mv_e = max(obj_es, key=lambda e: float(gt_disp[seg_c == e].mean()))
        mv_m = seg_c == mv_e
        gt_mv = float(gt_disp[mv_m].mean())
        other_obj = [e for e in obj_es if e != mv_e]
        instr = c["instruction"]
        ep_true, init = run(mdl, c, ci, seg, instr, K)
        pd_true = (ep_true - init).norm(dim=-1)
        m_true = float(pd_true[mv_m].mean())
        # §A1 DIRECTION (the blind-spot metric): pred vs GT centroid displacement of the mover entity.
        gt_ep = tr[K][ci]
        pred_d = ep_true[mv_m].mean(0) - init[mv_m].mean(0)
        gt_d = gt_ep[mv_m].mean(0) - init[mv_m].mean(0)
        dir_cos = float(F.cosine_similarity(pred_d[None], gt_d[None]))
        endp_err = float((ep_true[mv_m] - gt_ep[mv_m]).norm(dim=-1).median())
        dcoss.append(dir_cos); eerrs.append(endp_err)
        quiet = all(float(pd_true[seg_c == e].mean()) < 0.02 for e in other_obj)
        floor = m_true >= 0.25 * gt_mv
        swaps = [INSTR(n) for n in NOUNS if n not in instr][:args.n_swap]
        for sw in swaps:
            ep_sw, _ = run(mdl, c, ci, seg, sw, K)
            m_sw = float((ep_sw - init).norm(dim=-1)[mv_m].mean())
            r = m_sw / max(m_true, 1e-6); ratios.append(r)
            succ.append(1.0 if (floor and quiet and m_sw <= 0.5 * m_true) else 0.0)
            base_succ.append(0.0)   # all-static baseline: m_true=0 -> floor fails -> 0
        print(f"  {os.path.basename(cp)}: mover=id{mv_e} GTdisp={gt_mv*100:.1f}cm TRUEmove={m_true*100:.1f}cm "
              f"dir-cos={dir_cos:+.2f} endErr={endp_err*100:.0f}cm "
              f"floor={'Y' if floor else 'N'} quiet={'Y' if quiet else 'N'} swap/true={sum(ratios[-len(swaps):])/max(1,len(swaps)):.2f}")
    n = max(1, len(succ))
    print(f"\n=== SELECTION ACCURACY: {sum(succ)/n:.2f}  ({int(sum(succ))}/{n} clip-swap pairs) ===")
    print(f"    mean swap/true motion ratio: {sum(ratios)/max(1,len(ratios)):.2f}  (low = language suppresses correctly)")
    print(f"    mean DIRECTION cos: {sum(dcoss)/max(1,len(dcoss)):+.2f}  endpoint-err: {sum(eerrs)/max(1,len(eerrs))*100:.0f}cm  "
          f"(high cos / low err = correct TRAJECTORY, not just correct selection)")
    print(f"    all-static baseline accuracy: {sum(base_succ)/n:.2f}  (suppression cannot game the metric)")


if __name__ == "__main__":
    main()
