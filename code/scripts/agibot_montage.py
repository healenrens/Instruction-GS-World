"""Assemble all agibot_gt_*.json summaries into ONE overview montage + a sortable table.
Reads the per-episode PNGs (already saved by agibot_spatrack_eval.py) and stacks them, annotated
with the camera-motion verdict, sorted by cam_trans_frac (cleanest static at top, worst ego-motion at bottom).
Usage: agibot_montage.py [out_dir]
"""
import sys, os, glob, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.image as mpimg

out_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs"
jsons = sorted(glob.glob(os.path.join(out_dir, "agibot_gt_*.json")))
rows = []
for j in jsons:
    d = json.load(open(j))
    png = d["png"]
    if not os.path.exists(png):
        png = os.path.join(out_dir, os.path.basename(j).replace(".json", ".png"))
    if not os.path.exists(png):
        continue
    fcf = d["fixed_cam_false"]
    rows.append((fcf.get("cam_trans_frac", 0.0), d, png))
rows.sort(key=lambda r: r[0])

n = len(rows)
fig, axes = plt.subplots(n, 1, figsize=(16, 3.6 * n))
if n == 1: axes = [axes]
for ax, (trf, d, png) in zip(axes, rows):
    ax.imshow(mpimg.imread(png)); ax.axis("off")
    fct, fcf = d["fixed_cam_true"], d["fixed_cam_false"]
    verdict = ("STATIC-cam (fixed_cam=True OK)" if trf < 0.02
               else "MODERATE ego-motion (suspect)" if trf < 0.05
               else "STRONG ego-motion (fixed_cam=True WRONG)")
    ax.set_title(
        f"{d['task']} ep{d['episode']}  |  {d['label'][:60]}  |  {verdict}\n"
        f"cam_trans={fcf['cam_trans']:.4f} ({100*trf:.1f}% depth) rot={fcf['cam_rot_deg']:.2f}deg  |  "
        f"movers(T)={fct['movers']}->{fcf['movers']}(F)  mover_med2d={fct['mover_med2d']:.0f}px  "
        f"vis={fct['vis']:.2f}  reproj={fct['reproj_med']:.2f}px  z[{fct['zmin']:.2f},{fct['zmax']:.2f}]",
        fontsize=9, loc="left")
out = os.path.join(out_dir, "agibot_gt_OVERVIEW.png")
fig.tight_layout(); fig.savefig(out, dpi=85, bbox_inches="tight"); plt.close(fig)
print("SAVED", out, "with", n, "episodes")

# also dump a plain-text table sorted by ego-motion
print("\n%-12s %-4s %7s %6s %8s->%-6s %8s %5s %6s  %s" %
      ("task", "ep", "trans%", "rot", "movT", "movF", "mvmed2d", "vis", "reproj", "verdict"))
for trf, d, png in rows:
    fct, fcf = d["fixed_cam_true"], d["fixed_cam_false"]
    verdict = "STATIC" if trf < 0.02 else "MODER" if trf < 0.05 else "STRONG-EGO"
    print("%-12s %-4d %6.1f%% %5.1f %8d->%-6d %7.0f %5.2f %6.2f  %s" %
          (d["task"], d["episode"], 100*trf, fcf["cam_rot_deg"], fct["movers"], fcf["movers"],
           fct["mover_med2d"], fct["vis"], fct["reproj_med"], verdict))
