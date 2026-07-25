"""Make a 6-frame montage per rollout (logs/rollout_batch/<task>_s<seed>/rollout.mp4) + a label with the
success flag, so all rollouts can be eyeballed at a glance."""
import os, glob, subprocess
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image

BATCH = "logs/rollout_batch"
for d in sorted(glob.glob(f"{BATCH}/*/")):
    name = os.path.basename(d.rstrip("/"))
    mp4 = os.path.join(d, "rollout.mp4")
    log = os.path.join(d, "log.txt")
    if not os.path.exists(mp4):
        continue
    succ = "?"
    if os.path.exists(log):
        t = open(log, errors="ignore").read()
        succ = "SUCCESS" if "DONE success=True" in t else ("FAIL" if "DONE success=False" in t else "?")
        cnt = ""
        import re
        m = re.findall(r"take_action_cnt=(\d+)/(\d+)", t)
        if m:
            cnt = f"  ({m[-1][0]}/{m[-1][1]} steps)"
    tmp = f"/tmp/mz_{name}"; os.makedirs(tmp, exist_ok=True)
    subprocess.run(f"rm -f {tmp}/*.png && ffmpeg -y -loglevel error -i {mp4} {tmp}/f_%04d.png", shell=True)
    frames = sorted(glob.glob(f"{tmp}/f_*.png"))
    if not frames:
        print("no frames", name); continue
    idx = np.linspace(0, len(frames) - 1, min(6, len(frames))).astype(int)
    sel = [frames[i] for i in idx]
    fig, ax = plt.subplots(1, len(sel), figsize=(len(sel) * 2.4, 2.4))
    if len(sel) == 1:
        ax = [ax]
    for a, f in zip(ax, sel):
        a.imshow(Image.open(f)); a.axis("off")
    col = "green" if succ == "SUCCESS" else ("red" if succ == "FAIL" else "gray")
    fig.suptitle(f"{name}   ->  {succ}{cnt}", fontsize=12, color=col, y=0.99)
    plt.tight_layout()
    out = f"{BATCH}/montage_{name}.png"
    plt.savefig(out, dpi=72, bbox_inches="tight"); plt.close()
    print("montage", name, succ, "->", out)
print("DONE")
