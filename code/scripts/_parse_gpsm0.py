"""Parse logs/gps_m0.log: split by arm, average the noisy per-clip metrics over the converged half."""
import re
import statistics as st
import sys

lines = open(sys.argv[1]).read().splitlines()
arms, cur = {}, None
skips = {}
for ln in lines:
    if "ARM-B GPSToken" in ln:
        cur = "B-GPSToken(M256,b30)"; arms[cur] = []; skips[cur] = 0
    elif "ARM-A random" in ln:
        cur = "A-random(M256)"; arms[cur] = []; skips[cur] = 0
    elif cur and "[skip]" in ln:
        skips[cur] += 1
    elif cur and re.match(r"^e\d+ s\d+", ln):
        d = {"step": int(re.search(r" s(\d+)", ln).group(1))}
        for k in ["corr", "ratio", "dcos", "leak", "relSel"]:
            m = re.search(k + r"(-?\d+\.\d+)", ln)
            if m:
                d[k] = float(m.group(1))
        arms[cur].append(d)

for arm, rows in arms.items():
    conv = [r for r in rows if r["step"] >= 300]
    print(f"=== {arm}: {len(rows)} metric-lines, {skips.get(arm,0)} skips | avg over step>=300 (n={len(conv)}) ===")
    for k in ["corr", "ratio", "dcos", "leak", "relSel"]:
        vals = [r[k] for r in conv if k in r]
        if vals:
            neg = sum(1 for v in vals if v < 0) if k == "dcos" else 0
            extra = f"  (dcos<0 on {neg}/{len(vals)} clips)" if k == "dcos" else ""
            print(f"    {k:7s}: mean {st.mean(vals):.3f}  median {st.median(vals):.3f}{extra}")
