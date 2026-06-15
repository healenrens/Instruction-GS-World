"""Compare GPSToken reconstructions vs GT (PSNR + side-by-side montage). Tests whether the sparse
256-token representation captures our (out-of-domain) robot frames.
Usage: python _gps_recon_compare.py <gt_dir> <recon_dir> <out.png>"""
import glob
import os
import sys

import cv2
import numpy as np

gt_dir, recon_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]


def psnr(a, b):
    m = ((a.astype(float) - b.astype(float)) ** 2).mean()
    return 99.0 if m == 0 else 10 * np.log10(255.0 ** 2 / m)


rows = []
for g in sorted(glob.glob(f"{gt_dir}/*")):
    name = os.path.basename(g)
    r = os.path.join(recon_dir, name)
    if not os.path.exists(r):
        # recon filename may differ; try the only/first file by stem
        cands = glob.glob(os.path.join(recon_dir, os.path.splitext(name)[0] + "*"))
        if not cands:
            continue
        r = cands[0]
    gi, ri = cv2.imread(g), cv2.imread(r)
    if gi is None or ri is None:
        continue
    if gi.shape != ri.shape:
        ri = cv2.resize(ri, (gi.shape[1], gi.shape[0]))
    p = psnr(gi, ri)
    print(f"{name}: PSNR {p:.2f} dB")
    pair = np.concatenate([gi, ri], axis=1)
    cv2.putText(pair, f"{name}  GT | GPSToken-256recon  PSNR {p:.1f}", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(pair, f"{name}  GT | GPSToken-256recon  PSNR {p:.1f}", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    rows.append(pair)
if rows:
    cv2.imwrite(out, np.concatenate(rows, axis=0))
    print(f"-> {out}")
else:
    print("no matched recon/gt pairs found")
