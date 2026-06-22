"""Dump frame-0 of our clips as 512x512 PNGs for the GPSToken tokenizer reconstruction test
(their inference path requires resolution divisible by 256). Writes a list.txt for the CLI."""
import os
import sys

import cv2
import torch

OUT = "third_party/GPSToken/our_frames"
os.makedirs(OUT, exist_ok=True)
paths = []
for p in sys.argv[1:]:
    c = torch.load(p, map_location="cpu", weights_only=False)
    rgb = c["gt_rgb"][0].numpy()  # [H,W,3] uint8 RGB
    rgb512 = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
    name = os.path.basename(p).replace(".pt", "") + ".png"
    cv2.imwrite(f"{OUT}/{name}", rgb512[:, :, ::-1])  # RGB->BGR for imwrite
    paths.append(os.path.abspath(f"{OUT}/{name}"))
with open(f"{OUT}/list.txt", "w") as f:
    f.write("\n".join(paths) + "\n")
print(f"dumped {len(paths)} frames -> {OUT}/  (list.txt)")
