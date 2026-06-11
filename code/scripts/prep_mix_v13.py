"""§87 R4 vocab: assemble the v13 mix = LIBERO sim + AgiBot real + libero_90 BOOK (a genuinely NEW
object category). Book clips split train/held90 -> the unseen-noun-GENERALIZATION test (does training
on some book instances let the model move HELD-OUT book instances? heldtask was 0 = vocab ceiling).
Quality gate same as v12 (target disp 5-80 StV2-cm, 500-30k obj gaussians)."""
import glob
import os

import torch

WS = "/mnt/pfs/public/xuhaoming/instruct_gs_world"
SIM = os.path.join(WS, "data/libero_pi3_v2")
REAL = os.path.join(WS, "data/_agibot")
OUT = os.path.join(WS, "data/mix_v13")


def gate(f):
    c = torch.load(f, map_location="cpu", weights_only=False)
    io = c["is_obj"]
    if not int(io.sum()):
        return False
    med = float((c["traj"][-1][io] - c["traj"][0][io]).norm(dim=-1).median())
    return (0.05 <= med <= 0.8) and (500 <= int(io.sum()) <= 30000)


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "*.pt")):
        os.remove(f)
    n = {"sim": 0, "real": 0, "book_tr": 0, "book_held": 0}
    for f in sorted(glob.glob(os.path.join(SIM, "*.pt"))):       # LIBERO sim (keep splits)
        os.symlink(f, os.path.join(OUT, os.path.basename(f)))
        n["sim"] += 1
    real = [f for f in sorted(glob.glob(os.path.join(REAL, "clip_stv2_ep*.pt")),
                              key=lambda x: int(x.split("ep")[1].split(".")[0])) if gate(f)]
    for k, f in enumerate(real):                                  # AgiBot real
        ep = int(f.split("ep")[1].split(".")[0])
        sp = "heldreal" if k % 5 == 4 else "train"
        os.symlink(f, os.path.join(OUT, f"epi9{ep:05d}_r_{sp}.pt"))
        n["real"] += 1
    book = [f for f in sorted(glob.glob(os.path.join(REAL, "clip_stv2_lib90_ep*.pt")),
                              key=lambda x: int(x.split("ep")[1].split(".")[0])) if gate(f)]
    for k, f in enumerate(book):                                  # libero_90 BOOK (new noun)
        ep = int(f.split("ep")[1].split(".")[0])
        sp = "held90" if k % 4 == 3 else "train"                  # 25% held -> unseen-instance test
        os.symlink(f, os.path.join(OUT, f"epi8{ep:05d}_b_{sp}.pt"))
        n["book_tr" if sp == "train" else "book_held"] += 1
    print(f"[mix_v13] sim={n['sim']} real={n['real']} | book train={n['book_tr']} held90={n['book_held']} "
          f"(of {len(book)} book clips passing gate)")


if __name__ == "__main__":
    main()
