import torch, glob
for f in sorted(glob.glob("checkpoints/stream11_infonce/ckpt_*.pt")):
    ck = torch.load(f, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    bad = []
    tot = 0
    for k, v in sd.items():
        if torch.is_tensor(v) and v.is_floating_point():
            tot += 1
            if not torch.isfinite(v).all():
                bad.append(k)
    step = ck.get("step", "?")
    status = "CLEAN" if not bad else ("NaN/Inf in %d tensors e.g. %s" % (len(bad), bad[:3]))
    print("%s step=%s : %s (%d float tensors)" % (f, step, status, tot))
