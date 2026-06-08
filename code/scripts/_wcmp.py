import torch
a = torch.load("checkpoints/stream11_infonce/ckpt_0004000.pt", map_location="cpu", weights_only=False)["model"]
b = torch.load("checkpoints/stream11b_infonce/ckpt_last.pt", map_location="cpu", weights_only=False)["model"]
rows = []
for k in a:
    if k in b and torch.is_tensor(a[k]) and a[k].is_floating_point() and a[k].numel() > 0:
        na = a[k].float().norm().item()
        nb = b[k].float().norm().item()
        mb = b[k].float().abs().max().item()
        rows.append((nb / (na + 1e-9), k, na, nb, mb))
rows.sort(reverse=True)
print("=== TOP 16 weight-norm GROWERS (s4000 clean -> s6000 broken) ===")
print("%7s %9s %10s %10s  %s" % ("ratio", "normA", "normB", "maxB", "param"))
for ratio, k, na, nb, mb in rows[:16]:
    print("%7.2f %9.3f %10.3f %10.2f  %s" % (ratio, na, nb, mb, k))
big = [(k, b[k].float().abs().max().item()) for k in b
       if torch.is_tensor(b[k]) and b[k].is_floating_point() and b[k].numel() > 0
       and b[k].float().abs().max().item() > 50]
print("=== params with max-abs > 50 in broken ckpt (%d total) ===" % len(big))
for k, m in sorted(big, key=lambda x: -x[1])[:14]:
    print("  maxabs=%10.1f  %s" % (m, k))
