"""Verify subtask instructions + boundary-biased sampling distribution."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.data.streaming import StreamingClipDataset

for ratio in [4.0, 10.0]:
    ds = StreamingClipDataset(K=8, stride=8, seed=1, boundary_ratio=ratio, boundary_frac=0.35, middle_weight=0.3)
    it = iter(ds)
    n_b = 0; N = 300; weights = []; ex = []
    for i, item in enumerate(it):
        n_b += int(item["is_boundary"]); weights.append(item["boundary_weight"])
        if i < 4:
            ex.append((item["is_boundary"], item["instruction"][:60]))
        if i + 1 >= N:
            break
    print(f"\n[ratio={ratio}] index segments={len(ds.index)}  boundary frac={n_b/N:.2f} "
          f"(target {ratio/(ratio+1):.2f})  mean loss-weight={sum(weights)/len(weights):.2f}")
    for b, s in ex:
        print(f"    is_boundary={b}  {s!r}")
