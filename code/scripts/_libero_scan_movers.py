"""§50: scan LIBERO episodes for REAL object motion (failed/approach-only demos like epi400 have
none — the gripper moves, the object doesn't). Light: decodes ONLY the mask column.
Per episode: detected object id (max END-START centroid disp among non-robot ids) + that disp.
Usage: _libero_scan_movers.py <epi list...>"""
import sys, io, json
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import hf_hub_download

REPO = "binhng/libero_object_lerobot_mask_depth"


def scan(epi):
    t = pq.read_table(hf_hub_download(REPO, f"data/chunk-000/episode_{epi:06d}.parquet", repo_type="dataset"),
                      columns=["observation.images.image_mask", "task_index"])
    n = t.num_rows
    ti = int(t["task_index"][0].as_py())
    msk = []
    for i in range(0, n, 2):                                   # stride 2: enough for end-start
        cell = t["observation.images.image_mask"][i].as_py()
        m = np.array(Image.open(io.BytesIO(cell["bytes"]))) if isinstance(cell, dict) else np.array(cell)
        msk.append(m[..., 0] if m.ndim == 3 else m)
    msk = np.stack(msk)
    best, bid = -1.0, -1
    for i in [int(x) for x in np.unique(msk[0]) if int(x) not in (0, 8, 10)]:
        cs = np.array([np.argwhere(f == i)[:, [1, 0]].mean(0) if (f == i).sum() > 10
                       else [np.nan, np.nan] for f in msk])
        valid = np.where(~np.isnan(cs[:, 0]))[0]
        if len(valid) < 2:
            continue
        d = float(np.linalg.norm(cs[valid[-1]] - cs[valid[0]]))
        if d > best:
            best, bid = d, i
    return ti, bid, best, n


if __name__ == "__main__":
    epis = [int(x) for x in sys.argv[1:]]
    for e in epis:
        try:
            ti, bid, d, n = scan(e)
            tag = "MOVER" if d > 15 else "static"
            print(f"epi{e:03d} task{ti} obj_id={bid} disp={d:5.1f}px n={n}  {tag}", flush=True)
        except Exception as ex:
            print(f"epi{e:03d} ERROR {type(ex).__name__}: {ex}", flush=True)
