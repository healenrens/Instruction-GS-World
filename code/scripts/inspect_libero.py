"""Feasibility inspection of the LIBERO LeRobot mask+depth dataset for our clean-GT 3DGS pipeline.
The visual streams are stored IN the parquet (HF 'image' dtype = encoded bytes), not as videos.
Decodes frame 0 to check: DEPTH encoding (metric/usable?), MASK ids, object_of_interest, task language."""
import io
import os

os.environ.setdefault("HF_HOME", "/mnt/pfs/public/xuhaoming/hf_cache")
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "binhng/libero_object_lerobot_mask_depth"


def dl(rel):
    return hf_hub_download(REPO, rel, repo_type="dataset")


print("=== TASKS ===")
for i, line in enumerate(open(dl("meta/tasks.jsonl"))):
    if i < 8:
        print("  ", line.strip())

t = pq.read_table(dl("data/chunk-000/episode_000000.parquet"))
print("\n=== PARQUET schema ===")
for f in t.schema:
    print(f"  {f.name}: {f.type}")
print("rows", t.num_rows)
st = np.array(t["observation.state"].to_pylist())
print("state", st.shape, "state[0]", np.round(st[0], 3))


def decode(cell):
    if isinstance(cell, dict) and cell.get("bytes") is not None:
        return np.array(Image.open(io.BytesIO(cell["bytes"])))
    if isinstance(cell, (bytes, bytearray)):
        return np.array(Image.open(io.BytesIO(cell)))
    return np.array(cell)


print("\n=== FRAME-0 visual streams (decoded from parquet) ===")
for vk in ["observation.images.image", "observation.images.image_depth",
           "observation.images.image_mask", "observation.images.object_of_interest_mask"]:
    try:
        fr = decode(t[vk][0].as_py())
        u = np.unique(fr)
        extra = ""
        if "depth" in vk and fr.ndim == 3:
            extra = (f" ch_ranges={[(int(fr[..., c].min()), int(fr[..., c].max())) for c in range(fr.shape[2])]}"
                     f" ch_equal={all(np.array_equal(fr[..., 0], fr[..., c]) for c in range(1, fr.shape[2]))}")
        print(f"  {vk}: {fr.shape} {fr.dtype} min{fr.min()} max{fr.max()} "
              f"nuniq{len(u)}{(' uniq=' + str(u.tolist())) if len(u) <= 16 else ''}{extra}")
    except Exception as e:
        print(f"  {vk}: ERR {type(e).__name__} {e}")
print("\nNote: LIBERO=robosuite; agentview camera intrinsics derivable from fovy(45deg default)+256x256.")
