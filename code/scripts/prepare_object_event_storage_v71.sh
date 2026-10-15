#!/usr/bin/env bash
# Expand the existing portable archive on Alibaba PFS. No download or re-encoding.
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/mnt/pfs/xuhaoming/wm-align}"
PARTS_ROOT="${PARTS_ROOT:-/mnt/data/baidu/sync_data/wm-align/object_video_v69_portable.parts}"
PYTHON="${PYTHON:-/usr/bin/python3}"

mkdir -p "${WORK_ROOT}/datasets" "${WORK_ROOT}/metadata" "${WORK_ROOT}/logs" \
  "${WORK_ROOT}/outputs" "${WORK_ROOT}/models" "${WORK_ROOT}/tmp" \
  "${WORK_ROOT}/cache/huggingface" "${WORK_ROOT}/cache/torch" "${WORK_ROOT}/cache/uv"

export TMPDIR="${WORK_ROOT}/tmp"
export HF_HOME="${WORK_ROOT}/cache/huggingface"
export TORCH_HOME="${WORK_ROOT}/cache/torch"
export UV_CACHE_DIR="${WORK_ROOT}/cache/uv"
export XDG_CACHE_HOME="${WORK_ROOT}/cache"

printf '[object-event-data] source=%s\ndestination=%s\n' "${PARTS_ROOT}" "${WORK_ROOT}/datasets"
cat "${PARTS_ROOT}"/object_video_v69_portable.tar.part-* \
  | tar --extract --file=- --directory="${WORK_ROOT}/datasets" \
      --checkpoint=100000 --checkpoint-action=echo

"${PYTHON}" - "${WORK_ROOT}" <<'PY'
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = root / "datasets/object_video_v69_portable/manifest.json"
data = json.loads(manifest.read_text())
report = {
    "phase": "archive_extracted",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "manifest": str(manifest),
    "clips": len(data["entries"]),
    "source_partition_counts": dict(Counter(
        f"{row['source']}/{row['partition']}" for row in data["entries"])),
    "video_decode_performed": False,
    "reencoded": False,
}
path = root / "metadata/portable_extraction.json"
path.write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
PY
