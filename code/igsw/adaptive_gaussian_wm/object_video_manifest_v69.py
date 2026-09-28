"""Resolve portable dataset paths against the manifest, never the shell directory."""

import json
from pathlib import Path


def load_object_video_manifest_v69(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    return {**manifest, "root": str((path.parent / manifest["root"]).resolve())}


def resolve_object_video_case_v69(case, root):
    record = case["record"]
    return {**case, "record": {**record, "path": str((Path(root) / record["path"]).resolve())}}
