"""Read-only language coverage audit and fixed-window V70 manifest preparation.

Native schemas are those used by lerobot_agibot.py and
build_multisource_video_index_v53.py. Unknown schemas need a curated sidecar.
Sidecar entries: source, raw_video_path, frame_offset, frame_count, fps,
instruction, language_provenance. Optional start_frame/end_frame restrict an
instruction to an episode-relative, half-open interval at the history anchor.
"""

import csv
import json
import math
import re
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def usable_language_text_v70(text):
    return (isinstance(text, str) and bool(text.strip())
            and text.strip().casefold() not in ("unknown task", "xxxx")
            and not re.fullmatch(r"task[_ -]?\d+", text.strip(), re.IGNORECASE))


def _fields(row):
    return {key: {"type": type(value).__name__, "sample": value}
            for key, value in row.items() if key in
            ("episode_index", "length", "tasks", "task_index", "action_config",
             "instruction", "language", "language_instruction") or key.startswith("videos/")}


def _embedded_fields(entry, teacher):
    evidence = []
    def visit(value, location):
        if isinstance(value, dict):
            for key, child in value.items():
                field = location + "." + str(key)
                if key in ("instruction", "language_instruction", "language", "tasks"):
                    texts = child if isinstance(child, list) else [child]
                    for text in texts:
                        if isinstance(text, str):
                            evidence.append({"instruction": text, "field": field,
                                             "kind": "embedded_text_candidate", "original_origin_verified": False,
                                             "semantic_instruction_verified": False,
                                             "usable_text": bool(usable_language_text_v70(text))})
                if isinstance(child, (dict, list)):
                    visit(child, field)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, (dict, list)):
                    visit(child, f"{location}[{index}]")
    visit(entry, "entry")
    if teacher is not None:
        visit(teacher, "teacher")
    return evidence


def _embedded(entry, teacher):
    return [field for field in _embedded_fields(entry, teacher) if field["usable_text"]]


class LanguageSourcesV70:
    """Index original metadata by source + raw video + frame offset, not episode ID."""

    def __init__(self, source_roots=None, sidecar=None):
        self.source_roots = source_roots or {}
        self.schemas = []
        self.records = defaultdict(list)
        self.loaded = set()
        self.discovered_roots = set()
        self.video_roots = {}
        self.sidecar = defaultdict(list)
        if sidecar:
            path = Path(sidecar).resolve()
            for row in read_json(path)["entries"]:
                raw = str((path.parent / row["raw_video_path"]).resolve())
                self.sidecar[(row["source"], raw, int(row["frame_offset"]))].append(
                    {**row, "sidecar_path": str(path)})

    def discover(self, source, video):
        roots = self.source_roots.get(source, [])
        if isinstance(roots, (str, Path)):
            roots = [roots]
        for root in roots:
            root = Path(root).resolve()
            if (source, str(root)) in self.discovered_roots:
                continue
            self.discovered_roots.add((source, str(root)))
            paths = [root / "meta/info.json"] if (root / "meta/info.json").is_file() else sorted(root.rglob("meta/info.json"))
            for info in paths:
                self.load_root(source, info.parent.parent)
        directory = (source, str(Path(video).parent))
        if directory not in self.video_roots:
            self.video_roots[directory] = next((parent for parent in Path(video).parents
                                                if (parent / "meta/info.json").is_file()), None)
        if self.video_roots[directory] is not None:
            self.load_root(source, self.video_roots[directory])

    def load_root(self, source, root):
        identity = (source, str(root))
        if identity in self.loaded:
            return
        self.loaded.add(identity)
        info = read_json(root / "meta/info.json")
        version = str(info.get("codebase_version", ""))
        schema = {"source": source, "root": str(root), "codebase_version": version,
                  "info_keys": sorted(info), "features": info.get("features", {}),
                  "metadata_files": [], "row_count": 0, "sample_fields": []}
        self.schemas.append(schema)
        if version.startswith("v3"):
            paths = sorted((root / "meta/episodes").rglob("*.parquet"))
            import pyarrow.parquet as pq
            rows = []
            for path in paths:
                table = pq.read_table(path)
                schema["metadata_files"].append({"path": str(path), "columns": table.schema.names})
                rows.extend((str(path), i, row) for i, row in enumerate(table.to_pylist()))
        elif version.startswith("v2"):
            path = root / "meta/episodes.jsonl"
            rows = [(str(path), i, row) for i, row in enumerate(_jsonl(path))] if path.is_file() else []
            schema["metadata_files"].append({"path": str(path), "exists": path.is_file()})
        else:
            schema["status"] = "unsupported_codebase_version"
            return
        tasks_path = root / "meta/tasks.jsonl"
        tasks = {int(row["task_index"]): row["task"] for row in _jsonl(tasks_path)} if tasks_path.is_file() else {}
        task_field = "meta/tasks.jsonl.task"
        schema["task_table"] = {"path": str(tasks_path), "rows": len(tasks)}
        parquet_tasks = root / "meta/tasks.parquet"
        if not tasks and parquet_tasks.is_file():
            import pyarrow.parquet as pq
            table = pq.read_table(parquet_tasks)
            pandas = json.loads((table.schema.metadata or {}).get(b"pandas", b"{}"))
            index_columns = pandas.get("index_columns", [])
            schema["task_table"] = {"path": str(parquet_tasks), "columns": table.schema.names,
                                    "pandas_index_columns": index_columns, "rows": table.num_rows,
                                    "sample_rows": table.slice(0, 3).to_pylist()}
            if len(index_columns) == 1 and isinstance(index_columns[0], str) and index_columns[0] in table.schema.names and "task_index" in table.schema.names:
                text_column = index_columns[0]
                tasks = {int(row["task_index"]): row[text_column] for row in table.to_pylist()}
                tasks_path = parquet_tasks
                task_field = f"meta/tasks.parquet.{text_column} (declared pandas index)"
            else:
                schema["task_table"]["status"] = "unsupported_language_index_schema"
        schema["row_count"] = len(rows)
        schema["status"] = "recognized_video_mapping"
        cameras = [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]
        from .grounded_motion_sources_v68 import video_record
        for path, ordinal, row in rows:
            if len(schema["sample_fields"]) < 3:
                schema["sample_fields"].append({"metadata_path": path, "row": ordinal,
                                                "row_keys": sorted(row), "fields": _fields(row)})
            texts = row.get("tasks", [])
            texts = texts if isinstance(texts, list) else [texts]
            language = [{"instruction": text, "field": "tasks", "kind": "original_episode_instruction"}
                        for text in texts if usable_language_text_v70(text)]
            if not language and row.get("task_index") is not None and int(row["task_index"]) in tasks:
                text = tasks[int(row["task_index"])]
                if usable_language_text_v70(text):
                    language = [{"instruction": text, "field": "task_index->" + task_field,
                                 "task_index": int(row["task_index"]), "task_table": str(tasks_path),
                                 "kind": "original_episode_instruction"}]
            # The AgiBot reader explicitly supports per-frame task_index lookup.
            if not language and source == "agibot" and version.startswith("v2") and tasks and "data_path" in info:
                data = root / info["data_path"].format(episode_chunk=int(row["episode_index"]) // int(info["chunks_size"]),
                                                     episode_index=int(row["episode_index"]))
                if data.is_file():
                    import pyarrow.parquet as pq
                    if "task_index" in pq.ParquetFile(data).schema.names:
                        task_ids = {value for value in pq.read_table(data, columns=["task_index"])["task_index"].to_pylist() if value is not None}
                        for task_id in sorted(task_ids):
                            text = tasks.get(int(task_id))
                            if usable_language_text_v70(text):
                                language.append({"instruction": text, "field": "data.task_index->meta/tasks.jsonl.task",
                                                 "task_index": int(task_id), "data_path": str(data),
                                                 "task_table": str(tasks_path), "kind": "original_episode_instruction"})
            for camera in cameras:
                if version.startswith("v3") and row.get(f"videos/{camera}/file_index") is None:
                    continue
                record = video_record(root, info, row, camera)
                key = (source, str(Path(record["path"]).resolve()), int(record["frame_offset"]))
                self.records[key].append({**record, "metadata_path": path, "metadata_row": ordinal,
                                          "original_episode_index": int(row["episode_index"]), "camera": camera,
                                          "metadata_fields": _fields(row),
                                          "language": language, "segments": row.get("action_config") or []})

    def match(self, source, video, record, anchor=None):
        self.discover(source, video)
        key = (source, str(Path(video).resolve()), int(record["frame_offset"]))
        candidates = []
        for row in self.sidecar.get(key, []):
            if float(row["fps"]) == float(record["fps"]) and int(row["frame_count"]) == int(record["frame_count"]):
                if usable_language_text_v70(row["instruction"]):
                    if anchor is None or int(row.get("start_frame", 0)) <= anchor < int(row.get("end_frame", row["frame_count"])):
                        candidates.append({"instruction": row["instruction"], **row["language_provenance"],
                                           "kind": "user_curated_sidecar", "metadata_path": row["sidecar_path"],
                                           "raw_video_path": key[1], "frame_offset": key[2]})
        if candidates:
            return candidates
        for row in self.records.get(key, []):
            if row["frame_count"] != int(record["frame_count"]) or row["fps"] != float(record["fps"]):
                continue
            base = {name: row[name] for name in ("metadata_path", "metadata_row", "original_episode_index", "camera")}
            base.update(raw_video_path=key[1], frame_offset=key[2], frame_count=row["frame_count"], fps=row["fps"],
                        match_method="source/raw_video/frame_offset/frame_count/fps")
            segments = [s for s in row["segments"] if usable_language_text_v70(s.get("action_text"))]
            active = [s for s in segments if anchor is not None and int(s["start_frame"]) <= anchor < int(s["end_frame"])]
            if active:
                candidates.extend({**base, "instruction": s["action_text"], "field": "action_config.action_text",
                                   "kind": "original_timed_instruction", "start_frame": int(s["start_frame"]),
                                   "end_frame": int(s["end_frame"]), "selection": "history_anchor_only"} for s in active)
            else:
                candidates.extend({**base, **text} for text in row["language"])
                if anchor is None:
                    candidates.extend({**base, "instruction": s["action_text"], "field": "action_config.action_text",
                                       "kind": "original_timed_instruction", "start_frame": int(s["start_frame"]),
                                       "end_frame": int(s["end_frame"])} for s in segments)
        return candidates


def load_stage2_v70(path):
    path = Path(path).resolve()
    manifest = read_json(path)
    root = (path.parent / manifest["root"]).resolve()
    raw_paths = {}
    plan_path = path.parent / "export_plan.json"
    if plan_path.is_file():
        raw_paths = {str((root / asset["path"]).resolve()): asset["source"] for asset in read_json(plan_path)["assets"]}
    return manifest, root, raw_paths


def resolve_stage2_manifest_v70(manifest=None, stage2_run=None):
    """Stage 2 writes run.json.args.manifest, not a new dataset.json."""
    if manifest:
        return str(Path(manifest).resolve())
    run = Path(stage2_run).resolve()
    run = run / "run.json" if run.is_dir() else run
    return str(Path(read_json(run)["args"]["manifest"]).resolve())


class VerifiedLanguageTraceV70:
    """Authoritative 2026-10-06 server trace; failed joins never fall back."""

    def __init__(self, path):
        self.path = str(Path(path).resolve())
        self.rows = _jsonl(self.path)
        self.by_case = defaultdict(list)
        self.frame_tasks = {}
        for index, row in enumerate(self.rows):
            self.by_case[row["case_id"]].append((index, row))

    def lookup(self, entry, case, video):
        matches = self.by_case.get(entry.get("case_id", case.get("case_id")), [])
        if len(matches) != 1:
            return None, "trace_case_missing" if not matches else "trace_case_ambiguous"
        index, row = matches[0]
        record = case["record"]
        exact = (row["source"] == entry["source"] and int(row["current_episode_index"]) == int(entry["episode_index"])
                 and row["partition"] == entry["partition"] and int(row["first_frame"]) == int(case["first_frame"])
                 and int(row["last_frame"]) == int(case["last_frame"]) and float(row["fps"]) == float(record["fps"])
                 and int(row["frame_count"]) == int(record["frame_count"])
                 and 0 <= int(row["first_frame"]) <= int(row["last_frame"]) < int(row["frame_count"]))
        if entry["source"] == "robotwin":
            origin = row.get("cache_origin")
            exact = exact and origin is not None
            if origin is not None:
                exact = exact and Path(origin["filename"]).name == Path(record["path"]).name
        else:
            exact = (exact and str(Path(row["video"]).resolve()) == str(Path(video).resolve())
                     and int(row["offset_frames"]) == int(record["frame_offset"])
                     and int(row["frame_count"]) == int(record["frame_count"]))
        if not exact:
            return (index, row), "trace_case_contract_mismatch"
        if row["join_status"] != "verified":
            return (index, row), "trace_" + row["join_status"]
        return (index, row), "verified"

    def evidence(self, located):
        index, row = located
        base = {"trace_path": self.path, "trace_row": index, "source_verified": True,
                "original_origin_verified": True, "semantic_instruction_verified": False,
                "original_metadata": row}
        if row["source"] == "hy_embodied":
            if not row["clip_frame_rows_complete"]:
                return []
            return [{**base, "instruction": text, "kind": "verified_frame_task_annotation", "task_index": int(task_id),
                     "field": "data.task_index->clip_task_texts", "instruction_scope": "history_anchor_task_not_episode_goal"}
                    for task_id, text in row["clip_task_texts"].items()
                    if row["clip_task_kinds"][str(task_id)] == "natural_language_candidate" and usable_language_text_v70(text)]
        if (len(row["original_tasks"]) != 1 or row["original_task_kinds"] != ["natural_language_candidate"]
                or not usable_language_text_v70(row["original_tasks"][0])):
            return []
        return [{**base, "instruction": row["original_tasks"][0], "kind": "verified_original_episode_instruction",
                 "field": "original_tasks[0]", "instruction_scope": "episode_goal_not_precise_future_subtask"}]

    def instruction_at(self, located, anchor):
        candidates = self.evidence(located)
        if not candidates:
            return [], "trace_no_usable_natural_language"
        _, row = located
        if row["source"] != "hy_embodied":
            return candidates, "verified"
        key = (row["data_file"], int(row["original_episode_index"]))
        if not Path(row["data_file"]).is_file():
            return [], "trace_frame_task_data_missing"
        if key not in self.frame_tasks:
            import pyarrow.parquet as pq
            table = pq.read_table(row["data_file"], columns=["episode_index", "frame_index", "task_index"],
                                  filters=[("episode_index", "=", key[1])])
            self.frame_tasks[key] = {int(record["frame_index"]): int(record["task_index"])
                                     for record in table.to_pylist() if record["task_index"] is not None}
        frames = self.frame_tasks[key]
        task_id = frames.get(anchor)
        if task_id is None:
            return [], "trace_history_anchor_task_missing"
        selected = [candidate for candidate in candidates if candidate["task_index"] == task_id]
        if not selected:
            return [], "trace_history_anchor_task_not_natural_language"
        first, last = anchor, anchor
        while frames.get(first - 1) == task_id:
            first -= 1
        while frames.get(last + 1) == task_id:
            last += 1
        return [{**candidate, "start_frame": first, "end_frame": last + 1,
                 "history_anchor": anchor, "selection": "exact_original_episode_and_history_anchor"}
                for candidate in selected], "verified"


def inspect_language_sources_v70(stage2_manifest, source_roots=None, sidecar=None, progress=None, verified_trace=None):
    """Load every existing teacher.pt; distinguish embedded text from native joins."""
    import torch
    manifest, root, raw_paths = load_stage2_v70(stage2_manifest)
    sources = LanguageSourcesV70(source_roots, sidecar)
    trace = VerifiedLanguageTraceV70(verified_trace) if verified_trace else None
    rows, counts, template = [], defaultdict(Counter), []
    teacher_files_read = 0
    for index, entry in enumerate(manifest["entries"]):
        teacher_path = root / entry["path"]
        teacher = None
        if trace is None and teacher_path.is_file():
            teacher = torch.load(teacher_path, map_location="cpu", mmap=True, weights_only=False)
            teacher_files_read += 1
        case = entry["case"] if "case" in entry else teacher["case"]
        record = case["record"]
        video = str((root / record["path"]).resolve())
        raw_video = raw_paths.get(video, video)
        text_fields = _embedded_fields(entry, teacher)
        embedded = [field for field in text_fields if field["usable_text"]]
        trace_status = None
        if trace is not None:
            located, trace_status = trace.lookup(entry, case, raw_video)
            joined = trace.evidence(located) if trace_status == "verified" else []
            mapping = [located[1]] if located is not None else []
        else:
            joined = sources.match(entry["source"], raw_video, record)
            mapping = [{key: value for key, value in row.items() if key not in ("language", "segments")}
                       for row in sources.records.get((entry["source"], raw_video, int(record["frame_offset"])), [])]
        group = entry["group"]
        group_candidate = {"field": "entry.group", "value": group, "semantic_instruction_verified": False,
                           "source_verified": False,
                           "status": "not_language_source"}
        if entry["source"] in ("agibot", "bridge", "robomind"):
            group_candidate["status"] = "grouptext_candidate" if usable_language_text_v70(group) else "placeholder_or_task_id"
            matching = [e for e in joined if e["kind"] in ("original_episode_instruction", "verified_original_episode_instruction") and e["instruction"] == group]
            if matching:
                group_candidate.update(status="confirmed_against_original_metadata", source_verified=True,
                                       original_evidence=matching)
                embedded.append({"instruction": group, "field": "entry.group", "kind": "confirmed_group_instruction",
                                 "original_origin_verified": True, "source_verified": True,
                                 "semantic_instruction_verified": False, "original_evidence": matching})
            elif usable_language_text_v70(group) and any(e["kind"] in ("original_episode_instruction", "verified_original_episode_instruction") for e in joined):
                group_candidate["status"] = "conflicts_with_original_metadata"
                embedded.append({"instruction": group, "field": "entry.group", "kind": "grouptext_candidate",
                                 "original_origin_verified": False, "semantic_instruction_verified": False})
        for candidate in embedded:
            matched_text = [e for e in joined if e["kind"] in ("original_episode_instruction", "verified_original_episode_instruction")
                            and e["instruction"] == candidate["instruction"]]
            if matched_text:
                candidate.update(original_origin_verified=True, source_verified=True, original_evidence=matched_text)
        embedded_text = {e["instruction"] for e in embedded}
        episode_text = {e["instruction"] for e in joined if e["kind"] not in ("original_timed_instruction", "verified_frame_task_annotation")}
        conflicting_text = embedded_text | episode_text
        embedded_status = "ambiguous" if len(embedded_text) > 1 else "present" if embedded_text else "missing"
        lookup_status = "ambiguous" if len(episode_text) > 1 else "joinable" if joined else "missing"
        if len(conflicting_text) > 1:
            status = "ambiguous"
        elif embedded_text:
            status = "already_embedded"
        elif joined:
            status = "reliably_joinable"
        else:
            status = "missing"
        if trace is not None and trace_status != "verified":
            status = "ambiguous" if trace_status == "trace_case_ambiguous" else "missing"
        counts[entry["source"]][status] += 1
        reason = "conflicting_text_candidates" if status == "ambiguous" else ""
        if status == "missing":
            reason = "matched_metadata_without_usable_language" if mapping else "no_raw_video_time_metadata_mapping"
            if trace is not None:
                reason = trace_status if trace_status != "verified" else "trace_no_usable_natural_language"
        if not teacher_path.is_file():
            reason = "teacher_file_missing"
        if not Path(video).is_file():
            reason = "rgb_payload_missing"
        row = {"entry_index": index, "case_id": entry.get("case_id", case.get("case_id", "")),
               "source": entry["source"], "group": entry["group"], "episode_index": entry["episode_index"],
               "partition": entry["partition"], "status": status, "reason": reason,
               "embedded_status": embedded_status, "original_lookup_status": lookup_status,
               "trace_status": trace_status,
               "embedded_original_text_conflict": bool(embedded_text and episode_text and embedded_text != episode_text),
               "teacher_path": str(teacher_path), "teacher_exists": teacher_path.is_file(),
               "video_path": video, "video_exists": Path(video).is_file(), "raw_video_path": raw_video,
               "frame_offset": record["frame_offset"], "frame_count": record["frame_count"], "fps": record["fps"],
               "entry_keys": sorted(entry), "case_keys": sorted(case), "record_keys": sorted(record),
               "teacher_keys": sorted(teacher) if teacher is not None else [],
               "group_text_evidence": group_candidate,
               "original_mapping_evidence": mapping,
               "actual_embedded_text_fields": text_fields,
               "embedded_evidence": embedded, "original_metadata_evidence": joined}
        rows.append(row)
        if progress is not None and (len(rows) % 1000 == 0 or len(rows) == len(manifest["entries"])):
            progress(len(rows), len(manifest["entries"]))
        if status in ("missing", "ambiguous"):
            template.append({"source": entry["source"], "raw_video_path": raw_video,
                             "frame_offset": record["frame_offset"], "frame_count": record["frame_count"],
                             "fps": record["fps"], "instruction": "", "language_provenance": {"author": "", "source_note": ""}})
    statuses = ("already_embedded", "reliably_joinable", "missing", "ambiguous")
    per_source = {source: {status: count[status] for status in statuses} for source, count in counts.items()}
    return {"version": 70, "stage2_manifest": str(Path(stage2_manifest).resolve()), "entries_audited": len(rows),
            "unique_episodes": len({(r["source"], r["group"], r["episode_index"]) for r in rows}),
            "partition_counts": dict(Counter(r["partition"] for r in rows)),
            "source_counts": dict(Counter(r["source"] for r in rows)),
            "teacher_files_read": teacher_files_read, "per_source": per_source,
            "verified_trace": trace.path if trace is not None else None,
            "teacher_text_audit": "reused_verified_trace_no_teacher_loads" if trace is not None else "all_existing_teacher_files_read",
            "embedded_by_source": {source: dict(Counter(r["embedded_status"] for r in rows if r["source"] == source)) for source in counts},
            "original_lookup_by_source": {source: dict(Counter(r["original_lookup_status"] for r in rows if r["source"] == source)) for source in counts},
            "group_text_by_source": {source: dict(Counter(r["group_text_evidence"]["status"] for r in rows if r["source"] == source)) for source in counts},
            "totals": {status: sum(c[status] for c in counts.values()) for status in statuses},
            "embedded_and_native_lookup_separate": True, "schema_inspection": sources.schemas,
            "entries": rows, "curated_sidecar_template": {"version": 70, "entries": template}}


def write_language_audit_v70(report, output, csv_path=None, sidecar_template=None):
    write_json(output, report)
    csv_path = Path(csv_path or Path(output).with_suffix(".csv"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["entry_index", "case_id", "source", "group", "episode_index", "partition", "status", "reason",
               "embedded_status", "original_lookup_status", "embedded_original_text_conflict",
               "teacher_path", "teacher_exists", "video_path", "video_exists", "raw_video_path", "frame_offset",
               "frame_count", "fps", "embedded_evidence", "original_metadata_evidence"]
    columns.append("group_text_evidence")
    columns.append("original_mapping_evidence")
    columns.append("actual_embedded_text_fields")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in report["entries"]:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], (list, dict)) else row[key] for key in columns})
    if sidecar_template:
        write_json(sidecar_template, report["curated_sidecar_template"])


def uniform_language_windows_v70(case, raw_frames=None, max_windows=4):
    """Fixed 3s/16 history + 5s/25 future; no motion-dependent selection."""
    fps = float(case["record"]["fps"])
    first = int(case["first_frame"])
    last = min(int(case["last_frame"]), int(case["record"]["frame_count"]) - 1)
    if raw_frames is not None:
        last = min(last, first + int(raw_frames) - 1)
    low, high = first + math.ceil(3 * fps), last - math.ceil(5 * fps)
    if high < low:
        return []
    count = min(4, max_windows, high - low + 1)
    anchors = [round(low + (high - low) * (i + .5) / count) for i in range(count)]
    windows, seen = [], set()
    for anchor in anchors:
        history = [round(anchor - 3 * fps + i * 3 * fps / 15) for i in range(16)]
        future = [round(anchor + i * 5 * fps / 25) for i in range(1, 26)]
        indices = history + future
        if tuple(indices) not in seen:
            seen.add(tuple(indices))
            windows.append(indices)
    return windows


def prepare_language_manifest_v70(stage2_manifest, teacher_checkpoint, output, source_roots=None,
                                  sidecar=None, audit=None, max_windows=4, seed=17,
                                  tokenizer_path=None, instruction_tokens=512, tokenizer=None, verified_trace=None):
    """Keep V69 clips/episode partitions; exclude and report windows without text."""
    manifest, root, raw_paths = load_stage2_v70(stage2_manifest)
    sources = LanguageSourcesV70(source_roots, sidecar)
    trace = VerifiedLanguageTraceV70(verified_trace) if verified_trace else None
    audit = read_json(audit) if isinstance(audit, (str, Path)) else audit
    if audit is None and trace is None:
        audit = inspect_language_sources_v70(stage2_manifest, source_roots, sidecar)
    if tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    token_counts = {}
    audit_rows = {r["entry_index"]: r for r in audit["entries"]} if audit else {}
    by_source = defaultdict(set)
    for entry in manifest["entries"]:
        if entry["partition"] == "held":
            by_source[entry["source"]].add((entry["group"], int(entry["episode_index"])))
    diagnostic = set()
    rng = random.Random(seed)
    for source, identities in sorted(by_source.items()):
        episodes = sorted(identities)
        rng.shuffle(episodes)
        diagnostic.update((source, *episode) for episode in episodes[:len(episodes) // 2])
    entries, excluded, seen = [], [], set()
    trace_clip_counts = Counter()
    for index, entry in enumerate(manifest["entries"]):
        case = entry["case"]
        record = case["record"]
        video = str((root / record["path"]).resolve())
        raw_video = raw_paths.get(video, video)
        located = None
        if trace is not None:
            located, trace_status = trace.lookup(entry, case, raw_video)
            if trace_status != "verified":
                excluded.append({"entry_index": index, "case_id": entry.get("case_id", case.get("case_id")),
                                 "source": entry["source"], "reason": trace_status})
                continue
            if not trace.evidence(located):
                excluded.append({"entry_index": index, "case_id": entry.get("case_id", case.get("case_id")),
                                 "source": entry["source"], "reason": "trace_no_usable_natural_language"})
                continue
            trace_clip_counts[entry["source"]] += 1
            embedded = []
        else:
            embedded = [candidate for candidate in audit_rows[index]["embedded_evidence"]
                        if candidate.get("original_origin_verified") is True]
        episode_key = (entry["source"], entry["group"], int(entry["episode_index"]))
        partition = entry["partition"]
        if partition == "held":
            partition = "diagnostic" if episode_key in diagnostic else "test"
        if trace is None and not audit_rows[index]["teacher_exists"]:
            excluded.append({"entry_index": index, "reason": "teacher_file_missing"})
            continue
        if trace is None and audit_rows[index]["status"] == "ambiguous" and sidecar is None:
            excluded.append({"entry_index": index, "reason": "ambiguous_language"})
            continue
        if not Path(video).is_file():
            excluded.append({"entry_index": index, "reason": "rgb_payload_missing"})
            continue
        windows = uniform_language_windows_v70(case, entry.get("raw_frames"), max_windows)
        if not windows:
            excluded.append({"entry_index": index, "reason": "no_legal_8s_window"})
        for indices in windows:
            if trace is not None:
                candidates, trace_status = trace.instruction_at(located, indices[15])
                if trace_status != "verified":
                    excluded.append({"entry_index": index, "case_id": entry.get("case_id", case.get("case_id")),
                                     "source": entry["source"], "frame_indices": indices, "reason": trace_status})
                    continue
            else:
                native = sources.match(entry["source"], raw_video, record, indices[15])
                candidates = native if native and native[0]["kind"] in ("original_timed_instruction", "user_curated_sidecar") else embedded + native
            candidates = [candidate for candidate in candidates if usable_language_text_v70(candidate["instruction"])]
            texts = {c["instruction"] for c in candidates}
            if len(texts) != 1:
                excluded.append({"entry_index": index, "frame_indices": indices,
                                 "reason": "missing_language" if not texts else "ambiguous_language", "evidence": candidates})
                continue
            instruction = candidates[0]["instruction"]
            if tokenizer is not None:
                if instruction not in token_counts:
                    token_counts[instruction] = len(tokenizer.encode(instruction, add_special_tokens=False))
                if token_counts[instruction] > instruction_tokens:
                    excluded.append({"entry_index": index, "source": entry["source"], "frame_indices": indices,
                                     "reason": "instruction_token_budget_exceeded", "instruction": instruction,
                                     "instruction_tokens": token_counts[instruction], "budget": instruction_tokens})
                    continue
            identity = (entry["source"], raw_video, record["frame_offset"], tuple(indices))
            if identity in seen:
                continue
            seen.add(identity)
            window_id = f"v70_e{index:08d}_a{indices[15]:09d}"
            provenance = {**candidates[0], "raw_video_path": raw_video, "frame_offset": record["frame_offset"],
                          "fps": record["fps"], "history_anchor": indices[15]}
            provenance.pop("instruction")
            entries.append({"window_id": window_id, "case_id": entry.get("case_id", case.get("case_id", "")),
                            "source": entry["source"], "group": entry["group"],
                            "episode_index": entry["episode_index"], "partition": partition,
                            "stage2_partition": entry["partition"],
                            "instruction": instruction, "instruction_tokens": token_counts.get(instruction),
                            "language_provenance": provenance,
                            "case": {**case, "record": {**record, "path": video}},
                            "frame_indices": indices, "history_frames": 16,
                            "history_times": [(i - indices[15]) / float(record["fps"]) for i in indices[:16]],
                            "label_path": str((Path(output).resolve().parent / "labels" / f"{window_id}.pt"))})
    result = {"version": 70, "teacher_checkpoint": str(Path(teacher_checkpoint).resolve()),
              "stage2_manifest": str(Path(stage2_manifest).resolve()), "entries": entries,
              "report": {"input_clips": len(manifest["entries"]), "stage2_root": str(root),
                         "windows": len(entries), "excluded": excluded,
                         "verified_trace": {"path": trace.path, "rows": len(trace.rows),
                                            "authoritative": True, "fallback": False,
                                            "language_eligible_clips": sum(trace_clip_counts.values()),
                                            "language_eligible_clips_by_source": dict(trace_clip_counts),
                                            "teacher_files_loaded": 0} if trace is not None else None,
                         "exclusion_counts": dict(Counter(e["reason"] for e in excluded)),
                         "instruction_budget": {"tokens": instruction_tokens, "applied": tokenizer is not None,
                                                "tokenizer_path": str(Path(tokenizer_path).resolve()) if tokenizer_path else None,
                                                "add_special_tokens": False, "truncation": False,
                                                "chat_and_timestamp_overhead": "not included; logged by conditioner"},
                         "source_partition_counts": dict(Counter(f"{e['source']}/{e['partition']}" for e in entries)),
                         "schema_inspection": sources.schemas, "motion_filtering": False,
                         "split": {"unit": "source/group/episode", "seed": seed,
                                   "stage2_partition_preserved": True, "train": "unchanged",
                                   "held": "per-source sorted episode identities, Python Random(seed) shuffle; first floor(N/2) diagnostic, remainder test",
                                   "assignment_before_language_exclusion": True, "evaluation_selection_used": False,
                                   "diagnostic_episodes_assigned": len(diagnostic),
                                   "test_episodes_assigned": sum(len(v) for v in by_source.values()) - len(diagnostic)},
                         "window_sampling": "uniform_legal_anchor_bin_centers"}}
    write_json(output, result)
    return result
