"""Preparation-only network operations and official dataset manifests."""

import csv
import io
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from collections import Counter
import random
from pathlib import Path

import av
import h5py
from PIL import Image

from .io import read_json, write_json

PHYSION_BASE = "https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/"
SCENARIOS = ["Collide", "Contain", "Dominoes", "Drape", "Drop", "Link", "Roll", "Support"]


def stratified_development(rows, fraction, seed):
    groups = {}
    for row in rows:
        if row["split"] == "train":
            groups.setdefault((row["scenario"], row["label"]), []).append(row)
    for group in groups.values():
        random.Random(seed).shuffle(group)
        count = min(len(group) - 1, max(1, round(len(group) * fraction))) if fraction else 0
        for row in group[:count]:
            row["split"] = "dev"


def stratified_pilot(rows, limit, seed):
    selected = []
    for split in ("train", "dev", "test"):
        groups = {}
        for row in rows:
            if row["split"] == split:
                groups.setdefault((row["scenario"], row["label"]), []).append(row)
        rng = random.Random(seed)
        for group in groups.values():
            rng.shuffle(group)
        ordered = sorted(groups)
        rng.shuffle(ordered)
        candidates = [groups[key][index] for index in range(max(map(len, groups.values()), default=0))
                      for key in ordered if index < len(groups[key])]
        selected.extend(candidates[:limit])
    return selected


def official_physion_plan(root, scenarios, limit):
    rows = []
    for scenario in scenarios:
        for kind in ("readout_training", "testing"):
            name = f"{scenario}_{kind}_HDF5s.tar.gz"
            if scenario == "Roll" and kind == "readout_training":
                name = "Rollreadout_HDF5s.tar.gz"
            rows.append({"url": PHYSION_BASE + name, "mode": "tar_stream",
                         "destination": str(Path(root) / kind / scenario), "max_files": limit})
    return rows


def download_plan(config, plan_path=None):
    plan = read_json(plan_path) if plan_path else config.get("downloads", [])
    if config["benchmark"] == "physion" and plan_path is None:
        data = config["data"]
        plan = official_physion_plan(data["root"], data["scenarios"], data["download_limit"])
    for row in plan:
        destination = Path(row["destination"])
        if row["mode"] == "tar_stream":
            destination.mkdir(parents=True, exist_ok=True)
            count = 0
            # A pilot stops after N HDF5 members; it never downloads the dynamics corpus.
            with urllib.request.urlopen(row["url"]) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".hdf5"):
                        continue
                    output = destination / Path(member.name).name
                    if not output.exists():
                        with archive.extractfile(member) as source, output.with_suffix(".partial").open("wb") as target:
                            shutil.copyfileobj(source, target)
                        output.with_suffix(".partial").replace(output)
                    count += 1
                    print(f"download {output} files={count}", flush=True)
                    if row["max_files"] and count >= row["max_files"]:
                        break
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            # curl honors the server proxy and resumes authorized large archive downloads.
            subprocess.run(["curl", "--fail", "--location", "--retry", "0", "--continue-at", "-",
                            "--output", str(destination), row["url"]], check=True)
    write_json(Path(config["data"]["root"]) / "download_receipt.json", plan)


def unpack(archives, destination, join_parts=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archives = [Path(path) for path in archives]
    if join_parts:
        # Joining is explicit: independently compressed zip files must remain separate.
        joined = destination / "joined_archive.tgz"
        with joined.open("wb") as target:
            for path in archives:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, target)
        archives = [joined]
    for path in archives:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                archive.extractall(destination)
        else:
            with tarfile.open(path) as archive:
                archive.extractall(destination, filter="data")


def inspect_schema(path):
    path = Path(path)
    if path.suffix == ".hdf5":
        with h5py.File(path) as data:
            first = sorted(data["frames"])[0]
            fields = {}
            for group in ("static", f"frames/{first}"):
                data[group].visititems(lambda name, obj: fields.update(
                    {f"{group}/{name}": {"shape": list(obj.shape), "dtype": str(obj.dtype)}}
                    if isinstance(obj, h5py.Dataset) else {}))
            return {"path": str(path), "frame_count": len(data["frames"]), "fields": fields}
    if path.suffix == ".json":
        value = read_json(path)
        return {"path": str(path), "type": type(value).__name__, "count": len(value),
                "sample": value[:2] if isinstance(value, list) else list(value.items())[:2]}
    if path.suffix == ".csv":
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        return {"path": str(path), "count": len(rows), "sample": rows[:2]}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        first = next(container.decode(stream))
        return {"path": str(path), "fps": float(stream.average_rate),
                "frames": stream.frames, "size": [first.height, first.width],
                "codec": stream.codec_context.name, "first_pts": first.pts}


def physion_manifest(config):
    data, rows = config["data"], []
    indices = data["input_frame_indices"]
    for scenario in data["scenarios"]:
        for official_split, split in (("readout_training", "train"), ("testing", "test")):
            source_root = Path(data["root"]) / official_split / scenario
            files = sorted(source_root.rglob("*.hdf5"))
            for path in files:
                relative = path.relative_to(source_root)
                with h5py.File(path) as sample:
                    keys = sorted(sample["frames"])
                    stimulus = sample["static/stimulus_name"][()].decode()
                    # Labels are allowed to describe unseen outcomes; image access remains prefix-only.
                    label = int(any(bool(sample[f"frames/{key}/labels/target_contacting_zone"][()]) for key in keys))
                    image = Image.open(io.BytesIO(sample[f"frames/{keys[indices[0]]}/images/_img"][()].tobytes()))
                rows.append({"id": f"{scenario}/{official_split}/{relative.with_suffix('').as_posix()}",
                             "stimulus_name": stimulus, "source_relative_path": relative.as_posix(),
                             "path": str(path.resolve()), "format": "physion_hdf5",
                             "split": split, "scenario": scenario, "label": label,
                             "official_split": official_split, "frame_indices": indices,
                             "times": [i / data["fps"] for i in indices], "native_hw": [image.height, image.width],
                             "time_basis": {"kind": "nominal_frame_time", "nominal_fps": data["fps"],
                                            "measured_clock": False,
                                            "source": "https://arxiv.org/html/2106.08261v3",
                                            "paper_reported_movie_fps": 30,
                                            "paper_observed_prefix_seconds": 1.5,
                                            "provenance": "paper section2.2 reports movies rendered at30fps; times use configured frame-position scale, not HDF5 timestamps; custom prefix is not paper observed1.5s"},
                             "input_boundary": data["prefix_source"], "last_allowed_frame": indices[-1]})
    # Dev comes only from readout training, stratified within scenario and outcome.
    stratified_development(rows, data["dev_fraction"], config["seed"])
    return rows


def ssv2_manifest(config):
    data, rows = config["data"], []
    labels = read_json(data["labels"])
    for official_split, split in (("train", "train"), ("validation", "test")):
        for item in read_json(data[official_split]):
            template = item["template"].replace("[", "").replace("]", "")
            path = Path(data["videos"]) / f"{item['id']}{data['extension']}"
            rows.append({"id": str(item["id"]), "path": str(path.resolve()), "format": "video",
                         "split": split, "scenario": "ssv2", "label": int(labels[template]),
                         "official_split": official_split, "input_boundary": "entire annotated video",
                         "time_basis": {"kind": "video_pts_seconds", "measured_clock": True,
                                        "provenance": "decoded frame pts * time_base; video clock, not calibrated simulator clock"},
                         "sampling": "unique uniform indices; actual PTS; no padding by repetition"})
    stratified_development(rows, data["dev_fraction"], config["seed"])
    if data.get("pilot_per_split"):
        rows = stratified_pilot(rows, data["pilot_per_split"], config["seed"])
    return rows


def build_manifest(config):
    rows = {"physion": physion_manifest, "ssv2": ssv2_manifest}[config["benchmark"]](config)
    payload = {"benchmark": config["benchmark"], "protocol": config["protocol"],
               "scope": config["scope"], "num_classes": config["num_classes"], "rows": rows,
               "split_counts": dict(Counter(row["split"] for row in rows)), "data": config["data"],
               "class_counts": {split: dict(Counter(str(row["label"]) for row in rows if row["split"] == split))
                                for split in ("train", "dev", "test")},
               "evaluation_partition": "official validation" if config["benchmark"] == "ssv2" else "official testing"}
    write_json(config["manifest"], payload)
    print(payload["split_counts"], flush=True)
    return payload
