#!/usr/bin/env python3
"""Render actual RGB windows and their source mapping without loading a model."""

import argparse
from html import escape
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igsw.object_event_wm.codec import letterbox_video
from igsw.object_event_wm.data import build_video_dataset, collate_object_event_video, write_json
from igsw.object_event_wm.evaluation import balanced_indices, write_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--partition", default="development")
    parser.add_argument("--items-per-source", type=int, default=2)
    parser.add_argument("--preview-size", type=int, default=512)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset = build_video_dataset(args.manifest, partition=args.partition)
    records, excluded, sections = [], [], []
    torch.set_num_threads(4)
    for index in balanced_indices(dataset, args.items_per_source):
        sample = dataset[index]
        if "decode_error" in sample:
            excluded.append(sample)
            continue
        frames, _ = letterbox_video(collate_object_event_video([sample]), args.preview_size)
        frames = frames[0]
        case_dir = out / sample["case_id"]
        unit = sample.get("time_unit", "seconds")
        write_video(case_dir / "input_sequence.mp4", frames, sample["times"], unit)
        rows = []
        requested_times = ((-3., -1.5, 0., 1., 3., 5.) if unit == "seconds" else
                           np.linspace(float(sample["times"][0]), float(sample["times"][-1]), 6))
        for requested in requested_times:
            frame = int((sample["times"]-requested).abs().argmin())
            array = (frames[frame].permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
            preview = Image.fromarray(array).resize((256, 256))
            panel = Image.new("RGB", (256, 280), "white")
            panel.paste(preview)
            time_label = f'{float(sample["times"][frame]):+.3f}s' if unit == "seconds" else f'step {int(sample["times"][frame])}'
            ImageDraw.Draw(panel).text((8, 260), f'{time_label} frame={int(sample["frame_indices"][frame])}', fill="black")
            rows.append(panel)
        contact = Image.new("RGB", (256*len(rows), 280), "white")
        for column, panel in enumerate(rows):
            contact.paste(panel, (column*256, 0))
        contact.save(case_dir / "timeline.jpg")
        record = {key: sample[key] for key in ("case_id", "source", "camera", "media_path", "frame_offset",
                   "fps", "episode_index", "partition", "history_frames", "instruction", "language_provenance", "time_source")}
        for key in ("times", "frame_indices", "frame_valid", "native_hw"):
            record[key] = sample[key].tolist()
        record["preview_size"] = args.preview_size
        record["time_unit"] = unit
        record["preview_fps_not_physical"] = 5 if unit == "sampled_frame" else None
        record["observation_record"] = sample.get("observation_record")
        records.append(record)
        write_json(case_dir / "source.json", record)
        case_id = escape(sample["case_id"], quote=True)
        sections.append(f'<section><h2>{escape(sample["source"])} / {case_id}</h2>'
                        f'<p>Camera: {escape(sample["camera"])}; native: {record["native_hw"]}; '
                        f'Physical FPS: {record["fps"]}; history: {record["history_frames"]} frames; time unit: {unit}</p>'
                        f'<p>{escape(sample["instruction"] or "No episode-level instruction at this window")}</p>'
                        f'<img src="{case_id}/timeline.jpg" alt="History and future at recorded times">'
                        f'<video src="{case_id}/input_sequence.mp4" controls preload="none"></video>'
                        f'<p><a href="{case_id}/source.json">Frame times and original file mapping</a></p></section>')
        print(json.dumps({"event": "rgb_window_reviewed", "case_id": sample["case_id"],
                          "source": sample["source"], "camera": sample["camera"]}), flush=True)
    write_json(out / "report.json", {"manifest": str(Path(args.manifest).resolve()),
               "partition": args.partition, "cases": records, "excluded": excluded,
               "pretrained_model_used": False, "preview_only_resize": args.preview_size})
    (out / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Object Event RGB Windows</title><style>'
        'body{font:16px system-ui;margin:24px;color:#202124;background:#fff;letter-spacing:0}'
        'section{padding:24px 0;border-top:1px solid #ddd}h2{font-size:20px;overflow-wrap:anywhere}'
        'img{width:100%;height:auto}video{width:512px;max-width:100%}p{overflow-wrap:anywhere}'
        '</style><h1>Object Event RGB Windows</h1>'+''.join(sections)+'</html>')


if __name__ == "__main__":
    main()
