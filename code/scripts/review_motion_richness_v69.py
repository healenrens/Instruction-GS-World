#!/usr/bin/env python3
"""Video-backed motion diagnostics and blind comparison forms; never changes training sampling."""

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image, ImageDraw
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import write_json, decode_case
from igsw.adaptive_gaussian_wm.object_video_manifest_v69 import load_object_video_manifest_v69, resolve_object_video_case_v69
from igsw.adaptive_gaussian_wm.tracker_visual_review_media_v67 import rgb_image, write_video
from igsw.adaptive_gaussian_wm.video_file_decoder import VideoDecodeError
from igsw.adaptive_gaussian_wm.swanlab_tracking_v69 import (
    add_swanlab_arguments, start_swanlab_v69, log_values_v69, log_table_v69,
    log_video_v69, log_evidence_v69, set_results_v69,
)


def quantiles(value):
    value = value.float()
    if not len(value):
        return {"p50": None, "p90": None, "p95": None}
    return dict(zip(("p50", "p90", "p95"), value.quantile(torch.tensor([.5, .9, .95])).tolist()))


def window_metrics(data, first, stop):
    evidence = data["motion_evidence"]
    valid = evidence["valid"][first:stop].bool()
    compensated = evidence["compensated_coordinates"][first:stop].float()
    raw = data["native"]["tracks"][first:stop].float()
    spans, raw_spans, paths, selected = [], [], [], []
    fps = data["case"]["record"]["fps"]
    for point in range(valid.shape[1]):
        good = valid[:, point]
        if int(good.sum()) < 2:
            continue
        span = (compensated[good, point].quantile(.95, dim=0)-compensated[good, point].quantile(.05, dim=0)).norm()
        raw_span = (raw[good, point].quantile(.95, dim=0)-raw[good, point].quantile(.05, dim=0)).norm()
        adjacent = good[1:] & good[:-1]
        length = (compensated[1:, point]-compensated[:-1, point]).norm(dim=-1)[adjacent].sum()
        spans.append(span)
        raw_spans.append(raw_span)
        paths.append(length)
        if span >= evidence["threshold_px"][point]:
            selected.append(point)
    spans = torch.stack(spans) if spans else torch.empty(0)
    raw_spans = torch.stack(raw_spans) if raw_spans else torch.empty(0)
    paths = torch.stack(paths) if paths else torch.empty(0)
    h, w = data["case"]["height"], data["case"]["width"]
    diagonal = float((h*h+w*w)**.5)
    cells = set()
    for point in selected:
        xy = raw[valid[:, point], point] / torch.tensor([w, h]) * 16
        cells.update(map(tuple, xy.floor().long().clamp(0, 15).tolist()))
    return {"first_local_frame": first, "stop_local_frame": stop,
            "start_seconds": first/fps, "stop_seconds": (stop-1)/fps,
            "measurable_points": len(spans), "candidate_moving_points": len(selected),
            "compensated_span_px": quantiles(spans), "raw_span_px": quantiles(raw_spans),
            "relative_span_image_diagonal": quantiles(spans/diagonal), "observed_path_length_px": quantiles(paths),
            "motion_covered_grid_fraction": len(cells)/256,
            "background_fit_fraction": float(data["background_motion"]["valid"][first:stop].float().mean()),
            "moving_point_ids": selected, "score_is_not_object_motion_ground_truth": True}


def render_window(case, data, metrics, directory):
    first, stop = metrics["first_local_frame"], metrics["stop_local_frame"]
    fps = case["record"]["fps"]
    indices = torch.arange(first, stop, max(1, round(fps/5)))
    rgb = decode_case(case, indices+case["first_frame"], return_error=True)
    if isinstance(rgb, VideoDecodeError):
        return {"status": "decode_failed", "error": str(rgb)}
    native = data["native"]
    relay = data["relay_evidence"]["prediction"]
    selected = metrics["moving_point_ids"]
    def raw_frames():
        for image in rgb:
            yield rgb_image(image)
    def tracked_frames(show_relay=False):
        trail = Image.new("RGBA", (rgb.shape[-1], rgb.shape[-2]))
        trail_draw = ImageDraw.Draw(trail)
        previous = first
        for image, frame in zip(rgb, indices.tolist()):
            for point in selected:
                for left in range(previous+1, frame+1):
                    if bool(native["visibility"][left-1:left+1, point].all() & native["in_bounds"][left-1:left+1, point].all()):
                        trail_draw.line(tuple(native["tracks"][left-1:left+1, point].flatten().tolist()), fill=(255, 215, 0, 180), width=1)
            previous = frame
            canvas = Image.alpha_composite(rgb_image(image).convert("RGBA"), trail).convert("RGB")
            draw = ImageDraw.Draw(canvas)
            for point in selected:
                if bool(native["visibility"][frame, point] & native["in_bounds"][frame, point]):
                    x, y = native["tracks"][frame, point].tolist()
                    draw.ellipse((x-2, y-2, x+2, y+2), fill="#ffd700")
                if show_relay and bool(relay["visibility"][frame, point] & relay["in_bounds"][frame, point]):
                    x, y = relay["tracks"][frame, point].tolist()
                    draw.ellipse((x-2, y-2, x+2, y+2), fill="#00dfff")
            if show_relay:
                draw.text((4, 4), "yellow=first tracker; cyan=relay; agreement is not ground truth", fill="white", stroke_width=2, stroke_fill="black")
            yield canvas
    directory.mkdir(parents=True, exist_ok=True)
    write_video(directory / "raw.mp4", raw_frames(), 5)
    write_video(directory / "tracks.mp4", tracked_frames(), 5)
    write_video(directory / "relay.mp4", tracked_frames(True), 5)
    return {"status": "rendered", "raw": str(directory / "raw.mp4"), "tracks": str(directory / "tracks.mp4"), "relay": str(directory / "relay.mp4")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--items", type=int, default=400)
    p.add_argument("--visualize", type=int, default=400)
    p.add_argument("--window_seconds", type=float, default=2.)
    p.add_argument("--stride_seconds", type=float, default=2.)
    p.add_argument("--seed", type=int, default=17)
    add_swanlab_arguments(p, default_name="object_video_v69_motion_richness_review")
    args = p.parse_args()
    manifest = load_object_video_manifest_v69(args.manifest)
    root, out = Path(manifest["root"]), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    entries = list(manifest["entries"])
    random.Random(args.seed).shuffle(entries)
    entries = entries[:args.items] if args.items else entries
    rows, pairs = [], []
    body = ["<!doctype html><meta charset='utf-8'><title>Motion evidence review</title>",
            "<style>body{font:16px system-ui;max-width:1500px;margin:24px auto}video{width:48%}section{border-top:1px solid #aaa;padding:16px 0}pre{white-space:pre-wrap}</style>",
            "<h1>Motion evidence: raw video and measured tracks</h1><p>No composite richness score and no training sampling changes. Review raw clips before opening diagnostic numbers. Yellow is a threshold-selected candidate for this diagnostic, NOT the training top-75% pool or an object label; cyan is relay tracking.</p>"]
    for ordinal, entry in enumerate(entries):
        data = torch.load(root / entry["path"], map_location="cpu", weights_only=False)
        data["case"] = resolve_object_video_case_v69(data["case"], root)
        fps = data["case"]["record"]["fps"]
        length = round(args.window_seconds*fps)+1
        stride = max(1, round(args.stride_seconds*fps))
        case_rows = []
        for first in range(0, len(data["native"]["tracks"])-length+1, stride):
            row = {"case": entry["case_id"], "source": entry["source"], "window_id": f"{entry['case_id']}_f{first}",
                   **window_metrics(data, first, first+length)}
            if ordinal < args.visualize:
                row["media"] = render_window(data["case"], data, row, out / "videos" / row["window_id"])
            case_rows.append(row)
            rows.append(row)
            body.append(f"<section><h2>{html.escape(row['window_id'])}</h2>")
            if row.get("media", {}).get("status") == "rendered":
                for name in ("raw", "tracks", "relay"):
                    relative = Path(row["media"][name]).relative_to(out).as_posix()
                    body.append(f"<video controls preload='none' src='{html.escape(relative)}'></video>")
            body.append(f"<details><summary>Diagnostic values</summary><pre>{html.escape(json.dumps(row, indent=2))}</pre></details></section>")
        for left, right in zip(case_rows[:-1], case_rows[1:]):
            pairs.append({"left": left["window_id"], "right": right["window_id"], "source": entry["source"],
                          "partition": entry["partition"],
                          "winner": None, "reason": "", "confound": None,
                          "allowed_winners": ["left", "right", "equal", "uncertain"],
                          "confound_examples": ["camera_motion", "arm_only", "small_object", "tracker_drift", "occlusion"]})
        print(f"[motion-richness-v69] {ordinal+1}/{len(entries)} case={entry['case_id']} windows={len(case_rows)}", flush=True)
    report = {"status": "diagnostics_only_pending_human_review", "manifest": args.manifest, "rows": rows,
              "window_seconds": args.window_seconds, "clip_counts": dict(Counter(e["source"] for e in entries)),
              "sampling_modified": False, "same_clip_windows_do_not_measure_unsampled_episode_intervals": True}
    write_json(out / "report.json", report)
    write_json(out / "blind_comparisons.json", {"pairs": pairs})
    (out / "index.html").write_text("\n".join(body), encoding="utf-8")
    run = start_swanlab_v69(args, group="object-video-sequence-v69", job_type="motion-evidence", config=vars(args))
    if run is not None:
        table_rows = []
        for row in rows:
            media = row.get("media", {})
            table_rows.append([row["case"], row["source"], row["window_id"], row["start_seconds"], row["measurable_points"],
                               row["compensated_span_px"]["p90"], row["relative_span_image_diagonal"]["p90"],
                               row["motion_covered_grid_fraction"], media.get("raw", ""), media.get("tracks", "")])
            if media.get("status") == "rendered":
                for name in ("raw", "tracks", "relay"):
                    log_video_v69(run, f"motion_evidence/{row['case']}/{row['window_id']}/{name}", media[name])
        log_table_v69(run, "motion_evidence/windows",
                      ["case", "source", "window", "start_seconds", "measurable_points", "span_p90_px",
                       "relative_span_p90", "grid_coverage", "raw", "tracks"], table_rows)
        log_values_v69(run, {"motion_evidence/sampling_modified": False})
        set_results_v69(run, {"status": report["status"], "clip_counts": report["clip_counts"], "sampling_modified": False})
        log_evidence_v69(run, [out / "report.json", out / "blind_comparisons.json", out / "index.html"], base_path=out)
        run.finish()
    print(f"[motion-richness-v69] review={out / 'index.html'} comparisons={out / 'blind_comparisons.json'}", flush=True)


if __name__ == "__main__":
    main()
