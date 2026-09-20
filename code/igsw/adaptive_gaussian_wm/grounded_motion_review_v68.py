"""Inspection of the exact offline targets, camera alternatives and rejection reasons."""

import html
import json
from collections import Counter
from pathlib import Path
import zipfile

from PIL import Image, ImageDraw
import torch

from .grounded_motion_sources_v68 import alternate_views
from .grounded_tracker_export_v67 import styled_tracks
from .grounded_tracker_selection_media_v67 import render_object_selection
from .tracker_source_review_v67 import export_source_review
from .tracker_visual_review_media_v67 import draw_points, rgb_image, write_video
from .tracker_visual_review_v67 import decode_case, write_json
from .video_file_decoder import VideoDecodeError


def camera_overview(case, directory):
    views = (alternate_views(case) or [case]) if case["source"] == "robomind" else [case]
    tiles, mappings = [], []
    for view in views:
        indices = torch.linspace(0, view["record"]["frame_count"] - 1, 6).round().long().unique()
        video = decode_case(view, indices, return_error=True)
        strip = Image.new("RGB", (256 * len(indices), 224), "#202020")
        draw = ImageDraw.Draw(strip)
        if isinstance(video, VideoDecodeError):
            draw.text((3, 5), view["camera"], fill="white")
            draw.text((3, 50), "CAMERA OVERVIEW UNAVAILABLE: decode failed; see camera_mapping.json", fill="#ffb0a0")
            tiles.append(strip)
            mappings.append({"camera": view["camera"], "record": view["record"], "selected": view["camera"] == case["camera"],
                             "overview_status": "decode_failed", "error": str(video)})
            print(f"[camera-review] overview_decode_failed={view['record']['path']} error={video}", flush=True)
            continue
        for column, frame in enumerate(video):
            thumb = rgb_image(frame)
            thumb.thumbnail((256, 190))
            strip.paste(thumb, (column * 256, 30))
            draw.text((column * 256 + 3, 208), f"frame {int(indices[column])}", fill="white")
        draw.text((3, 5), view["camera"] + (" SELECTED" if view["camera"] == case["camera"] else ""), fill="white")
        tiles.append(strip)
        mappings.append({"camera": view["camera"], "record": view["record"], "selected": view["camera"] == case["camera"], "overview_status": "decoded"})
    if tiles:
        sheet = Image.new("RGB", (max(t.width for t in tiles), sum(t.height for t in tiles)))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, (0, i * 224))
        sheet.save(directory / "camera_overview.png")
    write_json(directory / "camera_mapping.json", {"case_id": case["case_id"], "views": mappings,
               "camera_role_evidence": case.get("camera_evidence"), "visual_confirmation": "pending user review"})


def render_motion_data(directory, rgb, native, queries, background, report, case, args, target_valid):
    export_source_review(case, directory, args.episode_overview_frames, record_decode_errors=True)
    camera_overview(case, directory)
    target_display = {**native, "visibility": native["visibility"] & target_valid}
    render_object_selection(directory, rgb, target_display, case, queries, report, args.display_width)
    styled = styled_tracks(native, queries, torch.arange(len(queries["xy"])))
    def frames():
        for index, image in enumerate(rgb):
            overlay = draw_points(rgb_image(image), styled, index, case["record"]["fps"], queries["labels"],
                                  "ALL raw points; not the training target mask")
            yield overlay.resize((args.display_width, round(overlay.height * args.display_width / overlay.width)))
    write_video(directory / "all_points_context.mp4", frames(), case["record"]["fps"])
    reference = background["reference"]
    write_json(directory / "background_fit.json", {"frames": background["frames"], "meaning": background["meaning"]})
    ref_queries = {"metadata": [{"role": "scene_context"} for _ in range(reference["tracks"].shape[1])]}
    ref_styled = styled_tracks(reference, ref_queries, torch.arange(reference["tracks"].shape[1]))
    def reference_frames():
        for index, image in enumerate(rgb):
            overlay = draw_points(rgb_image(image), ref_styled, index, case["record"]["fps"],
                ["reference"] * reference["tracks"].shape[1], f"Background reference; fit usable={bool(background['valid'][index])}")
            yield overlay.resize((args.display_width, round(overlay.height * args.display_width / overlay.width)))
    write_video(directory / "background_reference.mp4", reference_frames(), case["record"]["fps"])


def write_data_gallery(out, entries, selection, *, index_name="index.html", manifest_name="training_manifest.json"):
    esc = html.escape
    body = ["<!doctype html><meta charset='utf-8'><title>Grounded motion data v68</title>",
            "<style>body{font:16px system-ui;max-width:1250px;margin:24px auto}video{width:48%;vertical-align:top}img{max-width:100%}section{border-top:1px solid #bbb;padding:24px 0}pre{white-space:pre-wrap}</style>",
            "<h1>V68 离线运动数据</h1><p>主视频只画实际训练目标；全部点和机械臂上下文另列。75%是采样预算，不是准确率。</p>",
            f"<p><a href='selection.json'>来源/视角排除记录</a> · <a href='{esc(manifest_name)}'>当前展示 manifest</a></p>"]
    if index_name != "index.html":
        body.append("<p>PARTIAL REVIEW: only completed cases are shown; this is not the complete requested dataset.</p>")
    for entry in entries:
        if not entry["rendered"]:
            continue
        path = esc(str(Path(entry["path"]).parent))
        body.append(f"<section><h2>{esc(entry['case_id'])}</h2><p>{esc(entry['camera'])} · targets={entry['object_targets']}</p>")
        if entry["rendered"]:
            body.append(f"<h3>当前选择及同 episode 可用视角</h3><img src='{path}/camera_overview.png'>"
                        f"<p><a href='{path}/camera_mapping.json'>相机实际路径/时间映射</a> · <a href='{path}/source_review.json'>源文件映射</a></p>"
                        f"<h3>整段 episode 概览</h3><img src='{path}/episode_overview.png'>"
                        f"<h3>物体候选筛选前 / 实际75%目标</h3><video controls src='{path}/object_before_topk.mp4'></video>"
                        f"<video controls src='{path}/object_after_topk.mp4'></video>"
                        f"<h3>背景参考 / 全部点和上下文</h3><video controls src='{path}/background_reference.mp4'></video>"
                        f"<video controls src='{path}/all_points_context.mp4'></video>")
            refinement = json.loads((out / Path(entry["path"]).parent / "refinement.json").read_text())
            body.append("<h3>运动点与背景负提示重新得到的 SAM 支持区域</h3>")
            for view in refinement["views"]:
                body.append(f"<img loading='lazy' src='{path}/{esc(view['overlay'])}'>")
        body.append(f"<p><a href='{path}/motion_filter.json'>逐点筛选原因</a> · <a href='{path}/refinement.json'>SAM正负提示</a></p></section>")
    (out / index_name).write_text("\n".join(body), encoding="utf-8")


def write_review_bundle(out, entries, metadata, *, name="review_bundle.zip"):
    # Only reviewed cases contribute media; the manifest still lists all training clips.
    artifacts = set(metadata)
    for entry in entries:
        if entry["rendered"]:
            artifacts.update((out / Path(entry["path"]).parent).rglob("*"))
    with zipfile.ZipFile(out / name, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(artifacts):
            if path.is_file() and path.suffix in (".html", ".png", ".mp4", ".json"):
                archive.write(path, path.relative_to(out))


def upload_data(args, out, entries, configuration):
    if args.wandb_mode == "disabled":
        return
    import os
    import wandb
    os.environ.pop("WANDB_RUN_ID", None)
    os.environ.pop("WANDB_RESUME", None)
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=args.wandb_name,
                     group="grounded-motion-data-v68", job_type="offline-data", mode=args.wandb_mode,
                     dir=args.wandb_dir, config=configuration)
    table = wandb.Table(columns=["case", "source", "camera", "object_targets", "reference_usable_fraction", "targets", "cameras"])
    table_counts = Counter()
    failure_path = out / "decode_failures.json"
    failures = json.loads(failure_path.read_text())["cases"] if failure_path.is_file() else {}
    for row in entries:
        if args.render and not row["rendered"]:
            continue
        if not row["rendered"] and table_counts[row["source"]] >= 64:
            continue
        table_counts[row["source"]] += 1
        directory = out / Path(row["path"]).parent
        table.add_data(row["case_id"], row["source"], row["camera"], row["object_targets"], row["background_valid_fraction"],
            wandb.Video(str(directory / "object_after_topk.mp4"), format="mp4") if row["rendered"] else None,
            wandb.Image(str(directory / "camera_overview.png")) if row["rendered"] and (directory / "camera_overview.png").is_file() else None)
    run.log({"motion_data/cases": table, "motion_data/clips": len(entries),
             "motion_data/table_is_sample": sum(table_counts.values()) < len(entries),
             "motion_data/table_clips": sum(table_counts.values()),
             "motion_data/review_clips": sum(e["rendered"] for e in entries),
             "motion_data/decode_skipped_clips": len(failures),
             "motion_data/decode_failed_files": len({row["path"] for row in failures.values()}),
             "motion_data/target_points": sum(e["object_targets"] for e in entries),
             **{f"motion_data/clips_{source}": count for source, count in Counter(e["source"] for e in entries).items()}})
    artifact = wandb.Artifact(args.wandb_name, type="motion-data-review")
    for name in ("selection.json", "training_manifest.json", "decode_failures.json", "index.html", "review_bundle.zip"):
        if (out / name).is_file():
            artifact.add_file(str(out / name))
    run.log_artifact(artifact)
    run.finish()
