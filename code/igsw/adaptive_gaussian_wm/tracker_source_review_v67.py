"""Expose indexed file/episode provenance without claiming access to raw originals."""

import math

import av
from PIL import Image, ImageDraw
import torch

from .tracker_visual_review_v67 import decode_case, write_json
from .tracker_visual_review_media_v67 import rgb_image
from .video_file_decoder import VideoDecodeError


def export_source_review(case, directory, overview_frames, *, record_decode_errors=False):
    record = case["record"]
    container_info = {"adapter": record["adapter"]}
    indices = torch.linspace(0, record["frame_count"] - 1, min(overview_frames, record["frame_count"])).round().long().unique()
    rgb = decode_case(case, indices, return_error=record_decode_errors)
    if isinstance(rgb, VideoDecodeError):
        report = {"case_id": case["case_id"], "source": case["source"], "indexed_record": record,
                  "camera": case["camera"], "overview_status": "decode_failed", "error": str(rgb),
                  "overview_episode_frames": indices.tolist(), "raw_original_provenance": "unverified",
                  "selected_episode_frames": [case["first_frame"], case["last_frame"]]}
        sheet = Image.new("RGB", (1024, 96), "#202020")
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 16), "EPISODE OVERVIEW UNAVAILABLE: source decode failed; tracked clip is separate.", fill="#ffb0a0")
        draw.text((12, 48), "See source_review.json for the file path and decoder error.", fill="white")
        sheet.save(directory / "episode_overview.png")
        write_json(directory / "source_review.json", report)
        print(f"[source-review] overview_decode_failed={case['case_id']} error={rgb}", flush=True)
        return report
    if record["adapter"] != "rgb_episode_cache":
        with av.open(record["path"]) as container:
            stream = container.streams.video[0]
            container_info.update(
                {
                    "codec": stream.codec_context.name,
                    "native_hw": [stream.height, stream.width],
                    "average_fps": float(stream.average_rate)
                    if stream.average_rate
                    else None,
                    "duration_seconds": float(stream.duration * stream.time_base)
                    if stream.duration is not None
                    else None,
                    "stream_start_pts": stream.start_time,
                    "note": "the file may pack several episodes; this is not proof of an unedited original",
                }
            )
    tile_width, tile_height, columns = 256, 190, 4
    sheet = Image.new(
        "RGB",
        (columns * tile_width, math.ceil(len(indices) / columns) * tile_height),
        "#202020",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (frame, image) in enumerate(zip(indices.tolist(), rgb)):
        thumbnail = rgb_image(image)
        thumbnail.thumbnail((tile_width, tile_height - 30))
        left, top = index % columns * tile_width, index // columns * tile_height
        sheet.paste(thumbnail, (left + (tile_width - thumbnail.width) // 2, top))
        selected = case["first_frame"] <= frame <= case["last_frame"]
        draw.text(
            (left + 3, top + tile_height - 27),
            f"episode frame={frame} t={frame / record['fps']:.2f}s",
            fill="white",
        )
        if selected:
            draw.text(
                (left + 3, top + tile_height - 14),
                "inside tracked 10s window",
                fill="#70ee90",
            )
    sheet.save(directory / "episode_overview.png")
    report = {
        "case_id": case["case_id"],
        "source": case["source"],
        "indexed_record": record,
        "container": container_info,
        "selected_episode_frames": [case["first_frame"], case["last_frame"]],
        "selected_file_frames": [
            record["frame_offset"] + case["first_frame"],
            record["frame_offset"] + case["last_frame"],
        ],
        "episode_file_frame_range": [
            record["frame_offset"],
            record["frame_offset"] + record["frame_count"] - 1,
        ],
        "episode_seconds": (record["frame_count"] - 1) / record["fps"],
        "overview_episode_frames": indices.tolist(),
        "overview_status": "decoded",
        "overview_is_uniform_inspection_only": True,
        "raw_original_provenance": "unverified",
        "tracking_input": "consecutive indexed frames, not this overview or the 400ms display branch",
        "camera": case["camera"],
        "case_manifest_override": "supply corrected complete case records through existing CASE_MANIFEST; no raw path inferred",
    }
    write_json(directory / "source_review.json", report)
    return report
