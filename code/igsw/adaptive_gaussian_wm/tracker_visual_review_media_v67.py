"""Videos with stable point IDs, unfiltered visibility, and fixed native crops."""

from __future__ import annotations

import colorsys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def color(point):
    rgb = colorsys.hsv_to_rgb((point * 0.61803398875) % 1.0, 0.85, 1.0)
    return tuple(round(value * 255) for value in rgb)


def rgb_image(frame):
    return Image.fromarray(frame.permute(1, 2, 0).numpy())


def write_video(path, frames, fps, frame_times=None):
    path = Path(path)
    temporary = path.with_name(path.stem + ".tmp.mp4")
    with av.open(str(temporary), mode="w") as container:
        stream = None
        for index, image in enumerate(frames):
            if stream is None:
                width, height = image.size
                stream = container.add_stream(
                    "libx264", rate=Fraction(str(fps)).limit_denominator(10000)
                )
                stream.width, stream.height = width + width % 2, height + height % 2
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "20", "preset": "fast"}
                if frame_times is not None:
                    stream.time_base = Fraction(1, 90000)
                    stream.codec_context.time_base = Fraction(1, 90000)
            padded = Image.new("RGB", (stream.width, stream.height))
            padded.paste(image, (0, 0))
            frame = av.VideoFrame.from_ndarray(np.asarray(padded), format="rgb24")
            if frame_times is not None:
                frame.time_base = Fraction(1, 90000)
                frame.pts = round(float(frame_times[index]) * 90000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    temporary.replace(path)


def draw_points(image, prediction, frame, fps, labels, title, trails_seconds=0.25):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    coordinates = prediction["tracks"]
    visible = prediction["visibility"]
    indices = prediction["frame_indices"]
    supplied = int((prediction["query_local_frames"] == frame).sum())
    current_frame = int(indices[frame])
    earlier = [
        i
        for i in range(frame + 1)
        if current_frame - int(indices[i]) <= trails_seconds * fps
    ]
    draw.rectangle((0, 0, width, 37), fill=(12, 12, 12))
    draw.text(
        (7, 4),
        f"{title} | frame={current_frame} t={current_frame / fps:.3f}s",
        fill="white",
    )
    suffix = f" | {supplied} supplied query points" if supplied else ""
    draw.text(
        (7, 20),
        "filled=visible; ring=not visible; border X=offscreen" + suffix,
        fill="white",
    )
    for point, xy in enumerate(coordinates[frame]):
        x, y = xy.tolist()
        if not np.isfinite([x, y]).all():
            continue
        tint = color(point)
        for previous, following in zip(earlier, earlier[1:]):
            if bool(visible[previous, point] & visible[following, point]):
                line = [
                    tuple(coordinates[t, point].tolist()) for t in (previous, following)
                ]
                if np.isfinite(line).all():
                    draw.line(line, fill=tint, width=2)
        inside = 0 <= x < width and 0 <= y < height
        if inside:
            box = (x - 3, y - 3, x + 3, y + 3)
            draw.ellipse(
                box,
                outline=tint,
                fill=tint if bool(visible[frame, point]) else None,
                width=2,
            )
        else:
            x, y = min(max(x, 5), width - 6), min(max(y, 42), height - 6)
            draw.line((x - 3, y - 3, x + 3, y + 3), fill=tint, width=2)
            draw.line((x - 3, y + 3, x + 3, y - 3), fill=tint, width=2)
        if len(labels) <= 32:
            draw.text(
                (x + 4, y + 3),
                str(point),
                fill=tint,
                stroke_width=1,
                stroke_fill="black",
            )
    return image


def crop_boxes(case, xy, labels):
    width, height = case["width"], case["height"]
    if labels and labels[0] != "unlabelled_grid_point":
        groups = sorted(set(labels), key=lambda label: (-labels.count(label), label))[
            :3
        ]
        centers = [
            xy[[i for i, label in enumerate(labels) if label == group]]
            .mean(dim=0)
            .tolist()
            for group in groups
        ]
    else:
        centers = [
            [width * 0.5, height * 0.5],
            [width * 0.3, height * 0.7],
            [width * 0.7, height * 0.7],
        ]
        groups = ["fixed center", "fixed lower-left", "fixed lower-right"]
    side = max(32, round(min(width, height) * 0.3))
    boxes = []
    for group_index, ((x, y), label) in enumerate(zip(centers, groups)):
        left = int(min(max(x - side / 2, 0), width - side))
        top = int(min(max(y - side / 2, 0), height - side))
        boxes.append(
            {
                "box": [left, top, left + side, top + side],
                "label": f"group {group_index + 1}"
                if labels[0] != "unlabelled_grid_point"
                else label,
                "annotation_label": label,
            }
        )
    return boxes


def panel(image, boxes, width, prediction, frame):
    height = round(image.height * width / image.width)
    crop_width = width // len(boxes)
    canvas = Image.new("RGB", (width, height + crop_width + 18), (20, 20, 20))
    canvas.paste(image.resize((width, height)), (0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, spec in enumerate(boxes):
        crop = image.crop(spec["box"]).resize((crop_width, crop_width))
        crop_draw = ImageDraw.Draw(crop)
        left, top, right, bottom = spec["box"]
        point_rows = prediction["tracks"][frame].tolist()
        inside_count = sum(
            left <= x < right and top <= y < bottom for x, y in point_rows
        )
        for point, (x, y) in enumerate(point_rows):
            if inside_count <= 64 and left <= x < right and top <= y < bottom:
                crop_draw.text(
                    (
                        (x - left) * crop_width / (right - left) + 4,
                        (y - top) * crop_width / (bottom - top) + 3,
                    ),
                    str(point),
                    fill=color(point),
                    stroke_width=1,
                    stroke_fill="black",
                )
        canvas.paste(crop, (index * crop_width, height))
        draw.text(
            (index * crop_width + 3, height + crop_width + 2),
            spec["label"],
            fill="white",
        )
    return canvas


def render_pair(directory, rgb, native, sampled, case, xy, labels, display_width):
    directory = Path(directory)
    fps = case["record"]["fps"]
    first = int(native["frame_indices"][0])
    boxes = crop_boxes(case, xy, labels)
    paths = {
        key: directory / f"{key}.mp4" for key in ("native", "sampled", "comparison")
    }

    def annotated(prediction, title):
        for frame, original in enumerate(prediction["frame_indices"].tolist()):
            overlay = draw_points(
                rgb_image(rgb[original - first]), prediction, frame, fps, labels, title
            )
            yield panel(overlay, boxes, display_width, prediction, frame)

    write_video(paths["native"], annotated(native, "NATIVE consecutive input"), fps)
    stride = int(sampled["frame_indices"][1] - sampled["frame_indices"][0])
    times = (sampled["frame_indices"] - first).float() / fps
    write_video(
        paths["sampled"],
        annotated(sampled, f"SAMPLED input stride={stride}"),
        fps / stride,
        frame_times=times,
    )

    def comparison_frames():
        for frame, original in enumerate(sampled["frame_indices"].tolist()):
            source_image = rgb_image(rgb[original - first])
            native_overlay = draw_points(
                source_image,
                native,
                original - first,
                fps,
                labels,
                "NATIVE at shared timestamp",
            )
            sampled_overlay = draw_points(
                source_image, sampled, frame, fps, labels, "SAMPLED at shared timestamp"
            )
            left = panel(native_overlay, boxes, display_width, native, original - first)
            right = panel(sampled_overlay, boxes, display_width, sampled, frame)
            combined = Image.new("RGB", (display_width * 2, left.height))
            combined.paste(left, (0, 0))
            combined.paste(right, (display_width, 0))
            yield combined

    write_video(
        paths["comparison"], comparison_frames(), fps / stride, frame_times=times
    )
    preview = rgb_image(rgb[case["anchor_frame"] - first])
    preview.save(directory / "anchor.png")
    return {key: str(path.name) for key, path in paths.items()}, boxes
