"""Render the actual loader sample, not semantic-role-colored proposals."""

import html
from pathlib import Path

from PIL import Image, ImageDraw
import torch

from .grounded_motion_dataset_v68 import GroundedMotionDatasetV68
from .tracker_visual_review_media_v67 import rgb_image, write_video
from .tracker_visual_review_v67 import write_json


CURRENT_COLOR = (40, 210, 240)
SHORT_COLOR = (255, 210, 45)
LONG_COLOR = (245, 120, 225)


def render_motion_selection(directory, rgb, native, target_valid, selected_ids, width, fps):
    xy = native["tracks"][:, selected_ids].numpy()
    valid = target_valid[:, selected_ids].numpy()
    height, native_width = rgb.shape[-2:]
    trails = Image.new("RGBA", (native_width, height))
    trail_draw = ImageDraw.Draw(trails)
    def frames():
        for frame, original in enumerate(rgb):
            if frame:
                for before, after in zip(xy[frame - 1, valid[frame - 1] & valid[frame]],
                                         xy[frame, valid[frame - 1] & valid[frame]]):
                    trail_draw.line((*before.tolist(), *after.tolist()), fill=(*SHORT_COLOR, 255), width=2)
            image = rgb_image(original)
            image.paste(trails, (0, 0), trails)
            draw = ImageDraw.Draw(image)
            for x, y in xy[frame, valid[frame]].tolist():
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=SHORT_COLOR)
            draw.rectangle((0, 0, image.width, 38), fill=(12, 12, 12))
            draw.text((7, 4), f"ALL-ROLE TOP MOTION: {len(selected_ids)} tracks; frame={frame}", fill="white")
            draw.text((7, 21), "Yellow = selection only, NOT object / robot class", fill="white")
            yield image.resize((width, round(image.height * width / image.width)))
    write_video(Path(directory) / "selected_motion.mp4", frames(), fps)


def render_training_sample(sample, directory, width=640):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    h, w = sample["native_image_hw"].tolist()
    xy = (sample["coordinates"] + 1) * .5 * torch.tensor([w - 1, h - 1])
    current = sample["point_valid"][3]
    short, long = sample["target_valid"].unbind(0)
    titles = ["HISTORY 1", "HISTORY 2", "HISTORY 3", "CURRENT: appearance queries",
              "+1 SECOND: correspondence / transport", "+3 SECONDS: dynamics transport only"]
    counts = {"current_queries": int(current.sum()), "short_supervised": int(short.sum()),
              "long_supervised": int(long.sum()), "point_budget": len(sample["point_ids"])}
    panels = []
    history_panels = []
    frame_rows = []
    for time in range(6):
        original = rgb_image(sample["video_rgb"][time])
        original.save(directory / f"rgb_{time}.png")
        image = original.copy()
        draw = ImageDraw.Draw(image)
        mask, tint = ((current, CURRENT_COLOR) if time == 3 else
                      ((short, SHORT_COLOR) if time == 4 else (long, LONG_COLOR)))
        if time >= 3:
            for point in torch.where(mask)[0].tolist():
                x, y = xy[time, point].tolist()
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=tint)
        scaled_height = round(h * width / w)
        panel = Image.new("RGB", (width, scaled_height + 54), "#141414")
        panel.paste(image.resize((width, scaled_height)), (0, 54))
        label = ImageDraw.Draw(panel)
        active = time >= 4 or bool(sample["history_valid"][time])
        title = titles[time] if active else f"HISTORY {time + 1}: MASKED OUT (not observed)"
        label.text((7, 5), title, fill="white")
        label.text((7, 23), f"frame={int(sample['frame_indices'][time])}; "
                   f"time={float(sample['frame_times'][time]):.3f}s; drawn={int(mask.sum()) if time >= 3 else 0}", fill="white")
        panels.append(panel)
        if time < 4:
            raw_panel = Image.new("RGB", panel.size, "#141414")
            raw_panel.paste(original.resize((width, scaled_height)), (0, 54))
            ImageDraw.Draw(raw_panel).text((7, 5), title, fill="white")
            history_panels.append(raw_panel)
        frame_rows.append({"frame_index": int(sample["frame_indices"][time]), "rgb": f"rgb_{time}.png",
                           "observed_by_student": bool(time < 4 and active), "label": title})
    sheet = Image.new("RGB", (width * 3, panels[0].height * 2), "#141414")
    for time, panel in enumerate(panels):
        sheet.paste(panel, ((time % 3) * width, (time // 3) * panel.height))
    sheet.save(directory / "training_sample.png")
    history = Image.new("RGB", (width * 2, panels[0].height * 2), "#141414")
    for time, panel in enumerate(history_panels):
        history.paste(panel, ((time % 2) * width, (time // 2) * panel.height))
    history.save(directory / "student_input.png")
    valid_ids = torch.where(sample["point_ids"] >= 0)[0].tolist()
    rows = [{"loader_point_index": i, "track_id": int(sample["point_ids"][i]),
             "coordinates_native_px": xy[:, i].tolist(), "point_valid": sample["point_valid"][:, i].tolist(),
             "selected_for_motion": bool(sample["motion_mask"][i]), "short_target_valid": bool(short[i]),
             "long_target_valid": bool(long[i]),
             "tracker_observability_aux_target": sample["point_valid"][4:6, i].tolist(),
             "role_label": int(sample["roles"][i]), "region_label": int(sample["region_ids"][i])} for i in valid_ids]
    report = {"case_id": sample["case_id"], "sample_index": sample["sample_index"],
              "epoch": sample["data_epoch"], "seed": sample["sampler_seed"], "counts": counts,
              "selection_policy": sample["selection_policy"], "frames": frame_rows, "points": rows,
              "history_valid": sample["history_valid"].tolist(),
              "future_rgb_is_training_target_only": True, "rgb_reconstruction_loss": False,
              "appearance_teacher_validity": "loader points shown; frozen feature teacher additionally applies its own valid mask",
              "drawing": "cyan=current auxiliary queries; yellow=short targets; pink=long targets; no class colors"}
    write_json(directory / "training_sample.json", report)
    return report


def write_training_review(manifest, output, *, points=256, seed=17, epochs=(0,), width=640):
    """The same dataset/index/epoch path used by the training DataLoader."""
    dataset = GroundedMotionDatasetV68(manifest, points=points, seed=seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    body = ["<!doctype html><meta charset='utf-8'><title>Actual training samples</title>",
            "<style>body{font:16px system-ui;max-width:1500px;margin:24px auto}img{max-width:100%}section{border-top:1px solid #bbb;padding:20px 0}</style>",
            "<h1>实际训练 loader 样本</h1><p>直接调用训练 Dataset，不使用另一套展示采样。颜色只表示监督用途，不表示物体或夹爪。</p>",
            "<p>青色：当前外观辅助查询；黄色：+1秒轨迹监督；粉色：+3秒轨迹监督。后者只在 Dynamics 阶段使用。"
            "无效坐标不画点；JSON保留逐点有效性。未来图像不是 student history 输入，也没有RGB重建loss。</p>",
            f"<p>points={points}; seed={seed}; epochs={list(epochs)}。训练更换epoch会重新采样，图中是指定epoch的确切样本。</p>"]
    summaries = []
    for index, entry in enumerate(dataset.entries):
        if not entry["rendered"]:
            continue
        for epoch in epochs:
            directory = output / f"sample_{index:06d}_epoch{epoch}"
            report = render_training_sample(dataset[(index, epoch)], directory, width)
            relative = directory.name
            summaries.append({"path": relative, "case_id": report["case_id"], "counts": report["counts"],
                              "sample_index": index, "epoch": epoch})
            body.append(f"<section><h2>{html.escape(report['case_id'])}: epoch {epoch}</h2>"
                        f"<p>{html.escape(str(report['counts']))}</p><img loading='lazy' src='{relative}/training_sample.png'>"
                        f"<p><a href='{relative}/student_input.png'>Student实际历史输入（无点叠加）</a> · "
                        f"<a href='{relative}/training_sample.json'>实际point IDs、坐标、mask</a></p></section>")
            print(f"[training-data-review] sample={index} epoch={epoch} case={entry['case_id']} {report['counts']}", flush=True)
    write_json(output / "samples.json", {"manifest": str(Path(manifest).resolve()), "points": points,
               "seed": seed, "epochs": list(epochs), "samples": summaries})
    (output / "index.html").write_text("\n".join(body), encoding="utf-8")
    return summaries
