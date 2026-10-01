"""Native-coordinate evidence for observed State reconstruction, including controls."""

from pathlib import Path

from PIL import Image, ImageDraw

from .tracker_visual_review_media_v67 import color, rgb_image, write_video


def render_state_change_v69(batch, output, predictions, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    h, w = batch["native_hw"][0].tolist()
    scale = (batch["native_hw"][0, [1, 0]].float().cpu() - 1) * .5
    target = (batch["teacher"]["xy"][0].float().cpu() + 1) * scale
    decoded = {name: (value[0].float().cpu() + 1) * scale for name, value in predictions.items()}
    valid = batch["teacher"]["valid"][0].cpu()
    primary = batch["teacher"]["transport_weight"][0].bool().cpu()
    times = batch["times"][0].cpu()
    indices = batch["frame_valid"][0].nonzero().flatten().tolist()
    conditions = ("observed_state", "frozen_state", "frozen_tokens", "frozen_centers")
    banner = 48

    def comparison_frames():
        for frame in indices:
            source = rgb_image(batch["rgb"][0, frame, :, :h, :w].cpu())
            grid = Image.new("RGB", (w * 2, (h + banner) * 2), "black")
            past = [index for index in indices if times[frame]-1 <= times[index] <= times[frame]]
            for panel, condition in enumerate(conditions):
                image = Image.new("RGB", (w, h + banner), "black")
                image.paste(source, (0, banner))
                draw = ImageDraw.Draw(image)
                draw.text((5, 4), f"{condition}  t={float(times[frame]):.2f}s  OBSERVED RGB", fill="white")
                draw.text((5, 20), "yellow=transport target cyan=context target red=readout", fill="white")
                for point in valid[frame].nonzero().flatten().tolist():
                    tint = "#ffd700" if primary[point] else "#00dfff"
                    x, y = target[frame, point].tolist()
                    px, py = decoded[condition][frame, point].tolist()
                    for first, second in zip(past, past[1:]):
                        if not bool(valid[first, point] and valid[second, point]):
                            continue
                        a, b = target[first, point].tolist(), target[second, point].tolist()
                        draw.line((a[0], a[1]+banner, b[0], b[1]+banner), fill=tint, width=1)
                    draw.line((x, y+banner, px, py+banner), fill="#ff5555", width=1)
                    draw.ellipse((x-2, y+banner-2, x+2, y+banner+2), fill=tint)
                    draw.ellipse((px-2, py+banner-2, px+2, py+banner+2), fill="#ff5555")
                grid.paste(image, ((panel % 2) * w, (panel // 2) * (h + banner)))
            yield grid

    playback_times = [float(times[index] - times[indices[0]]) for index in indices]
    comparison = destination / "state_comparison.mp4"
    write_video(comparison, comparison_frames(), 5, frame_times=playback_times)
    centers = [(state.centers[0].float().cpu() + 1) * scale for state in output["observed_states"]]
    query_valid = output["queries"].valid[0].cpu()

    def query_frames():
        for frame in indices:
            image = Image.new("RGB", (w, h + banner), "black")
            image.paste(rgb_image(batch["rgb"][0, frame, :, :h, :w].cpu()), (0, banner))
            draw = ImageDraw.Draw(image)
            draw.text((5, 4), f"Observed State centers t={float(times[frame]):.2f}s", fill="white")
            draw.text((5, 20), "color=query index (not object label); root number + local carriers", fill="white")
            for query in query_valid.nonzero().flatten().tolist():
                tint = color(query)
                root_x, root_y = centers[frame][query, 0].tolist()
                draw.text((root_x+4, root_y+banner+4), str(query), fill=tint)
                for x, y in centers[frame][query].tolist():
                    draw.ellipse((x-3, y+banner-3, x+3, y+banner+3), outline=tint, width=2)
                    draw.line((root_x, root_y+banner, x, y+banner), fill=tint, width=1)
            yield image

    query_video = destination / "state_queries.mp4"
    write_video(query_video, query_frames(), 5, frame_times=playback_times)
    return {"comparison": str(comparison), "queries": str(query_video)}
