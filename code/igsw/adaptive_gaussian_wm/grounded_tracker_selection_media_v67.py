"""Compare the exact pre/post ranking target masks, without context clutter."""

import torch

from .grounded_tracker_export_v67 import styled_tracks
from .tracker_visual_review_media_v67 import draw_points, rgb_image, write_video


def render_object_selection(directory, rgb, native, case, queries, selection, width):
    outputs = {}
    for key, field, title in (
        (
            "before_topk",
            "object_motion_candidate_ids_before_topk",
            "Object motion candidates BEFORE ranking",
        ),
        ("after_topk", "object_target_ids", "Object motion targets AFTER ranking"),
    ):
        ids = torch.tensor(selection[field], dtype=torch.long)
        styled = styled_tracks(native, queries, ids)
        styled["legend"] = (
            "green=object candidate; context hidden here, retained in data"
        )
        labels = [queries["labels"][i] for i in ids.tolist()]

        def frames():
            for frame, image in enumerate(rgb):
                overlay = draw_points(
                    rgb_image(image),
                    styled,
                    frame,
                    case["record"]["fps"],
                    labels,
                    f"{title}: {len(ids)} points",
                )
                yield overlay.resize(
                    (width, round(overlay.height * width / overlay.width))
                )

        outputs[key] = f"object_{key}.mp4"
        write_video(directory / outputs[key], frames(), case["record"]["fps"])
    return outputs
