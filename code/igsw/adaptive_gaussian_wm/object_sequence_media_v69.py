"""Visualize actual sequence inputs and native-pixel targets without role-colored claims."""

from pathlib import Path

from PIL import ImageDraw

from .tracker_visual_review_media_v67 import rgb_image, write_video
from .tracker_visual_review_v67 import write_json


def render_sequence_v69(batch, output, destination, config, stage):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    th = config.history_frames
    h, w = batch["native_hw"][0].tolist()
    scale = batch["native_hw"][0, [1, 0]].float()-1
    target = (batch["teacher"]["xy"][0].detach().cpu()+1)*.5*scale.cpu()
    predicted = output["observed_positions"][0] if stage == "state" else output["rollout_positions"][0]
    predicted = (predicted.detach().float().cpu()+1)*.5*scale.cpu()
    valid = batch["teacher"]["valid"][0].cpu()
    motion = batch["teacher"]["transport_weight"][0].bool().cpu()
    frame_ids = batch["frame_valid"][0].nonzero().flatten().tolist()
    def frames():
        for index in frame_ids:
            image = rgb_image(batch["rgb"][0, index, :, :h, :w].cpu())
            draw = ImageDraw.Draw(image)
            local = index if stage == "state" else index-th
            for point in valid[index].nonzero().flatten().tolist():
                x, y = target[index, point].tolist()
                draw.ellipse((x-2, y-2, x+2, y+2), fill="#ffd700" if motion[point] else "#00dfff")
                if local >= 0:
                    px, py = predicted[local, point].tolist()
                    draw.line((x, y, px, py), fill="#ff5555", width=1)
                    draw.ellipse((px-2, py-2, px+2, py+2), fill="#ff5555")
            draw.rectangle((0, 0, w, 50), fill="black")
            label = "HISTORY INPUT" if index < th else "FUTURE TARGET ONLY"
            draw.text((5, 4), f"{label} t={float(batch['times'][0,index]):.2f}s", fill="white")
            draw.text((5, 19), "yellow=transport cyan=aux/context", fill="white")
            draw.text((5, 34), "red=model", fill="white")
            yield image
    times = [float(batch["times"][0, index]-batch["times"][0, frame_ids[0]]) for index in frame_ids]
    write_video(destination / "sequence.mp4", frames(), 5, frame_times=times)
    write_json(destination / "sequence.json", {"case_id": batch["case_id"][0], "stage": stage,
               "history_frames": th, "frame_indices": batch["frame_indices"][0].cpu().tolist(),
               "times": batch["times"][0].cpu().tolist(), "point_ids": batch["teacher"]["point_ids"][0].cpu().tolist(),
               "target_xy_px": target.tolist(), "prediction_xy_px": predicted.tolist(), "target_valid": valid.tolist(),
               "transport_weight": batch["teacher"]["transport_weight"][0].cpu().tolist(),
               "selection_status": batch["transport_selection_status"][0],
               "transport_loss_mask": (valid & motion[None]).tolist(),
               "observation_labels": batch["teacher"]["observation"][0].cpu().tolist(),
               "loss_scope": "state appearance/binding use observed measurements; transport uses selected 75 pool; latent terms are object-query level",
               "measurement_source": "independent annotation" if batch.get("independent_truth") else "frozen tracker; not ground truth"})
    return destination / "sequence.mp4"
