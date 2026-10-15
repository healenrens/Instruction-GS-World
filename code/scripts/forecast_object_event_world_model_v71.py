#!/usr/bin/env python3
"""Generate future RGB from a history-only request and an instruction."""

import argparse
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igsw.object_event_wm.codec import FrozenVisualCodec, letterbox_video
from igsw.object_event_wm.data import write_json
from igsw.object_event_wm.evaluation import write_video
from igsw.object_event_wm.prediction import read_history_request, observe_history, generate_from_history
from igsw.object_event_wm.prior_training import load_prior_model
from igsw.object_event_wm.text import FrozenInstructionEncoder, collate_text_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-checkpoint", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--history-request", required=True)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--text-model", default="/mnt/pfs/public/xuhaoming/model_zoo/Qwen3-VL-4B-Instruct")
    parser.add_argument("--future-seconds", type=float, default=5.)
    parser.add_argument("--future-frames", type=int, default=25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    model, prior, saved = load_prior_model(args.prior_checkpoint, device)
    codec = FrozenVisualCodec(args.assets).to(device)
    observed = read_history_request(args.history_request, device)
    text_encoder = FrozenInstructionEncoder(args.text_model).to(device) if args.instruction else None
    torch.cuda.synchronize()
    started = time.perf_counter()
    text, valid = None, None
    if text_encoder is not None:
        text, valid = collate_text_features(text_encoder([args.instruction]))
        text, valid = text.to(device), valid.to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
        states = observe_history(model, codec, observed)
        extent = args.future_frames if observed["time_unit"] == "sampled_frame" else args.future_seconds
        times = torch.arange(1, args.future_frames+1, device=device)[None]*extent/args.future_frames
        result = generate_from_history(model, codec, prior, states, times, text, valid,
                                        generator=torch.Generator(device=device).manual_seed(args.seed))
        history_rgb, _ = letterbox_video(observed, model.view_size(codec))
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_video(out / "forecast.mp4", torch.cat((history_rgb[0], result["rgb"][0])),
                torch.cat((observed["times"][0], times[0])), observed["time_unit"])
    torch.save({"effects": result["effects"].cpu(), "updates": result["updates"].cpu(),
                "times": times.cpu()}, out / "generated_effects.pt")
    write_json(out / "prediction.json", {"instruction": args.instruction, "seed": args.seed,
                "precision": args.precision,
                "view_size": model.view_size(codec), "codec_decoder_size": codec.image_size,
                "future_rgb_read": False, "teacher_checkpoint": saved["teacher_checkpoint"],
                "future_frames": args.future_frames,
                "future_seconds": args.future_seconds if observed["time_unit"] == "seconds" else None,
                "time_unit": observed["time_unit"],
                "preview_fps_not_physical": 5 if observed["time_unit"] == "sampled_frame" else None,
                "frame_times": torch.cat((observed["times"][0], times[0])).cpu().tolist(),
                "prediction_seconds_including_text_history_dynamics_render": elapsed})


if __name__ == "__main__":
    main()
