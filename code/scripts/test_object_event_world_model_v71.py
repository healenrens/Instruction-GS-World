#!/usr/bin/env python3
"""Real-asset end-to-end training, recovery, causal-prefix, and video output test."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
import av

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igsw.object_event_wm.codec import FrozenVisualCodec, letterbox_video
from igsw.object_event_wm.config import ModelConfig
from igsw.object_event_wm.data import ObjectEventVideoDataset, collate_object_event_video, write_json
from igsw.object_event_wm.evaluation import evaluate
from igsw.object_event_wm.model import ObjectEventWorldModel
from igsw.object_event_wm.prior import ObjectEffectPrior, prior_objective
from igsw.object_event_wm.prior_training import load_prior_model
from igsw.object_event_wm.text import TextFeatureStore
from igsw.object_event_wm.training import device_batch, read_checkpoint


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def exercise_resume(common, root):
    full, recovered = root/"uninterrupted", root/"recovered"
    subprocess.run([*common, "--out", str(full)], check=True)
    subprocess.run([*common, "--out", str(recovered), "--stop-after", "1"], check=True)
    subprocess.run([*common, "--out", str(recovered), "--resume", str(recovered/"latest.json")], check=True)
    first, second = read_checkpoint(full/"latest.json"), read_checkpoint(recovered/"latest.json")
    difference = max(float((value.float()-second["model"][name].float()).abs().max())
                     for name, value in first["model"].items())
    left, right = rows(full/"train.jsonl"), rows(recovered/"train.jsonl")
    order_equal = [row["case_ids_rank0"] for row in left] == [row["case_ids_rank0"] for row in right]
    result = {"resume_parameter_max_difference": difference, "sample_order_equal": order_equal,
              "uninterrupted_losses": [row["loss"] for row in left],
              "recovered_losses": [row["loss"] for row in right],
              "finite_gradients": all(torch.isfinite(torch.tensor(row["grad_norm"])).item() for row in left+right)}
    assert difference < 1e-5 and order_equal, result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--modes", default="global_step,object_step,object_segment")
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--text-cache", default="", help="Include the real language prior and history-only CLI path.")
    parser.add_argument("--prior-base-checkpoint", default="", help="Use a specified fixed teacher for the conditional prior path.")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_script = Path(__file__).with_name("train_object_event_world_model_v71.py")
    reports = {}
    for mode in args.modes.split(","):
        common = [sys.executable, str(train_script), "--manifest", args.manifest, "--assets", args.assets,
                  "--steps", "2", "--batch", "1", "--accum", "1", "--workers", "0",
                  "--log-every", "1", "--save-every", "1", "--snapshot-every", "0",
                  "--rollout-ramp", "1", "--warmup", "1", "--effect-mode", mode,
                  "--precision", args.precision, "--source-revision", args.source_revision]
        reports[mode] = exercise_resume(common, out/mode)
    selected = args.modes.split(",")[-1]
    saved = read_checkpoint(out/selected/"recovered/latest.json")
    model = ObjectEventWorldModel(ModelConfig(**saved["model_config"])).cuda().eval()
    model.load_state_dict(saved["model"])
    codec = FrozenVisualCodec(args.assets).cuda().eval()
    data = ObjectEventVideoDataset(args.manifest, partition="development")
    sample = data[0]
    batch = device_batch(collate_object_event_video([sample]), torch.device("cuda"))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
        images, mask = letterbox_video(batch, codec.image_size)
        features = codec.encode(images)
        valid = codec.patch_valid(mask)
        original = model.observe(features, valid, batch["times"])
        changed = features.clone()
        th = batch["history_frames"]
        changed[:, th:] = changed[:, th:].flip(1)
        other = model.observe(changed, valid, batch["times"])
        prefix_difference = max(float((a.tokens-b.tokens).abs().max())
                                for a, b in zip(original[:th], other[:th]))
    assert prefix_difference < 1e-6, prefix_difference
    model.requires_grad_(False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
        teacher = model(batch, codec, deterministic=True)
    prior = ObjectEffectPrior(model.config, text_width=model.config.width).cuda().train()
    optimizer = torch.optim.AdamW(prior.parameters(), lr=2e-4)
    history = teacher["observed_states"][:th]
    future_times = batch["times"][:, th:]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
        prior_loss, prior_parts = prior_objective(
            prior, history, teacher["predicted_states"], teacher["effects"],
            future_times, batch["frame_valid"][:, th:])
    prior_loss.backward()
    prior_norm = torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.)
    optimizer.step()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
        generated = prior.generate(history, future_times, model.dynamics,
                                   generator=torch.Generator(device="cuda").manual_seed(17))
        context = prior.condition(history)
        sources = [history[-1], *teacher["predicted_states"][:-1]]
        tokens = torch.stack([state.tokens for state in sources], 1)
        centers = torch.stack([state.centers for state in sources], 1)
        source_times = torch.stack([state.time for state in sources], 1)
        previous = torch.stack([row["value"] for row in teacher["effects"]], 1)
        previous = torch.cat((torch.zeros_like(previous[:, :1]), previous[:, :-1]), 1)
        original_prediction = prior.forward_sequence(context, tokens, centers, source_times, previous, future_times)
        changed_tokens = tokens.clone()
        changed_tokens[:, 1:] += 3
        changed_prediction = prior.forward_sequence(context, changed_tokens, centers, source_times, previous, future_times)
        prior_prefix_difference = float((original_prediction["mean"][:, 0]-changed_prediction["mean"][:, 0]).abs().max())
    assert prior_prefix_difference < 1e-6, prior_prefix_difference
    prior_report = {"loss": float(prior_loss.detach()), "gradient_norm": float(prior_norm),
                    "effect_shape": list(generated["effects"].shape),
                    "generated_state_count": len(generated["states"]),
                    "language_encoder_tested": False,
                    "later_teacher_forced_state_swap_first_prediction_difference": prior_prefix_difference,
                    "finite_generated_effects": bool(torch.isfinite(generated["effects"]).all()),
                    "parts": {key: float(value.detach()) for key, value in prior_parts.items()}}
    assert torch.isfinite(prior_norm) and prior_norm > 0 and prior_report["finite_generated_effects"], prior_report
    del teacher, prior, optimizer, generated
    reconstruction = evaluate(model, codec, data, out/"reconstruction", items_per_source=1,
                               precision=args.precision, videos_per_source=1)
    report = {"status": "passed", "source_revision": args.source_revision,
              "real_assets": codec.asset_config, "model_config": saved["model_config"],
              "modes": reports, "future_feature_swap_prefix_max_difference": prefix_difference,
              "codec_parameters_frozen": all(not p.requires_grad for p in codec.parameters()),
              "history_prior_update_rollout": prior_report,
              "reconstruction_cases": reconstruction["cases"],
              "reconstruction_sources": reconstruction["sources"],
              "scope": "engineering end-to-end execution; not a representation-quality result"}
    if args.text_cache:
        common = [sys.executable, str(train_script), "--manifest", args.manifest, "--assets", args.assets,
                  "--steps", "2", "--batch", "1", "--accum", "1", "--workers", "0",
                  "--log-every", "1", "--save-every", "1", "--snapshot-every", "0", "--warmup", "1",
                  "--stage", "prior", "--base-checkpoint", args.prior_base_checkpoint or str(out/selected/"recovered/latest.json"),
                  "--text-cache", args.text_cache, "--precision", args.precision,
                  "--source-revision", args.source_revision]
        report["language_prior_training"] = exercise_resume(common, out/"language_prior")
        conditional_model, conditional_prior, _ = load_prior_model(out/"language_prior/recovered/latest.json", torch.device("cuda"))
        text_store = TextFeatureStore(args.text_cache)
        conditional_data = ObjectEventVideoDataset(args.manifest, partition="development", sampling="language_windows")
        evaluated = evaluate(conditional_model, codec, conditional_data, out/"conditional_generation",
                             items_per_source=1, precision=args.precision, videos_per_source=1,
                             prior=conditional_prior, text_store=text_store, samples=2)
        first_case = rows(out/"conditional_generation/cases.jsonl")[0]
        forecast_script = Path(__file__).with_name("forecast_object_event_world_model_v71.py")
        subprocess.run([sys.executable, str(forecast_script), "--prior-checkpoint", str(out/"language_prior/recovered/latest.json"),
                        "--assets", args.assets, "--history-request",
                        str(out/"conditional_generation/videos"/first_case["case_id"]/"history_request.json"),
                        "--instruction", first_case["instruction"], "--text-model", text_store.metadata["model"],
                        "--precision", args.precision,
                        "--out", str(out/"history_only_forecast")], check=True)
        report["language_prior_training"].update(generation_cases=evaluated["cases"],
            frozen_teacher=all(not p.requires_grad for p in conditional_model.parameters()),
            history_only_forecast=json.loads((out/"history_only_forecast/prediction.json").read_text()))
        with av.open(str(out/"history_only_forecast/forecast.mp4")) as video:
            video_tick = float(video.streams.video[0].time_base)
            video_times = torch.tensor([frame.time for frame in video.decode(video=0)])
        requested = torch.tensor(report["language_prior_training"]["history_only_forecast"]["frame_times"])
        video_time_error = float((video_times-(requested-requested[0])).abs().max())
        assert video_time_error <= video_tick+1e-6, video_time_error
        report["language_prior_training"]["video_timestamp_max_error_seconds"] = video_time_error
        report["language_prior_training"]["video_time_base_seconds"] = video_tick
    write_json(out/"test_report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
