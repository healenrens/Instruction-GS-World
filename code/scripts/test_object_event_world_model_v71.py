#!/usr/bin/env python3
"""Real-asset end-to-end training, recovery, causal-prefix, and video output test."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igsw.object_event_wm.codec import FrozenVisualCodec, letterbox_video
from igsw.object_event_wm.config import ModelConfig
from igsw.object_event_wm.data import ObjectEventVideoDataset, collate_object_event_video, write_json
from igsw.object_event_wm.evaluation import evaluate
from igsw.object_event_wm.model import ObjectEventWorldModel
from igsw.object_event_wm.prior import ObjectEffectPrior, prior_objective
from igsw.object_event_wm.training import device_batch, read_checkpoint


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--modes", default="global_step,object_step,object_segment")
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--source-revision", default="")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_script = Path(__file__).with_name("train_object_event_world_model_v71.py")
    reports = {}
    for mode in args.modes.split(","):
        full, recovered = out/mode/"uninterrupted", out/mode/"recovered"
        common = [sys.executable, str(train_script), "--manifest", args.manifest, "--assets", args.assets,
                  "--steps", "2", "--batch", "1", "--accum", "1", "--workers", "0",
                  "--log-every", "1", "--save-every", "1", "--snapshot-every", "0",
                  "--rollout-ramp", "1", "--warmup", "1", "--effect-mode", mode,
                  "--precision", args.precision, "--source-revision", args.source_revision]
        subprocess.run([*common, "--out", str(full)], check=True)
        subprocess.run([*common, "--out", str(recovered), "--stop-after", "1"], check=True)
        subprocess.run([*common, "--out", str(recovered), "--resume", str(recovered/"latest.json")], check=True)
        first, second = read_checkpoint(full/"latest.json"), read_checkpoint(recovered/"latest.json")
        difference = max(float((value.float()-second["model"][name].float()).abs().max())
                         for name, value in first["model"].items())
        left, right = rows(full/"train.jsonl"), rows(recovered/"train.jsonl")
        order_equal = [r["case_ids_rank0"] for r in left] == [r["case_ids_rank0"] for r in right]
        reports[mode] = {"resume_parameter_max_difference": difference, "sample_order_equal": order_equal,
                         "uninterrupted_losses": [r["loss"] for r in left],
                         "recovered_losses": [r["loss"] for r in right],
                         "finite_gradients": all(torch.isfinite(torch.tensor(r["grad_norm"])).item() for r in left+right)}
        assert difference < 1e-5 and order_equal, reports[mode]
        del first, second
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
    prior_report = {"loss": float(prior_loss.detach()), "gradient_norm": float(prior_norm),
                    "effect_shape": list(generated["effects"].shape),
                    "generated_state_count": len(generated["states"]),
                    "language_encoder_tested": False,
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
    write_json(out/"test_report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
