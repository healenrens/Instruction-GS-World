#!/usr/bin/env python3
"""Remote single-GPU Stage2 expansion from the assessed real State checkpoint."""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from test_object_video_sequence_v69 import run_training
from igsw.adaptive_gaussian_wm.v69_runtime import add_v69_arguments
from igsw.adaptive_gaussian_wm.v69_config import ObjectVideoConfigV69, parameter_inventory
from igsw.adaptive_gaussian_wm.object_video_manifest_v69 import load_object_video_manifest_v69
from igsw.adaptive_gaussian_wm.v69_resume_diagnostics import compare_resume_v69, configure_reproducibility_v69
from igsw.adaptive_gaussian_wm.v69_association_verification import verify_association_interfaces_v69
from igsw.adaptive_gaussian_wm.object_video_sequence_dataset_v69 import ObjectVideoSequenceDatasetV69, collate_object_video_v69, move_batch_v69
from igsw.adaptive_gaussian_wm.pretrained_visual_encoder_v69 import PretrainedVisualEncoderV69
from igsw.adaptive_gaussian_wm.object_video_world_model_v69 import ObjectVideoWorldModelV69
from igsw.adaptive_gaussian_wm.state_change_evaluation_v69 import observe_state_v69
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import write_json


ASSESSED_STATE_CHECKPOINT = (
    "/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/"
    "object_video_v69_state_change_held_aed14bb_20261001_235841/checkpoint_snapshot.pt"
)


def frozen_checkpoint_invariants(source, trained):
    modules = {}
    for module in ("encoder", "target_encoder", "readout"):
        weights = {name: value for name, value in source["model"].items() if name.startswith(module + ".")}
        assert weights
        assert weights.keys() == {name for name in trained["model"] if name.startswith(module + ".")}
        assert all(torch.equal(value, trained["model"][name]) for name, value in weights.items()), module
        modules[module] = {"tensors": len(weights), "unchanged_exactly": True}
    assert source["perception"].keys() == trained["perception"].keys()
    assert all(torch.equal(value, trained["perception"][name]) for name, value in source["perception"].items())
    modules["perception"] = {"tensors": len(source["perception"]), "unchanged_exactly": True}
    return modules


def main():
    parser = add_v69_arguments(argparse.ArgumentParser(description=__doc__))
    parser.set_defaults(stage="dynamics", stage2_preset="large", dynamics_checkpoint_blocks=True,
                        state_checkpoint=ASSESSED_STATE_CHECKPOINT, steps=2, batch=1, global_batch=1, workers=0,
                        deterministic=True, resume_trace=True, wandb_name="v69_stage2_large_integration")
    args = parser.parse_args()
    configure_reproducibility_v69(True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = torch.load(args.state_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    assert source["args"]["stage"] == "state"
    if args.state_checkpoint == ASSESSED_STATE_CHECKPOINT:
        assert source["step"] == 6000
    run = None
    if args.wandb_mode != "disabled":
        import wandb
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ.pop("WANDB_RESUME", None)
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=args.wandb_name,
                         group="object-video-sequence-v69", job_type="stage2-large-integration",
                         mode=args.wandb_mode, config={**vars(args), "state_step": source["step"]})
    manifest = load_object_video_manifest_v69(args.manifest)
    grouped = defaultdict(list)
    for entry in manifest["entries"]:
        if entry["partition"] == "train":
            grouped[entry["source"]].append(entry)
    chosen = [entries[0] for _, entries in sorted(grouped.items())]
    test_manifest = out / "test_dataset.json"
    write_json(test_manifest, {**manifest, "entries": chosen,
                               "scope": "real native-resolution Stage2 runtime integration only"})
    steps = max(2, args.steps)
    split_step = steps // 2
    stage_out, baseline_out = out / "dynamics", out / "dynamics_uninterrupted"
    run_training(args, test_manifest, stage_out, "dynamics", steps, split_step, args.state_checkpoint)
    first = torch.load(stage_out / f"step_{split_step:07d}.pt", map_location="cpu", mmap=True, weights_only=False)
    assert first["step"] == split_step and first["optimizer"]["state"]
    initialization = json.loads((stage_out / "initialization.json").read_text())
    assert initialization["step"] == 0 and initialization["epoch"] == 0 and initialization["cursor"] == 0
    assert initialization["optimizer_state_entries"] == 0
    assert set(initialization["trainable_modules"]) == {"posterior", "dynamics"}
    assert initialization["state_initialization"]["step"] == source["step"]
    assert initialization["state_initialization"]["optimizer_inherited"] is False
    frozen_after_first_step = frozen_checkpoint_invariants(source, first)
    run_training(args, test_manifest, stage_out, "dynamics", steps, steps,
                 args.state_checkpoint, stage_out / "latest.pt")
    resumed = torch.load(stage_out / "latest.pt", map_location="cpu", mmap=True, weights_only=False)
    run_training(args, test_manifest, baseline_out, "dynamics", steps, steps, args.state_checkpoint)
    baseline = torch.load(baseline_out / "latest.pt", map_location="cpu", mmap=True, weights_only=False)
    comparison = compare_resume_v69(resumed, baseline, stage_out, baseline_out)
    comparison_path = out / "dynamics_resume_comparison.json"
    write_json(comparison_path, comparison)
    print(json.dumps({"event": "stage2_resume_comparison", "report": str(comparison_path), **comparison}), flush=True)
    if run:
        run.summary["resume_comparison"] = comparison
        artifact = wandb.Artifact(args.wandb_name + "-resume", type="resume-comparison")
        artifact.add_file(str(comparison_path))
        run.log_artifact(artifact)
    assert comparison["passed"], str(comparison_path)
    assert resumed["step"] == steps and resumed["scheduler"]["last_epoch"] == steps
    assert resumed["world_size"] == 1
    assert resumed["state_initialization"] == baseline["state_initialization"] == initialization["state_initialization"]
    assert resumed["architecture"] == resumed["config"]["architecture"]
    config = ObjectVideoConfigV69(**resumed["config"])
    assert config.width == 512 and config.heads == 8
    assert (config.dynamics_hidden_width, config.dynamics_attention_heads, config.dynamics_layers) == (1024, 16, 12)
    assert (config.posterior_hidden_width, config.posterior_attention_heads, config.posterior_layers) == (1024, 16, 4)
    assert config.dynamics_checkpoint_blocks
    assert (config.history_frames, config.future_frames, config.object_queries, config.effect_tokens, config.effect_dim) == (16, 25, 16, 4, 64)
    frozen_after_resume = frozen_checkpoint_invariants(source, resumed)
    parameter_updates = {}
    for module in ("posterior", "dynamics"):
        differences = [float((value-first["model"][name]).abs().max())
                       for name, value in resumed["model"].items() if name.startswith(module + ".")]
        assert max(differences) > 0, module
        parameter_updates[module] = {"max_abs_update_after_split": max(differences),
                                     "updated_tensors_after_split": sum(value > 0 for value in differences)}
    execution_metrics = [json.loads(line) for line in (stage_out / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in execution_metrics] == list(range(1, steps+1))
    for row in execution_metrics:
        assert not row["trainable_parameters_without_gradient"]
        assert all(math.isfinite(value) for value in row.values() if isinstance(value, (int, float)))
        assert all(value == 1. for name, value in row.items() if name.startswith("gradient_preclip/") and name.endswith("/finite"))
        assert all(row[f"gradient_preclip/{module}/l2"] > 0 for module in ("posterior", "dynamics"))
        assert all(name in row for name in ("observed_target_transport", "current_state_copy_transport", "last_observation_copy_transport"))
        if run:
            run.log({f"stage2/{key}": value for key, value in row.items() if isinstance(value, (int, float))}, step=row["step"])
    traces = sorted((stage_out / "resume_trace_rank0000").glob("*.pt"))
    assert len(traces) == steps
    tested_case_ids = set()
    effect_dtypes = set()
    for path in traces:
        trace = torch.load(path, map_location="cpu", weights_only=False)
        tested_case_ids.update(trace["case_id"])
        assert trace["effect_value"].shape == (1, 16, 4, 64)
        # Autocast may promote Gaussian sampling to FP32 after exp().
        assert trace["effect_value"].dtype in (torch.bfloat16, torch.float32)
        effect_dtypes.add(str(trace["effect_value"].dtype))
        assert trace["source_tokens"].shape == (1, 16, config.tokens_per_object, 512)
        assert trace["frame_indices"].shape == (1, 41)
        assert trace["effect_value"].isfinite().all()
        assert trace["posterior_query_gradient"].isfinite().all()
        assert trace["posterior_query_gradient"].shape == (4, 1024)
    del first, baseline
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    perception = PretrainedVisualEncoderV69(config.encoder, args.encoder_repository, args.encoder_weights,
                     args.encoder_frame_batch, history_seconds=config.history_seconds, saved_backbone=resumed["perception"]).to(device)
    model = ObjectVideoWorldModelV69(config, "dynamics").to(device).eval()
    model.load_state_dict(resumed["model"], strict=True)
    del source, resumed
    frozen_modules = {"perception": perception, "encoder": model.encoder,
                      "target_encoder": model.target_encoder, "readout": model.readout}
    assert all(not parameter.requires_grad and parameter.grad is None
               for module in frozen_modules.values() for parameter in module.parameters())
    dataset = ObjectVideoSequenceDatasetV69(test_manifest, config, args.seed, "train")
    batch = move_batch_v69(collate_object_video_v69([dataset[0]]), device)
    with torch.no_grad():
        fields = perception(batch["rgb"], batch["pixel_valid"], batch["times"], batch["native_hw"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = observe_state_v69(model, fields, batch)
        association = verify_association_interfaces_v69(model, output, batch)
    inventory = parameter_inventory({**frozen_modules, "posterior": model.posterior, "dynamics": model.dynamics})
    trainable = inventory["posterior"]["trainable_parameters"] + inventory["dynamics"]["trainable_parameters"]
    assert 350_000_000 < trainable < 365_000_000
    report = {"status": "passed_runtime_contract", "architecture": config.architecture, "config": config.to_dict(),
              "state_initialization": initialization["state_initialization"], "initialization": initialization,
              "parameter_inventory": inventory, "stage2_trainable_parameters": trainable,
              "training_dtype": "bfloat16_autocast", "effect_dtypes": sorted(effect_dtypes), "training_steps": steps,
              "effect_shape": [1, 16, 4, 64], "state_shape": [1, 16, config.tokens_per_object, 512],
              "native_hw": batch["native_hw"].cpu().tolist(), "selected_sources": sorted(grouped),
              "tested_case_ids": sorted(tested_case_ids),
              "frozen_after_first_step": frozen_after_first_step, "frozen_after_resume": frozen_after_resume,
              "parameter_updates": parameter_updates, "resume_comparison": comparison,
              "association_interfaces": association, "execution_metrics": execution_metrics,
              "objective_unchanged": True, "object_semantics_verified": False,
              "training_checkpoints_are_test_only": True}
    write_json(out / "test_report.json", report)
    if run:
        run.summary.update(report)
        artifact = wandb.Artifact(args.wandb_name, type="object-video-stage2-runtime-test")
        for path in (out / "test_report.json", stage_out / "model_inventory.json", stage_out / "initialization.json",
                     stage_out / "progress.json", stage_out / "metrics.jsonl", comparison_path):
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        run.finish()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
