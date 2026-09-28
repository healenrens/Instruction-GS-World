#!/usr/bin/env python3
"""Single-GPU full-model integration: both training stages, real resume, and causal inputs."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from igsw.adaptive_gaussian_wm.v69_runtime import add_v69_arguments
from igsw.adaptive_gaussian_wm.v69_config import ObjectVideoConfigV69, parameter_inventory
from igsw.adaptive_gaussian_wm.object_video_manifest_v69 import load_object_video_manifest_v69
from igsw.adaptive_gaussian_wm.v69_resume_diagnostics import compare_resume_v69
from igsw.adaptive_gaussian_wm.object_video_sequence_dataset_v69 import ObjectVideoSequenceDatasetV69, collate_object_video_v69, move_batch_v69
from igsw.adaptive_gaussian_wm.pretrained_visual_encoder_v69 import PretrainedVisualEncoderV69
from igsw.adaptive_gaussian_wm.object_video_world_model_v69 import ObjectVideoWorldModelV69
from igsw.adaptive_gaussian_wm.object_sequence_readout_v69 import appearance_binding_target_v69
from igsw.adaptive_gaussian_wm.tracker_visual_review_v67 import write_json


def run_training(args, manifest, out, stage, steps, stop_after, state_checkpoint="", resume=""):
    script = Path(__file__).with_name("train_object_video_sequence_v69.py")
    command = [sys.executable, str(script), "--manifest", str(manifest), "--out", str(out), "--stage", stage,
               "--encoder", args.encoder, "--encoder_repository", args.encoder_repository, "--encoder_weights", args.encoder_weights,
               "--encoder_frame_batch", str(args.encoder_frame_batch), "--steps", str(steps), "--stop_after", str(stop_after),
               "--batch", "1", "--global_batch", "1", "--workers", "0", "--seed", str(args.seed),
               "--log_every", "1", "--save_every", str(steps), "--recovery_every", "1", "--wandb_mode", "disabled",
               "--deterministic", "--resume_trace",
               "--source_revision", args.source_revision]
    if args.config:
        command += ["--config", args.config]
    if state_checkpoint:
        command += ["--state_checkpoint", str(state_checkpoint)]
    if resume:
        command += ["--resume", str(resume)]
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"):
        env.pop(name, None)
    print(f"[object-video-v69-test] phase={stage} stop_after={stop_after} resume={resume}", flush=True)
    subprocess.run(command, check=True, env=env)


def main():
    args = add_v69_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = None
    if args.wandb_mode != "disabled":
        import wandb
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ.pop("WANDB_RESUME", None)
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=args.wandb_name,
                         group="object-video-sequence-v69", job_type="single-gpu-integration", mode=args.wandb_mode, config=vars(args))
    manifest = load_object_video_manifest_v69(args.manifest)
    grouped = defaultdict(list)
    for entry in manifest["entries"]:
        if entry["partition"] == "train":
            grouped[entry["source"]].append(entry)
    chosen = [rows[0] for _, rows in sorted(grouped.items())]
    test_manifest = out / "test_dataset.json"
    write_json(test_manifest, {**manifest, "entries": chosen, "scope": "full-resolution real-clip integration, not a scientific evaluation"})
    steps = max(2, len(chosen)*2)
    half = min(steps-1, max(1, len(chosen)//2+1))
    resume_differences, resume_comparisons = {}, {}
    execution_metrics = {}
    for stage in ("state", "dynamics"):
        stage_out = out / stage
        parent = out / "state/latest.pt" if stage == "dynamics" else ""
        run_training(args, test_manifest, stage_out, stage, steps, half, parent)
        first = torch.load(stage_out / "latest.pt", map_location="cpu", weights_only=False)
        assert first["step"] == half
        assert first["world_size"] == 1
        assert first["optimizer"]["state"]
        del first
        run_training(args, test_manifest, stage_out, stage, steps, steps, parent, stage_out / "latest.pt")
        resumed = torch.load(stage_out / "latest.pt", map_location="cpu", weights_only=False)
        assert resumed["step"] == steps and resumed["scheduler"]["last_epoch"] == steps
        baseline_out = out / f"{stage}_uninterrupted"
        run_training(args, test_manifest, baseline_out, stage, steps, steps, parent)
        baseline = torch.load(baseline_out / "latest.pt", map_location="cpu", weights_only=False)
        comparison = compare_resume_v69(resumed, baseline, stage_out, baseline_out)
        comparison["numerical_mode"] = resumed["numerical_mode"]
        comparison_path = out / f"{stage}_resume_comparison.json"
        write_json(comparison_path, comparison)
        print(json.dumps({"event": "resume_comparison", "stage": stage, "report": str(comparison_path),
                          "passed": comparison["passed"], "model_max_abs_difference": comparison["model_max_abs_difference"],
                          "first_trace_mismatch": comparison["first_trace_mismatch"]}), flush=True)
        if run:
            run.summary[f"resume/{stage}"] = comparison
            artifact = wandb.Artifact(f"{args.wandb_name}-{stage}-resume", type="resume-comparison")
            artifact.add_file(str(comparison_path))
            run.log_artifact(artifact)
        assert comparison["passed"], f"resume comparison failed: {stage}; first divergence and numerical differences saved in {comparison_path}"
        resume_comparisons[stage] = comparison
        resume_differences[stage] = comparison["model_max_abs_difference"]
        execution_metrics[stage] = [json.loads(line) for line in (stage_out / "metrics.jsonl").read_text().splitlines()]
        assert all(not row["trainable_parameters_without_gradient"] for row in execution_metrics[stage])
        del baseline
        del resumed
    eval_manifest = out / "evaluation_dataset.json"
    write_json(eval_manifest, {**manifest, "entries": [{**entry, "partition": "held"} for entry in chosen],
                               "scope": "training clips reused solely for evaluator runtime integration; not held scientific evidence"})
    evaluation_out = out / "evaluation_runtime"
    subprocess.run([sys.executable, str(Path(__file__).with_name("evaluate_object_video_sequence_v69.py")),
                    "--manifest", str(eval_manifest), "--out", str(evaluation_out),
                    "--checkpoint", str(out / "dynamics/latest.pt"), "--items", "1", "--visualize", "1",
                    "--encoder_repository", args.encoder_repository, "--encoder_weights", args.encoder_weights,
                    "--encoder_frame_batch", str(args.encoder_frame_batch), "--wandb_mode", "disabled"], check=True)
    checkpoint = torch.load(out / "state/latest.pt", map_location="cpu", weights_only=False)
    config = ObjectVideoConfigV69(**checkpoint["config"])
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    perception = PretrainedVisualEncoderV69(config.encoder, args.encoder_repository, args.encoder_weights,
                     args.encoder_frame_batch, history_seconds=config.history_seconds, saved_backbone=checkpoint["perception"]).to(device)
    model = ObjectVideoWorldModelV69(config, "state").to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    dataset = ObjectVideoSequenceDatasetV69(test_manifest, config, args.seed)
    cpu_batch = collate_object_video_v69([dataset[0]])
    batch = move_batch_v69(cpu_batch, device)
    th = config.history_frames
    with torch.no_grad():
        fields = perception(batch["rgb"], batch["pixel_valid"], batch["times"], batch["native_hw"])
        changed_rgb = batch["rgb"].clone()
        changed_rgb[:, th:] = 255-changed_rgb[:, th:]
        changed = perception(changed_rgb, batch["pixel_valid"], batch["times"], batch["native_hw"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            query, states = model.encode_history(fields)
            changed_query, changed_states = model.encode_history(changed)
            source_difference = float((states[-1].tokens-changed_states[-1].tokens).abs().max())
            query_difference = float((query.features-changed_query.features).abs().max())
            assert source_difference < 1e-6 and query_difference < 1e-6
            assert torch.equal(query.xy, changed_query.xy)
            output = model(fields, batch)
            changed_teacher = {**batch["teacher"], "xy": batch["teacher"]["xy"].flip(2), "transport_weight": torch.zeros_like(batch["teacher"]["transport_weight"])}
            teacher_output = model(fields, {**batch, "teacher": changed_teacher})
            teacher_difference = float((output["source"].tokens-teacher_output["source"].tokens).abs().max())
            assert teacher_difference < 1e-6
            target, evidence = appearance_binding_target_v69(query, output["measurement_features"][:, th-1], fields.features[:, th-1], fields.valid[:, th-1])
            active = torch.cat((query.valid, query.valid.new_ones((1, 1))), -1).float()
            uniform = active[:, None] / active.sum(-1)[:, None, None]
            uniform_kl = (target * (target.clamp_min(1e-7).log()-uniform.clamp_min(1e-7).log())).sum(-1)
            uniform_margin = float((uniform_kl*evidence).sum()/evidence.sum().clamp_min(1))
            actual_target_variation = float(evidence.sum())
            assert not any(p.requires_grad for p in perception.parameters())
            assert not any(p.requires_grad for p in model.target_encoder.parameters())
            assert output["source"].tokens.shape[1:] == (config.object_queries, config.tokens_per_object, config.width)
    inventory = parameter_inventory({"perception": perception, "object_memory": model.encoder, "EMA_memory": model.target_encoder,
                                     "readout": model.readout, "posterior": model.posterior, "dynamics": model.dynamics})
    report = {"status": "passed_runtime_contract", "architecture": config.architecture, "config": config.to_dict(),
              "full_capacity_model": config.to_dict() == ObjectVideoConfigV69(encoder=config.encoder).to_dict(),
              "native_resolution": batch["native_hw"].cpu().tolist(), "parameter_inventory": inventory,
              "history_seconds_actual": float(batch["times"][0, th-1]-batch["times"][0, 0]),
              "future_seconds_actual": float(batch["times"][0, -1]), "history_frames": th, "future_frames": config.future_frames,
              "sources": sorted(grouped), "state_steps_with_resume": steps, "dynamics_steps_with_resume": steps,
              "resume_vs_uninterrupted_parameter_max_difference": resume_differences,
              "resume_comparisons": resume_comparisons,
              "deterministic_resume_test": True,
              "execution_metrics": execution_metrics,
              "future_swap_history_max_difference": source_difference, "future_swap_query_max_difference": query_difference,
              "teacher_swap_history_max_difference": teacher_difference, "uniform_binding_target_kl": uniform_margin,
              "binding_evidence_mass": actual_target_variation, "teacher_binding_is_weak_visual_affinity_not_object_GT": True,
              "evaluator_runtime_report": str(evaluation_out / "report.json"),
              "evaluation_clip_is_not_scientific_held_evidence": True,
              "object_semantics_verified": False, "training_checkpoints_are_test_only": True}
    write_json(out / "test_report.json", report)
    if run:
        run.summary.update(report)
        artifact = wandb.Artifact(args.wandb_name, type="object-video-runtime-test")
        artifact.add_file(str(out / "test_report.json"))
        artifact.add_dir(str(evaluation_out), name="evaluation_runtime")
        for stage in ("state", "dynamics"):
            artifact.add_file(str(out / stage / "model_inventory.json"), name=f"{stage}_model_inventory.json")
            artifact.add_file(str(out / stage / "progress.json"), name=f"{stage}_progress.json")
        run.log_artifact(artifact)
        run.finish()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
