#!/usr/bin/env python3
"""One CPU integration for paired sampling, decoded metrics and report comparison.

Uses the actual flow expert with an explicit tiny conditioner; not a GPU/Qwen test.
"""

from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed.checkpoint as dcp

from compare_language_object_effect_v70 import compare_reports
from evaluate_language_object_effect_v70 import diagnostic_indices, save_video, write_report
from test_language_effect_integration_v70 import FixtureConditioner, FixtureDataset, FixtureHistory
from igsw.adaptive_gaussian_wm.language_effect_model_v70 import LanguageEffectModelV70
from igsw.adaptive_gaussian_wm.language_effect_evaluation_v70 import trajectory_records
from igsw.adaptive_gaussian_wm.v70_evaluation_protocol import (
    effect_diagnostics, evaluation_plan, evaluation_summaries, paired_noise,
)
from igsw.adaptive_gaussian_wm.v70_checkpoint import load_model_checkpoint_v70


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    entries = [{"source": "a" if i < 3 else "b", "group": "g", "episode_index": i,
                "window_id": f"window_{i}_{w}", "frame_indices": [0, 1, 2, 3, 4],
                "instruction": f"move item {i}"} for i in range(6) for w in range(2)]
    plan = evaluation_plan(entries, diagnostic_indices(entries, 6, 17), 17)
    reverse = list(reversed(entries))
    assert plan == evaluation_plan(reverse, diagnostic_indices(reverse, 6, 17), 17)
    assert all(row["instruction"] != row["shuffled_instruction"] for row in plan)
    assert len({row["episode_index"] for row in plan}) == 6
    torch.manual_seed(17)
    model = LanguageEffectModelV70("fixture", conditioner=FixtureConditioner(),
                                  expert_kwargs={"num_blocks": 2, "width": 32, "heads": 4, "ffn_dim": 64}).eval()
    batch = torch.utils.data.default_collate([FixtureDataset()[(0, 0, 0)]])
    checkpoint = out / "model_snapshot"
    checkpoint.mkdir(exist_ok=True)
    dcp.save({"model": model.state_dict()}, checkpoint_id=checkpoint / "shards")
    (checkpoint/"metadata.json").write_text(json.dumps({"step": 2500}))
    restored = deepcopy(model)
    with torch.no_grad():
        next(restored.parameters()).zero_()
    load_model_checkpoint_v70(restored, checkpoint)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    history = FixtureHistory()(batch)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        condition = model.encode_condition(model.prepare_inputs(batch), history)
        noise = paired_noise(batch["target_mean"], 17, 2, 0)
        generated = model.sample(condition, steps=3, noise=noise)
        torch.randn(41)
        repeated = model.sample(condition, steps=3, noise=paired_noise(batch["target_mean"], 17, 2, 0))
        assert torch.equal(generated, repeated)
        auxiliary = effect_diagnostics(model, condition, batch["target_mean"], noise, generated)
        assert all(torch.isfinite(torch.tensor(value)) for value in auxiliary.values())
    truth = torch.zeros(1, 5, 2, 2)
    truth[0, 2:, 0, 0] = torch.tensor([.1, .2, .3])
    valid = torch.ones(1, 5, 2, dtype=torch.bool)
    valid[0, -1, 1] = False
    teacher = {"xy": truth, "valid": valid, "reference_xy": torch.zeros(1, 2, 2),
               "point_present": torch.ones(1, 2, dtype=torch.bool), "point_ids": torch.tensor([[7, 9]]),
               "transport_weight": torch.tensor([[1., 0.]]), "reference_index": torch.tensor([[1, 0]])}
    prediction = truth[:, 2:].clone()
    prediction[..., 0] += .02
    rows = trajectory_records(prediction, teacher, torch.tensor([[201, 201]]), 2, torch.tensor([[1., 3., 5.]]))
    assert abs(rows[0]["ade_px"] - 2) < 1e-5
    assert abs(rows[0]["relative_ade"] - .1) < 1e-5
    assert rows[1]["fde_px"] is None and rows[1]["error_5s_px"] is None
    assert rows[1]["relative_ade"] is None and abs(rows[1]["static_drift_px"] - 2) < 1e-5
    all_rows, cases = [], []
    for item in plan:
        current = []
        for name in ("posterior_mean", "persistence", "correct_language/sample_0",
                     "correct_language/sample_1", "no_language/sample_0", "shuffled_language/sample_0"):
            current.extend({**row, "window_id": item["window_id"], "source": item["source"], "condition": name} for row in rows)
        all_rows.extend(current)
        cases.append({**item, "trajectory_metrics": evaluation_summaries(current)})
    first_out = out / "first"
    first_out.mkdir(exist_ok=True)
    save_video(first_out/"fixture.mp4", torch.zeros(5, 3, 201, 201, dtype=torch.uint8),
               prediction[0], truth[0, 2:], valid[0, 2:], 2)
    cases[0]["videos"] = {"correct_language/sample_0": "fixture.mp4"}
    write_report(first_out, args, {"step": 2500}, all_rows, cases)
    first = json.loads((first_out/"report.json").read_text())
    second = deepcopy(first)
    second["checkpoint"]["step"] = 5000
    for case in second["cases"]:
        case["trajectory_metrics"]["correct_language/sample_0"]["ade_px"]["mean"] -= 1
    second_out = out / "second"
    second_out.mkdir(exist_ok=True)
    (second_out/"report.json").write_text(json.dumps(second))
    report = compare_reports(first, second, bootstrap=100)
    primary = report["comparisons"][report["primary"]]["summaries"]["all"]["ade_px"]
    assert primary["episodes"] == 6 and primary["delta_second_minus_first"] == -1
    assert primary["delta_ci95"] == [-1, -1]
    (out/"comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"test": "cpu_tiny_evaluation_integration", "status": "passed",
                      "fixed_plan_order_invariant": True, "paired_noise_repeatable": True,
                      "model_only_dcp_load": True, "video_written": True,
                      "trajectory_units_and_invisible_endpoint": True, "paired_episode_delta": primary,
                      "auxiliary": auxiliary}), flush=True)


if __name__ == "__main__":
    main()
