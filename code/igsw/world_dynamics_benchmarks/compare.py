"""Paired example-level comparisons, separate from tuning and training."""

import json
from pathlib import Path

import numpy as np

from .io import experiment_root, read_json, write_json


def compare_models(config, split="dev"):
    reports, cases = {}, {}
    for model in ("dino", "vjepa2", "state", "state_z"):
        root = experiment_root(config, model) / "attention"
        reports[model] = read_json(root / f"report_{split}.json")
        cases[model] = {row["id"]: row for row in map(json.loads, (root / f"cases_{split}.jsonl").read_text().splitlines())}
    shared = sorted(set.intersection(*(set(rows) for rows in cases.values())))
    comparisons = {}
    rng = np.random.default_rng(config["seed"])
    for baseline in ("state", "dino", "vjepa2"):
        differences = np.array([float(cases["state_z"][key]["prediction"] == cases["state_z"][key]["label"]) -
                               float(cases[baseline][key]["prediction"] == cases[baseline][key]["label"])
                               for key in shared])
        bootstrap = [rng.choice(differences, len(differences), replace=True).mean() for _ in range(2000)]
        comparisons["state_z_vs_" + baseline] = {
            "paired_accuracy_gain": float(differences.mean()),
            "paired_ci95_bootstrap": np.quantile(bootstrap, [.025, .975]).tolist(),
            "improved_cases": int((differences > 0).sum()), "regressed_cases": int((differences < 0).sum())}
    output = Path(config["output_root"]) / config["benchmark"] / "comparisons" / f"seed{config['seed']}" / config["attempt"]
    output.mkdir(parents=True, exist_ok=True)
    report = {"scope": config["scope"], "protocol": config["protocol"], "split": split,
              "evaluation_role": "development_selection" if split == "dev" else "held_evaluation",
              "paired_count": len(shared),
              "unpaired_counts": {model: len(rows) - len(shared) for model, rows in cases.items()},
              "comparisons": comparisons, "reports": reports,
              "interpretation": ("dev scores are selection-set results, not held generalization; " if split == "dev" else "")
                                + "task utility only; not object validity, deployment accuracy or official score reproduction"}
    write_json(output / f"comparison_{split}.json", report)
    with (output / f"paired_cases_{split}.jsonl").open("w") as stream:
        for key in shared:
            stream.write(json.dumps({"id": key, "models": {model: rows[key] for model, rows in cases.items()}}) + "\n")
    print(json.dumps({"comparison": str(output / f"comparison_{split}.json"), "comparisons": comparisons}), flush=True)
    return report
