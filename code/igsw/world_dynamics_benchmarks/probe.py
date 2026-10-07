"""Capacity-controlled token readout and a separate per-scenario linear reference."""

import csv
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import unique_indices
from .io import experiment_root, read_json, save_tensor, write_json


class TokenDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return torch.load(row["feature_path"], map_location="cpu", weights_only=False), row


def collate(items):
    length = max(len(item[0]["tokens"]) for item in items)
    values = torch.zeros(len(items), length, 1024)
    metadata = torch.zeros(len(items), length, 5)
    valid = torch.zeros(len(items), length, dtype=torch.bool)
    labels = torch.tensor([item[1]["label"] for item in items])
    for index, (feature, _) in enumerate(items):
        count = len(feature["tokens"])
        values[index, :count, :feature["tokens"].shape[-1]] = feature["tokens"]
        metadata[index, :count], valid[index, :count] = feature["metadata"], feature["valid"]
    return {"tokens": values, "metadata": metadata, "valid": valid, "labels": labels,
            "rows": [item[1] for item in items]}


class TokenProbe(nn.Module):
    def __init__(self, settings, classes):
        super().__init__()
        width = settings["width"]
        self.content = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, width))
        self.position = nn.Linear(5, width)
        self.query = nn.Parameter(torch.zeros(1, 1, width))
        self.attention = nn.MultiheadAttention(width, settings["heads"], batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, classes))

    def forward(self, batch):
        context = self.content(batch["tokens"]) + self.position(batch["metadata"])
        hidden, _ = self.attention(self.query.expand(len(context), -1, -1), context, context,
                                   key_padding_mask=~batch["valid"], need_weights=False)
        return self.head(hidden[:, 0])


def loader(rows, settings, seed=None):
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(TokenDataset(rows), batch_size=settings["batch_size"], shuffle=seed is not None,
                      generator=generator, collate_fn=collate, num_workers=0)


def to_device(batch, device):
    return {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in batch.items()}


def state_rng():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(saved):
    torch.set_rng_state(saved["torch"])
    np.random.set_state(saved["numpy"])
    random.setstate(saved["python"])
    if saved["cuda"]:
        torch.cuda.set_rng_state_all(saved["cuda"])


def wilson(correct):
    count, p, z = len(correct), float(np.mean(correct)), 1.96
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [float(center - radius), float(center + radius)]


def summary(cases):
    correct = np.array([case["prediction"] == case["label"] for case in cases])
    labels = sorted({case["label"] for case in cases})
    classes = {str(label): {"count": sum(case["label"] == label for case in cases),
                           "accuracy": float(np.mean([case["prediction"] == label for case in cases
                                                      if case["label"] == label]))} for label in labels}
    return {"count": len(cases), "accuracy": float(correct.mean()), "accuracy_ci95_wilson": wilson(correct),
            "balanced_accuracy_present_classes": float(np.mean([value["accuracy"] for value in classes.values()])),
            "top5": float(np.mean([case["label"] in case["top5"] for case in cases])), "classes": classes}


@torch.no_grad()
def attention_cases(model, rows, settings, device):
    model.eval()
    cases = []
    for batch in loader(rows, settings):
        inputs = to_device(batch, device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        probabilities = model(inputs).softmax(-1).cpu()
        elapsed = time.perf_counter() - start
        for index, row in enumerate(batch["rows"]):
            cases.append({"id": row["id"], "path": row["path"], "scenario": row["scenario"],
                "official_split": row["official_split"], "label": row["label"],
                "prediction": int(probabilities[index].argmax()), "probabilities": probabilities[index].tolist(),
                "top5": probabilities[index].topk(min(5, probabilities.shape[-1])).indices.tolist(),
                "readout_seconds_per_item_in_batch": elapsed / len(batch["rows"]),
                "valid_tokens": int(batch["valid"][index].sum())})
    return cases


def train_attention(config, model_name, manifest, resume=False, stop_step=None):
    settings, device = config["probe"], config["device"]
    root = experiment_root(config, model_name) / "attention"
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    model = TokenProbe(settings, manifest["num_classes"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    epoch, cursor, step, best = 0, 0, 0, -1.0
    if resume:
        saved = torch.load(root / "latest.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        epoch, cursor, step, best = saved["epoch"], saved["cursor"], saved["step"], saved["best"]
        restore_rng(saved["rng"])
    train = [row for row in manifest["rows"] if row["split"] == "train"]
    dev = [row for row in manifest["rows"] if row["split"] == "dev"]
    trace_path = root / "trace.jsonl"
    if resume:
        # An interrupted process can log updates after the most recent committed checkpoint.
        committed = [line for line in trace_path.read_text().splitlines()
                     if json.loads(line)["step"] <= step]
        temporary = trace_path.with_suffix(".partial")
        temporary.write_text("\n".join(committed) + "\n")
        temporary.replace(trace_path)
    # Exclusive creation makes a new run explicit. Resume appends after its committed data cursor.
    mode = "a" if resume else "x"
    with trace_path.open(mode) as trace:
        def save(next_epoch, next_cursor):
            save_tensor(root / "latest.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": next_epoch, "cursor": next_cursor, "step": step, "best": best,
                "rng": state_rng(), "config": config})

        for current_epoch in range(epoch, settings["epochs"]):
            batches = loader(train, settings, seed=config["seed"] + current_epoch)
            model.train()
            for batch_index, batch in enumerate(batches):
                if current_epoch == epoch and batch_index < cursor:
                    continue
                inputs = to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(inputs), inputs["labels"])
                loss.backward()
                optimizer.step()
                step += 1
                record = {"epoch": current_epoch, "batch": batch_index, "step": step,
                          "ids": [row["id"] for row in batch["rows"]], "loss": float(loss.detach())}
                trace.write(json.dumps(record) + "\n")
                trace.flush()
                print(json.dumps(record), flush=True)
                if step % settings["save_every"] == 0 or step == stop_step:
                    save(current_epoch, batch_index + 1)
                if step == stop_step:
                    return root
            metrics = summary(attention_cases(model, dev, settings, device))
            write_json(root / f"dev_epoch_{current_epoch:03d}.json", metrics)
            if metrics["accuracy"] > best:
                best = metrics["accuracy"]
                save_tensor(root / "best.pt", {"model": model.state_dict(), "epoch": current_epoch, "step": step,
                                               "selection": "internal train-derived dev accuracy", "config": config})
            save(current_epoch + 1, 0)
    return root


def linear_vector(row, budget):
    feature = torch.load(row["feature_path"], map_location="cpu", weights_only=False)
    valid = feature["valid"]
    padded = nn.functional.pad(feature["tokens"][valid].float(), (0, 1024 - feature["tokens"].shape[-1]))
    values = torch.cat((padded, feature["metadata"][valid]), -1)
    indices = unique_indices(len(values), budget)
    sampled = values[indices]
    # A shorter sequence is zero-padded, never duplicated to manufacture observations.
    return nn.functional.pad(sampled, (0, 0, 0, budget - len(sampled))).flatten().numpy()


def train_linear(config, model_name, manifest, resume=False):
    settings = config["probe"]
    root = experiment_root(config, model_name) / "linear"
    root.mkdir(parents=True, exist_ok=True)
    for scenario in sorted({row["scenario"] for row in manifest["rows"]}):
        path = root / (scenario + ".joblib")
        if resume and path.exists():
            continue
        rows = [row for row in manifest["rows"] if row["split"] == "train" and row["scenario"] == scenario]
        groups = {label: [row for row in rows if row["label"] == label] for label in {row["label"] for row in rows}}
        rng = random.Random(config["seed"])
        for group in groups.values():
            rng.shuffle(group)
        count = min(map(len, groups.values()))
        rows = [row for group in groups.values() for row in group[:count]]
        x = np.stack([linear_vector(row, settings["linear_tokens"]) for row in rows])
        y = np.array([row["label"] for row in rows])
        search = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=settings["linear_max_iter"])),
            {"logisticregression__C": np.logspace(*settings["linear_logspace"])},
            cv=StratifiedKFold(settings["linear_cv"], shuffle=True, random_state=config["seed"]),
            scoring="balanced_accuracy", n_jobs=1)
        search.fit(x, y)
        temporary = path.with_suffix(".partial")
        joblib.dump(search.best_estimator_, temporary)
        temporary.replace(path)
        write_json(root / (scenario + "_fit.json"), {"balanced_train_count": len(rows),
                   "best_C": search.best_params_, "train_only_cv_score": search.best_score_,
                   "feature_dim": x.shape[-1], "protocol": "per-scenario official-style logistic reference; bounded token flatten"})
    return root


def evaluate(config, model_name, protocol, manifest, split="dev"):
    settings = config["probe"]
    root = experiment_root(config, model_name) / protocol
    rows = [row for row in manifest["rows"] if row["split"] == split]
    if protocol == "attention":
        saved = torch.load(root / "best.pt", map_location=config["device"], weights_only=False)
        model = TokenProbe(settings, manifest["num_classes"]).to(config["device"])
        model.load_state_dict(saved["model"])
        cases = attention_cases(model, rows, settings, config["device"])
        parameters = sum(parameter.numel() for parameter in model.parameters())
    else:
        cases, parameters = [], {}
        for scenario in sorted({row["scenario"] for row in rows}):
            model = joblib.load(root / (scenario + ".joblib"))
            selected = [row for row in rows if row["scenario"] == scenario]
            x = np.stack([linear_vector(row, settings["linear_tokens"]) for row in selected])
            probabilities = model.predict_proba(x)
            classes = model.classes_
            parameters[scenario] = int(model[-1].coef_.size + model[-1].intercept_.size)
            for row, p in zip(selected, probabilities):
                cases.append({"id": row["id"], "path": row["path"], "scenario": scenario,
                    "official_split": row["official_split"], "label": row["label"],
                    "prediction": int(classes[p.argmax()]), "probabilities": p.tolist(),
                    "probability_classes": classes.tolist(), "top5": classes[p.argsort()[-5:][::-1]].tolist()})
    features_by_id = {row["id"]: row["feature_path"] for row in rows}
    with (root / f"cases_{split}.jsonl").open("w") as stream:
        for case in cases:
            feature = torch.load(features_by_id[case["id"]],
                                 map_location="cpu", weights_only=False)
            case.update({key: feature[key] for key in ("frame_indices", "times", "time_basis", "native_hw", "decode_seconds", "encoder_seconds", "token_sampling")})
            # Existing SSv2 natural-RGB caches precede the explicit cue metadata field.
            case["input_cue"] = feature.get("input_cue", {"kind": "natural_rgb"})
            stream.write(json.dumps(case) + "\n")
    report = {"benchmark": manifest["benchmark"], "scope": manifest["scope"], "model": model_name,
              "representation": manifest["representation"], "input_protocol": manifest["protocol"],
              "readout_protocol": protocol, "readout_parameters": parameters,
              "expected_num_classes": manifest["num_classes"],
              "observed_evaluation_class_count": len({row["label"] for row in cases}),
              "split": split,
              "evaluation_role": "development_selection" if split == "dev" else "held_evaluation",
              "evaluation_partition": "internal train-derived dev" if split == "dev" else (
                  "official validation" if manifest["benchmark"] == "ssv2" else "official testing"),
              "official_score_reproduction": False, "overall": summary(cases),
              "scenario_macro_accuracy": float(np.mean([summary([case for case in cases if case["scenario"] == scenario])[
                  "balanced_accuracy_present_classes"] for scenario in sorted({case["scenario"] for case in cases})])),
              "scenarios": {scenario: summary([case for case in cases if case["scenario"] == scenario])
                            for scenario in sorted({case["scenario"] for case in cases})}}
    write_json(root / f"report_{split}.json", report)
    with (root / f"confusion_{split}.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["truth", "prediction", "count"])
        from collections import Counter
        writer.writerows((truth, prediction, count) for (truth, prediction), count in
                        sorted(Counter((case["label"], case["prediction"]) for case in cases).items()))
    print(json.dumps(report), flush=True)
    return report


def run_probe(config, model_name, protocol, command, resume, split="dev"):
    manifest = read_json(experiment_root(config, model_name) / "features.json")
    if command == "probe":
        return (train_attention if protocol == "attention" else train_linear)(config, model_name, manifest, resume)
    return evaluate(config, model_name, protocol, manifest, split)
