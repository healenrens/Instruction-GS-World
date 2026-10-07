"""Configuration and atomic experiment artifacts."""

import json
import os
from pathlib import Path

import torch
import yaml


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def load_config(path):
    return yaml.safe_load(os.path.expandvars(Path(path).read_text()))


def experiment_root(config, model):
    return Path(config["output_root"]) / config["benchmark"] / model / f"seed{config['seed']}" / config["attempt"]
