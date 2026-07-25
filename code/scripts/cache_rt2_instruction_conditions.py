"""Cache one frozen Qwen text embedding per unique strict-causal instruction."""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from igsw.adaptive_gaussian_wm.conditioning import (  # noqa: E402
    CONDITION_CACHE_VERSION,
    TOKEN_CONDITION_CACHE_VERSION,
    normalize_instruction,
)
from igsw.dynamics.conditioning import (  # noqa: E402
    DEFAULT_QWEN_PATH,
    QwenVLEncoder,
)
from igsw.latent_particle_wm.pair_targets import (  # noqa: E402
    CAUSAL_PAIR_VERSION,
)


_PAIR_SUFFIX = re.compile(r"_t\d+_u\d+\.pt$")


def _representative_pairs(root: str) -> list[str]:
    representatives = {}
    for path in sorted(glob.glob(os.path.join(root, "*.pt"))):
        source_key = _PAIR_SUFFIX.sub("", os.path.basename(path))
        representatives.setdefault(source_key, path)
    if not representatives:
        raise ValueError(f"no strict-causal pair files in {root}")
    return list(representatives.values())


def _move_inputs(inputs: dict, device: torch.device) -> dict:
    result = {}
    for name, value in inputs.items():
        if not torch.is_tensor(value):
            result[name] = value
        elif value.is_floating_point():
            result[name] = value.to(device, dtype=torch.bfloat16)
        else:
            result[name] = value.to(device)
    return result


def _instruction_span_mask(
    input_ids: torch.Tensor,
    instruction_ids: torch.Tensor,
) -> torch.Tensor:
    if input_ids.ndim != 1 or instruction_ids.ndim != 1:
        raise ValueError("instruction span ids must be one-dimensional")
    width = int(instruction_ids.numel())
    if width == 0 or width > input_ids.numel():
        raise ValueError("instruction token span is empty or too long")
    matches = [
        start
        for start in range(input_ids.numel() - width + 1)
        if torch.equal(input_ids[start : start + width], instruction_ids)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"instruction token span must match exactly once, got {len(matches)}"
        )
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    mask[matches[0] : matches[0] + width] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_QWEN_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--token_features", action="store_true")
    args = parser.parse_args()
    for key, value in {
        "HF_HOME": "/mnt/pfs/public/xuhaoming/hf_cache",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": "/mnt/pfs/public/xuhaoming/.cache",
    }.items():
        os.environ.setdefault(key, value)

    instructions = set()
    for path in _representative_pairs(args.data):
        pair = torch.load(path, map_location="cpu", weights_only=False)
        if pair.get("pair_version") != CAUSAL_PAIR_VERSION:
            raise ValueError(f"causal pair version mismatch: {path}")
        instructions.add(normalize_instruction(pair.get("instruction", "")))
    ordered = sorted(instructions)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    encoder = QwenVLEncoder(args.model).to(device).eval()
    features = []
    token_features = []
    with torch.inference_mode():
        for index, instruction in enumerate(ordered, 1):
            inputs = _move_inputs(
                encoder.build_inputs(instruction, None),
                device,
            )
            hidden, _, _ = encoder(inputs)
            instruction_ids = encoder.processor.tokenizer(
                instruction,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0].to(device)
            instruction_mask = _instruction_span_mask(
                inputs["input_ids"][0],
                instruction_ids,
            )
            tokens = hidden[-1][instruction_mask].float().cpu()
            features.append(tokens.mean(dim=0))
            token_features.append(tokens)
            print(
                f"[condition-cache] {index}/{len(ordered)} {instruction!r}",
                flush=True,
            )
    feature_tensor = torch.stack(features).contiguous()
    feature_sha256 = hashlib.sha256(
        feature_tensor.numpy().tobytes()
    ).hexdigest()
    payload = {
        "version": (
            TOKEN_CONDITION_CACHE_VERSION
            if args.token_features
            else CONDITION_CACHE_VERSION
        ),
        "model": os.path.abspath(args.model),
        "instructions": ordered,
        "features": feature_tensor,
        "feature_dim": int(feature_tensor.shape[1]),
        "feature_sha256": feature_sha256,
        "pair_root": os.path.abspath(args.data),
        "pair_version": CAUSAL_PAIR_VERSION,
    }
    if args.token_features:
        max_tokens = max(len(value) for value in token_features)
        token_tensor = torch.zeros(
            len(token_features),
            max_tokens,
            feature_tensor.shape[1],
        )
        token_valid = torch.zeros(
            len(token_features),
            max_tokens,
            dtype=torch.bool,
        )
        for index, value in enumerate(token_features):
            token_tensor[index, : len(value)] = value
            token_valid[index, : len(value)] = True
        token_hasher = hashlib.sha256()
        token_hasher.update(token_tensor.numpy().tobytes())
        token_hasher.update(token_valid.numpy().tobytes())
        payload["token_features"] = token_tensor.contiguous()
        payload["token_valid"] = token_valid.contiguous()
        payload["token_feature_sha256"] = token_hasher.hexdigest()
    output = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = f"{output}.tmp.{os.getpid()}"
    torch.save(payload, temporary)
    os.replace(temporary, output)
    print(
        f"[condition-cache] DONE instructions={len(ordered)} "
        f"feature_dim={feature_tensor.shape[1]} out={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
