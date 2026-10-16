#!/usr/bin/env python3
"""Encode complete instructions with local T5, recording the effective token length."""

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import T5EncoderModel, T5Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igsw.object_event_wm.data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    instructions = sorted({""} | {window["instruction"] for row in manifest["entries"]
                                  for window in row["language_windows"]})
    tokenizer = T5Tokenizer.from_pretrained(args.model, local_files_only=True)
    model = T5EncoderModel.from_pretrained(args.model, local_files_only=True).to(args.device).eval().requires_grad_(False)
    lengths = [len(tokenizer(text, truncation=False)["input_ids"]) for text in instructions]
    sequence_length = max(32, max(lengths))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {"instructions": instructions, "width": model.config.d_model, "model": str(Path(args.model).resolve()),
                "manifest": str(Path(args.manifest).resolve()), "sequence_length": sequence_length,
                "nonempty_instructions": len(instructions)-1, "empty_instruction_cached": True,
                "upstream_token_limit": 32, "truncation": False, "token_lengths": lengths,
                "device": args.device, "torch_version": torch.__version__, "dtype": "float32",
                "above_upstream_token_limit": [index for index, length in enumerate(lengths) if length > 32],
                "padding_attention": "upstream recipe: padded T5 input without an attention mask"}
    write_json(out/"index.json", metadata)
    with torch.no_grad():
        for start in range(0, len(instructions), args.batch):
            tokens = tokenizer(instructions[start:start+args.batch], padding="max_length", max_length=sequence_length,
                               truncation=False, return_tensors="pt")["input_ids"].to(args.device)
            features = model(input_ids=tokens).last_hidden_state.cpu().float()
            for offset, value in enumerate(features):
                target = out/f"{start+offset:07d}.pt"
                temporary = target.with_suffix(".partial")
                torch.save(value.clone(), temporary)
                temporary.replace(target)
            print(json.dumps({"instructions_saved": min(start+args.batch, len(instructions)), "total": len(instructions)}), flush=True)
    write_json(out/"completion.json", {"instructions": len(instructions), "width": model.config.d_model,
                                       "sequence_length": sequence_length, "truncation": False})


if __name__ == "__main__":
    main()
