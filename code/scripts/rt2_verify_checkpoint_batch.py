"""Compare real-checkpoint single-sample inference with the batched training path.

The comparison uses prepared training clips and fixed ODE initial noise.  It
separates two effects:
  * single inference vs a batch of one (different Qwen/grid helper paths), and
  * a batch of one vs a padded multi-clip batch (batching and mask behavior).
"""
import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "scripts"))

from train_vla import _to_dev, build_batch_padded, make_model  # noqa: E402


def _model_args(saved, norm_stats):
    defaults = {
        "geom_mode": "xyz", "fdim": 128, "feat_source": "qwen", "dino_imgsize": 518,
        "traj_pred": 0, "img_loss": 1, "w_depth": 0.5, "cam_cond": 0, "L": 512,
        "beta": 30.0, "placement": "entropy", "wrist": 1, "action_dim": 14,
        "action_steps": 50, "d_act": 704, "n_heads_act": 11, "n_state_tokens": 1,
        "mlp_ratio": 4.0, "w_flow": 1.0, "w_act": 1.0,
    }
    values = {key: saved.get(key, value) for key, value in defaults.items()}
    values.update(init_from="", norm_stats=norm_stats)
    return SimpleNamespace(**values)


def _load_model(checkpoint, norm_stats, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = _model_args(payload.get("args", {}), norm_stats)
    model = make_model(args, device, Kf=12)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    missing = [key for key in missing if not key.startswith("encoder.")]
    if missing or unexpected:
        raise ValueError(f"checkpoint state mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    print(f"[verify] loaded checkpoint step={int(payload['step'])}: {checkpoint}", flush=True)
    return model, args, payload


def _max_abs(a, b):
    return float((a.float() - b.float()).abs().max().item())


def _mean_abs(a, b):
    return float((a.float() - b.float()).abs().mean().item())


def _cosine(a, b):
    return float(F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item())


def _run_single(model, clip, x0, n_steps):
    x, tok, layers, vlm, ctxm = model._trunk_features(clip)
    state = model.action_expert.embed_state(clip["anchor"].float()[None])
    z = model.action_expert.sample(
        layers, vlm, state, trunk_mask=None, vlm_mask=ctxm, n_steps=n_steps,
        device=x.device, dtype=torch.float32, x0=x0,
    )
    return {
        "x": x.detach(), "tok": tok.detach(),
        "layers": [value.detach() for value in layers],
        "vlm": [value.detach() for value in vlm],
        "action": model.act_norm.denormalize(z[0]).detach(),
    }


def _run_batch(model, clips, args, x0, n_steps):
    batch = build_batch_padded(clips, clips[0]["tok_xyz0"].device, args, model.encoder)
    x, tok, layers, vlm, ctxm = model._trunk_features_batch(batch)
    state = model.action_expert.embed_state(batch["anchor"].float())
    z = model.action_expert.sample(
        layers, vlm, state, trunk_mask=batch["tok_mask"], vlm_mask=ctxm, n_steps=n_steps,
        device=x.device, dtype=torch.float32, x0=x0,
    )
    return batch, {
        "x": x.detach(), "tok": tok.detach(),
        "layers": [value.detach() for value in layers],
        "vlm": [value.detach() for value in vlm],
        "action": model.act_norm.denormalize(z).detach(),
    }


def _compare(single, batched, index, token_count):
    action_a, action_b = single["action"], batched["action"][index]
    trunk_a, trunk_b = single["x"][0, :token_count], batched["x"][index, :token_count]
    tok_a, tok_b = single["tok"][:token_count], batched["tok"][index, :token_count]
    layer_diffs = [
        _max_abs(a[0, :token_count], b[index, :token_count])
        for a, b in zip(single["layers"], batched["layers"])
    ]
    layer_cosines = [
        _cosine(a[0, :token_count], b[index, :token_count])
        for a, b in zip(single["layers"], batched["layers"])
    ]
    vlm_diffs = [_max_abs(a[0], b[index]) for a, b in zip(single["vlm"], batched["vlm"])]
    vlm_cosines = [_cosine(a[0], b[index]) for a, b in zip(single["vlm"], batched["vlm"])]
    return {
        "token_feature_max_abs": _max_abs(tok_a, tok_b),
        "token_feature_cosine": _cosine(tok_a, tok_b),
        "trunk_final_max_abs": _max_abs(trunk_a, trunk_b),
        "trunk_final_cosine": _cosine(trunk_a, trunk_b),
        "trunk_layer_max_abs": max(layer_diffs),
        "trunk_layer_min_cosine": min(layer_cosines),
        "vlm_context_max_abs": max(vlm_diffs),
        "vlm_context_min_cosine": min(vlm_cosines),
        "action_max_abs": _max_abs(action_a, action_b),
        "action_mean_abs": _mean_abs(action_a, action_b),
        "action_cosine": _cosine(action_a, action_b),
        "action_reference_p95_abs": float(torch.quantile(action_a.float().abs(), 0.95).item()),
        "endpoint_max_abs": _max_abs(action_a.cumsum(0)[-1], action_b.cumsum(0)[-1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prep_cache", required=True)
    parser.add_argument("--norm_stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_clips", type=int, default=2)
    parser.add_argument("--clips", nargs="*", default=[],
                        help="optional prep-cache paths or basenames; overrides --num_clips selection")
    parser.add_argument("--n_steps", type=int, default=10)
    args_cli = parser.parse_args()

    if args_cli.clips:
        paths = [path if os.path.isabs(path) else os.path.join(args_cli.prep_cache, path)
                 for path in args_cli.clips]
    else:
        paths = sorted(glob.glob(os.path.join(args_cli.prep_cache, "*_train.pt")))[:args_cli.num_clips]
    missing_paths = [path for path in paths if not os.path.isfile(path)]
    if missing_paths:
        raise ValueError(f"prep-cache clips not found: {missing_paths}")
    if not paths:
        raise ValueError("no prep-cache clips selected")

    torch.manual_seed(0)
    device = "cuda"
    model, model_args, payload = _load_model(args_cli.checkpoint, args_cli.norm_stats, device)
    clips = [_to_dev(torch.load(path, map_location="cpu", weights_only=False), device) for path in paths]
    x0 = [torch.randn(1, model_args.action_steps, model_args.action_dim, device=device) for _ in clips]

    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    with torch.no_grad(), amp:
        single = [_run_single(model, clip, noise, args_cli.n_steps) for clip, noise in zip(clips, x0)]
        batch1 = [_run_batch(model, [clip], model_args, noise, args_cli.n_steps)[1]
                  for clip, noise in zip(clips, x0)]
        batch_n_data, batch_n = _run_batch(model, clips, model_args, torch.cat(x0, 0), args_cli.n_steps)

    rows = []
    for index, (path, clip) in enumerate(zip(paths, clips)):
        token_count = int(clip["M"])
        single_vs_batch1 = _compare(single[index], batch1[index], 0, token_count)
        batch1_as_single = {
            "x": batch1[index]["x"][0:1],
            "tok": batch1[index]["tok"][0],
            "layers": [value[0:1] for value in batch1[index]["layers"]],
            "vlm": [value[0:1] for value in batch1[index]["vlm"]],
            "action": batch1[index]["action"][0],
        }
        batch1_vs_batch_n = _compare(batch1_as_single, batch_n, index, token_count)
        rows.append({
            "clip": os.path.abspath(path),
            "tokens": token_count,
            "vlm_sequence": int(clip["vlm0"]["input_ids"].shape[1]),
            "single_vs_batch1": single_vs_batch1,
            "batch1_vs_batchN": batch1_vs_batch_n,
        })

    report = {
        "checkpoint": os.path.abspath(args_cli.checkpoint),
        "checkpoint_step": int(payload["step"]),
        "prep_cache": os.path.abspath(args_cli.prep_cache),
        "batch_size": len(clips),
        "padded_tokens": int(batch_n_data["tok_mask"].shape[1]),
        "n_steps": args_cli.n_steps,
        "clips": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
    with open(args_cli.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
