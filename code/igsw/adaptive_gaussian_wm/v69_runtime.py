"""Shared execution configuration, metric records, and foreground DDP utilities."""

import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist

from .v69_config import ObjectVideoConfigV69


def add_v69_arguments(parser):
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--encoder", choices=("dinov3_vitl16", "vjepa2_1_vitl16"), default="dinov3_vitl16")
    parser.add_argument("--encoder_repository", required=True)
    parser.add_argument("--encoder_weights", required=True)
    parser.add_argument("--encoder_frame_batch", type=int, default=2)
    parser.add_argument("--config", default="")
    parser.add_argument("--stage", choices=("state", "dynamics"), default="state")
    parser.add_argument("--state_checkpoint", default="")
    parser.add_argument("--posterior_geometry", choices=("inherit", "on", "off"), default="inherit")
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--global_batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--stop_after", type=int, default=0, help="Optional execution stop; scheduler still uses --steps.")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic kernels for controlled resume comparisons.")
    parser.add_argument("--resume_trace", action="store_true", help="Save per-microbatch data, RNG and posterior evidence for the integration test.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=2500)
    parser.add_argument("--recovery_every", type=int, default=250)
    parser.add_argument("--source_revision", default="local-unversioned")
    parser.add_argument("--wandb_project", default="instruct-gs-world")
    parser.add_argument("--wandb_entity", default="healenrenss-university-of-chinese-acadmic-and-science")
    parser.add_argument("--wandb_name", default="object_video_sequence_v69")
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    return parser


def config_from_args(args):
    values = json.loads(Path(args.config).read_text()) if args.config else {}
    config = ObjectVideoConfigV69(**{**values, "encoder": args.encoder})
    return replace(config, posterior_geometry=args.posterior_geometry == "on") if args.posterior_geometry != "inherit" else config


def batch_is_readable(batch, context, device):
    readable = torch.tensor(int("decode_errors" not in batch), device=device)
    if context.distributed:
        dist.all_reduce(readable, op=dist.ReduceOp.MIN)
    if "decode_errors" in batch:
        print(json.dumps({"event": "decode_skip", "rank": context.rank, "errors": batch["decode_errors"]}), flush=True)
    return bool(readable)


def case_metrics_v69(output, batch, config, stage):
    th = config.history_frames
    valid = batch["teacher"]["valid"] if stage == "state" else batch["teacher"]["valid"][:, th:]
    times = batch["times"] if stage == "state" else batch["times"][:, th:]
    rows = []
    for item in range(len(valid)):
        for frame in range(valid.shape[1]):
            all_values = output["epe_px"][item, frame][valid[item, frame]].detach().float().cpu()
            primary = valid[item, frame] & batch["teacher"]["transport_weight"][item].bool()
            primary_values = output["epe_px"][item, frame][primary].detach().float().cpu()
            row = {"case": batch["case_id"][item], "source": batch["source"][item],
                   "seconds": float(times[item, frame]), "valid_points": len(all_values),
                   "transport_points": len(primary_values), "epe_px": all_values.tolist(), "transport_epe_px": primary_values.tolist(),
                   "selection_status": batch["transport_selection_status"][item]}
            for name, values in (("all", all_values), ("transport", primary_values)):
                row[f"{name}_p50"] = float(values.quantile(.5)) if len(values) else None
                row[f"{name}_p90"] = float(values.quantile(.9)) if len(values) else None
            rows.append(row)
    return rows


def append_case_records(path, rows):
    with Path(path).open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
