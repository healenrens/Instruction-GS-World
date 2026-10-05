"""SwanLab SDK training scalars only; credentials remain in the environment."""

import json
import os
from pathlib import Path


def add_tracking_arguments_v70(parser):
    parser.add_argument("--swanlab_project", default=os.environ.get(
        "SWANLAB_PROJ_NAME", "instruct-gs-world"))
    parser.add_argument("--swanlab_workspace", default=os.environ.get("SWANLAB_WORKSPACE_NAME", ""))
    parser.add_argument("--swanlab_name", default="language_object_effect_v70")
    parser.add_argument("--swanlab_mode", choices=("online", "offline", "local", "disabled"),
                        default=os.environ.get("SWANLAB_MODE", "online"))
    return parser


def start_tracking_v70(args, config, saved=None):
    """Rank zero only. Restore the exact run ID, never infer resume from a directory."""
    if args.swanlab_mode == "disabled":
        return None, {"backend": "swanlab", "id": None, "mode": "disabled"}
    # The SDK's structured environment fields must not parse our plain names or
    # override checkpoint-owned project/workspace/run identity.
    for name in ("SWANLAB_PROJECT", "SWANLAB_WORKSPACE", "SWANLAB_RUN_ID", "SWANLAB_RESUME"):
        os.environ.pop(name, None)
    import swanlab
    saved = saved or {}
    run = swanlab.init(
        project=saved.get("project", args.swanlab_project),
        workspace=saved.get("workspace", args.swanlab_workspace) or None,
        name=args.swanlab_name, mode=args.swanlab_mode,
        log_dir=str(Path(args.out) / "swanlog"), config=config,
        id=saved.get("id"), resume="must" if saved.get("id") else "never",
    )
    tracking = {"backend": "swanlab", "id": run.id,
                "project": saved.get("project", args.swanlab_project),
                "workspace": saved.get("workspace", args.swanlab_workspace),
                "mode": args.swanlab_mode, "scope": "training_scalars_only"}
    path = Path(args.out) / "tracking.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(tracking, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return run, tracking


def log_training_scalars_v70(run, values, step):
    if run is not None:
        run.log({key: float(value) for key, value in values.items()}, step=step)


def finish_tracking_v70(run):
    if run is not None:
        run.finish()
