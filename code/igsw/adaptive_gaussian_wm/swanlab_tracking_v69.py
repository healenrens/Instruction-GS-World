"""V69 training cloud metrics, plus local-only non-training evidence."""

import json
from numbers import Real
import os
from pathlib import Path


def add_swanlab_arguments(parser, default_name="object_video_sequence_v69"):
    parser.add_argument("--swanlab_project", "--wandb_project", dest="swanlab_project",
                        default=os.environ.get("SWANLAB_PROJECT", "instruct-gs-world"))
    parser.add_argument("--swanlab_workspace", default=os.environ.get("SWANLAB_WORKSPACE", ""))
    parser.add_argument("--swanlab_name", "--wandb_name", dest="swanlab_name", default=default_name)
    parser.add_argument("--swanlab_mode", "--wandb_mode", dest="swanlab_mode",
                        choices=("online", "offline", "local", "disabled"), default=os.environ.get("SWANLAB_MODE", "online"))
    # Old submitted commands remain accepted; a W&B entity is not a SwanLab workspace.
    parser.add_argument("--wandb_entity", default="", help="Legacy argument; unused by SwanLab.")
    return parser


def start_swanlab_v69(args, group, job_type, config=None, checkpoint=None):
    if args.swanlab_mode == "online" and job_type not in ("state", "dynamics"):
        print(f"[swanlab-v69] training-only upload; {job_type} results remain local", flush=True)
        return None
    if args.swanlab_mode == "disabled":
        return None
    import swanlab
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state = (checkpoint.get("tracking") or {}) if checkpoint is not None else {}
    tracking_path = out / "tracking.json"
    if checkpoint is not None and not state and tracking_path.is_file():
        state = json.loads(tracking_path.read_text())
    continuing = state.get("backend") == "swanlab" and bool(state.get("id"))
    for name in ("SWANLAB_RUN_ID", "SWANLAB_RESUME"):
        os.environ.pop(name, None)
    project = state["project"] if continuing else args.swanlab_project
    workspace = state["workspace"] if continuing else args.swanlab_workspace
    run = swanlab.init(project=project, workspace=workspace or None, name=args.swanlab_name,
                       group=group, job_type=job_type, mode=args.swanlab_mode, log_dir=str(out / "swanlog"),
                       id=state["id"] if continuing else None, resume="must" if continuing else None,
                       config=config if config is not None else vars(args))
    tracking = {"backend": "swanlab", "id": run.id, "project": project, "workspace": workspace,
                "mode": args.swanlab_mode, "upload_scope": "training_scalars_only",
                "start_step": checkpoint["step"] if checkpoint is not None else 0}
    if args.swanlab_mode == "online":
        tracking.update(url=run.url, workspace=run.path.strip("/").split("/")[0])
    if checkpoint is not None and checkpoint.get("wandb_id"):
        tracking["previous_wandb_id"] = checkpoint["wandb_id"]
    elif state.get("previous_wandb_id"):
        tracking["previous_wandb_id"] = state["previous_wandb_id"]
    tracking_path.write_text(json.dumps(tracking, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "v69_tracking", **tracking}), flush=True)
    return run


def log_values_v69(run, values, step=None):
    import swanlab
    run.log({key: value if isinstance(value, Real) else
             swanlab.Text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
             for key, value in values.items()}, step=step)


def log_table_v69(run, name, columns, rows, step=None):
    import swanlab
    columns = list(columns)
    for start in range(0, max(1, len(rows)), 5000):
        data = [[row.get(key) for key in columns] if isinstance(row, dict) else list(row)
                for row in rows[start:start+5000]]
        table = swanlab.echarts.Table().add(columns, data)
        key = name if len(rows) <= 5000 else f"{name}_{start//5000:04d}"
        run.log({key: table}, step=step)


def set_results_v69(run, values):
    run.config.update({"results": {**run.config.get("results", {}), **values}})
    log_values_v69(run, {"results/report": values})


def log_video_v69(run, name, path, step=None):
    """SwanLab's public video media accepts GIF; original MP4 stays on shared storage."""
    import av
    import swanlab
    from PIL import Image
    path = Path(path)
    gif = path if path.suffix.lower() == ".gif" else path.with_suffix(".swanlab.gif")
    if path != gif:
        images = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            rate = float(stream.average_rate)
            stride = max(1, round(rate / 10))
            for index, frame in enumerate(container.decode(stream)):
                if index % stride == 0:
                    image = frame.to_image()
                    image.thumbnail((768, 768), Image.Resampling.LANCZOS)
                    images.append(image)
        images[0].save(gif, save_all=True, append_images=images[1:], duration=round(1000*stride/rate), loop=0)
    run.log({name: swanlab.Video(str(gif))}, step=step)


def log_evidence_v69(run, paths, base_path):
    """Upload readable reports as cloud media; public SwanLab has no artifact save API."""
    import swanlab
    base = Path(base_path).resolve()
    files = sorted({file.resolve() for path in (Path(value).resolve() for value in paths)
                    for file in (path.rglob("*") if path.is_dir() else [path])
                    if file.is_file() and "swanlog" not in file.relative_to(base).parts
                    and "evidence_upload" not in file.relative_to(base).parts})
    uploaded = []
    for path in files:
        name = path.relative_to(base).as_posix()
        key = "evidence/" + name
        if path.suffix in (".json", ".jsonl", ".csv"):
            with path.open(encoding="utf-8") as stream:
                part = 0
                content = stream.read(65536)
                while content:
                    run.log({f"{key}/part_{part:05d}": swanlab.Text(content, caption=name)})
                    part += 1
                    content = stream.read(65536)
            uploaded.append({"file": name, "parts": part, "representation": "text"})
        elif path.suffix == ".html":
            run.log({key: swanlab.Html(path)})
            uploaded.append({"file": name, "representation": "html"})
        elif path.suffix in (".png", ".jpg", ".jpeg"):
            run.log({key: swanlab.Image(str(path))})
            uploaded.append({"file": name, "representation": "image"})
    log_values_v69(run, {"evidence/files": uploaded})
