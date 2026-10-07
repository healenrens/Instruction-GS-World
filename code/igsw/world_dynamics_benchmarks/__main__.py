"""python -m igsw.world_dynamics_benchmarks --config FILE COMMAND."""

import argparse
import json

from .io import load_config
from .prepare import build_manifest, download_plan, inspect_schema, unpack, official_physion_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    download = sub.add_parser("download")
    download.add_argument("--plan")
    extract = sub.add_parser("unpack")
    extract.add_argument("--destination", required=True)
    extract.add_argument("--join-parts", action="store_true")
    extract.add_argument("archives", nargs="+")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("path")
    sub.add_parser("manifest")
    comparison = sub.add_parser("compare")
    comparison.add_argument("--split", choices=["dev", "test"], default="dev")
    export = sub.add_parser("export")
    export.add_argument("--model", choices=["dino", "vjepa2", "state", "state_z"], required=True)
    for command in ("probe", "evaluate"):
        runner = sub.add_parser(command)
        runner.add_argument("--model", choices=["dino", "vjepa2", "state", "state_z"], required=True)
        runner.add_argument("--protocol", choices=["attention", "linear"], default="attention")
        runner.add_argument("--resume", action="store_true")
        if command == "evaluate":
            runner.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "plan":
        import shutil
        from pathlib import Path
        root = Path(config["data"]["root"])
        root.mkdir(parents=True, exist_ok=True)
        plan = config.get("downloads", [])
        if config["benchmark"] == "physion":
            plan = official_physion_plan(root, config["data"]["scenarios"], config["data"]["download_limit"])
        print(json.dumps({"disk_free_bytes": shutil.disk_usage(root).free, "downloads": plan,
                          "scope": config["scope"],
                          "raw_feature_upper_bound_bytes_per_clip": config["export"]["token_budget"] * 1024 * 2,
                          "note": "FP16 cache; full SSv2 raw-token export can exceed 1TB per representation"}, indent=2))
    elif args.command == "download":
        download_plan(config, args.plan)
    elif args.command == "unpack":
        unpack(args.archives, args.destination, args.join_parts)
    elif args.command == "inspect":
        print(json.dumps(inspect_schema(args.path), indent=2))
    elif args.command == "manifest":
        build_manifest(config)
    elif args.command == "compare":
        from .compare import compare_models
        compare_models(config, args.split)
    elif args.command == "export":
        from .features import export_features
        export_features(config, args.model)
    else:
        from .probe import run_probe
        run_probe(config, args.model, args.protocol, args.command, args.resume,
                  args.split if args.command == "evaluate" else "dev")


if __name__ == "__main__":
    main()
