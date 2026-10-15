#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igsw.object_event_wm.training import parse_args, train


if __name__ == "__main__":
    train(parse_args(default_stage="prior"))
