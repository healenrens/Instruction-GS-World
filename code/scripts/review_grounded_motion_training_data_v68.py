#!/usr/bin/env python3
"""Export actual loader samples from a frozen training manifest, on CPU."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igsw.adaptive_gaussian_wm.grounded_motion_training_review_v68 import write_training_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", default="0")
    args = parser.parse_args()
    write_training_review(args.manifest, args.out, points=args.points, seed=args.seed,
                          epochs=tuple(int(e) for e in args.epochs.split(",")))


if __name__ == "__main__":
    main()
