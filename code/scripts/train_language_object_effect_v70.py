#!/usr/bin/env python3
"""Foreground torchrun entry point for native V70 FSDP training."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.distributed as dist

from igsw.adaptive_gaussian_wm.v70_training import add_training_arguments_v70, train_v70


def main():
    args = add_training_arguments_v70(argparse.ArgumentParser(description=__doc__)).parse_args()
    train_v70(args)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
