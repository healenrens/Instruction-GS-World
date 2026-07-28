#!/usr/bin/env python3
"""Compatibility entrypoint for the v30 Object Memory verifier."""

from __future__ import annotations

import os
import runpy


SCRIPT = os.path.join(
    os.path.dirname(__file__),
    "verify_object_memory_jepa_v28.py",
)
runpy.run_path(SCRIPT, run_name="__main__")
