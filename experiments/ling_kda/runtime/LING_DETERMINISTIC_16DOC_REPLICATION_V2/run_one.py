#!/usr/bin/env python3
"""Thin, semantics-preserving wrapper around the audited Ling repeatability runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUDITED_RUNNER = (
    ROOT
    / "experiments"
    / "ling_kda"
    / "runtime"
    / "repeatability_closure_v1"
    / "run_ling_repeatability.py"
)


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "ling_repeatability_16doc_audited_runner", AUDITED_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import audited runner: {AUDITED_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # The only behavioral extension: permit the original ordered documents 0..15.
    # All model, quantizer, rotation, warmup, prefill, decode, metric, and fixed-
    # config behavior remains in the audited runner without reimplementation.
    module.SELECTED_INDICES = tuple(range(16))
    module.main()


if __name__ == "__main__":
    main()
