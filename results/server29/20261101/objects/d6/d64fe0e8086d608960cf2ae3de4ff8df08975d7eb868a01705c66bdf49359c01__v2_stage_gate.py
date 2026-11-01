#!/usr/bin/env python3
"""Fail-closed stage gate for all V2 orchestration.

This deliberately has no subprocess-launch feature.  A caller must record every
worker exit code and the completed artifacts, then invoke this gate before it is
allowed to launch a later stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--actual", type=int, required=True)
    parser.add_argument("--worker-exit-codes", required=True, help="comma-separated integer exit codes")
    parser.add_argument("--artifacts-complete", choices=("PASS", "FAIL"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    exits = [int(value) for value in args.worker_exit_codes.split(",") if value.strip()]
    checks = {
        "EXPECTED_SAMPLE_COUNT": args.expected,
        "ACTUAL_SUCCESS_COUNT": args.actual,
        "WORKER_EXIT_CODES": exits,
        "WORKER_EXIT_GATE": "PASS" if exits and all(code == 0 for code in exits) else "FAIL",
        "COUNT_GATE": "PASS" if args.actual == args.expected else "FAIL",
        "ARTIFACT_COMPLETENESS": args.artifacts_complete,
    }
    passed = all(checks[key] == "PASS" for key in ("WORKER_EXIT_GATE", "COUNT_GATE", "ARTIFACT_COMPLETENESS"))
    checks.update({
        "STAGE_GATE": "PASS" if passed else "FAIL",
        "DO_NOT_START_NEXT_STAGE": not passed,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
