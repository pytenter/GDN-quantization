#!/usr/bin/env python3
"""Fail-closed consistency checks for the design-only artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
storage = json.loads((ROOT / "storage_estimate.json").read_text())
runtime = json.loads((ROOT / "runtime_estimate.json").read_text())
status = json.loads((ROOT / "final_status.json").read_text())

assert storage["token_layer_condition_rows"] == storage["tokens"] * storage["layers"] * storage["conditions"]
assert storage["raw_subtotal_bytes"] == sum(storage[k] for k in (
    "controlled_raw_bytes", "recurrent_raw_bytes", "function_space_raw_bytes", "reducer_projected_bytes"
))
assert storage["free_at_audit_gib"] >= storage["minimum_free_before_launch_gib"]
assert storage["STORAGE_READINESS"] == "PASS"
assert runtime["memory"]["five_row_bundle_authorized"] is False
assert status["FULL60_CAPTURE_DESIGN_READY"] == "YES"
assert status["FULL60_EXECUTION_READY"] == "NO"
assert status["FULL60_STARTED"] is False
for key in ("MODEL_MODIFIED", "QUANTIZER_MODIFIED", "ROTATION_MODIFIED", "METRICS_MODIFIED",
            "REDUCER_SEMANTICS_MODIFIED", "NEW_GENERATION_STARTED", "NEW_SCORING_STARTED"):
    assert status[key] is False
for name in ("DESIGN_DOCUMENT.md", "STORAGE_ESTIMATE.md", "RUNTIME_ESTIMATE.md",
             "IMPLEMENTATION_PLAN.md", "READINESS_AUDIT.md",
             "QWEN_GDN_STAGE_B_FULL60_MECHANISM_CAPTURE_DESIGN_V1_REPORT.md"):
    assert (ROOT / name).is_file(), name
print("FULL60_CAPTURE_DESIGN_VALIDATION=PASS")
