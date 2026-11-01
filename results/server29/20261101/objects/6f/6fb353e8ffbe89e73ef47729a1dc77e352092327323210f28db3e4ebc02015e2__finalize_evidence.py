#!/usr/bin/env python3
"""Validate the frozen failure boundary and hash compact PP2 evidence once."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"


def read_json(name: str) -> dict:
    return json.loads((ANALYSIS / name).read_text(encoding="utf-8"))


def main() -> None:
    stage = read_json("stage_load_summary.json")
    forward = read_json("pp2_teacher_forward_parity.json")
    localization = read_json("first_forward_divergence_localization.json")
    verdict = read_json("final_verdict.json")
    concise = read_json("forward_parity.json")
    rank0 = read_json("pp2_teacher_rank0.json")
    rank1 = read_json("pp2_teacher_rank1.json")
    reference = read_json("single_teacher_reference.json")
    partition = read_json("partition_inventory.json")
    config_partition = json.loads((ROOT / "configs/pp2_partition.json").read_text(encoding="utf-8"))
    assert stage["STAGE_LOCAL_LOAD_GATE"] == "PASS"
    assert forward["PP2_TEACHER_FORWARD_GATE"] == "FAIL"
    assert verdict["PP2_STATUS"] == "FAIL"
    assert concise["FORWARD_PARITY_GATE"] == "FAIL"
    assert localization["first_divergent_block"] == 0
    assert localization["first_forward_divergence_localization"] == "COMPLETE"
    assert localization["early_submodule_comparison"]["block.0.input_layernorm"]["sha256_equal"]
    assert not localization["early_submodule_comparison"]["block.0.linear_attn"]["sha256_equal"]
    assert rank0["input_ids_sha256"] == rank1["input_ids_sha256"] == reference["sample_token_hash"]
    assert rank0["boundary_after_block15"]["sha256"] == rank1["boundary_after_block15"]["sha256"]
    assert rank0["boundary_after_block15"]["sha256"] != reference["boundary_after_block15"]["sha256"]
    assert rank0["forward_wire_bytes"] == 524288
    assert partition == config_partition
    assert partition["global_rotation_parameter_count"] == 195072
    assert verdict["FORMAL_C5_C6_STARTED"] is False and verdict["AIME_STARTED"] is False
    assert not list(ANALYSIS.glob("*.error.json"))
    assert not list(ANALYSIS.glob("h1_*.json"))
    assert not list(ANALYSIS.glob("h4_*.json"))
    assert not list(ANALYSIS.glob("h32_*.json"))

    required = [
        "preregistration.json", "configs/environment.json", "configs/pp2_partition.json",
        "analysis/partition_inventory.json", "analysis/stage_load_summary.json",
        "analysis/single_teacher_reference.json", "analysis/pp2_teacher_forward_parity.json",
        "analysis/first_forward_divergence_localization.json", "analysis/final_verdict.json",
        "reports/CANONICAL_C5_C6_MATH_DERIVATION.md", "reports/PP2_DESIGN.md",
        "reports/FORWARD_PARITY_REPORT.md", "reports/FINAL_REPORT.md",
    ]
    for relative in required:
        assert (ROOT / relative).is_file(), relative

    manifest = ROOT / "hashes/artifact_sha256.txt"
    manifest.parent.mkdir(exist_ok=True)
    lines = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == manifest or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative}\n")
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(lines)
    print(json.dumps({"evidence_gate": "PASS", "files_hashed": len(lines),
                      "manifest": str(manifest)}, sort_keys=True))


if __name__ == "__main__":
    main()
