#!/usr/bin/env python3
"""Freeze the deployment-closure verdict, diff, report, and artifact hashes."""

import difflib
import hashlib
import json
from pathlib import Path


ROOT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1")
ORIGINAL = Path("/data/zypan/consolidation/GDN-quantization-qwen3090-20260921/experiments/qwen_gdn/aime26/81920/run_qwen_aime26_formal.py")
INTEGRATED = ROOT / "scripts/run_qwen_c5_c6_formal.py"


def load(name):
    return json.loads((ROOT / "analysis" / name).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path, text):
    with path.open("x", encoding="utf-8") as f:
        f.write(text)


def main():
    stage1 = load("stage1_gate_summary.json")
    static = load("loader_integration_static_audit_v2.json")
    functional = load("functional_probe_gate_summary_v2.json")
    dynamic = load("repeatability_and_hadamard_guard_v2.json")["gates"]
    gates = {
        "C5_SYNC": stage1["C5_SYNC"],
        "C6_SYNC": stage1["C6_SYNC"],
        "C5_FILE_HASH_GATE": stage1["C5_FILE_HASH_GATE"],
        "C6_FILE_HASH_GATE": stage1["C6_FILE_HASH_GATE"],
        "C5_FROZEN_STRUCTURE_GATE": stage1["C5_FROZEN_STRUCTURE_GATE"],
        "C6_FROZEN_STRUCTURE_GATE": stage1["C6_FROZEN_STRUCTURE_GATE"],
        "FROZEN_LOADER_INTEGRATION_STATIC_GATE": static["FROZEN_LOADER_INTEGRATION_STATIC_GATE"],
        "FROZEN_LOADER_INTEGRATION_DYNAMIC_GATE": dynamic["FROZEN_LOADER_INTEGRATION_DYNAMIC_GATE"],
        "HADAMARD_REGRESSION_GUARD": dynamic["HADAMARD_REGRESSION_GUARD"],
        "C5_FUNCTIONAL_PROBE_GATE": functional["C5_FUNCTIONAL_PROBE_GATE"],
        "C6_FUNCTIONAL_PROBE_GATE": functional["C6_FUNCTIONAL_PROBE_GATE"],
        "C5_REPEATABILITY_GATE": dynamic["C5_REPEATABILITY_GATE"],
        "C6_REPEATABILITY_GATE": dynamic["C6_REPEATABILITY_GATE"],
    }
    verdict = "PASS" if all(value == "PASS" for value in gates.values()) else "FAIL"
    summary = {
        "task": "QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1",
        "gates": gates,
        "C5_FILE_SHA256": "0266c5065267cf452d72cfe29a2a4d9a24fdfb6a89d7bec04b4d796c86f857c9",
        "C6_FILE_SHA256": "67e3731213ad3fb0c088c781ddd912aaaad5af75f9532b289442690a7b09f39c",
        "canonical_original_runner_sha256": sha(ORIGINAL),
        "integrated_runner_sha256": sha(INTEGRATED),
        "AIME_used": False,
        "QWEN_C5_C6_DEPLOYMENT_CLOSURE": verdict,
    }
    write_once(ROOT / "analysis/deployment_closure_summary_v2.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")

    diff = difflib.unified_diff(
        ORIGINAL.read_text(encoding="utf-8").splitlines(True),
        INTEGRATED.read_text(encoding="utf-8").splitlines(True),
        fromfile=str(ORIGINAL), tofile=str(INTEGRATED),
    )
    write_once(ROOT / "analysis/canonical_runner_final_diff.patch", "".join(diff))

    lines = ["# QWEN C5/C6 deployment closure", "", "| Gate | Result |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in gates.items()]
    lines += ["", f"`QWEN_C5_C6_DEPLOYMENT_CLOSURE = {verdict}`", "",
              "- C5 and C6 were synchronized only from verified GitHub commit `fe7105fbc44b49db7a5bab1bf693c25e0f9e63fa`.",
              "- Both payloads are 24-layer FP32 128×128 direct frozen matrices with exact file and per-layer hashes.",
              "- No matrix was reconstructed from theta.",
              "- The original canonical runner and baseline artifacts were not modified.",
              "- The integration preserves the original Hadamard class, quantizer, sampling, and cache helpers exactly by AST hash and dynamic fixed-prompt comparison.",
              "- The final runner additionally records start/end UTC, config hash, retry count, and a canonical per-record payload SHA without changing model semantics.",
              "- Fixed functional probes used `TRAIN:first-row`, 32 tokens, and no AIME data.",
              "- The first C5 outer-gate failure is preserved in logs and was caused solely by a hand-copied SHA constant typo; its completed probe was revalidated without rerunning C5.",
              "- The first repeatability attempt stopped before token generation because the harness compared two different hash encodings; the corrected run is preserved separately.", ""]
    write_once(ROOT / "reports/FINAL_REPORT_V2.md", "\n".join(lines))

    paths = sorted(p for p in ROOT.rglob("*") if p.is_file() and p not in {ROOT / "hashes/artifact_sha256.txt", ROOT / "hashes/artifact_sha256_v2.txt"})
    hash_lines = [f"{sha(path)}  {path.relative_to(ROOT)}" for path in paths]
    write_once(ROOT / "hashes/artifact_sha256_v2.txt", "\n".join(hash_lines) + "\n")
    print(json.dumps(summary, indent=2))
    if verdict != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
