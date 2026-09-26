#!/usr/bin/env python3
"""Close the preregistered trajectory experiment without starting any new stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"
REPORTS = ROOT / "reports"
HASHES = ROOT / "hashes"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write_new(path: Path, value) -> None:
    if path.exists():
        raise RuntimeError(f"refusing overwrite {path}")
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    protocol = ROOT / "configs/frozen_trajectory_protocol.json"
    config_copy = ANALYSIS / "frozen_trajectory_protocol.json"
    if sha(protocol) != sha(config_copy):
        raise RuntimeError("frozen protocol copies differ")
    singles = [read(ANALYSIS / f"single_{index:02d}.json") for index in range(1, 11)]
    duals = [read(ANALYSIS / f"dual_{index:02d}.json") for index in range(1, 6)]
    if any(len(row["updates"]) != 8 or row["protocol_sha256"] != sha(protocol)
           for row in [*singles, *duals]):
        raise RuntimeError("incomplete/mismatched 8-update trajectory")
    single = read(ANALYSIS / "single_gpu_10run_summary.json")
    dual = read(ANALYSIS / "dual_gpu_5run_summary.json")
    probe = read(ANALYSIS / "update8_functional_probe.json")
    frozen = ROOT / "analysis/frozen_single_gpu_trajectory_envelope.json"
    frozen_hash = read(ANALYSIS / "frozen_single_gpu_trajectory_envelope_hash.json")
    functional = ROOT / "analysis/frozen_single_gpu_functional_envelope.json"
    functional_hash = read(ANALYSIS / "frozen_single_gpu_functional_envelope_hash.json")
    if sha(frozen) != frozen_hash["sha256"] or sha(functional) != functional_hash["sha256"]:
        raise RuntimeError("frozen envelope hash mismatch")
    for index in range(1, 11):
        read(ANALYSIS / f"single_{index:02d}_probe.json")
    for index in range(1, 6):
        read(ANALYSIS / f"dual_{index:02d}_probe.json")
        if read(ANALYSIS / f"dual_{index:02d}_forward_gate.json")["DUAL_GPU_FORWARD_GATE"] != "PASS":
            raise RuntimeError("per-run dual forward gate failed")
    forward_gate = dual["DUAL_GPU_FORWARD_GATE"]
    trajectory_gate = dual["DUAL_GPU_TRAJECTORY_GATE"]
    probe_gate = probe["UPDATE8_FUNCTIONAL_PROBE_GATE"]
    topology = "VERIFIED_WITH_CONTROLLED_NUMERICAL_VARIABILITY" if (
        forward_gate == trajectory_gate == probe_gate == "PASS") else "FAIL"
    verdict = {
        "TASK": "QWEN_DUAL_GPU_TRAJECTORY_EQUIVALENCE_CLOSURE_V1",
        "SINGLE_GPU_BITWISE_BACKWARD": "FAIL",
        "SINGLE_GPU_TRAJECTORY_ENVELOPE": "FROZEN",
        "DUAL_GPU_FORWARD_GATE": forward_gate,
        "DUAL_GPU_TRAJECTORY_GATE": trajectory_gate,
        "UPDATE8_FUNCTIONAL_PROBE_GATE": probe_gate,
        "DUAL_GPU_TOPOLOGY": topology,
        "NEXT_STAGE_AUTHORIZED": "YES" if topology.startswith("VERIFIED") else "NO",
        "STOP_POINT": "STOP_AFTER_REPORT_WAIT_FOR_MANUAL_NEXT_DECISION",
        "single_runs": 10, "dual_runs": 5, "updates_per_run": 8,
        "single_pair_count": 45, "single_dual_pair_count": 50,
        "frozen_protocol_sha256": sha(protocol),
        "frozen_trajectory_envelope_sha256": sha(frozen),
        "frozen_functional_envelope_sha256": sha(functional),
        "trajectory_failure_count": dual["failure_count"],
        "functional_failure_count": probe["failure_count"],
        "first_trajectory_failure": dual["first_failure"],
        "first_functional_failure": probe["first_failure"],
    }
    write_new(ANALYSIS / "final_verdict.json", verdict)
    REPORTS.mkdir(exist_ok=True)
    report = REPORTS / "final_report.md"
    if report.exists():
        raise RuntimeError(f"refusing overwrite {report}")
    losses_single = [[row["loss"] for row in run["updates"]] for run in singles]
    losses_dual = [[row["loss"] for row in run["updates"]] for run in duals]
    lines = [
        "# Qwen dual-GPU trajectory equivalence closure",
        "",
        f"- Verdict: **{topology}**.",
        f"- Forward exact gate: **{forward_gate}**; trajectory envelope gate: **{trajectory_gate}**; fixed non-AIME endpoint probe: **{probe_gate}**.",
        "- Single-GPU bitwise backward repeatability: **FAIL** (established in prior preserved closure).",
        "- 10 fresh-process single trajectories and 5 fresh-process dual trajectories; 8 H32 Adam updates each from theta=0; no formal C5/C6 training or AIME generation.",
        "- Single reference: all 45 pairwise differences, per update and per layer, frozen as strict observed maxima without multiplier before dual comparison.",
        f"- Cross comparisons: 50 single-versus-dual pairs; {dual['comparison_rows']} trajectory metric cells; {dual['failure_count']} outside the frozen envelope.",
        f"- Endpoint probe: {probe['metric_comparisons']} metric cells; {probe['failure_count']} outside the frozen 45-pair single endpoint envelope.",
        "",
        "## First out-of-envelope evidence",
        "",
        f"- Trajectory: `{json.dumps(dual['first_failure'], sort_keys=True)}`",
        f"- Functional probe: `{json.dumps(probe['first_failure'], sort_keys=True)}`",
        "",
        "## Per-update loss range",
        "",
        "| Update | Single-GPU min–max | Dual-GPU min–max | Cross-envelope failures |",
        "|---:|---:|---:|---:|",
    ]
    for update in range(8):
        s = [row[update] for row in losses_single]
        d = [row[update] for row in losses_dual]
        lines.append(f"| {update+1} | {min(s):.9g}–{max(s):.9g} | {min(d):.9g}–{max(d):.9g} | {dual['failures_by_update'][str(update+1)]} |")
    lines += [
        "",
        "## Provenance and stop point",
        "",
        f"- Frozen protocol SHA-256: `{sha(protocol)}`.",
        f"- Frozen trajectory envelope SHA-256: `{sha(frozen)}`.",
        f"- Frozen functional envelope SHA-256: `{sha(functional)}`.",
        "- Full update-level, layer-level, raw forward/QDQ, gradient and rotation artifacts, pairwise trajectory comparisons, probe data and hashes are retained under this closure directory.",
        "- Gradient and parameter hashes are provenance only, not equality gates, because the single-GPU reference is not bitwise backward deterministic.",
        f"- NEXT_STAGE_AUTHORIZED = **{verdict['NEXT_STAGE_AUTHORIZED']}**. No next-stage work was started; wait for a separate decision even if verified.",
        "",
    ]
    with report.open("x") as handle:
        handle.write("\n".join(lines))
    HASHES.mkdir(exist_ok=True)
    manifest = HASHES / "artifact_sha256.txt"
    if manifest.exists():
        raise RuntimeError(f"refusing overwrite {manifest}")
    paths = sorted(path for path in ROOT.rglob("*")
                   if path.is_file() and path != manifest and
                   "__pycache__" not in path.parts and path.suffix != ".pyc")
    with manifest.open("x") as handle:
        for path in paths:
            handle.write(f"{sha(path)}  {path.relative_to(ROOT).as_posix()}\n")
    print(json.dumps({"DUAL_GPU_TOPOLOGY": topology, "artifact_files": len(paths),
                      "manifest_sha256": sha(manifest), "final_report_sha256": sha(report)},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
