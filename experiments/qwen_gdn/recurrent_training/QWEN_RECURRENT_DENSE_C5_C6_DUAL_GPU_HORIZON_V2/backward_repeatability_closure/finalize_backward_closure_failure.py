#!/usr/bin/env python3
"""Finalize a stopped closure only when a frozen one-step topology gate fails."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"
REPORTS = ROOT / "reports"
HASHES = ROOT / "hashes"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def worst(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    def ratio(row: dict) -> float:
        allowed = row["envelope"]
        value = row["value"]
        if allowed in (None, 0):
            return float("inf") if value not in (None, 0) else 1.0
        return float(value) / float(allowed)
    return max(rows, key=ratio)


def main() -> None:
    single5 = read(ANALYSIS / "single_gpu_repeatability_5run.json")
    single10 = read(ANALYSIS / "single_gpu_repeatability_10run.json")
    first = read(ANALYSIS / "first_divergent_gradient.json")
    source = read(ANALYSIS / "backward_nondeterminism_source_audit.json")
    deterministic = read(ANALYSIS / "deterministic_mode_diagnostic.json")
    dual = read(ANALYSIS / "dual_gpu_repeatability_5run.json")
    if dual["DUAL_GPU_FORWARD_SEMANTICS"] != "PASS":
        raise RuntimeError("forward gate failed; use separate semantic-failure report")
    keys = ("DUAL_GPU_GRADIENT_WITHIN_SINGLE_NOISE_ENVELOPE",
            "DUAL_GPU_INTERNAL_REPEATABILITY",
            "DUAL_GPU_ONE_STEP_UPDATE_WITHIN_SINGLE_NOISE_ENVELOPE")
    if all(dual[key] == "PASS" for key in keys):
        raise RuntimeError("one-step gates pass: run required 8-update trajectories, not this failure finalizer")
    gradient_failures = dual["cross_gradient_failures"]
    internal_failures = dual["dual_internal_failures"]
    update_failures = dual["cross_update_failures"]
    verdict = {
        "SINGLE_GPU_FORWARD_REPEATABILITY": single5["SINGLE_GPU_FORWARD_REPEATABILITY"],
        "SINGLE_GPU_BACKWARD_BITWISE_REPEATABILITY": single5["SINGLE_GPU_BACKWARD_BITWISE_REPEATABILITY"],
        "SINGLE_GPU_EXACT_BACKWARD_CLOSURE": "NOT_ACHIEVABLE",
        "FIRST_DIVERGENT_GRADIENT_LAYER": first["FIRST_DIVERGENT_GRADIENT_LAYER"],
        "FIRST_DIVERGENT_GRADIENT_PARAMETER": first["FIRST_DIVERGENT_GRADIENT_PARAMETER"],
        "BACKWARD_NONDETERMINISM_SOURCE": source["BACKWARD_NONDETERMINISM_SOURCE"],
        "FIRST_REPORTED_NONDETERMINISTIC_OP": deterministic["FIRST_REPORTED_NONDETERMINISTIC_OP"],
        "BACKWARD_NUMERICAL_PROTOCOL": "FROZEN_SINGLE_GPU_ENVELOPE",
        "DUAL_GPU_GRADIENT_GATE": "WITHIN_SINGLE_NOISE_ENVELOPE" if dual[keys[0]] == "PASS" else "FAIL",
        "DUAL_GPU_INTERNAL_REPEATABILITY": dual[keys[1]],
        "DUAL_GPU_ONE_STEP_UPDATE_GATE": "WITHIN_SINGLE_NOISE_ENVELOPE" if dual[keys[2]] == "PASS" else "FAIL",
        "DUAL_GPU_SHORT_TRAJECTORY_GATE": "NOT_RUN",
        "DUAL_GPU_TRAINING_TOPOLOGY": "FAIL",
        "DUAL_GPU_TOPOLOGY_SPECIFIC_NUMERICAL_EFFECT": "PRESENT_BY_FROZEN_ENVELOPE_GATE",
        "H128_H64_RESOURCE_SEARCH": "NOT_RUN",
        "FOUR_GPU_TOPOLOGY": "NOT_RUN",
        "FULL_C5_C6_TRAINING": "NOT_RUN",
        "AIME_GENERATION": "NOT_RUN",
        "gradient_failures": len(gradient_failures),
        "dual_internal_failures": len(internal_failures),
        "update_failures": len(update_failures),
        "worst_gradient_failure": worst(gradient_failures),
        "worst_dual_internal_failure": worst(internal_failures),
        "worst_update_failure": worst(update_failures),
        "single_gradient_envelope_sha256": sha(ANALYSIS / "frozen_single_gpu_noise_envelope.json"),
        "single_update_envelope_sha256": sha(ANALYSIS / "frozen_single_gpu_update_envelope.json"),
        "numerical_protocol_sha256": sha(ROOT / "protocol_amendments/backward_numerical_equivalence_v1.json"),
        "source_artifacts_sha256": {
            name: sha(ANALYSIS / name) for name in (
                "single_gpu_repeatability_5run.json", "single_gpu_repeatability_10run.json",
                "first_divergent_gradient.json", "deterministic_mode_diagnostic.json",
                "backward_nondeterminism_source_audit.json", "dual_gpu_repeatability_5run.json")
        },
    }
    write_new(ANALYSIS / "closure_verdict.json", verdict)
    max_global = read(ANALYSIS / "frozen_single_gpu_noise_envelope.json")["global"]["relative_l2"]
    update_global = read(ANALYSIS / "frozen_single_gpu_update_envelope.json")["global"]["relative_l2"]
    report = (
        "# QWEN_BACKWARD_REPEATABILITY_AND_TOPOLOGY_CLOSURE_V1\n\n"
        "**Verdict: DUAL_GPU_TRAINING_TOPOLOGY = FAIL under the prospectively frozen "
        "single-GPU numerical-envelope protocol.** Stop here. This does not prove a particular kernel "
        "is the cause; it means at least one required dual comparison exceeded the permitted intrinsic "
        "same-topology single-GPU variability. No 8-update trajectory, four-GPU topology, H128/H64 "
        "search, formal training, or AIME generation was started.\n\n"
        f"The five initial independent single-GPU runs had exact input, teacher target, 240 sampled "
        f"state/QDQ points and losses, but bitwise gradient repeatability failed. The first divergent "
        f"parameter was `{first['FIRST_DIVERGENT_GRADIENT_PARAMETER']}`; "
        f"{sum(bool(v) for v in first['different_pairs_by_layer'].values())}/24 GDN layers varied. "
        "The independent diagnostic reproduces the prior V2 reference's eight loss components and "
        "all 240 sampled forward/QDQ hashes exactly.\n\n"
        f"Strict PyTorch deterministic mode stopped before backward at "
        f"`{deterministic['FIRST_REPORTED_NONDETERMINISTIC_OP']}` inside Qwen's gated-delta fallback. "
        "The operator was not bypassed or replaced. The observed backward variability source is "
        f"classified {source['BACKWARD_NONDETERMINISM_SOURCE']}; the cumsum error is evidence of an "
        "unsupported deterministic operator, not proof of causation.\n\n"
        f"Ten additional fresh-process single-GPU one-step runs formed 45 pairs. Their frozen global "
        f"relative-L2 maxima were gradient {max_global:.9g} and Adam update {update_global:.9g}. "
        "The strict per-layer and global maxima had no safety multiplier and were hashed before "
        "the five new dual-GPU runs. The five dual runs kept forward, QDQ, writeback and BPTT exact.\n\n"
        "| Gate | Result | Violations |\n|---|---|---:|\n"
        f"| Single-vs-dual gradient | {dual[keys[0]]} | {len(gradient_failures)} |\n"
        f"| Dual internal repeatability | {dual[keys[1]]} | {len(internal_failures)} |\n"
        f"| Single-vs-dual one-step update | {dual[keys[2]]} | {len(update_failures)} |\n\n"
        "All 50 cross-topology gradient pairs, 10 dual-internal pairs and 50 update pairs were "
        "retained; no outlier, layer or metric was dropped. Exact hashes remain provenance only "
        "because the single topology itself is not bitwise backward-repeatable.\n\n"
        "Required next action: manual decision on a new, explicit protocol. This task does not "
        "authorize loosening the envelope or proceeding to resource search.\n"
    )
    path = REPORTS / "final_report.md"
    if path.exists():
        raise RuntimeError("refusing to overwrite final report")
    path.write_text(report)
    HASHES.mkdir(exist_ok=True)
    manifest = HASHES / "artifact_sha256.txt"
    if manifest.exists():
        raise RuntimeError("refusing to overwrite hash manifest")
    files = sorted(path for path in ROOT.rglob("*") if path.is_file()
                   and path != manifest and "__pycache__" not in path.parts)
    manifest.write_text("".join(f"{sha(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in files))
    print(json.dumps({key: verdict[key] for key in (
        "DUAL_GPU_GRADIENT_GATE", "DUAL_GPU_INTERNAL_REPEATABILITY",
        "DUAL_GPU_ONE_STEP_UPDATE_GATE", "DUAL_GPU_TRAINING_TOPOLOGY",
        "gradient_failures", "dual_internal_failures", "update_failures")}, indent=2))


if __name__ == "__main__":
    main()
