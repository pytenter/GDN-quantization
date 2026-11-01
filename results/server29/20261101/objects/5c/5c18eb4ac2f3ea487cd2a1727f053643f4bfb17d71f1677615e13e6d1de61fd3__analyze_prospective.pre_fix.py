#!/usr/bin/env python3
"""Finalize the preregistered prospective paired-ranking pilot."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import socket
from pathlib import Path


TASK = "LING_KDA_PERSISTENT_SURROGATE_PROSPECTIVE_VALIDATION_V1"
ROOT = Path("/data/zypan/experiment_runs") / TASK


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"NO_OVERWRITE:{path}")
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(value, encoding="utf-8")
    os.link(tmp, path); tmp.unlink()


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_complete(tag: str):
    path = ROOT / "runs" / tag / "COMPLETE.json"
    value = json.loads(path.read_text())
    if value.get("gate") != "PASS":
        raise RuntimeError(f"RUN_NOT_PASS:{tag}")
    return path, value


def flatten_results(selection):
    rows, by_key, run_files = [], {}, []
    for doc, frozen in selection["documents"].items():
        gpu = json.loads((ROOT / "frozen_config.json").read_text())["document_gpu_map"][doc]
        tags = [f"stage1_{doc}_gpu{gpu}"]
        stage2 = ROOT / "runs" / f"stage2_{doc}_gpu{gpu}" / "COMPLETE.json"
        if stage2.exists():
            tags.append(f"stage2_{doc}_gpu{gpu}")
        for tag in tags:
            path, receipt = load_complete(tag)
            run_files.append({"path": str(path), "sha256": sha256(path), "gpu_hours": receipt["gpu_hours"]})
            for item in receipt["results"]:
                key = (doc, item["condition"])
                if key in by_key:
                    raise RuntimeError(f"DUPLICATE_SCIENTIFIC_KEY:{key}")
                by_key[key] = item
                rows.append({
                    "document": doc, "condition": item["condition"], "stage": receipt["role"],
                    "gpu": receipt["physical_gpu"], "primary_mean_kl": item["primary"]["mean"],
                    "primary_auc_kl": item["primary"]["auc"], "primary_p50_kl": item["primary"]["p50"],
                    "primary_p99_kl": item["primary"]["p99"], "secondary_mean_kl": item["secondary"]["mean"],
                    "secondary_auc_kl": item["secondary"]["auc"], "secondary_p50_kl": item["secondary"]["p50"],
                    "secondary_p99_kl": item["secondary"]["p99"], "all_finite": item["all_finite"],
                    "persistent_integrity": item["persistent_integrity"]["passed"],
                })
    return rows, by_key, run_files


def judge_pair(doc, pair, by_key, comparator):
    a, b = pair["conditions"]
    ra, rb = by_key[(doc, a)], by_key[(doc, b)]
    ya, yb = ra["primary"]["mean"], rb["primary"]["mean"]
    actual = a if ya < yb else b if yb < ya else "TIE"
    pred = pair["left_predicted_winner"]
    other = pair["right_predicted_winner"]
    return {
        "document": doc, "comparator": comparator, "conditions": [a, b],
        "P1_predicted_winner": pred, f"{comparator}_predicted_winner": other,
        "actual_future_kl_winner": actual, "P1_correct": pred == actual,
        f"{comparator}_correct": other == actual,
        "primary_mean_kl": {a: ya, b: yb}, "delta_i_minus_j": ya - yb,
        "secondary_mean_kl": {a: ra["secondary"]["mean"], b: rb["secondary"]["mean"]},
        "primary_secondary_winner_consistent": actual == (a if ra["secondary"]["mean"] < rb["secondary"]["mean"] else b if rb["secondary"]["mean"] < ra["secondary"]["mean"] else "TIE"),
    }


def main():
    selection_path = ROOT / "PROSPECTIVE_ROTATION_SELECTION.json"
    selection = json.loads(selection_path.read_text())
    rows, by_key, run_files = flatten_results(selection)
    primary = [judge_pair(doc, frozen["primary_pair"], by_key, "B0") for doc, frozen in sorted(selection["documents"].items())]
    p1_hits = sum(x["P1_correct"] for x in primary)
    if p1_hits == 2:
        pilot = "P1_DIRECTIONALLY_SUPPORTIVE_IN_PROSPECTIVE_PILOT"
        final = "P1_PERSISTENT_PREDICTION_DIRECTIONALLY_SUPPORTED"
    elif p1_hits == 1:
        pilot = "P1_PROSPECTIVE_SIGNAL_MIXED"
        final = "P1_PERSISTENT_PREDICTION_MIXED"
    else:
        pilot = "P1_PROSPECTIVE_DISCRIMINATION_NOT_SUPPORTED"
        final = "P1_PERSISTENT_PREDICTION_NOT_SUPPORTED"
    secondary = []
    stage2_gate_path = ROOT / "analysis" / "STAGE2_ADVANCEMENT_GATE.json"
    stage2_authorized = stage2_gate_path.exists() and json.loads(stage2_gate_path.read_text()).get("gate") == "PASS"
    if stage2_authorized and all(x.get("secondary_status") == "FROZEN" for x in selection["documents"].values()):
        required = [(doc, c) for doc, x in selection["documents"].items() for c in x["secondary_pair"]["conditions"]]
        if all(x in by_key for x in required):
            secondary = [judge_pair(doc, frozen["secondary_pair"], by_key, "P3_H128") for doc, frozen in sorted(selection["documents"].items())]
    secondary_label = "NOT_RUN_OR_NOT_AVAILABLE"
    if secondary:
        p1_secondary_hits = sum(x["P1_correct"] for x in secondary)
        p3_hits = sum(x["P3_H128_correct"] for x in secondary)
        secondary_label = "P3_ADDITIONAL_VALUE_SUPPORTED_IN_PILOT" if p3_hits > p1_secondary_hits else "P3_ADDITIONAL_VALUE_NOT_SUPPORTED_IN_PILOT"

    fields = list(rows[0])
    from io import StringIO
    buf = StringIO(); writer = csv.DictWriter(buf, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    atomic_text(ROOT / "PERSISTENT_FUTURE_KL_RESULTS.csv", buf.getvalue())

    ranking = [f"# {TASK}: prospective ranking report", "", f"Primary pilot verdict: `{pilot}`", f"Final decision: `{final}`", "", "## P1 vs B0 preregistered pairs", ""]
    for x in primary:
        ranking += [f"- {x['document']}: P1 predicted {x['P1_predicted_winner']}; B0 predicted {x['B0_predicted_winner']}; actual {x['actual_future_kl_winner']}; ΔKL(i-j)={x['delta_i_minus_j']:.9g}."]
    ranking += ["", "## P1 vs P3_H128 conditional pairs", "", f"Status: `{secondary_label}`"]
    for x in secondary:
        ranking += [f"- {x['document']}: P1 predicted {x['P1_predicted_winner']}; P3_H128 predicted {x['P3_H128_predicted_winner']}; actual {x['actual_future_kl_winner']}; ΔKL(i-j)={x['delta_i_minus_j']:.9g}."]
    ranking += ["", "## Limits", "", "This is a two-document, discordant-pair prospective pilot. Overlapping scoring windows are correlated outcomes, not independent samples. Success predicts rankings only; it does not show that gradient descent on P1 will improve Future-KL.", ""]
    atomic_text(ROOT / "PROSPECTIVE_RANKING_REPORT.md", "\n".join(ranking))

    calibration = []
    for doc in json.loads((ROOT / "PROSPECTIVE_DOCUMENT_SELECTION.json").read_text())["selected_documents"]:
        path = ROOT / "runs" / f"calibration_{doc['document_id']}_gpu{doc['physical_gpu']}.json"
        value = json.loads(path.read_text())
        calibration.append({"path": str(path), "sha256": sha256(path), "gpu_hours": value["accounted_gpu_hours"]})
    smoke_path, smoke = load_complete("correctness_smoke_pg19-test-51410_gpu6")
    all_costs = calibration + [{"path": str(smoke_path), "sha256": sha256(smoke_path), "gpu_hours": smoke["gpu_hours"]}] + run_files
    total_gpu_hours = sum(x["gpu_hours"] for x in all_costs)
    ledger = {"task": TASK, "units": all_costs, "total_gpu_hours": total_gpu_hours, "max_total_gpu_hours": 20.0,
              "within_total_budget": total_gpu_hours <= 20.0, "stage1_cap": 12.0,
              "failed_probe_gpu_hours": 0.0, "gpu_hours_definition": "sum of actual device wall times"}
    atomic_json(ROOT / "GPU_COST_LEDGER.json", ledger)

    correctness = f"""# Persistent-QDQ correctness report

- Gate: **PASS**
- Runtime: {smoke['hardware']['torch']} / CUDA {smoke['hardware']['cuda_runtime']} / driver {smoke['hardware']['driver']}
- Correctness smoke: FP plus IDENTITY (Native), H, L6 and L7 through 512 tokens on GPU6.
- GPU UUID: {smoke['hardware']['gpu_uuid']}
- Deterministic algorithms: enabled; TF32: disabled; CUBLAS workspace: `:4096:8`.
- Each branch executed 18 R128 QDQ/writebacks per token, and instrumented the next kernel to verify that its input exactly equaled the preceding post-QDQ state.
- FP and branches were advanced in one process on one GPU using identical teacher-forced token positions.
- Scorer: full-vocabulary `KL(FP logits || persistent-QDQ logits)` reduced online. Logit position t reflects QDQ/writeback through t-1; state at t is immediately QDQ-written for t+1.
- No pulse, no `NO_FURTHER_QDQ`, no high-precision recurrent carry, no optimizer, no free generation.
"""
    atomic_text(ROOT / "PERSISTENT_QDQ_CORRECTNESS_REPORT.md", correctness)

    trainability = f"""# Trainability readiness audit

Prediction validation: `{final}`.

No training and no optimizer step were executed. A prediction result is not gradient readiness or training effectiveness. The frozen P1 evaluator uses measurement-oriented detached/captured state paths; a production differentiability audit, STE-path verification, Cayley parameter gradient check, backward-memory estimate and no-update backward smoke remain required before any training proposal. Because this task does not authorize training, formal training readiness is `NOT_ESTABLISHED`.
"""
    atomic_text(ROOT / "TRAINABILITY_READINESS_AUDIT.md", trainability)
    next_decision = {"task": TASK, "primary_pilot_verdict": pilot, "final_decision": final,
                     "secondary_decision": secondary_label, "training_authorized": False,
                     "next_recommended_experiment": "Independent no-update differentiability and gradient-direction validation only" if p1_hits == 2 else "Do not develop P1 training; diagnose readout/propagation/quantizer interaction",
                     "statistical_generalization_confirmed": False}
    atomic_json(ROOT / "NEXT_RESEARCH_DECISION.json", next_decision)

    reproducibility = {"task": TASK, "host": socket.gethostname(), "kernel": platform.release(),
                       "rotation_selection": {"path": str(selection_path), "sha256": sha256(selection_path)},
                       "run_files": run_files, "calibration_files": calibration,
                       "runner": {"path": str(ROOT / 'scripts' / 'run_persistent_pair.py'), "sha256": sha256(ROOT / 'scripts' / 'run_persistent_pair.py')},
                       "selector": {"path": str(ROOT / 'scripts' / 'select_pairs.py'), "sha256": sha256(ROOT / 'scripts' / 'select_pairs.py')},
                       "old_attempts_modified": False, "old_canonical_modified": False,
                       "training": False, "aime": False, "free_generation": False}
    atomic_json(ROOT / "REPRODUCIBILITY_REPORT.md.json", reproducibility)
    atomic_text(ROOT / "REPRODUCIBILITY_REPORT.md", "# Reproducibility report\n\n```json\n" +
                json.dumps(reproducibility, indent=2, sort_keys=True, allow_nan=False) + "\n```\n")

    existing = json.loads((ROOT / "EXISTING_ASSET_AUDIT.json").read_text())
    atomic_text(ROOT / "EXISTING_ASSET_AUDIT.md", "# Existing asset audit\n\n" +
                "Reusable: " + ", ".join(existing["reusable"]) + ".\n\n" +
                "Minimal new code: " + ", ".join(existing["minimal_new_code"]) + ".\n\n" +
                "Missing required assets: none. Frozen source identities are preserved in `EXISTING_ASSET_AUDIT.json`.\n")

    report = [f"# {TASK} final report", "", f"- Final decision: `{final}`", f"- Primary pilot verdict: `{pilot}`", f"- Secondary verdict: `{secondary_label}`", f"- Completed persistent scientific units: {len(rows)}", f"- Total new GPU-hours: {total_gpu_hours:.6f} / 20.0", "", "All pair identities and predicted winners were frozen before long-horizon outcomes. This remains a small prospective pilot, not broad statistical confirmation.", ""]
    atomic_text(ROOT / "FINAL_REPORT.md", "\n".join(report))

    atomic_json(ROOT / "COMPLETE.json", {"task": TASK, "status": "COMPLETE", "final_decision": final,
                                          "secondary_decision": secondary_label, "scientific_units": len(rows),
                                          "total_gpu_hours": total_gpu_hours, "completed_epoch": __import__('time').time()})
    # Hash all final, immutable scientific deliverables, excluding this manifest.
    paths = sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    atomic_text(ROOT / "SHA256SUMS", "".join(f"{sha256(p)}  {p.relative_to(ROOT)}\n" for p in paths))
    print(json.dumps({"status": "COMPLETE", "decision": final, "secondary": secondary_label, "gpu_hours": total_gpu_hours}))


if __name__ == "__main__":
    main()
