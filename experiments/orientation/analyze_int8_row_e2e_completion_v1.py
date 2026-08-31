#!/usr/bin/env python3
import json, math, os, statistics, time
from collections import defaultdict
import os
from pathlib import Path

ROOT = Path(os.environ.get("GDN_DATA_ROOT", "/path/to/gdn_data_root"))
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
SHARDS = RESULTS / "gdn_end2end_bit_axis_screening_v1_shards"
OUT_JSON = RESULTS / "gdn_int8_row_e2e_completion_v1.json"
OUT_RECORDS = RESULTS / "gdn_int8_row_e2e_completion_v1_records.jsonl"
OUT_REPORT = REPORTS / "gdn_int8_row_e2e_completion_v1.md"
SOURCE_RESULT = RESULTS / "gdn_end2end_bit_axis_screening_v1.json"
PROTOCOL = RESULTS / "gdn_end2end_bit_axis_screening_v1_protocol.json"
TASK = "GDN_INT8_ORIENTATION_RECURRENCE_MECHANISM_V1_P0"
EPS = 1e-12


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def read_rows():
    rows = []
    for p in sorted(SHARDS.glob("shard_*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r["_source_shard"] = p.name
                    rows.append(r)
    return rows


def pct(vals, q):
    vals = sorted(vals)
    if not vals:
        return None
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return vals[lo] if lo == hi else vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def summarize(vals):
    toks = [int(r.get("output_token_count") or 0) for r in vals]
    n = len(vals)
    return {
        "N": n,
        "correct": sum(bool(r.get("correct")) for r in vals),
        "accuracy": sum(bool(r.get("correct")) for r in vals) / max(n, 1),
        "truncation_count": sum(bool(r.get("truncated")) for r in vals),
        "truncation_rate": sum(bool(r.get("truncated")) for r in vals) / max(n, 1),
        "nonfinite_count": sum(bool(r.get("nonfinite")) for r in vals),
        "nonfinite_rate": sum(bool(r.get("nonfinite")) for r in vals) / max(n, 1),
        "completed_count": sum(bool(r.get("completed")) for r in vals),
        "mean_generation_length": sum(toks) / max(n, 1),
        "median_generation_length": statistics.median(toks) if toks else None,
        "p90_generation_length": pct(toks, 0.90),
        "p95_generation_length": pct(toks, 0.95),
        "max_generation_length": max(toks) if toks else None,
        "exactly_32768_count": sum(t == 32768 for t in toks),
        "EOS_count": sum(bool(r.get("completed")) and not bool(r.get("truncated")) for r in vals),
        "exception_count": sum(bool(r.get("exception")) for r in vals),
    }


def sample_record(r):
    toks = int(r.get("output_token_count") or 0)
    truncated = bool(r.get("truncated"))
    eos_reached = bool(r.get("completed")) and not truncated
    finish_reason = "length" if truncated else ("eos_or_stop_token" if eos_reached else "nonfinite_or_exception")
    return {
        "problem_id": r.get("problem_id"),
        "prompt_id": r.get("prompt_sha1"),
        "benchmark": r.get("benchmark"),
        "quantizer": r.get("quantizer"),
        "correct": bool(r.get("correct")),
        "generated_new_tokens": toks,
        "EOS_reached": eos_reached,
        "finish_reason": finish_reason,
        "truncated": truncated,
        "nonfinite": bool(r.get("nonfinite")),
        "final_answer": r.get("predicted_answer"),
        "reference_answer": r.get("reference_answer"),
        "output_path": str(SHARDS / r.get("_source_shard", "")),
        "max_token_hit": bool(r.get("max_token_hit")),
        "stop_token_ids": r.get("stop_token_ids"),
    }


def main():
    rows = read_rows()
    wanted = {"FP_STATE", "INT8-row", "INT8-column"}
    filtered = [r for r in rows if r.get("benchmark") == "MATH-500" and r.get("quantizer") in wanted]
    byq = defaultdict(list)
    for r in filtered:
        byq[r.get("quantizer")].append(r)

    OUT_RECORDS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_RECORDS.open("w", encoding="utf-8") as f:
        for r in sorted(byq.get("INT8-row", []), key=lambda x: (x.get("problem_index", -1), x.get("problem_id", ""))):
            f.write(json.dumps(sample_record(r), sort_keys=True, ensure_ascii=False) + "\n")

    comparison = {q: summarize(byq.get(q, [])) for q in ["FP_STATE", "INT8-row", "INT8-column"]}
    fp_by_pid = {r["problem_id"]: r for r in byq.get("FP_STATE", [])}
    col_by_pid = {r["problem_id"]: r for r in byq.get("INT8-column", [])}
    row_pairs = []
    for r in byq.get("INT8-row", []):
        pid = r["problem_id"]
        row_pairs.append({
            "problem_id": pid,
            "fp_correct": bool(fp_by_pid.get(pid, {}).get("correct")),
            "int8_row_correct": bool(r.get("correct")),
            "int8_column_correct": bool(col_by_pid.get(pid, {}).get("correct")) if pid in col_by_pid else None,
            "fp_truncated": bool(fp_by_pid.get(pid, {}).get("truncated")),
            "int8_row_truncated": bool(r.get("truncated")),
            "int8_column_truncated": bool(col_by_pid.get(pid, {}).get("truncated")) if pid in col_by_pid else None,
        })

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8")) if PROTOCOL.exists() else {}
    obj = {
        "task": TASK,
        "timestamp": now(),
        "source_shards": str(SHARDS),
        "protocol_source": str(PROTOCOL),
        "source_end2end_summary": str(SOURCE_RESULT),
        "protocol_metadata": {
            "model_path": "from /experiments/qwen35_gdn_quant/run_config.json",
            "transformers_path": "os.environ.get(GDN_TRANSFORMERS_SRC, /path/to/transformers-qwen35)",
            "dtype": "bfloat16 model load in end-to-end harness",
            "prefill_precision": "FP32/normal model state, no quantization during prefill",
            "decode_precision": "recurrent state fake-quantized after each decode update for quantized configs",
            "generation_max_new_tokens": 32768,
            "max_length_used": False,
            "truncation_definition": "generated_new_tokens == 32768 AND final token is not in configured stop_token_ids",
            "backend_finish_reason_used": False,
            "layer_scope": "24 GDN layers",
            "qrange": {"INT8": [-127, 127]},
            "zero_point": 0,
            "rounding": "torch.round, signed symmetric fake quantization",
            "SMOKE_GATE": protocol.get("SMOKE_GATE", "UNKNOWN"),
            "QUANTIZER_PROTOCOL_GATE": protocol.get("QUANTIZER_PROTOCOL_GATE", "UNKNOWN"),
        },
        "run_status": "COMPLETE" if len(byq.get("INT8-row", [])) == 30 else "INCOMPLETE",
        "comparison": comparison,
        "paired_rows": row_pairs,
        "supported_conclusion": (
            "INT8-row remains strongly degraded relative to FP_STATE and INT8-column on the fixed MATH-500 N=30 subset."
            if len(byq.get("INT8-row", [])) == 30 and comparison["INT8-row"]["accuracy"] + 0.15 < comparison["INT8-column"]["accuracy"]
            else "Do not make a final INT8-row pathology claim until N=30 is complete."
        ),
    }
    OUT_JSON.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# GDN INT8-row E2E Completion V1",
        "",
        "## OBSERVATION",
        "- Protocol gates: SMOKE=%s, QUANTIZER=%s." % (obj["protocol_metadata"]["SMOKE_GATE"], obj["protocol_metadata"]["QUANTIZER_PROTOCOL_GATE"]),
        "- `max_new_tokens=32768`; no `max_length` cap is used by the harness.",
        "- Truncation means `generated_new_tokens == 32768` and no configured EOS/stop token was reached.",
        "",
        "## RESULTS",
        "```json",
        json.dumps(comparison, indent=2, sort_keys=True),
        "```",
        "",
        "## HYPOTHESIS",
        "- INT8-row may be a finite-but-pathological orientation under every-token recurrent feedback quantization.",
        "",
        "## SUPPORTED CONCLUSION",
        "- %s" % obj["supported_conclusion"],
        "",
        "## NEGATIVE RESULT",
        "- This P0 output alone does not establish causality or a deployable method.",
        "",
        "## PROPOSED FOLLOW-UP",
        "- If P0 is complete and row degradation remains, run P1-A state-change mechanism on selected row-pathological and control prompts.",
    ]
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
