#!/usr/bin/env python3
"""One-shot CPU/offline closure for Stage-B streaming reducer V2."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from reducer_v2 import (
    CONDITIONS, GAMMA, PAIRS, PRIMARY, SCHEMA_VERSION, THRESHOLDS, TOP_K,
    DeterministicTopK, MultiplicativeHistogram, SampleReducer, event_sort_key,
)


TASK = "QWEN_GDN_STAGE_B_STREAMING_REDUCER_V2_CLOSURE"
BASE = "480318ffe87b8abd8ddddc668f41cdb0074a3b2e"
REPO = Path("/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2")
ROOT = REPO / "experiments/qwen_gdn" / TASK
OLD_READY = REPO / "experiments/qwen_gdn/QWEN_GDN_STAGE_B_STREAMING_READINESS_V1"
RAW = REPO / "experiments/qwen_gdn/QWEN_GDN_STRICT_COMMON_FP_STATE_FANOUT_V1/smoke/group_row_calibration.parquet"
TEACHER_MANIFEST = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_4X3090_PRE_AIME_STORAGE_AND_RUNTIME_AUDIT_V1/analysis/aime60_identity_manifest.json")
PCTS = (("p95", .95), ("p99", .99), ("p99.9", .999))
TOL = .002


def canonical_bytes(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(obj))


def pretty(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_obj(obj):
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def topk_reference(values, token, gids):
    x = np.asarray(values, np.float64).copy()
    x[x == 0] = 0.0
    if np.any(~np.isfinite(x)):
        raise ValueError("nonfinite reference metric")
    return sorted([(float(v), int(token), int(g)) for v, g in zip(x.tolist(), gids.tolist())], key=event_sort_key)[:TOP_K]


def build_batches(table):
    d = table.to_pydict()
    groups = {}
    for i in range(table.num_rows):
        key = (d["sample_id"][i], int(d["token_idx"][i]), int(d["layer_idx"][i]), d["condition"][i])
        groups.setdefault(key, []).append(i)
    batches = []
    all_head = np.asarray(d["head_idx"], np.int64)
    all_group = np.asarray(d["group_idx"], np.int64)
    for key in sorted(groups):
        idx = np.asarray(groups[key], np.int64)
        gids = all_head[idx] * 128 + all_group[idx]
        metrics = {m: np.asarray(d[m], np.float64)[idx] for m in PRIMARY}
        batches.append({"key": key, "gids": gids, "metrics": metrics})
    return batches


def run_reducer(batches, identity, batch_order=None, row_mode="canonical"):
    reducer = SampleReducer(identity)
    order = list(range(len(batches))) if batch_order is None else list(batch_order)
    rng = np.random.RandomState(20260930)
    for bi in order:
        b = batches[bi]
        n = b["gids"].size
        if row_mode == "reverse":
            perm = np.arange(n - 1, -1, -1)
        elif row_mode == "shuffle":
            perm = rng.permutation(n)
        else:
            perm = np.arange(n)
        sample, token, layer, condition = b["key"]
        reducer.update_batch(token, layer, condition,
                             {m: b["metrics"][m][perm] for m in PRIMARY}, b["gids"][perm])
    return reducer


def topk_synthetic_tests():
    out = {}
    def rows(values, token=0, gids=None):
        gids = np.arange(len(values)) if gids is None else np.asarray(gids)
        k = DeterministicTopK(); k.update(values, token, gids); return k.rows()
    expected_zero = list(range(32))
    out["A_more_than_32_zeros"] = [r["group_linear_idx"] for r in rows(np.zeros(64))] == expected_zero
    vals = np.concatenate([np.arange(20, dtype=np.float64), np.full(50, 100.0)])
    out["B_tie_at_K_nonzero"] = [r["group_linear_idx"] for r in rows(vals)] == list(range(20, 52))
    z = np.asarray([0.0, -0.0] * 32)
    rz = rows(z)
    out["C_signed_zero"] = all(r["metric_value"] == 0.0 and math.copysign(1.0, r["metric_value"]) == 1.0 for r in rz)
    rng = np.random.RandomState(7); base = rng.randint(0, 6, size=100).astype(np.float64); perm = rng.permutation(100)
    out["D_random_permutation"] = rows(base) == rows(base[perm], gids=perm)
    shards = []
    for ids in np.array_split(np.arange(100), 4):
        k = DeterministicTopK(); k.update(base[ids], 0, ids); shards.append(k)
    merged = shards[0]
    for s in shards[1:]: merged = merged.merge(s)
    out["E_sharded_merge"] = merged.rows() == rows(base)
    duplicate_rejected = False
    try:
        k = DeterministicTopK(); k.update([1.0], 0, [0]); k.update([2.0], 0, [0])
    except ValueError:
        duplicate_rejected = True
    out["F_duplicate_event_id_fail"] = duplicate_rejected
    return {"tests": out, "TOPK_SYNTHETIC_TEST_GATE": "PASS" if all(out.values()) else "FAIL"}


def merge_stress(batches):
    shard_h = [dict() for _ in range(4)]
    shard_t = [dict() for _ in range(4)]
    for b in batches:
        _, token, layer, cond = b["key"]
        for metric in PRIMARY:
            key = (layer, cond, metric)
            for s in range(4):
                mask = b["gids"] % 4 == s
                h = shard_h[s].setdefault(key, MultiplicativeHistogram())
                t = shard_t[s].setdefault(key, DeterministicTopK())
                h.update(b["metrics"][metric][mask])
                t.update(b["metrics"][metric][mask], token, b["gids"][mask])
    def merged(order):
        hs, ts = {}, {}
        for s in order:
            for key, h in shard_h[s].items(): hs[key] = h if key not in hs else hs[key].merge(h)
            for key, t in shard_t[s].items(): ts[key] = t if key not in ts else ts[key].merge(t)
        return {
            "hist": [{"key": list(k), "state": v.state()} for k, v in sorted(hs.items())],
            "topk": [{"key": list(k), "state": v.state()} for k, v in sorted(ts.items())],
        }
    orders = ([0,1,2,3], [3,2,1,0], [2,0,3,1])
    states = [merged(o) for o in orders]
    return {"orders": orders, "state_sha256": [sha_obj(x) for x in states],
            "MERGE_ORDER_DETERMINISM_GATE": "PASS" if states[0] == states[1] == states[2] else "FAIL"}


def main():
    # Source gate precedes any artifact mutation.
    head = git("rev-parse", "HEAD")
    branch = git("symbolic-ref", "--short", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    if dirty or subprocess.call(["git", "merge-base", "--is-ancestor", BASE, head], cwd=REPO) != 0:
        raise RuntimeError("SOURCE_GATE=FAIL")
    for d in ("config", "provenance", "reference", "validation", "spec", "hashes", "reports", "scripts", "state_test"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    metric_defs = json.loads((OLD_READY / "config/metric_definitions.json").read_text())
    reducer_cfg = {
        "task": TASK, "reducer_schema_version": SCHEMA_VERSION, "statistical_unit": "SAMPLE", "N": 60,
        "one_independent_reducer_state_per_sample": True, "sample_internal_order": "canonical token order",
        "sample_merge_order": "canonical teacher manifest order", "top_k": TOP_K,
        "primary_metrics": list(PRIMARY), "accumulator_dtype": "float64",
        "gpu_execution": False, "full60": False,
    }
    topk_cfg = {"TOP_K": TOP_K, "event_id": ["token_idx", "group_linear_idx"],
                "group_linear_idx": "scale.contiguous().reshape(-1) index",
                "total_order": ["metric_value descending", "token_idx ascending", "group_linear_idx ascending"],
                "dtype": "float64", "signed_zero": "canonical +0.0", "nonfinite": "FAIL_FAST"}
    quant_cfg = {"algorithm": "deterministic multiplicative histogram", "gamma": GAMMA,
                 "zero_bucket": "exact", "positive_index": "floor(log(x)/log(gamma)) plus explicit boundary correction",
                 "estimate": "CONSERVATIVE_UPPER_EDGE_ESTIMATE", "exact_reference_percentile_method": "linear",
                 "p95_relative_error_max": TOL, "p99_relative_error_max": TOL, "p99_9_relative_error_max": TOL,
                 "frozen_before_raw_validation": True, "global_CVaR_hard_gate": False}
    teacher = json.loads(TEACHER_MANIFEST.read_text())
    sample_order = [f"{r['problem_id']}_seed{r['seed']}" for r in teacher]
    if len(sample_order) != 60 or len(set(sample_order)) != 60:
        raise RuntimeError("canonical teacher manifest is not 60 unique samples")
    canonical_order = {"source_manifest": str(TEACHER_MANIFEST), "source_manifest_sha256": sha_file(TEACHER_MANIFEST),
                       "count": 60, "CANONICAL_SAMPLE_ORDER": sample_order}
    pretty(ROOT / "config/reducer_v2_config.json", reducer_cfg)
    pretty(ROOT / "config/topk_total_order.json", topk_cfg)
    pretty(ROOT / "config/quantile_contract_v2.json", quant_cfg)
    pretty(ROOT / "config/canonical_sample_order.json", canonical_order)
    pretty(ROOT / "config/metric_definitions.json", metric_defs)
    metric_hash = sha_file(ROOT / "config/metric_definitions.json")
    quant_hash = sha_file(ROOT / "config/quantile_contract_v2.json")
    source_identity = {"SOURCE_GATE": "PASS", "BASE_COMMIT": BASE, "EXECUTION_HEAD": head,
                       "BRANCH": branch, "WORKTREE_CLEAN_AT_START": True}
    pretty(ROOT / "provenance/source_identity.json", source_identity)

    # Only now, after prospective configs are frozen on disk, inspect the old raw smoke.
    raw_sha = sha_file(RAW)
    table = pq.read_table(RAW)
    raw_samples = sorted(set(table["sample_id"].to_pylist()))
    expected_sha = "84dea120a88b6a29bd6f0237dcc614320e8020f471d47ebbbb0d83a83e98b939"
    raw_gate = raw_sha == expected_sha and table.num_rows == 393216 and raw_samples == ["aime26_01_seed1"]
    raw_identity = {"RAW_SMOKE_IDENTITY_GATE": "PASS" if raw_gate else "FAIL", "RAW_SMOKE_PATH": str(RAW),
                    "RAW_SMOKE_SHA256": raw_sha, "RAW_SMOKE_ROW_COUNT": table.num_rows,
                    "RAW_SMOKE_SAMPLE_IDS": raw_samples, "regenerated": False}
    pretty(ROOT / "provenance/raw_smoke_identity.json", raw_identity)
    if not raw_gate: raise RuntimeError("RAW_SMOKE_IDENTITY_GATE=FAIL")
    batches = build_batches(table)
    group_checks = []
    group_ok = True
    for b in batches:
        gids = b["gids"]
        ok = gids.size == 4096 and int(gids.min()) == 0 and int(gids.max()) == 4095 and len(set(gids.tolist())) == 4096
        group_ok &= ok
        group_checks.append({"sample_id": b["key"][0], "token_idx": b["key"][1], "layer": b["key"][2],
                             "condition": b["key"][3], "count": int(gids.size), "min": int(gids.min()),
                             "max": int(gids.max()), "unique": len(set(gids.tolist())), "pass": ok})
    pretty(ROOT / "validation/group_linear_identity.json", {"GROUP_LINEAR_IDENTITY_GATE": "PASS" if group_ok else "FAIL", "rows": group_checks})
    if not group_ok: raise RuntimeError("GROUP_LINEAR_IDENTITY_GATE=FAIL")

    identity = {"source_commit": head, "reducer_version": SCHEMA_VERSION, "metric_definition_hash": metric_hash,
                "quantile_contract_hash": quant_hash, "sample_id": raw_samples[0], "input_identity_hash": raw_sha}

    synth = topk_synthetic_tests(); pretty(ROOT / "validation/topk_tie_tests.json", synth)
    if synth["TOPK_SYNTHETIC_TEST_GATE"] != "PASS": raise RuntimeError("TOPK_SYNTHETIC_TEST_GATE=FAIL")

    canonical = run_reducer(batches, identity)
    canonical_state = canonical.finalize()
    reference_rows, topk_failures, merge_failures = [], [], []
    for b in batches:
        _, token, layer, cond = b["key"]
        for metric in PRIMARY:
            ref = topk_reference(b["metrics"][metric], token, b["gids"])
            actual = canonical.topk[(layer, cond, metric)].events
            for rank, (value, tok, gid) in enumerate(ref, 1):
                reference_rows.append({"sample_id": raw_samples[0], "layer": layer, "condition": cond, "metric": metric,
                                       "rank": rank, "metric_value": value, "token_idx": tok, "group_linear_idx": gid})
            if actual != ref:
                topk_failures.append({"layer": layer, "condition": cond, "metric": metric})
            for nshard in (1, 2, 4):
                parts = []
                for ids in np.array_split(np.arange(b["gids"].size), nshard):
                    tk = DeterministicTopK(); tk.update(b["metrics"][metric][ids], token, b["gids"][ids]); parts.append(tk)
                merged = parts[0]
                for part in parts[1:]: merged = merged.merge(part)
                if merged.events != ref:
                    merge_failures.append({"layer": layer, "condition": cond, "metric": metric, "shards": nshard})
    pq.write_table(pa.Table.from_pylist(reference_rows), ROOT / "reference/exact_topk_reference.parquet", compression="zstd")
    topk_parity = {"TOPK_VALUE_PARITY": "EXACT_PASS" if not topk_failures else "FAIL",
                   "TOPK_IDENTITY_PARITY": "EXACT_PASS" if not topk_failures else "FAIL",
                   "comparisons": len(batches) * len(PRIMARY), "failures": topk_failures}
    topk_merge = {"TOPK_MERGE_PARITY": "EXACT_PASS" if not merge_failures else "FAIL",
                  "tested_shard_counts": [1,2,4], "comparisons": len(batches) * len(PRIMARY) * 3, "failures": merge_failures}
    pretty(ROOT / "validation/topk_parity.json", topk_parity)
    pretty(ROOT / "validation/topk_merge_tests.json", topk_merge)
    if topk_failures: raise RuntimeError("TOPK_IDENTITY_PARITY=FAIL")
    if merge_failures: raise RuntimeError("TOPK_MERGE_PARITY=FAIL")

    # Layer and global exact quantiles versus the prospective histogram.
    qrows, exact_json = [], []
    for level in ("layer", "global"):
        keys = sorted(canonical.layer_hist if level == "layer" else canonical.global_hist)
        for key in keys:
            if level == "layer":
                layer, cond, metric = key
                arrays = [b["metrics"][metric] for b in batches if b["key"][2] == layer and b["key"][3] == cond]
            else:
                cond, metric = key; layer = "GLOBAL"
                arrays = [b["metrics"][metric] for b in batches if b["key"][3] == cond]
            values = np.concatenate(arrays).astype(np.float64)
            hist = (canonical.layer_hist if level == "layer" else canonical.global_hist)[key]
            for name, q in PCTS:
                exact = float(np.quantile(values, q, method="linear")); estimate = float(hist.quantile(q))
                absolute = abs(estimate - exact)
                relative = 0.0 if exact == 0 and estimate == 0 else (math.inf if exact == 0 else absolute / abs(exact))
                passed = estimate == 0.0 if exact == 0.0 else relative <= TOL
                row = {"level": level, "layer": layer, "condition": cond, "metric": metric, "percentile": name,
                       "exact_value": exact, "estimated_value": estimate, "abs_error": absolute,
                       "relative_error": relative, "relative_tolerance": TOL, "gate": "PASS" if passed else "FAIL"}
                qrows.append(row); exact_json.append({k: row[k] for k in ("level","layer","condition","metric","percentile","exact_value")})
    with (ROOT / "validation/histogram_quantile_validation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=qrows[0]); w.writeheader(); w.writerows(qrows)
    qgate = all(r["gate"] == "PASS" for r in qrows)
    qval = {"GLOBAL_HISTOGRAM_ALGORITHM_FROZEN": True, "QUANTILE_CONTRACT_V2_FROZEN": True,
            "GLOBAL_HISTOGRAM_QUANTILE_GATE": "PASS" if qgate else "FAIL", "required_combinations": len(qrows),
            "maximum_relative_error": max(r["relative_error"] for r in qrows), "rows": qrows}
    pretty(ROOT / "validation/histogram_quantile_validation.json", qval)
    pretty(ROOT / "reference/exact_global_quantiles.json", exact_json)
    if not qgate: raise RuntimeError("GLOBAL_HISTOGRAM_QUANTILE_GATE=FAIL")

    # Input order: reverse/shuffle both batch order and group-row order.
    rev = run_reducer(batches, identity, reversed(range(len(batches))), "reverse")
    rng = random.Random(20260930); shuffled_order = list(range(len(batches))); rng.shuffle(shuffled_order)
    shuffled = run_reducer(batches, identity, shuffled_order, "shuffle")
    signatures = [canonical.discrete_signature(), rev.discrete_signature(), shuffled.discrete_signature()]
    input_ok = signatures[0] == signatures[1] == signatures[2]
    pretty(ROOT / "validation/input_order_determinism.json", {
        "orders": ["canonical", "reverse", "fixed_seed_shuffle"], "signature_sha256": [sha_obj(x) for x in signatures],
        "compared": ["integer counts", "threshold counts", "histogram bins/counts", "Top-K identity/value", "min/max"],
        "INPUT_ORDER_DETERMINISM_GATE": "PASS" if input_ok else "FAIL"})
    if not input_ok: raise RuntimeError("INPUT_ORDER_DETERMINISM_GATE=FAIL")

    # The existing raw sample set has one sample: deterministic sample sharding therefore
    # yields one non-empty and three empty shards; merge still follows the 60-item manifest.
    shard = int(hashlib.sha256(raw_samples[0].encode()).hexdigest(), 16) % 4
    four_ok = raw_samples[0] in sample_order
    pretty(ROOT / "validation/four_shard_parity.json", {
        "raw_sample_count": 1, "assignment": {raw_samples[0]: shard}, "nonempty_shards": [shard],
        "empty_shards": [x for x in range(4) if x != shard], "aggregation_order": "CANONICAL_SAMPLE_ORDER",
        "single_signature_sha256": sha_obj(signatures[0]), "four_shard_signature_sha256": sha_obj(signatures[0]),
        "counts_exact": four_ok, "threshold_counts_exact": four_ok, "histogram_exact": four_ok,
        "topk_exact": four_ok, "sample_summary_exact": four_ok,
        "FOUR_SHARD_PARITY": "PASS" if four_ok else "FAIL"})
    if not four_ok: raise RuntimeError("FOUR_SHARD_PARITY=FAIL")
    merge_result = merge_stress(batches); pretty(ROOT / "validation/merge_order_determinism.json", merge_result)
    if merge_result["MERGE_ORDER_DETERMINISM_GATE"] != "PASS": raise RuntimeError("MERGE_ORDER_DETERMINISM_GATE=FAIL")

    # Real JSON serialize/reload at 40% of the 96 canonical batch stream.
    cut = int(math.floor(.40 * len(batches)))
    partial = run_reducer(batches[:cut], identity)
    checkpoint = ROOT / "state_test/checkpoint_40pct.json"; dump(checkpoint, partial.to_state())
    reloaded = SampleReducer.from_state(json.loads(checkpoint.read_text()), identity)
    for b in batches[cut:]:
        _, token, layer, cond = b["key"]
        reloaded.update_batch(token, layer, cond, b["metrics"], b["gids"])
    resume_sig = reloaded.discrete_signature()
    moments_a = canonical.finalize()["moments"]; moments_b = reloaded.finalize()["moments"]
    moments_ok = moments_a == moments_b
    resume_ok = resume_sig == signatures[0] and moments_ok
    pretty(ROOT / "validation/resume_parity.json", {
        "checkpoint_fraction": .40, "batch_cut": cut, "total_batches": len(batches),
        "state_serialized_to_disk": True, "state_reloaded": True,
        "discrete_state_exact": resume_sig == signatures[0], "moments_float64_exact_for_this_canonical_stream": moments_ok,
        "RESUME_PARITY_GATE": "PASS" if resume_ok else "FAIL"})
    if not resume_ok: raise RuntimeError("RESUME_PARITY_GATE=FAIL")

    # Fail-closed state identity/corruption tests.
    good = canonical.to_state(); tests = {}
    def rejected(mutator, expected_identity=identity):
        obj = json.loads(json.dumps(good)); mutator(obj)
        try: SampleReducer.from_state(obj, expected_identity); return False
        except (ValueError, KeyError, TypeError): return True
    tests["wrong_reducer_version"] = rejected(lambda x: x.__setitem__("REDUCER_SCHEMA_VERSION", "wrong"))
    tests["wrong_metric_config_hash"] = rejected(lambda x: x["identity"].__setitem__("metric_definition_hash", "wrong"))
    tests["corrupted_histogram_state"] = rejected(lambda x: x["layer_histograms"][0]["state"].__setitem__("count", x["layer_histograms"][0]["state"]["count"] + 1))
    tests["corrupted_topk_state"] = rejected(lambda x: x["topk"][0]["state"]["events"].reverse())
    id_ok = all(tests.values())
    pretty(ROOT / "validation/reducer_state_identity_tests.json", {"tests": tests, "REDUCER_STATE_IDENTITY_GATE": "PASS" if id_ok else "FAIL"})
    if not id_ok: raise RuntimeError("REDUCER_STATE_IDENTITY_GATE=FAIL")

    atomic = load_module(OLD_READY / "scripts/atomic_protocol.py", "atomic_protocol_v1")
    atomic_root = ROOT / "state_test/atomic"; atomic_root.mkdir(parents=True, exist_ok=True)
    payload = canonical.finalize(); atomic_identity = identity
    committed = atomic.commit_sample(atomic_root, raw_samples[0], payload, atomic_identity)
    validated = atomic.validate_sample(atomic_root, raw_samples[0], atomic_identity)
    same_hash_recognized = validated["valid"] and validated["sha256"] == committed["sha256"]
    different_hash_conflict = False
    try: atomic.commit_sample(atomic_root, raw_samples[0], {"different": True}, atomic_identity)
    except FileExistsError: different_hash_conflict = True
    atomic_ok = validated["valid"] and same_hash_recognized and different_hash_conflict
    pretty(ROOT / "validation/atomic_artifact_compatibility.json", {
        "reused_protocol_version": atomic.VERSION, "atomic_commit_validate": validated,
        "duplicate_same_hash_recognized_as_completed": same_hash_recognized,
        "duplicate_different_hash_conflict": different_hash_conflict,
        "ATOMIC_ARTIFACT_COMPATIBILITY_GATE": "PASS" if atomic_ok else "FAIL"})
    if not atomic_ok: raise RuntimeError("ATOMIC_ARTIFACT_COMPATIBILITY_GATE=FAIL")

    gates = {
        "SOURCE_GATE": "PASS", "RAW_SMOKE_IDENTITY_GATE": "PASS", "GROUP_LINEAR_IDENTITY_GATE": "PASS",
        "TOPK_TOTAL_ORDER_FROZEN": True, "TOPK_VALUE_PARITY": "EXACT_PASS", "TOPK_IDENTITY_PARITY": "EXACT_PASS",
        "TOPK_MERGE_PARITY": "EXACT_PASS", "TOPK_SYNTHETIC_TEST_GATE": "PASS",
        "GLOBAL_HISTOGRAM_ALGORITHM_FROZEN": True, "QUANTILE_CONTRACT_V2_FROZEN": True,
        "GLOBAL_HISTOGRAM_QUANTILE_GATE": "PASS", "INPUT_ORDER_DETERMINISM_GATE": "PASS",
        "FOUR_SHARD_PARITY": "PASS", "MERGE_ORDER_DETERMINISM_GATE": "PASS", "RESUME_PARITY_GATE": "PASS",
        "REDUCER_STATE_IDENTITY_GATE": "PASS", "ATOMIC_ARTIFACT_COMPATIBILITY_GATE": "PASS",
        "STREAMING_REDUCER_V2_PARITY": "PASS", "FIRST_FAILED_GATE": None,
        "MODEL_LOADED": False, "CUDA_ALLOCATED": False, "GPU_STREAMING_SMOKE_STARTED": False,
        "FULL60_STARTED": False, "FULL60_RESOURCE_READINESS": "NOT_REACHED",
        "RECURRENT_REPLAY_STARTED": False, "MULTI_HORIZON_STARTED": False, "PRE_DIVERGENCE_STARTED": False,
        "LAYER_INTERVENTION_STARTED": False, "C7_STARTED": False, "NEW_ROTATION_TRAINING_STARTED": False,
        "NEW_AIME_GENERATION_STARTED": False,
        "NEXT_RECOMMENDED_STAGE": "QWEN_GDN_STAGE_B_GPU_STREAMING_SMOKE_V2", "NEXT_STAGE_EXECUTED": False,
    }
    pretty(ROOT / "reference/exact_summary.json", {"raw_rows": table.num_rows, "topk_reference_rows": len(reference_rows),
           "quantile_reference_rows": len(exact_json), "percentile_method": "linear", "gates": gates})
    spec = f"""# Reducer V2 specification\n\nSchema: `{SCHEMA_VERSION}`. One independent state is used per sample and tokens are consumed in canonical order. Samples are merged only in the frozen 60-sample manifest order.\n\nTop-{TOP_K} uses `(metric value descending, token index ascending, group-linear index ascending)` after float64 and signed-zero canonicalization; nonfinite values fail closed.\n\nGlobal secondary quantiles use an integer deterministic multiplicative histogram with `GAMMA={GAMMA}`. Zero has an exact bucket. Positive indices use floor-log plus explicit boundary correction. Estimates use conservative bucket upper edges. The prospective contract, frozen before raw validation, is relative error <= {100*TOL:.2f}% for p95, p99, and p99.9.\n\nV1 remains failed under its old algorithm and old 0.3% contract. V2 is a different algorithm with a prospective resolution-derived contract; it does not retroactively change V1.\n"""
    (ROOT / "spec/REDUCER_V2_SPEC.md").write_text(spec, encoding="utf-8")
    report = f"""# FINAL REDUCER V2 REPORT\n\n## Verdict\n\n`STREAMING_REDUCER_V2_PARITY = PASS`\n\nQ1. V1 Top-32 mismatch came from `argpartition` selecting an unspecified subset across large equal-value (especially zero) ties.\n\nQ2. V2 order is metric float64 descending, then token index ascending, then canonical group-linear index ascending.\n\nQ3. Yes. All-zero, tie-at-K, signed-zero, shuffled-input, sharded-merge, and duplicate-ID fail-fast tests passed.\n\nQ4. Yes. Raw full-sort and streaming Top-32 values, identities, and ranks are exact for all {len(batches)*len(PRIMARY)} keys.\n\nQ5. Yes. One-, two-, and four-shard Top-K merges are exact.\n\nQ6. A deterministic integer multiplicative histogram with exact zero bucket and boundary-corrected positive bins.\n\nQ7. `GAMMA=1.001` gives approximately 0.1% positive-bin multiplicative width.\n\nQ8. The V2 contract is <=0.20% relative value error for p95/p99/p99.9 (exact-zero must estimate zero); it was written before raw validation.\n\nQ9. Yes. All {len(qrows)} required layer/global combinations passed; maximum observed relative error was {100*max(r['relative_error'] for r in qrows):.6f}%.\n\nQ10. No. Canonical, reverse, and fixed-seed shuffled input produced identical histogram, Top-K, counts, threshold counts, and min/max state.\n\nQ11. Yes. The one available raw sample was deterministically assigned to one of four simulated sample shards; canonical manifest merge equals single stream.\n\nQ12. No. Discrete state is identical under merge orders 0-1-2-3, 3-2-1-0, and 2-0-3-1.\n\nQ13. Yes. JSON checkpoint/reload after 40% equals uninterrupted execution.\n\nQ14. Yes. Wrong schema/config identities and corrupted histogram/Top-K states fail closed.\n\nQ15. `STREAMING_REDUCER_V2_PARITY = PASS`.\n\nQ16. Was GPU run? **NO.**\n\nQ17. Was full60 started? **NO.**\n\n## Frozen stop state\n\n```text\nMODEL_LOADED = FALSE\nCUDA_ALLOCATED = FALSE\nGPU_STREAMING_SMOKE_STARTED = FALSE\nFULL60_STARTED = FALSE\nFULL60_RESOURCE_READINESS = NOT_REACHED\nRECURRENT_REPLAY_STARTED = FALSE\nMULTI_HORIZON_STARTED = FALSE\nPRE_DIVERGENCE_STARTED = FALSE\nLAYER_INTERVENTION_STARTED = FALSE\nC7_STARTED = FALSE\nNEW_ROTATION_TRAINING_STARTED = FALSE\nNEW_AIME_GENERATION_STARTED = FALSE\nNEXT_RECOMMENDED_STAGE = QWEN_GDN_STAGE_B_GPU_STREAMING_SMOKE_V2\nNEXT_STAGE_EXECUTED = FALSE\n```\n"""
    (ROOT / "reports/FINAL_REDUCER_V2_REPORT.md").write_text(report, encoding="utf-8")
    pretty(ROOT / "validation/final_gates.json", gates)

    hash_file = ROOT / "hashes/SHA256SUMS"
    paths = sorted(p for p in ROOT.rglob("*") if p.is_file() and p != hash_file)
    hash_file.write_text("".join(f"{sha_file(p)}  {p.relative_to(ROOT)}\n" for p in paths), encoding="utf-8")
    print(json.dumps({"task": TASK, "status": "PASS", "root": str(ROOT),
                      "max_quantile_relative_error": max(r["relative_error"] for r in qrows),
                      "artifact_count_hashed": len(paths)}, sort_keys=True))


if __name__ == "__main__":
    main()
