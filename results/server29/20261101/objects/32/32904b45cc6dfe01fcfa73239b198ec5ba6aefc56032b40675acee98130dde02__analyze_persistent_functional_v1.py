#!/usr/bin/env python3
"""Frozen output-space analysis after all teacher-forced replay shards complete."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import socket
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
OUT = ROOT / "experiments/QWEN_PERSISTENT_FUNCTIONAL_SURROGATE_V1"
HELPER = OUT / "analyze_persistent_surrogate.py"
HORIZONS = (1, 4, 8, 16, 32, 64)
METRICS = ("kl", "delta_nll", "delta_margin")
CANDIDATES = ("Identity", "Hadamard")
BAD_PAIR = ("aime26_28", 2)
SEED = 20260924

spec = importlib.util.spec_from_file_location("frozen_persistent_stat_helpers", HELPER)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checked_save_json(path: Path, obj):
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    S.dump_json(path, obj)


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if not len(finite):
        return {"n": int(len(a)), "valid": 0, "invalid": int(len(a)), "mean": None, "median": None, "p90": None, "p95": None, "cvar95": None}
    p95 = float(np.quantile(finite, .95))
    return {
        "n": int(len(a)), "valid": int(len(finite)), "invalid": int(len(a) - len(finite)),
        "mean": float(finite.mean()), "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, .90)), "p95": p95,
        "cvar95": float(finite[finite >= p95].mean()),
    }


def one_sided_permutation(a, b, repeats=10000, seed=SEED):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    observed = float(np.median(a) - np.median(b))
    rng = np.random.default_rng(seed)
    pool = np.concatenate([a, b])
    hits = 0
    for _ in range(repeats):
        shuffled = rng.permutation(pool)
        hits += np.median(shuffled[:len(a)]) - np.median(shuffled[len(a):]) <= observed
    return {"observed_rescue_minus_failure_median": observed, "p_one_sided_more_negative": (hits + 1) / (repeats + 1), "permutations": repeats}


def group_for(label):
    identity = label["int8_c128"]
    hadamard = label["int8_c128_key_h"]
    if identity and hadamard:
        return "stable_correct"
    if not identity and hadamard:
        return "hadamard_rescue"
    if not identity and not hadamard:
        return "stable_failure"
    return "identity_only_correct"


def valid_metric_row(frame):
    return np.isfinite(frame[list(METRICS)].to_numpy(dtype=np.float64)).all(axis=1)


def prepare_source():
    audit = read_json(OUT / "artifact_manifest.json")
    protocol = read_json(OUT / "anchor_protocol.json")
    replay_parity = read_json(OUT / "replay_parity/parity_results.json")
    fp_parity = read_json(OUT / "replay_parity/fp_rotation_parity.json")
    if audit["ARTIFACT_AUDIT"] != "PASS" or replay_parity["REPLAY_PARITY_GATE"] != "PASS" or fp_parity["FP_OUTPUT_SPACE_PARITY"] != "PASS":
        raise RuntimeError("source and replay gates must all pass")
    for shard in range(4):
        state = read_json(OUT / f"replay/shard_{shard:02d}/status.json")
        if state["status"] != "COMPLETE" or state["completed"] != 15 or state["assigned"] != 15:
            raise RuntimeError(f"shard {shard} is not complete")
    inventory = {(r["problem_id"], int(r["seed"])): r for r in protocol["trajectory_anchors"]}
    labels = S.load_labels()
    if len(inventory) != 60:
        raise RuntimeError("anchor inventory does not contain 60 unique trajectories")
    return audit, protocol, replay_parity, fp_parity, inventory, labels


def collect_replay(inventory, labels, runtime_hash):
    local_files = sorted(OUT.glob("replay/shard_*/*_local.parquet"))
    persistent_files = sorted(OUT.glob("replay/shard_*/*_persistent.parquet"))
    if len(local_files) != 60 or len(persistent_files) != 60:
        raise RuntimeError(f"expected 60 local/persistent files, found {len(local_files)}/{len(persistent_files)}")
    continuous_path = OUT / "token_metrics/continuous_metrics.parquet"
    anchor_path = OUT / "token_metrics/anchor_horizon_metrics.parquet"
    if continuous_path.exists() or anchor_path.exists():
        raise RuntimeError("refusing to overwrite combined token metrics")
    continuous_path.parent.mkdir(parents=True, exist_ok=True)
    temp_continuous = continuous_path.with_suffix(".parquet.tmp")
    temp_anchor = anchor_path.with_suffix(".parquet.tmp")
    if temp_continuous.exists() or temp_anchor.exists():
        raise RuntimeError("stale partial combined metrics require inspection")
    writer_continuous = None
    writer_anchor = None
    trajectory_rows = []
    horizon_rows = []
    seen = set()
    total_continuous = total_anchor = invalid_continuous = invalid_anchor = 0
    try:
        for local_path in local_files:
            stem = local_path.name.removesuffix("_local.parquet")
            persistent_path = local_path.with_name(stem + "_persistent.parquet")
            meta_path = local_path.with_name(stem + "_meta.json")
            if not persistent_path.exists() or not meta_path.exists():
                raise RuntimeError(f"incomplete replay unit {stem}")
            meta = read_json(meta_path)
            key = (meta["problem_id"], int(meta["seed"]))
            if key in seen or key not in inventory or key not in labels:
                raise RuntimeError(f"duplicate or unregistered sample {key}")
            seen.add(key)
            source = inventory[key]
            if meta["anchors"] != source["anchors"] or int(meta["generated_tokens"]) != int(source["generated_tokens"]):
                raise RuntimeError(f"anchor provenance mismatch {key}")
            local = pd.read_parquet(local_path)
            anchor = pd.read_parquet(persistent_path)
            n = int(source["generated_tokens"])
            if len(local) != 2 * (n - 1) or len(anchor) != 2 * 4 * len(HORIZONS):
                raise RuntimeError(f"row count mismatch {key}: {len(local)}, {len(anchor)}")
            reference = local[local["candidate"] == "Identity"].sort_values("input_timestep")
            sequence = reference["input_token_id"].astype(int).tolist() + [int(reference["target_token_id"].iloc[-1])]
            observed_hash = hashlib.sha256(json.dumps(sequence, separators=(",", ":")).encode()).hexdigest()
            if observed_hash != source["token_ids_sha256"]:
                raise RuntimeError(f"teacher token provenance mismatch {key}")
            other = local[local["candidate"] == "Hadamard"].sort_values("input_timestep")
            if not np.array_equal(reference["input_token_id"].to_numpy(), other["input_token_id"].to_numpy()) or not np.array_equal(reference["target_token_id"].to_numpy(), other["target_token_id"].to_numpy()):
                raise RuntimeError(f"continuous token pairing mismatch {key}")
            if set(anchor["horizon"].unique()) != set(HORIZONS) or set(anchor["anchor_timestep"].unique()) != set(source["anchors"]):
                raise RuntimeError(f"horizon or anchor mismatch {key}")
            for _, item in anchor.iterrows():
                if int(item["input_token_id"]) != sequence[int(item["input_timestep"])] or int(item["target_token_id"]) != sequence[int(item["target_timestep"])]:
                    raise RuntimeError(f"anchor teacher token mismatch {key}")
            if set(local["candidate"].unique()) != set(CANDIDATES) or set(anchor["candidate"].unique()) != set(CANDIDATES):
                raise RuntimeError(f"candidate mismatch {key}")
            group = group_for(labels[key]) if key != BAD_PAIR else "symmetric_numerical_exclusion"
            fp_correct = bool(labels[key]["fp_state"])
            clean = key != BAD_PAIR
            for frame in (local, anchor):
                frame["valid_flag"] = valid_metric_row(frame)
                frame["runtime_hash"] = runtime_hash
                frame["trajectory_hash"] = source["token_ids_sha256"]
                frame["clean59"] = clean
            invalid_continuous += int((~local["valid_flag"]).sum())
            invalid_anchor += int((~anchor["valid_flag"]).sum())
            total_continuous += len(local)
            total_anchor += len(anchor)
            local_table = pa.Table.from_pandas(local, preserve_index=False)
            anchor_table = pa.Table.from_pandas(anchor, preserve_index=False)
            if writer_continuous is None:
                writer_continuous = pq.ParquetWriter(temp_continuous, local_table.schema, compression="zstd")
                writer_anchor = pq.ParquetWriter(temp_anchor, anchor_table.schema, compression="zstd")
            writer_continuous.write_table(local_table)
            writer_anchor.write_table(anchor_table)
            for candidate in CANDIDATES:
                local_part = local[local["candidate"] == candidate]
                base = {"problem_id": key[0], "seed": key[1], "canonical_index": int(meta["canonical_index"]), "candidate": candidate, "clean59": clean, "group": group, "fp_correct": fp_correct, "outcome_correct": bool(labels[key]["int8_c128" if candidate == "Identity" else "int8_c128_key_h"]), "generated_tokens": n}
                trajectory = base.copy()
                for metric in METRICS:
                    values = stats(local_part[metric])
                    for field, value in values.items():
                        trajectory[f"continuous_{metric}_{field}"] = value
                means_by_horizon = {metric: [] for metric in METRICS}
                p95_by_horizon = {metric: [] for metric in METRICS}
                for horizon in HORIZONS:
                    part = anchor[(anchor["candidate"] == candidate) & (anchor["horizon"] == horizon)].sort_values("anchor_ordinal")
                    if len(part) != 4 or part["anchor_timestep"].astype(int).tolist() != source["anchors"]:
                        raise RuntimeError(f"incomplete anchor-horizon pair {key}/{candidate}/{horizon}")
                    record = dict(base, horizon=horizon, anchor_count=len(part))
                    for metric in METRICS:
                        values = stats(part[metric])
                        for field, value in values.items():
                            record[f"{metric}_{field}"] = value
                        means_by_horizon[metric].append(values["mean"])
                        p95_by_horizon[metric].append(values["p95"])
                    horizon_rows.append(record)
                for metric in METRICS:
                    trajectory[f"persistent_{metric}_equal_horizon_auc"] = float(np.mean(means_by_horizon[metric])) if all(v is not None for v in means_by_horizon[metric]) else None
                    trajectory[f"persistent_{metric}_equal_horizon_p95"] = float(np.mean(p95_by_horizon[metric])) if all(v is not None for v in p95_by_horizon[metric]) else None
                trajectory_rows.append(trajectory)
            if len(seen) % 5 == 0:
                print(json.dumps({"event": "aggregation_progress", "trajectories": len(seen), "total": 60}), flush=True)
    finally:
        if writer_continuous is not None:
            writer_continuous.close()
        if writer_anchor is not None:
            writer_anchor.close()
    if len(seen) != 60:
        raise RuntimeError("not all registered trajectories were found")
    os.replace(temp_continuous, continuous_path)
    os.replace(temp_anchor, anchor_path)
    return pd.DataFrame(trajectory_rows), pd.DataFrame(horizon_rows), {"trajectories": len(seen), "continuous_rows": total_continuous, "anchor_rows": total_anchor, "invalid_continuous_rows": invalid_continuous, "invalid_anchor_rows": invalid_anchor, "token_pairing_exact": True}


def within_rotation(trajectory):
    results = {}
    panel = trajectory[(trajectory["clean59"]) & (trajectory["fp_correct"])]
    for candidate in CANDIDATES:
        part = panel[panel["candidate"] == candidate]
        y = (~part["outcome_correct"].astype(bool)).astype(int).to_numpy()
        metrics = {}
        for metric in METRICS:
            score = part[f"persistent_{metric}_equal_horizon_auc"].to_numpy(dtype=np.float64)
            good = np.isfinite(score)
            yy, xx = y[good], score[good]
            correct, failure = xx[yy == 0], xx[yy == 1]
            metrics[metric] = {
                "n": int(len(xx)), "failures": int(yy.sum()), "auroc": S.auroc(yy, xx),
                "auroc_bootstrap_95ci": S.bootstrap_auc(yy, xx, seed=SEED),
                "spearman_risk_vs_failure": S.spearman(xx, yy),
                "median_correct": float(np.median(correct)) if len(correct) else None,
                "median_failure": float(np.median(failure)) if len(failure) else None,
                "cliffs_delta_failure_vs_correct": S.cliffs_delta(failure, correct),
            }
        results[candidate] = {"panel": "clean59_FP_correct", "n": int(len(part)), "metrics": metrics}
    return results


def paired_analysis(trajectory, horizon):
    paired = []
    for metric in METRICS:
        for point in ("AUC",) + HORIZONS:
            frame = trajectory if point == "AUC" else horizon[horizon["horizon"] == point]
            frame = frame[frame["clean59"]]
            column = f"persistent_{metric}_equal_horizon_auc" if point == "AUC" else f"{metric}_mean"
            pivot = frame.pivot(index=["problem_id", "seed", "group"], columns="candidate", values=column)
            if len(pivot) != 59 or pivot.isna().any().any():
                raise RuntimeError(f"incomplete clean59 pair {metric}/{point}")
            for (problem_id, seed, group), row in pivot.iterrows():
                paired.append({"problem_id": problem_id, "seed": int(seed), "group": group, "metric": metric, "horizon": point, "Identity": float(row["Identity"]), "Hadamard": float(row["Hadamard"]), "delta": float(row["Hadamard"] - row["Identity"])})
    paired = pd.DataFrame(paired)
    summaries = {}
    for metric in METRICS:
        points = {}
        for point in ("AUC",) + HORIZONS:
            part = paired[(paired["metric"] == metric) & (paired["horizon"] == point)]
            groups = {}
            for name, group_frame in part.groupby("group"):
                values = group_frame["delta"].to_numpy(dtype=np.float64)
                improved = int((values < 0).sum())
                groups[name] = {"n": len(values), "improved": improved, "p_improved": improved / len(values), "p_improved_wilson_95ci": S.wilson(improved, len(values)), "median_delta": float(np.median(values)), "median_delta_bootstrap_95ci": S.bootstrap_median(values, seed=SEED), "two_sided_sign_p": S.exact_sign_p(improved, int((values > 0).sum()))}
            rescue = part[part["group"] == "hadamard_rescue"]["delta"].to_numpy(dtype=np.float64)
            failure = part[part["group"] == "stable_failure"]["delta"].to_numpy(dtype=np.float64)
            comparison = one_sided_permutation(rescue, failure)
            comparison["cliffs_delta_rescue_vs_stable_failure"] = S.cliffs_delta(rescue, failure)
            points[str(point)] = {"groups": groups, "rescue_vs_stable_failure": comparison}
        medians = [points[str(h)]["groups"]["hadamard_rescue"]["median_delta"] for h in HORIZONS]
        points["horizon_trend"] = {"rescue_median_delta_by_horizon": dict(zip(map(str, HORIZONS), medians)), "spearman_log_h_vs_median_delta": S.spearman(np.log(np.asarray(HORIZONS)), medians)}
        summaries[metric] = points
    return paired, summaries


def cross_rotation(trajectory, horizon):
    result = {}
    for metric in METRICS:
        points = {}
        for point in ("AUC",) + HORIZONS:
            frame = trajectory if point == "AUC" else horizon[horizon["horizon"] == point]
            frame = frame[frame["clean59"]]
            column = f"persistent_{metric}_equal_horizon_auc" if point == "AUC" else f"{metric}_mean"
            values = {candidate: frame[frame["candidate"] == candidate][column].to_numpy(dtype=np.float64) for candidate in CANDIDATES}
            if any(len(v) != 59 or not np.isfinite(v).all() for v in values.values()):
                raise RuntimeError(f"incomplete aggregate {metric}/{point}")
            medians = {candidate: float(np.median(values[candidate])) for candidate in CANDIDATES}
            points[str(point)] = {"median_trajectory_risk": medians, "PAIRWISE_DIRECTION_CHECK": "CORRECT" if medians["Hadamard"] < medians["Identity"] else "INCORRECT"}
        result[metric] = points
    return result


def classify_and_decide(cross, rescue, invalid_rows):
    classifications = {}
    for metric in METRICS:
        if invalid_rows:
            classifications[metric] = "INSUFFICIENT_EVIDENCE"
        elif cross[metric]["AUC"]["PAIRWISE_DIRECTION_CHECK"] != "CORRECT":
            classifications[metric] = "FAILED_CROSS_ROTATION"
        else:
            r = rescue[metric]["AUC"]
            g = r["groups"]["hadamard_rescue"]
            f = r["groups"]["stable_failure"]
            c = r["rescue_vs_stable_failure"]
            good = g["p_improved"] > .5 and g["median_delta"] < f["median_delta"] and c["cliffs_delta_rescue_vs_stable_failure"] < 0 and c["p_one_sided_more_negative"] < .10
            classifications[metric] = "PROMISING_ROTATION_SURROGATE" if good else "FAILED_RESCUE_ASSOCIATION"
    gate = "PASS" if classifications["kl"] == "PROMISING_ROTATION_SURROGATE" else "CONDITIONAL" if invalid_rows else "FAIL"
    failure_reason = None
    if gate == "FAIL":
        failure_reason = "WRONG_CROSS_ROTATION_DIRECTION" if classifications["kl"] == "FAILED_CROSS_ROTATION" else "NO_RESCUE_ASSOCIATION"
    qualified = [h for h in HORIZONS if cross["kl"][str(h)]["PAIRWISE_DIRECTION_CHECK"] == "CORRECT" and rescue["kl"][str(h)]["groups"]["hadamard_rescue"]["p_improved"] > .5]
    medians = rescue["kl"]["horizon_trend"]["rescue_median_delta_by_horizon"]
    if len(qualified) == len(HORIZONS):
        pattern = "GROWS_WITH_HORIZON" if medians["64"] < medians["1"] else "LOCAL_ALREADY_PRESENT"
    elif not qualified:
        pattern = "NO_PERSISTENT_ADVANTAGE"
    elif 1 not in qualified and set(qualified).issubset({32, 64}):
        pattern = "ONLY_LONG_HORIZON"
    else:
        pattern = "MIXED"
    return classifications, gate, failure_reason, pattern


def make_report(audit, replay_parity, fp_parity, trajectory, cross, rescue, within, classifications, gate, failure_reason, pattern, totals):
    groups = Counter(trajectory[(trajectory["candidate"] == "Identity") & (trajectory["clean59"])]["group"])
    lines = ["# Qwen Persistent Functional Surrogate V1", "", "## 1. Replay parity", "", f"`REPLAY_PARITY_GATE = {replay_parity['REPLAY_PARITY_GATE']}`; frozen token IDs, FP margin, delta-z, top1 changes and repeat-anchor determinism passed. The earlier v2 parity failure and v3 correction remain in the amendments and logs.", "", "## 2. FP full-path output-space parity", "", f"`FP_OUTPUT_SPACE_PARITY = {fp_parity['FP_OUTPUT_SPACE_PARITY']}`. Across {fp_parity['summary']['observations']} unquantized teacher-forced observations, relative-logit-L2 p95={fp_parity['summary']['logit_relative_l2_p95']:.6g}, full-vocab KL p95={fp_parity['summary']['full_vocab_kl_p95']:.6g}, top1 agreement={fp_parity['summary']['top1_agreement']:.3%}, maximum absolute logit difference={fp_parity['summary']['logit_max_abs_error']:.6g}.", "", "## 3. KL, DeltaNLL and DeltaMargin by horizon", "", "All values below are medians across clean59 trajectory-level means over four fixed anchors. Lower is better; signed secondary metrics were not clamped.", "", "| h | KL I | KL H | DeltaNLL I | DeltaNLL H | DeltaMargin I | DeltaMargin H | Rescue P(DeltaKL<0) |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for h in HORIZONS:
        v = [cross[m][str(h)]["median_trajectory_risk"] for m in METRICS]
        p = rescue["kl"][str(h)]["groups"]["hadamard_rescue"]["p_improved"]
        lines.append(f"| {h} | {v[0]['Identity']:.6g} | {v[0]['Hadamard']:.6g} | {v[1]['Identity']:.6g} | {v[1]['Hadamard']:.6g} | {v[2]['Identity']:.6g} | {v[2]['Hadamard']:.6g} | {p:.1%} |")
    primary = cross["kl"]["AUC"]["median_trajectory_risk"]
    r = rescue["kl"]["AUC"]
    lines += ["", "## 4. Persistent KL cross-rotation direction", "", f"`PAIRWISE_DIRECTION_CHECK = {cross['kl']['AUC']['PAIRWISE_DIRECTION_CHECK']}`. Equal-horizon KL trajectory AUC median: Identity={primary['Identity']:.6g}, Hadamard={primary['Hadamard']:.6g}. Frozen reasoning accuracy is 18/59 versus 41/59. This is a pairwise direction check, not a two-point Spearman test.", "", "## 5. Hadamard rescue majority", "", f"Groups reconstructed by real IDs: {dict(groups)}. On the primary equal-horizon KL aggregation, {r['groups']['hadamard_rescue']['improved']}/{r['groups']['hadamard_rescue']['n']} rescue pairs have KL_H<KL_I; Wilson 95% CI={r['groups']['hadamard_rescue']['p_improved_wilson_95ci']}. Per-horizon probabilities are in the table and `rescue_analysis/horizon_rescue_statistics.json`.", "", "## 6. Horizon trend", "", f"Rescue median DeltaKL by fixed horizon: {rescue['kl']['horizon_trend']['rescue_median_delta_by_horizon']}. Secondary Spearman across log horizon={rescue['kl']['horizon_trend']['spearman_log_h_vs_median_delta']:.6g}. Bootstrap CIs and paired sign proportions are saved for every horizon.", "", "## 7. Rescue versus stable failure", "", f"Primary DeltaKL medians: rescue={r['groups']['hadamard_rescue']['median_delta']:.6g}, stable-failure={r['groups']['stable_failure']['median_delta']:.6g}; Cliff's delta={r['rescue_vs_stable_failure']['cliffs_delta_rescue_vs_stable_failure']:.6g}; one-sided 10000-permutation p={r['rescue_vs_stable_failure']['p_one_sided_more_negative']:.6g}. Stable-correct is retained as a control in the saved group statistics.", "", "## 8. Local versus persistent", "", f"`PERSISTENT_ADVANTAGE_PATTERN = {pattern}`. Compare h=1 with every later preregistered horizon in the table; no horizon was selected using outcomes. Continuous from-sequence divergence is stored separately and is not called an anchor-reset horizon.", "", "## 9. M5 and persistent KL roles", "", f"Frozen raw-M5 within-rotation failure AUROC was approximately 0.851 (Identity) and 0.780 (Hadamard), but M5 failed cross-rotation direction and rescue. Primary equal-horizon KL within-rotation AUROC is {within['Identity']['metrics']['kl']['auroc']:.6g} (Identity) and {within['Hadamard']['metrics']['kl']['auroc']:.6g} (Hadamard); full CIs, Spearman, medians and effects are saved. M5 remains a within-rotation diagnostic; KL's proposed role is judged by common-output-space rotation and rescue tests.", "", "## 10. Persistent functional surrogate decision", "", f"`PERSISTENT_FUNCTIONAL_SURROGATE_GATE = {gate}`. Metric classifications: {classifications}. Failure reason: {failure_reason or 'none'}. Replay rows: {totals}; source checkpoint SHA256={audit['checkpoint_combined_sha256']}.", "", "## 11. Learnable rotation decision", "", f"`LEARNABLE_ROTATION_GATE = {'CONDITIONAL' if gate == 'PASS' else 'FAIL'}`. This experiment ends without training or choosing a new rotation. The independent Qwen 64-document Future-KL remains supplementary and was not pooled with AIME clean59.", ""]
    return "\n".join(lines)


def main():
    start = time.time()
    protected = [OUT / "results.json", OUT / "statistical_analysis.json", OUT / "final_report.md", OUT / "run_manifest.json"]
    if any(path.exists() for path in protected):
        raise RuntimeError("refusing to overwrite formal analysis output")
    audit, protocol, replay_parity, fp_parity, inventory, labels = prepare_source()
    runtime_hash = sha256(OUT / "replay_persistent_functional_v3.py")
    trajectory, horizon, totals = collect_replay(inventory, labels, runtime_hash)
    for folder in ("trajectory_metrics", "cross_rotation_analysis", "rescue_analysis", "within_rotation_analysis"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(OUT / "trajectory_metrics/trajectory_summary.csv", index=False)
    horizon.to_csv(OUT / "trajectory_metrics/horizon_summary.csv", index=False)
    group_frame = trajectory[(trajectory["candidate"] == "Identity") & (trajectory["clean59"])][["problem_id", "seed", "group", "fp_correct"]].sort_values(["problem_id", "seed"])
    group_frame.to_csv(OUT / "rescue_analysis/group_manifest.csv", index=False)
    cross = cross_rotation(trajectory, horizon)
    paired, rescue = paired_analysis(trajectory, horizon)
    paired.to_csv(OUT / "rescue_analysis/paired_deltas.csv", index=False)
    within = within_rotation(trajectory)
    invalid_clean = bool(
        trajectory[trajectory["clean59"]][[f"persistent_{m}_equal_horizon_auc" for m in METRICS]].isna().any().any()
        or (horizon[horizon["clean59"]][[f"{m}_invalid" for m in METRICS]].to_numpy() > 0).any()
    )
    classifications, gate, failure_reason, pattern = classify_and_decide(cross, rescue, invalid_clean)
    checked_save_json(OUT / "cross_rotation_analysis/pairwise_direction.json", cross)
    checked_save_json(OUT / "rescue_analysis/horizon_rescue_statistics.json", rescue)
    for candidate in CANDIDATES:
        checked_save_json(OUT / f"within_rotation_analysis/{candidate.lower()}.json", within[candidate])
    statistics = {"cross_rotation": cross, "rescue": rescue, "within_rotation": within, "row_audit": totals, "groups": group_frame["group"].value_counts().to_dict(), "classifications": classifications, "PERSISTENT_ADVANTAGE_PATTERN": pattern}
    checked_save_json(OUT / "statistical_analysis.json", statistics)
    result = {
        "task": "QWEN_PERSISTENT_FUNCTIONAL_SURROGATE_V1", "ARTIFACT_AUDIT": "PASS",
        "REPLAY_PARITY_GATE": "PASS", "FP_OUTPUT_SPACE_PARITY": "PASS",
        "PRIMARY_METRIC": "FULL_VOCAB_PERSISTENT_KL", "PAIRWISE_DIRECTION": cross["kl"]["AUC"]["PAIRWISE_DIRECTION_CHECK"],
        "RESCUE_ASSOCIATION": "PASS" if classifications["kl"] == "PROMISING_ROTATION_SURROGATE" else "FAIL",
        "PERSISTENT_ADVANTAGE_PATTERN": pattern,
        "BEST_HORIZON_AGGREGATION": "PREREGISTERED_EQUAL_WEIGHT_H1_4_8_16_32_64",
        "WITHIN_ROTATION_KL_SIGNAL": {candidate: within[candidate]["metrics"]["kl"]["auroc"] for candidate in CANDIDATES},
        "PERSISTENT_FUNCTIONAL_SURROGATE_GATE": gate, "PERSISTENT_FAILURE_REASON": failure_reason,
        "LEARNABLE_ROTATION_GATE": "CONDITIONAL" if gate == "PASS" else "FAIL",
        "metric_classifications": classifications, "row_audit": totals,
        "NEW_FREE_GENERATION": "NO", "NEW_ROTATION_TRAINING": "NO", "MODEL_MODIFIED": "NO",
        "QUANTIZER_MODIFIED": "NO", "SCORER_MODIFIED": "NO", "M5_REDESIGNED": "NO",
    }
    checked_save_json(OUT / "results.json", result)
    report = make_report(audit, replay_parity, fp_parity, trajectory, cross, rescue, within, classifications, gate, failure_reason, pattern, totals)
    (OUT / "final_report.md").open("x", encoding="utf-8").write(report)
    outputs = {}
    for rel in ("artifact_manifest.json", "anchor_protocol.json", "replay_parity/parity_results.json", "replay_parity/fp_rotation_parity.json", "token_metrics/continuous_metrics.parquet", "token_metrics/anchor_horizon_metrics.parquet", "trajectory_metrics/trajectory_summary.csv", "trajectory_metrics/horizon_summary.csv", "cross_rotation_analysis/pairwise_direction.json", "rescue_analysis/group_manifest.csv", "rescue_analysis/paired_deltas.csv", "rescue_analysis/horizon_rescue_statistics.json", "within_rotation_analysis/identity.json", "within_rotation_analysis/hadamard.json", "statistical_analysis.json", "results.json", "final_report.md"):
        path = OUT / rel
        outputs[rel] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    checked_save_json(OUT / "run_manifest.json", {"task": result["task"], "status": "COMPLETE", "completed_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.time() - start, "host": socket.gethostname(), "analysis_script_sha256": sha256(Path(__file__)), "replay_script_sha256": runtime_hash, "outputs": outputs, "new_generation": False})
    print("\nQWEN_PERSISTENT_FUNCTIONAL_SURROGATE_V1")
    for key in ("ARTIFACT_AUDIT", "REPLAY_PARITY_GATE", "FP_OUTPUT_SPACE_PARITY", "PRIMARY_METRIC", "PAIRWISE_DIRECTION", "RESCUE_ASSOCIATION", "PERSISTENT_ADVANTAGE_PATTERN", "BEST_HORIZON_AGGREGATION", "WITHIN_ROTATION_KL_SIGNAL", "PERSISTENT_FUNCTIONAL_SURROGATE_GATE", "LEARNABLE_ROTATION_GATE", "NEW_FREE_GENERATION", "NEW_ROTATION_TRAINING", "MODEL_MODIFIED", "QUANTIZER_MODIFIED", "SCORER_MODIFIED", "M5_REDESIGNED"):
        print(f"{key} = {result[key]}")


if __name__ == "__main__":
    main()
