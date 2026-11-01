#!/usr/bin/env python3
"""Frozen trajectory-stability analysis; run only after all formal shards complete."""
from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
OUT = ROOT / "experiments/QWEN_HADAMARD_TRAJECTORY_STABILITY_ANALYSIS_V1"
PRIOR = ROOT / "experiments/QWEN_PERSISTENT_FUNCTIONAL_SURROGATE_V1"
ROTATIONS = ("Identity", "Hadamard")
GDN_LAYERS = tuple(i for i in range(32) if i % 4 != 3)
SEED = 20260924
THRESHOLDS = {"state": .1, "readout": .1, "kl": .1}
PRIMARY = ("readout_onset_fraction", "readout_late_log_growth_mean")
METRIC_ORIENTATION = {
    "state_median": -1, "readout_median": -1, "kl_median": -1,
    "state_onset_fraction": 1, "readout_onset_fraction": 1, "kl_onset_fraction": 1,
    "top1_onset_fraction": 1, "margin_onset_fraction": 1,
    "state_late_log_growth_mean": -1, "readout_late_log_growth_mean": -1,
    "kl_late_log_growth_mean": -1,
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def new_json(path, obj):
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def finite_stats(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"n": int(len(values)), "invalid": int(len(values)), "median": None, "p95": None, "cvar95": None}
    p95 = float(np.quantile(finite, .95))
    return {"n": int(len(values)), "invalid": int(len(values)-len(finite)), "median": float(np.median(finite)), "p95": p95, "cvar95": float(finite[finite >= p95].mean())}


def onset(values, threshold=None, event=None):
    values = np.asarray(values)
    mask = (values > threshold) if event is None else event
    positions = np.flatnonzero(mask)
    # n is one position after the last observed t; a non-event is right-censored there.
    return (int(positions[0]), True) if len(positions) else (len(values), False)


def growth(values):
    x = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.diff(np.log(x + 1e-6))
    g[~np.isfinite(g)] = np.nan
    n = len(g)
    cuts = (0, math.ceil(n/3), math.ceil(2*n/3), n)
    result = {}
    for name, lo, hi in zip(("early", "middle", "late"), cuts[:-1], cuts[1:]):
        segment = g[lo:hi]
        valid = segment[np.isfinite(segment)]
        result[f"{name}_log_growth_mean"] = float(valid.mean()) if len(valid) else None
        result[f"{name}_log_growth_median"] = float(np.median(valid)) if len(valid) else None
        result[f"{name}_log_growth_p95"] = float(np.quantile(valid, .95)) if len(valid) else None
        positive = valid[valid > 0]
        result[f"{name}_positive_log_growth_p95"] = float(np.quantile(positive, .95)) if len(positive) else None
        result[f"{name}_invalid_growth"] = int(len(segment)-len(valid))
    return result


def bootstrap_median(x, repeats=5000, seed=SEED):
    x = np.asarray(x, dtype=np.float64)
    if not len(x) or not np.isfinite(x).all():
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(repeats, len(x)), replace=True)
    return [float(v) for v in np.quantile(np.median(draws, axis=1), [.025, .975])]


def cliff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(((a[:, None] > b).sum() - (a[:, None] < b).sum()) / (len(a)*len(b)))


def exact_sign_p(positive, negative):
    n = positive + negative
    if not n:
        return 1.0
    tail = sum(math.comb(n,k) for k in range(min(positive,negative)+1)) / (2**n)
    return float(min(1.0, 2*tail))


def rankdata(values):
    x = np.asarray(values)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i+j+1)/2
        i = j
    return ranks


def permutation_greater(a, b, repeats=10000, seed=SEED):
    a, b = np.asarray(a), np.asarray(b)
    obs = float(np.median(a)-np.median(b))
    pool = np.concatenate([a,b]); rng = np.random.default_rng(seed)
    more = less = 0
    for _ in range(repeats):
        perm = rng.permutation(pool)
        trial = float(np.median(perm[:len(a)]) - np.median(perm[len(a):]))
        more += trial >= obs
        less += trial <= obs
    return {"median_rescue_minus_failure_benefit": obs, "p_one_sided_more_benefit": (more+1)/(repeats+1), "p_one_sided_less_benefit": (less+1)/(repeats+1), "permutations": repeats, "cliffs_delta_benefit": cliff(a,b)}


def auroc(y, score):
    y, score = np.asarray(y), np.asarray(score)
    good = np.isfinite(score)
    y, score = y[good], score[good]
    n1, n0 = int(y.sum()), int((1-y).sum())
    if not n1 or not n0:
        return None
    ranks = rankdata(score)
    return float((ranks[y==1].sum() - n1*(n1+1)/2)/(n1*n0))


def spearman_descriptive(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return {"n":int(len(x)),"spearman":None}
    return {"n":int(len(x)),"spearman":float(np.corrcoef(rankdata(x),rankdata(y))[0,1])}


def source_labels():
    prior_audit = load_json(PRIOR / "artifact_manifest.json")
    path = Path(prior_audit["frozen_outcomes"]["path"])
    labels = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line); key = (row["problem_id"], int(row["seed"]))
            labels.setdefault(key, {})[row["condition"]] = bool(row["v4_correct"])
    exclude = load_json(PRIOR / "preregistration.json")["primary_aime_panel"]["excluded_pair"]
    bad = (exclude["problem_id"], int(exclude["seed"]))
    return labels, bad


def generation_secondary(trajectories):
    source = Path(load_json(PRIOR/"artifact_manifest.json")["frozen_outcomes"]["path"])
    condition_to_rotation = {"int8_c128":"Identity","int8_c128_key_h":"Hadamard"}
    lookup = {}
    with source.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            rotation = condition_to_rotation.get(row["condition"])
            if rotation:
                lookup[(row["problem_id"],int(row["seed"]),rotation)] = row
    fields = ("late_repeated_16gram_fraction","late_unique_4gram_ratio","num_candidate_changes","self_correction_count","longest_repeated_span_fraction")
    stability = ("readout_onset_fraction","readout_late_log_growth_mean")
    linked = []
    for _, item in trajectories[trajectories.clean59].iterrows():
        key = (item.problem_id,int(item.seed),item.rotation)
        if key not in lookup: raise RuntimeError(f"missing frozen generation behavior {key}")
        source_row = lookup[key]
        record = {"problem_id":key[0],"seed":key[1],"rotation":key[2],"group":item.group}
        for name in fields[:-1]:
            value = source_row.get(name)
            record[name] = float(value) if value is not None else np.nan
        span, count = source_row.get("longest_repeated_span"), source_row.get("num_generated_tokens")
        record["longest_repeated_span_fraction"] = float(span)/float(count) if span is not None and count else np.nan
        for name in stability: record[name] = float(item[name])
        linked.append(record)
    frame = pd.DataFrame(linked)
    if len(frame) != 118 or frame.duplicated(["problem_id","seed","rotation"]).any():
        raise RuntimeError("secondary generation linkage is not exact clean59 pairing")
    within = {}
    for rotation in ROTATIONS:
        part = frame[frame.rotation==rotation]
        within[rotation] = {metric:{behavior:spearman_descriptive(part[metric],part[behavior]) for behavior in fields} for metric in stability}
    paired = {}
    for metric in stability:
        a=frame.pivot(index=["problem_id","seed"],columns="rotation",values=metric)
        paired[metric]={}
        for behavior in fields:
            b=frame.pivot(index=["problem_id","seed"],columns="rotation",values=behavior)
            if not a.index.equals(b.index) or len(a)!=59 or len(b)!=59:
                raise RuntimeError("secondary pair mismatch")
            paired[metric][behavior]=spearman_descriptive(a.Hadamard-a.Identity,b.Hadamard-b.Identity)
    return frame, {"status":"AVAILABLE_SECONDARY","secondary":True,"no_gold_conditioned_field_selection":True,"new_generation":False,"within_rotation":within,"paired_H_minus_I":paired,"interpretation":"Descriptive association only; teacher-forced and free-generation trajectories are not identical interventions. No free-generation first-divergence position was inferred."}


def group_for(label):
    identity, had = label["int8_c128"], label["int8_c128_key_h"]
    if identity and had: return "stable_correct"
    if not identity and had: return "hadamard_rescue"
    if not identity and not had: return "stable_failure"
    return "identity_only_correct"


def summarize_rotation(layer, logit, n, base):
    summary, layer_rows = {}, []
    rotation = base["rotation"]
    part = layer[layer.rotation == rotation].sort_values(["token_position", "layer"])
    z = logit[logit.rotation == rotation].sort_values("token_position")
    if len(part) != n*24 or len(z) != n:
        raise RuntimeError(f"row count mismatch {base}")
    if not np.array_equal(part.token_position.to_numpy(), np.repeat(np.arange(n),24)) or not np.array_equal(part.layer.to_numpy(), np.tile(GDN_LAYERS,n)):
        raise RuntimeError(f"incomplete layer-time grid {base}")
    if not np.array_equal(z.token_position.to_numpy(), np.arange(n)):
        raise RuntimeError(f"incomplete logit-time grid {base}")
    matrices = {name: part[name].to_numpy(dtype=np.float64).reshape(n,24) for name in ("D_state", "D_readout", "fp_readout_norm")}
    token = {name: np.median(matrices[f"D_{name}"], axis=1) for name in ("state", "readout")}
    token["kl"] = z.D_logit_KL.to_numpy(dtype=np.float64)
    token["state_layer_p95"] = np.quantile(matrices["D_state"], .95, axis=1)
    token["readout_layer_p95"] = np.quantile(matrices["D_readout"], .95, axis=1)
    for name in ("state", "readout", "kl"):
        x = token[name]
        s = finite_stats(x)
        for stat in ("median", "p95", "cvar95"):
            summary[f"{name}_{stat}"] = s[stat]
        summary[f"{name}_invalid_tokens"] = s["invalid"]
        t, occurred = onset(x, THRESHOLDS[name])
        summary[f"{name}_onset"] = t
        summary[f"{name}_onset_fraction"] = t/n
        summary[f"{name}_onset_event"] = occurred
        summary.update({f"{name}_{key}": value for key, value in growth(x).items()})
    agreement = z.top1_agreement.to_numpy(dtype=np.int8)
    margin = z.q_margin_on_fp_top1.to_numpy(dtype=np.float64)
    for name, event in (("top1", agreement == 0), ("margin", margin <= 0)):
        t, occurred = onset(event, event=event)
        summary[f"{name}_onset"] = t
        summary[f"{name}_onset_fraction"] = t/n
        summary[f"{name}_onset_event"] = occurred
    summary["top1_agreement_rate"] = float(np.mean(agreement == 1))
    summary["fp_entropy_median"] = finite_stats(z.fp_entropy)["median"]
    summary["q_entropy_median"] = finite_stats(z.q_entropy)["median"]
    summary["fp_margin_median"] = finite_stats(z.fp_margin)["median"]
    summary["q_margin_on_fp_top1_median"] = finite_stats(margin)["median"]
    summary["fp_readout_denominator_lt_1e8"] = int((matrices["fp_readout_norm"] < 1e-8).sum())
    for j, layer_id in enumerate(GDN_LAYERS):
        record = dict(base, layer=int(layer_id))
        for name in ("state", "readout"):
            x = matrices[f"D_{name}"][:,j]
            s = finite_stats(x)
            record[f"{name}_median"] = s["median"]
            record[f"{name}_p95"] = s["p95"]
            record[f"{name}_cvar95"] = s["cvar95"]
            record[f"{name}_invalid"] = s["invalid"]
            record[f"{name}_onset"], record[f"{name}_onset_event"] = onset(x, THRESHOLDS[name])
            record[f"{name}_late_log_growth_mean"] = growth(x)["late_log_growth_mean"]
        layer_rows.append(record)
    frame = pd.DataFrame({**base, "token_position": np.arange(n, dtype=np.int32), "D_state_layer_median": token["state"], "D_readout_layer_median": token["readout"], "D_state_layer_p95": token["state_layer_p95"], "D_readout_layer_p95": token["readout_layer_p95"], "D_logit_KL": token["kl"], "top1_agreement": agreement, "q_margin_on_fp_top1": margin, "fp_entropy": z.fp_entropy.to_numpy(), "q_entropy": z.q_entropy.to_numpy()})
    return summary, layer_rows, frame


def analyze_replay(labels, bad, inventory):
    files = sorted(OUT.glob("trajectory_metrics/shard_*/*_meta.json"))
    if len(files) != 60:
        raise RuntimeError(f"expected 60 complete units, got {len(files)}")
    token_path = OUT / "divergence_analysis/token_summary.parquet"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    if token_path.exists() or token_path.with_suffix(".parquet.tmp").exists():
        raise RuntimeError("refusing to overwrite token summary")
    writer = None; rows = []; layers = []; seen = set()
    audit = {"trajectories": 0, "layer_rows": 0, "logit_rows": 0, "invalid_clean59_state": 0, "invalid_clean59_readout": 0, "invalid_clean59_kl": 0, "fp_readout_denominator_lt_1e8": 0, "token_pairing_exact": True, "all_unit_hashes_valid": True}
    try:
        for path in files:
            meta = load_json(path); key = (meta["problem_id"], int(meta["seed"]))
            if key in seen or key not in labels or key not in inventory or meta["token_ids_sha256"] != inventory[key]:
                raise RuntimeError(f"source ID or token hash mismatch {key}")
            seen.add(key)
            n = int(meta["generated_tokens"])-1
            stem = str(path).removesuffix("_meta.json")
            lp, zp = Path(stem+"_layers.parquet"), Path(stem+"_logits.parquet")
            if sha(lp) != meta["layer_file_sha256"] or sha(zp) != meta["logit_file_sha256"]:
                raise RuntimeError(f"unit file hash mismatch {key}")
            layer = pd.read_parquet(lp); logit = pd.read_parquet(zp)
            if len(layer) != 2*n*24 or len(logit) != 2*n:
                raise RuntimeError(f"unit rows mismatch {key}")
            check = logit.pivot(index="token_position", columns="rotation", values=["input_token_id", "target_token_id"])
            for field in ("input_token_id", "target_token_id"):
                if not np.array_equal(check[field]["Identity"].to_numpy(), check[field]["Hadamard"].to_numpy()):
                    raise RuntimeError(f"rotation token pairing mismatch {key}")
            source_tokens = check["input_token_id"]["Identity"].astype(int).tolist() + [int(check["target_token_id"]["Identity"].iloc[-1])]
            if hashlib.sha256(json.dumps(source_tokens, separators=(",", ":")).encode()).hexdigest() != inventory[key]:
                raise RuntimeError(f"source sequence mismatch {key}")
            clean = key != bad; group = group_for(labels[key]) if clean else "symmetric_numerical_exclusion"
            for rotation in ROTATIONS:
                outcome_key = "int8_c128" if rotation == "Identity" else "int8_c128_key_h"
                base = {"problem_id": key[0], "seed": key[1], "rotation": rotation, "clean59": clean, "group": group, "fp_correct": bool(labels[key]["fp_state"]), "outcome_correct": bool(labels[key][outcome_key]), "token_count": n}
                summary, layer_rows, frame = summarize_rotation(layer, logit, n, base)
                rows.append({**base, **summary}); layers.extend(layer_rows)
                if clean:
                    for name in ("state", "readout", "kl"):
                        audit[f"invalid_clean59_{name}"] += summary[f"{name}_invalid_tokens"]
                if rotation == "Identity":
                    audit["fp_readout_denominator_lt_1e8"] += summary["fp_readout_denominator_lt_1e8"]
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(token_path)+".tmp", table.schema, compression="zstd")
                writer.write_table(table)
            audit["trajectories"] += 1
            audit["layer_rows"] += len(layer); audit["logit_rows"] += len(logit)
            if len(seen) % 5 == 0:
                print(json.dumps({"event": "analysis_progress", "trajectories": len(seen), "total": 60}), flush=True)
    finally:
        if writer is not None:
            writer.close()
    if len(seen) != 60:
        raise RuntimeError("not all source units were analyzed")
    os.replace(str(token_path)+".tmp", token_path)
    return pd.DataFrame(rows), pd.DataFrame(layers), audit


def pair_and_test(trajectories):
    clean = trajectories[trajectories.clean59]
    pairs = []
    for metric, orientation in METRIC_ORIENTATION.items():
        pivot = clean.pivot(index=["problem_id", "seed", "group"], columns="rotation", values=metric)
        if len(pivot) != 59 or pivot.isna().any().any():
            raise RuntimeError(f"incomplete clean59 paired metric {metric}")
        for (problem_id, seed, group), r in pivot.iterrows():
            raw = float(r["Hadamard"]-r["Identity"])
            pairs.append({"problem_id": problem_id, "seed": seed, "group": group, "metric": metric, "Identity": float(r["Identity"]), "Hadamard": float(r["Hadamard"]), "delta_H_minus_I": raw, "benefit": orientation*raw})
    frame = pd.DataFrame(pairs); result = {}
    for metric in METRIC_ORIENTATION:
        part = frame[frame.metric == metric]; groups = {}
        for group, g in part.groupby("group"):
            values = g.benefit.to_numpy(dtype=float)
            positive, negative = int((values > 0).sum()), int((values < 0).sum())
            groups[group] = {"n": len(values), "median_benefit": float(np.median(values)), "median_benefit_bootstrap_95ci": bootstrap_median(values), "improved": positive, "p_improved": positive/len(values), "two_sided_sign_p": exact_sign_p(positive, negative)}
        rescue = part[part.group == "hadamard_rescue"].benefit.to_numpy(dtype=float)
        failure = part[part.group == "stable_failure"].benefit.to_numpy(dtype=float)
        result[metric] = {"groups": groups, "rescue_vs_stable_failure": permutation_greater(rescue, failure)}
    return frame, result


def orientation_result(pair, metric, good="lower"):
    x = pair[pair.metric == metric].delta_H_minus_I.to_numpy(dtype=float)
    med = float(np.median(x)); ci = bootstrap_median(x)
    direction = "LOWER" if ci[1] < 0 else "HIGHER" if ci[0] > 0 else "MIXED"
    return {"median_H_minus_I": med, "bootstrap_95ci": ci, "result": direction, "beneficial": direction == good}


def main():
    started = time.time()
    protected = [OUT/x for x in ("results.json", "statistical_analysis.json", "final_report.md", "run_manifest.json")]
    if any(x.exists() for x in protected):
        raise RuntimeError("refusing to overwrite formal result")
    artifact = load_json(OUT/"artifact_manifest.json")
    parity = load_json(OUT/"replay_parity/parity_results.json")
    prereg = load_json(OUT/"preregistration.json")
    if artifact["ARTIFACT_AUDIT"] != "PASS" or parity["REPLAY_PARITY_GATE"] != "PASS" or prereg["status"] != "FROZEN_BEFORE_FORMAL_STABILITY_COMPARISON":
        raise RuntimeError("source/prereg/parity gate missing")
    for index in range(4):
        state = load_json(OUT/f"trajectory_metrics/shard_{index:02d}/status.json")
        if state["status"] != "COMPLETE" or state["assigned"] != 15 or state["completed"] != 15:
            raise RuntimeError(f"shard {index} incomplete")
    labels, bad = source_labels()
    protocol = load_json(PRIOR/"anchor_protocol.json")
    inventory = {(x["problem_id"],int(x["seed"])):x["token_ids_sha256"] for x in protocol["trajectory_anchors"]}
    trajectories, layer_summary, audit = analyze_replay(labels,bad,inventory)
    for folder in ("trajectory_metrics", "divergence_analysis", "rescue_analysis", "generation_analysis", "figures"):
        (OUT/folder).mkdir(parents=True,exist_ok=True)
    trajectories.to_csv(OUT/"trajectory_metrics/trajectory_summary.csv",index=False)
    layer_summary.to_csv(OUT/"trajectory_metrics/layer_summary.csv",index=False)
    groups = trajectories[(trajectories.rotation == "Identity") & trajectories.clean59][["problem_id","seed","group","fp_correct"]].sort_values(["problem_id","seed"])
    if groups.group.value_counts().to_dict() != {"stable_correct":18,"hadamard_rescue":23,"stable_failure":18}:
        raise RuntimeError("frozen clean59 group count mismatch")
    groups.to_csv(OUT/"rescue_analysis/group_manifest.csv",index=False)
    paired, rescue = pair_and_test(trajectories)
    paired.to_csv(OUT/"rescue_analysis/paired_metrics.csv",index=False)
    new_json(OUT/"rescue_analysis/rescue_statistics.json",rescue)
    instant = {name:orientation_result(paired,f"{name}_median") for name in ("state","readout","kl")}
    onset_results = {name:orientation_result(paired,f"{name}_onset_fraction",good="HIGHER") for name in ("state","readout","kl","top1","margin")}
    growth_results = {name:orientation_result(paired,f"{name}_late_log_growth_mean") for name in ("state","readout","kl")}
    r_on = rescue[PRIMARY[0]]; r_gr = rescue[PRIMARY[1]]
    def qualifies(item):
        g = item["groups"]["hadamard_rescue"]
        return g["median_benefit_bootstrap_95ci"][0] > 0 and item["rescue_vs_stable_failure"]["p_one_sided_more_benefit"] < .05
    valid = all(audit[f"invalid_clean59_{name}"] == 0 for name in ("state","readout","kl"))
    strong_opposite_state = (rescue["state_late_log_growth_mean"]["groups"]["hadamard_rescue"]["median_benefit_bootstrap_95ci"][1] < 0 and rescue["state_late_log_growth_mean"]["rescue_vs_stable_failure"]["p_one_sided_less_benefit"] < .05)
    if not valid:
        mechanism = "INCONCLUSIVE"
    elif qualifies(r_on) and qualifies(r_gr) and not strong_opposite_state:
        mechanism = "SUPPORTED"
    elif qualifies(r_on) or qualifies(r_gr):
        mechanism = "PARTIAL"
    else:
        mechanism = "NOT_SUPPORTED"
    hint = "TRAJECTORY_STABILITY_BASED" if mechanism == "SUPPORTED" else "NO_STABILITY_EVIDENCE_FOUND"
    within = {}
    for rotation in ROTATIONS:
        panel = trajectories[(trajectories.clean59)&(trajectories.fp_correct)&(trajectories.rotation==rotation)]
        y = (~panel.outcome_correct.astype(bool)).astype(int).to_numpy()
        within[rotation] = {"n":len(panel),"failure_n":int(y.sum()),"readout_late_growth_failure_auroc":auroc(y,panel.readout_late_log_growth_mean),"negative_readout_onset_failure_auroc":auroc(y,-panel.readout_onset_fraction),"state_late_growth_failure_auroc":auroc(y,panel.state_late_log_growth_mean)}
    generation_frame, generation = generation_secondary(trajectories)
    generation_frame.to_csv(OUT/"generation_analysis/linked_behavior.csv",index=False)
    new_json(OUT/"generation_analysis/secondary_associations.json",generation)
    statistics = {"row_audit":audit,"groups":groups.group.value_counts().to_dict(),"instantaneous":instant,"onset":onset_results,"growth":growth_results,"rescue":rescue,"within_rotation":within,"strong_opposite_state":strong_opposite_state,"mechanism":mechanism}
    new_json(OUT/"statistical_analysis.json",statistics)
    results = {"task":"QWEN_HADAMARD_TRAJECTORY_STABILITY_ANALYSIS_V1","REPLAY_PARITY_GATE":"PASS","STATE_DIVERGENCE_RESULT":instant["state"]["result"],"READOUT_DIVERGENCE_RESULT":instant["readout"]["result"],"LOGIT_DIVERGENCE_RESULT":instant["kl"]["result"],"DIVERGENCE_ONSET_RESULT":onset_results["readout"]["result"],"ERROR_GROWTH_RESULT":growth_results["readout"]["result"],"RESCUE_STABILITY_RESULT":"PASS" if qualifies(r_on) and qualifies(r_gr) else "FAIL","HADAMARD_STABILITY_MECHANISM":mechanism,"LEARNABLE_ROTATION_OBJECTIVE_HINT":hint,"row_audit":audit,"NEW_FREE_GENERATION":"NO","NEW_ROTATION_TRAINING":"NO","MODEL_MODIFIED":"NO","QUANTIZER_MODIFIED":"NO","SCORER_MODIFIED":"NO"}
    new_json(OUT/"results.json",results)
    lines = ["# Qwen Hadamard Trajectory Stability Analysis V1","","## 1. Replay parity","",f"`REPLAY_PARITY_GATE = PASS`. State and KL match the frozen captures exactly; repeated replay is deterministic. The initial topk-vs-argmax observer mismatch and its semantics-preserving repair remain in the parity amendment.","","## 2. State, readout and logit divergence","","The state metric is canonical-basis post-QDQ recurrent-state relative error. The readout metric is the actual GDN norm-input output before RMSNorm, gate and out projection, in the shared value basis. Logit divergence is exact full-vocabulary KL at the same teacher-forced history. None is conflated with the others.",""]
    for name in ("state","readout","kl"):
        v = instant[name]
        lines.append(f"- {name}: median paired H-I instantaneous divergence={v['median_H_minus_I']:.6g}, bootstrap 95% CI={v['bootstrap_95ci']}; classification={v['result']}.")
    lines += ["","## 3. Divergence onset and survival","", "Thresholds were frozen before formal analysis: state/readout relative 0.1, KL 0.1 nat; margin collapse at <=0 and first top1 mismatch. Non-events are right-censored at trajectory end. All 24 GDN layers contribute to the primary state/readout median.",""]
    for name in ("state","readout","kl","top1","margin"):
        v=onset_results[name]; lines.append(f"- {name}: paired normalized onset H-I median={v['median_H_minus_I']:.6g}, 95% CI={v['bootstrap_95ci']}; {v['result']}.")
    lines += ["","## 4. Error growth","","Late-stage growth is the mean adjacent-token log change in the final third of the fixed FP token trajectory; early/middle/late, median and tails are retained in the trajectory file.",""]
    for name in ("state","readout","kl"):
        v=growth_results[name]; lines.append(f"- {name}: paired late log-growth H-I median={v['median_H_minus_I']:.6g}, 95% CI={v['bootstrap_95ci']}; {v['result']}.")
    lines += ["","## 5. Rescue mechanism (primary)","",f"Real-ID clean59 groups: {statistics['groups']}. A positive benefit means later onset or lower growth. Rescue versus stable-failure is tested with a one-sided 10,000-permutation contrast; full bootstrap CIs, effects and sign proportions are in rescue_statistics.json.",""]
    for metric in PRIMARY:
        entry=rescue[metric]; g=entry["groups"]["hadamard_rescue"]; contrast=entry["rescue_vs_stable_failure"]
        lines.append(f"- {metric}: rescue median benefit={g['median_benefit']:.6g}, 95% CI={g['median_benefit_bootstrap_95ci']}, improved={g['improved']}/{g['n']}; rescue-vs-failure Cliff's delta={contrast['cliffs_delta_benefit']:.6g}, one-sided p={contrast['p_one_sided_more_benefit']:.6g}; preregistered test {'PASS' if qualifies(entry) else 'FAIL'}.")
    lines += ["","## 6. Existing free-generation behavior (secondary)","", "Frozen non-gold behavior fields (late repetition, late 4-gram diversity, candidate changes, self-corrections and relative longest repeated span) were linked by real sample ID. Within-rotation and paired-change Spearman correlations are in generation_analysis/secondary_associations.json. They are descriptive and do not affect the mechanism gate; no free-generation first-divergence token was inferred.","", "## 7. Classification and implications","",f"`HADAMARD_STABILITY_MECHANISM = {mechanism}`. `LEARNABLE_ROTATION_OBJECTIVE_HINT = {hint}`. This is an observational teacher-forced mechanism test, not a new rotation objective trained or a causal explanation of free-generation accuracy.","",f"Replay audit: {audit}. Existing generation annotations: {generation['status']}.","","No new AIME generation or rotation training; no model, quantizer or scorer modification.",""]
    (OUT/"final_report.md").open("x",encoding="utf-8").write("\n".join(lines))
    output_paths = [OUT/x for x in ("artifact_manifest.json","preregistration.json","parity_amendment_v1.json","analysis_amendment_v1.json","analysis_amendment_v2.json","generation_amendment_v1.json","replay_parity/parity_results.json","divergence_analysis/token_summary.parquet","trajectory_metrics/trajectory_summary.csv","trajectory_metrics/layer_summary.csv","rescue_analysis/group_manifest.csv","rescue_analysis/paired_metrics.csv","rescue_analysis/rescue_statistics.json","generation_analysis/linked_behavior.csv","generation_analysis/secondary_associations.json","statistical_analysis.json","results.json","final_report.md")]
    manifest={"task":results["task"],"status":"COMPLETE","host":socket.gethostname(),"completed_at":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.time()-started,"replay_script_sha256":sha(OUT/"replay_stability_v2.py"),"analysis_script_sha256":sha(Path(__file__)),"outputs":{str(p.relative_to(OUT)):{"bytes":p.stat().st_size,"sha256":sha(p)} for p in output_paths}}
    new_json(OUT/"run_manifest.json",manifest)
    print("\nQWEN_HADAMARD_TRAJECTORY_STABILITY_ANALYSIS_V1\n")
    for key in ("REPLAY_PARITY_GATE","STATE_DIVERGENCE_RESULT","READOUT_DIVERGENCE_RESULT","LOGIT_DIVERGENCE_RESULT","DIVERGENCE_ONSET_RESULT","ERROR_GROWTH_RESULT","RESCUE_STABILITY_RESULT","HADAMARD_STABILITY_MECHANISM","LEARNABLE_ROTATION_OBJECTIVE_HINT","NEW_FREE_GENERATION","NEW_ROTATION_TRAINING","MODEL_MODIFIED","QUANTIZER_MODIFIED","SCORER_MODIFIED"):
        print(f"{key} = {results[key]}")


if __name__ == "__main__":
    main()
