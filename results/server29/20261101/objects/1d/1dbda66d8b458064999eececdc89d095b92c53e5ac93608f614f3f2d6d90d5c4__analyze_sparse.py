#!/usr/bin/env python3
"""Summarize the frozen sparse replay without treating token/layer rows as iid."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1")


def atomic_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def q(values, p):
    values = [float(x) for x in values if pd.notna(x)]
    return float(np.quantile(values, p)) if values else None


def main():
    metrics = pd.read_parquet(ROOT / "sparse_metrics.parquet")
    endpoint = pd.read_csv(ROOT / "paired_outcomes.csv")
    endpoint = endpoint.set_index("sample_id")
    required_conditions = {"H", "C5", "C6"}
    if set(metrics.condition.unique()) != required_conditions:
        raise RuntimeError(f"condition identity mismatch {set(metrics.condition.unique())}")
    if metrics[["sample_id", "condition", "token_index_consumed", "scope", "layer"]].duplicated().any():
        raise RuntimeError("duplicate metric identity")

    layer_scopes = {
        "state_drift": "state_fp_relative_post_qdq_physical",
        "local_qdq": "state_local_qdq_native_basis",
        "function": "linear_attn_out_proj_output_fp_relative",
    }
    unit_rows = []
    for (sample, cond, pos), group in metrics.groupby(["sample_id", "condition", "token_index_consumed"]):
        row = {"sample_id": sample, "condition": cond, "token_index_consumed": int(pos)}
        for short, scope in layer_scopes.items():
            g = group[group.scope == scope]
            if len(g) != 24:
                raise RuntimeError(f"missing layer rows {sample}/{cond}/{pos}/{scope}: {len(g)}")
            row[f"{short}_mean_relative_l2"] = float(g.relative_l2.mean())
            row[f"{short}_max_relative_l2"] = float(g.relative_l2.max())
            row[f"{short}_mean_absolute_l2"] = float(g.absolute_l2.mean())
        mg = group[group.scope == "model_output"]
        if len(mg) != 1:
            raise RuntimeError(f"missing model row {sample}/{cond}/{pos}: {len(mg)}")
        mr = mg.iloc[0]
        for col in ("kl_fp_to_condition", "delta_nll_fixed_fp_next_token", "margin_degradation", "top1_agreement"):
            row[col] = float(mr[col]) if col != "top1_agreement" else bool(mr[col])
        unit_rows.append(row)
    unit = pd.DataFrame(unit_rows)
    unit.to_csv(ROOT / "analysis/sparse_unit_summary.csv", index=False)

    metric_cols = [x for x in unit.columns if x not in {"sample_id", "condition", "token_index_consumed", "top1_agreement"}]
    comparisons = []
    for learned in ("C5", "C6"):
        category_col = f"{learned}_vs_H_category"
        for sample in sorted(unit.sample_id.unique()):
            cat = str(endpoint.loc[sample, category_col])
            a = unit[(unit.sample_id == sample) & (unit.condition == learned)].set_index("token_index_consumed")
            h = unit[(unit.sample_id == sample) & (unit.condition == "H")].set_index("token_index_consumed")
            for pos in sorted(set(a.index) & set(h.index)):
                row = {"comparison": f"{learned}_vs_H", "sample_id": sample, "endpoint_category": cat, "token_index_consumed": int(pos)}
                for col in metric_cols:
                    row[f"delta_{col}"] = float(a.loc[pos, col]) - float(h.loc[pos, col])
                row["learned_top1_agreement"] = bool(a.loc[pos, "top1_agreement"])
                row["hadamard_top1_agreement"] = bool(h.loc[pos, "top1_agreement"])
                comparisons.append(row)
    comp = pd.DataFrame(comparisons)
    comp.to_csv(ROOT / "analysis/sparse_learned_vs_hadamard.csv", index=False)

    group_rows = []
    for (comparison, category), g in comp.groupby(["comparison", "endpoint_category"]):
        samples = sorted(g.sample_id.unique())
        latest = g.sort_values("token_index_consumed").groupby("sample_id", as_index=False).tail(1)
        row = {"comparison": comparison, "endpoint_category": category, "sample_count": len(samples), "sample_ids": ";".join(samples)}
        for col in [x for x in comp.columns if x.startswith("delta_")]:
            row[f"all_positions_median_{col}"] = float(g[col].median())
            row[f"latest_position_median_{col}"] = float(latest[col].median())
            row[f"latest_fraction_learned_lower_{col}"] = float((latest[col] < 0).mean())
        group_rows.append(row)
    pd.DataFrame(group_rows).to_csv(ROOT / "analysis/sparse_group_summary.csv", index=False)

    layer = metrics[metrics.scope == "state_fp_relative_post_qdq_physical"].copy()
    rankings = []
    for learned in ("C5", "C6"):
        category_col = f"{learned}_vs_H_category"
        for category in ("reference_only", "candidate_only"):
            samples = [s for s in layer.sample_id.unique() if str(endpoint.loc[s, category_col]) == category]
            if not samples:
                continue
            latest_pos = layer[layer.sample_id.isin(samples)].groupby("sample_id").token_index_consumed.max().to_dict()
            for sample in samples:
                for lyr in sorted(layer.layer.unique()):
                    pos = latest_pos[sample]
                    a = layer[(layer.sample_id == sample) & (layer.condition == learned) & (layer.layer == lyr) & (layer.token_index_consumed == pos)].iloc[0]
                    h = layer[(layer.sample_id == sample) & (layer.condition == "H") & (layer.layer == lyr) & (layer.token_index_consumed == pos)].iloc[0]
                    rankings.append({"comparison": f"{learned}_vs_H", "endpoint_category": category, "sample_id": sample, "layer": int(lyr), "latest_position": int(pos), "delta_state_drift_relative_l2": float(a.relative_l2-h.relative_l2)})
    rank = pd.DataFrame(rankings)
    rank_agg = rank.groupby(["comparison", "endpoint_category", "layer"], as_index=False).agg(mean_delta=("delta_state_drift_relative_l2", "mean"), median_delta=("delta_state_drift_relative_l2", "median"), sample_count=("sample_id", "nunique"))
    rank_agg["rank_descending_mean_delta"] = rank_agg.groupby(["comparison", "endpoint_category"])["mean_delta"].rank(method="first", ascending=False).astype(int)
    rank_agg.to_csv(ROOT / "analysis/sparse_layer_ranking.csv", index=False)

    findings = {
        "status": "COMPLETE",
        "semantic": "FP_PATH_PERSISTENT_REPLAY",
        "sample_count": int(unit.sample_id.nunique()),
        "conditions": sorted(unit.condition.unique()),
        "capture_positions": sorted(int(x) for x in unit.token_index_consumed.unique()),
        "unit_rows": int(len(unit)),
        "raw_metric_rows": int(len(metrics)),
        "late_history": "UNTESTED_BEYOND_2048_RECURRENT_UPDATES",
        "inference_boundary": "Targeted endpoint-stratified panel; layer/token rows are repeated measures, not independent experimental samples.",
        "model_metric_ranges": {
            col: {"min": float(unit[col].min()), "median": float(unit[col].median()), "max": float(unit[col].max())}
            for col in ("kl_fp_to_condition", "delta_nll_fixed_fp_next_token", "margin_degradation")
        },
        "group_summary_file": "analysis/sparse_group_summary.csv",
        "layer_ranking_file": "analysis/sparse_layer_ranking.csv",
    }
    atomic_json(ROOT / "analysis/sparse_findings.json", findings)
    print(json.dumps(findings, sort_keys=True))


if __name__ == "__main__":
    main()
