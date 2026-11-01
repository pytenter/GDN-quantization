#!/usr/bin/env python3
"""Equal-document aggregation for SCREEN and frozen CHECK evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


ROOT = Path("experiments/QWEN_GDN_SHARED_OBJECTIVE_PARETO_AND_DEV_SCREEN_V1")
ANALYSIS = ROOT / "analysis"
LOWER_METRICS = [
    "state_relative_mse",
    "functional_relative_mse",
    "downstream_final_hidden_relative_mse",
    "full_vocab_kl_fp_to_condition",
    "condition_fixed_reference_nll",
    "delta_nll_fixed_reference_token",
    "fp_reference_margin_degradation",
]
ABS_METRICS = LOWER_METRICS + ["condition_margin_for_fp_reference", "top1_agreement_with_fp"]


def aggregate_capture(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"document_id", "partition", "condition", "token_index_consumed", *ABS_METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"missing metric columns: {sorted(missing)}")
    if not frame[list(LOWER_METRICS)].applymap(lambda x: pd.notna(x)).all().all():
        raise RuntimeError("nonfinite/missing capture metric")
    grouped = frame.groupby(["partition", "document_id", "condition"], sort=False)
    docs = grouped[ABS_METRICS].mean().reset_index()
    captures = grouped.size().rename("capture_count").reset_index()
    docs = docs.merge(captures, on=["partition", "document_id", "condition"], validate="one_to_one")
    for partition, sub in docs.groupby("partition"):
        counts = sub.groupby("document_id").size()
        if not (counts == 5).all():
            raise RuntimeError(f"{partition} does not contain five candidate conditions per document")
    return docs


def add_hadamard_deltas(docs: pd.DataFrame) -> pd.DataFrame:
    had = docs[docs.condition == "HADAMARD"].set_index(["partition", "document_id"])
    output = docs.copy()
    for metric in ABS_METRICS:
        lookup = output.set_index(["partition", "document_id"]).index.map(had[metric])
        output[f"{metric}_minus_hadamard"] = output[metric].to_numpy() - lookup.to_numpy()
    return output


def summarize(docs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (partition, condition), group in docs.groupby(["partition", "condition"], sort=False):
        row = {"partition": partition, "condition": condition, "documents": len(group)}
        for metric in ABS_METRICS:
            values = group[metric]
            delta = group[f"{metric}_minus_hadamard"]
            row[f"{metric}__absolute_mean"] = float(values.mean())
            row[f"{metric}__absolute_median"] = float(values.median())
            row[f"{metric}__mean_delta_vs_hadamard"] = float(delta.mean())
            row[f"{metric}__median_delta_vs_hadamard"] = float(delta.median())
            row[f"{metric}__documents_delta_lt_zero"] = int((delta < 0).sum())
            row[f"{metric}__documents_delta_gt_zero"] = int((delta > 0).sum())
            if metric in LOWER_METRICS:
                row[f"{metric}__mean_positive_regression"] = float(delta.clip(lower=0).mean())
                row[f"{metric}__mean_improvement"] = float((-delta.clip(upper=0)).mean())
                row[f"{metric}__maximum_positive_regression"] = float(delta.max())
                row[f"{metric}__maximum_regression_document_id"] = str(group.loc[delta.idxmax(), "document_id"])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("screen", "final"), required=True)
    args = parser.parse_args()
    screen = aggregate_capture(ANALYSIS / "screen_capture_metrics.parquet")
    if args.phase == "screen":
        docs = add_hadamard_deltas(screen)
        docs.to_csv(ANALYSIS / "screen_results.csv", index=False)
        summarize(docs).to_csv(ANALYSIS / "screen_summary.csv", index=False)
        print(json.dumps({"status": "PASS", "phase": "screen", "documents": docs.document_id.nunique(), "rows": len(docs)}, sort_keys=True))
        return
    hypothesis = json.loads((ANALYSIS / "screen_hypothesis.json").read_text())
    if hypothesis.get("status") != "FROZEN_BEFORE_CHECK":
        raise RuntimeError("SCREEN hypothesis was not frozen before CHECK analysis")
    check = aggregate_capture(ANALYSIS / "check_capture_metrics.parquet")
    docs = add_hadamard_deltas(pd.concat([screen, check], ignore_index=True))
    summary = summarize(docs)
    docs.to_csv(ANALYSIS / "independent_dev_per_document.csv", index=False)
    summary.to_csv(ANALYSIS / "independent_dev_summary.csv", index=False)
    docs[docs.partition == "CHECK"].to_csv(ANALYSIS / "check_results.csv", index=False)
    print(json.dumps({"status": "PASS", "phase": "final", "documents": docs.document_id.nunique(), "rows": len(docs)}, sort_keys=True))


if __name__ == "__main__":
    main()
