#!/usr/bin/env python3
"""CPU-only validation and Pareto analysis for the frozen H64 candidates."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("experiments/QWEN_GDN_SHARED_OBJECTIVE_PARETO_AND_DEV_SCREEN_V1")
SOURCE_ROOT = Path("experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1")
SOURCE_CSV = SOURCE_ROOT / "analysis/h64_validation_cross_objective.csv"


KNOWN = {
    ("C5", 7): (0.008787583752564387, 0.05271048276335932),
    ("C5", 63): (0.007916185131762177, 0.04950929437109153),
    ("C5", 113): (0.008100550487142755, 0.046988552749098744),
    ("C6", 7): (0.008776775443038787, 0.050523607427749084),
    ("C6", 119): (0.008231279271058156, 0.0523448915919289),
    ("C6", 125): (0.008251136421677074, 0.04901648649683921),
}


def nondominated(frame: pd.DataFrame) -> set[int]:
    keep: set[int] = set()
    for i, row in frame.iterrows():
        dominated = False
        for j, other in frame.iterrows():
            if i == j:
                continue
            if (
                other.state_relative_mse <= row.state_relative_mse
                and other.functional_relative_mse <= row.functional_relative_mse
                and (
                    other.state_relative_mse < row.state_relative_mse
                    or other.functional_relative_mse < row.functional_relative_mse
                )
            ):
                dominated = True
                break
        if not dominated:
            keep.add(i)
    return keep


def dominators(frame: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    return frame[
        (frame.state_relative_mse <= row.state_relative_mse)
        & (frame.functional_relative_mse <= row.functional_relative_mse)
        & (
            (frame.state_relative_mse < row.state_relative_mse)
            | (frame.functional_relative_mse < row.functional_relative_mse)
        )
    ]


def ident(row: pd.Series) -> str:
    return f"{row['condition']} u{int(row['update'])}"


def pct(new: float, base: float) -> float:
    return 100.0 * (new / base - 1.0)


def main() -> None:
    out = ROOT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE_CSV)
    required = {
        "condition", "update", "target_exposures", "selected",
        "state_relative_mse", "functional_relative_mse", "combined", "primary",
        "checkpoint_path", "checkpoint_sha256",
    }
    if set(df.columns) != required:
        raise RuntimeError(f"unexpected columns: {list(df.columns)}")
    if len(df) != 40 or df.condition.value_counts().to_dict() != {"C5": 20, "C6": 20}:
        raise RuntimeError(f"unexpected candidate coverage: {df.condition.value_counts().to_dict()}")
    if df.duplicated(["condition", "update"]).any() or df.checkpoint_sha256.duplicated().any():
        raise RuntimeError("duplicate checkpoint identity")
    for col in ["state_relative_mse", "functional_relative_mse", "combined", "primary"]:
        if not df[col].map(lambda x: math.isfinite(float(x))).all():
            raise RuntimeError(f"non-finite values in {col}")

    # Cross-check every historical row against the preserved original summaries,
    # including shared document/capture coverage.
    coverage = {}
    for condition in ("C5", "C6"):
        path = SOURCE_ROOT / f"training_provenance/analysis/{condition}_validation_summary.json"
        source = json.loads(path.read_text(encoding="utf-8"))
        if source["candidate_count"] != 20 or len(source["candidates"]) != 20:
            raise RuntimeError(f"{condition}: original summary candidate count mismatch")
        for candidate in source["candidates"]:
            key = (condition, int(candidate["update"]))
            match = df[(df.condition == condition) & (df["update"] == candidate["update"])]
            if len(match) != 1:
                raise RuntimeError(f"missing source candidate {key}")
            row = match.iloc[0]
            checks = {
                "target_exposures": int(row.target_exposures) == int(candidate["target_exposures"]),
                "checkpoint_path": row.checkpoint_path == candidate["checkpoint_path"],
                "checkpoint_sha256": row.checkpoint_sha256 == candidate["checkpoint_sha256"],
                "state": math.isclose(row.state_relative_mse, candidate["validation"]["state"], rel_tol=0, abs_tol=1e-15),
                "functional": math.isclose(row.functional_relative_mse, candidate["validation"]["functional"], rel_tol=0, abs_tol=1e-15),
                "primary": math.isclose(row.primary, candidate["primary"], rel_tol=0, abs_tol=1e-15),
            }
            if not all(checks.values()):
                raise RuntimeError(f"source disagreement {key}: {checks}")
            coverage[key] = (int(candidate["validation"]["documents"]), int(candidate["validation"]["samples"]))
    if set(coverage.values()) != {(16, 128)}:
        raise RuntimeError(f"not one shared 16x128 protocol: {set(coverage.values())}")
    for key, expected in KNOWN.items():
        row = df[(df.condition == key[0]) & (df["update"] == key[1])].iloc[0]
        actual = (float(row.state_relative_mse), float(row.functional_relative_mse))
        if not all(math.isclose(a, b, rel_tol=0, abs_tol=1e-12) for a, b in zip(actual, expected)):
            raise RuntimeError(f"known-value mismatch for {key}: {actual} != {expected}")

    c5_front = nondominated(df[df.condition == "C5"])
    c6_front = nondominated(df[df.condition == "C6"])
    global_front = nondominated(df)
    df["state_rank"] = df.state_relative_mse.rank(method="min", ascending=True).astype(int)
    df["functional_rank"] = df.functional_relative_mse.rank(method="min", ascending=True).astype(int)
    df["pareto_c5"] = [i in c5_front for i in df.index]
    df["pareto_c6"] = [i in c6_front for i in df.index]
    df["pareto_global"] = [i in global_front for i in df.index]
    df["dominated_by_count"] = [len(dominators(df, row)) for _, row in df.iterrows()]
    df["documents"] = 16
    df["captures"] = 128
    df = df.rename(columns={"state_relative_mse": "state_mse", "functional_relative_mse": "functional_mse"})
    all_cols = [
        "condition", "update", "target_exposures", "selected", "state_mse", "functional_mse",
        "state_rank", "functional_rank", "pareto_c5", "pareto_c6", "pareto_global",
        "dominated_by_count", "documents", "captures", "checkpoint_path", "checkpoint_sha256",
    ]
    df[all_cols].to_csv(out / "pareto_all_checkpoints.csv", index=False)

    frontier_rows = []
    for scope, idxs in (("C5", c5_front), ("C6", c6_front), ("GLOBAL", global_front)):
        for i in sorted(idxs, key=lambda x: (df.loc[x, "state_mse"], df.loc[x, "functional_mse"])):
            r = df.loc[i]
            frontier_rows.append({
                "frontier_scope": scope, "condition": r.condition, "update": int(r["update"]),
                "target_exposures": int(r.target_exposures), "state_mse": r.state_mse,
                "functional_mse": r.functional_mse, "checkpoint_sha256": r.checkpoint_sha256,
            })
    pd.DataFrame(frontier_rows).to_csv(out / "pareto_frontier.csv", index=False)

    mandatory = [("C5", 63), ("C5", 113), ("C6", 119), ("C6", 125)]
    canonical = {"C5": 63, "C6": 125}
    details = []
    for condition, update in mandatory:
        row = df[(df.condition == condition) & (df["update"] == update)].iloc[0]
        base = df[(df.condition == condition) & (df["update"] == canonical[condition])].iloc[0]
        dom = dominators(df.rename(columns={"state_mse": "state_relative_mse", "functional_mse": "functional_relative_mse"}), row.rename({"state_mse": "state_relative_mse", "functional_mse": "functional_relative_mse"}))
        details.append({
            "id": f"{condition} u{update}", "pareto_family": bool(row[f"pareto_{condition.lower()}"]),
            "pareto_global": bool(row.pareto_global), "dominated_by": [ident(x) for _, x in dom.iterrows()],
            "state_pct_vs_family_canonical": pct(row.state_mse, base.state_mse),
            "functional_pct_vs_family_canonical": pct(row.functional_mse, base.functional_mse),
        })

    c5_dominates_c6_125 = []
    target_c6 = df[(df.condition == "C6") & (df["update"] == 125)].iloc[0]
    for _, r in df[df.condition == "C5"].iterrows():
        if r.state_mse <= target_c6.state_mse and r.functional_mse <= target_c6.functional_mse and (r.state_mse < target_c6.state_mse or r.functional_mse < target_c6.functional_mse):
            c5_dominates_c6_125.append(ident(r))
    c6_dominates_c5_63 = []
    target_c5 = df[(df.condition == "C5") & (df["update"] == 63)].iloc[0]
    for _, r in df[df.condition == "C6"].iterrows():
        if r.state_mse <= target_c5.state_mse and r.functional_mse <= target_c5.functional_mse and (r.state_mse < target_c5.state_mse or r.functional_mse < target_c5.functional_mse):
            c6_dominates_c5_63.append(ident(r))

    c5_global = df[(df.condition == "C5") & df.pareto_global]
    c6_global = df[(df.condition == "C6") & df.pareto_global]
    lines = [
        "# Phase A — offline Pareto summary", "",
        "## Integrity gate", "",
        "PASS: 40 unique finite checkpoint rows match the preserved original C5/C6 validation summaries exactly. Every row records 16 documents and 128 captures on the same frozen panel. Known prompt values agree within 1e-12.", "",
        "## Frontier counts", "",
        f"- C5-only frontier: {len(c5_front)} checkpoints — {', '.join(ident(df.loc[i]) for i in sorted(c5_front, key=lambda i: int(df.loc[i, 'update'])))}.",
        f"- C6-only frontier: {len(c6_front)} checkpoints — {', '.join(ident(df.loc[i]) for i in sorted(c6_front, key=lambda i: int(df.loc[i, 'update'])))}.",
        f"- Combined frontier: {len(global_front)} checkpoints — {', '.join(ident(df.loc[i]) for i in sorted(global_front, key=lambda i: (str(df.loc[i, 'condition']), int(df.loc[i, 'update']))))}.",
        f"- Global frontier family occupancy: C5={len(c5_global)}, C6={len(c6_global)}.", "",
        "## Mandatory checkpoints", "",
        "| Checkpoint | Family Pareto | Global Pareto | Dominated by | State Δ vs family canonical | Functional Δ vs family canonical |",
        "|---|---|---|---|---:|---:|",
    ]
    for d in details:
        lines.append(f"| {d['id']} | {d['pareto_family']} | {d['pareto_global']} | {', '.join(d['dominated_by']) or 'none'} | {d['state_pct_vs_family_canonical']:+.3f}% | {d['functional_pct_vs_family_canonical']:+.3f}% |")
    lines += [
        "", "## Explicit questions", "",
        f"- C5 checkpoints dominating canonical C6 u125: {', '.join(c5_dominates_c6_125) or 'none'}.",
        f"- C6 checkpoints dominating canonical C5 u63: {', '.join(c6_dominates_c5_63) or 'none'}.",
        f"- Does one family occupy most of the combined frontier? {'YES — C5 occupies all combined-frontier points.' if len(c5_global) == len(global_front) else 'NO — both families are represented.'}",
        "- Pareto position is not interpreted as AIME accuracy.", "",
    ]
    (out / "pareto_summary.md").write_text("\n".join(lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=180)
    colors = {"C5": "#1f77b4", "C6": "#d62728"}
    for condition in ("C5", "C6"):
        part = df[df.condition == condition]
        ax.scatter(part.state_mse, part.functional_mse, s=36, alpha=0.70, c=colors[condition], label=condition)
    gf = df[df.pareto_global].sort_values("state_mse")
    ax.plot(gf.state_mse, gf.functional_mse, color="#2ca02c", linewidth=1.5, linestyle="--", label="combined Pareto frontier")
    for condition, update in mandatory:
        r = df[(df.condition == condition) & (df["update"] == update)].iloc[0]
        ax.scatter([r.state_mse], [r.functional_mse], s=90, facecolors="none", edgecolors="black", linewidths=1.4)
        ax.annotate(f"{condition} u{update}", (r.state_mse, r.functional_mse), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("State relative-MSE (lower is better)")
    ax.set_ylabel("Functional relative-MSE (lower is better)")
    ax.set_title("H64 checkpoint shared-validation Pareto map (no AIME outcomes)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "pareto_plot.png")
    plt.close(fig)

    audit = {
        "status": "PASS", "historical_rows": len(df), "families": {"C5": 20, "C6": 20},
        "documents_per_candidate": 16, "captures_per_candidate": 128,
        "duplicate_checkpoint_ids": 0, "finite_values": True,
        "source_csv": str(SOURCE_CSV), "mandatory": details,
        "c5_dominates_canonical_c6_u125": c5_dominates_c6_125,
        "c6_dominates_canonical_c5_u63": c6_dominates_c5_63,
        "global_frontier": [ident(df.loc[i]) for i in sorted(global_front, key=lambda i: (str(df.loc[i, "condition"]), int(df.loc[i, "update"])))],
    }
    (out / "phase_a_integrity.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "global_frontier": audit["global_frontier"]}, sort_keys=True))


if __name__ == "__main__":
    main()
