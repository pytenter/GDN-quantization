#!/usr/bin/env python3
"""Freeze prospective discordant pairs without reading any long-horizon outcome."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/data/zypan/experiment_runs/LING_KDA_PERSISTENT_SURROGATE_PROSPECTIVE_VALIDATION_V1")


def atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"NO_OVERWRITE:{path}")
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(value, encoding="utf-8")
    os.link(tmp, path)
    tmp.unlink()


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ordinal_ranks(rows, key):
    ordered = sorted(rows, key=lambda x: (float(x[key]), x["condition"]))
    return {x["condition"]: i + 1 for i, x in enumerate(ordered)}


def pair_candidates(rows, left, right, excluded=frozenset()):
    ranks_l, ranks_r = ordinal_ranks(rows, left), ordinal_ranks(rows, right)
    by = {x["condition"]: x for x in rows}
    names = sorted(by)
    found = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            dl = float(by[a][left]) - float(by[b][left])
            dr = float(by[a][right]) - float(by[b][right])
            if dl == 0 or dr == 0 or dl * dr >= 0:
                continue
            gap_l, gap_r = abs(ranks_l[a] - ranks_l[b]), abs(ranks_r[a] - ranks_r[b])
            if min(gap_l, gap_r) < 3:
                continue
            used = len({a, b} & set(excluded))
            found.append({
                "conditions": [a, b],
                "left_metric": left,
                "right_metric": right,
                "left_scores": {a: by[a][left], b: by[b][left]},
                "right_scores": {a: by[a][right], b: by[b][right]},
                "left_rank_gap": gap_l,
                "right_rank_gap": gap_r,
                "minimum_rank_gap": min(gap_l, gap_r),
                "left_predicted_winner": a if float(by[a][left]) < float(by[b][left]) else b,
                "right_predicted_winner": a if float(by[a][right]) < float(by[b][right]) else b,
                "overlap_with_primary_count": used,
            })
    found.sort(key=lambda x: (x["overlap_with_primary_count"], -x["minimum_rank_gap"],
                              -x["left_rank_gap"], -x["right_rank_gap"], tuple(x["conditions"])))
    return found


def main():
    config = json.loads((ROOT / "frozen_config.json").read_text())
    selection = json.loads((ROOT / "PROSPECTIVE_DOCUMENT_SELECTION.json").read_text())
    output_rows = []
    pair_freeze = {
        "task": config["experiment"],
        "outcome_blind": True,
        "long_horizon_artifacts_read": False,
        "risk_direction": config["proxy_risk_direction"],
        "primary_rule": config["primary_pair_rule"],
        "secondary_rule": config["secondary_pair_rule"],
        "documents": {},
    }
    for doc in selection["selected_documents"]:
        document = doc["document_id"]
        tag = f"calibration_{document}_gpu{doc['physical_gpu']}"
        path = ROOT / "runs" / f"{tag}.json"
        run = json.loads(path.read_text())
        if not run.get("valid") or run["document"] != document or len(run["results"]) != 16:
            raise RuntimeError(f"CALIBRATION_INVALID:{document}")
        rows = []
        for item in run["results"]:
            row = {"document": document, "gpu": doc["physical_gpu"], "condition": item["condition"],
                   **{k: v for k, v in item["aggregate"].items() if isinstance(v, (int, float))}}
            rows.append(row)
            output_rows.append(row)
        primary = pair_candidates(rows, "P1", "B0")
        if not primary:
            pair_freeze["documents"][document] = {
                "primary_status": "PRIMARY_DISCORDANT_PAIR_NOT_FOUND",
                "secondary_status": "NOT_EVALUATED_WITHOUT_PRIMARY",
                "calibration_run": {"path": str(path), "sha256": sha256(path)},
            }
            continue
        chosen_primary = primary[0]
        secondary = pair_candidates(rows, "P1", "P3_H128", chosen_primary["conditions"])
        pair_freeze["documents"][document] = {
            "primary_status": "FROZEN",
            "primary_pair": chosen_primary,
            "primary_candidate_count": len(primary),
            "secondary_status": "FROZEN" if secondary else "P1_VS_P3_DISCRIMINATION_NOT_AVAILABLE",
            "secondary_pair": secondary[0] if secondary else None,
            "secondary_candidate_count": len(secondary),
            "calibration_run": {"path": str(path), "sha256": sha256(path)},
        }
    fields = sorted({k for row in output_rows for k in row})
    from io import StringIO
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader(); writer.writerows(output_rows)
    atomic_text(ROOT / "CALIBRATION_PROXY_SCORES.csv", buffer.getvalue())
    pair_freeze["primary_pairs_complete"] = all(
        x.get("primary_status") == "FROZEN" for x in pair_freeze["documents"].values()
    )
    pair_freeze["secondary_pairs_complete"] = all(
        x.get("secondary_status") == "FROZEN" for x in pair_freeze["documents"].values()
    )
    pair_freeze["calibration_csv_sha256"] = sha256(ROOT / "CALIBRATION_PROXY_SCORES.csv")
    atomic_json(ROOT / "PROSPECTIVE_ROTATION_SELECTION.json", pair_freeze)
    if not pair_freeze["primary_pairs_complete"]:
        atomic_json(ROOT / "STOP.json", {"status": "PRIMARY_DISCORDANT_PAIR_NOT_FOUND", "stage": "PAIR_SELECTION"})
        raise RuntimeError("PRIMARY_DISCORDANT_PAIR_NOT_FOUND")
    print(json.dumps({"primary_pairs_complete": True,
                      "secondary_pairs_complete": pair_freeze["secondary_pairs_complete"],
                      "documents": pair_freeze["documents"]}, sort_keys=True))


if __name__ == "__main__":
    main()
