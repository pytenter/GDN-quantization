#!/usr/bin/env python3
"""Create additive diagnostic rescoring artifacts without modifying source JSONL."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from aime26_common import DIAGNOSTIC_SCORER_VERSION, STRICT_SCORER_VERSION
from aime26_diagnostic_scorer import score_aime_layers


TASK = "AIME26_SCORER_ROBUSTNESS_AND_DIAGNOSTIC_EXTRACTION_V1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-configuration", action="append")
    parser.add_argument("--expected-per-configuration", type=int)
    args = parser.parse_args()

    output_rows = []
    for source in args.input:
        source = source.resolve()
        source_hash = sha256(source)
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            original = json.loads(line)
            scoring = score_aime_layers(original.get("response", ""), original["gold_answer"])
            if bool(original.get("correct")) != scoring["strict_correct"]:
                raise RuntimeError(
                    f"frozen strict score mismatch at {source}:{line_number}: "
                    f"stored={original.get('correct')} recomputed={scoring['strict_correct']}"
                )
            output_rows.append(
                {
                    "task": TASK,
                    "model": original.get("model"),
                    "configuration": original.get("configuration", original.get("method")),
                    "problem_id": original.get("problem_id"),
                    "seed": int(original.get("seed")),
                    "original_result_path": str(source),
                    "original_result_sha256": source_hash,
                    "original_line_number": line_number,
                    "original_protocol_version": original.get("protocol_version"),
                    "original_git_commit": original.get("git_commit"),
                    "original_correct_preserved": bool(original.get("correct")),
                    **scoring,
                }
            )

    identities = [
        (
            row["model"],
            row["configuration"],
            row["problem_id"],
            row["seed"],
            row["original_result_path"],
            row["original_line_number"],
        )
        for row in output_rows
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate diagnostic rescore identity")

    by_configuration = defaultdict(list)
    for row in output_rows:
        by_configuration[(row["model"], row["configuration"])].append(row)

    summaries = {}
    for (model, configuration), rows in sorted(by_configuration.items()):
        total = len(rows)
        strict_correct = sum(row["strict_correct"] for row in rows)
        diagnostic_correct = sum(row["diagnostic_correct"] for row in rows)
        nonboxed_diagnostic_correct = sum(
            row["diagnostic_correct"]
            and row["answer_format_compliance"] != "BOXED_COMPLIANT"
            for row in rows
        )
        format_only = sum(
            row["diagnostic_correct"] and not row["strict_correct"] for row in rows
        )
        key = f"{model}/{configuration}"
        summaries[key] = {
            "model": model,
            "configuration": configuration,
            "N_total": total,
            "N_boxed_compliant": sum(
                row["answer_format_compliance"] == "BOXED_COMPLIANT" for row in rows
            ),
            "N_nonboxed_but_diagnostic_correct": nonboxed_diagnostic_correct,
            "N_unextractable": sum(
                row["diagnostic_parse_status"] == "NO_ANSWER_FOUND" for row in rows
            ),
            "strict_correct": strict_correct,
            "diagnostic_correct": diagnostic_correct,
            "strict_accuracy": strict_correct / total if total else None,
            "diagnostic_accuracy": diagnostic_correct / total if total else None,
            "format_only_loss": format_only,
            "format_failure_rate": format_only / total if total else None,
        }

    known = {}
    for problem_id, seed in (("aime26_01", 2), ("aime26_02", 2)):
        matches = [
            row for row in output_rows
            if row["model"] == "Ling-3.0-tiny"
            and row["configuration"] == "LING_FP_STATE"
            and row["problem_id"] == problem_id
            and row["seed"] == seed
        ]
        name = f"{problem_id}_seed{seed}"
        known[name] = matches[0] if len(matches) == 1 else {"match_count": len(matches)}

    ling_fp_present = any(
        row["model"] == "Ling-3.0-tiny" and row["configuration"] == "LING_FP_STATE"
        for row in output_rows
    )
    known_ling_gate = (
        all(
            isinstance(value, dict)
            and value.get("strict_correct") is False
            and value.get("diagnostic_correct") is True
            and value.get("diagnostic_extracted_answer") == expected
            and value.get("diagnostic_parse_source") == "FINAL_ANSWER_LABEL"
            for (name, value), expected in zip(sorted(known.items()), ("277", "62"))
        )
        if ling_fp_present else None
    )

    actual_configurations = {row["configuration"] for row in output_rows}
    expected_configurations = set(args.expected_configuration or actual_configurations)
    result_count_gate = (
        actual_configurations == expected_configurations
        and (
            args.expected_per_configuration is None
            or all(
                len(by_configuration[(next(row["model"] for row in output_rows if row["configuration"] == configuration), configuration)])
                == args.expected_per_configuration
                for configuration in expected_configurations
            )
        )
    )

    report = {
        "task": TASK,
        "status": "COMPLETE" if result_count_gate and known_ling_gate is not False else "INCOMPLETE",
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "diagnostic_scorer_version": DIAGNOSTIC_SCORER_VERSION,
        "strict_scorer_modified": "NO",
        "primary_formal_metric": "STRICT",
        "rows": len(output_rows),
        "result_count_gate": "PASS" if result_count_gate else "FAIL",
        "expected_configurations": sorted(expected_configurations),
        "expected_per_configuration": args.expected_per_configuration,
        "summaries": summaries,
        "known_ling_cases": known,
        "known_ling_cases_gate": "PASS" if known_ling_gate is True else "NOT_APPLICABLE" if known_ling_gate is None else "FAIL",
        "historical_artifacts_overwritten": "NO",
        "formal_generation_semantics_changed": "NO",
        "future_scorer_recommendation": "KEEP_STRICT_PLUS_DIAGNOSTIC",
    }

    if known_ling_gate is False:
        raise RuntimeError("known Ling diagnostic extraction gate failed")
    if not result_count_gate:
        raise RuntimeError("diagnostic rescore result-count gate failed")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
