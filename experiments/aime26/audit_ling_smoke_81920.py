#!/usr/bin/env python3
"""Validate the 3x6 Ling AIME26 81920-token smoke and compare old truncations."""

import argparse
import hashlib
import json
from pathlib import Path


FILES = {
    "LING_FP_STATE": "fp_state.jsonl",
    "LING_INT8_R128": "int8_r128.jsonl",
    "LING_INT8_R128_VALUE_HADAMARD": "int8_r128_value_h.jsonl",
}
EXPECTED_MAX_NEW_TOKENS = 81920
OLD_MAX_NEW_TOKENS = 65536
EXPECTED_KEYS = {(f"aime26_{problem:02d}", seed) for problem in range(1, 4) for seed in (1, 2)}


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(row):
    return str(row["problem_id"]), int(row["seed"])


def finish_type(row):
    finish = row.get("finish_reason")
    return finish.get("type") if isinstance(finish, dict) else finish


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    old_by_method = {}
    new_by_method = {}
    old_files = []
    for method, filename in FILES.items():
        old_path = args.old_dir / filename
        new_path = args.new_dir / filename
        old_rows = load_jsonl(old_path)
        new_rows = load_jsonl(new_path)
        old_by_method[method] = {key(row): row for row in old_rows}
        new_by_method[method] = {key(row): row for row in new_rows}
        old_files.append(
            {
                "path": str(old_path),
                "sha256": sha256(old_path),
                "rows": len(old_rows),
                "protocol": "OLD_PROTOCOL_MAX_NEW_TOKENS_65536",
            }
        )

    marker = {
        "status": "OLD_PROTOCOL_MAX_NEW_TOKENS_65536",
        "max_new_tokens": OLD_MAX_NEW_TOKENS,
        "mix_with_current_protocol": False,
        "artifacts": old_files,
    }
    marker_path = args.old_dir / "OLD_PROTOCOL_MAX_NEW_TOKENS_65536.json"
    if marker_path.exists():
        if json.loads(marker_path.read_text(encoding="utf-8")) != marker:
            raise RuntimeError(f"existing historical marker differs: {marker_path}")
    else:
        marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result_count_gate = all(len(rows) == 6 for rows in new_by_method.values())
    unique_gate = all(set(rows) == EXPECTED_KEYS for rows in new_by_method.values())
    successful_gate = all(
        row.get("successful_sample") is True and row.get("runtime_error") is None
        for rows in new_by_method.values()
        for row in rows.values()
    )
    config_gate = all(
        row.get("max_new_tokens") == EXPECTED_MAX_NEW_TOKENS
        and row.get("protocol_version") == "GDN_KDA_AIME26_OFFICIAL_SAMPLING_81920_2SEED_FORMAL_V2"
        and row.get("temperature") == 1.0
        and row.get("top_p") == 0.95
        and row.get("top_k") == 20
        and row.get("thinking") is True
        and row.get("do_sample") is True
        and int(row.get("seed")) in (1, 2)
        and int(row.get("sampling_seed")) == int(row.get("seed"))
        for rows in new_by_method.values()
        for row in rows.values()
    )
    runtime_config_gate = all(
        row.get("ling_effective_max_new_tokens") == EXPECTED_MAX_NEW_TOKENS
        and row.get("ling_effective_max_new_tokens_gate") is True
        and row.get("runtime_effective_sampling_params", {}).get("max_new_tokens") == EXPECTED_MAX_NEW_TOKENS
        and row.get("runtime_effective_sampling_params", {}).get("temperature") == 1.0
        and row.get("runtime_effective_sampling_params", {}).get("top_p") == 0.95
        and row.get("runtime_effective_sampling_params", {}).get("top_k") == 20
        and int(row.get("runtime_effective_sampling_params", {}).get("sampling_seed")) == int(row.get("seed"))
        for rows in new_by_method.values()
        for row in rows.values()
    )
    rotation_semantics_gate = all(
        row.get("kda_rotation_semantics_version") == "CORRECTED_PREFILL_ENDPOINT_V2"
        and row.get("redundant_prefill_endpoint_rotation") is False
        for row in new_by_method["LING_INT8_R128_VALUE_HADAMARD"].values()
    )
    token_provenance_gate = all(
        row.get("token_provenance_valid") is True
        and isinstance(row.get("generated_token_ids"), list)
        and len(row["generated_token_ids"]) == row.get("generated_token_count")
        and row.get("finish_reason") is not None
        and isinstance(row.get("eos_seen"), bool)
        and isinstance(row.get("truncated"), bool)
        for rows in new_by_method.values()
        for row in rows.values()
    )
    nonfinite_gate = all(
        row.get("nonfinite") is False
        and row.get("finite_checked_output_tokens") == row.get("generated_token_count")
        for rows in new_by_method.values()
        for row in rows.values()
    )

    input_hash_gate = True
    for sample_key in EXPECTED_KEYS:
        new_hashes = {
            rows[sample_key].get("input_ids_hash")
            for rows in new_by_method.values()
            if sample_key in rows
        }
        old_hashes = {
            rows[sample_key].get("input_ids_hash")
            for rows in old_by_method.values()
            if sample_key in rows
        }
        if (
            len(new_hashes) != 1
            or len(old_hashes) != 1
            or None in new_hashes
            or None in old_hashes
            or new_hashes != old_hashes
        ):
            input_hash_gate = False

    prior_truncations = []
    for method, old_rows in old_by_method.items():
        for sample_key, old in sorted(old_rows.items()):
            if not old.get("truncated"):
                continue
            new = new_by_method[method].get(sample_key)
            if new is None or new.get("runtime_error"):
                classification = "ERROR"
            elif new.get("truncated"):
                classification = "STILL_TRUNCATED_AT_81920"
            elif finish_type(new) == "stop" and new.get("generated_token_count", 0) < EXPECTED_MAX_NEW_TOKENS:
                classification = "RESOLVED_BEFORE_81920"
            else:
                classification = "OTHER_TERMINATION"
            prior_truncations.append(
                {
                    "method": method,
                    "problem_id": sample_key[0],
                    "seed": sample_key[1],
                    "old_finish_reason": old.get("finish_reason"),
                    "old_generated_token_count": old.get("output_tokens"),
                    "new_finish_reason": None if new is None else new.get("finish_reason"),
                    "new_generated_token_count": None if new is None else new.get("generated_token_count"),
                    "old_truncated": True,
                    "new_truncated": None if new is None else new.get("truncated"),
                    "classification": classification,
                }
            )

    summaries = {}
    for method, rows in new_by_method.items():
        values = list(rows.values())
        summaries[method] = {
            "rows": len(values),
            "correct": sum(bool(row.get("correct")) for row in values),
            "truncated": sum(bool(row.get("truncated")) for row in values),
            "errors": sum(not bool(row.get("successful_sample")) for row in values),
        }

    report = {
        "task": "GDN_KDA_AIME26_OFFICIAL_SAMPLING_81920_2SEED_FORMAL_V2_SMOKE",
        "old_max_new_tokens": OLD_MAX_NEW_TOKENS,
        "new_max_new_tokens": EXPECTED_MAX_NEW_TOKENS,
        "ling_max_new_tokens_config_gate": config_gate,
        "ling_runtime_effective_config_gate": runtime_config_gate,
        "ling_input_hash_gate": input_hash_gate,
        "ling_unique_problem_seed_gate": unique_gate,
        "ling_token_provenance_gate": token_provenance_gate,
        "ling_nonfinite_gate": nonfinite_gate,
        "ling_kda_corrected_rotation_semantics_gate": rotation_semantics_gate,
        "ling_result_count_gate": result_count_gate and successful_gate,
        "expected_rows": 18,
        "actual_rows": sum(value["rows"] for value in summaries.values()),
        "summaries": summaries,
        "previously_truncated_samples": prior_truncations,
        "formal_started": False,
        "formal_release": "HOLD",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

    gates = [
        config_gate,
        runtime_config_gate,
        input_hash_gate,
        unique_gate,
        token_provenance_gate,
        nonfinite_gate,
        rotation_semantics_gate,
        result_count_gate,
        successful_gate,
    ]
    if not all(gates):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
