#!/usr/bin/env python3
"""Final release gates for Ling AIME26 formal execution."""

import copy
import json
import subprocess
import tempfile
from pathlib import Path

import run_ling_sglang_aime26 as runner
from aime26_common import extract_aime_answer, load_frozen_dataset, score_aime


REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "artifacts/aime26_v1/formal/dataset/aime26_frozen.jsonl"
SMOKE_ROOT = REPO / "artifacts/aime26_v2/official_sampling_81920/ling/smoke"
PARITY_ROOT = REPO / "artifacts/aime26_v2/official_sampling_81920/ling/parity"
LAUNCH_ROOT = REPO / "artifacts/aime26_v2/official_sampling_81920/ling/launch"
CRITICAL_FILES = (
    "experiments/aime26/aime26_common.py",
    "experiments/aime26/aime26_diagnostic_scorer.py",
    "experiments/aime26/rescore_aime26_diagnostic.py",
    "experiments/aime26/test_aime26_diagnostic_scorer.py",
    "experiments/aime26/run_ling_sglang_aime26.py",
    "experiments/aime26/launch_ling_sglang.py",
    "experiments/aime26/launch_ling_sglang_server.sh",
    "experiments/aime26/sglang_kda_runtime_patch.py",
    "experiments/aime26/test_aime26_scorer.py",
    "experiments/aime26/ling_deterministic_replay.py",
    "experiments/aime26/audit_ling_formal_preflight.py",
    "experiments/aime26/audit_ling_formal_status.py",
    "experiments/aime26/run_ling_smoke_81920.sh",
    "experiments/aime26/audit_ling_smoke_81920.py",
    "experiments/aime26/run_ling_formal_worker.sh",
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gate_scorer():
    cases = (
        ("\\boxed{42}", "42", "42", True), ("\\boxed{-7}", "-7", "-7", True),
        ("\\boxed{\\frac{29}{50}}", "29/50", "29/50", True),
        ("\\boxed{\\frac{2}{4}}", "1/2", "1/2", True), ("\\boxed{0.125}", "1/8", "0.125", True),
        ("\\boxed{1} then \\boxed{2}", "2", "2", True), ("\\boxed{\\frac{1}{}}", "", "1", False),
        ("no answer", "", "0", False),
    )
    return all(extract_aime_answer(text) == extracted and score_aime(text, gold) == (extracted, correct)
               for text, extracted, gold, correct in cases)


def valid_template():
    method = runner.METHODS["fp_state"][0]
    return {
        "protocol_version": runner.FORMAL_PROTOCOL_VERSION, "model": "Ling-3.0-tiny", "method": method,
        "problem_id": "aime26_01", "seed": 1,
        "formal_identity": {"model": "Ling-3.0-tiny", "configuration": method, "problem_id": "aime26_01",
                            "seed": 1, "protocol_version": runner.FORMAL_PROTOCOL_VERSION},
        "successful_sample": True, "runtime_error": None, "nonfinite": False, "response": "x",
        "generated_token_ids": [1], "generated_token_count": 1, "output_tokens": 1,
        "finish_reason": {"type": "stop", "matched": 1}, "eos_seen": True, "truncated": False,
        "token_provenance_valid": True, "max_new_tokens": runner.MAX_NEW_TOKENS,
        "ling_effective_max_new_tokens": runner.MAX_NEW_TOKENS,
        "runtime_effective_sampling_params": {"max_new_tokens": runner.MAX_NEW_TOKENS,
            "temperature": runner.TEMPERATURE, "top_p": runner.TOP_P, "top_k": runner.TOP_K, "sampling_seed": 1},
        "thinking": True,
    }


def gate_resume():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); good = valid_template(); invalid = []
        for field, value in (("runtime_error", "boom"), ("response", ""), ("generated_token_ids", [])):
            row = copy.deepcopy(good); row[field] = value; invalid.append(row)
        path = root / "rows.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in [good] + invalid) + "\n", encoding="utf-8")
        if len(runner.read_valid_done(path, "fp_state")) != 1:
            return False
        corrupt = root / "corrupt.jsonl"; corrupt.write_text("{broken\n", encoding="utf-8")
        duplicate = root / "duplicate.jsonl"
        duplicate.write_text(json.dumps(good) + "\n" + json.dumps(good) + "\n", encoding="utf-8")
        for test in (corrupt, duplicate):
            try:
                runner.read_valid_done(test, "fp_state"); return False
            except RuntimeError:
                pass
    return True


def phase_layers(events, event_name, mode, phase):
    return {int(event["layer"]) for event in events if event.get("event") == event_name
            and event.get("mode") == mode and event.get("phase") == phase}


def main():
    dataset = load_frozen_dataset(DATASET)
    smoke = {key: read_jsonl(SMOKE_ROOT / f"{key}.jsonl") for key in runner.METHODS}
    all_smoke = [row for rows in smoke.values() for row in rows]
    hashes = {}
    for rows in smoke.values():
        for row in rows:
            hashes.setdefault((row["problem_id"], int(row["seed"])), set()).add(row["input_ids_hash"])
    r_events = read_jsonl(PARITY_ROOT / "smoke81920_r128_audit.jsonl")
    h_events = read_jsonl(PARITY_ROOT / "smoke81920_r128_h_v2_audit.jsonl")
    layers = {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22}
    gate_d = json.loads((PARITY_ROOT / "gate_d_hadamard.json").read_text(encoding="utf-8"))
    canonical = [(method, row["problem_id"], seed) for method in runner.METHODS for row in dataset for seed in runner.SEEDS]
    shards = [[unit for index, unit in enumerate(canonical) if index % runner.FORMAL_WORKERS == worker]
              for worker in range(runner.FORMAL_WORKERS)]
    replay_path = LAUNCH_ROOT / "deterministic_replay.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else {}
    critical_tracked = subprocess.run(["git", "ls-files", "--error-unmatch", *CRITICAL_FILES], cwd=REPO,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    critical_clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *CRITICAL_FILES], cwd=REPO).returncode == 0
    q_modes = ((r_events, "int8_r128"), (h_events, "int8_r128_value_h"))
    gates = {
        "SCORER_CORRECTNESS": gate_scorer(), "CODE_PROVENANCE": critical_tracked and critical_clean,
        "TOKEN_PROVENANCE": all(r.get("token_provenance_valid") is True and r.get("token_ids_source")
            in {"response.output_ids", "meta_info.output_token_logprobs"} and isinstance(r.get("generated_token_ids"), list)
            and r["generated_token_ids"] and len(r["generated_token_ids"]) == r.get("generated_token_count")
            and r.get("runtime_error") is None for r in all_smoke),
        "NONFINITE_AUDIT": all(r.get("nonfinite") is False for r in all_smoke),
        "RESUME_INTEGRITY": gate_resume(),
        "FINAL_COUNT_INTEGRITY": len(canonical) == 180 and all(sum(u[0] == method for u in canonical) == 60 for method in runner.METHODS),
        "DETERMINISTIC_REPLAY": replay.get("gate") == "PASS" and replay.get("exact_token_id_match") is True,
        "INPUT_HASH_GATE": len(hashes) == 6 and all(len(values) == 1 for values in hashes.values()),
        "QUANTIZATION_PATH_GATE": all(phase_layers(events, "kda_state_qdq", mode, phase) == layers
            for events, mode in q_modes for phase in ("prefill", "decode")),
        "ROTATION_PATH_GATE": gate_d.get("gate") == "PASS"
            and gate_d.get("KDA_ROTATION_SEMANTICS_VERSION") == runner.KDA_ROTATION_SEMANTICS_VERSION
            and gate_d.get("REDUNDANT_PREFILL_ENDPOINT_ROTATION") == "NO"
            and gate_d.get("prefill_endpoint_basis_gate") == "PASS"
            and all(phase_layers(h_events, "kda_value_hadamard", "int8_r128_value_h", phase) == layers
                    for phase in ("prefill", "decode")),
        "LING_RUNTIME_EFFECTIVE_CONFIG": all(
            r.get("thinking") is True and r.get("do_sample") is True
            and r.get("runtime_effective_sampling_params") == {
                "temperature": runner.TEMPERATURE, "top_p": runner.TOP_P,
                "top_k": runner.TOP_K, "max_new_tokens": runner.MAX_NEW_TOKENS,
                "sampling_seed": int(r["seed"]),
            } for r in all_smoke
        ),
        "LING_MAX_NEW_TOKENS_81920": runner.MAX_NEW_TOKENS == 81920 and all(r.get("max_new_tokens") == 81920 for r in all_smoke),
        "LING_SHARD_DISJOINTNESS": len(set().union(*(set(shard) for shard in shards))) == sum(map(len, shards)),
        "LING_SHARD_COVERAGE": sum(map(len, shards)) == 180 and sorted(map(len, shards)) == [90, 90],
    }
    report = {
        "task": "GDN_KDA_AIME26_OFFICIAL_SAMPLING_81920_2SEED_FORMAL_V2_PREFLIGHT_LING",
        "formal_release": "GO" if all(gates.values()) else "HOLD",
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "dataset_rows": len(dataset), "assigned_units": len(canonical), "worker_units": list(map(len, shards)),
        "duplicate_units": len(canonical) - len(set(canonical)),
        "kda_rotation_semantics_version": runner.KDA_ROTATION_SEMANTICS_VERSION,
        "redundant_prefill_endpoint_rotation": "NO",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "critical_files": list(CRITICAL_FILES),
    }
    LAUNCH_ROOT.mkdir(parents=True, exist_ok=True)
    (LAUNCH_ROOT / "preflight_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["formal_release"] == "GO" else 2)


if __name__ == "__main__":
    main()
