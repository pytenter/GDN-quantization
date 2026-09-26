#!/usr/bin/env python3
"""Final release gates for Qwen AIME26 formal execution."""

import copy
import json
import subprocess
import tempfile
from pathlib import Path

import run_qwen_aime26_formal as runner
from aime26_common import extract_aime_answer, score_aime


REPO = Path(__file__).resolve().parents[2]
LAUNCH_ROOT = REPO / "artifacts/aime26_v2/official_sampling_81920/qwen/launch"
CRITICAL_FILES = (
    "experiments/aime26/aime26_common.py",
    "experiments/aime26/run_qwen_aime26_formal.py",
    "experiments/aime26/test_aime26_scorer.py",
    "experiments/aime26/qwen_deterministic_replay.py",
    "experiments/aime26/audit_qwen_formal_preflight.py",
    "experiments/aime26/audit_qwen_formal_status.py",
    "experiments/aime26/run_qwen_smoke_81920.sh",
    "experiments/aime26/run_qwen_formal_worker.sh",
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gate_scorer():
    cases = (
        ("\\boxed{42}", "42", "42", True),
        ("\\boxed{-7}", "-7", "-7", True),
        ("\\boxed{\\frac{29}{50}}", "29/50", "29/50", True),
        ("\\boxed{\\frac{2}{4}}", "1/2", "1/2", True),
        ("\\boxed{0.125}", "1/8", "0.125", True),
        ("\\boxed{1} then \\boxed{2}", "2", "2", True),
        ("\\boxed{\\frac{1}{}}", "", "1", False),
        ("no answer", "", "0", False),
    )
    return all(extract_aime_answer(text) == extracted and score_aime(text, gold) == (extracted, correct)
               for text, extracted, gold, correct in cases)


def valid_template():
    method = runner.METHODS["fp_state"]
    return {
        "protocol_version": runner.PROTOCOL_VERSION,
        "model": "Qwen3.5-9B",
        "method": method,
        "problem_id": "aime26_01",
        "seed": 1,
        "formal_identity": {
            "model": "Qwen3.5-9B", "configuration": method, "problem_id": "aime26_01",
            "seed": 1, "protocol_version": runner.PROTOCOL_VERSION,
        },
        "runtime_error": None, "nonfinite": False, "response": "x", "generated_token_ids": [1],
        "generated_token_count": 1, "output_tokens": 1, "finish_reason": "eos", "eos_seen": True,
        "truncated": False, "max_new_tokens": runner.MAX_NEW_TOKENS,
        "effective_generation_config": {
            "max_new_tokens": runner.MAX_NEW_TOKENS, "temperature": runner.TEMPERATURE,
            "top_p": runner.TOP_P, "top_k": runner.TOP_K, "thinking": True, "do_sample": True,
            "min_p": runner.MIN_P, "presence_penalty": runner.PRESENCE_PENALTY,
            "repetition_penalty": runner.REPETITION_PENALTY,
            "presence_penalty_scope": "generated_output_tokens_only",
            "sampling_seed": 1,
        },
    }


def gate_resume():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        good = valid_template()
        invalid = []
        for field, value in (("runtime_error", "boom"), ("response", ""), ("generated_token_ids", [])):
            row = copy.deepcopy(good); row[field] = value; invalid.append(row)
        path = root / "rows.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in [good] + invalid) + "\n", encoding="utf-8")
        if len(runner.read_done(path, "fp_state")) != 1:
            return False
        corrupt = root / "corrupt.jsonl"; corrupt.write_text("{broken\n", encoding="utf-8")
        duplicate = root / "duplicate.jsonl"
        duplicate.write_text(json.dumps(good) + "\n" + json.dumps(good) + "\n", encoding="utf-8")
        for test in (corrupt, duplicate):
            try:
                runner.read_done(test, "fp_state")
                return False
            except RuntimeError:
                pass
    return True


def main():
    dataset = runner.load_frozen_dataset(runner.DATASET)
    smoke_root = runner.OUT_ROOT / "smoke"
    smoke = {key: read_jsonl(smoke_root / ("int8_c128_key_h.jsonl" if key == "int8_c128_key_h" else f"{key}.jsonl"))
             for key in runner.METHODS}
    hashes = {}
    for rows in smoke.values():
        for row in rows:
            hashes.setdefault((row["problem_id"], int(row["seed"])), set()).add(row["input_ids_hash"])
    all_smoke = [row for rows in smoke.values() for row in rows]
    quant_rows = smoke["int8_c128"] + smoke["int8_c128_key_h"]
    h_rows = smoke["int8_c128_key_h"]
    canonical = [(method, row["problem_id"], seed) for method in runner.METHODS for row in dataset for seed in (1, 2)]
    shards = [[unit for index, unit in enumerate(canonical) if index % runner.FORMAL_WORKERS == worker]
              for worker in range(runner.FORMAL_WORKERS)]
    replay_path = LAUNCH_ROOT / "deterministic_replay.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else {}
    critical_tracked = subprocess.run(["git", "ls-files", "--error-unmatch", *CRITICAL_FILES], cwd=REPO,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    critical_clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *CRITICAL_FILES], cwd=REPO).returncode == 0
    gates = {
        "SCORER_CORRECTNESS": gate_scorer(),
        "CODE_PROVENANCE": critical_tracked and critical_clean,
        "TOKEN_PROVENANCE": all(isinstance(r.get("generated_token_ids"), list) and r["generated_token_ids"]
                                  and len(r["generated_token_ids"]) == r.get("output_tokens")
                                  and r.get("finish_reason") in {"eos", "length"} and r.get("runtime_error") is None
                                  for r in all_smoke),
        "NONFINITE_AUDIT": all(r.get("nonfinite") is False for r in all_smoke),
        "RESUME_INTEGRITY": gate_resume(),
        "FINAL_COUNT_INTEGRITY": len(canonical) == 180 and all(sum(u[0] == method for u in canonical) == 60 for method in runner.METHODS),
        "DETERMINISTIC_REPLAY": replay.get("gate") == "PASS" and replay.get("exact_token_id_match") is True,
        "INPUT_HASH_GATE": len(hashes) == 6 and all(len(values) == 1 for values in hashes.values()),
        "QUANTIZATION_PATH_GATE": all(
            r["quantizer_audit"]["calls"] > 0
            and r["quantizer_audit"]["layers"] == list(runner.GDN_LAYERS)
            and r["quantizer_audit"]["prefill_quantized"] is False
            and all(shape[-2:] == [1, 128] for shape in r["quantizer_audit"]["scale_shapes"].values())
            for r in quant_rows
        ),
        "ROTATION_PATH_GATE": all(
            r["hadamard_audit"]["calls"] > 0
            and r["hadamard_audit"]["layers"] == list(runner.GDN_LAYERS)
            and r["hadamard_audit"]["phase_counts"].get("chunk_prefill", 0) > 0
            and r["hadamard_audit"]["phase_counts"].get("recurrent_decode", 0) > 0
            for r in h_rows
        ),
        "QWEN_RUNTIME_EFFECTIVE_CONFIG": all(
            r.get("effective_generation_config") == {
                "thinking": True, "do_sample": True, "temperature": runner.TEMPERATURE,
                "top_p": runner.TOP_P, "top_k": runner.TOP_K, "min_p": runner.MIN_P,
                "presence_penalty": runner.PRESENCE_PENALTY,
                "repetition_penalty": runner.REPETITION_PENALTY,
            "presence_penalty_scope": "generated_output_tokens_only",
                "max_new_tokens": runner.MAX_NEW_TOKENS, "sampling_seed": int(r["seed"]),
            } for r in all_smoke
        ),
        "QWEN_MAX_NEW_TOKENS_81920": runner.MAX_NEW_TOKENS == 81920 and all(r.get("max_new_tokens") == 81920 for r in all_smoke),
        "QWEN_SHARD_DISJOINTNESS": len(set().union(*(set(shard) for shard in shards))) == sum(map(len, shards)),
        "QWEN_SHARD_COVERAGE": sum(map(len, shards)) == 180 and sorted(map(len, shards)) == [45, 45, 45, 45],
    }
    report = {
        "task": "GDN_KDA_AIME26_OFFICIAL_SAMPLING_81920_2SEED_FORMAL_V2_PREFLIGHT_QWEN",
        "formal_release": "GO" if all(gates.values()) else "HOLD",
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "dataset_rows": len(dataset), "assigned_units": len(canonical), "worker_units": list(map(len, shards)),
        "duplicate_units": len(canonical) - len(set(canonical)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "critical_files": list(CRITICAL_FILES),
    }
    LAUNCH_ROOT.mkdir(parents=True, exist_ok=True)
    (LAUNCH_ROOT / "preflight_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["formal_release"] == "GO" else 2)


if __name__ == "__main__":
    main()
