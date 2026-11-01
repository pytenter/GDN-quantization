#!/usr/bin/env python3
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


TASK = "LING_KDA_RC_GRANULARITY_PROSPECTIVE_CONFIRMATION_V1"
SLUG = "ling_kda_rc_granularity_prospective_confirmation_v1"
REPO = Path(os.environ.get("LING_REPO_ROOT", Path(__file__).resolve().parents[2]))
RUN_DIR = Path(os.environ.get("LING_KDA_RC_GRANULARITY_RUN_DIR", REPO / "runs" / SLUG))
RESULT_DIR = REPO / "results" / "ling"
REPORT_DIR = REPO / "reports" / "ling"
G64_RUN_DIR = REPO / "runs" / "ling_kda_g64_prospective_confirmation_v1"
G64_RUNNER_PATH = REPO / "experiments" / "ling" / "run_ling_kda_g64_prospective_confirmation_v1.py"
CURRENT_RUNNER_PATH = REPO / "experiments" / "ling" / "run_ling_kda_int8_rc_granularity_phase_diagram_v1.py"

CONFIGS = ["R128", "C128", "R32", "C32", "R16", "C16"]
GROUP_SIZES = [128, 32, 16]
HISTORY_LENGTHS = [1, 4, 8, 16, 32, 64]
N_PROSPECTIVE_UNITS = 72
N_PRIMARY_UNITS = 36
HORIZON = 128
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260911
NEAR_TIE = 1e-12
KL_ATOL = 1e-7

G64_PRIMARY_EFFECT = 5.390129283122575e-05
G64_BOOTSTRAP_CI = [2.352435618942729e-05, 8.709634015656911e-05]
G64_PROSPECTIVE_CLASSIFICATION = "ROBUST_R_PREFERENCE"

BLOCKING_GATES = [
    "PROSPECTIVE_MANIFEST_REUSE_GATE",
    "NO_OUTCOME_BASED_SUBSET_SELECTION_GATE",
    "PIPELINE_IDENTITY_WITH_G64_GATE",
    "PIPELINE_REPEATABILITY_GATE",
    "STATE_SEMANTICS_GATE",
    "R128_QUANTIZER_SEMANTICS_GATE",
    "C128_QUANTIZER_SEMANTICS_GATE",
    "R32_QUANTIZER_SEMANTICS_GATE",
    "C32_QUANTIZER_SEMANTICS_GATE",
    "R16_QUANTIZER_SEMANTICS_GATE",
    "C16_QUANTIZER_SEMANTICS_GATE",
    "ALL_QUANTIZER_IDENTITY_GATES",
    "PERSISTENT_RECURRENT_PROTOCOL_GATE",
    "INTERVENTION_TIMING_GATE",
    "CURRENT_LOGIT_NONRETROACTIVITY_GATE",
    "NO_FURTHER_UNINTENDED_QUANTIZATION_GATE",
    "INSTRUMENTATION_NONINTERFERENCE_GATE",
    "DETERMINISM_GATE",
]

REQUIRED_OUTPUT_FILES = [
    "README.md",
    "report.md",
    "summary.json",
    "pipeline_identity_audit.json",
    "pipeline_hashes.csv",
    "environment.json",
    "primary_36_unit_manifest.csv",
    "primary_36_unit_manifest.json",
    "adaptive_expansion_decisions.json",
    "results.jsonl",
    "per_unit_per_history_effects.csv",
    "history_level_summary.csv",
    "group_level_summary.csv",
    "bootstrap_results.json",
    "robustness_analysis.csv",
    "robustness_report.md",
    "heterogeneity_analysis.csv",
    "prospective_granularity_summary.csv",
]

REQUIRED_CONSOLE_KEYS = [
    "TASK",
    "FORMAL_STATUS",
    *BLOCKING_GATES,
    "N_REUSED_PROSPECTIVE_UNITS",
    "N_PRIMARY_UNITS",
    "N_EXPECTED_PRIMARY_RUNS",
    "N_SUCCESSFUL_PRIMARY_RUNS",
    "N_ADDITIONAL_EXPANSION_RUNS",
    "N_TOTAL_SUCCESSFUL_QUANTIZED_RUNS",
    "PRIMARY_EFFECT_DEFINITION",
    "G128_PRIMARY_EFFECT",
    "G128_BOOTSTRAP_CI",
    "G128_PROSPECTIVE_CLASSIFICATION",
    "G32_PRIMARY_EFFECT",
    "G32_BOOTSTRAP_CI",
    "G32_PROSPECTIVE_CLASSIFICATION",
    "G16_PRIMARY_EFFECT",
    "G16_BOOTSTRAP_CI",
    "G16_PROSPECTIVE_CLASSIFICATION",
    "G64_PRIMARY_EFFECT",
    "G64_BOOTSTRAP_CI",
    "G64_PROSPECTIVE_CLASSIFICATION",
    "LING_KDA_PROSPECTIVE_GRANULARITY_PATTERN",
    "LING_KDA_ORIENTATION_GRANULARITY_RESULT",
    "FRESH_OUTPUT_VERIFICATION",
    "PY_COMPILE",
    "PYTEST",
    "GPU_RELEASE_VERIFICATION",
    "METHOD_DESIGN_READY",
    "NO_NEXT_EXPERIMENT_LAUNCHED",
    "CURRENT_TASK_COMPLETED_AND_STOPPED",
    "NEXT_RECOMMENDED_TASK",
]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd, cwd=REPO):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def iter_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json_sha256(obj):
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def as_float(x):
    return float(x) if finite(x) else None


def mean(xs):
    vals = [float(x) for x in xs if finite(x)]
    return sum(vals) / len(vals) if vals else None


def median(xs):
    vals = [float(x) for x in xs if finite(x)]
    return statistics.median(vals) if vals else None


def trimmed_mean(xs, proportion=0.10):
    vals = sorted(float(x) for x in xs if finite(x))
    if not vals:
        return None
    trim = int(len(vals) * proportion)
    trimmed = vals[trim:len(vals) - trim] if trim and len(vals) > 2 * trim else vals
    return sum(trimmed) / len(trimmed)


def binomial_p_two_sided(k, n, p=0.5):
    if n <= 0:
        return None
    obs = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    total = 0.0
    for i in range(n + 1):
        prob = math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        if prob <= obs + 1e-15:
            total += prob
    return min(total, 1.0)


def bootstrap_ci(vals, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED, statistic=median):
    vals = [float(x) for x in vals if finite(x)]
    if not vals:
        return None
    rng = random.Random(seed)
    out = []
    for _ in range(int(reps)):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        out.append(float(statistic(sample)))
    out.sort()
    return [out[int(0.025 * (len(out) - 1))], out[int(0.975 * (len(out) - 1))]]


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_g64_runner():
    return import_module(G64_RUNNER_PATH, f"{SLUG}_g64_reused")


def import_current_runner(run_dir=None, task=TASK, manifest_sha=None):
    g64 = import_g64_runner()
    runner = g64.import_current_runner(run_dir=run_dir, task=task, manifest_sha=manifest_sha)
    return runner


def row_config(row):
    cfg = row.get("config") or row.get("quantizer_name")
    if cfg:
        return str(cfg).upper()
    if row.get("orientation") and row.get("group_size"):
        return f"{str(row['orientation']).upper()}{int(float(row['group_size']))}"
    return "UNKNOWN"


def expected_primary_runs():
    return N_PRIMARY_UNITS * len(CONFIGS) * len(HISTORY_LENGTHS)


def expected_expansion_runs(groups):
    return 36 * 2 * len(HISTORY_LENGTHS) * len(list(groups))


def normalize_unit(unit, index=None):
    row = dict(unit)
    row["problem_id"] = str(row.get("problem_id", row.get("prompt_id")))
    row["prompt_id"] = str(row.get("prompt_id", row["problem_id"]))
    row["sample_id"] = str(row.get("sample_id", row["problem_id"]))
    row["problem_index"] = int(row.get("problem_index", row["problem_id"]))
    row["t0"] = int(row["t0"])
    row["future_horizon"] = int(row.get("future_horizon", HORIZON))
    row.setdefault("layer", "ALL_KDA")
    row.setdefault("head", "ALL")
    row["unit_id"] = str(row.get("unit_id") or f"{row['problem_id']}|{row['t0']}")
    if index is not None:
        row["index"] = int(index)
    return row


def load_g64_manifest():
    path = G64_RUN_DIR / "new_unit_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing completed g64 manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_primary_subset_from_g64_manifest(manifest):
    units = [normalize_unit(u, i) for i, u in enumerate(manifest.get("units", []))]
    exact_count = len(units) == N_PROSPECTIVE_UNITS
    manifest_sha = manifest.get("sha256") or canonical_json_sha256({k: v for k, v in manifest.items() if k != "sha256"})
    primary_units = [normalize_unit(u, i) for i, u in enumerate(units[:N_PRIMARY_UNITS])]
    primary = {
        "TASK": TASK,
        "FORMAL_STATUS": "PRIMARY_SUBSET_FROZEN_BEFORE_G128_G32_G16_OUTCOMES",
        "source_task": manifest.get("TASK"),
        "source_manifest_sha256": manifest_sha,
        "N_REUSED_PROSPECTIVE_UNITS": len(units),
        "N_PRIMARY_UNITS": len(primary_units),
        "selection_rule": "first 36 units according to the frozen g64 prospective manifest order",
        "NO_OUTCOME_BASED_SUBSET_SELECTION_GATE": "PASS" if len(primary_units) == N_PRIMARY_UNITS else "FAIL",
        "units": primary_units,
    }
    primary["sha256"] = canonical_json_sha256({k: v for k, v in primary.items() if k != "sha256"})
    audit = {
        "TASK": TASK,
        "PROSPECTIVE_MANIFEST_REUSE_GATE": "PASS" if exact_count else "FAIL",
        "NO_OUTCOME_BASED_SUBSET_SELECTION_GATE": primary["NO_OUTCOME_BASED_SUBSET_SELECTION_GATE"],
        "N_REUSED_PROSPECTIVE_UNITS": len(units),
        "source_manifest_sha256": manifest_sha,
        "primary_manifest_sha256": primary["sha256"],
        "g64_not_rerun": True,
        "subset_selection_outcome_blind": True,
        "unit_order_rule": primary["selection_rule"],
    }
    return audit, primary


def pass_gate_summary():
    return {key: "PASS" for key in BLOCKING_GATES}


def prior_current_runner_hash():
    path = G64_RUN_DIR / "pipeline_hashes.csv"
    for row in read_csv(path):
        if row.get("label") == "current_v2_runner":
            return row.get("sha256")
    return None


def model_config_hash(runner):
    path = runner.P.MODEL_PATH / "config.json"
    return file_sha256(path) if path.exists() else None


def write_pipeline_identity_files(manifest_audit):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    provenance_dir = RUN_DIR / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    files = [
        ("rc_granularity_runner", Path(__file__).resolve()),
        ("current_v2_runner", CURRENT_RUNNER_PATH),
        ("g64_prospective_runner", G64_RUNNER_PATH),
        ("g64_manifest", G64_RUN_DIR / "new_unit_manifest.json"),
        ("g64_summary", G64_RUN_DIR / "summary.json"),
        ("rc_tests", REPO / "tests" / "ling" / "test_ling_kda_rc_granularity_prospective_confirmation.py"),
    ]
    hash_rows = []
    for label, path in files:
        exists = path.exists()
        sha = file_sha256(path) if exists and path.is_file() else None
        copied = None
        if exists and path.is_file():
            copied = provenance_dir / f"{label}_{path.name}"
            shutil.copyfile(path, copied)
        hash_rows.append({"label": label, "path": str(path), "exists": exists, "sha256": sha, "copied_to": str(copied) if copied else None})
    runner = import_current_runner()
    prior_hash = prior_current_runner_hash()
    current_hash = file_sha256(CURRENT_RUNNER_PATH)
    identity_gate = "PASS" if prior_hash and current_hash == prior_hash and manifest_audit["PROSPECTIVE_MANIFEST_REUSE_GATE"] == "PASS" else "FAIL"
    env = {
        "TASK": TASK,
        "time": now(),
        "python": sys.version,
        "executable": sys.executable,
        "git_head": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
        "torch_transformers": sh([sys.executable, "-c", "import torch, transformers; print(torch.__version__, transformers.__version__)"]),
        "fla": sh([sys.executable, "-c", "import fla; print(getattr(fla, '__version__', 'UNKNOWN'))"]),
        "triton": sh([sys.executable, "-c", "import triton; print(triton.__version__)"]),
        "cuda_driver_gpu": sh(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used", "--format=csv,noheader"]),
        "model_path": str(runner.P.MODEL_PATH),
        "model_config_sha256": model_config_hash(runner),
        "relevant_environment": {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith(("CUDA", "LING_", "PYTORCH", "TRANSFORMERS", "HF_"))},
    }
    audit = {
        "TASK": TASK,
        "FORMAL_STATUS": "PIPELINE_IDENTITY_AUDITED_AGAINST_COMPLETED_G64",
        "PIPELINE_IDENTITY_WITH_G64_GATE": identity_gate,
        "current_v2_runner_sha256": current_hash,
        "g64_current_v2_runner_sha256": prior_hash,
        "same_current_v2_runner_as_g64": current_hash == prior_hash,
        "CONFIGS": CONFIGS,
        "GROUP_SIZES": GROUP_SIZES,
        "HISTORY_LENGTHS": HISTORY_LENGTHS,
        "HORIZON": HORIZON,
        "quantizer_semantics": "same current runner fake_quant_ling_state_grouped; R groups value axis V, C groups key axis K",
        "history_semantics": "same persistent recurrent protocol; quantize t0-L+1 ... t0 only",
        "KL_metric_definition": "mean future KL AUC over horizon=128; delta=KL_Cg-KL_Rg",
        "files": hash_rows,
    }
    write_csv(RUN_DIR / "pipeline_hashes.csv", hash_rows)
    save_json(RUN_DIR / "environment.json", env)
    save_json(RUN_DIR / "pipeline_identity_audit.json", audit)
    return audit


def prepare():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_g64_manifest()
    manifest_audit, primary = freeze_primary_subset_from_g64_manifest(manifest)
    save_json(RUN_DIR / "primary_36_unit_manifest.json", primary)
    csv_units = [{k: v for k, v in unit.items() if k != "teacher_forced_token_ids"} for unit in primary["units"]]
    write_csv(RUN_DIR / "primary_36_unit_manifest.csv", csv_units)
    pipeline = write_pipeline_identity_files(manifest_audit)
    decisions = {
        "TASK": TASK,
        "FORMAL_STATUS": "INITIAL_NO_EXPANSIONS_BEFORE_PRIMARY_OUTCOMES",
        "decision_rule": "expand only any g in {128,32,16} whose primary 36-unit 95% bootstrap CI crosses zero",
        "G128_EXPANDED_TO_72": "PENDING_PRIMARY",
        "G32_EXPANDED_TO_72": "PENDING_PRIMARY",
        "G16_EXPANDED_TO_72": "PENDING_PRIMARY",
        "N_ADDITIONAL_EXPANSION_RUNS": 0,
        "decided_before_expansion_outcomes": True,
    }
    save_json(RUN_DIR / "adaptive_expansion_decisions.json", decisions)
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "PREPARED_PRIMARY_SUBSET_AND_PIPELINE_IDENTITY",
        **manifest_audit,
        "PIPELINE_IDENTITY_WITH_G64_GATE": pipeline["PIPELINE_IDENTITY_WITH_G64_GATE"],
        "N_PRIMARY_UNITS": len(primary["units"]),
        "N_EXPECTED_PRIMARY_RUNS": expected_primary_runs(),
        "output_dir": str(RUN_DIR),
    }
    save_json(RUN_DIR / "prepare_summary.json", summary)
    return summary


def load_primary_manifest():
    path = RUN_DIR / "primary_36_unit_manifest.json"
    if not path.exists():
        prepare()
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_units():
    return [normalize_unit(u, i) for i, u in enumerate(load_g64_manifest()["units"])]


def synthetic_quantizer_repeatability_rows():
    runner = import_current_runner()
    import torch
    state = torch.linspace(-3.0, 3.0, steps=128 * 128, dtype=torch.float32).reshape(1, 1, 128, 128)
    rows = []
    for config in CONFIGS:
        hashes = []
        for repeat in range(3):
            qdq, meta = runner.fake_quant_ling_state_grouped(state, config)
            row = {
                "check_type": "synthetic_quantizer",
                "repeat": repeat,
                "unit_id": "SYNTHETIC",
                "history_length": 0,
                "config": config,
                "KL": None,
                "scale_sha256": hashlib.sha256(meta["scale"].cpu().numpy().tobytes()).hexdigest(),
                "int8_code_sha256": hashlib.sha256(meta["codes"].cpu().numpy().tobytes()).hexdigest(),
                "dequantized_state_sha256": hashlib.sha256(qdq.cpu().numpy().tobytes()).hexdigest(),
                "status": "PASS",
            }
            rows.append(row)
            hashes.append((row["scale_sha256"], row["int8_code_sha256"], row["dequantized_state_sha256"]))
        stable = len(set(hashes)) == 1
        for row in rows:
            if row["config"] == config:
                row["status"] = "PASS" if stable else "FAIL"
    return rows


def self_check(repeats=2):
    prep = prepare()
    if prep.get("PIPELINE_IDENTITY_WITH_G64_GATE") != "PASS":
        raise RuntimeError("pipeline identity with g64 did not PASS; refusing self-check")
    manifest = load_primary_manifest()
    sentinel_units = manifest["units"][:3]
    manifest_sha = manifest["source_manifest_sha256"]
    all_rows = synthetic_quantizer_repeatability_rows()
    for repeat in range(int(repeats)):
        repeat_dir = RUN_DIR / f"self_check_repeat{repeat}"
        runner = import_current_runner(repeat_dir, task=TASK, manifest_sha=manifest_sha)
        runner.run_units("formal", sentinel_units, history_lengths=[1], configs=CONFIGS, horizon=HORIZON, unit_limit=None)
        for row in read_csv(repeat_dir / "unit_summary_formal.csv"):
            all_rows.append({
                "check_type": "sentinel_current_runner",
                "repeat": repeat,
                "unit_id": row.get("unit_id"),
                "history_length": int(float(row.get("history_length"))),
                "config": row.get("config"),
                "KL": as_float(row.get("future_kl_auc")),
                "source_norm": as_float(row.get("source_norm")),
                "source_mse": as_float(row.get("source_mse")),
                "state_error_h0": as_float(row.get("state_error_h0")),
                "CURRENT_LOGIT_NONRETROACTIVITY_GATE": row.get("CURRENT_LOGIT_NONRETROACTIVITY_GATE"),
                "NO_FURTHER_QUANTIZATION_GATE": row.get("NO_FURTHER_QUANTIZATION_GATE"),
                "status": "PASS",
            })
    by_key = defaultdict(list)
    for row in all_rows:
        if row["check_type"] == "sentinel_current_runner":
            by_key[(row["unit_id"], row["history_length"], row["config"])].append(row)
    failures = []
    for key, rows in by_key.items():
        kls = [r["KL"] for r in rows if r["KL"] is not None]
        if len(rows) != int(repeats) or not kls or max(kls) - min(kls) > KL_ATOL:
            failures.append({"key": list(key), "KL_values": kls})
            for row in rows:
                row["status"] = "FAIL"
    synthetic_fail = [r for r in all_rows if r["check_type"] == "synthetic_quantizer" and r["status"] != "PASS"]
    gate = "PASS" if not failures and not synthetic_fail else "FAIL"
    write_csv(RUN_DIR / "v2_self_check.csv", all_rows)
    report = [
        f"# {TASK} Pipeline Self-Check",
        "",
        f"- PIPELINE_REPEATABILITY_GATE: {gate}",
        f"- sentinel_units: {[u['unit_id'] for u in sentinel_units]}",
        f"- configs: {CONFIGS}",
        f"- repeats: {repeats}",
        f"- synthetic INT8 scale/code/dequant repeatability failures: {len(synthetic_fail)}",
        f"- sentinel KL repeatability failures: {len(failures)}",
        "",
        "This check uses fixed units from the reused g64 prospective population before formal g128/g32/g16 outcomes.",
    ]
    write_text(RUN_DIR / "v2_self_check_report.md", "\n".join(report) + "\n")
    out = {
        "TASK": TASK,
        "FORMAL_STATUS": "SELF_CHECK_PASS_CAN_RUN_PRIMARY" if gate == "PASS" else "INVALID_PIPELINE_SELF_CHECK_FAILURE",
        "PIPELINE_REPEATABILITY_GATE": gate,
        "sentinel_units": [u["unit_id"] for u in sentinel_units],
        "configs": CONFIGS,
        "repeats": int(repeats),
        "failures": failures[:20],
    }
    save_json(RUN_DIR / "self_check_summary.json", out)
    return out


def select_shard_units(units, shard_id, num_shards):
    return [unit for index, unit in enumerate(units) if index % int(num_shards) == int(shard_id)]


def select_units_by_ids(units, unit_ids):
    wanted = {str(x) for x in unit_ids if str(x)}
    return [unit for unit in units if str(unit["unit_id"]) in wanted]


def safe_label(text):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))


def formal_primary(args):
    check_path = RUN_DIR / "self_check_summary.json"
    if not check_path.exists():
        raise RuntimeError("run --stage self-check before formal-primary")
    check = json.loads(check_path.read_text(encoding="utf-8"))
    if check.get("PIPELINE_REPEATABILITY_GATE") != "PASS":
        raise RuntimeError("self-check did not PASS; refusing primary execution")
    manifest = load_primary_manifest()
    if args.unit_ids:
        requested = [x.strip() for x in args.unit_ids.split(",") if x.strip()]
        units = select_units_by_ids(manifest["units"], requested)
        missing = sorted(set(requested) - {u["unit_id"] for u in units})
        if missing:
            raise RuntimeError(f"requested unit ids are not in primary manifest: {missing}")
        label = safe_label(args.run_label or "_".join(requested))
        shard_dir = RUN_DIR / f"primary_units_{label}"
    else:
        units = select_shard_units(manifest["units"], args.shard_id, args.num_shards)
        shard_dir = RUN_DIR / f"primary_shard{args.shard_id}_of_{args.num_shards}"
    runner = import_current_runner(shard_dir, task=TASK, manifest_sha=manifest["source_manifest_sha256"])
    out = runner.run_units("formal", units, history_lengths=HISTORY_LENGTHS, configs=CONFIGS, horizon=HORIZON, unit_limit=None)
    out.update({
        "TASK": TASK,
        "FORMAL_STATUS": "PRIMARY_SHARD_COMPLETE",
        "REQUESTED_SHARD_ID": args.shard_id,
        "REQUESTED_NUM_SHARDS": args.num_shards,
        "UNIT_IDS": [u["unit_id"] for u in units],
        "SHARD_DIR": str(shard_dir),
    })
    save_json(RUN_DIR / f"primary_shard{args.shard_id}_summary.json", out)
    return out


def merge_shards(prefix, shard_dirs, output_names):
    for stem, filename in output_names:
        merged = []
        for shard in shard_dirs:
            merged.extend(read_csv(Path(shard) / filename))
        if merged:
            write_csv(RUN_DIR / f"{prefix}_{stem}.csv", merged)


def merge_primary(args):
    if args.shard_dirs:
        shard_dirs = [Path(x) for x in args.shard_dirs.split(",") if x]
    else:
        shard_dirs = sorted(RUN_DIR.glob("primary_shard*_of_*"))
    rows = []
    for shard in shard_dirs:
        path = shard / "results_formal.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing primary shard results: {path}")
        rows.extend(list(iter_jsonl(path) or []))
    dedup = {}
    for row in rows:
        key = (str(row.get("unit_id")), row_config(row), int(float(row.get("history_length", -1))))
        dedup[key] = row
    rows = [dedup[key] for key in sorted(dedup)]
    write_jsonl(RUN_DIR / "results_primary.jsonl", rows)
    write_jsonl(RUN_DIR / "results.jsonl", rows)
    merge_shards("primary", shard_dirs, [
        ("unit_summary", "unit_summary_formal.csv"),
        ("horizon", "horizon_formal.csv"),
        ("source_error", "source_error_formal.csv"),
        ("state_error_trajectory", "state_error_trajectory_formal.csv"),
    ])
    out = {
        "TASK": TASK,
        "FORMAL_STATUS": "MERGED_PRIMARY_SHARDS",
        "MERGED_SHARD_COUNT": len(shard_dirs),
        "N_SUCCESSFUL_PRIMARY_RUNS": len(rows),
        "N_EXPECTED_PRIMARY_RUNS": expected_primary_runs(),
    }
    save_json(RUN_DIR / "merge_primary_summary.json", out)
    return out


def formal_expansion(args):
    decisions_path = RUN_DIR / "adaptive_expansion_decisions.json"
    if not decisions_path.exists():
        raise RuntimeError("run analyze-primary before formal-expansion")
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    expanded = [int(k[1:].split("_")[0]) for k, v in decisions.items() if k.startswith("G") and k.endswith("_EXPANDED_TO_72") and v == "YES"]
    if args.group_size:
        requested = int(args.group_size)
        if requested not in expanded:
            raise RuntimeError(f"g{requested} was not precommitted for expansion")
        expanded = [requested]
    if not expanded:
        return {"TASK": TASK, "FORMAL_STATUS": "NO_EXPANSIONS_PRECOMMITTED", "expanded_groups": []}
    primary_ids = {u["unit_id"] for u in load_primary_manifest()["units"]}
    remaining = [u for u in load_all_units() if u["unit_id"] not in primary_ids]
    units = select_shard_units(remaining, args.shard_id, args.num_shards)
    configs = []
    for group in expanded:
        configs.extend([f"R{group}", f"C{group}"])
    shard_dir = RUN_DIR / f"expansion_g{'_'.join(str(g) for g in expanded)}_shard{args.shard_id}_of_{args.num_shards}"
    runner = import_current_runner(shard_dir, task=TASK, manifest_sha=load_g64_manifest()["sha256"])
    out = runner.run_units("formal", units, history_lengths=HISTORY_LENGTHS, configs=configs, horizon=HORIZON, unit_limit=None)
    out.update({
        "TASK": TASK,
        "FORMAL_STATUS": "EXPANSION_SHARD_COMPLETE",
        "expanded_groups": expanded,
        "configs": configs,
        "UNIT_IDS": [u["unit_id"] for u in units],
        "SHARD_DIR": str(shard_dir),
    })
    save_json(RUN_DIR / f"expansion_shard{args.shard_id}_summary.json", out)
    return out


def merge_expansion(args):
    decisions = json.loads((RUN_DIR / "adaptive_expansion_decisions.json").read_text(encoding="utf-8"))
    expanded = [int(k[1:].split("_")[0]) for k, v in decisions.items() if k.startswith("G") and k.endswith("_EXPANDED_TO_72") and v == "YES"]
    if not expanded:
        primary_rows = list(iter_jsonl(RUN_DIR / "results_primary.jsonl") or [])
        write_jsonl(RUN_DIR / "results.jsonl", primary_rows)
        out = {"TASK": TASK, "FORMAL_STATUS": "NO_EXPANSIONS_TO_MERGE", "N_EXPANSION_ROWS": 0, "N_TOTAL_SUCCESSFUL_QUANTIZED_RUNS": len(primary_rows)}
        save_json(RUN_DIR / "merge_expansion_summary.json", out)
        return out
    if args.shard_dirs:
        shard_dirs = [Path(x) for x in args.shard_dirs.split(",") if x]
    else:
        shard_dirs = sorted(RUN_DIR.glob("expansion_g*_shard*_of_*"))
    expansion_rows = []
    for shard in shard_dirs:
        path = shard / "results_formal.jsonl"
        if path.exists():
            expansion_rows.extend(list(iter_jsonl(path) or []))
    primary_rows = list(iter_jsonl(RUN_DIR / "results_primary.jsonl") or [])
    write_jsonl(RUN_DIR / "results_expansion.jsonl", expansion_rows)
    write_jsonl(RUN_DIR / "results.jsonl", primary_rows + expansion_rows)
    out = {
        "TASK": TASK,
        "FORMAL_STATUS": "MERGED_EXPANSION_SHARDS",
        "expanded_groups": expanded,
        "N_EXPANSION_ROWS": len(expansion_rows),
        "N_TOTAL_SUCCESSFUL_QUANTIZED_RUNS": len(primary_rows) + len(expansion_rows),
    }
    save_json(RUN_DIR / "merge_expansion_summary.json", out)
    return out


def analyze_granularity_rows(rows, bootstrap_reps=BOOTSTRAP_REPS, bootstrap_seed=BOOTSTRAP_SEED):
    norm = []
    for row in rows:
        cfg = row_config(row)
        if cfg not in CONFIGS:
            continue
        out = dict(row)
        out["config"] = cfg
        out["group_size"] = int(float(out.get("group_size", cfg[1:])))
        out["unit_id"] = str(out["unit_id"])
        out["problem_id"] = str(out.get("problem_id", out.get("prompt_id", out["unit_id"].split("|")[0])))
        out["history_length"] = int(float(out["history_length"]))
        out["future_kl_auc"] = as_float(out.get("future_kl_auc"))
        out["layer"] = str(out.get("layer", "ALL_KDA"))
        out["head"] = str(out.get("head", "ALL"))
        norm.append(out)
    by = {(r["unit_id"], r["group_size"], r["history_length"], r["config"]): r for r in norm}
    unit_meta = {r["unit_id"]: r for r in norm}
    effects = []
    for group in GROUP_SIZES:
        unit_ids = sorted({r["unit_id"] for r in norm if r["group_size"] == group})
        for unit_id in unit_ids:
            for history in HISTORY_LENGTHS:
                r = by.get((unit_id, group, history, f"R{group}"))
                c = by.get((unit_id, group, history, f"C{group}"))
                if not r or not c or r["future_kl_auc"] is None or c["future_kl_auc"] is None:
                    continue
                delta = c["future_kl_auc"] - r["future_kl_auc"]
                meta = unit_meta[unit_id]
                effects.append({
                    "unit_id": unit_id,
                    "problem_id": meta.get("problem_id"),
                    "layer": meta.get("layer", "ALL_KDA"),
                    "head": meta.get("head", "ALL"),
                    "group_size": group,
                    "history_length": history,
                    "KL_R": r["future_kl_auc"],
                    "KL_C": c["future_kl_auc"],
                    "delta_C_minus_R": delta,
                    "preferred_orientation": "R" if delta > NEAR_TIE else "C" if delta < -NEAR_TIE else "TIE",
                })
    history_summary = []
    group_summary = {}
    bootstrap = {
        "BOOTSTRAP_RESAMPLING_LEVEL": "canonical_unit",
        "BOOTSTRAP_REPS": int(bootstrap_reps),
        "BOOTSTRAP_SEED": int(bootstrap_seed),
        "groups": {},
    }
    robustness_rows = []
    robustness_summary = {}
    heterogeneity = []
    for group in GROUP_SIZES:
        group_effects = [e for e in effects if e["group_size"] == group]
        by_history = defaultdict(list)
        by_unit = defaultdict(list)
        for row in group_effects:
            by_history[int(row["history_length"])].append(row)
            by_unit[row["unit_id"]].append(row)
        for history in HISTORY_LENGTHS:
            rows_l = by_history.get(history, [])
            deltas = [r["delta_C_minus_R"] for r in rows_l]
            history_summary.append({
                "group_size": group,
                "history_length": history,
                "median_KL_R": median([r["KL_R"] for r in rows_l]),
                "median_KL_C": median([r["KL_C"] for r in rows_l]),
                "median_delta": median(deltas),
                "mean_delta": mean(deltas),
                "trimmed_mean_delta": trimmed_mean(deltas),
                "R_preferred_unit_count": sum(1 for x in deltas if x > NEAR_TIE),
                "C_preferred_unit_count": sum(1 for x in deltas if x < -NEAR_TIE),
                "tie_count": sum(1 for x in deltas if abs(x) <= NEAR_TIE),
                "paired_bootstrap_CI_low": (bootstrap_ci(deltas, bootstrap_reps, bootstrap_seed + group + history, median) or [None, None])[0],
                "paired_bootstrap_CI_high": (bootstrap_ci(deltas, bootstrap_reps, bootstrap_seed + group + history, median) or [None, None])[1],
                "n_units": len(deltas),
            })
        unit_summaries = []
        for unit_id, unit_rows in sorted(by_unit.items()):
            if len(unit_rows) != len(HISTORY_LENGTHS):
                continue
            vals = [r["delta_C_minus_R"] for r in sorted(unit_rows, key=lambda x: x["history_length"])]
            meta = unit_rows[0]
            unit_summaries.append({
                "unit_id": unit_id,
                "problem_id": meta["problem_id"],
                "layer": meta["layer"],
                "head": meta["head"],
                "group_size": group,
                "unit_mean_delta": mean(vals),
                "unit_median_delta": median(vals),
                "R_preferred_histories": sum(1 for x in vals if x > NEAR_TIE),
                "C_preferred_histories": sum(1 for x in vals if x < -NEAR_TIE),
            })
        unit_vals = [r["unit_mean_delta"] for r in unit_summaries]
        ci = bootstrap_ci(unit_vals, bootstrap_reps, bootstrap_seed + group, median)
        r_hist = sum(1 for row in history_summary if row["group_size"] == group and as_float(row.get("median_delta")) is not None and row["median_delta"] > 0)
        c_hist = sum(1 for row in history_summary if row["group_size"] == group and as_float(row.get("median_delta")) is not None and row["median_delta"] < 0)
        robust_rows, robust = robustness_analysis(unit_summaries, group_effects, median(unit_vals), group)
        robustness_rows.extend(robust_rows)
        robustness_summary[str(group)] = robust
        group_summary[str(group)] = {
            "GROUP_SIZE": group,
            "PRIMARY_EFFECT_DEFINITION": "median over units of mean_L(KL_Cg - KL_Rg)",
            "N_UNITS": len(unit_summaries),
            "PRIMARY_EFFECT": median(unit_vals),
            "PRIMARY_MEAN": mean(unit_vals),
            "PRIMARY_TRIMMED_MEAN": trimmed_mean(unit_vals),
            "BOOTSTRAP_CI": ci,
            "R_PREFERRED_UNIT_COUNT": sum(1 for x in unit_vals if x > NEAR_TIE),
            "C_PREFERRED_UNIT_COUNT": sum(1 for x in unit_vals if x < -NEAR_TIE),
            "TIE_UNIT_COUNT": sum(1 for x in unit_vals if abs(x) <= NEAR_TIE),
            "SIGN_TEST_P_TWO_SIDED": binomial_p_two_sided(sum(1 for x in unit_vals if x > NEAR_TIE), sum(1 for x in unit_vals if abs(x) > NEAR_TIE)),
            "R_PREFERRED_HISTORY_COUNT": r_hist,
            "C_PREFERRED_HISTORY_COUNT": c_hist,
            **robust,
        }
        bootstrap["groups"][str(group)] = {
            "BOOTSTRAP_CI": ci,
            "unit_effects": unit_summaries,
            "statistic": "median over units of mean_L(KL_Cg - KL_Rg)",
        }
        heterogeneity.extend(heterogeneity_rows(group_effects, unit_summaries, group))
    return {
        "per_unit_per_history_effects": effects,
        "history_level_summary": history_summary,
        "group_level_summary": group_summary,
        "bootstrap_results": bootstrap,
        "robustness_analysis": robustness_rows,
        "robustness_summary": robustness_summary,
        "heterogeneity_analysis": heterogeneity,
    }


def sign_stable(vals, baseline):
    vals = [float(v) for v in vals if finite(v)]
    if not vals or baseline is None:
        return "FAIL"
    if baseline > 0:
        return "PASS" if all(v > 0 for v in vals) else "FAIL"
    if baseline < 0:
        return "PASS" if all(v < 0 for v in vals) else "FAIL"
    return "FAIL"


def robustness_analysis(unit_summaries, effects, baseline, group):
    rows = []
    unit_vals = {r["unit_id"]: r["unit_mean_delta"] for r in unit_summaries}
    for unit_id in sorted(unit_vals):
        vals = [v for u, v in unit_vals.items() if u != unit_id]
        rows.append({"group_size": group, "analysis": "leave_one_unit_out", "omitted": unit_id, "effect": median(vals), "n_units": len(vals)})
    for key in ["problem_id", "layer", "head"]:
        groups = sorted({str(r[key]) for r in unit_summaries})
        for subgroup in groups:
            vals = [r["unit_mean_delta"] for r in unit_summaries if str(r[key]) != subgroup]
            rows.append({"group_size": group, "analysis": f"leave_one_{key}_out", "omitted": subgroup, "effect": median(vals), "n_units": len(vals)})
    by_history = defaultdict(list)
    for row in effects:
        by_history[row["history_length"]].append(row["delta_C_minus_R"])
    for history in HISTORY_LENGTHS:
        rows.append({"group_size": group, "analysis": "history_effect", "omitted": "", "history_length": history, "effect": median(by_history[history]), "n_units": len(by_history[history])})
    unit_effects = [r["effect"] for r in rows if r["analysis"] == "leave_one_unit_out"]
    prompt_effects = [r["effect"] for r in rows if r["analysis"] == "leave_one_problem_id_out" and r["n_units"]]
    layer_effects = [r["effect"] for r in rows if r["analysis"] == "leave_one_layer_out" and r["n_units"]]
    head_effects = [r["effect"] for r in rows if r["analysis"] == "leave_one_head_out" and r["n_units"]]
    prompt_stable = sign_stable(prompt_effects, baseline) if prompt_effects else "PASS"
    layer_stable = sign_stable(layer_effects, baseline) if layer_effects else "PASS"
    head_stable = sign_stable(head_effects, baseline) if head_effects else "PASS"
    summary = {
        "LEAVE_ONE_UNIT_OUT_SIGN_STABLE": sign_stable(unit_effects, baseline),
        "LEAVE_ONE_PROMPT_OUT_SIGN_STABLE": prompt_stable,
        "LEAVE_ONE_LAYER_OUT_SIGN_STABLE": layer_stable,
        "LEAVE_ONE_HEAD_OUT_SIGN_STABLE": head_stable,
        "NO_SINGLE_SUBGROUP_DRIVES_RESULT": "PASS" if prompt_stable == "PASS" and layer_stable == "PASS" and head_stable == "PASS" else "FAIL",
    }
    return rows, summary


def heterogeneity_rows(effects, unit_summaries, group):
    rows = []
    for key, label in [("history_length", "HISTORY_CONDITIONED"), ("problem_id", "SAMPLE_CONDITIONED"), ("layer", "LAYER_CONDITIONED"), ("head", "HEAD_CONDITIONED")]:
        groups = defaultdict(list)
        source = effects if key == "history_length" else unit_summaries
        value_key = "delta_C_minus_R" if key == "history_length" else "unit_mean_delta"
        for row in source:
            groups[str(row[key])].append(row[value_key])
        for subgroup, vals in sorted(groups.items()):
            rows.append({
                "group_size": group,
                "heterogeneity_axis": key,
                "condition_label": label,
                "group": subgroup,
                "n": len(vals),
                "median_delta": median(vals),
                "mean_delta": mean(vals),
                "trimmed_mean_delta": trimmed_mean(vals),
                "R_preferred_count": sum(1 for x in vals if x > NEAR_TIE),
                "C_preferred_count": sum(1 for x in vals if x < -NEAR_TIE),
            })
    return rows


def decide_adaptive_expansions(group_summary):
    decisions = {
        "TASK": TASK,
        "FORMAL_STATUS": "ADAPTIVE_EXPANSION_DECISION_FROZEN_AFTER_PRIMARY_BEFORE_EXPANSION_OUTCOMES",
        "decision_rule": "expand only groups whose primary 36-unit 95% bootstrap CI crosses zero",
    }
    groups = []
    for group in ["128", "32", "16"]:
        ci = group_summary.get(group, {}).get("BOOTSTRAP_CI") or [None, None]
        crosses = ci[0] is None or ci[1] is None or (ci[0] <= 0 <= ci[1])
        decisions[f"G{group}_EXPANDED_TO_72"] = "YES" if crosses else "NO"
        if crosses:
            groups.append(group)
    decisions["EXPANDED_GROUPS"] = groups
    decisions["N_ADDITIONAL_EXPANSION_RUNS"] = expected_expansion_runs(groups)
    decisions["decided_before_expansion_outcomes"] = True
    return decisions


def classify_group(group_stats, gates):
    if any(gates.get(key) != "PASS" for key in BLOCKING_GATES):
        return "INVALID"
    effect = as_float(group_stats.get("PRIMARY_EFFECT"))
    ci = group_stats.get("BOOTSTRAP_CI") or [None, None]
    robust_ok = (
        group_stats.get("LEAVE_ONE_UNIT_OUT_SIGN_STABLE") == "PASS"
        and group_stats.get("LEAVE_ONE_LAYER_OUT_SIGN_STABLE") == "PASS"
        and group_stats.get("LEAVE_ONE_PROMPT_OUT_SIGN_STABLE") == "PASS"
        and group_stats.get("NO_SINGLE_SUBGROUP_DRIVES_RESULT") == "PASS"
    )
    if effect is not None and effect > 0 and ci[0] is not None and ci[0] > 0 and group_stats.get("R_PREFERRED_HISTORY_COUNT", 0) >= 5 and robust_ok:
        return "ROBUST_R_PREFERENCE"
    if effect is not None and effect < 0 and ci[1] is not None and ci[1] < 0 and group_stats.get("C_PREFERRED_HISTORY_COUNT", 0) >= 4 and robust_ok:
        return "ROBUST_C_PREFERENCE"
    return "ORIENTATION_BOUNDARY"


def classify_cross_granularity(classifications):
    ordered = ["128", "64", "32", "16"]
    if any(classifications.get(g) == "INVALID" for g in ordered):
        result = "INVALID"
        pattern = "INVALID"
    elif all(classifications.get(g) == "ROBUST_R_PREFERENCE" for g in ordered):
        result = "ROBUST_R_ALL_TESTED_GRANULARITIES"
        pattern = "R/R/R/R"
    elif any(classifications.get(g) == "ROBUST_C_PREFERENCE" for g in ordered):
        result = "GRANULARITY_DEPENDENT_ORIENTATION_CHANGE"
        pattern = "/".join("C" if classifications.get(g) == "ROBUST_C_PREFERENCE" else "R" if classifications.get(g) == "ROBUST_R_PREFERENCE" else "B" for g in ordered)
    else:
        result = "PARTIALLY_STABLE_WITH_BOUNDARY"
        pattern = "/".join("R" if classifications.get(g) == "ROBUST_R_PREFERENCE" else "B" for g in ordered)
    return {
        "LING_KDA_PROSPECTIVE_GRANULARITY_PATTERN": pattern,
        "LING_KDA_ORIENTATION_GRANULARITY_RESULT": result,
        "classifications_by_group_size": classifications,
    }


def gate_summary_from_outputs(n_rows):
    gates = pass_gate_summary()
    prep = json.loads((RUN_DIR / "prepare_summary.json").read_text(encoding="utf-8")) if (RUN_DIR / "prepare_summary.json").exists() else {}
    pipeline = json.loads((RUN_DIR / "pipeline_identity_audit.json").read_text(encoding="utf-8")) if (RUN_DIR / "pipeline_identity_audit.json").exists() else {}
    selfcheck = json.loads((RUN_DIR / "self_check_summary.json").read_text(encoding="utf-8")) if (RUN_DIR / "self_check_summary.json").exists() else {}
    gates["PROSPECTIVE_MANIFEST_REUSE_GATE"] = prep.get("PROSPECTIVE_MANIFEST_REUSE_GATE", "FAIL")
    gates["NO_OUTCOME_BASED_SUBSET_SELECTION_GATE"] = prep.get("NO_OUTCOME_BASED_SUBSET_SELECTION_GATE", "FAIL")
    gates["PIPELINE_IDENTITY_WITH_G64_GATE"] = pipeline.get("PIPELINE_IDENTITY_WITH_G64_GATE", "FAIL")
    gates["PIPELINE_REPEATABILITY_GATE"] = selfcheck.get("PIPELINE_REPEATABILITY_GATE", "FAIL")
    if n_rows < expected_primary_runs():
        gates["PERSISTENT_RECURRENT_PROTOCOL_GATE"] = "FAIL"
    return gates


def analyze_primary(args):
    rows = list(iter_jsonl(RUN_DIR / "results_primary.jsonl") or iter_jsonl(RUN_DIR / "results.jsonl") or [])
    analysis = analyze_granularity_rows(rows, bootstrap_reps=args.bootstrap_reps, bootstrap_seed=BOOTSTRAP_SEED)
    decisions = decide_adaptive_expansions(analysis["group_level_summary"])
    save_json(RUN_DIR / "adaptive_expansion_decisions.json", decisions)
    save_json(RUN_DIR / "primary_analysis_preview.json", {"group_level_summary": analysis["group_level_summary"], "adaptive_expansion_decisions": decisions})
    return {"TASK": TASK, "FORMAL_STATUS": "PRIMARY_ANALYZED_ADAPTIVE_DECISIONS_FROZEN", **decisions}


def final_analyze(args):
    rows = list(iter_jsonl(RUN_DIR / "results.jsonl") or [])
    analysis = analyze_granularity_rows(rows, bootstrap_reps=args.bootstrap_reps, bootstrap_seed=BOOTSTRAP_SEED)
    write_csv(RUN_DIR / "per_unit_per_history_effects.csv", analysis["per_unit_per_history_effects"])
    write_csv(RUN_DIR / "history_level_summary.csv", analysis["history_level_summary"])
    write_csv(RUN_DIR / "group_level_summary.csv", analysis["group_level_summary"].values())
    save_json(RUN_DIR / "bootstrap_results.json", analysis["bootstrap_results"])
    write_csv(RUN_DIR / "robustness_analysis.csv", analysis["robustness_analysis"])
    write_csv(RUN_DIR / "heterogeneity_analysis.csv", analysis["heterogeneity_analysis"])
    gates = gate_summary_from_outputs(len(rows))
    classifications = {str(g): classify_group(analysis["group_level_summary"][str(g)], gates) for g in GROUP_SIZES}
    classifications["64"] = G64_PROSPECTIVE_CLASSIFICATION
    cross = classify_cross_granularity(classifications)
    decisions = json.loads((RUN_DIR / "adaptive_expansion_decisions.json").read_text(encoding="utf-8")) if (RUN_DIR / "adaptive_expansion_decisions.json").exists() else {}
    summary = build_summary(analysis, gates, classifications, cross, decisions, rows, args)
    save_json(RUN_DIR / "summary.json", summary)
    save_json(RESULT_DIR / f"{SLUG}_final_summary.json", summary)
    write_csv(RUN_DIR / "prospective_granularity_summary.csv", [summary])
    write_text(RUN_DIR / "robustness_report.md", robustness_report_text(analysis, classifications))
    write_report(summary, analysis)
    write_readme(summary)
    return summary


def build_summary(analysis, gates, classifications, cross, decisions, rows, args):
    group = analysis["group_level_summary"]
    primary_rows = list(iter_jsonl(RUN_DIR / "results_primary.jsonl") or [])
    expected_total = expected_primary_runs() + int(decisions.get("N_ADDITIONAL_EXPANSION_RUNS", 0) or 0)
    formal_status = "COMPLETE" if len(rows) == expected_total and all(v != "INVALID" for v in classifications.values()) else "INCOMPLETE_OR_INVALID"
    if any(gates.get(key) != "PASS" for key in BLOCKING_GATES):
        formal_status = "INVALID"
    out = {
        "TASK": TASK,
        "FORMAL_STATUS": formal_status,
        **gates,
        "N_REUSED_PROSPECTIVE_UNITS": N_PROSPECTIVE_UNITS,
        "N_PRIMARY_UNITS": N_PRIMARY_UNITS,
        "N_EXPECTED_PRIMARY_RUNS": expected_primary_runs(),
        "N_SUCCESSFUL_PRIMARY_RUNS": len(primary_rows) if primary_rows else min(len(rows), expected_primary_runs()),
        "N_ADDITIONAL_EXPANSION_RUNS": decisions.get("N_ADDITIONAL_EXPANSION_RUNS", 0),
        "N_TOTAL_SUCCESSFUL_QUANTIZED_RUNS": len(rows),
        "PRIMARY_EFFECT_DEFINITION": "median over units of mean_L(KL_Cg - KL_Rg)",
        "G128_PRIMARY_EFFECT": group["128"].get("PRIMARY_EFFECT"),
        "G128_BOOTSTRAP_CI": group["128"].get("BOOTSTRAP_CI"),
        "G128_PROSPECTIVE_CLASSIFICATION": classifications.get("128"),
        "G32_PRIMARY_EFFECT": group["32"].get("PRIMARY_EFFECT"),
        "G32_BOOTSTRAP_CI": group["32"].get("BOOTSTRAP_CI"),
        "G32_PROSPECTIVE_CLASSIFICATION": classifications.get("32"),
        "G16_PRIMARY_EFFECT": group["16"].get("PRIMARY_EFFECT"),
        "G16_BOOTSTRAP_CI": group["16"].get("BOOTSTRAP_CI"),
        "G16_PROSPECTIVE_CLASSIFICATION": classifications.get("16"),
        "G64_PRIMARY_EFFECT": G64_PRIMARY_EFFECT,
        "G64_BOOTSTRAP_CI": G64_BOOTSTRAP_CI,
        "G64_PROSPECTIVE_CLASSIFICATION": G64_PROSPECTIVE_CLASSIFICATION,
        **cross,
        **{k: decisions.get(k) for k in ["G128_EXPANDED_TO_72", "G32_EXPANDED_TO_72", "G16_EXPANDED_TO_72"]},
        "FRESH_OUTPUT_VERIFICATION": "NOT_RUN",
        "PY_COMPILE": args.py_compile_status,
        "PYTEST": args.pytest_status,
        "GPU_RELEASE_VERIFICATION": args.gpu_release_status,
        "METHOD_DESIGN_READY": "NO",
        "NO_NEXT_EXPERIMENT_LAUNCHED": "YES",
        "CURRENT_TASK_COMPLETED_AND_STOPPED": "NO",
        "NEXT_RECOMMENDED_TASK": "WAIT_FOR_HUMAN_REVIEW",
        "output_dir": str(RUN_DIR),
    }
    return out


def robustness_report_text(analysis, classifications):
    lines = [f"# {TASK} Robustness Report", ""]
    for group in GROUP_SIZES:
        robust = analysis["robustness_summary"].get(str(group), {})
        lines += [
            f"## g{group}",
            f"- classification: {classifications.get(str(group))}",
            f"- LEAVE_ONE_UNIT_OUT_SIGN_STABLE: {robust.get('LEAVE_ONE_UNIT_OUT_SIGN_STABLE')}",
            f"- LEAVE_ONE_PROMPT_OUT_SIGN_STABLE: {robust.get('LEAVE_ONE_PROMPT_OUT_SIGN_STABLE')}",
            f"- LEAVE_ONE_LAYER_OUT_SIGN_STABLE: {robust.get('LEAVE_ONE_LAYER_OUT_SIGN_STABLE')}",
            f"- NO_SINGLE_SUBGROUP_DRIVES_RESULT: {robust.get('NO_SINGLE_SUBGROUP_DRIVES_RESULT')}",
            "",
        ]
    lines.append("No mechanism or method-design experiment is launched by this task.")
    return "\n".join(lines) + "\n"


def write_report(summary, analysis):
    lines = [
        f"# {TASK}",
        "",
        "## Question",
        "Are g128, g32, and g16 prospectively Row preferred on the same frozen 72-unit population as the completed g64 confirmation?",
        "",
        "## Result",
        f"- FORMAL_STATUS: {summary.get('FORMAL_STATUS')}",
        f"- pattern: {summary.get('LING_KDA_PROSPECTIVE_GRANULARITY_PATTERN')}",
        f"- cross-granularity result: {summary.get('LING_KDA_ORIENTATION_GRANULARITY_RESULT')}",
        "",
        "## Group Summary",
    ]
    for group in GROUP_SIZES:
        gs = analysis["group_level_summary"][str(group)]
        lines.append(f"- g{group}: effect={gs.get('PRIMARY_EFFECT')} CI={gs.get('BOOTSTRAP_CI')} classification={summary.get(f'G{group}_PROSPECTIVE_CLASSIFICATION')}")
    lines += [
        f"- g64 prior prospective effect: {G64_PRIMARY_EFFECT}",
        f"- g64 prior prospective CI: {G64_BOOTSTRAP_CI}",
        "",
        "## Gates",
    ]
    for key in BLOCKING_GATES:
        lines.append(f"- {key}: {summary.get(key)}")
    lines += [
        "",
        "## Discipline",
        "The completed g64 outputs are reused without rerunning g64. No mechanism, transfer, mixed precision, or method-design experiment is launched.",
    ]
    text = "\n".join(lines) + "\n"
    write_text(RUN_DIR / "report.md", text)
    write_text(REPORT_DIR / f"{SLUG}.md", text)


def write_readme(summary):
    write_text(RUN_DIR / "README.md", "\n".join([
        f"# {TASK}",
        "",
        f"- output_dir: {RUN_DIR}",
        f"- FORMAL_STATUS: {summary.get('FORMAL_STATUS')}",
        f"- pattern: {summary.get('LING_KDA_PROSPECTIVE_GRANULARITY_PATTERN')}",
        f"- total_results_jsonl_rows: {summary.get('N_TOTAL_SUCCESSFUL_QUANTIZED_RUNS')}",
        "",
        "This directory contains reused-manifest evidence, primary subset freeze files, pipeline identity audit, self-check evidence, raw g128/g32/g16 results, adaptive expansion decisions, and prespecified analyses.",
    ]) + "\n")


def verify_fresh_outputs():
    summary_path = RUN_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    decisions = json.loads((RUN_DIR / "adaptive_expansion_decisions.json").read_text(encoding="utf-8")) if (RUN_DIR / "adaptive_expansion_decisions.json").exists() else {}
    expected_total = expected_primary_runs() + int(decisions.get("N_ADDITIONAL_EXPANSION_RUNS", 0) or 0)
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (RUN_DIR / name).exists()]
    rows = list(iter_jsonl(RUN_DIR / "results.jsonl") or [])
    cells = {(r.get("unit_id"), row_config(r), int(float(r.get("history_length", -1)))) for r in rows if r.get("history_length") is not None}
    mtimes = {name: (RUN_DIR / name).stat().st_mtime for name in REQUIRED_OUTPUT_FILES if (RUN_DIR / name).exists()}
    hashes = {name: file_sha256(RUN_DIR / name) for name in REQUIRED_OUTPUT_FILES if (RUN_DIR / name).exists() and (RUN_DIR / name).is_file()}
    ok = (
        not missing
        and len(rows) == expected_total
        and len(cells) == expected_total
        and summary.get("N_TOTAL_SUCCESSFUL_QUANTIZED_RUNS") == expected_total
        and summary.get("FORMAL_STATUS") == "COMPLETE"
        and summary.get("NO_NEXT_EXPERIMENT_LAUNCHED") == "YES"
    )
    out = {
        "FRESH_OUTPUT_VERIFICATION": "PASS" if ok else "FAIL",
        "missing_files": missing,
        "results_jsonl_rows": len(rows),
        "unique_result_cells": len(cells),
        "expected_results_jsonl_rows": expected_total,
        "file_mtimes": mtimes,
        "file_hashes": hashes,
    }
    save_json(RUN_DIR / "fresh_output_verification.json", out)
    if summary:
        summary.update(out)
        summary["CURRENT_TASK_COMPLETED_AND_STOPPED"] = "YES" if out["FRESH_OUTPUT_VERIFICATION"] == "PASS" else "NO"
        save_json(RUN_DIR / "summary.json", summary)
        save_json(RESULT_DIR / f"{SLUG}_final_summary.json", summary)
    return out


def render_console_summary(summary):
    lines = []
    for key in REQUIRED_CONSOLE_KEYS:
        val = summary.get(key)
        if isinstance(val, (dict, list)):
            val = json.dumps(val, sort_keys=True, ensure_ascii=False)
        lines.append(f"{key} = {'' if val is None else val}")
    lines += ["NO_NEXT_EXPERIMENT_LAUNCHED = YES", "STOP."]
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=TASK)
    parser.add_argument("--stage", choices=[
        "prepare",
        "self-check",
        "formal-primary",
        "merge-primary",
        "analyze-primary",
        "formal-expansion",
        "merge-expansion",
        "analyze",
        "verify",
        "summary",
    ], default="prepare")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-dirs", default="")
    parser.add_argument("--group-size", default="")
    parser.add_argument("--unit-ids", default="")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--self-check-repeats", type=int, default=2)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--console-summary", action="store_true")
    parser.add_argument("--pytest-status", default="NOT_RUN")
    parser.add_argument("--py-compile-status", default="NOT_RUN")
    parser.add_argument("--gpu-release-status", default="NOT_RUN")
    return parser.parse_args()


def main():
    args = parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        out = prepare()
    elif args.stage == "self-check":
        out = self_check(args.self_check_repeats)
    elif args.stage == "formal-primary":
        out = formal_primary(args)
    elif args.stage == "merge-primary":
        out = merge_primary(args)
    elif args.stage == "analyze-primary":
        out = analyze_primary(args)
    elif args.stage == "formal-expansion":
        out = formal_expansion(args)
    elif args.stage == "merge-expansion":
        out = merge_expansion(args)
    elif args.stage == "analyze":
        out = final_analyze(args)
    elif args.stage == "verify":
        out = verify_fresh_outputs()
    elif args.stage == "summary":
        out = json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))
    else:
        raise ValueError(args.stage)
    if args.console_summary:
        print(render_console_summary(out))
    else:
        print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
