#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO / "experiments" / "ling" / "run_ling_kda_decay_scalarization_causal_v1.py"


def import_runner():
    spec = importlib.util.spec_from_file_location("ling_decay_scalar_runner", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def convert_row(row):
    out = {}
    for k, v in row.items():
        if v == "":
            out[k] = None
            continue
        try:
            if k in {"problem_index", "t0", "tau", "teacher_token_id"}:
                out[k] = int(float(v))
            elif k in {"C_GREATER_R"}:
                out[k] = str(v).lower() == "true"
            elif k in {"unit_id", "problem_id"}:
                out[k] = v
            else:
                out[k] = float(v)
        except Exception:
            out[k] = v
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def run_shard(args):
    runner = import_runner()
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
    os.environ.setdefault("FLA_FLASH_KDA", "0")
    os.environ.setdefault("FLA_TILELANG", "0")

    stage0 = load_json(runner.RUN_DIR / "stage0_audit.json")
    stage_a = load_json(runner.RUN_DIR / "stageA_operator_panel.json")
    if stage0.get("LAMBDA1_FP_LOGIT_IDENTITY") != "PASS":
        raise SystemExit("Stage0 gate is not PASS")
    if stage_a.get("STAGE_A_GATE") == "OPERATOR_CAUSAL_SIGNAL_NEGATIVE":
        raise SystemExit("StageA gate is negative")

    canonical = runner.import_module(runner.CANONICAL_RUNNER, f"ling_canonical_runner_scalar_shard{args.shard_id}")
    ctl = runner.import_module(runner.CONTROLLED_RUNNER, f"ling_controlled_runner_scalar_shard{args.shard_id}")
    canonical.MODEL_PATH = Path(os.environ.get("LING_MODEL_PATH", "/data/zypan/models/Ling-3.0-tiny"))
    canonical.LEGACY_DATA = Path(os.environ.get("LING_AIME24_DATA", runner.REPO / "runs" / "ling_kda_aime24_int8_rc_end_to_end_smoke_v1" / "aime24_HuggingFaceH4_aime_2024_train.json"))

    units = runner.load_canonical_units()
    shard_units = [unit for i, unit in enumerate(units) if i % args.num_shards == args.shard_id]
    print(f"[{runner.now()}] shard {args.shard_id}/{args.num_shards} units={[u['unit_id'] for u in shard_units]}", flush=True)
    torch, model, tokenizer = canonical.load_model_and_tokenizer()
    scalarizer = runner.DecayScalarizer()
    scalarizer.install(model)
    dataset = {str(r["problem_id"]): r for r in canonical.load_dataset()}
    fp_records = ctl.load_fp_records()
    layers = canonical.kda_layers_from_config(load_json(canonical.MODEL_PATH / "config.json"))

    unit_rows = []
    horizon_rows = []
    fp_by_unit_lambda = {}
    for unit in shard_units:
        for lam in runner.LAMBDAS:
            print(f"[{runner.now()}] shard={args.shard_id} StageB unit={unit['unit_id']} lambda={lam:.2f}", flush=True)
            u, h, fp_logits = runner.run_unit_lambda(canonical, torch, model, tokenizer, dataset, fp_records, scalarizer, layers, unit, lam)
            unit_rows.append(u)
            horizon_rows.extend(h)
            fp_by_unit_lambda[(unit["unit_id"], lam)] = fp_logits

    for row in unit_rows:
        if row["lambda"] == 1.0:
            row["FP_DRIFT_KL_vs_lambda1"] = 0.0
            continue
        ref = fp_by_unit_lambda[(row["unit_id"], 1.0)]
        cur = fp_by_unit_lambda[(row["unit_id"], row["lambda"])]
        vals = []
        for a, b in zip(ref, cur):
            logp = torch.log_softmax(a.float(), dim=-1)
            logq = torch.log_softmax(b.float(), dim=-1)
            p = torch.softmax(a.float(), dim=-1)
            vals.append(float(torch.sum(p * (logp - logq), dim=-1).mean().item()))
        row["FP_DRIFT_KL_vs_lambda1"] = runner.mean(vals)

    write_jsonl(runner.RUN_DIR / f"formal_per_unit_shard{args.shard_id}.jsonl", unit_rows)
    write_csv(runner.RUN_DIR / f"formal_horizon_shard{args.shard_id}.csv", horizon_rows)
    runner.save_json(runner.RUN_DIR / f"formal_shard{args.shard_id}_summary.json", {
        "TASK": runner.TASK,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "units": [u["unit_id"] for u in shard_units],
        "unit_rows": len(unit_rows),
        "horizon_rows": len(horizon_rows),
    })
    print(f"[{runner.now()}] shard={args.shard_id} complete unit_rows={len(unit_rows)} horizon_rows={len(horizon_rows)}", flush=True)


def finalize(args):
    runner = import_runner()
    stage0 = load_json(runner.RUN_DIR / "stage0_audit.json")
    stage_a = load_json(runner.RUN_DIR / "stageA_operator_panel.json")
    unit_rows = []
    horizon_rows = []
    for sid in range(args.num_shards):
        unit_rows.extend(list(iter_jsonl(runner.RUN_DIR / f"formal_per_unit_shard{sid}.jsonl") or []))
        horizon_rows.extend(convert_row(r) for r in read_csv(runner.RUN_DIR / f"formal_horizon_shard{sid}.csv"))
    unit_rows = sorted(unit_rows, key=lambda r: (int(r["problem_index"]), int(r["t0"]), float(r["lambda"])))
    horizon_rows = sorted(horizon_rows, key=lambda r: (int(r["problem_index"]), int(r["t0"]), float(r["lambda"]), int(r["tau"])))
    write_jsonl(runner.RUN_DIR / "formal_per_unit.jsonl", unit_rows)
    write_csv(runner.RUN_DIR / "formal_per_unit.csv", unit_rows)
    write_csv(runner.RUN_DIR / "formal_horizon.csv", horizon_rows)
    summary, stats = runner.aggregate_formal(unit_rows, horizon_rows, stage_a)
    config = load_json(runner.RUN_DIR / "config.json")
    runner.make_report(config, runner.load_canonical_units(), stage0, stage_a, summary, stats, [], [], unit_rows)
    runner.save_json(runner.DOC_DIR / "final_classification.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize(args)
    else:
        if args.shard_id is None:
            raise SystemExit("--shard-id is required unless --finalize is set")
        run_shard(args)


if __name__ == "__main__":
    main()
