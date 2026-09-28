#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(command, env):
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--trace-root", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--trainer", type=Path)
    parser.add_argument("--run-one-update", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    trainer = args.trainer or (
        repo_root / "experiments" / "LING_RECURRENT_DENSE_L6_L7_V1" / "scripts" / "recurrent_dense_train.py"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["LING_MODEL_PATH"] = str(args.model_root.resolve())

    gradient_json = args.output_root / "gradient_feasibility.json"
    run([
        args.python, str(trainer), "--phase", "smoke",
        "--legacy-repo", str(args.runtime_root.resolve()),
        "--trace-dir", str(args.trace_root.resolve()),
        "--output-file", str(gradient_json),
    ], env)
    gradient = json.loads(gradient_json.read_text(encoding="utf-8"))
    required = {
        "GRADIENT_GATE": "PASS",
        "REAL_RECURRENT_WRITEBACK_GATE": "PASS",
        "RECURRENT_STATE_PROVENANCE_GATE": "PASS",
        "TRAIN_QDQ_MATCH": "PASS",
        "STE_FORWARD_EXACT_QDQ": "PASS",
    }
    for key, value in required.items():
        if gradient.get(key) != value:
            raise RuntimeError(f"{key}={gradient.get(key)!r}, expected {value!r}")
    for objective in ("L6_RECURRENT_DENSE_STATE", "L7_RECURRENT_DENSE_FUNCTIONAL"):
        row = gradient["per_objective_gradient"][objective]
        if not row.get("finite") or not row.get("nonzero"):
            raise RuntimeError(f"gradient smoke failed for {objective}: {row}")

    result = {
        "CLEAN_ROOM_L6_GRADIENT_SMOKE_GATE": "PASS",
        "CLEAN_ROOM_L7_GRADIENT_SMOKE_GATE": "PASS",
        "one_update_requested": args.run_one_update,
    }
    if args.run_one_update:
        for short, objective in (
            ("l6", "L6_RECURRENT_DENSE_STATE"),
            ("l7", "L7_RECURRENT_DENSE_FUNCTIONAL"),
        ):
            output = args.output_root / f"{short}_one_update"
            run([
                args.python, str(trainer), "--phase", "train",
                "--legacy-repo", str(args.runtime_root.resolve()),
                "--trace-dir", str(args.trace_root.resolve()),
                "--objective", objective,
                "--gradient-horizon", "128",
                "--steps", "1",
                "--validation-interval", "1",
                "--early-stop-validations", "10",
                "--train-sequence-limit", "1",
                "--validation-sequence-limit", "1",
                "--seed", "0", "--lr", "0.003",
                "--output-dir", str(output),
            ], env)
            summary = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
            if summary.get("status") != "PASS" or summary.get("steps_completed") != 1:
                raise RuntimeError(f"one-update smoke failed for {objective}")
            result[f"CLEAN_ROOM_{short.upper()}_ONE_UPDATE_GATE"] = "PASS"
    (args.output_root / "clean_room_smoke_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
