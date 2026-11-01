#!/usr/bin/env python3
"""One independent CPU FrozenV4 pass over exactly60 released legal payloads.

The runner never imports a model, allocates CUDA, changes trajectories or
selects candidates using gold. Tokenizer re-decode precedes gold comparison.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Independent CPU scoring requires explicitly empty CVD")
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    import resident_record_validation as validation
    control = importlib.import_module("resident_execution_control")
    control.verify_sources(root)
    control.verify_model_and_external(root, fresh_external=True)
    complete = json.loads((root / "FORMAL_GENERATIONS_COMPLETE.json").read_text())
    terminal = json.loads((root / "SESSION_RESULT.json").read_text())
    validation.require(complete["count"] == 60
            and terminal.get("formal_completed") == 60
            and terminal.get("all_own_processes_released") is True,
            "Independent scoring requires complete60 and actual terminal/release receipts")
    jobs = json.loads((root / "FORMAL_JOB_MANIFEST.json").read_text())["jobs"]
    validation.require(len(jobs) == 60 and len({job["id"] for job in jobs}) == 60,
                       "Formal job manifest is not the unique60")
    gold_rows = [json.loads(line) for line in (root / "protocol/aime26_frozen.jsonl").read_text().splitlines()
                 if line.strip()]
    gold = {row["problem_id"]: row for row in gold_rows}
    validation.require(len(gold_rows) == len(gold) == 30, "Frozen gold roster is not the original30")
    scorer_path = root / "protocol/aime26_scorer_v4.py"
    validation.require(validation.sha(scorer_path) == validation.SCORER_SHA, "Original FrozenV4 bytes differ")
    name = "_resident_exact_original_frozen_v4"
    spec = importlib.util.spec_from_file_location(name, scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    sys.modules[name] = scorer
    spec.loader.exec_module(scorer)
    sys.path.insert(0, "/data/zypan/transformers-qwen35/src")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/data/zypan/modelscope_models/Qwen3.5-9B",
                         trust_remote_code=True, local_files_only=True)
    scores = []
    for job in jobs:
        directory = root / "formal" / job["id"]
        interval = json.loads((directory / "RESOURCE_INTERVAL.json").read_text())
        validation.require(interval["exit_returncode"] == 0 and interval["release"] == "PASS",
                           "Scoring encountered an unreleased or failed process interval")
        wrapper = json.loads((directory / "RESULT.json").read_text())
        row = validation.validate(wrapper, job, root)
        response = tokenizer.decode(row["generated_token_ids"], skip_special_tokens=True)
        validation.require(response == row["response"], "Exact sampledtoken re-decode differs from worker text")
        frozen = gold[job["problem_id"]]
        extraction = scorer.extract_v4(response, row["finish_reason"], frozen["problem"])
        correct = bool(scorer.compare_with_gold(extraction, frozen["answer"]))
        scores.append(dict(sample_id=job["id"], problem_id=job["problem_id"], condition=job["condition"],
                           seed=1, gpu=job["gpu"], gpu_uuid=job["gpu_uuid"],
                           correct=correct, extracted_answer=extraction.normalized_value,
                           extraction=extraction.to_dict(), gold=frozen["answer"],
                           generated_token_count=row["generated_token_count"], finish_reason=row["finish_reason"],
                           wall_seconds=row["wall_seconds"], TPS=row["TPS"],
                           peak_allocated_bytes=row["peak_allocated_bytes"],
                           payload_sha256=wrapper["payload_sha256"]))
    validation.require({(row["condition"], row["problem_id"]) for row in scores}
                       == {(condition, problem) for condition in ("C1", "S3") for problem in gold},
                       "Independent score coverage is not both original30 sets")
    result = dict(status="PASS_CPU_ORIGINAL_FROZEN_V4_EXACT60_AND_TOKEN_REDECODE",
                  scorer_sha256=validation.sha(scorer_path), dataset_sha256=validation.sha(root / "protocol/aime26_frozen.jsonl"),
                  source_freeze_sha256=validation.sha(root / "SOURCE_FREEZE.json"),
                  formal_permit_sha256=validation.sha(root / "FORMAL_GPU_PERMIT.json"),
                  formal_authorization_sha256=validation.sha(root / "AUTHORIZATION_FORMAL.json"),
                  CUDA_visible_devices="", model_loaded=False, new_GPU_dispatches=0,
                  scores=scores)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(validation.canonical(result) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"], score_count=len(scores))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
