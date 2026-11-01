#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/data/zypan")
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
SHARDS = ROOT / "results" / "gdn_end2end_bit_axis_screening_v1_shards"
CLAIMS = SHARDS / "p0_int8_row_missing_claims"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_end2end_bit_axis_screening as e


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def done_ids():
    done = set()
    for p in sorted(SHARDS.glob("shard_*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("benchmark") == "MATH-500" and r.get("quantizer") == "INT8-row":
                    done.add(r.get("problem_id"))
    return done


def claim(problem_id, worker_id):
    CLAIMS.mkdir(parents=True, exist_ok=True)
    safe = problem_id.replace("/", "__")
    path = CLAIMS / safe
    try:
        os.mkdir(path)
    except FileExistsError:
        return None
    (path / "worker.json").write_text(
        json.dumps({"worker_id": worker_id, "problem_id": problem_id, "time": now()}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-id", type=int, required=True)
    args = ap.parse_args()

    print("[%s] worker %d start" % (now(), args.worker_id), flush=True)
    jobs = e.planned_jobs(False, ["int8_row"], ["MATH-500"])
    job_by_pid = {item["problem_id"]: (ck, item, max_new) for ck, item, max_new in jobs}
    ordered = [item["problem_id"] for ck, item, max_new in jobs]

    e.run_protocol()
    torch, _transformers, model, tokenizer = e.load_model()
    qcfg = e.cfg_dict("int8_row")
    out_path = SHARDS / ("shard_p0_int8_row_queue_%03d.jsonl" % args.worker_id)
    t0 = time.time()
    completed = 0

    while True:
        done = done_ids()
        target = None
        claim_path = None
        for pid in ordered:
            if pid in done:
                continue
            claim_path = claim(pid, args.worker_id)
            if claim_path is not None:
                target = pid
                break
        if target is None:
            print("[%s] worker %d no unclaimed missing jobs; exit" % (now(), args.worker_id), flush=True)
            break
        if target in done_ids():
            print("[%s] worker %d claimed already-done %s; skip" % (now(), args.worker_id, target), flush=True)
            continue
        ck, item, max_new = job_by_pid[target]
        print("[%s] worker %d running %s" % (now(), args.worker_id, target), flush=True)
        try:
            rec = e.generate_one(torch, model, tokenizer, item, qcfg, max_new)
        except Exception as exc:
            rec = {
                "task": e.TASK,
                "benchmark": item["benchmark"],
                "problem_id": item["problem_id"],
                "problem_index": item["problem_index"],
                "quantizer": qcfg["quantizer"],
                "bit_width": qcfg["bit_width"],
                "axis": qcfg["axis"],
                "seed": e.SEED + int(item["problem_index"]),
                "correct": False,
                "predicted_answer": "",
                "reference_answer": item["reference_answer"],
                "output_token_count": 0,
                "completed": False,
                "nonfinite": False,
                "truncated": False,
                "max_token_hit": False,
                "response": "",
                "ref": item.get("solution"),
                "exception": repr(exc),
            }
        rec["run_elapsed_sec"] = time.time() - t0
        rec["p0_queue_worker_id"] = args.worker_id
        rec["p0_claim_path"] = str(claim_path)
        append_jsonl(out_path, rec)
        completed += 1
        print("[%s] worker %d wrote %s completed=%d" % (now(), args.worker_id, target, completed), flush=True)


if __name__ == "__main__":
    main()
