#!/usr/bin/env python3
"""Formal-path Stage-B reducer parity runner for gated P0/P1 phases."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


TASK = "QWEN_GDN_STAGE_B_STREAMING_ACCELERATION_V1"
SOURCE_BASE = "255a41b2217b7044926c0b4d41dc89eaeed6d509"
REPO = Path("/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2")
ROOT = REPO / "experiments/QWEN_GDN_STAGE_B_STREAMING_ACCELERATION_V1"
OLD_SMOKE = REPO / "experiments/qwen_gdn/QWEN_GDN_STRICT_COMMON_FP_STATE_FANOUT_V1/scripts/stage_b_smoke.py"
ORACLE_PATH = REPO / "experiments/qwen_gdn/QWEN_GDN_STAGE_B_STREAMING_REDUCER_V3_CLOSURE/scripts/reducer_v3.py"
ACCEL_PATH = ROOT / "code/accelerated_reducer.py"
FORMAL_PATH = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py")
DEPLOY = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1/deployment")
TEACHER = Path("/data/zypan/worktrees/aime26-sglang-rotation-v1/artifacts/aime26_v2/official_sampling_81920/qwen/formal/shards")
SMOKE_IDENTITY_PATH = REPO / "experiments/qwen_gdn/QWEN_GDN_STAGE_B_GPU_STREAMING_SMOKE_V3/smoke_identity.json"
CONDITIONS = ("Native", "Hadamard", "C5", "C6")
SMOKE_UNITS = (("aime26_01", 1), ("aime26_01", 2))
SMOKE_TOKENS = 32
FULL_TEACHER_TOKENS = 1854451
EPS = 1e-12
EXPECTED_ROTATION = {
    "C5": "0266c5065267cf452d72cfe29a2a4d9a24fdfb6a89d7bec04b4d796c86f857c9",
    "C6": "67e3731213ad3fb0c088c781ddd912aaaad5af75f9532b289442690a7b09f39c",
}


def loadmod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def token_sha(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def load_rows():
    rows = []
    for path in sorted(TEACHER.glob("worker*/fp_state.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    rows.sort(key=lambda row: (int(row["canonical_index"]), int(row["seed"])))
    index = {(row["problem_id"], int(row["seed"])): row for row in rows}
    return rows, [index[key] for key in SMOKE_UNITS]


def section_equal(oracle_state, accel_state):
    checks = {
        "histogram_integer_counts": oracle_state["layer_histograms"] == accel_state["layer_histograms"] and oracle_state["global_histograms"] == accel_state["global_histograms"],
        "tail_p99_cvar99_max": oracle_state["tail_moments"] == accel_state["tail_moments"],
        "threshold_counts": oracle_state["paired"] == accel_state["paired"],
        "topk_identity": oracle_state["extreme_topk"] == accel_state["extreme_topk"],
        "moments": oracle_state["moments"] == accel_state["moments"],
        "cursor_and_tokens": oracle_state["processing_cursor"] == accel_state["processing_cursor"] and oracle_state["tokens_seen"] == accel_state["tokens_seen"],
    }
    checks["full_state_exact"] = canonical_bytes(oracle_state) == canonical_bytes(accel_state)
    return checks


class DualAccumulator:
    def __init__(self, oracle, accelerated, v3):
        self.oracle = oracle
        self.accelerated = accelerated
        self.v3 = v3
        self.buffer = {}
        self.sample_id = oracle.identity["sample_id"]
        self.oracle_seconds = 0.0
        self.accelerated_seconds = 0.0

    @staticmethod
    def metrics(source, scale, codes, dequant):
        minimum = source.amin(dim=-2)
        maximum = source.amax(dim=-2)
        absmax = source.abs().amax(dim=-2)
        power = source.square().mean(dim=-2)
        rms = power.sqrt()
        err = dequant - source
        l2 = err.square().sum(dim=-2).sqrt()
        source_l2 = source.square().sum(dim=-2).sqrt()
        return {"scale": scale.squeeze(-2), "papr_linear": absmax / rms.clamp_min(EPS), "qdq_relative_l2": l2 / source_l2.clamp_min(EPS), "absmax": absmax, "range": maximum - minimum}

    def update(self, sample_id, token_idx, condition, layer, source, scale, codes, dequant):
        if sample_id != self.sample_id:
            raise RuntimeError("sample identity mismatch")
        key = (int(layer), condition)
        if key in self.buffer:
            raise RuntimeError("duplicate capture")
        tensors = self.metrics(source, scale, codes, dequant)
        self.buffer[key] = {"group_ids": np.arange(self.v3.N_GROUPS, dtype=np.int64), "metrics": {name: value.detach().float().reshape(-1).cpu().numpy().astype(np.float64) for name, value in tensors.items()}}
        if len(self.buffer) == len(self.v3.LAYERS) * len(self.v3.CONDITIONS):
            started = time.perf_counter(); self.oracle.update_token(token_idx, self.buffer); self.oracle_seconds += time.perf_counter() - started
            started = time.perf_counter(); self.accelerated.update_token(token_idx, self.buffer); self.accelerated_seconds += time.perf_counter() - started
            self.buffer = {}

    def finish(self):
        if self.buffer:
            raise RuntimeError("partial token buffer")


def main():
    if len(sys.argv) != 5 or sys.argv[1] != "--phase" or sys.argv[3] != "--gpu" or sys.argv[2] not in ("p0", "p1"):
        raise SystemExit("usage: run_parity.py --phase p0|p1 --gpu N")
    phase = sys.argv[2]
    requested_gpu = int(sys.argv[4])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(requested_gpu)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    start_status = git("status", "--porcelain")
    start_head = git("rev-parse", "HEAD")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_BASE, start_head], cwd=REPO).returncode == 0
    if start_status or not ancestor:
        raise RuntimeError(f"SOURCE_GATE=FAIL dirty={bool(start_status)} head={start_head}")

    old = loadmod(OLD_SMOKE, "accel_frozen_fanout")
    v3 = loadmod(ORACLE_PATH, "accel_v3_oracle")
    accel = loadmod(ACCEL_PATH, "accel_p0_module")
    formal = loadmod(FORMAL_PATH, "accel_formal_runner")
    rows, selected = load_rows()
    if len(rows) != 60 or sum(len(row["generated_token_ids"]) for row in rows) != FULL_TEACHER_TOKENS:
        raise RuntimeError("teacher identity mismatch")
    expected_phase = "P0_HISTOGRAM_ONLY" if phase == "p0" else "P1_HISTOGRAM_AND_TOPK"
    if accel.ACCELERATION_PHASE != expected_phase:
        raise RuntimeError(f"{phase.upper()} runner accelerator phase mismatch")
    if phase == "p1":
        p0 = json.loads((ROOT / "parity/p0_parity.json").read_text())
        if p0.get("STREAMING_REDUCER_ACCEL_P0_PARITY") != "PASS":
            raise RuntimeError("P1 forbidden before P0 PASS")
    expected_smoke = json.loads(SMOKE_IDENTITY_PATH.read_text())["samples"]

    provenance = {
        "SOURCE_GATE": "PASS", "source_head": start_head, "source_branch": git("symbolic-ref", "--short", "HEAD"),
        "oracle_path": str(ORACLE_PATH), "oracle_sha256": sha(ORACLE_PATH),
        "accelerated_path": str(ACCEL_PATH), "accelerated_sha256": sha(ACCEL_PATH),
        "formal_path": str(FORMAL_PATH), "formal_sha256": sha(FORMAL_PATH),
        "smoke_identity_path": str(SMOKE_IDENTITY_PATH), "smoke_identity_sha256": sha(SMOKE_IDENTITY_PATH),
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "phase": accel.ACCELERATION_PHASE, "samples": [f"{p}_seed{s}" for p, s in SMOKE_UNITS], "tokens": 64,
    }
    dump(ROOT / f"audit/{phase}_source_identity.json", provenance)

    model, tokenizer = formal.load_model_and_tokenizer()
    device = next(model.parameters()).device
    banks, rotations = {}, {}
    for condition in ("C5", "C6"):
        rotation = DEPLOY / condition / "frozen_rotation_fp32.pt"
        manifest = DEPLOY / condition / "deployment_manifest.json"
        if sha(rotation) != EXPECTED_ROTATION[condition]:
            raise RuntimeError("rotation identity mismatch")
        banks[condition], rotations[condition] = formal.load_frozen_rotation(str(rotation), str(manifest), device, condition.lower())
    h_patch = formal.QwenKeyHadamardPatch(model)
    c5_patch = formal.QwenFrozenRotationPatch(model, banks["C5"], rotations["C5"])
    c6_patch = formal.QwenFrozenRotationPatch(model, banks["C6"], rotations["C6"])

    from transformers import DynamicCache
    results, all_checks = [], []
    with old.StrictCommonFPFanout(model, formal, h_patch, c5_patch, c6_patch) as fanout, torch.inference_mode():
        for row in selected:
            sample_id = f"{row['problem_id']}_seed{int(row['seed'])}"
            expected = expected_smoke[sample_id]
            tokens = list(map(int, row["generated_token_ids"][:SMOKE_TOKENS]))
            if len(row["generated_token_ids"]) != expected["teacher_tokens"] or token_sha(tokens) != expected["prefix_sha256"] or token_sha(row["generated_token_ids"]) != expected["full_sha256"]:
                raise RuntimeError("sample identity mismatch")
            identity = {"sample_id": sample_id, "source_commit": SOURCE_BASE, "reducer_schema_version": v3.SCHEMA_VERSION, "input_identity_hash": expected["prefix_sha256"]}
            oracle = v3.SampleReducerV3(identity)
            reducer_class = accel.AcceleratedSampleReducerP0 if phase == "p0" else accel.AcceleratedSampleReducerP1
            accelerated = reducer_class(identity)
            accumulator = DualAccumulator(oracle, accelerated, v3)
            observer = old.QuantizerObserver(formal.quantize_c128, accumulator)
            audit = {condition: {"calls": 0, "layers": set(), "scale_shapes": {}, "code_min": 0, "code_max": 0} for condition in CONDITIONS}
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
            prompt = tokenizer.apply_chat_template(row["raw_messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"].to(device)
            attention = encoded.get("attention_mask"); attention = attention.to(device) if attention is not None else None
            cache = DynamicCache(config=model.config, offloading=True, offload_only_non_sliding=True)
            fanout.capture = False
            prefill = model(input_ids=input_ids, attention_mask=attention, past_key_values=cache, use_cache=True, logits_to_keep=1)
            past = prefill.past_key_values
            del prefill, input_ids, attention
            offset = len(fanout.state_checks)
            for token_idx, token_id in enumerate(tokens):
                fanout.begin(sample_id, token_idx)
                current = torch.tensor([[token_id]], dtype=torch.long, device=device)
                output = model(input_ids=current, past_key_values=past, use_cache=True, logits_to_keep=1)
                past = output.past_key_values
                fanout.end()
                for condition in CONDITIONS:
                    states = {layer: formal.get_state(past, layer).detach().clone() for layer in formal.GDN_LAYERS} if condition == "Native" else fanout.local_states[condition]
                    observer.run(old.FakeCache(states, max(formal.GDN_LAYERS)), audit[condition], sample_id, token_idx, condition, formal.GDN_LAYERS)
                    del states
                del output, current
                if (token_idx + 1) % 4 == 0:
                    print(json.dumps({"event": f"{phase}_parity_progress", "sample_id": sample_id, "token": token_idx + 1, "total": len(tokens)}), flush=True)
            torch.cuda.synchronize(device)
            accumulator.finish()
            oracle_state, accel_state = oracle.to_state(), accelerated.to_state()
            checks = section_equal(oracle_state, accel_state)
            fanout_checks = fanout.state_checks[offset:]
            common_fp = len(fanout_checks) == 2 * len(formal.GDN_LAYERS) and all(len(set(item["branch_sha256"].values())) == 1 and all(item["branch_torch_equal"].values()) for item in fanout_checks)
            quantizer_ok = all(audit[c]["calls"] == len(tokens) and audit[c]["layers"] == set(formal.GDN_LAYERS) for c in CONDITIONS)
            result = {"sample_id": sample_id, "tokens": len(tokens), "checks": checks, "oracle_sha256": hashlib.sha256(canonical_bytes(oracle_state)).hexdigest(), "accelerated_sha256": hashlib.sha256(canonical_bytes(accel_state)).hexdigest(), "oracle_reducer_seconds": accumulator.oracle_seconds, "accelerated_reducer_seconds": accumulator.accelerated_seconds, "reducer_speedup": accumulator.oracle_seconds / accumulator.accelerated_seconds, "COMMON_FP_GATE": "PASS" if common_fp else "FAIL", "FORMAL_QUANTIZER_GATE": "PASS" if quantizer_ok else "FAIL"}
            results.append(result); all_checks.extend(fanout_checks)
            if not all(checks.values()) or not common_fp or not quantizer_ok:
                gate = f"STREAMING_REDUCER_ACCEL_{phase.upper()}_PARITY"
                dump(ROOT / f"parity/{phase}_parity.json", {gate: "FAIL", "results": results})
                raise RuntimeError(f"{gate}=FAIL")
            del past, cache, oracle, accelerated, accumulator, observer
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)

    total_oracle = sum(item["oracle_reducer_seconds"] for item in results)
    total_accel = sum(item["accelerated_reducer_seconds"] for item in results)
    gate = f"STREAMING_REDUCER_ACCEL_{phase.upper()}_PARITY"
    output = {gate: "PASS", "samples": 2, "tokens": 64, "same_model": True, "same_seed": True, "same_sample_order": True, "same_token_order": True, "results": results, "aggregate": {"oracle_reducer_seconds": total_oracle, "accelerated_reducer_seconds": total_accel, "reducer_speedup": total_oracle / total_accel}, "forbidden": {"FULL60_STARTED": False, "RECURRENT_REPLAY_STARTED": False, "MULTI_HORIZON_STARTED": False, "LAYER_INTERVENTION_STARTED": False, "C7_STARTED": False, "NEW_ROTATION_TRAINED": False}}
    dump(ROOT / f"parity/{phase}_parity.json", output)
    dump(ROOT / f"benchmark/{phase}_reducer_benchmark.json", output["aggregate"])
    (ROOT / f"reports/{phase.upper()}_PARITY_REPORT.md").write_text(f"# {phase.upper()} parity report\n\n`{gate} = PASS`\n\n64/64 tokens and both samples matched exactly.\n\nReducer-only speedup: **{total_oracle/total_accel:.3f}x**.\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "gate": gate, "reducer_speedup": total_oracle / total_accel}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
