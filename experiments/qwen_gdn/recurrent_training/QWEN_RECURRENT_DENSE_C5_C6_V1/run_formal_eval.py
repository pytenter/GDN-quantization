#!/usr/bin/env python3
"""Dense Qwen AIME26 evaluation wrapper.

This module deliberately reuses the frozen manual/HF formal runner.  It only
replaces the existing key-Hadamard patch with H128 followed by the selected
per-layer Cayley correction, and narrows scheduling to AIME26 problems 11..30
at generation seed 1.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch


TASK = "QWEN_RECURRENT_DENSE_C5_C6_V1"
PROTOCOL_VERSION = TASK
EXPERIMENT = Path(__file__).resolve().parent
REPO = EXPERIMENT.parents[1]
BASE_REPO = Path("/data/zypan/worktrees/aime26-sglang-rotation-v1")
BASE_RUNNER = BASE_REPO / "experiments/aime26/run_qwen_aime26_formal.py"
DATASET = BASE_REPO / "artifacts/aime26_v1/formal/dataset/aime26_frozen.jsonl"
CAYLEY_SOURCE = REPO / "experiments/shared/rotation/cayley_rotation.py"
CALIBRATION_CORPUS = REPO / "CALIBRATION_RAW_TEXTS.jsonl"
CHECKPOINTS = {
    "C5": EXPERIMENT / "checkpoints/c5/best_c5_seed0.pt",
    "C6": EXPERIMENT / "checkpoints/c6/best_c6_seed0.pt",
}
CONDITIONS = {
    "C5": "INT8_C128_RECURRENT_DENSE_STATE",
    "C6": "INT8_C128_RECURRENT_DENSE_FUNCTIONAL",
}
EXPECTED_OBJECTIVE = {"C5": "DENSE_STATE", "C6": "DENSE_FUNCTIONAL"}
GDN_LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30)
FORMAL_MAX_NEW_TOKENS = 81920
FORMAL_WORKERS = 4
GATE_TOKEN_LIMIT = 32


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if str(BASE_RUNNER.parent) not in sys.path:
    sys.path.insert(0, str(BASE_RUNNER.parent))
BASE = import_file(BASE_RUNNER, "qwen_aime26_frozen_formal_runner")
CAYLEY = import_file(CAYLEY_SOURCE, "qwen_dense_eval_cayley")
CANONICAL_PATCH = BASE.QwenKeyHadamardPatch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def rows_last20() -> list[dict]:
    rows = BASE.load_frozen_dataset(DATASET)
    selected = rows[10:]
    expected = [f"aime26_{index:02d}" for index in range(11, 31)]
    actual = [row["problem_id"] for row in selected]
    if actual != expected or [int(row["problem_idx"]) for row in selected] != list(range(11, 31)):
        raise RuntimeError(f"last-20 manifest mismatch: {actual}")
    return selected


def load_bank(condition: str, device: torch.device):
    path = CHECKPOINTS[condition]
    if not path.is_file():
        raise FileNotFoundError(f"CHECKPOINT_NOT_FOUND: {path}")
    actual_hash = sha256(path)
    summary = EXPERIMENT / "checkpoints" / condition.lower() / "training_summary.json"
    expected_hash = json.loads(summary.read_text(encoding="utf-8"))["checkpoint_sha256"]
    if actual_hash != expected_hash:
        raise RuntimeError(f"checkpoint hash mismatch for {condition}: {actual_hash} != {expected_hash}")
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    expected_keys = {"task", "model", "condition", "objective", "training_mode", "seed", "lr", "target_exposures", "optimizer_updates", "gradient_horizon", "numerical_rollout", "validation", "bank", "layer_ids"}
    if set(payload) != expected_keys:
        raise RuntimeError(f"checkpoint schema mismatch for {condition}: {sorted(payload)}")
    if payload["task"] != TASK:
        raise RuntimeError(f"checkpoint task mismatch for {condition}")
    if payload["model"] != "Qwen3.5-9B/GDN" or payload["condition"] != condition or payload["objective"] != EXPECTED_OBJECTIVE[condition]:
        raise RuntimeError(f"checkpoint semantic mismatch for {condition}")
    if int(payload["seed"]) != 0 or tuple(payload["layer_ids"]) != GDN_LAYERS:
        raise RuntimeError(f"checkpoint seed/layer mapping mismatch for {condition}")
    if len(payload["bank"]) != len(GDN_LAYERS):
        raise RuntimeError(f"checkpoint bank size mismatch for {condition}")
    if any(tuple(value.shape) != (8128,) or value.dtype != torch.float32 for value in payload["bank"].values()):
        raise RuntimeError(f"checkpoint parameter shape/dtype mismatch for {condition}")
    bank = CAYLEY.PerLayerCayleyRotations(GDN_LAYERS)
    bank.load_state_dict(payload["bank"], strict=True)
    bank.to(device)
    bank.eval()
    gate = bank.runtime_gate(threshold=1.0e-4)
    if gate["status"] != "PASS":
        raise RuntimeError(f"orthogonality gate failed for {condition}: {gate}")
    with torch.no_grad():
        matrices = {layer: bank.layer(layer).matrix().detach().contiguous() for layer in GDN_LAYERS}
    metadata = {key: payload[key] for key in ("task", "model", "condition", "objective", "training_mode", "seed", "lr", "target_exposures", "optimizer_updates", "gradient_horizon", "validation", "layer_ids")}
    metadata.update({"path": str(path), "sha256": actual_hash})
    return matrices, metadata, gate


class DenseKeyPatch(CANONICAL_PATCH):
    """Canonical H128 followed by a per-layer dense SO(128) correction."""

    def __init__(self, model, matrices: dict[int, torch.Tensor]):
        super().__init__(model)
        self.matrices = matrices

    def _transform(self, value, use_norm):
        value = BASE.normalize_qk(value) if use_norm else value
        layer = int(self.current_layer)
        if layer not in self.matrices:
            raise RuntimeError(f"dense rotation invoked on unmapped layer {layer}")
        transformed = value.float().matmul(BASE.hadamard(128, dtype=torch.float32, device=value.device))
        return transformed.matmul(self.matrices[layer])


def configure_base(condition: str, matrices: dict[int, torch.Tensor], max_new_tokens: int) -> None:
    label = CONDITIONS[condition]
    BASE.METHODS["int8_c128_key_h"] = label
    BASE.PROTOCOL_VERSION = PROTOCOL_VERSION
    BASE.MAX_NEW_TOKENS = int(max_new_tokens)
    BASE.FORMAL_WORKERS = FORMAL_WORKERS
    BASE.DATASET = DATASET
    BASE.QwenKeyHadamardPatch = lambda model: DenseKeyPatch(model, matrices)


def runner_revision() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    except Exception:
        commit = "UNKNOWN"
    return {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve()), "repo_commit": commit}


def enrich_record(record: dict, condition: str, checkpoint: dict, gate: dict) -> dict:
    record["task"] = TASK
    record["condition_id"] = condition
    record["condition"] = CONDITIONS[condition]
    record["inherited_sampling_protocol"] = "GDN_KDA_AIME26_OFFICIAL_SAMPLING_81920_2SEED_FORMAL_V2"
    record["dense_runtime"] = {
        "side": "KEY",
        "order": "qk_l2norm -> canonical_H128 -> per_layer_DeltaR -> GDN",
        "cache_basis_continuity": "prefill_and_decode_remain_in_H128_DeltaR_basis",
        "state_re_rotation": False,
        "head_sharing": "one_128x128_matrix_per_recurrent_layer_shared_across_32_heads",
        "matrix_dtype": "float32",
        "readout_before_qdq_writeback": True,
        "checkpoint": checkpoint,
        "orthogonality_gate": {key: gate[key] for key in ("status", "threshold", "max_abs_rt_r_minus_i", "all_finite", "all_proper")},
        "wrapper": runner_revision(),
    }
    return record


def record_valid(row: dict, condition: str) -> bool:
    if row.get("task") != TASK or row.get("condition_id") != condition or row.get("condition") != CONDITIONS[condition]:
        return False
    if int(row.get("seed", -1)) != 1 or row.get("problem_id") not in {f"aime26_{index:02d}" for index in range(11, 31)}:
        return False
    return bool(BASE.is_valid_record(row, "int8_c128_key_h"))


def completed(output: Path, marker: Path, condition: str) -> bool:
    if not output.is_file() or not marker.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata.get("sha256") != sha256(output):
            return False
        row = json.loads(output.read_text(encoding="utf-8"))
        return record_valid(row, condition)
    except Exception:
        return False


def run_formal(condition: str, worker_id: int, worker_count: int) -> None:
    if worker_count != FORMAL_WORKERS or worker_id not in range(worker_count):
        raise SystemExit(f"formal requires --worker-count {FORMAL_WORKERS} and worker id 0..3")
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if physical_gpu not in {"0", "1", "2", "3"}:
        raise SystemExit("set CUDA_VISIBLE_DEVICES to one physical GPU 0..3")
    model, tokenizer = BASE.load_model_and_tokenizer()
    device = next(model.parameters()).device
    matrices, checkpoint, gate = load_bank(condition, device)
    configure_base(condition, matrices, FORMAL_MAX_NEW_TOKENS)
    jobs = [(index, row) for index, row in enumerate(rows_last20()) if index % worker_count == worker_id]
    output_dir = EXPERIMENT / "outputs" / CONDITIONS[condition]
    attempts = EXPERIMENT / "logs" / f"{condition}_worker{worker_id}_attempts.jsonl"
    for local_index, (canonical_index, item) in enumerate(jobs, 1):
        stem = f"{item['problem_id']}_seed1"
        output = output_dir / f"{stem}.json"
        marker = output_dir / f"{stem}.complete.json"
        if completed(output, marker, condition):
            print(f"[{time.strftime('%F %T')}] skip {condition} {stem}", flush=True)
            continue
        print(f"[{time.strftime('%F %T')}] {local_index}/{len(jobs)} {condition} {stem}", flush=True)
        try:
            record = BASE.generate_one(model, tokenizer, item, 1, "int8_c128_key_h", physical_gpu, worker_id, canonical_index)
            record = enrich_record(record, condition, checkpoint, gate)
            if not record_valid(record, condition):
                raise RuntimeError("generated record failed frozen protocol validation")
            atomic_json(output, record)
            atomic_json(marker, {
                "task": TASK,
                "condition": CONDITIONS[condition],
                "problem_id": item["problem_id"],
                "generation_seed": 1,
                "sha256": sha256(output),
                "size": output.stat().st_size,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
        except Exception:
            attempts.parent.mkdir(parents=True, exist_ok=True)
            attempt = {
                "condition": condition,
                "problem_id": item["problem_id"],
                "seed": 1,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "traceback": traceback.format_exc(),
            }
            with attempts.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(attempt, sort_keys=True, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            raise


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def non_eval_text() -> str:
    rows = [json.loads(line) for line in CALIBRATION_CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    heldout = next(row for row in rows if row["split"] == "HELDOUT")
    return heldout["raw_text"][:600]


def capture_fixed(model, tokenizer, patch_factory, forced_tokens=None, steps: int = 4, quantized: bool = True):
    _messages, input_ids = BASE.render_input(tokenizer, non_eval_text())
    input_ids = input_ids.to(next(model.parameters()).device)
    attention_mask = torch.ones_like(input_ids)
    tokens, logits_hashes, state_hashes = [], [], []
    final_logits = None
    final_states = None
    quant_audit = {"calls": 0, "layers": set(), "scale_shapes": {}, "code_min": 0, "code_max": 0}
    context = patch_factory(model) if patch_factory is not None else contextlib.nullcontext()
    with context, torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past = output.past_key_values
        for step in range(steps):
            logits = output.logits[:, -1, :].float()
            logits_hashes.append(tensor_sha256(logits))
            token = int(forced_tokens[step]) if forced_tokens is not None else int(torch.argmax(logits, dim=-1).item())
            tokens.append(token)
            current = torch.tensor([[token]], dtype=torch.long, device=input_ids.device)
            output = model(input_ids=current, past_key_values=past, use_cache=True)
            past = output.past_key_values
            if quantized:
                BASE.quantize_c128(past, quant_audit)
            state_hashes.append({str(layer): tensor_sha256(BASE.get_state(past, layer)) for layer in GDN_LAYERS})
        final_logits = output.logits[:, -1, :].detach().float().cpu()
        final_states = {layer: BASE.get_state(past, layer).detach().float().cpu().clone() for layer in GDN_LAYERS}
    return {
        "tokens": tokens,
        "logits_hashes": logits_hashes,
        "state_hashes": state_hashes,
        "quantizer_calls": quant_audit["calls"],
        "quantizer_layers": sorted(quant_audit["layers"]),
        "input_ids_hash": BASE.ids_sha256(input_ids[0].tolist()),
    }, final_logits, final_states


def short_generation(model, tokenizer, condition: str, matrices, checkpoint, gate) -> dict:
    configure_base(condition, matrices, GATE_TOKEN_LIMIT)
    item = {"problem_id": "non_eval_gate", "problem_idx": 0, "problem": non_eval_text(), "answer": "0"}
    first = enrich_record(BASE.generate_one(model, tokenizer, item, 314159, "int8_c128_key_h", os.environ.get("CUDA_VISIBLE_DEVICES", ""), 0, 0), condition, checkpoint, gate)
    second = enrich_record(BASE.generate_one(model, tokenizer, item, 314159, "int8_c128_key_h", os.environ.get("CUDA_VISIBLE_DEVICES", ""), 0, 0), condition, checkpoint, gate)
    return {
        "exact_token_id_match": first["generated_token_ids"] == second["generated_token_ids"],
        "run1_token_ids": first["generated_token_ids"],
        "run2_token_ids": second["generated_token_ids"],
        "runtime_error": [first["runtime_error"], second["runtime_error"]],
        "nonfinite": [first["nonfinite"], second["nonfinite"]],
        "finish_reason": [first["finish_reason"], second["finish_reason"]],
        "quantizer_calls": [first["quantizer_audit"]["calls"], second["quantizer_audit"]["calls"]],
        "rotation_layers": [first["hadamard_audit"]["layers"], second["hadamard_audit"]["layers"]],
        "peak_gpu_memory_mb": max(first["peak_gpu_memory_mb"], second["peak_gpu_memory_mb"]),
        "gate": "PASS" if first["generated_token_ids"] == second["generated_token_ids"] and not any([first["runtime_error"], second["runtime_error"], first["nonfinite"], second["nonfinite"]]) else "FAIL",
    }


def run_gate() -> None:
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if physical_gpu not in {"0", "1", "2", "3"}:
        raise SystemExit("set CUDA_VISIBLE_DEVICES to one physical GPU 0..3")
    model, tokenizer = BASE.load_model_and_tokenizer()
    device = next(model.parameters()).device
    identity = torch.eye(128, device=device, dtype=torch.float32)
    step0_matrices = {layer: identity for layer in GDN_LAYERS}
    step0_meta = {"kind": "UNTRAINED_HADAMARD_INITIALIZATION", "step": 0}
    step0_gate = {"status": "PASS", "threshold": 1.0e-4, "max_abs_rt_r_minus_i": 0.0, "all_finite": True, "all_proper": True}

    BASE.MAX_NEW_TOKENS = GATE_TOKEN_LIMIT
    canonical, canonical_logits, canonical_states = capture_fixed(model, tokenizer, CANONICAL_PATCH)
    step0_factory = lambda current_model: DenseKeyPatch(current_model, step0_matrices)
    step0, step0_logits, step0_states = capture_fixed(model, tokenizer, step0_factory, canonical["tokens"])
    final_logit_max_abs = float((canonical_logits - step0_logits).abs().max().item())
    final_state_max_abs = max(float((canonical_states[layer] - step0_states[layer]).abs().max().item()) for layer in GDN_LAYERS)
    step0_exact = canonical["tokens"] == step0["tokens"] and canonical["logits_hashes"] == step0["logits_hashes"] and canonical["state_hashes"] == step0["state_hashes"]
    del canonical_logits, canonical_states, step0_logits, step0_states

    reference = Path("/data/zypan/consolidation/GDN-quantization-qwen3090-20260921/results/aime26/81920/qwen_gdn/evidence/reference_closure/qwen_operator_equivalence.json")
    result = {
        "task": TASK,
        "status": "PASS" if step0_exact else "FAIL",
        "C5C6_STEP0_REGRESSION_PARITY": "PASS" if step0_exact else "FAIL",
        "non_evaluation_input": {"source": str(CALIBRATION_CORPUS), "input_ids_hash": canonical["input_ids_hash"], "steps": 4},
        "dense_step0_vs_canonical_hadamard": {
            "exact_token_ids": canonical["tokens"] == step0["tokens"],
            "exact_logit_hashes": canonical["logits_hashes"] == step0["logits_hashes"],
            "exact_state_hashes_after_qdq": canonical["state_hashes"] == step0["state_hashes"],
            "final_logits_max_abs": final_logit_max_abs,
            "final_state_max_abs": final_state_max_abs,
            "gate": "PASS" if step0_exact else "FAIL",
            "c2b_required": False if step0_exact else True,
            "step0_metadata": step0_meta,
            "step0_orthogonality": step0_gate,
        },
        "prior_frozen_evidence": {
            "fp_no_quant_rotation_equivalence": {"path": str(reference), "sha256": sha256(reference), "payload": json.loads(reference.read_text(encoding="utf-8"))},
        },
        "gpu": physical_gpu,
        "runner": runner_revision(),
    }
    atomic_json(EXPERIMENT / "analysis/step0_regression_parity.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gate", "formal"), required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS))
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=FORMAL_WORKERS)
    args = parser.parse_args()
    if args.mode == "gate":
        if args.condition is not None:
            raise SystemExit("gate audits all candidates; omit --condition")
        run_gate()
        return
    if args.condition not in CONDITIONS:
        raise SystemExit("formal requires --condition")
    run_formal(args.condition, args.worker_id, args.worker_count)


if __name__ == "__main__":
    main()
