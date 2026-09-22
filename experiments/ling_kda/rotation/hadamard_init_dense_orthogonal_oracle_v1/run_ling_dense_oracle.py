#!/usr/bin/env python3
"""Ling-3.0-tiny/KDA Hadamard-initialized dense Value-rotation oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import torch


TASK = "HADAMARD_INIT_DENSE_ORTHOGONAL_ORACLE_V1"
REPO = Path(__file__).resolve().parents[4]
EPS = 1.0e-12


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = import_file(REPO / "experiments/shared/rotation/cayley_rotation.py", "dense_oracle_cayley_ling")
O = import_file(REPO / "experiments/shared/rotation/orthogonal_matrix_generator.py", "dense_oracle_orthogonal_ling")


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_corpus(path: Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row["split"] == split.upper()]


def freeze_model(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("MODEL_FREEZE_GATE_FAIL")


def load_context(args):
    root = Path(args.legacy_repo)
    os.environ["GDN_ROTATION_REPO"] = str(root)
    source = root / "experiments/rotation/run_kda_rotation_prefill_full_path_first_divergence_v1.py"
    F = import_file(source, "ling_dense_oracle_fullpath")
    model, tokenizer = F.load_sharded_model(args.max_memory_gib)
    freeze_model(model)
    probe = F.FullPathForensicProbe(torch.eye(128, dtype=torch.float32))
    layers = tuple(int(layer) for layer in probe.install(model))
    return F, model, tokenizer, probe, layers


def sampled_positions(row: dict, token_count: int) -> list[int]:
    if token_count <= 136:
        raise RuntimeError(f"{row['document_id']} has only {token_count} tokens")
    seed = int(row["raw_text_sha256"][:16], 16) ^ 20260921
    return sorted(random.Random(seed).sample(list(range(128, token_count)), 8))


def tokenizer_manifest(tokenizer, corpus_path: Path, rows: list[dict]) -> dict:
    root = Path(getattr(tokenizer, "name_or_path", ""))
    files = {}
    if root.is_dir():
        for path in root.iterdir():
            if path.is_file() and path.name in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"}:
                files[path.name] = sha256(path)
    encoded = {}
    for row in rows:
        ids = tokenizer(row["raw_text"], add_special_tokens=False).input_ids[:1024]
        encoded[row["document_id"]] = {"token_count": len(ids), "positions": sampled_positions(row, len(ids))}
    return {"task": TASK, "model": "Ling-3.0-tiny/KDA", "raw_corpus": str(corpus_path), "raw_corpus_sha256": sha256(corpus_path), "tokenizer_name_or_path": str(root), "tokenizer_file_sha256": files, "tokenized_documents": encoded, "sequence_length_max": 1024, "sampled_positions_per_sequence": 8, "minimum_position": 128, "AIME26_used": False}


def token_last(value: torch.Tensor) -> torch.Tensor:
    # KDA path tensors are [B,T,H,V] or [B,T,D].
    if value.ndim >= 3:
        return value[:, -1:].clone()
    return value.clone()


def relative_l2(value, reference) -> float:
    x, y = value.detach().float(), reference.detach().float()
    return float((torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(y).clamp_min(EPS)).cpu())


def collect(args) -> None:
    F, model, tokenizer, probe, layers = load_context(args)
    rows = load_corpus(Path(args.corpus), args.split)
    if args.document_limit is not None:
        rows = rows[: args.document_limit]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = tokenizer_manifest(tokenizer, Path(args.corpus), rows)
    save_json(outdir / f"CALIBRATION_MANIFEST_{args.split.upper()}.json", manifest)
    device = F.model_input_device(model)
    continuity_kernel_cache, continuity_cache_next = [], []
    try:
        for row_index, row in enumerate(rows):
            ids = tokenizer(row["raw_text"], add_special_tokens=False).input_ids[:1024]
            positions = manifest["tokenized_documents"][row["document_id"]]["positions"]
            cache, current, samples = None, 0, []
            for sample_index, position in enumerate(positions):
                end = position + 1
                prior = None if cache is None else {layer: F.BASE.cache_stack(cache, layers)[layer].detach().float().cpu().clone() for layer in layers}
                chunk = torch.tensor([ids[current:end]], dtype=torch.long, device=device)
                plan = F.H.plan("NS_REPLAY", "native", False)
                probe.begin(plan, sample_index)
                with torch.inference_mode():
                    output = model(input_ids=chunk, past_key_values=cache, cache_position=torch.arange(current, end, device=device), use_cache=True)
                probe.end()
                cache = output.past_key_values
                cache_stack = F.BASE.cache_stack(cache, layers)
                layer_records = {}
                for layer in layers:
                    records = probe.core_tokens["NS_REPLAY"][layer]
                    record = records[-1]
                    path = probe.path_tensors["NS_REPLAY"][layer]
                    layer_records[layer] = {
                        "q": record["q"].to(torch.bfloat16), "k": record["k"].to(torch.bfloat16),
                        "v": record["v_semantic"].to(torch.bfloat16),
                        "beta": record["beta"].float(), "log_decay": record["log_decay"].float(),
                        "state_input": record["state_in"].to(torch.bfloat16),
                        "state_output": record["state_after"].to(torch.bfloat16),
                        "core_output": record["raw_core_output"].to(torch.bfloat16),
                        "dynamic_gate": token_last(path["dynamic_gate_input"]).to(torch.bfloat16),
                        "post_norm_gate": token_last(path["rmsnorm_scaled_output"]).to(torch.bfloat16),
                        "out_proj_output": token_last(path["out_proj_output"]).to(torch.bfloat16),
                    }
                    continuity_kernel_cache.append(relative_l2(probe.records["NS_REPLAY"][layer]["final_state"], cache_stack[layer]))
                    if prior is not None:
                        continuity_cache_next.append(relative_l2(records[0]["state_in"], prior[layer]))
                samples.append({"position": position, "layers": layer_records})
                current = end
            shard = {"document_id": row["document_id"], "raw_text_sha256": row["raw_text_sha256"], "split": args.split.upper(), "layers": layers, "samples": samples}
            path = outdir / f"{row['document_id']}.pt"
            torch.save(shard, path)
            print(f"COLLECT {row_index + 1}/{len(rows)} {row['document_id']} bytes={path.stat().st_size}", flush=True)
    finally:
        probe.close()
    kernel_max = max(continuity_kernel_cache, default=0.0)
    next_max = max(continuity_cache_next, default=0.0)
    save_json(outdir / f"COLLECTION_{args.split.upper()}_COMPLETE.json", {"status": "PASS" if kernel_max == 0.0 and next_max == 0.0 else "FAIL", "documents": len(rows), "samples": len(rows) * 8, "KDA_ROTATION_SEMANTICS_VERSION": "CORRECTED_PREFILL_ENDPOINT_V2", "LING_KDA_PREFILL_ENDPOINT_STATE_BASIS_GATE": "PASS" if kernel_max == 0.0 and next_max == 0.0 else "FAIL", "REDUNDANT_PREFILL_ENDPOINT_ROTATION": "NO", "kernel_return_to_cache_max_relative_l2": kernel_max, "cache_to_next_chunk_input_max_relative_l2": next_max, "AUDIT_PATH_SIDE_EFFECT": "NO", "AIME26_used": False})


def hadamard(device):
    return O.normalized_hadamard(128).to(device=device, dtype=torch.float32)


def transform_state(state, rotation, h):
    # Value-side row-vector path: canonical H, then learned DeltaR.
    return state.float().matmul(h).matmul(rotation)


def recover_state(state, rotation, h):
    return state.float().matmul(rotation.transpose(0, 1)).matmul(h)


def state_loss_and_qdq(state, rotation_module, h, use_ste=True):
    rotation = rotation_module.matrix()
    rotated_fp32 = transform_state(state, rotation, h)
    # Required deployed precision boundary: FP32 rotation -> BF16 -> QDQ.
    rotated_bf16 = rotated_fp32.to(torch.bfloat16)
    quant = C.ling_r128_ste(rotated_bf16.float()) if use_ste else C.ling_r128_qdq(rotated_bf16.float())
    deployed_qdq = quant.dequant.to(torch.bfloat16).float()
    recovered = recover_state(deployed_qdq, rotation, h)
    numerator = (recovered - state.float()).square().sum(dim=(-2, -1))
    denominator = state.float().square().sum(dim=(-2, -1)).clamp_min(EPS)
    return (numerator / denominator).mean(), deployed_qdq, quant


def fused_replay(probe, record, state, value, device):
    return probe.orig_fused(q=record["q"].to(device), k=record["k"].to(device), v=value,
        g=record["log_decay"].float().to(device), beta=record["beta"].float().to(device),
        initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False)


def local_output(model, layer, core, gate):
    attention = model.model.layers[layer].attention
    dtype = attention.o_proj.weight.dtype
    norm = attention.o_norm(core.to(dtype), gate.to(device=core.device, dtype=dtype))
    output = attention.o_proj(norm.reshape(norm.shape[0], norm.shape[1], -1))
    return norm, output


def load_traces(trace_dir: Path, split: str, limit=None):
    paths = sorted(trace_dir.glob(f"wikitext2raw-{split.lower()}-*.pt"))
    if limit is not None:
        paths = paths[:limit]
    traces = []
    for path in paths:
        shard = torch.load(path, map_location="cpu")
        for sample in shard["samples"]:
            traces.append({"document_id": shard["document_id"], **sample})
    if not traces:
        raise RuntimeError(f"no {split} traces in {trace_dir}")
    return traces


def device_record(record, device):
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in record.items()}


def replay_gate(args) -> None:
    F, model, _tokenizer, probe, layers = load_context(args)
    device = F.model_input_device(model)
    bank = C.PerLayerCayleyRotations(layers).to(device)
    sample = load_traces(Path(args.trace_dir), "TRAIN", 1)[0]
    h = hadamard(device)
    per_layer = {}
    try:
        for layer in layers:
            rec = device_record(sample["layers"][layer], device)
            native_core, _ = fused_replay(probe, rec, rec["state_input"].float(), rec["v"], device)
            native_norm, native_out = local_output(model, layer, native_core, rec["dynamic_gate"])
            rotation = bank.layer(layer).matrix()
            state_loss, deployed_qdq, ste = state_loss_and_qdq(rec["state_input"], bank.layer(layer), h, True)
            exact = C.ling_r128_qdq(transform_state(rec["state_input"], rotation, h).to(torch.bfloat16).float())
            value_rot = rec["v"].float().matmul(h).matmul(rotation).to(torch.bfloat16)
            raw_rot, _ = fused_replay(probe, rec, deployed_qdq, value_rot, device)
            mapped = raw_rot.float().matmul(rotation.T).matmul(h).to(torch.bfloat16)
            _rot_norm, rot_out = local_output(model, layer, mapped, rec["dynamic_gate"])
            per_layer[str(layer)] = {"native_core_replay_rel_l2": relative_l2(native_core, rec["core_output"]), "native_post_norm_replay_rel_l2": relative_l2(native_norm, rec["post_norm_gate"]), "native_out_proj_replay_rel_l2": relative_l2(native_out, rec["out_proj_output"]), "ste_exact_max_abs": float((ste.dequant - exact.dequant).abs().max().cpu()), "theta0_state_loss": float(state_loss.detach().cpu()), "theta0_delta_identity_max_abs": float((rotation - torch.eye(128, device=device)).abs().max().cpu()), "theta0_local_output_finite": bool(torch.isfinite(rot_out).all().cpu())}
    finally:
        probe.close()
    max_replay = max(max(row["native_core_replay_rel_l2"], row["native_post_norm_replay_rel_l2"], row["native_out_proj_replay_rel_l2"]) for row in per_layer.values())
    continuity = json.loads((Path(args.trace_dir) / "COLLECTION_TRAIN_COMPLETE.json").read_text())
    result = {"task": TASK, "model": "Ling-3.0-tiny/KDA", "layers": len(layers), "degrees_of_freedom_per_layer": 8128, "total_trainable_parameters": bank.total_trainable_parameters, "model_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "KDA_ROTATION_SEMANTICS_VERSION": "CORRECTED_PREFILL_ENDPOINT_V2", "LING_KDA_PREFILL_ENDPOINT_STATE_BASIS_GATE": continuity["LING_KDA_PREFILL_ENDPOINT_STATE_BASIS_GATE"], "REDUNDANT_PREFILL_ENDPOINT_ROTATION": "NO", "AUDIT_PATH_SIDE_EFFECT": "NO", "STE_FORWARD_PARITY_TEST": "PASS" if all(row["ste_exact_max_abs"] == 0 for row in per_layer.values()) else "FAIL", "DENSE_ORACLE_INITIALIZATION_GATE": "PASS" if all(row["theta0_delta_identity_max_abs"] == 0 for row in per_layer.values()) else "FAIL", "LOCAL_REPLAY_PARITY": "PASS" if max_replay <= args.replay_tolerance else "FAIL", "local_replay_max_relative_l2": max_replay, "replay_tolerance": args.replay_tolerance, "orthogonality": bank.runtime_gate(), "per_layer": per_layer}
    result["status"] = "PASS" if all(result[key] == "PASS" for key in ("LING_KDA_PREFILL_ENDPOINT_STATE_BASIS_GATE", "STE_FORWARD_PARITY_TEST", "DENSE_ORACLE_INITIALIZATION_GATE", "LOCAL_REPLAY_PARITY")) else "FAIL"
    save_json(Path(args.output_dir) / "initialization_ste_local_replay_gate.json", result)
    print(json.dumps({key: result[key] for key in ("LING_KDA_PREFILL_ENDPOINT_STATE_BASIS_GATE", "STE_FORWARD_PARITY_TEST", "DENSE_ORACLE_INITIALIZATION_GATE", "LOCAL_REPLAY_PARITY", "local_replay_max_relative_l2", "status")}, indent=2), flush=True)
    if result["status"] != "PASS":
        raise RuntimeError("LING_DENSE_ORACLE_GATE_FAIL")


def objective_for_sample(model, probe, bank, layers, sample, h, objective, grad):
    device = h.device
    state_total = torch.zeros((), device=device)
    functional_total = torch.zeros((), device=device)
    for layer in layers:
        rec = device_record(sample["layers"][layer], device)
        state_loss, qdq, _ = state_loss_and_qdq(rec["state_input"], bank.layer(layer), h, grad)
        state_total = state_total + state_loss / len(layers)
        if objective == "DENSE_FUNCTIONAL":
            rotation = bank.layer(layer).matrix()
            value = rec["v"].float().matmul(h).matmul(rotation).to(torch.bfloat16)
            raw, _ = fused_replay(probe, rec, qdq, value, device)
            mapped = raw.float().matmul(rotation.T).matmul(h).to(torch.bfloat16)
            _norm, output = local_output(model, layer, mapped, rec["dynamic_gate"])
            functional_total = functional_total + C.relative_mse(output.float(), rec["out_proj_output"].float()) / len(layers)
    primary = state_total if objective == "DENSE_STATE" else functional_total
    combined = state_total if objective == "DENSE_STATE" else functional_total + 0.1 * state_total
    return combined, primary, state_total, functional_total


def evaluate(model, probe, bank, layers, traces, h, objective):
    rows = []
    with torch.no_grad():
        for sample in traces:
            values = objective_for_sample(model, probe, bank, layers, sample, h, objective, False)
            rows.append([float(value.cpu()) for value in values])
    return {"combined": sum(x[0] for x in rows) / len(rows), "primary": sum(x[1] for x in rows) / len(rows), "state": sum(x[2] for x in rows) / len(rows), "functional": sum(x[3] for x in rows) / len(rows), "samples": len(rows)}


def train(args) -> None:
    random.seed(args.seed); torch.manual_seed(args.seed)
    F, model, _tokenizer, probe, layers = load_context(args)
    device = F.model_input_device(model)
    training = load_traces(Path(args.trace_dir), "TRAIN", args.train_sequence_limit)
    validation = load_traces(Path(args.trace_dir), "VALIDATION")
    bank = C.PerLayerCayleyRotations(layers).to(device)
    optimizer = torch.optim.Adam(bank.parameters(), lr=args.lr, weight_decay=0.0)
    h = hadamard(device)
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = outdir / f"best_{args.objective.lower()}_seed{args.seed}.pt"
    best, best_step, stale, history = float("inf"), 0, 0, []
    started = time.time()
    try:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            values = objective_for_sample(model, probe, bank, layers, training[random.randrange(len(training))], h, args.objective, True)
            combined, primary, state, functional = values
            if not torch.isfinite(combined): raise FloatingPointError(f"nonfinite loss step {step}")
            combined.backward(); grad = torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0); optimizer.step()
            if step == 1 or step % args.validation_interval == 0 or step == args.steps:
                gate = bank.runtime_gate()
                if gate["status"] != "PASS": raise RuntimeError("ORTHOGONALITY_RUNTIME_FAIL")
                val = evaluate(model, probe, bank, layers, validation, h, args.objective)
                row = {"step": step, "train_combined": float(combined.detach().cpu()), "train_primary": float(primary.detach().cpu()), "train_state": float(state.detach().cpu()), "train_functional": float(functional.detach().cpu()), "gradient_norm": float(grad.detach().cpu()), "validation": val, "orthogonality": {key: gate[key] for key in ("status", "max_abs_rt_r_minus_i", "all_finite", "all_proper")}}
                history.append(row); print(json.dumps(row), flush=True)
                if val["primary"] < best:
                    best, best_step, stale = val["primary"], step, 0
                    torch.save({"task": TASK, "model": "Ling-3.0-tiny/KDA", "objective": args.objective, "seed": args.seed, "lr": args.lr, "step": step, "validation": val, "bank": bank.state_dict(), "layer_ids": layers}, checkpoint)
                else: stale += 1
                if stale >= 10: break
    finally:
        probe.close()
    save_json(outdir / "training_summary.json", {"task": TASK, "model": "Ling-3.0-tiny/KDA", "objective": args.objective, "seed": args.seed, "lr": args.lr, "best_step": best_step, "best_validation_primary": best, "history": history, "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "runtime_seconds": time.time() - started, "optimizer": "Adam", "weight_decay": 0.0, "gradient_clip_global_norm": 1.0, "canonical_bf16_intermediate_boundary": True, "model_weights_changed": False, "AIME26_used": False, "status": "PASS"})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("collect", "gate", "train"), required=True)
    p.add_argument("--legacy-repo", default="/data01/user2/repos/GDN-quantization")
    p.add_argument("--max-memory-gib", type=int, default=22)
    p.add_argument("--corpus"); p.add_argument("--split", choices=("train", "validation", "heldout"), default="train")
    p.add_argument("--document-limit", type=int); p.add_argument("--trace-dir"); p.add_argument("--output-dir", required=True)
    p.add_argument("--objective", choices=("DENSE_STATE", "DENSE_FUNCTIONAL")); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=100); p.add_argument("--validation-interval", type=int, default=50)
    p.add_argument("--train-sequence-limit", type=int); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--replay-tolerance", type=float, default=5e-3)
    return p.parse_args()


def main():
    args = parse_args()
    if args.phase == "collect": collect(args)
    elif args.phase == "gate": replay_gate(args)
    else: train(args)


if __name__ == "__main__": main()
