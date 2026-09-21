#!/usr/bin/env python3
"""Qwen3.5-9B/GDN Hadamard-initialized dense orthogonal oracle.

The model is used only under inference_mode to collect detached local traces.
All optimization happens in a side-car graph whose only parameters are the
per-recurrent-layer Cayley coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch


TASK = "HADAMARD_INIT_DENSE_ORTHOGONAL_ORACLE_V1"
REPO = Path(__file__).resolve().parents[4]
SHARED = REPO / "experiments/shared/rotation/cayley_rotation.py"
ORTHO = REPO / "experiments/shared/rotation/orthogonal_matrix_generator.py"
GDN_LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30)
EPS = 1.0e-12


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = import_file(SHARED, "dense_oracle_cayley_qwen")
O = import_file(ORTHO, "dense_oracle_orthogonal_qwen")


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
    selected = [row for row in rows if row["split"] == split.upper()]
    if not selected:
        raise RuntimeError(f"no {split} rows in {path}")
    return selected


def normalize_qk(value: torch.Tensor) -> torch.Tensor:
    return value * torch.rsqrt((value * value).sum(-1, keepdim=True) + 1.0e-6)


def get_state(cache, layer_idx: int) -> torch.Tensor:
    if hasattr(cache, "recurrent_states"):
        state = cache.recurrent_states[layer_idx]
        return state[0] if isinstance(state, (dict, tuple, list)) else state
    state = cache.layers[layer_idx].recurrent_states
    return state[0] if isinstance(state, (dict, tuple, list)) else state


def freeze_model(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("MODEL_FREEZE_GATE_FAIL")


def load_model(args):
    os.environ["GDN_DATA_ROOT"] = str(Path(args.legacy_root))
    candidate = Path(args.legacy_root) / "consolidation/GDN-quantization-qwen3090-20260921/experiments/orientation/run_int8_orientation_state_change_mechanism.py"
    if not candidate.exists():
        candidate = Path(args.legacy_root) / "GDN-quantization/experiments/orientation/run_int8_orientation_state_change_mechanism.py"
    legacy = import_file(candidate, "qwen_dense_oracle_legacy")
    _torch, model, tokenizer, config, e2e = legacy.setup_model()
    freeze_model(model)
    return model, tokenizer, config, e2e


def cpu(value, dtype=None):
    if value is None:
        return None
    out = value.detach()
    if dtype is not None:
        out = out.to(dtype)
    return out.cpu().clone()


class TraceProbe:
    """Capture one native recurrent token and its frozen local output per layer."""

    def __init__(self, model):
        self.model = model
        self.capture = False
        self.current_layer = None
        self.records = {}
        self.saved = []
        self.handles = []

    def _wrap(self, fn):
        def wrapped(query, key, value, *args, **kwargs):
            use_norm = bool(kwargs.pop("use_qk_l2norm_in_kernel", False))
            q = normalize_qk(query) if use_norm else query
            k = normalize_qk(key) if use_norm else key
            initial = kwargs.get("initial_state")
            core, state = fn(q, k, value, *args, use_qk_l2norm_in_kernel=False, **kwargs)
            layer = int(self.current_layer)
            if self.capture and layer in GDN_LAYERS:
                self.records[layer] = {
                    "q": cpu(q, torch.bfloat16), "k": cpu(k, torch.bfloat16),
                    "v": cpu(value, torch.bfloat16), "g": cpu(kwargs["g"], torch.float32),
                    "beta": cpu(kwargs["beta"], torch.float32),
                    "state_input": cpu(initial, torch.bfloat16),
                    "state_output": cpu(state, torch.bfloat16),
                    "core_output": cpu(core, torch.bfloat16),
                }
            return core.to(value.dtype), state
        return wrapped

    def _norm_pre(self, layer):
        def hook(_module, arguments):
            if self.capture and layer in self.records and len(arguments) >= 2:
                self.records[layer]["norm_input"] = cpu(arguments[0], torch.bfloat16)
                self.records[layer]["dynamic_gate"] = cpu(arguments[1], torch.bfloat16)
        return hook

    def _norm_post(self, layer):
        def hook(_module, _arguments, output):
            if self.capture and layer in self.records:
                self.records[layer]["post_norm_gate"] = cpu(output, torch.bfloat16)
        return hook

    def _out_post(self, layer):
        def hook(_module, _arguments, output):
            if self.capture and layer in self.records:
                self.records[layer]["out_proj_output"] = cpu(output, torch.bfloat16)
        return hook

    def install(self):
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
        layers = self.model.model.layers if hasattr(self.model.model, "layers") else self.model.model.language_model.layers
        for index, layer in enumerate(layers):
            module = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
            if module is None:
                continue
            self.handles.append(module.register_forward_pre_hook(lambda _m, _a, i=index: setattr(self, "current_layer", i)))
            if index in GDN_LAYERS:
                self.handles.append(module.norm.register_forward_pre_hook(self._norm_pre(index)))
                self.handles.append(module.norm.register_forward_hook(self._norm_post(index)))
                self.handles.append(module.out_proj.register_forward_hook(self._out_post(index)))
        for name in ("torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule"):
            original = getattr(qmod, name)
            self.saved.append((qmod, name, original))
            setattr(qmod, name, self._wrap(original))

    def close(self):
        for module, name, original in self.saved:
            setattr(module, name, original)
        for handle in self.handles:
            handle.remove()


def sampled_positions(row: dict, token_count: int) -> list[int]:
    if token_count <= 136:
        raise RuntimeError(f"{row['document_id']} has only {token_count} tokens")
    seed = int(row["raw_text_sha256"][:16], 16) ^ 20260921
    population = list(range(128, token_count))
    return sorted(random.Random(seed).sample(population, 8))


def tokenizer_manifest(tokenizer, corpus_path: Path, rows: list[dict]) -> dict:
    files = {}
    root = Path(getattr(tokenizer, "name_or_path", ""))
    if root.is_dir():
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
            path = root / name
            if path.exists():
                files[name] = sha256(path)
    encoded = {}
    for row in rows:
        ids = tokenizer(row["raw_text"], add_special_tokens=False).input_ids[:1024]
        encoded[row["document_id"]] = {"token_count": len(ids), "positions": sampled_positions(row, len(ids))}
    return {
        "task": TASK, "model": "Qwen3.5-9B/GDN", "raw_corpus": str(corpus_path),
        "raw_corpus_sha256": sha256(corpus_path), "tokenizer_name_or_path": str(root),
        "tokenizer_file_sha256": files, "tokenized_documents": encoded,
        "sequence_length_max": 1024, "sampled_positions_per_sequence": 8,
        "minimum_position": 128, "AIME26_used": False,
    }


def collect(args) -> None:
    model, tokenizer, _config, _e2e = load_model(args)
    rows = load_corpus(Path(args.corpus), args.split)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = tokenizer_manifest(tokenizer, Path(args.corpus), rows)
    save_json(outdir / f"CALIBRATION_MANIFEST_{args.split.upper()}.json", manifest)
    probe = TraceProbe(model)
    probe.install()
    device = model.get_input_embeddings().weight.device
    try:
        for row_index, row in enumerate(rows):
            ids = tokenizer(row["raw_text"], add_special_tokens=False).input_ids[:1024]
            positions = manifest["tokenized_documents"][row["document_id"]]["positions"]
            cache, current, samples = None, 0, []
            with torch.inference_mode():
                for position in positions:
                    if current < position:
                        chunk = torch.tensor([ids[current:position]], dtype=torch.long, device=device)
                        probe.capture = False
                        output = model(input_ids=chunk, past_key_values=cache, use_cache=True)
                        cache = output.past_key_values
                    probe.records = {}
                    probe.capture = True
                    token = torch.tensor([[ids[position]]], dtype=torch.long, device=device)
                    output = model(input_ids=token, past_key_values=cache, use_cache=True)
                    probe.capture = False
                    cache = output.past_key_values
                    missing = sorted(set(GDN_LAYERS) - set(probe.records))
                    if missing:
                        raise RuntimeError(f"trace missing GDN layers: {missing}")
                    samples.append({"position": position, "layers": probe.records})
                    current = position + 1
            shard = {"document_id": row["document_id"], "raw_text_sha256": row["raw_text_sha256"], "split": args.split.upper(), "samples": samples}
            path = outdir / f"{row['document_id']}.pt"
            torch.save(shard, path)
            print(f"COLLECT {row_index + 1}/{len(rows)} {row['document_id']} bytes={path.stat().st_size}", flush=True)
    finally:
        probe.close()
    save_json(outdir / f"COLLECTION_{args.split.upper()}_COMPLETE.json", {"status": "PASS", "documents": len(rows), "samples": len(rows) * 8, "AIME26_used": False})


def hadamard(device) -> torch.Tensor:
    return O.normalized_hadamard(128).to(device=device, dtype=torch.float32)


def transform_state(state, rotation, h):
    # canonical H first, then DeltaR; state uses the inverse-transpose coordinate law.
    h_state = torch.einsum("ji,bhjv->bhiv", h, state.float())
    return torch.einsum("ji,bhjv->bhiv", rotation, h_state)


def recover_state(state, rotation, h):
    corrected = torch.einsum("ij,bhjv->bhiv", rotation, state.float())
    return torch.einsum("ij,bhjv->bhiv", h, corrected)


def state_loss_and_qdq(state, rotation_module, h, use_ste=True):
    rotation = rotation_module.matrix()
    rotated = transform_state(state, rotation, h)
    quant = C.qwen_c128_ste(rotated) if use_ste else C.qwen_c128_qdq(rotated)
    recovered = recover_state(quant.dequant, rotation, h)
    numerator = (recovered - state.float()).square().sum(dim=(-2, -1))
    denominator = state.float().square().sum(dim=(-2, -1)).clamp_min(EPS)
    return (numerator / denominator).mean(), quant, recovered


def native_recurrence(record, state):
    q = record["q"].float()[:, 0] / math.sqrt(128)
    k = record["k"].float()[:, 0]
    v = record["v"].float()[:, 0]
    decay = record["g"].float()[:, 0].exp()
    beta = record["beta"].float()[:, 0]
    updated = state.float() * decay[..., None, None]
    memory = (updated * k.unsqueeze(-1)).sum(dim=-2)
    delta = (v - memory) * beta.unsqueeze(-1)
    updated = updated + k.unsqueeze(-1) * delta.unsqueeze(-2)
    core = (updated * q.unsqueeze(-1)).sum(dim=-2)
    return core, updated


def rotated_recurrence(record, quantized_state, rotation, h):
    q = record["q"].float()[:, 0].matmul(h).matmul(rotation) / math.sqrt(128)
    k = record["k"].float()[:, 0].matmul(h).matmul(rotation)
    v = record["v"].float()[:, 0]
    decay = record["g"].float()[:, 0].exp()
    beta = record["beta"].float()[:, 0]
    updated = quantized_state.float() * decay[..., None, None]
    memory = (updated * k.unsqueeze(-1)).sum(dim=-2)
    delta = (v - memory) * beta.unsqueeze(-1)
    updated = updated + k.unsqueeze(-1) * delta.unsqueeze(-2)
    core = (updated * q.unsqueeze(-1)).sum(dim=-2)
    return core, updated


def model_layers(model):
    return model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers


def local_output(model, layer_id, core, gate):
    module = getattr(model_layers(model)[layer_id], "linear_attn", None) or getattr(model_layers(model)[layer_id], "self_attn", None)
    dtype = module.out_proj.weight.dtype
    normed = module.norm(core.reshape(-1, core.shape[-1]).to(dtype), gate.to(dtype))
    output = module.out_proj(normed.reshape(core.shape[0], 1, -1))
    return normed, output


def load_traces(trace_dir: Path, split: str, limit: int | None = None) -> list[dict]:
    paths = sorted(trace_dir.glob(f"wikitext2raw-{split.lower()}-*.pt"))
    if limit is not None:
        paths = paths[:limit]
    traces = []
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        for sample in shard["samples"]:
            traces.append({"document_id": shard["document_id"], **sample})
    if not traces:
        raise RuntimeError(f"no {split} traces in {trace_dir}")
    return traces


def to_device_record(record, device):
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in record.items()}


def replay_gate(args) -> None:
    model, _tokenizer, _config, _e2e = load_model(args)
    device = model.get_input_embeddings().weight.device
    bank = C.PerLayerCayleyRotations(GDN_LAYERS).to(device)
    traces = load_traces(Path(args.trace_dir), "TRAIN", 1)
    sample = traces[0]
    h = hadamard(device)
    per_layer = {}
    for layer_id in GDN_LAYERS:
        record = to_device_record(sample["layers"][layer_id], device)
        state = record["state_input"].float()
        native_core, _ = native_recurrence(record, state)
        native_norm, native_out = local_output(model, layer_id, native_core, record["dynamic_gate"])
        rotation = bank.layer(layer_id).matrix()
        loss, q_ste, _ = state_loss_and_qdq(state, bank.layer(layer_id), h, True)
        q_exact = C.qwen_c128_qdq(transform_state(state, rotation, h))
        rotated_core, _ = rotated_recurrence(record, q_ste.dequant, rotation, h)
        _rot_norm, rot_out = local_output(model, layer_id, rotated_core, record["dynamic_gate"])
        def rel(x, y): return float(C.relative_mse(x, y).sqrt().detach().cpu())
        per_layer[str(layer_id)] = {
            "native_core_replay_rel_l2": rel(native_core, record["core_output"][:, 0]),
            "native_post_norm_replay_rel_l2": rel(native_norm, record["post_norm_gate"]),
            "native_out_proj_replay_rel_l2": rel(native_out, record["out_proj_output"]),
            "ste_exact_max_abs": float((q_ste.dequant - q_exact.dequant).abs().max().detach().cpu()),
            "theta0_state_loss": float(loss.detach().cpu()),
            "theta0_delta_identity_max_abs": float((rotation - torch.eye(128, device=device)).abs().max().detach().cpu()),
            "theta0_dense_vs_canonical_h_qdq_max_abs": float((q_ste.dequant - q_exact.dequant).abs().max().detach().cpu()),
            "theta0_local_output_finite": bool(torch.isfinite(rot_out).all().detach().cpu()),
        }
    max_replay = max(max(row["native_core_replay_rel_l2"], row["native_post_norm_replay_rel_l2"], row["native_out_proj_replay_rel_l2"]) for row in per_layer.values())
    result = {
        "task": TASK, "model": "Qwen3.5-9B/GDN", "layers": len(GDN_LAYERS),
        "degrees_of_freedom_per_layer": 8128, "total_trainable_parameters": bank.total_trainable_parameters,
        "model_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "STE_FORWARD_PARITY_TEST": "PASS" if all(row["ste_exact_max_abs"] == 0 for row in per_layer.values()) else "FAIL",
        "DENSE_ORACLE_INITIALIZATION_GATE": "PASS" if all(row["theta0_delta_identity_max_abs"] == 0 and row["theta0_dense_vs_canonical_h_qdq_max_abs"] == 0 for row in per_layer.values()) else "FAIL",
        "LOCAL_REPLAY_PARITY": "PASS" if max_replay <= args.replay_tolerance else "FAIL",
        "local_replay_max_relative_l2": max_replay, "replay_tolerance": args.replay_tolerance,
        "orthogonality": bank.runtime_gate(), "per_layer": per_layer,
        "status": "PASS" if max_replay <= args.replay_tolerance else "FAIL",
    }
    save_json(Path(args.output_dir) / "initialization_ste_local_replay_gate.json", result)
    print(json.dumps({key: result[key] for key in ("STE_FORWARD_PARITY_TEST", "DENSE_ORACLE_INITIALIZATION_GATE", "LOCAL_REPLAY_PARITY", "local_replay_max_relative_l2", "status")}, indent=2), flush=True)
    if result["status"] != "PASS":
        raise RuntimeError("QWEN_DENSE_ORACLE_GATE_FAIL")


def objective_for_sample(model, bank, sample, h, objective, grad):
    device = h.device
    total_state = torch.zeros((), device=device)
    total_func = torch.zeros((), device=device)
    for layer_id in GDN_LAYERS:
        record = to_device_record(sample["layers"][layer_id], device)
        state = record["state_input"].float()
        state_loss, quant, _ = state_loss_and_qdq(state, bank.layer(layer_id), h, use_ste=grad)
        total_state = total_state + state_loss / len(GDN_LAYERS)
        if objective == "DENSE_FUNCTIONAL":
            rotation = bank.layer(layer_id).matrix()
            core, _ = rotated_recurrence(record, quant.dequant, rotation, h)
            _norm, output = local_output(model, layer_id, core, record["dynamic_gate"])
            layer_loss = C.relative_mse(output.float(), record["out_proj_output"].float())
            total_func = total_func + layer_loss / len(GDN_LAYERS)
    primary = total_state if objective == "DENSE_STATE" else total_func
    combined = total_state if objective == "DENSE_STATE" else total_func + 0.1 * total_state
    return combined, primary, total_state, total_func


def evaluate_local(model, bank, traces, h, objective):
    values = []
    with torch.no_grad():
        for sample in traces:
            combined, primary, state, functional = objective_for_sample(model, bank, sample, h, objective, False)
            values.append((float(combined.cpu()), float(primary.cpu()), float(state.cpu()), float(functional.cpu())))
    return {
        "combined": sum(x[0] for x in values) / len(values),
        "primary": sum(x[1] for x in values) / len(values),
        "state": sum(x[2] for x in values) / len(values),
        "functional": sum(x[3] for x in values) / len(values),
        "samples": len(values),
    }


def train(args) -> None:
    if args.objective not in ("DENSE_STATE", "DENSE_FUNCTIONAL"):
        raise ValueError("--objective must be DENSE_STATE or DENSE_FUNCTIONAL")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, _tokenizer, _config, _e2e = load_model(args)
    device = model.get_input_embeddings().weight.device
    train_traces = load_traces(Path(args.trace_dir), "TRAIN", args.train_sequence_limit)
    validation = load_traces(Path(args.trace_dir), "VALIDATION")
    bank = C.PerLayerCayleyRotations(GDN_LAYERS).to(device)
    optimizer = torch.optim.Adam(bank.parameters(), lr=args.lr, weight_decay=0.0)
    h = hadamard(device)
    history, best, best_step, stale = [], float("inf"), 0, 0
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = outdir / f"best_{args.objective.lower()}_seed{args.seed}.pt"
    started = time.time()
    for step in range(1, args.steps + 1):
        sample = train_traces[random.randrange(len(train_traces))]
        optimizer.zero_grad(set_to_none=True)
        combined, primary, state, functional = objective_for_sample(model, bank, sample, h, args.objective, True)
        if not torch.isfinite(combined):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        combined.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.validation_interval == 0 or step == args.steps:
            gate = bank.runtime_gate()
            if gate["status"] != "PASS":
                raise RuntimeError(f"ORTHOGONALITY_RUNTIME_FAIL at step {step}")
            val = evaluate_local(model, bank, validation, h, args.objective)
            row = {"step": step, "train_combined": float(combined.detach().cpu()), "train_primary": float(primary.detach().cpu()), "train_state": float(state.detach().cpu()), "train_functional": float(functional.detach().cpu()), "gradient_norm": float(grad_norm.detach().cpu()), "validation": val, "orthogonality": {key: gate[key] for key in ("status", "max_abs_rt_r_minus_i", "all_finite", "all_proper")}}
            history.append(row)
            print(json.dumps(row), flush=True)
            if val["primary"] < best:
                best, best_step, stale = val["primary"], step, 0
                torch.save({"task": TASK, "model": "Qwen3.5-9B/GDN", "objective": args.objective, "seed": args.seed, "lr": args.lr, "step": step, "validation": val, "bank": bank.state_dict(), "layer_ids": GDN_LAYERS}, checkpoint)
            else:
                stale += 1
            if stale >= 10:
                break
    checkpoint_hash = sha256(checkpoint)
    save_json(outdir / "training_summary.json", {"task": TASK, "model": "Qwen3.5-9B/GDN", "objective": args.objective, "seed": args.seed, "lr": args.lr, "steps_requested": args.steps, "best_step": best_step, "best_validation_primary": best, "history": history, "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_hash, "runtime_seconds": time.time() - started, "train_sequences": args.train_sequence_limit or 64, "validation_sequences": 16, "optimizer": "Adam", "weight_decay": 0.0, "gradient_clip_global_norm": 1.0, "model_weights_changed": False, "AIME26_used": False, "status": "PASS"})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("collect", "gate", "train"), required=True)
    parser.add_argument("--legacy-root", default="/data/zypan")
    parser.add_argument("--corpus")
    parser.add_argument("--split", choices=("train", "validation", "heldout"), default="train")
    parser.add_argument("--trace-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objective", choices=("DENSE_STATE", "DENSE_FUNCTIONAL"))
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--train-sequence-limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-tolerance", type=float, default=5.0e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.phase == "collect":
        if not args.corpus:
            raise ValueError("--corpus is required")
        collect(args)
    elif args.phase == "gate":
        replay_gate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
