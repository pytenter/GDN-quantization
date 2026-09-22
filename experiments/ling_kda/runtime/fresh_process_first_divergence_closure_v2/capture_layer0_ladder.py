#!/usr/bin/env python3
"""Capture a compact, ordered layer-0 Ling/KDA prefill ladder.

Tensor payloads are temporary and exist only so a second process can compute an
exact difference.  The durable record contains hashes and scalar summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "repeatability_closure_v1" / "run_ling_repeatability.py"
ORDER = (
    "input_token_ids",
    "embedding_output",
    "layer0_input_hidden",
    "pre_norm_output",
    "q_projection_raw",
    "q_short_conv_output",
    "k_projection_raw",
    "k_short_conv_output",
    "v_projection_raw",
    "v_short_conv_output",
    "decay_projection_a_raw",
    "decay_projection_b_raw",
    "beta_projection_raw",
    "final_q_driver",
    "final_k_driver",
    "final_v_driver_native",
    "final_v_driver_rotated",
    "final_beta_driver",
    "final_log_decay_driver",
    "internal_q_l2norm_output",
    "internal_k_l2norm_output",
    "internal_g_chunk_cumsum_output",
    "internal_w_output",
    "internal_u_output",
    "internal_qg_output",
    "internal_kg_output",
    "internal_v_new_output",
    "internal_Aqk_output",
    "internal_Akk_output",
    "internal_intermediate_state_h",
    "chunk_kda_raw_output",
    "chunk_kda_mapped_output",
    "chunk_kda_final_state",
    "prefill_cache_layer0",
    "target_gate_projection_output",
    "target_o_norm_weight",
    "target_o_norm_input",
    "target_o_norm_gate_input",
    "target_o_norm_output",
    "target_o_proj_input",
    "target_o_proj_output",
    "target_attention_output",
    "layer0_kda_rmsnorm_input",
    "layer0_dynamic_gate_input",
    "layer0_dynamic_gate_output",
    "layer0_rmsnorm_scaled_output",
    "layer0_out_proj_input",
    "layer0_out_proj_output",
    "layer0_attention_output",
    "layer0_post_attention_rmsnorm_input",
    "layer0_post_attention_rmsnorm_output",
    "layer0_mlp_output",
    "layer0_output_hidden",
    "layer1_input_hidden",
    "layer1_pre_norm_output",
    "layer1_attention_input",
    "layer1_attention_output",
    "layer1_post_attention_rmsnorm_input",
    "layer1_post_attention_rmsnorm_output",
    "layer1_mlp_output",
    "layer1_output_hidden",
    "layer2_input_hidden",
)


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V1MOD = import_file(V1, "ling_repeatability_v1_for_first_divergence_v2")
ISOLATION = import_file(HERE / "isolate_chunk_kda.py", "ling_chunk_kda_isolation_helpers_v2")


def bytes_of(value: torch.Tensor) -> bytes:
    cpu = value.detach().contiguous().cpu()
    return cpu.view(torch.uint8).numpy().tobytes()


def signature(value: torch.Tensor) -> dict:
    cpu = value.detach().contiguous().cpu()
    numeric = cpu.double()
    finite_mask = torch.isfinite(numeric)
    finite = numeric[finite_mask]
    return {
        "sha256": hashlib.sha256(bytes_of(cpu)).hexdigest(),
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "device": str(value.device),
        "min": float(finite.min()) if finite.numel() else None,
        "max": float(finite.max()) if finite.numel() else None,
        "mean": float(finite.mean()) if finite.numel() else None,
        "norm": float(torch.linalg.vector_norm(finite)) if finite.numel() else None,
        "nonfinite": int((~finite_mask).sum()),
    }


def canonical_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def parameter_hash(module) -> dict:
    digest = hashlib.sha256()
    count = 0
    names = []
    for name, value in sorted(module.named_parameters(recurse=True)):
        digest.update(name.encode())
        digest.update(bytes_of(value))
        count += value.numel()
        names.append(name)
    return {"sha256": digest.hexdigest(), "parameter_count": count, "parameter_names": names}


def file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clone(value):
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    return value.detach().contiguous().cpu().clone() if torch.is_tensor(value) else None


def apply_diagnostic_gated_rmsnorm_config() -> bool:
    if os.environ.get("LING_GATED_RMSNORM_FIXED_CONFIG") != "BT16_W8":
        return False
    import triton
    import fla.modules.fused_norm_gate as module
    current = module.layer_norm_gated_fwd_kernel
    while current is not None:
        if hasattr(current, "configs"):
            current.configs = [triton.Config({"BT": 16}, num_warps=8, num_stages=3)]
            if isinstance(getattr(current, "cache", None), dict):
                current.cache.clear()
        current = getattr(current, "fn", None)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-repo", default="/data01/user2/repos/GDN-quantization")
    parser.add_argument("--max-memory-gib", type=int, default=22)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--state-checkpoint", required=True)
    parser.add_argument("--functional-checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--same-process-repeats", type=int, default=1)
    parser.add_argument("--target-layer", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    gated_rmsnorm_fixed = apply_diagnostic_gated_rmsnorm_config()
    F, model, tokenizer, layers, device, h, _dm, _rm, checkpoints = V1MOD.load_context(args)
    documents = sorted(V1MOD.R.load_corpus(Path(args.corpus), "HELDOUT"), key=lambda row: int(row["panel_index"]))
    document = documents[0]
    ids = tokenizer(document["raw_text"], add_special_tokens=False).input_ids[:1024]
    prefix = torch.tensor([ids[:128]], dtype=torch.long, device=device)
    mask = torch.ones_like(prefix)
    target_layer = int(args.target_layer)
    target_module = model.model.layers[target_layer]
    attention = target_module.attention
    layer1 = model.model.layers[1]
    layer2 = model.model.layers[2]
    captures: dict[str, torch.Tensor] = {"input_token_ids": prefix.detach().cpu().clone()}
    handles = []

    def post(name):
        def hook(_module, _inputs, output):
            value = clone(output)
            if value is not None:
                captures[name] = value
        return hook

    def pre(name):
        def hook(_module, inputs):
            value = clone(inputs[0] if inputs else None)
            if value is not None:
                captures[name] = value
        return hook

    def o_norm_pre(_module, inputs):
        if inputs:
            value = clone(inputs[0])
            if value is not None:
                captures["target_o_norm_input"] = value
        if len(inputs) >= 2:
            value = clone(inputs[1])
            if value is not None:
                captures["target_o_norm_gate_input"] = value

    handles.append(model.get_input_embeddings().register_forward_hook(post("embedding_output")))
    handles.append(target_module.register_forward_pre_hook(pre("layer0_input_hidden")))
    handles.append(target_module.input_layernorm.register_forward_hook(post("pre_norm_output")))
    if hasattr(attention, "g_proj"):
        handles.append(attention.g_proj.register_forward_hook(post("target_gate_projection_output")))
    handles.append(attention.o_norm.register_forward_pre_hook(o_norm_pre))
    handles.append(attention.o_norm.register_forward_hook(post("target_o_norm_output")))
    handles.append(attention.o_proj.register_forward_pre_hook(pre("target_o_proj_input")))
    handles.append(attention.o_proj.register_forward_hook(post("target_o_proj_output")))
    handles.append(attention.register_forward_hook(post("target_attention_output")))
    handles.append(layer1.register_forward_pre_hook(pre("layer1_input_hidden")))
    handles.append(layer1.register_forward_hook(post("layer1_output_hidden")))
    handles.append(layer2.register_forward_pre_hook(pre("layer2_input_hidden")))
    handles.append(layer1.input_layernorm.register_forward_hook(post("layer1_pre_norm_output")))
    handles.append(layer1.attention.register_forward_pre_hook(pre("layer1_attention_input")))
    handles.append(layer1.attention.register_forward_hook(post("layer1_attention_output")))
    handles.append(layer1.post_attention_layernorm.register_forward_pre_hook(pre("layer1_post_attention_rmsnorm_input")))
    handles.append(layer1.post_attention_layernorm.register_forward_hook(post("layer1_post_attention_rmsnorm_output")))
    handles.append(layer1.mlp.register_forward_hook(post("layer1_mlp_output")))
    module_map = {
        "q_proj": "q_projection_raw", "q_conv1d": "q_short_conv_output",
        "k_proj": "k_projection_raw", "k_conv1d": "k_short_conv_output",
        "v_proj": "v_projection_raw", "v_conv1d": "v_short_conv_output",
        "f_a_proj": "decay_projection_a_raw", "f_b_proj": "decay_projection_b_raw",
        "f_proj": "decay_projection_b_raw", "b_proj": "beta_projection_raw",
    }
    for module_name, capture_name in module_map.items():
        module = getattr(attention, module_name, None)
        if module is not None:
            handles.append(module.register_forward_hook(post(capture_name)))

    probe = F.FullPathForensicProbe(h)
    installed_layers = tuple(int(x) for x in probe.install(model))
    if target_layer not in installed_layers:
        raise RuntimeError(f"target layer {target_layer} is not KDA: {installed_layers}")
    import fla.ops.kda.chunk as chunk_module
    import fla.ops.kda.chunk_fwd as chunk_fwd_module
    original_l2norm = chunk_module.l2norm_fwd
    original_cumsum = chunk_fwd_module.chunk_local_cumsum
    original_chunk_kda_fwd = chunk_module.chunk_kda_fwd
    l2_calls = 0

    def capture_l2norm(*call_args, **call_kwargs):
        nonlocal l2_calls
        result = original_l2norm(*call_args, **call_kwargs)
        if probe.branch == "RP" and probe.current_layer == target_layer:
            name = "internal_q_l2norm_output" if l2_calls % 2 == 0 else "internal_k_l2norm_output"
            captures[name] = clone(result[0])
            l2_calls += 1
        return result

    def capture_cumsum(*call_args, **call_kwargs):
        result = original_cumsum(*call_args, **call_kwargs)
        if probe.branch == "RP" and probe.current_layer == target_layer:
            captures["internal_g_chunk_cumsum_output"] = clone(result)
        return result

    def capture_chunk_kda_fwd(*call_args, **call_kwargs):
        result = original_chunk_kda_fwd(*call_args, **call_kwargs)
        if probe.branch == "RP" and probe.current_layer == target_layer:
            for name, index in (
                ("internal_g_chunk_cumsum_output", 2),
                ("internal_w_output", 5),
                ("internal_u_output", 6),
                ("internal_qg_output", 7),
                ("internal_kg_output", 8),
                ("internal_v_new_output", 9),
                ("internal_Aqk_output", 3),
                ("internal_Akk_output", 4),
                ("internal_intermediate_state_h", 10),
            ):
                value = clone(result[index])
                if value is not None:
                    captures[name] = value
        return result

    chunk_module.l2norm_fwd = capture_l2norm
    chunk_fwd_module.chunk_local_cumsum = capture_cumsum
    chunk_module.chunk_kda_fwd = capture_chunk_kda_fwd
    same_process = []
    try:
        for repeat in range(args.same_process_repeats):
            if repeat:
                captures.clear()
                captures["input_token_ids"] = prefix.detach().cpu().clone()
                l2_calls = 0
            probe.begin(F.H.plan("RP", "rotated", False), repeat)
            with torch.inference_mode():
                output = model(
                    input_ids=prefix,
                    attention_mask=mask,
                    cache_position=torch.arange(128, device=device),
                    use_cache=True,
                )
            probe.end()
            record = probe.records["RP"][target_layer]
            captures.update({
                "final_q_driver": clone(record["q"]),
                "final_k_driver": clone(record["k"]),
                "final_v_driver_native": clone(record["v_semantic"]),
                "final_v_driver_rotated": clone(record["v"]),
                "final_beta_driver": clone(record["beta"]),
                "final_log_decay_driver": clone(record["log_decay"]),
                "chunk_kda_raw_output": clone(record["raw_output"]),
                "chunk_kda_mapped_output": clone(record["output"]),
                "chunk_kda_final_state": clone(record["final_state"]),
                "prefill_cache_layer0": clone(F.BASE.cache_stack(output.past_key_values, installed_layers)[target_layer]),
                "target_o_norm_weight": clone(attention.o_norm.weight),
            })
            path = probe.path_tensors["RP"][target_layer]
            for source, target in {
                "rmsnorm_input": "layer0_kda_rmsnorm_input",
                "dynamic_gate_input": "layer0_dynamic_gate_input",
                "dynamic_gate_output": "layer0_dynamic_gate_output",
                "rmsnorm_scaled_output": "layer0_rmsnorm_scaled_output",
                "out_proj_input": "layer0_out_proj_input",
                "out_proj_output": "layer0_out_proj_output",
                "attention_output": "layer0_attention_output",
                "post_attention_rmsnorm_input": "layer0_post_attention_rmsnorm_input",
                "post_attention_rmsnorm_output": "layer0_post_attention_rmsnorm_output",
                "mlp_output": "layer0_mlp_output",
                "layer_output_hidden": "layer0_output_hidden",
            }.items():
                value = clone(path.get(source))
                if value is not None:
                    captures[target] = value
            same_process.append({name: signature(captures[name]) for name in ORDER if name in captures})
    finally:
        for handle in handles:
            handle.remove()
        chunk_module.l2norm_fwd = original_l2norm
        chunk_fwd_module.chunk_local_cumsum = original_cumsum
        chunk_module.chunk_kda_fwd = original_chunk_kda_fwd
        probe.close()

    model_root = Path(str(getattr(model.config, "_name_or_path", "")))
    tok_root = Path(str(getattr(tokenizer, "name_or_path", "")))
    provenance = {
        "model_config": canonical_hash(model.config.to_dict()),
        "tokenizer_config": canonical_hash(getattr(tokenizer, "init_kwargs", {})),
        "model_config_file": file_hash(model_root / "config.json"),
        "tokenizer_config_file": file_hash(tok_root / "tokenizer_config.json"),
        "target_layer_index": target_layer,
        "target_layer": parameter_hash(target_module),
        "input_layernorm": parameter_hash(target_module.input_layernorm),
        "q_proj": parameter_hash(attention.q_proj),
        "k_proj": parameter_hash(attention.k_proj),
        "v_proj": parameter_hash(attention.v_proj),
        "b_proj": parameter_hash(attention.b_proj),
        "f_projection": parameter_hash(attention.f_proj if hasattr(attention, "f_proj") else attention.f_a_proj),
        "q_conv1d": parameter_hash(attention.q_conv1d),
        "k_conv1d": parameter_hash(attention.k_conv1d),
        "v_conv1d": parameter_hash(attention.v_conv1d),
        "rotation_checkpoint": checkpoints,
        "input_token_ids": signature(captures["input_token_ids"])["sha256"],
    }
    record = {
        "task": "LING_FRESH_PROCESS_FIRST_DIVERGENCE_CLOSURE_V2",
        "run_id": args.run_id,
        "pid": os.getpid(),
        "process_started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "environment": {key: os.environ.get(key) for key in (
            "CUBLAS_WORKSPACE_CONFIG", "CUDA_LAUNCH_BLOCKING", "PYTORCH_CUDA_ALLOC_CONF",
            "TRITON_CACHE_DIR", "FLA_CACHE_MODE", "FLA_CONFIG_DIR", "LING_GATED_RMSNORM_FIXED_CONFIG",
        )},
        "diagnostic_gated_rmsnorm_fixed_config_applied": gated_rmsnorm_fixed,
        "document_id": document["document_id"],
        "target_layer": target_layer,
        "raw_text_sha256": document["raw_text_sha256"],
        "token_ids_sha256": signature(captures["input_token_ids"])["sha256"],
        "condition": "Hadamard",
        "basis": "rotated",
        "packed_prefill": True,
        "capture_order": [name for name in ORDER if name in captures],
        "tensor_signatures": {name: signature(captures[name]) for name in ORDER if name in captures},
        "same_process_signatures": same_process,
        "autotune": ISOLATION.autotune_snapshot(),
        "provenance": provenance,
        "runtime_seconds": time.time() - started,
        "KDA_ROTATION_SEMANTICS_VERSION": "CORRECTED_PREFILL_ENDPOINT_V2",
        "REDUNDANT_PREFILL_ENDPOINT_ROTATION": "NO",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"ladder_run_{args.run_id:02d}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    torch.save({"captures": captures}, args.output_dir / f"ladder_run_{args.run_id:02d}.temporary.pt")
    print(json.dumps({"run_id": args.run_id, "capture_order": record["capture_order"], "runtime_seconds": record["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
