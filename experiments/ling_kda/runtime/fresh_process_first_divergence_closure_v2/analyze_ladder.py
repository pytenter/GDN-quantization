#!/usr/bin/env python3
"""Analyze fresh-process ladder captures and retain only compact evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def exact_diff(values: list[torch.Tensor]) -> dict:
    reference = values[0].double()
    comparisons = []
    for index, value in enumerate(values[1:], 1):
        current = value.double()
        delta = current - reference
        differing = delta != 0
        locations = differing.nonzero()
        comparisons.append({
            "left_run": 0,
            "right_run": index,
            "bitwise_equal": bool(torch.equal(values[0], value)),
            "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
            "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference).clamp_min(1e-12)),
            "different_elements": int(differing.sum()),
            "first_different_index": None if not locations.numel() else [int(x) for x in locations[0].tolist()],
        })
    return {
        "all_five_bitwise_equal": all(item["bitwise_equal"] for item in comparisons),
        "max_abs_across_run0_pairs": max(item["max_abs"] for item in comparisons),
        "max_relative_l2_across_run0_pairs": max(item["relative_l2"] for item in comparisons),
        "max_different_elements_across_run0_pairs": max(item["different_elements"] for item in comparisons),
        "comparisons": comparisons,
    }


def selected_autotune_configs(record: dict) -> dict:
    selected = {}
    for name, row in record.get("autotune", {}).items():
        cache = row.get("cache", {})
        if cache:
            selected[name] = {"best_config": row.get("best_config"), "cache": cache}
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-fixed-input", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(path.read_text()) for path in sorted(args.input_dir.glob("ladder_run_*.json"))]
    payload_paths = sorted(args.input_dir.glob("ladder_run_*.temporary.pt"))
    if len(records) != 5 or len(payload_paths) != 5:
        raise RuntimeError(f"expected 5 records and payloads, got {len(records)} and {len(payload_paths)}")
    provenance = [record["provenance"] for record in records]
    provenance_match = all(value == provenance[0] for value in provenance[1:])
    order = records[0]["capture_order"]
    payloads = [torch.load(path, map_location="cpu", weights_only=False)["captures"] for path in payload_paths]
    comparisons = {name: exact_diff([payload[name] for payload in payloads]) for name in order}
    first = next((name for name in order if not comparisons[name]["all_five_bitwise_equal"]), None)
    if first is None:
        operator = None
        fixed_input_name = None
    else:
        index = order.index(first)
        fixed_input_name = order[index - 1] if index else None
        operator_map = {
            "q_projection_raw": "layer0.attention.q_proj/torch.nn.Linear",
            "q_short_conv_output": "layer0.attention.q_conv1d/ShortConvolution",
            "k_projection_raw": "layer0.attention.k_proj/torch.nn.Linear",
            "k_short_conv_output": "layer0.attention.k_conv1d/ShortConvolution",
            "v_projection_raw": "layer0.attention.v_proj/torch.nn.Linear",
            "v_short_conv_output": "layer0.attention.v_conv1d/ShortConvolution",
            "decay_projection_a_raw": "layer0.attention.f_a_proj/torch.nn.Linear",
            "decay_projection_b_raw": "layer0.attention.f_b_proj/torch.nn.Linear",
            "beta_projection_raw": "layer0.attention.b_proj/torch.nn.Linear",
            "internal_q_l2norm_output": "fla.modules.l2norm_fwd",
            "internal_k_l2norm_output": "fla.modules.l2norm_fwd",
            "internal_g_chunk_cumsum_output": "fla.ops.utils.chunk_local_cumsum",
            "internal_w_output": "fla.ops.kda.chunk_kda_fwd_intra/w formation",
            "internal_u_output": "fla.ops.kda.chunk_kda_fwd_intra/u formation",
            "internal_qg_output": "fla.ops.kda.chunk_kda_fwd_intra/qg formation",
            "internal_kg_output": "fla.ops.kda.chunk_kda_fwd_intra/kg formation",
            "internal_v_new_output": "fla.ops.kda.chunk_kda_fwd_intra/v update",
            "internal_Aqk_output": "fla.ops.kda.chunk_kda_fwd/Aqk formation",
            "internal_Akk_output": "fla.ops.kda.chunk_kda_fwd/Akk formation",
            "internal_intermediate_state_h": "fla.ops.kda.chunk_kda_fwd/history formation",
            "chunk_kda_raw_output": "fla.ops.gla.chunk.chunk_gla_fwd_kernel_o",
            "chunk_kda_mapped_output": "KDA value-basis map-back",
            "chunk_kda_final_state": "fla.ops.kda.chunk_kda",
            "target_gate_projection_output": "target attention g_proj/torch.nn.Linear",
            "target_o_norm_output": "target attention o_norm/gated RMSNorm",
            "target_o_proj_output": "target attention o_proj/torch.nn.Linear",
            "target_attention_output": "target attention return boundary",
            "layer0_rmsnorm_scaled_output": "layer0.attention.o_norm",
            "layer0_out_proj_output": "layer0.attention.o_proj/torch.nn.Linear",
            "layer0_attention_output": "layer0.attention",
            "layer0_post_attention_rmsnorm_output": "layer0.post_attention_layernorm",
            "layer0_mlp_output": "layer0.mlp",
            "layer0_output_hidden": "layer0 residual merge",
            "layer1_pre_norm_output": "layer1.input_layernorm",
            "layer1_attention_output": "layer1.attention",
            "layer1_post_attention_rmsnorm_output": "layer1.post_attention_layernorm",
            "layer1_mlp_output": "layer1.mlp",
            "layer1_output_hidden": "layer1 residual merge",
            "layer2_input_hidden": "layer1-to-layer2 boundary",
        }
        operator = operator_map.get(first, "UPSTREAM_CONTEXT_OR_RUNTIME_STATE_DEPENDENCE")
    if fixed_input_name:
        args.keep_fixed_input.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"name": fixed_input_name, "tensor": payloads[0][fixed_input_name]}, args.keep_fixed_input)
    temporary_bytes = sum(path.stat().st_size for path in payload_paths)
    result = {
        "task": "LING_FRESH_PROCESS_FIRST_DIVERGENCE_CLOSURE_V2",
        "fresh_process_runs": 5,
        "diagnostic_sample": records[0]["document_id"],
        "condition": records[0]["condition"],
        "MODEL_LOAD_PROVENANCE_MATCH": "PASS" if provenance_match else "FAIL",
        "provenance_sha256": [canonical_hash(value) for value in provenance],
        "capture_ladder": order,
        "tensor_comparisons": comparisons,
        "FIRST_TRULY_DIVERGENT_UPSTREAM_TENSOR": first,
        "FIRST_NONDETERMINISTIC_OPERATOR_CANDIDATE": operator,
        "fixed_operator_input": fixed_input_name,
        "selected_autotune_configs_by_run": [selected_autotune_configs(record) for record in records],
        "temporary_tensor_bytes_before_cleanup": temporary_bytes,
        "temporary_ladder_tensors_cleaned": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for path in payload_paths:
        path.unlink()
    print(json.dumps({key: result[key] for key in ("MODEL_LOAD_PROVENANCE_MATCH", "FIRST_TRULY_DIVERGENT_UPSTREAM_TENSOR", "FIRST_NONDETERMINISTIC_OPERATOR_CANDIDATE")}, indent=2))


if __name__ == "__main__":
    main()
