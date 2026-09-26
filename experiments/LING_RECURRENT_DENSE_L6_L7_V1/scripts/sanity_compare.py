#!/usr/bin/env python3
"""Non-AIME H/L4/L5/L6/L7 local and persistent sanity panel."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rotations(path: Path, device) -> dict[int, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {int(layer): value.to(device=device, dtype=torch.float32) for layer, value in payload["rotations"].items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--legacy-repo", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--h", required=True)
    parser.add_argument("--l4", required=True)
    parser.add_argument("--l5", required=True)
    parser.add_argument("--l6", required=True)
    parser.add_argument("--l7", required=True)
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-memory-gib", type=int, default=22)
    args = parser.parse_args()
    rt = import_file(Path(args.trainer), "recurrent_sanity_runtime")

    class Context:
        legacy_repo = args.legacy_repo
        max_memory_gib = args.max_memory_gib

    fullpath, cayley, _ortho, model, _tokenizer, probe, layers = rt.load_context(Context())
    device = fullpath.model_input_device(model)
    paths = rt.trace_paths(Path(args.trace_dir), "VALIDATION", args.sequences)
    conditions = {
        "H_fixed": Path(args.h),
        "L4_old_single_step_Dense_State": Path(args.l4),
        "L5_old_single_step_Dense_Functional": Path(args.l5),
        "L6_recurrent_Dense_State": Path(args.l6),
        "L7_recurrent_Dense_Functional": Path(args.l7),
    }
    results = {}
    try:
        with torch.no_grad():
            for name, rotation_path in conditions.items():
                rotations = load_rotations(rotation_path, device)
                sums = {
                    "single_step_state_error": 0.0,
                    "recurrent_state_error": 0.0,
                    "local_functional_error": 0.0,
                    "multi_step_persistent_functional_error": 0.0,
                }
                count = 0
                for path in paths:
                    trace = rt.load_trace(path)
                    records = {
                        layer: rt.to_device_record(trace["layer_records"][layer], device) for layer in layers
                    }
                    teacher = {layer: records[layer]["initial_teacher_state"].float() for layer in layers}
                    recurrent = {
                        layer: rt.exact_qdq(cayley, teacher[layer].matmul(rotations[layer]), False)
                        .dequant.to(torch.bfloat16)
                        .float()
                        for layer in layers
                    }
                    for token in range(int(trace["sequence_length"])):
                        for layer in layers:
                            rec, r = records[layer], rotations[layer]
                            teacher_prev = teacher[layer]
                            _teacher_out, teacher_next = rt.fused(
                                probe, rec, teacher_prev, rec["v"][:, token : token + 1], token
                            )
                            single_qdq = rt.exact_qdq(cayley, teacher_prev.matmul(r), False).dequant.to(torch.bfloat16).float()
                            single_native = single_qdq.matmul(r.T)
                            sums["single_step_state_error"] += float(
                                rt.relative_state_loss(single_native, teacher_prev).cpu()
                            )
                            value_rot = rec["v"][:, token : token + 1].float().matmul(r).to(torch.bfloat16)
                            local_core_rot, _ = rt.fused(probe, rec, single_qdq, value_rot, token)
                            local_out = rt.local_output(
                                model,
                                layer,
                                local_core_rot.float().matmul(r.T).to(torch.bfloat16),
                                rec["dynamic_gate"][:, token : token + 1],
                            )
                            target = rec["out_proj_output"][:, token : token + 1]
                            sums["local_functional_error"] += float(rt.relative_mse(local_out, target).cpu())
                            persistent_core_rot, persistent_unquant = rt.fused(
                                probe, rec, recurrent[layer], value_rot, token
                            )
                            recurrent_post = (
                                rt.exact_qdq(cayley, persistent_unquant, False).dequant.to(torch.bfloat16).float()
                            )
                            persistent_native = recurrent_post.matmul(r.T)
                            sums["recurrent_state_error"] += float(
                                rt.relative_state_loss(persistent_native, teacher_next).cpu()
                            )
                            persistent_out = rt.local_output(
                                model,
                                layer,
                                persistent_core_rot.float().matmul(r.T).to(torch.bfloat16),
                                rec["dynamic_gate"][:, token : token + 1],
                            )
                            sums["multi_step_persistent_functional_error"] += float(
                                rt.relative_mse(persistent_out, target).cpu()
                            )
                            teacher[layer] = teacher_next
                            recurrent[layer] = recurrent_post
                            count += 1
                    del trace, records, teacher, recurrent
                    torch.cuda.empty_cache()
                results[name] = {
                    "rotation_path": str(rotation_path.resolve()),
                    "tokens_times_layers": count,
                    **{key: value / count for key, value in sums.items()},
                }
                print(json.dumps({"condition": name, **results[name]}), flush=True)
    finally:
        probe.close()
    payload = {
        "task": "LING_RECURRENT_DENSE_L6_L7_V1",
        "status": "PASS",
        "AIME26_used": False,
        "selection_use": "NONE_SANITY_ONLY",
        "validation_sequences": len(paths),
        "metrics": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
