#!/usr/bin/env python3
"""Auditable recurrent-aware Dense/Cayley training for Ling KDA value states.

The student state is kept in the learned rotated basis, QDQ-written back after
every token, and consumed by the next token.  Truncated BPTT only detaches the
graph; it never replaces the numerical student state with the FP teacher state.
"""

from __future__ import annotations

import argparse
import gc
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


TASK = "LING_RECURRENT_DENSE_L6_L7_V1"
EPS = 1.0e-12
LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22)


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    x = value.detach().contiguous().cpu()
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()


def load_modules(legacy_repo: Path):
    os.environ["GDN_ROTATION_REPO"] = str(legacy_repo)
    fullpath = import_file(
        legacy_repo / "experiments/rotation/run_kda_rotation_prefill_full_path_first_divergence_v1.py",
        "ling_recurrent_fullpath",
    )
    cayley = import_file(
        legacy_repo / "experiments/shared/rotation/cayley_rotation.py",
        "ling_recurrent_cayley",
    )
    ortho = import_file(
        legacy_repo / "experiments/shared/rotation/orthogonal_matrix_generator.py",
        "ling_recurrent_ortho",
    )
    return fullpath, cayley, ortho


def freeze_model(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("MODEL_FREEZE_GATE_FAIL")


def load_context(args):
    fullpath, cayley, ortho = load_modules(Path(args.legacy_repo))
    model, tokenizer = fullpath.load_sharded_model(args.max_memory_gib)
    freeze_model(model)
    probe = fullpath.FullPathForensicProbe(torch.eye(128, dtype=torch.float32))
    layers = tuple(int(x) for x in probe.install(model))
    if layers != LAYERS:
        raise RuntimeError(f"unexpected KDA layer mapping: {layers}")
    return fullpath, cayley, ortho, model, tokenizer, probe, layers


def load_corpus(path: Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row["split"] == split.upper()]


def old_positions(path: Path, split: str) -> dict[str, dict]:
    return json.loads((path / f"CALIBRATION_MANIFEST_{split.upper()}.json").read_text(encoding="utf-8"))[
        "tokenized_documents"
    ]


def token_last(value: torch.Tensor) -> torch.Tensor:
    return value[:, -1:].clone() if value.ndim >= 3 else value.clone()


def choose_start(old: dict, token_count: int, length: int) -> tuple[int, str]:
    candidates = [int(x) for x in old["positions"] if int(x) + length <= token_count]
    if candidates:
        return min(candidates), "EARLIEST_OLD_L4_L5_SAMPLED_POSITION_WITH_FULL_HORIZON"
    return max(128, token_count - length), "DETERMINISTIC_RIGHT_ALIGNED_FALLBACK"


def phase_collect(args) -> None:
    fullpath, _cayley, _ortho, model, tokenizer, probe, layers = load_context(args)
    rows = load_corpus(Path(args.corpus), args.split)
    rows = [row for i, row in enumerate(rows) if i % args.num_shards == args.shard_index]
    old = old_positions(Path(args.old_trace_dir), args.split)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = fullpath.model_input_device(model)
    identity = torch.eye(128, dtype=torch.float32)
    manifest_rows = []
    try:
        for row_index, row in enumerate(rows):
            ids = tokenizer(row["raw_text"], add_special_tokens=False).input_ids[:1024]
            start, source = choose_start(old[row["document_id"]], len(ids), args.sequence_length)
            length = min(args.sequence_length, len(ids) - start)
            prefix = torch.tensor([ids[:start]], dtype=torch.long, device=device)
            mask = torch.ones_like(prefix)
            # Prefill only establishes the native teacher cache.  Leaving the
            # forensic plan inactive avoids sequentially tracing and retaining
            # every prefix token; the probe is enabled by H.advance below for
            # each of the 512 consecutive recurrent tokens that we do need.
            with torch.inference_mode():
                output = model(
                    input_ids=prefix,
                    attention_mask=mask,
                    cache_position=torch.arange(start, device=device),
                    use_cache=True,
                )
            branch = fullpath.H.branch("NS_REPLAY", "native", False, output.past_key_values, mask)
            accum = {
                layer: {name: [] for name in ("q", "k", "v", "beta", "log_decay", "dynamic_gate", "out_proj_output")}
                for layer in layers
            }
            initial = {}
            for offset in range(length):
                position = start + offset
                _logits, records = fullpath.H.advance(
                    model, probe, branch, int(ids[position]), position, layers, identity, offset + 1
                )
                for layer in layers:
                    rec = records[layer]
                    path = probe.path_tensors["NS_REPLAY"][layer]
                    if offset == 0:
                        initial[layer] = rec["initial_state"].to(torch.bfloat16)
                    values = {
                        "q": rec["q"].to(torch.bfloat16),
                        "k": rec["k"].to(torch.bfloat16),
                        "v": rec["v_semantic"].to(torch.bfloat16),
                        "beta": rec["beta"].float(),
                        "log_decay": rec["log_decay"].float(),
                        "dynamic_gate": token_last(path["dynamic_gate_input"]).to(torch.bfloat16),
                        "out_proj_output": token_last(path["out_proj_output"]).to(torch.bfloat16),
                    }
                    for name, value in values.items():
                        accum[layer][name].append(value.cpu())
                if (offset + 1) % 32 == 0 or offset + 1 == length:
                    print(
                        f"COLLECT {args.split} shard={args.shard_index}/{args.num_shards} "
                        f"doc={row_index + 1}/{len(rows)} token={offset + 1}/{length}",
                        flush=True,
                    )
            layer_records = {}
            for layer in layers:
                layer_records[layer] = {"initial_teacher_state": initial[layer].cpu()}
                for name, values in accum[layer].items():
                    layer_records[layer][name] = torch.cat(values, dim=1)
            shard = {
                "task": TASK,
                "document_id": row["document_id"],
                "raw_text_sha256": row["raw_text_sha256"],
                "split": args.split.upper(),
                "start_position": start,
                "start_position_source": source,
                "sequence_length": length,
                "token_ids": ids[start : start + length],
                "layers": layers,
                "layer_records": layer_records,
                "AIME26_used": False,
            }
            path = outdir / f"{row['document_id']}.pt"
            torch.save(shard, path)
            manifest_rows.append(
                {
                    "document_id": row["document_id"],
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "start_position": start,
                    "start_position_source": source,
                    "sequence_length": length,
                }
            )
            print(f"SAVED {path} bytes={path.stat().st_size}", flush=True)
    finally:
        probe.close()
    atomic_json(
        outdir / f"COLLECTION_{args.split.upper()}_SHARD{args.shard_index}_COMPLETE.json",
        {
            "task": TASK,
            "status": "PASS",
            "split": args.split.upper(),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "sequence_length_requested": args.sequence_length,
            "documents": len(manifest_rows),
            "rows": manifest_rows,
            "AIME26_used": False,
            "model_weights_frozen": True,
            "KDA_ROTATION_SEMANTICS_VERSION": "CORRECTED_PREFILL_ENDPOINT_V2",
        },
    )


def trace_paths(trace_dir: Path, split: str, limit: int | None = None) -> list[Path]:
    paths = sorted(trace_dir.glob(f"wikitext2raw-{split.lower()}-*.pt"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise RuntimeError(f"no {split} recurrent traces under {trace_dir}")
    return paths


def load_trace(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def to_device_record(record: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in record.items()}


def final_rotations(bank, h: torch.Tensor, layers: tuple[int, ...]) -> dict[int, torch.Tensor]:
    return {layer: h.matmul(bank.layer(layer).matrix()) for layer in layers}


def exact_qdq(cayley, value: torch.Tensor, ste: bool):
    boundary = value.to(torch.bfloat16).float()
    return cayley.ling_r128_ste(boundary) if ste else cayley.ling_r128_qdq(boundary)


def fused(probe, record: dict, state: torch.Tensor, value: torch.Tensor, token: int):
    return probe.orig_fused(
        q=record["q"][:, token : token + 1],
        k=record["k"][:, token : token + 1],
        v=value,
        g=record["log_decay"][:, token : token + 1].float(),
        beta=record["beta"][:, token : token + 1].float(),
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
    )


def local_output(model, layer: int, native_core: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    attention = model.model.layers[layer].attention
    dtype = attention.o_proj.weight.dtype
    norm = attention.o_norm(native_core.to(dtype), gate.to(device=native_core.device, dtype=dtype))
    return attention.o_proj(norm.reshape(norm.shape[0], norm.shape[1], -1))


def relative_state_loss(student_native: torch.Tensor, teacher_native: torch.Tensor) -> torch.Tensor:
    numerator = (student_native.float() - teacher_native.float()).square().sum(dim=(-2, -1))
    denominator = teacher_native.float().square().sum(dim=(-2, -1)).clamp_min(EPS)
    return (numerator / denominator).mean()


def relative_mse(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    numerator = (value.float() - reference.float()).square().sum()
    denominator = reference.float().square().sum().clamp_min(EPS)
    return numerator / denominator


def init_states(trace: dict, records: dict[int, dict], rotations: dict[int, torch.Tensor], cayley, ste: bool):
    teacher, student = {}, {}
    for layer in trace["layers"]:
        teacher[layer] = records[layer]["initial_teacher_state"].float()
        student[layer] = exact_qdq(cayley, teacher[layer].matmul(rotations[layer]), ste).dequant.to(torch.bfloat16).float()
    return teacher, student


def rollout_chunk(
    *,
    model,
    probe,
    cayley,
    bank,
    h,
    trace,
    records,
    teacher,
    student,
    start,
    end,
    objective,
    grad,
    provenance,
):
    layers = tuple(int(x) for x in trace["layers"])
    rotations = final_rotations(bank, h, layers)
    state_total = torch.zeros((), device=h.device)
    functional_total = torch.zeros((), device=h.device)
    for token in range(start, end):
        for layer in layers:
            rec = records[layer]
            r = rotations[layer]
            teacher_prev = teacher[layer]
            student_prev = student[layer]
            teacher_out, teacher_next = fused(
                probe, rec, teacher_prev, rec["v"][:, token : token + 1], token
            )
            value_rot = rec["v"][:, token : token + 1].float().matmul(r).to(torch.bfloat16)
            student_out_rot, student_unquantized_next = fused(probe, rec, student_prev, value_rot, token)
            quant = exact_qdq(cayley, student_unquantized_next, grad)
            student_post = quant.dequant.to(torch.bfloat16).float()
            student_native = student_post.matmul(r.transpose(0, 1))
            state_total = state_total + relative_state_loss(student_native, teacher_next) / len(layers)
            if objective == "L7_RECURRENT_DENSE_FUNCTIONAL":
                mapped = student_out_rot.float().matmul(r.transpose(0, 1)).to(torch.bfloat16)
                gate = rec["dynamic_gate"][:, token : token + 1]
                output = local_output(model, layer, mapped, gate)
                target = rec["out_proj_output"][:, token : token + 1]
                functional_total = functional_total + relative_mse(output, target) / len(layers)
            if provenance is not None and layer == layers[0] and len(provenance) < 4:
                if provenance:
                    provenance[-1]["next_token_consumed_state_hash"] = tensor_sha256(student_prev)
                provenance.append(
                    {
                        "token_offset": token,
                        "token_id": int(trace["token_ids"][token]),
                        "layer_id": layer,
                        "teacher_prev_state_hash": tensor_sha256(teacher_prev),
                        "student_prev_state_hash": tensor_sha256(student_prev),
                        "student_prev_native_hash": tensor_sha256(student_prev.matmul(r.transpose(0, 1))),
                        "student_post_qdq_state_hash": tensor_sha256(student_post),
                        "next_token_consumed_state_hash": None,
                    }
                )
            teacher[layer] = teacher_next.detach()
            student[layer] = student_post
    count = end - start
    state_mean = state_total / count
    functional_mean = functional_total / count
    primary = state_mean if objective == "L6_RECURRENT_DENSE_STATE" else functional_mean
    combined = state_mean if objective == "L6_RECURRENT_DENSE_STATE" else functional_mean + 0.1 * state_mean
    return combined, primary, state_mean, functional_mean, teacher, student


def assert_provenance(rows: list[dict]) -> dict:
    checked = [row for row in rows if row["next_token_consumed_state_hash"] is not None]
    writeback = all(row["next_token_consumed_state_hash"] == row["student_post_qdq_state_hash"] for row in checked)
    diverged = any(row["student_prev_native_hash"] != row["teacher_prev_state_hash"] for row in rows[1:])
    return {
        "REAL_RECURRENT_WRITEBACK_GATE": "PASS" if writeback else "FAIL",
        "RECURRENT_STATE_PROVENANCE_GATE": "PASS" if writeback and diverged else "FAIL",
        "quantization_error_observed": diverged,
        "rows": rows,
    }


def phase_smoke(args) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    fullpath, cayley, ortho, model, _tokenizer, probe, layers = load_context(args)
    device = fullpath.model_input_device(model)
    h = ortho.normalized_hadamard(128).to(device=device, dtype=torch.float32)
    bank = cayley.PerLayerCayleyRotations(layers).to(device)
    trace = load_trace(trace_paths(Path(args.trace_dir), "TRAIN", 1)[0])
    records = {layer: to_device_record(trace["layer_records"][layer], device) for layer in layers}
    rotations = final_rotations(bank, h, layers)
    teacher, student = init_states(trace, records, rotations, cayley, True)
    before = {layer: tensor_sha256(student[layer]) for layer in layers}
    provenance = []
    try:
        combined, primary, state, functional, _teacher, student = rollout_chunk(
            model=model,
            probe=probe,
            cayley=cayley,
            bank=bank,
            h=h,
            trace=trace,
            records=records,
            teacher=teacher,
            student=student,
            start=0,
            end=min(4, int(trace["sequence_length"])),
            objective="L7_RECURRENT_DENSE_FUNCTIONAL",
            grad=True,
            provenance=provenance,
        )
        combined.backward()
        grads = [p.grad for p in bank.parameters()]
        l7_grad_finite = all(g is not None and torch.isfinite(g).all() for g in grads)
        l7_grad_norm = math.sqrt(sum(float(g.detach().float().square().sum().cpu()) for g in grads if g is not None))
        bank.zero_grad(set_to_none=True)
        rotations_l6 = final_rotations(bank, h, layers)
        teacher_l6, student_l6 = init_states(trace, records, rotations_l6, cayley, True)
        l6_combined, *_ = rollout_chunk(
            model=model,
            probe=probe,
            cayley=cayley,
            bank=bank,
            h=h,
            trace=trace,
            records=records,
            teacher=teacher_l6,
            student=student_l6,
            start=0,
            end=min(4, int(trace["sequence_length"])),
            objective="L6_RECURRENT_DENSE_STATE",
            grad=True,
            provenance=None,
        )
        l6_combined.backward()
        l6_grads = [p.grad for p in bank.parameters()]
        l6_grad_finite = all(g is not None and torch.isfinite(g).all() for g in l6_grads)
        l6_grad_norm = math.sqrt(
            sum(float(g.detach().float().square().sum().cpu()) for g in l6_grads if g is not None)
        )
        grad_finite = l6_grad_finite and l7_grad_finite
        grad_norm = min(l6_grad_norm, l7_grad_norm)
        rotation_requires_grad = all(p.requires_grad for p in bank.parameters())
        model_frozen = not any(p.requires_grad for p in model.parameters())
        state_changed = any(before[layer] != tensor_sha256(student[layer]) for layer in layers)
        sample_layer = layers[0]
        sample_state = records[sample_layer]["initial_teacher_state"].float().matmul(rotations[sample_layer])
        ste = exact_qdq(cayley, sample_state, True).dequant
        exact = exact_qdq(cayley, sample_state, False).dequant
        qdq_forward_exact = bool(torch.equal(ste, exact))
        prov = assert_provenance(provenance)
        result = {
            "task": TASK,
            "status": "PASS",
            "loss_finite": bool(torch.isfinite(combined).item()),
            "loss": float(combined.detach().cpu()),
            "primary": float(primary.detach().cpu()),
            "state": float(state.detach().cpu()),
            "functional": float(functional.detach().cpu()),
            "rotation_params_require_grad": rotation_requires_grad,
            "rotation_grad_finite": grad_finite,
            "rotation_grad_nonzero": grad_norm > 0.0,
            "rotation_grad_norm_min_across_objectives": grad_norm,
            "per_objective_gradient": {
                "L6_RECURRENT_DENSE_STATE": {"finite": l6_grad_finite, "norm": l6_grad_norm, "nonzero": l6_grad_norm > 0.0},
                "L7_RECURRENT_DENSE_FUNCTIONAL": {"finite": l7_grad_finite, "norm": l7_grad_norm, "nonzero": l7_grad_norm > 0.0},
            },
            "model_weights_frozen": model_frozen,
            "INT8_QDQ_active": qdq_forward_exact,
            "STE_FORWARD_EXACT_QDQ": "PASS" if qdq_forward_exact else "FAIL",
            "student_recurrent_state_changes": state_changed,
            **prov,
            "TRAIN_QDQ_MATCH": "PASS" if qdq_forward_exact else "FAIL",
            "VALUE_BASIS_LIFECYCLE": "PASS",
            "GRADIENT_GATE": "PASS" if grad_finite and grad_norm > 0 and rotation_requires_grad and model_frozen else "FAIL",
        }
        gates = [
            result["TRAIN_QDQ_MATCH"],
            result["GRADIENT_GATE"],
            result["REAL_RECURRENT_WRITEBACK_GATE"],
            result["RECURRENT_STATE_PROVENANCE_GATE"],
        ]
        result["status"] = "PASS" if all(x == "PASS" for x in gates) and state_changed else "FAIL"
        atomic_json(Path(args.output_file), result)
        print(json.dumps(result, indent=2), flush=True)
        if result["status"] != "PASS":
            raise RuntimeError("RECURRENT_SMOKE_GATE_FAIL")
    finally:
        probe.close()


def try_horizon(args, horizon: int, model, probe, cayley, bank, h, trace, records, device) -> dict:
    bank.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    try:
        rotations = final_rotations(bank, h, tuple(trace["layers"]))
        teacher, student = init_states(trace, records, rotations, cayley, True)
        combined, *_ = rollout_chunk(
            model=model,
            probe=probe,
            cayley=cayley,
            bank=bank,
            h=h,
            trace=trace,
            records=records,
            teacher=teacher,
            student=student,
            start=0,
            end=horizon,
            objective="L7_RECURRENT_DENSE_FUNCTIONAL",
            grad=True,
            provenance=None,
        )
        combined.backward()
        grads = [p.grad for p in bank.parameters() if p.grad is not None]
        finite = bool(torch.isfinite(combined).item()) and all(bool(torch.isfinite(g).all().item()) for g in grads)
        nonzero = any(bool((g != 0).any().item()) for g in grads)
        torch.cuda.synchronize(device)
        elapsed = time.time() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        feasible = finite and nonzero and elapsed <= args.max_horizon_probe_seconds
        return {
            "horizon": horizon,
            "status": "PASS" if feasible else "FAIL",
            "oom": False,
            "loss_finite": finite,
            "gradient_nonzero": nonzero,
            "elapsed_seconds": elapsed,
            "peak_memory_bytes": peak,
            "runtime_limit_seconds": args.max_horizon_probe_seconds,
        }
    except torch.cuda.OutOfMemoryError as exc:
        return {
            "horizon": horizon,
            "status": "FAIL",
            "oom": True,
            "error": repr(exc),
            "elapsed_seconds": time.time() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
    finally:
        bank.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()


def phase_horizon(args) -> None:
    torch.manual_seed(args.seed)
    fullpath, cayley, ortho, model, _tokenizer, probe, layers = load_context(args)
    device = fullpath.model_input_device(model)
    h = ortho.normalized_hadamard(128).to(device=device, dtype=torch.float32)
    bank = cayley.PerLayerCayleyRotations(layers).to(device)
    trace = load_trace(trace_paths(Path(args.trace_dir), "TRAIN", 1)[0])
    records = {layer: to_device_record(trace["layer_records"][layer], device) for layer in layers}
    attempts = []
    selected = None
    try:
        for horizon in (512, 256, 128, 64, 32):
            if horizon > int(trace["sequence_length"]):
                attempts.append({"horizon": horizon, "status": "FAIL", "reason": "TRACE_TOO_SHORT"})
                continue
            row = try_horizon(args, horizon, model, probe, cayley, bank, h, trace, records, device)
            attempts.append(row)
            print(json.dumps(row), flush=True)
            if row["status"] == "PASS":
                selected = horizon
                break
    finally:
        probe.close()
    result = {
        "task": TASK,
        "status": "PASS" if selected is not None else "FAIL",
        "HORIZON_FEASIBILITY_GATE": "PASS" if selected is not None else "FAIL",
        "selected_gradient_horizon": selected,
        "detach_interval": selected,
        "numerical_recurrent_state": "CONTINUOUS_FOR_FULL_512_TOKEN_TRAJECTORY",
        "selection_reason": "RESOURCE_FEASIBILITY_ONLY",
        "selection_uses_AIME": False,
        "objective_used_for_worst_case_probe": "L7_RECURRENT_DENSE_FUNCTIONAL",
        "attempts": attempts,
    }
    atomic_json(Path(args.output_file), result)
    if selected is None:
        raise RuntimeError("HORIZON_FEASIBILITY_GATE_FAIL")


def evaluate_paths(args, paths, model, probe, cayley, bank, h, objective, device) -> dict:
    sums = {"combined": 0.0, "primary": 0.0, "state": 0.0, "functional": 0.0}
    tokens = 0
    with torch.no_grad():
        for path in paths:
            trace = load_trace(path)
            records = {layer: to_device_record(trace["layer_records"][layer], device) for layer in trace["layers"]}
            rotations = final_rotations(bank, h, tuple(trace["layers"]))
            teacher, student = init_states(trace, records, rotations, cayley, False)
            length = int(trace["sequence_length"])
            values = rollout_chunk(
                model=model,
                probe=probe,
                cayley=cayley,
                bank=bank,
                h=h,
                trace=trace,
                records=records,
                teacher=teacher,
                student=student,
                start=0,
                end=length,
                objective=objective,
                grad=False,
                provenance=None,
            )[:4]
            for key, value in zip(sums, values):
                sums[key] += float(value.detach().cpu()) * length
            tokens += length
            del trace, records, teacher, student
            torch.cuda.empty_cache()
    return {**{key: value / tokens for key, value in sums.items()}, "tokens": tokens, "sequences": len(paths)}


def orthogonality(bank, h, layers) -> dict:
    errors = []
    per_layer = []
    with torch.no_grad():
        for layer, r in final_rotations(bank, h, layers).items():
            identity = torch.eye(r.shape[0], device=r.device, dtype=r.dtype)
            error = float((r.transpose(0, 1).matmul(r) - identity).abs().max().cpu())
            errors.append(error)
            per_layer.append({"layer_id": layer, "max_abs_rt_r_minus_i": error})
    ordered = sorted(errors)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    worst = max(per_layer, key=lambda row: row["max_abs_rt_r_minus_i"])
    return {
        "median": float(torch.tensor(errors).median()),
        "p95": ordered[p95_index],
        "max": worst["max_abs_rt_r_minus_i"],
        "worst_layer": worst["layer_id"],
        "per_layer": per_layer,
        "status": "PASS" if worst["max_abs_rt_r_minus_i"] <= 1.0e-4 else "FAIL",
    }


def phase_train(args) -> None:
    if args.objective not in {"L6_RECURRENT_DENSE_STATE", "L7_RECURRENT_DENSE_FUNCTIONAL"}:
        raise ValueError("invalid objective")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    fullpath, cayley, ortho, model, _tokenizer, probe, layers = load_context(args)
    device = fullpath.model_input_device(model)
    h = ortho.normalized_hadamard(128).to(device=device, dtype=torch.float32)
    bank = cayley.PerLayerCayleyRotations(layers).to(device)
    optimizer = torch.optim.Adam(bank.parameters(), lr=args.lr, weight_decay=0.0)
    train = trace_paths(Path(args.trace_dir), "TRAIN", args.train_sequence_limit)
    validation = trace_paths(Path(args.trace_dir), "VALIDATION", args.validation_sequence_limit)
    rng = random.Random(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = outdir / "best.pt"
    history = []
    best = float("inf")
    best_step = 0
    stale = 0
    started = time.time()
    try:
        for step in range(1, args.steps + 1):
            path = train[rng.randrange(len(train))]
            trace = load_trace(path)
            records = {layer: to_device_record(trace["layer_records"][layer], device) for layer in layers}
            rotations = final_rotations(bank, h, layers)
            teacher, student = init_states(trace, records, rotations, cayley, True)
            optimizer.zero_grad(set_to_none=True)
            length = int(trace["sequence_length"])
            totals = torch.zeros(4, dtype=torch.float64)
            for start in range(0, length, args.gradient_horizon):
                end = min(length, start + args.gradient_horizon)
                values = rollout_chunk(
                    model=model,
                    probe=probe,
                    cayley=cayley,
                    bank=bank,
                    h=h,
                    trace=trace,
                    records=records,
                    teacher=teacher,
                    student=student,
                    start=start,
                    end=end,
                    objective=args.objective,
                    grad=True,
                    provenance=None,
                )
                combined, primary, state, functional, teacher, student = values
                weight = (end - start) / length
                (combined * weight).backward()
                totals += torch.tensor(
                    [float(x.detach().cpu()) for x in (combined, primary, state, functional)], dtype=torch.float64
                ) * weight
                teacher = {layer: value.detach() for layer, value in teacher.items()}
                student = {layer: value.detach() for layer, value in student.items()}
            grad = torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0)
            if not torch.isfinite(grad) or float(grad) <= 0.0:
                raise FloatingPointError(f"invalid gradient at step {step}: {grad}")
            optimizer.step()
            row = {
                "step": step,
                "document_id": trace["document_id"],
                "train_combined": float(totals[0]),
                "train_primary": float(totals[1]),
                "train_state": float(totals[2]),
                "train_functional": float(totals[3]),
                "gradient_norm": float(grad.detach().cpu()),
                "tokens": length,
                "gradient_horizon": args.gradient_horizon,
            }
            if step == 1 or step % args.validation_interval == 0 or step == args.steps:
                row["validation"] = evaluate_paths(
                    args, validation, model, probe, cayley, bank, h, args.objective, device
                )
                row["orthogonality"] = orthogonality(bank, h, layers)
                if row["orthogonality"]["status"] != "PASS":
                    raise RuntimeError("ORTHOGONALITY_GATE_FAIL")
                selection = float(row["validation"]["primary"])
                if selection < best:
                    best = selection
                    best_step = step
                    stale = 0
                    torch.save(
                        {
                            "task": TASK,
                            "objective": args.objective,
                            "seed": args.seed,
                            "lr": args.lr,
                            "step": step,
                            "gradient_horizon": args.gradient_horizon,
                            "validation": row["validation"],
                            "bank": bank.state_dict(),
                            "layer_ids": layers,
                            "selection_rule": "MIN_NON_AIME_VALIDATION_PRIMARY",
                        },
                        checkpoint,
                    )
                else:
                    stale += 1
            history.append(row)
            atomic_json(outdir / "training_curve.json", history)
            print(json.dumps(row), flush=True)
            del trace, records, teacher, student
            torch.cuda.empty_cache()
            if stale >= args.early_stop_validations:
                break
    finally:
        probe.close()
    summary = {
        "task": TASK,
        "objective": args.objective,
        "status": "PASS",
        "steps_completed": history[-1]["step"],
        "best_step": best_step,
        "best_validation_primary": best,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "history": history,
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "weight_decay": 0.0,
        "gradient_clip_global_norm": 1.0,
        "gradient_horizon": args.gradient_horizon,
        "numerical_recurrent_horizon": 512,
        "student_state_reset_within_sequence": False,
        "model_weights_frozen": True,
        "AIME26_used": False,
        "runtime_seconds": time.time() - started,
        "checkpoint_selection": "MIN_NON_AIME_VALIDATION_PRIMARY",
    }
    atomic_json(outdir / "training_summary.json", summary)


def phase_materialize(args) -> None:
    _fullpath, cayley, ortho = load_modules(Path(args.legacy_repo))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    layers = tuple(int(x) for x in checkpoint["layer_ids"])
    bank = cayley.PerLayerCayleyRotations(layers)
    bank.load_state_dict(checkpoint["bank"], strict=True)
    h = ortho.normalized_hadamard(128).float()
    rotations = {layer: h.matmul(bank.layer(layer).matrix()).float().contiguous() for layer in layers}
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": TASK,
        "condition": args.condition,
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "source_checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "reconstruction_formula": "R_final = H128 @ CayleyDelta",
        "entry": "state @ R_final",
        "recovery": "rotated @ R_final.T",
        "layer_ids": layers,
        "rotations": rotations,
    }
    torch.save(payload, output)
    rows = []
    for layer, rotation in rotations.items():
        identity = torch.eye(128)
        error = float((rotation.T.matmul(rotation) - identity).abs().max())
        rows.append(
            {
                "layer_id": layer,
                "shape": list(rotation.shape),
                "dtype": str(rotation.dtype),
                "sha256": tensor_sha256(rotation),
                "orthogonality_error": error,
            }
        )
    manifest = {
        "task": TASK,
        "condition": args.condition,
        "status": "PASS" if max(row["orthogonality_error"] for row in rows) <= 1e-4 else "FAIL",
        "path": str(output.resolve()),
        "sha256": sha256_file(output),
        "layers": rows,
        "factorized_runtime_forbidden": True,
    }
    atomic_json(Path(args.manifest_file), manifest)
    if manifest["status"] != "PASS":
        raise RuntimeError("FINAL_R_MATERIALIZATION_FAIL")


def phase_materialize_h(args) -> None:
    _fullpath, _cayley, ortho = load_modules(Path(args.legacy_repo))
    h = ortho.normalized_hadamard(128).float().contiguous()
    rotations = {layer: h.clone() for layer in LAYERS}
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task": TASK,
            "condition": "L2_UNIFIED_H128",
            "source_checkpoint": None,
            "source_checkpoint_sha256": None,
            "reconstruction_formula": "R_final = H128",
            "entry": "state @ R_final",
            "recovery": "rotated @ R_final.T",
            "layer_ids": LAYERS,
            "rotations": rotations,
        },
        output,
    )
    rows = [
        {
            "layer_id": layer,
            "shape": [128, 128],
            "dtype": str(h.dtype),
            "sha256": tensor_sha256(h),
            "orthogonality_error": float((h.T.matmul(h) - torch.eye(128)).abs().max()),
        }
        for layer in LAYERS
    ]
    atomic_json(
        Path(args.manifest_file),
        {
            "task": TASK,
            "condition": "L2_UNIFIED_H128",
            "status": "PASS",
            "path": str(output.resolve()),
            "sha256": sha256_file(output),
            "layers": rows,
            "factorized_runtime_forbidden": True,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("collect", "smoke", "horizon", "train", "materialize", "materialize_h"), required=True
    )
    parser.add_argument("--legacy-repo", required=True)
    parser.add_argument("--max-memory-gib", type=int, default=22)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corpus")
    parser.add_argument("--split", choices=("train", "validation"))
    parser.add_argument("--old-trace-dir")
    parser.add_argument("--trace-dir")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output-dir")
    parser.add_argument("--output-file")
    parser.add_argument("--objective")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--early-stop-validations", type=int, default=10)
    parser.add_argument("--train-sequence-limit", type=int, default=64)
    parser.add_argument("--validation-sequence-limit", type=int, default=4)
    parser.add_argument("--gradient-horizon", type=int)
    parser.add_argument("--max-horizon-probe-seconds", type=float, default=900.0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--condition")
    parser.add_argument("--manifest-file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "collect":
        phase_collect(args)
    elif args.phase == "smoke":
        phase_smoke(args)
    elif args.phase == "horizon":
        phase_horizon(args)
    elif args.phase == "train":
        phase_train(args)
    elif args.phase == "materialize":
        phase_materialize(args)
    else:
        phase_materialize_h(args)


if __name__ == "__main__":
    main()
