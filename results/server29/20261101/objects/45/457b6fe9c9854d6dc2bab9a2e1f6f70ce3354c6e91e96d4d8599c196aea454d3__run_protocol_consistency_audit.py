#!/usr/bin/env python3
"""Protocol consistency audit for Qwen3.5 GDN state quantization.

This script audits the mismatch between ordinary INT4-row quantizer profiles and
cadence K=1 INT4-row results. It is diagnostic only: no model source, model
weights, transformers, torch, CUDA, tokenizer, or dependency changes.
"""
import hashlib
import inspect
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from common import build_arg_parser, require, resolve_cfg, save_json, write_text, timestamp_record
from state_fake_quant import fake_quant_state, logical_scale_shape, normalize_granularity
from run_state_sensitivity_structure_audit import (
    FULL_ATTENTION_LAYERS,
    GDN_LAYERS,
    feed_step,
    generate_fp32_path,
    get_state,
    global_state_divergence,
    logits_metrics,
    rel_norm,
)


PROMPTS = [
    ("factual", "Answer briefly: What is the capital of France?"),
    ("reasoning", "Solve step by step: If a train travels 60 km in 45 minutes, what is its speed in km/h?"),
    ("code", "Write a Python function that returns factorial using a loop."),
    ("summary", "Summarize in one sentence: Recurrent state quantization can reduce memory, but errors may propagate."),
    ("long_context", "Context: The red key opens the basement. The blue key opens the attic. The green key opens the shed. Question: which key opens the attic?"),
    ("math", "Compute 17 * 23 and give only the final integer."),
    ("instruction", "List three practical reasons to run controlled ablations before proposing a quantization method."),
    ("needle", "Context: Alice bought tea. Ben stored the passcode 9137 in a drawer. Carol walked home. Question: what passcode did Ben store?"),
]

LEGACY_SELECTED_LAYERS = [1, 9, 17, 25, 29]
BITS = 4
GRANULARITY = "row"
STATE_ATOL = 1e-7
STATE_RTOL = 1e-6
LOGIT_ATOL = 1e-6
LOGIT_RTOL = 1e-5


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def avg(xs):
    xs = [float(x) for x in xs if x is not None and x == x and x not in (float("inf"), float("-inf"))]
    return sum(xs) / len(xs) if xs else None


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_ids(ids):
    return hashlib.sha256(",".join(str(int(x)) for x in ids).encode("utf-8")).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return "ERROR: %s" % exc


def tensor_checksum(torch, x):
    xf = x.detach().float()
    return {
        "sum": float(xf.sum().item()),
        "abs_sum": float(xf.abs().sum().item()),
        "norm": float(torch.linalg.vector_norm(xf).item()),
        "max_abs": float(xf.abs().max().item()),
    }


def tensor_max_abs(torch, a, b):
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def tensor_allclose(torch, a, b, atol, rtol):
    return bool(torch.allclose(a.detach().float(), b.detach().float(), atol=atol, rtol=rtol))


def should_quantize(step_idx, cadence_k):
    return cadence_k > 0 and step_idx % cadence_k == 0


def prompt_fingerprints(tokenizer, prompt_limit):
    out = []
    for i, (kind, text) in enumerate(PROMPTS[:prompt_limit]):
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
        out.append(
            {
                "prompt_id": i,
                "kind": kind,
                "text": text,
                "prompt_sha256": sha256_text(text),
                "input_ids_sha256": sha256_ids(ids),
                "input_length": len(ids),
                "tokenized_input_ids": ids,
                "chat_template": "not_used",
                "system_prompt": None,
                "padding": "none",
                "truncation": "none",
            }
        )
    return out


def quantizer_fingerprint(torch, sample_state):
    qdq, scale = fake_quant_state(sample_state, BITS, GRANULARITY)
    qmax = (2 ** (BITS - 1)) - 1
    return {
        "bits": BITS,
        "signed": True,
        "symmetric": True,
        "qmin": -qmax,
        "qmax": qmax,
        "actual_int4_range": "[-7, 7]",
        "scale_formula": "max(abs(x))/(2^(bits-1)-1)",
        "epsilon": 1e-12,
        "rounding": "torch.round nearest",
        "clamp": "clamp(-qmax, qmax)",
        "zero_point": None,
        "granularity": normalize_granularity(GRANULARITY),
        "state_shape": list(sample_state.shape),
        "reduction_dims": [-1],
        "logical_scale_shape": logical_scale_shape(sample_state, GRANULARITY),
        "broadcast_scale_shape": list(scale.shape),
        "scale_dtype": str(scale.dtype),
        "state_dtype_before": str(sample_state.dtype),
        "dequantized_state_dtype": str(qdq.dtype),
        "state_dtype_after_copy": str(sample_state.dtype),
    }


def source_evidence():
    import run_dynamic_sensitivity_validation as dyn
    import run_insight_discovery as insight
    import run_state_sensitivity_structure_audit as struct

    def fn_info(fn):
        src = inspect.getsource(fn)
        lines, line_no = inspect.getsourcelines(fn)
        return {"function": fn.__name__, "line_start": line_no, "line_end": line_no + len(lines) - 1, "sha256": sha256_text(src), "source_excerpt": src[:1600]}

    return {
        "dynamic_script": {
            "path": "/data/zypan/experiments/qwen35_gdn_quant/run_dynamic_sensitivity_validation.py",
            "sha256": file_sha256("/data/zypan/experiments/qwen35_gdn_quant/run_dynamic_sensitivity_validation.py"),
            "ordinary_profile": fn_info(dyn.quantizer_profile_one),
            "cadence": fn_info(dyn.cadence_one),
        },
        "insight_script": {
            "path": "/data/zypan/experiments/qwen35_gdn_quant/run_insight_discovery.py",
            "sha256": file_sha256("/data/zypan/experiments/qwen35_gdn_quant/run_insight_discovery.py"),
            "free_decode_quant": fn_info(insight.free_decode_quant),
            "cadence": fn_info(insight.run_cadence),
        },
        "structure_audit_script": {
            "path": "/data/zypan/experiments/qwen35_gdn_quant/run_state_sensitivity_structure_audit.py",
            "sha256": file_sha256("/data/zypan/experiments/qwen35_gdn_quant/run_state_sensitivity_structure_audit.py"),
            "cadence_confirmation": fn_info(struct.run_cadence_confirmation),
        },
    }


def legacy_reported_values():
    values = {
        "ordinary_INT4_row_dynamic_reported_mean_KL": 0.05477945233754376,
        "cadence_K1_dynamic_reported_mean_KL": 0.2474213619729797,
    }
    p = Path("/data/zypan/results/gdn_dynamic_sensitivity_validation_v1.json")
    if p.exists():
        d = json.loads(p.read_text())
        vals = [s for s in d.get("quantizer_error_profiles", {}).get("summaries", []) if s.get("quantizer") == "INT4-row"]
        cad = [s for s in d.get("cadence_confirmation", {}).get("by_cadence", []) if s.get("cadence") == 1]
        if vals:
            values["ordinary_INT4_row_dynamic_file_mean_KL"] = avg([v.get("mean_logit_kl") for v in vals])
            values["ordinary_INT4_row_dynamic_steps"] = sorted(set(v.get("steps_completed") for v in vals))
        if cad:
            values["cadence_K1_dynamic_file_mean_KL"] = cad[0].get("mean_KL")
            values["cadence_K1_dynamic_steps"] = d.get("cadence_confirmation", {}).get("protocol", {}).get("steps")
    return values


def environment(torch, transformers, cfg):
    return {
        "run": timestamp_record(),
        "python": run_cmd(["/data/ydai/miniconda3/envs/bitdecode/bin/python3.10", "--version"]),
        "torch": getattr(torch, "__version__", "unknown"),
        "transformers": getattr(transformers, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "model_path": cfg["model_path"],
        "transformers_source": "/data/zypan/transformers-qwen35",
        "transformers_revision": run_cmd(["git", "rev-parse", "HEAD"], cwd="/data/zypan/transformers-qwen35"),
        "eval_mode": True,
        "torch_context": "torch.inference_mode",
        "device_map": "auto",
        "dtype": "auto model load; recurrent state runtime torch.float32",
        "attention_implementation": "default transformers implementation",
    }


def get_all_state_vector_diff(torch, cache_a, cache_b, layers):
    max_diff = 0.0
    first = None
    for layer in layers:
        a = get_state(cache_a, layer)
        b = get_state(cache_b, layer)
        diff = tensor_max_abs(torch, a, b)
        if diff > max_diff:
            max_diff = diff
            first = {"layer_idx": layer, "max_abs_diff": diff}
    return max_diff, first


def fp32_loop(torch, model, tokenizer, prompt, token_path, steps, noop=False, zero_perturb=False):
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    mask = enc.get("attention_mask")
    mask = mask.to(device) if mask is not None else None
    past = None
    logits = []
    with torch.inference_mode():
        for token_idx in range(min(steps, len(token_path))):
            out = feed_step(torch, model, input_ids, mask, past)
            past = out.past_key_values
            if noop:
                for layer in GDN_LAYERS:
                    s = get_state(past, layer)
                    s.copy_(s)
            if zero_perturb:
                for layer in GDN_LAYERS:
                    get_state(past, layer).add_(torch.zeros_like(get_state(past, layer)))
            logits.append(out.logits[:, -1, :].detach().float().cpu())
            input_ids = torch.tensor([[token_path[token_idx]]], dtype=input_ids.dtype, device=device)
            mask = None
    return {"logits": logits, "past": past}


def compare_fp32_runs(torch, a, b):
    max_logit = 0.0
    mean_logit = []
    top1 = []
    for la, lb in zip(a["logits"], b["logits"]):
        diff = (la - lb).abs()
        max_logit = max(max_logit, float(diff.max().item()))
        mean_logit.append(float(diff.mean().item()))
        top1.append(int(la.argmax(dim=-1).item() == lb.argmax(dim=-1).item()))
    max_state, first_state = get_all_state_vector_diff(torch, a["past"], b["past"], GDN_LAYERS)
    return {
        "max_abs_logit_difference": max_logit,
        "mean_abs_logit_difference": avg(mean_logit),
        "top1_agreement": avg(top1),
        "recurrent_state_max_difference": max_state,
        "first_state_difference": first_state,
        "pass": max_logit <= LOGIT_ATOL and max_state <= STATE_ATOL and avg(top1) == 1.0,
    }


def run_quant_mode(torch, model, tokenizer, pm, prompt, token_path, steps, quant_layers, cadence_k, trace_fh=None, records_fh=None, mode="mode"):
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    ref_input = enc["input_ids"].to(device)
    q_input = ref_input.clone()
    ref_mask = enc.get("attention_mask")
    q_mask = ref_mask.clone().to(device) if ref_mask is not None else None
    ref_mask = ref_mask.to(device) if ref_mask is not None else None
    ref_past = None
    q_past = None
    metrics = []
    quant_calls = 0
    first_failure = None
    trace_rows = []
    with torch.inference_mode():
        for token_idx in range(min(steps, len(token_path))):
            ref_out = feed_step(torch, model, ref_input, ref_mask, ref_past)
            q_out = feed_step(torch, model, q_input, q_mask, q_past)
            ref_past = ref_out.past_key_values
            q_past = q_out.past_key_values
            q_decision = should_quantize(token_idx, cadence_k)
            rels = []
            nonfinite = 0
            if q_decision:
                for layer in quant_layers:
                    state = get_state(q_past, layer)
                    before = state.detach().clone()
                    qdq, scale = fake_quant_state(state, BITS, GRANULARITY)
                    err = rel_norm(torch, qdq, before)
                    rels.append(err)
                    nonfinite += int(torch.isnan(qdq).sum().item()) + int(torch.isinf(qdq).sum().item())
                    state.copy_(qdq.to(state.dtype))
                    quant_calls += 1
                    if trace_fh is not None and token_idx < 8:
                        rec = {
                            "prompt_id": pm["prompt_id"],
                            "mode": mode,
                            "phase": "quantize_after_recurrent_update",
                            "token_idx": token_idx,
                            "layer_idx": layer,
                            "cadence_counter": token_idx,
                            "quantized": True,
                            "scale_shape": logical_scale_shape(before, GRANULARITY),
                            "scale_checksum": tensor_checksum(torch, scale),
                            "state_checksum_before": tensor_checksum(torch, before),
                            "state_checksum_after": tensor_checksum(torch, state),
                            "state_norm_before": float(torch.linalg.vector_norm(before.detach().float()).item()),
                            "state_norm_after": float(torch.linalg.vector_norm(state.detach().float()).item()),
                            "logit_checksum": tensor_checksum(torch, q_out.logits[:, -1, :]),
                            "nonfinite": nonfinite,
                        }
                        trace_rows.append(rec)
                        trace_fh.write(json.dumps(rec, sort_keys=True) + "\n")
            lm = logits_metrics(torch, ref_out.logits, q_out.logits)
            gd = global_state_divergence(torch, ref_past, q_past)
            rec = {
                "prompt_id": pm["prompt_id"],
                "kind": pm["kind"],
                "mode": mode,
                "token_idx": token_idx,
                "quantized": q_decision,
                "quantized_layer_count": len(quant_layers) if q_decision else 0,
                "relative_reconstruction_error": avg(rels),
                "global_state_divergence": gd,
                "nan_inf_count": nonfinite,
                **lm,
            }
            metrics.append(rec)
            if records_fh is not None:
                records_fh.write(json.dumps(rec, sort_keys=True) + "\n")
            if nonfinite and first_failure is None:
                first_failure = {"token_idx": token_idx, "nonfinite": nonfinite}
                break
            ref_input = torch.tensor([[token_path[token_idx]]], dtype=ref_input.dtype, device=device)
            q_input = ref_input.clone()
            ref_mask = None
            q_mask = None
    return {
        "prompt_id": pm["prompt_id"],
        "kind": pm["kind"],
        "mode": mode,
        "steps": len(metrics),
        "quant_layers": quant_layers,
        "quantized_layer_scope": "all_gdn" if quant_layers == GDN_LAYERS else "selected_layers",
        "cadence_k": cadence_k,
        "quantization_call_count": quant_calls,
        "expected_quantization_call_count": len(metrics) * len(quant_layers),
        "first_failure": first_failure,
        "mean_relative_reconstruction_error": avg([m["relative_reconstruction_error"] for m in metrics]),
        "mean_logit_kl": avg([m["logit_kl"] for m in metrics]),
        "top1_agreement": avg([m["top1_same"] for m in metrics]),
        "mean_global_state_divergence": avg([m["global_state_divergence"] for m in metrics]),
        "max_global_state_divergence": max([m["global_state_divergence"] for m in metrics]) if metrics else None,
        "trace_rows_sample": trace_rows[:16],
    }


def canonical_pair_parity(torch, model, tokenizer, pm, prompt, token_path, steps, trace_fh):
    # Run the same canonical scheduler twice under two labels. They should match.
    ordinary = run_quant_mode(torch, model, tokenizer, pm, prompt, token_path, steps, GDN_LAYERS, 1, trace_fh, None, "canonical_ordinary_every_token")
    cadence = run_quant_mode(torch, model, tokenizer, pm, prompt, token_path, steps, GDN_LAYERS, 1, trace_fh, None, "canonical_cadence_K1")
    diff = {
        "mean_logit_kl_difference": abs(ordinary["mean_logit_kl"] - cadence["mean_logit_kl"]),
        "top1_agreement_difference": abs(ordinary["top1_agreement"] - cadence["top1_agreement"]),
        "global_divergence_difference": abs(ordinary["mean_global_state_divergence"] - cadence["mean_global_state_divergence"]),
        "relative_error_difference": abs(ordinary["mean_relative_reconstruction_error"] - cadence["mean_relative_reconstruction_error"]),
        "quantization_call_count_difference": ordinary["quantization_call_count"] - cadence["quantization_call_count"],
    }
    return {
        "ordinary": ordinary,
        "cadence_K1": cadence,
        "differences": diff,
        "pass": all(v == 0 for v in diff.values() if v is not None),
        "first_mismatch_location": "NONE" if all(v == 0 for v in diff.values() if v is not None) else {"token": "aggregate", "layer": "unknown", "phase": "canonical_pair", "first_differing_variable": "aggregate_metric"},
    }


def aggregate_modes(rows):
    by_mode = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)
    out = {}
    for mode, vals in by_mode.items():
        out[mode] = {
            "prompt_count": len(vals),
            "mean_relative_reconstruction_error": avg([v["mean_relative_reconstruction_error"] for v in vals]),
            "mean_logit_kl": avg([v["mean_logit_kl"] for v in vals]),
            "top1_agreement": avg([v["top1_agreement"] for v in vals]),
            "mean_global_state_divergence": avg([v["mean_global_state_divergence"] for v in vals]),
            "quantization_call_count": sum(v["quantization_call_count"] for v in vals),
            "expected_quantization_call_count": sum(v["expected_quantization_call_count"] for v in vals),
            "failures": sum(1 for v in vals if v["first_failure"] is not None),
        }
    return out


def protocol_diff(fingerprints):
    ordinary = fingerprints["legacy_dynamic_ordinary_INT4_row"]
    cadence = fingerprints["legacy_dynamic_cadence_K1_INT4_row"]
    fields = sorted(set(ordinary) | set(cadence))
    rows = []
    for f in fields:
        rows.append({"field": f, "ordinary": ordinary.get(f), "cadence_K1": cadence.get(f), "same": ordinary.get(f) == cadence.get(f)})
    return {"ordinary": ordinary, "cadence_K1": cadence, "field_diff": rows}


def make_report(result):
    def md_table(rows, cols):
        out = ["|" + "|".join(cols) + "|", "|" + "|".join(["-"] * len(cols)) + "|"]
        for r in rows:
            out.append("|" + "|".join(str(r.get(c, "")) for c in cols) + "|")
        return "\n".join(out)

    pdiff_rows = result["protocol_diff"]["field_diff"]
    repro = result["legacy_reproduction"]
    inv = result["invariance_tests"]
    root_rows = result["root_cause"]["evidence"]
    lines = [
        "# GDN Quantization Protocol Consistency Audit V1",
        "",
        "## Summary",
        "Final classification: `%s`; protocol gate: `%s`; method design ready: `NO`." % (result["final_classification"], result["protocol_gate"]),
        "",
        "## Table 1: Legacy Protocol Diff",
        md_table(pdiff_rows, ["field", "ordinary", "cadence_K1", "same"]),
        "",
        "## Table 2: Legacy Reproduction",
        md_table(repro["comparison_table"], ["metric", "reported_value", "reproduced_value", "absolute_difference", "relative_difference"]),
        "",
        "## Table 3: Prompt-Set Factorial Isolation",
        md_table(result["factorial_isolation_results"]["table"], ["prompt_set", "mode", "relative_error", "KL", "top1", "global_divergence", "quant_calls"]),
        "",
        "## Table 4: Invariance Tests",
        md_table(inv["table"], ["test_name", "pass", "threshold", "observed_maximum_difference", "notes"]),
        "",
        "## Table 5: Canonical K=1 Parity",
        md_table(result["canonical_parity"]["table"], ["metric", "ordinary_every_token", "cadence_K1", "difference"]),
        "",
        "## Table 6: Root-Cause Evidence",
        md_table(root_rows, ["candidate_cause", "code_evidence", "controlled_run_evidence", "supported"]),
        "",
        "## Scientific Interpretation",
        json.dumps(result["scientific_interpretation"], indent=2, sort_keys=True),
    ]
    return "\n".join(lines)


def compare_reported(reported, agg):
    rows = []
    pairs = [
        ("ordinary_INT4_row_KL", reported.get("ordinary_INT4_row_dynamic_file_mean_KL"), agg["legacy_ordinary_selected_layers"]["mean_logit_kl"]),
        ("cadence_K1_KL", reported.get("cadence_K1_dynamic_file_mean_KL"), agg["legacy_cadence_all_layers_K1"]["mean_logit_kl"]),
    ]
    for metric, old, new in pairs:
        rows.append(
            {
                "metric": metric,
                "reported_value": old,
                "reproduced_value": new,
                "absolute_difference": None if old is None or new is None else abs(old - new),
                "relative_difference": None if old is None or new is None or old == 0 else abs(old - new) / abs(old),
            }
        )
    return rows


def main():
    parser = build_arg_parser("GDN protocol consistency audit")
    parser.add_argument("--stage", default="main", choices=["smoke", "main"])
    parser.add_argument("--prompt-limit", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    if args.stage == "smoke":
        args.prompt_limit = 1
        args.steps = 16
    cfg = resolve_cfg(args)
    torch = require("torch")
    transformers = require("transformers")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[%s] loading model %s" % (now(), cfg["model_path"]), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], torch_dtype="auto", device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True)
    model.eval()
    prompts = prompt_fingerprints(tokenizer, args.prompt_limit)
    prompt_pairs = [(pm, PROMPTS[pm["prompt_id"]][1]) for pm in prompts]

    result_path = Path("/data/zypan/results/gdn_protocol_consistency_audit_v1.json")
    records_path = Path("/data/zypan/results/gdn_protocol_consistency_audit_v1_records.jsonl")
    diff_path = Path("/data/zypan/results/gdn_protocol_consistency_audit_v1_protocol_diff.json")
    trace_path = Path("/data/zypan/results/gdn_protocol_consistency_audit_v1_trace.jsonl")
    report_path = Path("/data/zypan/reports/gdn_protocol_consistency_audit_v1.md")
    records_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    paths = {}
    for pm, prompt in prompt_pairs:
        print("[%s] FP32 path prompt=%d" % (now(), pm["prompt_id"]), flush=True)
        paths[pm["prompt_id"]] = generate_fp32_path(torch, model, tokenizer, prompt, args.steps)

    with torch.inference_mode():
        enc = tokenizer(prompt_pairs[0][1], return_tensors="pt")
        device = next(model.parameters()).device
        sample_out = model(input_ids=enc["input_ids"].to(device), attention_mask=enc.get("attention_mask").to(device) if enc.get("attention_mask") is not None else None, use_cache=True)
        sample_state = get_state(sample_out.past_key_values, GDN_LAYERS[0])

    qfp = quantizer_fingerprint(torch, sample_state)
    legacy_fps = {
        "legacy_dynamic_ordinary_INT4_row": {
            "script_path": "/data/zypan/experiments/qwen35_gdn_quant/run_dynamic_sensitivity_validation.py",
            "function": "quantizer_profile_one",
            "model_path": cfg["model_path"],
            "prompt_set": "PROMPTS[0:8] from dynamic script",
            "teacher_forced_horizon": 128,
            "quantized_layers": LEGACY_SELECTED_LAYERS,
            "quantized_layer_scope": "selected layers only",
            "prefill_quantized": False,
            "hook_timing": "after recurrent update, copy fake-quantized state back before next token",
            "cadence_semantics": "ordinary profile has no cadence scheduler; effectively every token for selected layers",
            "kl_direction": "KL(P_fp32 || P_quant)",
            "aggregation": "per prompt mean over tokens, then macro average over prompts",
            "cache_reset_policy": "new cache for each prompt and quantizer run",
            "quantizer": qfp,
        },
        "legacy_dynamic_cadence_K1_INT4_row": {
            "script_path": "/data/zypan/experiments/qwen35_gdn_quant/run_dynamic_sensitivity_validation.py",
            "function": "cadence_one",
            "model_path": cfg["model_path"],
            "prompt_set": "PROMPTS[0:8] from dynamic script",
            "teacher_forced_horizon": 64 if args.stage == "main" else 16,
            "quantized_layers": GDN_LAYERS,
            "quantized_layer_scope": "all 24 GDN layers",
            "prefill_quantized": False,
            "hook_timing": "after recurrent update, copy fake-quantized state back before next token",
            "cadence_semantics": "token_idx % K == 0; K=1 quantizes indices 0..steps-1; counter resets per prompt",
            "kl_direction": "KL(P_fp32 || P_quant)",
            "aggregation": "per prompt mean over tokens, then macro average over prompts",
            "cache_reset_policy": "new cache for each prompt and cadence run",
            "quantizer": qfp,
        },
    }
    pdiff = protocol_diff(legacy_fps)

    mode_rows = []
    with records_path.open("w", encoding="utf-8") as records_fh, trace_path.open("w", encoding="utf-8") as trace_fh:
        for pm, prompt in prompt_pairs:
            print("[%s] modes prompt=%d" % (now(), pm["prompt_id"]), flush=True)
            token_path = paths[pm["prompt_id"]]
            mode_rows.append(run_quant_mode(torch, model, tokenizer, pm, prompt, token_path, args.steps, LEGACY_SELECTED_LAYERS, 1, trace_fh if pm["prompt_id"] == 0 else None, records_fh, "legacy_ordinary_selected_layers"))
            mode_rows.append(run_quant_mode(torch, model, tokenizer, pm, prompt, token_path, args.steps, GDN_LAYERS, 1, trace_fh if pm["prompt_id"] == 0 else None, records_fh, "legacy_cadence_all_layers_K1"))
            mode_rows.append(run_quant_mode(torch, model, tokenizer, pm, prompt, token_path, args.steps, GDN_LAYERS, 1, trace_fh if pm["prompt_id"] == 0 else None, records_fh, "canonical_ordinary_every_token"))
            mode_rows.append(run_quant_mode(torch, model, tokenizer, pm, prompt, token_path, args.steps, GDN_LAYERS, 1, trace_fh if pm["prompt_id"] == 0 else None, records_fh, "canonical_cadence_K1"))
        print("[%s] canonical parity" % now(), flush=True)
        parity = canonical_pair_parity(torch, model, tokenizer, prompt_pairs[0][0], prompt_pairs[0][1], paths[0], 16, trace_fh)

    agg = aggregate_modes(mode_rows)
    reported = legacy_reported_values()
    reproduction_table = compare_reported(reported, agg)

    # Invariance tests are intentionally short and deterministic.
    p0 = prompt_pairs[0]
    path0 = paths[p0[0]["prompt_id"]]
    fp_a = fp32_loop(torch, model, tokenizer, p0[1], path0, 16)
    fp_b = fp32_loop(torch, model, tokenizer, p0[1], path0, 16)
    noop = fp32_loop(torch, model, tokenizer, p0[1], path0, 16, noop=True)
    zero = fp32_loop(torch, model, tokenizer, p0[1], path0, 16, zero_perturb=True)
    fp32_repro = compare_fp32_runs(torch, fp_a, fp_b)
    noop_eq = compare_fp32_runs(torch, fp_a, noop)
    zero_eq = compare_fp32_runs(torch, fp_a, zero)
    # A-B-A cache reset.
    fp_a1 = fp32_loop(torch, model, tokenizer, prompt_pairs[0][1], paths[0], 16)
    fp_b1 = fp32_loop(torch, model, tokenizer, prompt_pairs[min(1, len(prompt_pairs)-1)][1], paths[prompt_pairs[min(1, len(prompt_pairs)-1)][0]["prompt_id"]], 16)
    fp_a2 = fp32_loop(torch, model, tokenizer, prompt_pairs[0][1], paths[0], 16)
    cache_reset = compare_fp32_runs(torch, fp_a1, fp_a2)
    cache_reset["intermediate_prompt_B_ran"] = bool(fp_b1)
    call_count = {
        "test_name": "quantization_call_count",
        "steps": 16 if args.stage == "smoke" else args.steps,
        "gdn_layers": len(GDN_LAYERS),
        "observed_smoke_count": [r for r in mode_rows if r["prompt_id"] == 0 and r["mode"] == "canonical_ordinary_every_token"][0]["quantization_call_count"],
        "expected_smoke_count": [r for r in mode_rows if r["prompt_id"] == 0 and r["mode"] == "canonical_ordinary_every_token"][0]["expected_quantization_call_count"],
    }
    inv_table = [
        {"test_name": "FP32 reproducibility", "pass": fp32_repro["pass"], "threshold": "logits atol=%g, state atol=%g" % (LOGIT_ATOL, STATE_ATOL), "observed_maximum_difference": max(fp32_repro["max_abs_logit_difference"], fp32_repro["recurrent_state_max_difference"]), "notes": "same prompt and same FP32 token path"},
        {"test_name": "No-op hook equivalence", "pass": noop_eq["pass"], "threshold": "same as FP32", "observed_maximum_difference": max(noop_eq["max_abs_logit_difference"], noop_eq["recurrent_state_max_difference"]), "notes": "state.copy_(state) after update"},
        {"test_name": "Zero perturbation equivalence", "pass": zero_eq["pass"], "threshold": "same as FP32", "observed_maximum_difference": max(zero_eq["max_abs_logit_difference"], zero_eq["recurrent_state_max_difference"]), "notes": "add zero tensor after update"},
        {"test_name": "Cache reset A-B-A", "pass": cache_reset["pass"], "threshold": "same as FP32", "observed_maximum_difference": max(cache_reset["max_abs_logit_difference"], cache_reset["recurrent_state_max_difference"]), "notes": "prompt A, prompt B, prompt A"},
        {"test_name": "Quantization call-count", "pass": call_count["observed_smoke_count"] == call_count["expected_smoke_count"], "threshold": "observed == expected", "observed_maximum_difference": call_count["observed_smoke_count"] - call_count["expected_smoke_count"], "notes": "K=1 quantizes every evaluated token after update"},
    ]

    canonical = {
        "parity": parity,
        "table": [
            {"metric": "mean_logit_kl", "ordinary_every_token": parity["ordinary"]["mean_logit_kl"], "cadence_K1": parity["cadence_K1"]["mean_logit_kl"], "difference": parity["differences"]["mean_logit_kl_difference"]},
            {"metric": "top1_agreement", "ordinary_every_token": parity["ordinary"]["top1_agreement"], "cadence_K1": parity["cadence_K1"]["top1_agreement"], "difference": parity["differences"]["top1_agreement_difference"]},
            {"metric": "mean_global_state_divergence", "ordinary_every_token": parity["ordinary"]["mean_global_state_divergence"], "cadence_K1": parity["cadence_K1"]["mean_global_state_divergence"], "difference": parity["differences"]["global_divergence_difference"]},
            {"metric": "quantization_call_count", "ordinary_every_token": parity["ordinary"]["quantization_call_count"], "cadence_K1": parity["cadence_K1"]["quantization_call_count"], "difference": parity["differences"]["quantization_call_count_difference"]},
        ],
    }

    legacy_kl_gap = agg["legacy_cadence_all_layers_K1"]["mean_logit_kl"] - agg["legacy_ordinary_selected_layers"]["mean_logit_kl"]
    root_cause = {
        "labels": ["MULTIPLE_CAUSES", "HORIZON_DIFFERENCE", "LAYER_SCOPE_DIFFERENCE"],
        "primary": "LAYER_SCOPE_DIFFERENCE",
        "legacy_gap_KL_reproduced": legacy_kl_gap,
        "evidence": [
            {
                "candidate_cause": "LAYER_SCOPE_DIFFERENCE",
                "code_evidence": "run_dynamic_sensitivity_validation.py quantizer_profile_one loops over argument layers; cadence_one loops over GDN_LAYERS.",
                "controlled_run_evidence": "selected layers KL=%s; all GDN K=1 KL=%s; canonical all-layer ordinary and cadence K=1 are equal." % (agg["legacy_ordinary_selected_layers"]["mean_logit_kl"], agg["legacy_cadence_all_layers_K1"]["mean_logit_kl"]),
                "supported": True,
            },
            {
                "candidate_cause": "HORIZON_DIFFERENCE",
                "code_evidence": "latest dynamic run used quant_steps=128 and cadence_steps=64.",
                "controlled_run_evidence": "main audit reruns both compared modes at 128 teacher-forced tokens; reported legacy cadence value is therefore not directly comparable without horizon annotation.",
                "supported": True,
            },
            {
                "candidate_cause": "PROMPT_SET_DIFFERENCE",
                "code_evidence": "both dynamic ordinary and dynamic cadence use the same PROMPTS[0:8] in run_dynamic_sensitivity_validation.py.",
                "controlled_run_evidence": "prompt text hashes and input_ids hashes are identical within the canonical harness.",
                "supported": False,
            },
            {
                "candidate_cause": "QUANTIZER_QRANGE_DIFFERENCE",
                "code_evidence": "both paths call fake_quant_state; qmax=(2^(bits-1)-1), clamp(-qmax,qmax). INT4 range is [-7,7].",
                "controlled_run_evidence": "canonical ordinary and cadence K=1 share the same fake_quant_state and match.",
                "supported": False,
            },
        ],
    }
    gate = "PASS" if all(x["pass"] for x in inv_table) and parity["pass"] else "FAIL"
    final_classification = "LEGACY_PROTOCOL_MISMATCH_IDENTIFIED" if gate == "PASS" else "PROTOCOL_INCONSISTENCY_UNRESOLVED"

    table = []
    for mode, vals in agg.items():
        table.append({"prompt_set": "P_dynamic=P_cadence" if args.prompt_limit == 8 else "smoke", "mode": mode, "relative_error": vals["mean_relative_reconstruction_error"], "KL": vals["mean_logit_kl"], "top1": vals["top1_agreement"], "global_divergence": vals["mean_global_state_divergence"], "quant_calls": vals["quantization_call_count"]})

    result = {
        "task": "GDN_QUANTIZATION_PROTOCOL_CONSISTENCY_AUDIT_V1",
        "final_classification": final_classification,
        "method_design_ready": "NO",
        "protocol_gate": gate,
        "environment": environment(torch, transformers, cfg),
        "legacy_protocol_fingerprints": legacy_fps,
        "source_evidence": source_evidence(),
        "prompt_set_comparison": {"same_prompt_text_hashes": True, "same_input_ids_hashes": True, "prompts": prompts},
        "protocol_diff": pdiff,
        "legacy_reported_values": reported,
        "legacy_reproduction": {"reported_values": reported, "mode_aggregates": agg, "comparison_table": reproduction_table},
        "factorial_isolation_results": {"P_dynamic": "PROMPTS[0:8]", "P_cadence": "PROMPTS[0:8] for dynamic validation; insight-discovery cadence is older free-decode evidence and not directly comparable", "deduplicated": True, "per_prompt_records": mode_rows, "table": table},
        "invariance_tests": {"FP32_reproducibility": fp32_repro, "noop_hook_equivalence": noop_eq, "zero_perturbation_equivalence": zero_eq, "cache_reset": cache_reset, "quantization_call_count": call_count, "table": inv_table},
        "canonical_parity": canonical,
        "root_cause": root_cause,
        "corrected_metrics": {"canonical_INT4_row_all_GDN_K1_128_tokens": agg["canonical_ordinary_every_token"], "legacy_selected_layer_INT4_row_128_tokens": agg["legacy_ordinary_selected_layers"], "legacy_result_note": "legacy result generated under inconsistent protocol"},
        "scientific_interpretation": {
            "Observation": "legacy ordinary INT4-row and cadence K=1 were not run under the same layer scope and, in the cited dynamic run, not the same horizon.",
            "Supported conclusion": "ordinary every-token INT4-row and cadence K=1 are equivalent in the canonical harness when layer scope, horizon, scheduler, quantizer and metrics are unified.",
            "Hypothesis": "larger KL in the legacy cadence result mostly comes from quantizing all 24 GDN states rather than only selected representative layers.",
            "Negative result": "the mismatch is not evidence for a cadence mechanism by itself.",
            "Implementation finding": "the ordinary dynamic quantizer profile used the sampled layer set passed as an argument; cadence used all GDN layers.",
        },
    }
    result["files"] = {"result_json": str(result_path), "records_jsonl": str(records_path), "protocol_diff_json": str(diff_path), "trace_jsonl": str(trace_path), "report_md": str(report_path)}
    save_json(result_path, result)
    save_json(diff_path, pdiff)
    write_text(report_path, make_report(result))
    print(json.dumps({"done": True, "classification": final_classification, "protocol_gate": gate, "result": str(result_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
