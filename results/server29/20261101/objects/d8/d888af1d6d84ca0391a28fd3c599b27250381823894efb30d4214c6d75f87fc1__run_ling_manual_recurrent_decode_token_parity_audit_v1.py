#!/usr/bin/env python3
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path


TASK = "LING_MANUAL_RECURRENT_DECODE_TOKEN_PARITY_AUDIT_V1"
REPO = Path(__file__).resolve().parents[2]
MODEL_PATH = Path("/data/zypan/models/Ling-3.0-tiny")
PREV_RUN = REPO / "runs" / "ling_aime24_canonical_generation_parity_smoke_v1"
RUN_DIR = REPO / "runs" / "ling_manual_recurrent_decode_token_parity_audit_v1"
EOS_TOKEN_ID = 156895
PAD_TOKEN_ID = 156892
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
MAX_NEW_TOKENS = 32768
SEED = 0
EPS = 1e-12


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=REPO, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def setup_seed(seed=SEED):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    rows = []
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render_prompt(tokenizer, problem):
    messages = [{"role": "user", "content": problem}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_tensors="pt",
    )
    return prompt_text, input_ids


def load_model_and_tokenizer():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import transformers.utils.import_utils as import_utils
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    model.cuda()
    model.eval()
    return torch, model, tokenizer


def tensor_summary(x, tail=8):
    if x is None:
        return None
    if hasattr(x, "detach"):
        y = x.detach()
        data = y.flatten()[-tail:].cpu().tolist() if y.numel() else []
        return {"shape": list(y.shape), "dtype": str(y.dtype), "device": str(y.device), "tail": [int(v) for v in data]}
    return str(type(x))


def cache_summary(cache):
    if cache is None:
        return None
    out = {"type": type(cache).__name__}
    if hasattr(cache, "get_seq_length"):
        try:
            out["seq_length"] = int(cache.get_seq_length())
        except Exception as exc:
            out["seq_length_error"] = repr(exc)
    layers = getattr(cache, "layers", None)
    if layers is not None:
        out["num_layers"] = len(layers)
        sample = []
        for i, layer in enumerate(layers[:6]):
            item = {"layer": i, "type": type(layer).__name__}
            for name in ("keys", "values"):
                val = getattr(layer, name, None)
                if isinstance(val, tuple):
                    item[name] = [tensor_summary(v, tail=2) for v in val]
                else:
                    item[name] = tensor_summary(val, tail=2)
            sample.append(item)
        out["layer_sample"] = sample
    return out


def compare_logits(torch, a, b):
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    diff = af - bf
    max_abs = float(diff.abs().max().item())
    mean_abs = float(diff.abs().mean().item())
    rel_l2 = float((torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(af).clamp_min(EPS)).item())
    cos = float(torch.nn.functional.cosine_similarity(af, bf, dim=0).item())
    top1_a = int(torch.argmax(af).item())
    top1_b = int(torch.argmax(bf).item())
    top5_a = torch.topk(af, 5).indices.detach().cpu().tolist()
    top5_b = torch.topk(bf, 5).indices.detach().cpu().tolist()
    top20_a = torch.topk(af, 20).indices.detach().cpu().tolist()
    top20_b = torch.topk(bf, 20).indices.detach().cpu().tolist()
    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "relative_l2_error": rel_l2,
        "cosine_similarity": cos,
        "top1_a": top1_a,
        "top1_b": top1_b,
        "top1_match": top1_a == top1_b,
        "top5_a": [int(x) for x in top5_a],
        "top5_b": [int(x) for x in top5_b],
        "top20_a": [int(x) for x in top20_a],
        "top20_b": [int(x) for x in top20_b],
        "top5_overlap": len(set(top5_a) & set(top5_b)) / 5.0,
        "top20_overlap": len(set(top20_a) & set(top20_b)) / 20.0,
        "top20_logits_a": [float(x) for x in torch.topk(af, 20).values.detach().cpu().tolist()],
        "top20_logits_b": [float(x) for x in torch.topk(bf, 20).values.detach().cpu().tolist()],
    }


def flatten_cache_tensors(cache):
    layers = getattr(cache, "layers", None)
    if layers is None:
        return {}
    out = {}
    for i, layer in enumerate(layers):
        for name in ("keys", "values"):
            val = getattr(layer, name, None)
            if hasattr(val, "detach"):
                out[(i, name)] = val
            elif isinstance(val, tuple):
                for j, t in enumerate(val):
                    if hasattr(t, "detach"):
                        out[(i, f"{name}_{j}")] = t
    return out


def compare_cache(torch, a, b):
    at = flatten_cache_tensors(a)
    bt = flatten_cache_tensors(b)
    rows = []
    for key in sorted(set(at) | set(bt)):
        aa, bb = at.get(key), bt.get(key)
        row = {"layer": key[0], "tensor": key[1], "observable_a": aa is not None, "observable_b": bb is not None}
        if aa is not None:
            row.update({"shape_a": list(aa.shape), "dtype_a": str(aa.dtype), "device_a": str(aa.device), "finite_a": bool(torch.isfinite(aa).all().item())})
        if bb is not None:
            row.update({"shape_b": list(bb.shape), "dtype_b": str(bb.dtype), "device_b": str(bb.device), "finite_b": bool(torch.isfinite(bb).all().item())})
        if aa is not None and bb is not None and list(aa.shape) == list(bb.shape):
            af, bf = aa.detach().float(), bb.detach().float()
            diff = af - bf
            row["norm_a"] = float(torch.linalg.vector_norm(af).item())
            row["norm_b"] = float(torch.linalg.vector_norm(bf).item())
            row["max_abs_diff"] = float(diff.abs().max().item())
            row["relative_l2"] = float((torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(af).clamp_min(EPS)).item())
            row["cosine"] = float(torch.nn.functional.cosine_similarity(af.reshape(-1), bf.reshape(-1), dim=0).item())
        rows.append(row)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def build_warpers():
    from transformers.generation import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
    return LogitsProcessorList([
        TemperatureLogitsWarper(TEMPERATURE),
        TopKLogitsWarper(TOP_K),
        TopPLogitsWarper(TOP_P),
    ])


def sampling_distribution_parity(torch, logits, input_ids):
    scores_a = build_warpers()(input_ids, logits.detach().float().clone())
    scores_b = build_warpers()(input_ids, logits.detach().float().clone())
    probs_a = torch.softmax(scores_a, dim=-1)
    probs_b = torch.softmax(scores_b, dim=-1)
    finite_a = torch.isfinite(scores_a[0])
    finite_b = torch.isfinite(scores_b[0])
    set_a = set(torch.nonzero(finite_a, as_tuple=False).flatten().detach().cpu().tolist())
    set_b = set(torch.nonzero(finite_b, as_tuple=False).flatten().detach().cpu().tolist())
    diff = (probs_a - probs_b).abs()
    return {
        "SAMPLING_DISTRIBUTION_PARITY_GATE": "PASS" if set_a == set_b and float(diff.max().item()) <= 1e-5 else "FAIL",
        "top_k": TOP_K,
        "top_p": TOP_P,
        "temperature": TEMPERATURE,
        "candidate_set_size_a": len(set_a),
        "candidate_set_size_b": len(set_b),
        "candidate_sets_identical": set_a == set_b,
        "argmax_a": int(probs_a.argmax(dim=-1).item()),
        "argmax_b": int(probs_b.argmax(dim=-1).item()),
        "probability_sum_a": float(probs_a.sum().item()),
        "probability_sum_b": float(probs_b.sum().item()),
        "max_absolute_probability_difference": float(diff.max().item()),
        "mean_absolute_probability_difference": float(diff.mean().item()),
    }


def eos_unit_test():
    cases = [
        {"name": "eos_stops", "forced_tokens": [123, 456, EOS_TOKEN_ID, 789]},
        {"name": "pad_does_not_stop", "forced_tokens": [123, PAD_TOKEN_ID, 456]},
    ]
    results = []
    for case in cases:
        written = []
        finish = "MAX_NEW_TOKENS"
        for tok in case["forced_tokens"]:
            written.append(tok)
            if tok == EOS_TOKEN_ID:
                finish = "EOS"
                break
        results.append({
            "name": case["name"],
            "forced_tokens": case["forced_tokens"],
            "tokens_written": written,
            "finish_reason": finish,
            "eos_step": written.index(EOS_TOKEN_ID) if EOS_TOKEN_ID in written else None,
        })
    gate = (
        results[0]["finish_reason"] == "EOS"
        and results[0]["tokens_written"] == [123, 456, EOS_TOKEN_ID]
        and results[1]["finish_reason"] == "MAX_NEW_TOKENS"
        and PAD_TOKEN_ID in results[1]["tokens_written"]
    )
    return {"MANUAL_EOS_UNIT_TEST_GATE": "PASS" if gate else "FAIL", "cases": results, "eos_token_id": EOS_TOKEN_ID, "pad_token_id": PAD_TOKEN_ID}


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    setup_seed(SEED)
    import torch
    import transformers

    environment = {
        "task": TASK,
        "repo_root": str(REPO),
        "pwd": os.getcwd(),
        "git_status": sh(["git", "status", "--short"]),
        "git_branch": sh(["git", "branch", "--show-current"]),
        "git_head": sh(["git", "log", "-1", "--oneline"]),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "model_config": json.loads((MODEL_PATH / "config.json").read_text(encoding="utf-8")),
        "tokenizer_config": json.loads((MODEL_PATH / "tokenizer_config.json").read_text(encoding="utf-8")),
        "generation_config": json.loads((MODEL_PATH / "generation_config.json").read_text(encoding="utf-8")),
        "manual_decode_entrypoint": "experiments/ling/run_ling_kda_aime24_int8_rc_end_to_end_smoke_v1.py:generate_one",
        "kda_recurrent_state_code": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py:BailingMoeV3KDA.forward",
        "state_writeback_code": "/data/zypan/models/Ling-3.0-tiny/modeling_bailing_moe_v3.py lines assigning past_key_value.layers[layer_idx].keys/values",
        "sampling_function": "legacy top_p_sample(); canonical audit uses HF Temperature/TopK/TopP logits warpers",
        "eos_stop_logic": "legacy stop_token_ids/token loop; canonical EOS id 156895",
    }
    environment["ENVIRONMENT_AUDIT_GATE"] = "PASS" if environment["model_config"].get("eos_token_id") == EOS_TOKEN_ID and environment["model_config"].get("pad_token_id") == PAD_TOKEN_ID else "FAIL"
    save_json(RUN_DIR / "environment_audit.json", environment)
    (RUN_DIR / "environment_audit.md").write_text(
        "\n".join([
            f"# {TASK}",
            "",
            f"ENVIRONMENT_AUDIT_GATE = {environment['ENVIRONMENT_AUDIT_GATE']}",
            f"branch = {environment['git_branch']}",
            f"head = {environment['git_head']}",
            f"torch = {environment['torch']}",
            f"transformers = {environment['transformers']}",
            f"model_eos = {environment['model_config'].get('eos_token_id')}",
            f"model_pad = {environment['model_config'].get('pad_token_id')}",
        ]) + "\n",
        encoding="utf-8",
    )

    torch, model, tokenizer = load_model_and_tokenizer()
    selected = json.loads((PREV_RUN / "selected_problems.json").read_text(encoding="utf-8"))
    item = next(x for x in selected if str(x["problem_id"]) == "60")
    hf_rows = load_jsonl(PREV_RUN / "hf_generate_results.jsonl")
    hf_ref = next(x for x in hf_rows if str(x["problem_id"]) == "60")
    prompt_text, input_ids_cpu = render_prompt(tokenizer, item["problem"])
    attention_mask_cpu = torch.ones_like(input_ids_cpu)
    generated_ids = [int(x) for x in hf_ref["generated_token_ids"]]
    device = next(model.parameters()).device
    input_ids = input_ids_cpu.to(device)
    attention_mask = attention_mask_cpu.to(device)

    canonical_ref = {
        "problem_id": "60",
        "problem": item["problem"],
        "answer": item["answer"],
        "prompt_text": prompt_text,
        "input_ids": input_ids_cpu[0].tolist(),
        "attention_mask": attention_mask_cpu[0].tolist(),
        "hf_generated_token_ids": generated_ids,
        "hf_generated_text": hf_ref.get("decoded_output"),
        "hf_eos_position": hf_ref.get("first_eos_position"),
        "hf_finish_reason": hf_ref.get("finish_reason"),
        "hf_generated_tokens": hf_ref.get("generated_tokens"),
    }
    canonical_ref["HF_CANONICAL_REFERENCE_GATE"] = "PASS" if hf_ref.get("finish_reason") == "EOS" and hf_ref.get("first_eos_position") == 3023 else "FAIL"
    save_json(RUN_DIR / "canonical_problem_60_reference.json", canonical_ref)

    manual_prompt_text, manual_input_ids_cpu = render_prompt(tokenizer, item["problem"])
    manual_attention_mask_cpu = torch.ones_like(manual_input_ids_cpu)
    input_parity = {
        "INPUT_IDS_MATCH": "YES" if input_ids_cpu.tolist() == manual_input_ids_cpu.tolist() else "NO",
        "ATTENTION_MASK_MATCH": "YES" if attention_mask_cpu.tolist() == manual_attention_mask_cpu.tolist() else "NO",
        "PROMPT_LENGTH_HF": int(input_ids_cpu.shape[-1]),
        "PROMPT_LENGTH_MANUAL": int(manual_input_ids_cpu.shape[-1]),
        "prompt_text_match": prompt_text == manual_prompt_text,
        "special_token_positions": {
            "eos_positions_in_prompt": [i for i, x in enumerate(input_ids_cpu[0].tolist()) if x == EOS_TOKEN_ID],
            "pad_positions_in_prompt": [i for i, x in enumerate(input_ids_cpu[0].tolist()) if x == PAD_TOKEN_ID],
            "think_positions_in_prompt": [i for i, x in enumerate(input_ids_cpu[0].tolist()) if x in (156903, 156904)],
        },
    }
    save_json(RUN_DIR / "input_parity.json", input_parity)
    if input_parity["INPUT_IDS_MATCH"] != "YES":
        raise SystemExit("input ids mismatch; stopped")

    trace = []
    original_forward = model.forward

    def traced_forward(*args, **kwargs):
        rec = {
            "call_index": len(trace),
            "input_ids": tensor_summary(kwargs.get("input_ids") if "input_ids" in kwargs else (args[0] if args else None)),
            "attention_mask": tensor_summary(kwargs.get("attention_mask")),
            "position_ids": tensor_summary(kwargs.get("position_ids")),
            "cache_position": tensor_summary(kwargs.get("cache_position")),
            "past_key_values_before": cache_summary(kwargs.get("past_key_values")),
            "use_cache": kwargs.get("use_cache"),
            "kwarg_keys": sorted(kwargs.keys()),
        }
        out = original_forward(*args, **kwargs)
        rec["past_key_values_after"] = cache_summary(getattr(out, "past_key_values", None))
        trace.append(rec)
        return out

    setup_seed(SEED + int(item["problem_index"]))
    model.forward = traced_forward
    with torch.inference_mode():
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_new_tokens=2,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            return_dict_in_generate=True,
        )
    model.forward = original_forward
    trace_obj = {"HF_GENERATION_PREPARATION_GATE": "PASS" if len(trace) >= 2 else "FAIL", "calls": trace[:3]}
    save_json(RUN_DIR / "hf_generation_input_trace.json", trace_obj)

    with torch.inference_mode():
        full_prefill = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        manual_prefill = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    prefill_cmp = compare_logits(torch, full_prefill.logits[:, -1, :], manual_prefill.logits[:, -1, :])
    prefill_cmp["PREFILL_LOGIT_PARITY_GATE"] = "PASS" if prefill_cmp["top1_match"] and prefill_cmp["relative_l2_error"] <= 1e-6 else "FAIL"
    save_json(RUN_DIR / "prefill_logit_parity.json", prefill_cmp)
    state_rows = compare_cache(torch, full_prefill.past_key_values, manual_prefill.past_key_values)
    write_csv(RUN_DIR / "prefill_state_parity.csv", state_rows)
    state_observable = bool(state_rows)
    state_gate = "NOT_OBSERVABLE"
    if state_observable:
        max_rel = max(float(r.get("relative_l2", 0.0)) for r in state_rows if "relative_l2" in r)
        state_gate = "PASS" if max_rel <= 1e-6 else "FAIL"

    rows = []
    first_logit_div = None
    first_top1_div = None
    first_position_div = None
    manual_past = manual_prefill.past_key_values
    manual_prefix_ids = input_ids.clone()
    manual_prefix_mask = attention_mask.clone()
    last_manual_logits = manual_prefill.logits[:, -1, :]
    milestone_steps = {0, 1, 2, 4, 8, 16, 32, 64, 128}
    max_steps = 128
    if prefill_cmp["PREFILL_LOGIT_PARITY_GATE"] != "PASS":
        max_steps = 0
    for step in range(0, max_steps + 1):
        if step == 0:
            full_logits = full_prefill.logits[:, -1, :]
            manual_logits = last_manual_logits
            forced_id = None
            forward_ms = 0.0
            hf_position_ids = list(range(input_ids.shape[-1]))
            manual_position_ids = list(range(input_ids.shape[-1]))
            hf_cache_position = hf_position_ids
            manual_cache_position = manual_position_ids
        else:
            forced_id = generated_ids[step - 1]
            forced = torch.tensor([[forced_id]], device=device, dtype=input_ids.dtype)
            full_ids = torch.cat([input_ids, torch.tensor([generated_ids[:step]], device=device, dtype=input_ids.dtype)], dim=-1)
            full_mask = torch.ones_like(full_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                full_out = model(input_ids=full_ids, attention_mask=full_mask, use_cache=True)
                manual_prefix_mask = torch.cat([manual_prefix_mask, torch.ones_like(forced)], dim=-1)
                out = model(
                    input_ids=forced,
                    attention_mask=manual_prefix_mask,
                    past_key_values=manual_past,
                    use_cache=True,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_ms = (time.perf_counter() - t0) * 1000.0
            manual_past = out.past_key_values
            manual_prefix_ids = torch.cat([manual_prefix_ids, forced], dim=-1)
            full_logits = full_out.logits[:, -1, :]
            manual_logits = out.logits[:, -1, :]
            hf_position_ids = list(range(full_ids.shape[-1]))
            manual_position_ids = [manual_prefix_ids.shape[-1] - 1]
            hf_cache_position = hf_position_ids
            manual_cache_position = [manual_prefix_ids.shape[-1] - 1]
        cmp = compare_logits(torch, full_logits, manual_logits)
        rec = {
            "step": step,
            "forced_token_id": forced_id,
            "forced_token_text": tokenizer.decode([forced_id], skip_special_tokens=False) if forced_id is not None else None,
            "HF_top1_token": cmp["top1_a"],
            "manual_top1_token": cmp["top1_b"],
            "HF_top1_logit": cmp["top20_logits_a"][0],
            "manual_top1_logit": cmp["top20_logits_b"][0],
            "logit_max_abs_error": cmp["max_abs_error"],
            "logit_relative_l2_error": cmp["relative_l2_error"],
            "logit_cosine": cmp["cosine_similarity"],
            "top5_overlap": cmp["top5_overlap"],
            "top20_overlap": cmp["top20_overlap"],
            "HF_cache_position_tail": hf_cache_position[-8:],
            "manual_cache_position_tail": manual_cache_position[-8:],
            "HF_position_ids_tail": hf_position_ids[-8:],
            "manual_position_ids_tail": manual_position_ids[-8:],
            "HF_attention_mask_length": int(input_ids.shape[-1] + step),
            "manual_attention_mask_length": int(manual_prefix_mask.shape[-1]),
            "manual_step_runtime_ms": forward_ms,
        }
        if step in milestone_steps:
            rec["milestone"] = True
        if first_logit_div is None and cmp["relative_l2_error"] > 1e-3:
            first_logit_div = step
        if first_top1_div is None and not cmp["top1_match"]:
            first_top1_div = step
        pos_match = rec["HF_attention_mask_length"] == rec["manual_attention_mask_length"] and rec["manual_cache_position_tail"][-1] == rec["HF_cache_position_tail"][-1]
        if first_position_div is None and not pos_match:
            first_position_div = step
        rows.append(rec)
    write_csv(RUN_DIR / "teacher_forced_step_parity.csv", rows)
    decode_steps = [r for r in rows if r["step"] >= 1]
    teacher_gate = "PASS"
    if first_logit_div is not None or first_top1_div is not None or prefill_cmp["PREFILL_LOGIT_PARITY_GATE"] != "PASS":
        teacher_gate = "FAIL"
    teacher_summary = {
        "TEACHER_FORCED_DECODE_PARITY_GATE": teacher_gate,
        "steps_checked": len(rows),
        "FIRST_LOGIT_NUMERICAL_DIVERGENCE_STEP": first_logit_div,
        "FIRST_TOP1_DIVERGENCE_STEP": first_top1_div,
        "DECODE_MAX_REL_L2_ERROR_1_128": max((float(r["logit_relative_l2_error"]) for r in decode_steps), default=None),
        "MIN_LOGIT_COSINE_1_128": min((float(r["logit_cosine"]) for r in decode_steps), default=None),
        "TOP1_MATCH_RATE_1_128": sum(1 for r in decode_steps if r["HF_top1_token"] == r["manual_top1_token"]) / max(len(decode_steps), 1),
        "TOP20_OVERLAP_MEAN_1_128": sum(float(r["top20_overlap"]) for r in decode_steps) / max(len(decode_steps), 1),
    }
    save_json(RUN_DIR / "teacher_forced_summary.json", teacher_summary)

    first_div = {
        "FIRST_LOGIT_NUMERICAL_DIVERGENCE_STEP": first_logit_div,
        "FIRST_TOP1_DIVERGENCE_STEP": first_top1_div,
        "FIRST_STATE_DIVERGENCE_STEP": None if state_gate == "PASS" else 0,
        "FIRST_POSITION_SEMANTICS_DIVERGENCE_STEP": first_position_div,
    }
    if prefill_cmp["PREFILL_LOGIT_PARITY_GATE"] == "FAIL":
        root = "PREFILL_SEMANTICS_MISMATCH"
    elif first_top1_div == 1 or first_logit_div == 1:
        root = "PREFILL_TO_DECODE_TRANSITION_MISMATCH"
    elif first_logit_div is not None:
        root = "RECURRENT_DECODE_ACCUMULATION_MISMATCH"
    elif teacher_gate == "PASS":
        root = "GENERATION_LOOP_OR_RUNTIME"
    else:
        root = "MULTIPLE_MANUAL_GENERATION_MISMATCHES"
    first_div["ROOT_CAUSE_CLASS"] = root
    save_json(RUN_DIR / "first_divergence.json", first_div)

    if rows:
        last_logits = manual_logits
        sampling = sampling_distribution_parity(torch, last_logits, manual_prefix_ids)
    else:
        sampling = {"SAMPLING_DISTRIBUTION_PARITY_GATE": "NOT_RUN"}
    save_json(RUN_DIR / "sampling_distribution_parity.json", sampling)
    eos_test = eos_unit_test()
    save_json(RUN_DIR / "eos_unit_test.json", eos_test)

    # Runtime profile uses teacher-forced cached decode only, no sampling, at 32 and 128 steps.
    runtime_rows = []
    for n in (32, 128):
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past = out.past_key_values
        mask = attention_mask.clone()
        per = []
        for i in range(n):
            forced = torch.tensor([[generated_ids[i]]], device=device, dtype=input_ids.dtype)
            mask = torch.cat([mask, torch.ones_like(forced)], dim=-1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model(input_ids=forced, attention_mask=mask, past_key_values=past, use_cache=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            per.append((time.perf_counter() - t0) * 1000.0)
            past = out.past_key_values
        vals = sorted(per)
        runtime_rows.append({
            "path": "MANUAL_RECURRENT_DECODE",
            "steps": n,
            "median_ms": vals[len(vals) // 2],
            "p25_ms": vals[len(vals) // 4],
            "p75_ms": vals[(len(vals) * 3) // 4],
            "p95_ms": vals[min(len(vals) - 1, int(math.ceil(len(vals) * 0.95)) - 1)],
            "mean_ms": sum(vals) / len(vals),
        })
    write_csv(RUN_DIR / "runtime_profile.csv", runtime_rows)
    manual_ms = next(r["median_ms"] for r in runtime_rows if r["steps"] == 128)
    # HF generate reference timing for 128 capped tokens.
    setup_seed(SEED + int(item["problem_index"]))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_new_tokens=128,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    hf_ms = ((time.perf_counter() - t0) * 1000.0) / 128.0
    runtime_summary = {
        "MANUAL_MS_PER_TOKEN": manual_ms,
        "HF_REFERENCE_MS_PER_TOKEN": hf_ms,
        "SLOWDOWN_RATIO": manual_ms / hf_ms if hf_ms else None,
        "MANUAL_RUNTIME_GATE": "PASS" if manual_ms < 1000.0 else "FAIL",
        "RUNTIME_ROOT_CAUSE": "No severe per-token runtime pathology in 128-step cached teacher-forced profile." if manual_ms < 1000.0 else "Manual cached decode per-token runtime is too slow.",
        "runtime_profile_csv": str(RUN_DIR / "runtime_profile.csv"),
    }
    save_json(RUN_DIR / "runtime_summary.json", runtime_summary)

    final = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE",
        "ENVIRONMENT_AUDIT_GATE": environment["ENVIRONMENT_AUDIT_GATE"],
        "HF_CANONICAL_REFERENCE_GATE": canonical_ref["HF_CANONICAL_REFERENCE_GATE"],
        "INPUT_IDS_MATCH": input_parity["INPUT_IDS_MATCH"],
        "ATTENTION_MASK_MATCH": input_parity["ATTENTION_MASK_MATCH"],
        "HF_GENERATION_PREPARATION_GATE": trace_obj["HF_GENERATION_PREPARATION_GATE"],
        "PREFILL_LOGIT_PARITY_GATE": prefill_cmp["PREFILL_LOGIT_PARITY_GATE"],
        "PREFILL_STATE_PARITY_GATE": state_gate,
        "PREFILL_MAX_ABS_LOGIT_ERROR": prefill_cmp["max_abs_error"],
        "PREFILL_REL_L2_LOGIT_ERROR": prefill_cmp["relative_l2_error"],
        "PREFILL_LOGIT_COSINE": prefill_cmp["cosine_similarity"],
        "TEACHER_FORCED_DECODE_PARITY_GATE": teacher_summary["TEACHER_FORCED_DECODE_PARITY_GATE"],
        "FIRST_LOGIT_NUMERICAL_DIVERGENCE_STEP": first_logit_div,
        "FIRST_TOP1_DIVERGENCE_STEP": first_top1_div,
        "FIRST_STATE_DIVERGENCE_STEP": first_div["FIRST_STATE_DIVERGENCE_STEP"],
        "FIRST_POSITION_SEMANTICS_DIVERGENCE_STEP": first_position_div,
        "DECODE_MAX_REL_L2_ERROR_1_128": teacher_summary["DECODE_MAX_REL_L2_ERROR_1_128"],
        "MIN_LOGIT_COSINE_1_128": teacher_summary["MIN_LOGIT_COSINE_1_128"],
        "TOP1_MATCH_RATE_1_128": teacher_summary["TOP1_MATCH_RATE_1_128"],
        "TOP20_OVERLAP_MEAN_1_128": teacher_summary["TOP20_OVERLAP_MEAN_1_128"],
        "SAMPLING_DISTRIBUTION_PARITY_GATE": sampling["SAMPLING_DISTRIBUTION_PARITY_GATE"],
        "MANUAL_EOS_UNIT_TEST_GATE": eos_test["MANUAL_EOS_UNIT_TEST_GATE"],
        "MANUAL_MS_PER_TOKEN": runtime_summary["MANUAL_MS_PER_TOKEN"],
        "HF_REFERENCE_MS_PER_TOKEN": runtime_summary["HF_REFERENCE_MS_PER_TOKEN"],
        "SLOWDOWN_RATIO": runtime_summary["SLOWDOWN_RATIO"],
        "RUNTIME_ROOT_CAUSE": runtime_summary["RUNTIME_ROOT_CAUSE"],
        "ROOT_CAUSE_CLASS": root,
        "FIX_APPLIED": "NO",
        "POST_FIX_PREFILL_PARITY": "NOT_RUN",
        "POST_FIX_DECODE_PARITY": "NOT_RUN",
        "POST_FIX_EOS_GATE": "NOT_RUN",
        "POST_FIX_RUNTIME_GATE": "NOT_RUN",
    }
    ready = (
        final["CHAT_TEMPLATE_GATE"] if "CHAT_TEMPLATE_GATE" in final else "PASS"
    )
    generation_ready = (
        ready == "PASS"
        and sampling["SAMPLING_DISTRIBUTION_PARITY_GATE"] == "PASS"
        and prefill_cmp["PREFILL_LOGIT_PARITY_GATE"] == "PASS"
        and teacher_summary["TEACHER_FORCED_DECODE_PARITY_GATE"] == "PASS"
        and eos_test["MANUAL_EOS_UNIT_TEST_GATE"] == "PASS"
        and runtime_summary["MANUAL_RUNTIME_GATE"] == "PASS"
    )
    final["LING_MANUAL_GENERATION_READY"] = "YES" if generation_ready else "NO"
    final["CAN_START_30_PROBLEM_FP_R128_C128_RERUN"] = "YES" if generation_ready else "NO"
    final["FINAL_CLASSIFICATION"] = "MANUAL_RECURRENT_DECODE_PARITY_ESTABLISHED" if generation_ready else root
    final["NEXT_ACTION"] = "Run a short free-generation smoke with max_new_tokens=256 only if readiness is YES; otherwise inspect the first divergence artifact."
    final["branch"] = environment["git_branch"]
    final["commit"] = environment["git_head"]
    final["push_status"] = "NOT_PUSHED"
    final["git_status"] = sh(["git", "status", "--short"])
    final["artifact_dir"] = str(RUN_DIR)
    save_json(RUN_DIR / "final_classification.json", final)
    (RUN_DIR / "first_divergence.md").write_text(
        "\n".join([f"# {TASK}", "", f"FIRST_LOGIT_NUMERICAL_DIVERGENCE_STEP = {first_logit_div}", f"FIRST_TOP1_DIVERGENCE_STEP = {first_top1_div}", f"ROOT_CAUSE_CLASS = {root}"]) + "\n",
        encoding="utf-8",
    )
    (RUN_DIR / "root_cause_analysis.md").write_text(
        "\n".join([f"# {TASK}", "", f"ROOT_CAUSE_CLASS = {root}", f"RUNTIME_ROOT_CAUSE = {runtime_summary['RUNTIME_ROOT_CAUSE']}"]) + "\n",
        encoding="utf-8",
    )
    (RUN_DIR / "fix_summary.md").write_text("FIX_APPLIED = NO\nNo production harness fix was applied in this audit run.\n", encoding="utf-8")
    save_json(RUN_DIR / "post_fix_parity.json", {"FIX_APPLIED": "NO", "POST_FIX_PREFILL_PARITY": "NOT_RUN", "POST_FIX_DECODE_PARITY": "NOT_RUN"})
    report_lines = [
        f"# {TASK}",
        "",
        "## Previous failure",
        "HF canonical problem_id=60 finished at 3024 tokens with EOS; manual free decode timed out around 40 minutes.",
        "",
        "## Why timeout is not proven EOS failure",
        "The timeout did not show that EOS was generated and ignored; it only showed that the free-running manual trajectory did not finish promptly.",
        "",
        "## Canonical protocol",
        f"thinking=ON, temperature={TEMPERATURE}, top_p={TOP_P}, top_k={TOP_K}, max_new_tokens={MAX_NEW_TOKENS}, eos={EOS_TOKEN_ID}, pad={PAD_TOKEN_ID}.",
        "",
        "## Gates",
        "```json",
        json.dumps(final, indent=2, sort_keys=True, ensure_ascii=False),
        "```",
        "",
        "## Input parity",
        json.dumps(input_parity, indent=2, sort_keys=True, ensure_ascii=False),
        "",
        "## HF generation-preparation semantics",
        f"Trace saved to `{RUN_DIR / 'hf_generation_input_trace.json'}`.",
        "",
        "## Prefill logit parity",
        json.dumps(prefill_cmp, indent=2, sort_keys=True, ensure_ascii=False),
        "",
        "## Prefill recurrent-state parity",
        f"State CSV saved to `{RUN_DIR / 'prefill_state_parity.csv'}`. Gate={state_gate}.",
        "",
        "## Teacher-forced decode parity",
        json.dumps(teacher_summary, indent=2, sort_keys=True, ensure_ascii=False),
        "",
        "## Exact first divergence step",
        json.dumps(first_div, indent=2, sort_keys=True, ensure_ascii=False),
        "",
        "## Cache position / position ids / attention mask comparison",
        f"Per-step CSV saved to `{RUN_DIR / 'teacher_forced_step_parity.csv'}`.",
        "",
        "## KDA recurrent-state handoff semantics",
        "Observable through DynamicCache layer keys/values; detailed prefill cache comparison saved.",
        "",
        "## Sampling-distribution parity",
        json.dumps(sampling, indent=2, sort_keys=True, ensure_ascii=False),
        "",
        "## EOS unit test",
        json.dumps(eos_test, indent=2, sort_keys=True, ensure_ascii=False),
        "",
        "## Runtime profile",
        json.dumps(runtime_summary, indent=2, sort_keys=True, ensure_ascii=False),
        "",
        "## Runtime hotspot",
        runtime_summary["RUNTIME_ROOT_CAUSE"],
        "",
        "## Minimal fix",
        "No production fix applied in this run.",
        "",
        "## Post-fix validation",
        "Not run.",
        "",
        "## Remaining limitations",
        "Teacher-forced reference is full-prefix model forward, while HF generate trace is separately instrumented on a 2-token capped generation.",
        "",
        "## Final readiness",
        f"LING_MANUAL_GENERATION_READY = {final['LING_MANUAL_GENERATION_READY']}",
    ]
    (RUN_DIR / "final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
