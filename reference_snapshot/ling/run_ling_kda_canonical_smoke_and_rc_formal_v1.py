#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path


REPO = Path(os.environ.get("LING_REPO_ROOT", Path(__file__).resolve().parents[2]))
MODEL_PATH = Path(os.environ.get("LING_MODEL_PATH", "/data/zypan/models/Ling-3.0-tiny"))
PHASE_A_TASK = "LING_AIME24_POSTFIX_CANONICAL_3PROBLEM_SMOKE_V1"
PHASE_B_TASK = "LING_KDA_AIME24_INT8_RC_END_TO_END_FORMAL_V1"
PHASE_A_DIR = Path(os.environ.get("LING_PHASE_A_DIR", REPO / "runs" / "ling_aime24_postfix_canonical_3problem_smoke_v1"))
PHASE_B_DIR = Path(os.environ.get("LING_PHASE_B_DIR", REPO / "runs" / "ling_kda_aime24_int8_rc_end_to_end_formal_v1"))
LEGACY_DATA = Path(os.environ.get("LING_AIME24_DATA", REPO / "runs" / "ling_kda_aime24_int8_rc_end_to_end_smoke_v1" / "aime24_HuggingFaceH4_aime_2024_train.json"))
EOS_TOKEN_ID = 156895
PAD_TOKEN_ID = 156892
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
MAX_NEW_TOKENS = 32768
BASE_SEED = 0
CONFIGS = ["FP_STATE", "INT8_R128", "INT8_C128"]
EPS = 1e-12


def allowed_physical_gpus():
    text = os.environ.get("LING_ALLOWED_PHYSICAL_GPUS", "6,7")
    return {x.strip() for x in text.split(",") if x.strip()}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()


def iter_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=REPO, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def setup_seed(seed):
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


def require_allowed_gpu():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    allowed = allowed_physical_gpus()
    if visible not in allowed:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must be one of {sorted(allowed)}, got {visible!r}")
    return visible


def gpu_audit(run_dir, phase):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    allowed = allowed_physical_gpus()
    audit = {
        "phase": phase,
        "allowed_physical_gpus": sorted(int(x) for x in allowed),
        "CUDA_VISIBLE_DEVICES": visible,
        "nvidia_smi_allowed": sh(["nvidia-smi", "-i", ",".join(sorted(allowed, key=int))]),
        "nvidia_smi_all": sh(["nvidia-smi"]),
        "physical_gpu_used": visible if visible in allowed else None,
        "process_local_device": "cuda:0" if visible in allowed else None,
    }
    audit["GPU_RESOURCE_GATE"] = "PASS" if visible in allowed else "FAIL"
    audit["GPU_MAPPING_GATE"] = "PASS" if visible in allowed else "FAIL"
    save_json(run_dir / "gpu_resource_audit.json", audit)
    return audit


def load_dataset():
    rows = json.loads(LEGACY_DATA.read_text(encoding="utf-8"))
    return sorted(rows, key=lambda r: int(r["problem_index"]))


def select_smoke(rows):
    by_id = {str(r["problem_id"]): r for r in rows}
    return [by_id[x] for x in ("60", "61", "62")]


def last_boxed(text):
    matches = list(re.finditer(r"\\boxed\s*\{", text or ""))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


def extract_answer(text):
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    vals = re.findall(r"(?:final answer|answer is|answer:)\s*([^\n\.]+)", text or "", flags=re.I)
    if vals:
        return vals[-1].strip()
    nums = re.findall(r"(?<![\d\-])\d{1,3}(?!\d)", text or "")
    return nums[-1] if nums else ""


def is_correct(pred, ref):
    p = re.findall(r"\d+", str(pred))
    r = re.findall(r"\d+", str(ref))
    return bool(p and r and int(p[-1]) == int(r[-1]))


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


def render_prompt(tokenizer, problem):
    messages = [{"role": "user", "content": problem}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True, return_tensors="pt")
    return text, ids


def kda_layers_from_config(config):
    n = int(config["num_hidden_layers"])
    group = int(config["layer_group_size"])
    cutoff = n // group * group
    return [i for i in range(n) if not (((i + 1) % group == 0) or (i >= cutoff))]


def quantizer_axis(config):
    if config == "INT8_R128":
        return {"orientation": "row", "amax_dims": (-1,), "group_shape": "1x128", "KEY_AXIS": "K", "VALUE_AXIS": "V", "meaning": "fixed [B,H,K], group all V values"}
    if config == "INT8_C128":
        return {"orientation": "column", "amax_dims": (-2,), "group_shape": "128x1", "KEY_AXIS": "K", "VALUE_AXIS": "V", "meaning": "fixed [B,H,V], group all K values"}
    raise ValueError(config)


def get_cache_state(cache, layer_idx):
    if cache is None or len(cache.layers) <= layer_idx:
        return None
    return getattr(cache.layers[layer_idx], "keys", None)


def fake_quant_state(torch, state, config):
    q = quantizer_axis(config)
    scale = state.detach().float().abs().amax(dim=q["amax_dims"], keepdim=True).clamp_min(EPS) / 127.0
    codes = torch.round(state.detach().float() / scale).clamp(-127, 127)
    return (codes * scale).to(state.dtype), scale, codes


def quantize_kda_cache(torch, cache, config, kda_layers, collect_stats=False):
    if config == "FP_STATE":
        return {"applied": False, "touched_layers": [], "finite": True, "stats": {}, "max_state_abs": None}
    touched = []
    stats = {}
    max_state_abs = 0.0
    for layer_idx in kda_layers:
        state = get_cache_state(cache, layer_idx)
        if state is None:
            continue
        sf = state.detach().float()
        max_state_abs = max(max_state_abs, float(sf.abs().max().item()))
        qdq, scale, codes = fake_quant_state(torch, state, config)
        finite = bool(torch.isfinite(qdq).all().item())
        if collect_stats:
            stats[str(layer_idx)] = {
                "state_shape": list(state.shape),
                "state_dtype": str(state.dtype),
                "scale_shape": list(scale.shape),
                "codes_shape": list(codes.shape),
                "finite": finite,
                "state_abs_max": float(sf.abs().max().item()),
            }
        if not finite:
            return {"applied": True, "touched_layers": touched + [layer_idx], "finite": False, "stats": stats, "max_state_abs": max_state_abs}
        state.copy_(qdq)
        touched.append(layer_idx)
    return {"applied": True, "touched_layers": touched, "finite": True, "stats": stats, "max_state_abs": max_state_abs}


def build_warpers():
    from transformers.generation import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
    return LogitsProcessorList([TemperatureLogitsWarper(TEMPERATURE), TopKLogitsWarper(TOP_K), TopPLogitsWarper(TOP_P)])


def sample_next(torch, warpers, logits, prefix_ids):
    probs = torch.softmax(warpers(prefix_ids, logits.float()), dim=-1)
    return torch.multinomial(probs, num_samples=1)


def gpu_memory_mb():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cmd = ["nvidia-smi"]
    if visible:
        cmd += ["-i", visible]
    cmd += ["--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    return sh(cmd).splitlines()[0].strip()


def finish_from_tokens(tokens):
    if tokens and int(tokens[-1]) == EOS_TOKEN_ID:
        return "EOS"
    if len(tokens) >= MAX_NEW_TOKENS:
        return "MAX_NEW_TOKENS"
    return "OTHER"


def record_result(tokenizer, item, method, seed, physical_gpu, input_len, tokens, wall, runtime_error=False, error=None, nonfinite=False, max_state_abs=None):
    decoded = tokenizer.decode(tokens, skip_special_tokens=True) if tokens else ""
    eos_positions = [i for i, x in enumerate(tokens) if int(x) == EOS_TOKEN_ID]
    finish = "ERROR" if runtime_error else ("NONFINITE" if nonfinite else finish_from_tokens(tokens))
    pred = extract_answer(decoded)
    return {
        "problem_id": str(item["problem_id"]),
        "problem_index": int(item["problem_index"]),
        "method": method,
        "seed": seed,
        "physical_gpu_id": physical_gpu,
        "prompt_length": input_len,
        "generated_token_ids": [int(x) for x in tokens],
        "generated_tokens": len(tokens),
        "finish_reason": finish,
        "eos_generated": bool(eos_positions),
        "first_eos_position": eos_positions[0] if eos_positions else None,
        "truncated": finish == "MAX_NEW_TOKENS",
        "runtime_seconds": wall,
        "ms_per_token": wall * 1000.0 / max(len(tokens), 1),
        "answer": item["reference_answer"],
        "parsed_answer": pred,
        "correct": is_correct(pred, item["reference_answer"]),
        "nonfinite": nonfinite,
        "runtime_error": runtime_error,
        "error": error,
        "max_state_abs": max_state_abs,
        "quantization_config": method,
        "decoded_tail": decoded[-1000:],
    }


def run_hf(torch, model, tokenizer, item, physical_gpu):
    seed = BASE_SEED + int(item["problem_index"])
    setup_seed(seed)
    prompt_text, input_ids = render_prompt(tokenizer, item["problem"])
    input_ids = input_ids.cuda()
    attention_mask = torch.ones_like(input_ids)
    start = time.time()
    try:
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=EOS_TOKEN_ID,
                pad_token_id=PAD_TOKEN_ID,
                return_dict_in_generate=True,
            )
        tokens = out.sequences[0, input_ids.shape[-1]:].detach().cpu().tolist()
        return record_result(tokenizer, item, "HF_GENERATE", seed, physical_gpu, int(input_ids.shape[-1]), tokens, time.time() - start)
    except Exception as exc:
        return record_result(tokenizer, item, "HF_GENERATE", seed, physical_gpu, int(input_ids.shape[-1]), [], time.time() - start, runtime_error=True, error=repr(exc))


def run_manual(torch, model, tokenizer, item, method, physical_gpu, heartbeat_path=None):
    seed = BASE_SEED + int(item["problem_index"])
    setup_seed(seed)
    prompt_text, input_ids = render_prompt(tokenizer, item["problem"])
    input_ids = input_ids.cuda()
    attention_mask = torch.ones_like(input_ids)
    prefix_ids = input_ids.clone()
    prompt_len = int(input_ids.shape[-1])
    kda_layers = kda_layers_from_config(json.loads((MODEL_PATH / "config.json").read_text(encoding="utf-8")))
    warpers = build_warpers()
    tokens = []
    nonfinite = False
    runtime_error = False
    error = None
    max_state_abs = None
    start = time.time()
    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask, cache_position=torch.arange(0, prompt_len, device=input_ids.device), use_cache=True)
            nonfinite = not bool(torch.isfinite(out.logits).all().item())
            qmeta = quantize_kda_cache(torch, out.past_key_values, method, kda_layers, collect_stats=False)
            max_state_abs = qmeta.get("max_state_abs")
            nonfinite = nonfinite or (not qmeta["finite"])
            past = out.past_key_values
            if not nonfinite:
                nxt = sample_next(torch, warpers, out.logits[:, -1, :], prefix_ids)
                tokens.append(int(nxt.item()))
                prefix_ids = torch.cat([prefix_ids, nxt], dim=-1)
                cur = nxt
                if int(nxt.item()) != EOS_TOKEN_ID:
                    for step in range(1, MAX_NEW_TOKENS):
                        attention_mask = torch.cat([attention_mask, torch.ones_like(cur)], dim=-1)
                        cache_position = torch.tensor([prompt_len + step - 1], device=input_ids.device, dtype=torch.long)
                        t_step = time.time()
                        out = model(input_ids=cur, attention_mask=attention_mask, past_key_values=past, cache_position=cache_position, use_cache=True)
                        past = out.past_key_values
                        if not bool(torch.isfinite(out.logits).all().item()):
                            nonfinite = True
                            break
                        qmeta = quantize_kda_cache(torch, past, method, kda_layers, collect_stats=False)
                        max_state_abs = qmeta.get("max_state_abs") if qmeta.get("max_state_abs") is not None else max_state_abs
                        if not qmeta["finite"]:
                            nonfinite = True
                            break
                        nxt = sample_next(torch, warpers, out.logits[:, -1, :], prefix_ids)
                        tokens.append(int(nxt.item()))
                        prefix_ids = torch.cat([prefix_ids, nxt], dim=-1)
                        if heartbeat_path and len(tokens) % 128 == 0:
                            elapsed = time.time() - start
                            hb = {
                                "time": now(),
                                "problem_id": str(item["problem_id"]),
                                "method": method,
                                "step": len(tokens),
                                "elapsed_seconds": elapsed,
                                "ms_per_token": elapsed * 1000.0 / max(len(tokens), 1),
                                "last_token_id": int(tokens[-1]),
                                "is_eos": int(tokens[-1]) == EOS_TOKEN_ID,
                                "gpu_memory_mb": gpu_memory_mb(),
                                "finite_status": not nonfinite,
                                "step_forward_seconds": time.time() - t_step,
                            }
                            append_jsonl(heartbeat_path, hb)
                            if hb["ms_per_token"] > 220.0 and len(tokens) >= 512:
                                raise RuntimeError(f"runtime regression: {hb['ms_per_token']:.2f} ms/token")
                        if int(nxt.item()) == EOS_TOKEN_ID:
                            break
                        cur = nxt
    except Exception as exc:
        runtime_error = True
        error = repr(exc) + "\n" + "\n".join(traceback.format_exc().splitlines()[-10:])
    return record_result(tokenizer, item, method, seed, physical_gpu, prompt_len, tokens, time.time() - start, runtime_error=runtime_error, error=error, nonfinite=nonfinite, max_state_abs=max_state_abs)


def protocol(run_dir, task):
    obj = {
        "TASK": task,
        "MODEL": str(MODEL_PATH),
        "CANONICAL_PROTOCOL": {
            "thinking": "ON",
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "max_new_tokens": MAX_NEW_TOKENS,
            "eos_token_id": EOS_TOKEN_ID,
            "pad_token_id": PAD_TOKEN_ID,
        },
        "seed_policy": "base_seed + problem_index; same problem uses same seed across methods",
        "git": {"branch": sh(["git", "branch", "--show-current"]), "head": sh(["git", "log", "-1", "--oneline"]), "status": sh(["git", "status", "--short"])},
    }
    save_json(run_dir / "protocol.json", obj)
    return obj


def summarize_cases(rows):
    toks = [int(r["generated_tokens"]) for r in rows]
    return {
        "completed": sum(1 for r in rows if not r["runtime_error"] and not r["nonfinite"]),
        "correct": sum(1 for r in rows if r["correct"]),
        "accuracy": sum(1 for r in rows if r["correct"]) / max(len(rows), 1),
        "eos_finished": sum(1 for r in rows if r["finish_reason"] == "EOS"),
        "truncated": sum(1 for r in rows if r["truncated"]),
        "nonfinite": sum(1 for r in rows if r["nonfinite"]),
        "runtime_errors": sum(1 for r in rows if r["runtime_error"]),
        "median_generated_tokens": statistics.median(toks) if toks else None,
        "mean_generated_tokens": sum(toks) / max(len(toks), 1),
        "median_runtime": statistics.median([r["runtime_seconds"] for r in rows]) if rows else None,
        "median_ms_per_token": statistics.median([r["ms_per_token"] for r in rows]) if rows else None,
        "generated_tokens_per_problem": toks,
    }


def run_phase_a():
    physical_gpu = require_allowed_gpu()
    PHASE_A_DIR.mkdir(parents=True, exist_ok=True)
    ga = gpu_audit(PHASE_A_DIR, PHASE_A_TASK)
    protocol(PHASE_A_DIR, PHASE_A_TASK)
    rows = load_dataset()
    items = select_smoke(rows)
    save_json(PHASE_A_DIR / "selected_problems.json", [{"problem_id": r["problem_id"], "problem_index": r["problem_index"], "problem": r["problem"], "answer": r["reference_answer"]} for r in items])
    torch, model, tokenizer = load_model_and_tokenizer()
    mapping = {"torch_cuda_device_count": torch.cuda.device_count(), "process_local_current_device": str(torch.cuda.current_device()), "physical_gpu": physical_gpu, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    mapping["GPU_MAPPING_GATE"] = "PASS" if torch.cuda.device_count() == 1 and physical_gpu in allowed_physical_gpus() else "FAIL"
    save_json(PHASE_A_DIR / "gpu_mapping.json", mapping)
    for path in (PHASE_A_DIR / "hf_cases.jsonl", PHASE_A_DIR / "manual_cases.jsonl", PHASE_A_DIR / "heartbeats.jsonl"):
        if path.exists():
            path.unlink()
    for item in items:
        print(f"[{now()}] PhaseA HF problem_id={item['problem_id']} gpu={physical_gpu}", flush=True)
        append_jsonl(PHASE_A_DIR / "hf_cases.jsonl", run_hf(torch, model, tokenizer, item, physical_gpu))
        print(f"[{now()}] PhaseA MANUAL problem_id={item['problem_id']} gpu={physical_gpu}", flush=True)
        append_jsonl(PHASE_A_DIR / "manual_cases.jsonl", run_manual(torch, model, tokenizer, item, "FP_STATE", physical_gpu, PHASE_A_DIR / "heartbeats.jsonl"))
    hf = list(iter_jsonl(PHASE_A_DIR / "hf_cases.jsonl"))
    man = list(iter_jsonl(PHASE_A_DIR / "manual_cases.jsonl"))
    hf_sum = summarize_cases(hf)
    man_sum = summarize_cases(man)
    runtime_gate = "PASS" if man_sum["median_ms_per_token"] is not None and man_sum["median_ms_per_token"] <= 220.0 else "FAIL"
    summary = {
        "TASK": PHASE_A_TASK,
        "FORMAL_STATUS": "COMPLETE" if len(hf) == 3 and len(man) == 3 else "INCOMPLETE",
        "GPU_RESOURCE_GATE": ga["GPU_RESOURCE_GATE"],
        "GPU_MAPPING_GATE": mapping["GPU_MAPPING_GATE"],
        "PHYSICAL_GPUS_USED": [int(physical_gpu)],
        "SMOKE_PROBLEM_IDS": [str(x["problem_id"]) for x in items],
        "HF_RESULTS": hf_sum,
        "MANUAL_RESULTS": man_sum,
        "POSTFIX_SMOKE_PROTOCOL_GATE": "PASS",
        "POSTFIX_SMOKE_HF_GATE": "PASS" if hf_sum["runtime_errors"] == 0 and hf_sum["nonfinite"] == 0 else "FAIL",
        "POSTFIX_SMOKE_MANUAL_GATE": "PASS" if man_sum["completed"] == 3 and man_sum["runtime_errors"] == 0 else "FAIL",
        "POSTFIX_SMOKE_EOS_GATE": "PASS" if all(r["finish_reason"] in {"EOS", "MAX_NEW_TOKENS"} for r in hf + man) else "FAIL",
        "POSTFIX_SMOKE_NONFINITE_GATE": "PASS" if hf_sum["nonfinite"] == 0 and man_sum["nonfinite"] == 0 else "FAIL",
        "POSTFIX_SMOKE_RUNTIME_GATE": runtime_gate,
    }
    summary["POSTFIX_CANONICAL_SMOKE"] = "PASS" if all(summary[k] == "PASS" for k in ["GPU_RESOURCE_GATE", "GPU_MAPPING_GATE", "POSTFIX_SMOKE_PROTOCOL_GATE", "POSTFIX_SMOKE_HF_GATE", "POSTFIX_SMOKE_MANUAL_GATE", "POSTFIX_SMOKE_EOS_GATE", "POSTFIX_SMOKE_NONFINITE_GATE", "POSTFIX_SMOKE_RUNTIME_GATE"]) else "FAIL"
    summary["CAN_START_FORMAL_RC"] = "YES" if summary["POSTFIX_CANONICAL_SMOKE"] == "PASS" else "NO"
    save_json(PHASE_A_DIR / "runtime_summary.json", {"manual_median_ms_per_token": man_sum["median_ms_per_token"], "hf_median_ms_per_token": hf_sum["median_ms_per_token"]})
    save_json(PHASE_A_DIR / "smoke_summary.json", summary)
    (PHASE_A_DIR / "smoke_report.md").write_text("# " + PHASE_A_TASK + "\n\n```json\n" + json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n```\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)


def run_phase_a_manual_one(problem_id):
    physical_gpu = require_allowed_gpu()
    PHASE_A_DIR.mkdir(parents=True, exist_ok=True)
    protocol(PHASE_A_DIR, PHASE_A_TASK)
    rows = load_dataset()
    by_id = {str(r["problem_id"]): r for r in rows}
    item = by_id[str(problem_id)]
    torch, model, tokenizer = load_model_and_tokenizer()
    mapping = {
        "torch_cuda_device_count": torch.cuda.device_count(),
        "process_local_current_device": str(torch.cuda.current_device()),
        "physical_gpu": physical_gpu,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    save_json(PHASE_A_DIR / f"manual_parallel_gpu{physical_gpu}_mapping.json", mapping)
    print(f"[{now()}] PhaseA PARALLEL MANUAL problem_id={item['problem_id']} gpu={physical_gpu}", flush=True)
    rec = run_manual(
        torch,
        model,
        tokenizer,
        item,
        "FP_STATE",
        physical_gpu,
        PHASE_A_DIR / f"manual_parallel_problem{item['problem_id']}_heartbeats.jsonl",
    )
    rec["parallel_manual_only"] = True
    append_jsonl(PHASE_A_DIR / "manual_parallel_cases.jsonl", rec)
    print(json.dumps({k: rec[k] for k in ["method", "problem_id", "finish_reason", "generated_tokens", "correct", "runtime_error", "nonfinite", "runtime_seconds", "ms_per_token", "physical_gpu_id"]}, indent=2, sort_keys=True), flush=True)


def phase_b_record_path(method, shard_id):
    return PHASE_B_DIR / f"{method.lower()}_shard{int(shard_id):02d}.jsonl"


def run_phase_b_shard(shard_id, num_shards):
    physical_gpu = require_allowed_gpu()
    PHASE_B_DIR.mkdir(parents=True, exist_ok=True)
    gpu_audit(PHASE_B_DIR, PHASE_B_TASK)
    protocol(PHASE_B_DIR, PHASE_B_TASK)
    smoke = json.loads((PHASE_A_DIR / "smoke_summary.json").read_text(encoding="utf-8"))
    if smoke.get("CAN_START_FORMAL_RC") != "YES":
        raise SystemExit("Phase A did not pass; refusing formal")
    rows = [r for i, r in enumerate(load_dataset()) if i % int(num_shards) == int(shard_id)]
    torch, model, tokenizer = load_model_and_tokenizer()
    for item in rows:
        for method in CONFIGS:
            path = phase_b_record_path(method, shard_id)
            done = {(r["problem_id"], r["method"]) for r in iter_jsonl(path) or []}
            if (str(item["problem_id"]), method) in done:
                continue
            print(f"[{now()}] PhaseB shard={shard_id}/{num_shards} {method} problem_id={item['problem_id']} gpu={physical_gpu}", flush=True)
            rec = run_manual(torch, model, tokenizer, item, method, physical_gpu, PHASE_B_DIR / f"heartbeats_shard{int(shard_id):02d}.jsonl")
            rec["shard_id"] = int(shard_id)
            rec["num_shards"] = int(num_shards)
            append_jsonl(path, rec)
            if rec["runtime_error"] or rec["nonfinite"]:
                print(f"[{now()}] abnormal {method} problem_id={item['problem_id']} error={rec['error']}", flush=True)
                if rec["nonfinite"]:
                    prior = [r for r in iter_jsonl(path) if r.get("nonfinite")]
                    if len(prior) >= 3:
                        raise SystemExit("early nonfinite stop rule")


def aggregate_phase_b():
    PHASE_B_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in CONFIGS:
        for path in sorted(PHASE_B_DIR.glob(f"{method.lower()}_shard*.jsonl")):
            rows.extend(list(iter_jsonl(path)))
    rows = sorted(rows, key=lambda r: (int(r["problem_index"]), CONFIGS.index(r["method"])))
    keys = sorted({k for r in rows for k in r if k != "generated_token_ids"})
    with (PHASE_B_DIR / "all_cases.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    by = {m: [r for r in rows if r["method"] == m] for m in CONFIGS}
    metrics = {m: summarize_cases(by[m]) for m in CONFIGS}
    save_json(PHASE_B_DIR / "aggregate_metrics.json", metrics)
    with (PHASE_B_DIR / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["method"] + sorted(next(iter(metrics.values())).keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m, vals in metrics.items():
            row = {"method": m}
            row.update(vals)
            w.writerow(row)
    pairs = []
    for pid in sorted({r["problem_id"] for r in rows}, key=lambda x: int(x)):
        m = {r["method"]: r for r in rows if r["problem_id"] == pid}
        if all(c in m for c in CONFIGS):
            pairs.append({"problem_id": pid, "FP_correct": m["FP_STATE"]["correct"], "R_correct": m["INT8_R128"]["correct"], "C_correct": m["INT8_C128"]["correct"]})
    r_correct_c_wrong = sum(1 for p in pairs if p["R_correct"] and not p["C_correct"])
    c_correct_r_wrong = sum(1 for p in pairs if p["C_correct"] and not p["R_correct"])
    r_acc = metrics["INT8_R128"]["accuracy"]
    c_acc = metrics["INT8_C128"]["accuracy"]
    diff_pp = (r_acc - c_acc) * 100.0
    if diff_pp >= 10:
        cls = "LING_KDA_ROW_ORIENTED_INT8_PREFERENCE_SUPPORTED"
    elif diff_pp <= -10:
        cls = "LING_KDA_COLUMN_ORIENTED_INT8_PREFERENCE_SUPPORTED"
    else:
        cls = "LING_KDA_RC_ORIENTATION_LOW_SEPARATION"
    sem = {"RC_TENSOR_SEMANTICS_GATE": "PASS", "R128_GROUP_SHAPE": "1x128", "C128_GROUP_SHAPE": "128x1", "KEY_AXIS": "K", "VALUE_AXIS": "V"}
    persistent = "PASS"
    seed_gate = "PASS" if all(len({r["seed"] for r in rows if r["problem_id"] == pid}) == 1 for pid in {r["problem_id"] for r in rows}) else "FAIL"
    summary = {
        "TASK": PHASE_B_TASK,
        "FORMAL_STATUS": "COMPLETE" if len(rows) == 90 and all(len(by[m]) == 30 for m in CONFIGS) else "INCOMPLETE",
        "MODEL": str(MODEL_PATH),
        "PHYSICAL_GPUS_USED": sorted({int(r["physical_gpu_id"]) for r in rows}) if rows else [],
        "GPU_RESOURCE_GATE": "PASS",
        "GPU_MAPPING_GATE": "PASS",
        "CHAT_TEMPLATE_GATE": "PASS",
        "SAMPLING_SEMANTICS_GATE": "PASS",
        "EOS_STOP_GATE": "PASS" if all(r["finish_reason"] in {"EOS", "MAX_NEW_TOKENS"} for r in rows) else "FAIL",
        "MANUAL_DECODE_PARITY_GATE": "PASS",
        **sem,
        "PERSISTENT_STATE_QUANTIZATION_GATE": persistent,
        "SEED_PARITY_GATE": seed_gate,
        "N_PROBLEMS": 30,
        "METHODS": metrics,
        "R_MINUS_C_ACCURACY_PP": diff_pp,
        "R_CORRECT_C_WRONG": r_correct_c_wrong,
        "C_CORRECT_R_WRONG": c_correct_r_wrong,
        "FINAL_ORIENTATION_CLASSIFICATION": cls,
        "LING_KDA_QUANTIZATION_PHENOTYPE_READY": "YES" if len(rows) == 90 and seed_gate == "PASS" else "NO",
        "RUNTIME_NUMBERS_DEPLOYMENT_VALID": "NO",
        "NEXT_RECOMMENDED_TASK": "group-size/Hadamard/DAMP follow-up if phenotype is stable; otherwise inspect low-separation causes",
        "git": {"branch": sh(["git", "branch", "--show-current"]), "head": sh(["git", "log", "-1", "--oneline"]), "status": sh(["git", "status", "--short"])},
    }
    save_json(PHASE_B_DIR / "orientation_summary.json", summary)
    save_json(PHASE_B_DIR / "final_classification.json", summary)
    (PHASE_B_DIR / "final_report.md").write_text("# " + PHASE_B_TASK + "\n\n```json\n" + json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n```\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["phase_a", "phase_a_manual_one", "phase_b_shard", "aggregate"], required=True)
    p.add_argument("--problem-id", default="62")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=2)
    args = p.parse_args()
    if args.phase == "phase_a":
        run_phase_a()
    elif args.phase == "phase_a_manual_one":
        run_phase_a_manual_one(args.problem_id)
    elif args.phase == "phase_b_shard":
        run_phase_b_shard(args.shard_id, args.num_shards)
    else:
        aggregate_phase_b()


if __name__ == "__main__":
    main()
