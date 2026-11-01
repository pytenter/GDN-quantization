#!/usr/bin/env python3
import csv
import json
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
SEED = 0
EPS = 1e-12


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


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


def cmp_logits(torch, a, b):
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    d = af - bf
    top20_a = set(torch.topk(af, 20).indices.detach().cpu().tolist())
    top20_b = set(torch.topk(bf, 20).indices.detach().cpu().tolist())
    return {
        "max_abs": float(d.abs().max().item()),
        "rel_l2": float((torch.linalg.vector_norm(d) / torch.linalg.vector_norm(af).clamp_min(EPS)).item()),
        "cosine": float(torch.nn.functional.cosine_similarity(af, bf, dim=0).item()),
        "ref_top1": int(af.argmax().item()),
        "manual_top1": int(bf.argmax().item()),
        "top1_match": int(af.argmax().item()) == int(bf.argmax().item()),
        "top20_overlap": len(top20_a & top20_b) / 20.0,
    }


def write_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    setup_seed(SEED)
    import torch

    torch, model, tokenizer = load_model_and_tokenizer()
    selected = json.loads((PREV_RUN / "selected_problems.json").read_text(encoding="utf-8"))
    item = next(x for x in selected if str(x["problem_id"]) == "60")
    hf_ref = next(x for x in load_jsonl(PREV_RUN / "hf_generate_results.jsonl") if str(x["problem_id"]) == "60")
    generated_ids = [int(x) for x in hf_ref["generated_token_ids"]]
    messages = [{"role": "user", "content": item["problem"]}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_tensors="pt",
    ).cuda()
    prompt_len = input_ids.shape[-1]
    attention_mask = torch.ones_like(input_ids)
    prefill_cache_position = torch.arange(0, prompt_len, device=input_ids.device, dtype=torch.long)

    with torch.inference_mode():
        ref = model(input_ids=input_ids, attention_mask=attention_mask, cache_position=prefill_cache_position, use_cache=True)
        manual_old = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        manual_fixed = model(input_ids=input_ids, attention_mask=attention_mask, cache_position=prefill_cache_position, use_cache=True)
    old_past = manual_old.past_key_values
    fixed_past = manual_fixed.past_key_values
    ref_past = ref.past_key_values
    old_mask = attention_mask.clone()
    fixed_mask = attention_mask.clone()
    ref_mask = attention_mask.clone()
    rows = []
    first_old_rel = None
    first_old_top1 = None
    first_fixed_rel = None
    first_fixed_top1 = None
    old_times = []
    fixed_times = []
    for step in range(0, 129):
        if step == 0:
            old_cmp = cmp_logits(torch, ref.logits[:, -1, :], manual_old.logits[:, -1, :])
            fixed_cmp = cmp_logits(torch, ref.logits[:, -1, :], manual_fixed.logits[:, -1, :])
            forced_id = None
            old_ms = fixed_ms = 0.0
            actual_ref_cache_position = list(range(prompt_len))[-8:]
            actual_old_cache_position = list(range(prompt_len))[-8:]
            actual_fixed_cache_position = list(range(prompt_len))[-8:]
        else:
            forced_id = generated_ids[step - 1]
            forced = torch.tensor([[forced_id]], device=input_ids.device, dtype=input_ids.dtype)
            ref_mask = torch.cat([ref_mask, torch.ones_like(forced)], dim=-1)
            old_mask = torch.cat([old_mask, torch.ones_like(forced)], dim=-1)
            fixed_mask = torch.cat([fixed_mask, torch.ones_like(forced)], dim=-1)
            cache_position = torch.tensor([prompt_len + step - 1], device=input_ids.device, dtype=torch.long)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                ref = model(input_ids=forced, attention_mask=ref_mask, past_key_values=ref_past, cache_position=cache_position, use_cache=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ref_ms = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            with torch.inference_mode():
                old = model(input_ids=forced, past_key_values=old_past, use_cache=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            old_ms = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            with torch.inference_mode():
                fixed = model(input_ids=forced, attention_mask=fixed_mask, past_key_values=fixed_past, cache_position=cache_position, use_cache=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            fixed_ms = (time.perf_counter() - t0) * 1000.0
            ref_past = ref.past_key_values
            old_past = old.past_key_values
            fixed_past = fixed.past_key_values
            old_cmp = cmp_logits(torch, ref.logits[:, -1, :], old.logits[:, -1, :])
            fixed_cmp = cmp_logits(torch, ref.logits[:, -1, :], fixed.logits[:, -1, :])
            old_times.append(old_ms)
            fixed_times.append(fixed_ms)
            actual_ref_cache_position = [int(cache_position.item())]
            actual_old_cache_position = ["omitted; model infers from DynamicCache.get_seq_length()"]
            actual_fixed_cache_position = [int(cache_position.item())]
        row = {
            "step": step,
            "forced_token_id": forced_id,
            "forced_token_text": tokenizer.decode([forced_id], skip_special_tokens=False) if forced_id is not None else None,
            "ref_cache_position": actual_ref_cache_position,
            "manual_old_cache_position": actual_old_cache_position,
            "manual_fixed_cache_position": actual_fixed_cache_position,
            "old_rel_l2": old_cmp["rel_l2"],
            "old_cosine": old_cmp["cosine"],
            "old_top1_match": old_cmp["top1_match"],
            "old_top20_overlap": old_cmp["top20_overlap"],
            "fixed_rel_l2": fixed_cmp["rel_l2"],
            "fixed_cosine": fixed_cmp["cosine"],
            "fixed_top1_match": fixed_cmp["top1_match"],
            "fixed_top20_overlap": fixed_cmp["top20_overlap"],
            "old_ms": old_ms,
            "fixed_ms": fixed_ms,
        }
        if first_old_rel is None and old_cmp["rel_l2"] > 1e-3:
            first_old_rel = step
        if first_old_top1 is None and not old_cmp["top1_match"]:
            first_old_top1 = step
        if first_fixed_rel is None and fixed_cmp["rel_l2"] > 1e-6:
            first_fixed_rel = step
        if first_fixed_top1 is None and not fixed_cmp["top1_match"]:
            first_fixed_top1 = step
        rows.append(row)
    write_csv(RUN_DIR / "hf_cached_vs_manual_decode_probe.csv", rows)
    summary = {
        "reference": "HF generate-style cached teacher-forced decode: explicit growing attention_mask and cache_position.",
        "manual_old": "legacy/current manual decode: no attention_mask/cache_position on decode calls.",
        "manual_fixed": "minimal aligned manual decode: explicit growing attention_mask and cache_position.",
        "prompt_len": int(prompt_len),
        "OLD_FIRST_LOGIT_NUMERICAL_DIVERGENCE_STEP": first_old_rel,
        "OLD_FIRST_TOP1_DIVERGENCE_STEP": first_old_top1,
        "FIXED_FIRST_LOGIT_NUMERICAL_DIVERGENCE_STEP": first_fixed_rel,
        "FIXED_FIRST_TOP1_DIVERGENCE_STEP": first_fixed_top1,
        "OLD_MAX_REL_L2_1_128": max(r["old_rel_l2"] for r in rows[1:]),
        "FIXED_MAX_REL_L2_1_128": max(r["fixed_rel_l2"] for r in rows[1:]),
        "OLD_MIN_COSINE_1_128": min(r["old_cosine"] for r in rows[1:]),
        "FIXED_MIN_COSINE_1_128": min(r["fixed_cosine"] for r in rows[1:]),
        "OLD_TOP1_MATCH_RATE_1_128": sum(1 for r in rows[1:] if r["old_top1_match"]) / 128.0,
        "FIXED_TOP1_MATCH_RATE_1_128": sum(1 for r in rows[1:] if r["fixed_top1_match"]) / 128.0,
        "OLD_TOP20_OVERLAP_MEAN_1_128": sum(r["old_top20_overlap"] for r in rows[1:]) / 128.0,
        "FIXED_TOP20_OVERLAP_MEAN_1_128": sum(r["fixed_top20_overlap"] for r in rows[1:]) / 128.0,
        "OLD_MEDIAN_MS": sorted(old_times)[len(old_times) // 2],
        "FIXED_MEDIAN_MS": sorted(fixed_times)[len(fixed_times) // 2],
        "ROOT_CAUSE_CLASS": "POSITION_OR_CACHE_POSITION_MISMATCH" if first_old_rel == 1 and first_fixed_rel is None else "MULTIPLE_MANUAL_GENERATION_MISMATCHES",
        "FIX_VALIDATION_GATE": "PASS" if first_fixed_rel is None and first_fixed_top1 is None else "FAIL",
    }
    save_json(RUN_DIR / "hf_cached_vs_manual_decode_probe_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
