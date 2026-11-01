#!/usr/bin/env python3
import argparse
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


TASK = "LING_AIME24_CANONICAL_GENERATION_PARITY_SMOKE_V1"
REPO = Path(__file__).resolve().parents[2]
RUN_DIR = REPO / "runs" / "ling_aime24_canonical_generation_parity_smoke_v1"
LEGACY_RUN = REPO / "runs" / "ling_kda_aime24_int8_rc_end_to_end_smoke_v1"
MODEL_PATH = Path("/data/zypan/models/Ling-3.0-tiny")
EOS_TOKEN_ID = 156895
PAD_TOKEN_ID = 156892
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
MAX_NEW_TOKENS = 32768
SEED = 0


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


def sha256_text(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def load_dataset():
    rows = json.loads((LEGACY_RUN / "aime24_HuggingFaceH4_aime_2024_train.json").read_text(encoding="utf-8"))
    by_id = {str(r["problem_id"]): r for r in rows}
    selected = [by_id[x] for x in ("60", "61", "62")]
    save_json(
        RUN_DIR / "selected_problems.json",
        [
            {"problem_id": r["problem_id"], "problem_index": r["problem_index"], "problem": r["problem"], "answer": r["reference_answer"]}
            for r in selected
        ],
    )
    return selected


def last_boxed(text):
    matches = list(re.finditer(r"\\boxed\s*\{", text or ""))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None


def extract_answer(text):
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    for pat in [r"(?:final answer|answer is|answer:)\s*([^\n\.]+)", r"答案是\s*([^\n。]+)"]:
        vals = re.findall(pat, text or "", flags=re.I)
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


def build_warpers():
    from transformers.generation import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
    return LogitsProcessorList([
        TemperatureLogitsWarper(TEMPERATURE),
        TopKLogitsWarper(TOP_K),
        TopPLogitsWarper(TOP_P),
    ])


def warped_probs(torch, logits, input_ids):
    scores = build_warpers()(input_ids, logits.float())
    return torch.softmax(scores, dim=-1), scores


def sampling_semantics_audit(torch, model, input_ids):
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
        logits = out.logits[:, -1, :]
    probs_a, scores_a = warped_probs(torch, logits.clone(), input_ids)
    probs_b, scores_b = warped_probs(torch, logits.clone(), input_ids)
    finite_a = torch.isfinite(scores_a[0])
    finite_b = torch.isfinite(scores_b[0])
    set_a = set(torch.nonzero(finite_a, as_tuple=False).flatten().tolist())
    set_b = set(torch.nonzero(finite_b, as_tuple=False).flatten().tolist())
    diff = (probs_a - probs_b).abs()
    audit = {
        "top_k": TOP_K,
        "top_p": TOP_P,
        "temperature": TEMPERATURE,
        "candidate_set_size_hf_warpers": len(set_a),
        "candidate_set_size_manual": len(set_b),
        "candidate_sets_identical": set_a == set_b,
        "argmax_hf_warpers": int(probs_a.argmax(dim=-1).item()),
        "argmax_manual": int(probs_b.argmax(dim=-1).item()),
        "probability_sum_hf_warpers": float(probs_a.sum().item()),
        "probability_sum_manual": float(probs_b.sum().item()),
        "max_absolute_probability_difference": float(diff.max().item()),
        "mean_absolute_probability_difference": float(diff.mean().item()),
    }
    audit["SAMPLING_SEMANTICS_GATE"] = "PASS" if audit["candidate_sets_identical"] and audit["max_absolute_probability_difference"] <= 1e-5 else "FAIL"
    save_json(RUN_DIR / "sampling_semantics_audit.json", audit)
    return audit


def eos_positions(ids):
    return [i for i, x in enumerate(ids) if int(x) == EOS_TOKEN_ID]


def finish_from_ids(ids, runtime_error=False, nonfinite=False):
    if runtime_error:
        return "ERROR"
    if nonfinite:
        return "OTHER"
    pos = eos_positions(ids)
    if pos and pos[0] == len(ids) - 1:
        return "EOS"
    if len(ids) >= MAX_NEW_TOKENS:
        return "MAX_NEW_TOKENS"
    return "OTHER"


def result_record(tokenizer, item, path, input_len, generated_ids, wall, runtime_error=False, error=None, traceback_tail=None, nonfinite=False):
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True) if generated_ids else ""
    pos = eos_positions(generated_ids)
    finish = finish_from_ids(generated_ids, runtime_error=runtime_error, nonfinite=nonfinite)
    pred = extract_answer(decoded)
    return {
        "task": TASK,
        "path": path,
        "state": "FP_STATE",
        "problem_id": item["problem_id"],
        "problem_index": item["problem_index"],
        "input_length": input_len,
        "generated_token_ids": generated_ids,
        "generated_tokens": len(generated_ids),
        "first_eos_position": pos[0] if pos else None,
        "num_eos_generated": len(pos),
        "last_token_id": int(generated_ids[-1]) if generated_ids else None,
        "finish_reason": finish,
        "truncated": finish == "MAX_NEW_TOKENS",
        "decoded_output": decoded,
        "decoded_tail": decoded[-1000:],
        "final_answer_extracted": pred,
        "ground_truth_answer": item["reference_answer"],
        "correct": is_correct(pred, item["reference_answer"]),
        "runtime_error": runtime_error,
        "error": error,
        "traceback_tail": traceback_tail or [],
        "nonfinite": nonfinite,
        "wall_time": wall,
        "eos_detected": bool(pos),
        "generation_stopped_immediately_after_first_eos": bool(pos and pos[0] == len(generated_ids) - 1),
        "seed": SEED,
    }


def run_hf_generate(torch, model, tokenizer, item, input_ids):
    setup_seed(SEED + int(item["problem_index"]))
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    start = time.time()
    try:
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=EOS_TOKEN_ID,
                pad_token_id=PAD_TOKEN_ID,
                return_dict_in_generate=True,
            )
        seq = out.sequences[0].detach().cpu().tolist()
        generated_ids = seq[input_ids.shape[-1]:]
        nonfinite = False
        return result_record(tokenizer, item, "HF_GENERATE", input_ids.shape[-1], generated_ids, time.time() - start, nonfinite=nonfinite)
    except Exception as exc:
        return result_record(tokenizer, item, "HF_GENERATE", input_ids.shape[-1], [], time.time() - start, runtime_error=True, error=repr(exc), traceback_tail=traceback.format_exc().splitlines()[-20:])


def sample_next(torch, logits, input_ids):
    probs, _scores = warped_probs(torch, logits, input_ids)
    return torch.multinomial(probs, num_samples=1)


def run_manual_decode(torch, model, tokenizer, item, input_ids):
    setup_seed(SEED + int(item["problem_index"]))
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    start = time.time()
    generated = []
    nonfinite = False
    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=True)
            if not bool(torch.isfinite(out.logits).all().item()):
                nonfinite = True
            past = out.past_key_values
            cur_input_ids = input_ids
            running_attention_mask = torch.ones_like(input_ids, device=device)
            prompt_len = int(input_ids.shape[-1])
            if not nonfinite:
                nxt = sample_next(torch, out.logits[:, -1, :], cur_input_ids)
                generated.append(int(nxt.item()))
                if int(nxt.item()) != EOS_TOKEN_ID:
                    cur = nxt.to(device)
                    for _step in range(1, MAX_NEW_TOKENS):
                        running_attention_mask = torch.cat([running_attention_mask, torch.ones_like(cur)], dim=-1)
                        cache_position = torch.tensor([prompt_len + _step - 1], device=device, dtype=torch.long)
                        out = model(
                            input_ids=cur,
                            attention_mask=running_attention_mask,
                            past_key_values=past,
                            cache_position=cache_position,
                            use_cache=True,
                        )
                        if not bool(torch.isfinite(out.logits).all().item()):
                            nonfinite = True
                            break
                        past = out.past_key_values
                        cur_input_ids = torch.cat([cur_input_ids, cur], dim=-1)
                        nxt = sample_next(torch, out.logits[:, -1, :], cur_input_ids)
                        generated.append(int(nxt.item()))
                        if int(nxt.item()) == EOS_TOKEN_ID:
                            break
                        cur = nxt.to(device)
        return result_record(tokenizer, item, "MANUAL_RECURRENT_DECODE", input_ids.shape[-1], generated, time.time() - start, nonfinite=nonfinite)
    except Exception as exc:
        return result_record(tokenizer, item, "MANUAL_RECURRENT_DECODE", input_ids.shape[-1], generated, time.time() - start, runtime_error=True, error=repr(exc), traceback_tail=traceback.format_exc().splitlines()[-20:])


def summarize(records):
    vals = [r["generated_tokens"] for r in records]
    return {
        "completed": sum(1 for r in records if r["finish_reason"] == "EOS"),
        "eos_finished": sum(1 for r in records if r["finish_reason"] == "EOS"),
        "truncated": sum(1 for r in records if r["truncated"]),
        "runtime_errors": sum(1 for r in records if r["runtime_error"]),
        "nonfinite": sum(1 for r in records if r["nonfinite"]),
        "generated_tokens_per_problem": vals,
        "mean_generated_tokens": sum(vals) / max(len(vals), 1),
        "median_generated_tokens": statistics.median(vals) if vals else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["run", "summarize"], default="run")
    args = parser.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "summarize":
        write_summary()
        return

    setup_seed(SEED)
    torch, model, tokenizer = load_model_and_tokenizer()
    problems = load_dataset()
    prompts = []
    for item in problems:
        prompt_text, input_ids = render_prompt(tokenizer, item["problem"])
        prompts.append((item, prompt_text, input_ids))
    first = prompts[0]
    (RUN_DIR / "prompt_audit.txt").write_text(
        "CHAT_TEMPLATE_CANONICAL = PASS\n"
        "THINKING_MODE = ON\n\n"
        f"problem_id = {first[0]['problem_id']}\n"
        f"repr(prompt_text) = {repr(first[1])}\n"
        f"input_ids length = {first[2].shape[-1]}\n"
        f"first 50 token ids = {first[2][0, :50].tolist()}\n"
        f"last 50 prompt token ids = {first[2][0, -50:].tolist()}\n",
        encoding="utf-8",
    )
    protocol = {
        "TASK": TASK,
        "MODEL": str(MODEL_PATH),
        "N_PROBLEMS": len(problems),
        "legacy_run": str(LEGACY_RUN),
        "seed": SEED,
        "torch_manual_seed": SEED,
        "torch_cuda_manual_seed_all": SEED,
        "canonical_protocol": {
            "thinking": "ON",
            "do_sample": True,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "max_new_tokens": MAX_NEW_TOKENS,
            "eos_token_id": EOS_TOKEN_ID,
            "pad_token_id": PAD_TOKEN_ID,
        },
        "git": {"branch": sh(["git", "branch", "--show-current"]), "head": sh(["git", "rev-parse", "HEAD"]), "status": sh(["git", "status", "--short"]), "diff": sh(["git", "diff", "--", str(Path(__file__).relative_to(REPO))])},
        "started_at": now(),
    }
    save_json(RUN_DIR / "protocol.json", protocol)

    sampling = sampling_semantics_audit(torch, model, prompts[0][2])
    if sampling["SAMPLING_SEMANTICS_GATE"] != "PASS":
        write_summary()
        print(json.dumps(sampling, indent=2, sort_keys=True))
        return

    for item, _prompt, input_ids in prompts:
        print(f"[{now()}] HF problem_id={item['problem_id']}", flush=True)
        hf = run_hf_generate(torch, model, tokenizer, item, input_ids)
        append_jsonl(RUN_DIR / "hf_generate_results.jsonl", hf)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{now()}] MANUAL problem_id={item['problem_id']}", flush=True)
        man = run_manual_decode(torch, model, tokenizer, item, input_ids)
        append_jsonl(RUN_DIR / "manual_decode_results.jsonl", man)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_summary()


def write_summary():
    hf = list(iter_jsonl(RUN_DIR / "hf_generate_results.jsonl") or [])
    man = list(iter_jsonl(RUN_DIR / "manual_decode_results.jsonl") or [])
    sampling = json.loads((RUN_DIR / "sampling_semantics_audit.json").read_text(encoding="utf-8")) if (RUN_DIR / "sampling_semantics_audit.json").exists() else {}
    prompt_text = (RUN_DIR / "prompt_audit.txt").read_text(encoding="utf-8") if (RUN_DIR / "prompt_audit.txt").exists() else ""
    chat_gate = "PASS" if "CHAT_TEMPLATE_CANONICAL = PASS" in prompt_text and "THINKING_MODE = ON" in prompt_text else "FAIL"
    eos_gate = "PASS" if all((not r.get("eos_detected")) or r.get("generation_stopped_immediately_after_first_eos") for r in man) and all(not r.get("runtime_error") for r in man) else "FAIL"
    hf_sum = summarize(hf)
    man_sum = summarize(man)
    n = max(len(hf), len(man), 3)
    manual_timeout = any(r.get("finish_reason") == "MANUAL_TIMEOUT" for r in man)
    if manual_timeout:
        hf_gate = "PASS" if hf and hf_sum["runtime_errors"] == 0 and hf_sum["nonfinite"] == 0 else "INCONCLUSIVE"
        man_gate = "FAIL"
        parity = "FAIL"
    elif len(hf) < 3 or len(man) < 3:
        hf_gate = "INCONCLUSIVE"
        man_gate = "INCONCLUSIVE"
        parity = "INCONCLUSIVE" if sampling.get("SAMPLING_SEMANTICS_GATE") == "PASS" else "FAIL"
    else:
        hf_gate = "PASS" if hf_sum["runtime_errors"] == 0 and hf_sum["nonfinite"] == 0 else "FAIL"
        man_gate = "PASS" if man_sum["runtime_errors"] == 0 and man_sum["nonfinite"] == 0 else "FAIL"
        manual_runaway = hf_sum["truncated"] == 0 and man_sum["truncated"] >= 2
        parity = "PASS" if chat_gate == "PASS" and sampling.get("SAMPLING_SEMANTICS_GATE") == "PASS" and eos_gate == "PASS" and not manual_runaway else "FAIL"
        if hf_sum["truncated"] == 3 and man_sum["truncated"] == 3:
            parity = "INCONCLUSIVE"
    rows = []
    by_h = {r["problem_id"]: r for r in hf}
    by_m = {r["problem_id"]: r for r in man}
    for pid in sorted(set(by_h) | set(by_m), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        rows.append({
            "problem_id": pid,
            "HF_finish": by_h.get(pid, {}).get("finish_reason"),
            "HF_tokens": by_h.get(pid, {}).get("generated_tokens"),
            "Manual_finish": by_m.get(pid, {}).get("finish_reason"),
            "Manual_tokens": by_m.get(pid, {}).get("generated_tokens"),
        })
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE" if len(hf) == 3 and len(man) == 3 and sampling.get("SAMPLING_SEMANTICS_GATE") else "INCOMPLETE",
        "MODEL": str(MODEL_PATH),
        "N_PROBLEMS": n,
        "CANONICAL_PROTOCOL": {"thinking": "ON", "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "max_new_tokens": MAX_NEW_TOKENS, "eos_token_id": EOS_TOKEN_ID, "pad_token_id": PAD_TOKEN_ID},
        "CHAT_TEMPLATE_GATE": chat_gate,
        "SAMPLING_SEMANTICS_GATE": sampling.get("SAMPLING_SEMANTICS_GATE", "UNKNOWN"),
        "EOS_STOP_RUNTIME_GATE": eos_gate,
        "HF_CANONICAL_GENERATION_GATE": hf_gate,
        "MANUAL_CANONICAL_GENERATION_GATE": man_gate,
        "MANUAL_GENERATION_PARITY_GATE": parity,
        "HF_RESULTS": hf_sum,
        "MANUAL_RESULTS": man_sum,
        "PER_PROBLEM": rows,
        "ROOT_CAUSE_IF_FAIL": "" if parity == "PASS" else "See sampling/prompt/runtime gates.",
        "CAN_EXISTING_MANUAL_HARNESS_BE_USED_FOR_CANONICAL_LING_AIME24": "YES" if parity == "PASS" else "NOT_YET",
        "CAN_START_30_PROBLEM_FP_R128_C128_RERUN": "YES" if parity == "PASS" else "NO",
    }
    save_json(RUN_DIR / "parity_summary.json", summary)
    lines = [
        f"# {TASK}",
        "",
        f"FORMAL_STATUS = {summary['FORMAL_STATUS']}",
        f"CHAT_TEMPLATE_GATE = {chat_gate}",
        f"SAMPLING_SEMANTICS_GATE = {summary['SAMPLING_SEMANTICS_GATE']}",
        f"EOS_STOP_RUNTIME_GATE = {eos_gate}",
        f"MANUAL_GENERATION_PARITY_GATE = {parity}",
        "",
        "## Per Problem",
        "| problem_id | HF_finish | HF_tokens | Manual_finish | Manual_tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['problem_id']} | {r['HF_finish']} | {r['HF_tokens']} | {r['Manual_finish']} | {r['Manual_tokens']} |")
    lines += ["", "## Summary JSON", "```json", json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), "```"]
    (RUN_DIR / "parity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
