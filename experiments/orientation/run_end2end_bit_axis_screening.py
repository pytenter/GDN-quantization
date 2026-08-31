#!/usr/bin/env python3
"""End-to-end bit/axis screening for Qwen3.5 GDN state quantization."""
import argparse
import os
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("GDN_DATA_ROOT", "/path/to/gdn_data_root"))
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
RESULT = ROOT / "results" / "gdn_end2end_bit_axis_screening_v1.json"
RECORDS = ROOT / "results" / "gdn_end2end_bit_axis_screening_v1_records.jsonl"
PROTOCOL = ROOT / "results" / "gdn_end2end_bit_axis_screening_v1_protocol.json"
SHARDS = ROOT / "results" / "gdn_end2end_bit_axis_screening_v1_shards"
SUBSET = ROOT / "results" / "gdn_end2end_bit_axis_screening_v1_math_subset.json"
REPORT = ROOT / "reports" / "gdn_end2end_bit_axis_screening_v1.md"
CONFIG_PATH = EXP / "run_config.json"
TRANSFORMERS_SRC = ROOT / "transformers-qwen35"

MATH_ARROW = os.environ.get(
    "GDN_MATH500_ARROW",
    "/path/to/HuggingFaceH4___math-500/default/0.0.0/math-500-test.arrow",
)
AIME_ARROW = os.environ.get(
    "GDN_AIME2024_ARROW",
    "/path/to/HuggingFaceH4___aime_2024/default/0.0.0/aime_2024-train.arrow",
)

TASK = "GDN_END2END_BIT_AXIS_SCREENING_V1"
SEED = 20260827
MATH_N = 30
MATH_MAX_NEW = 32768
AIME_MAX_NEW = 32768
SMOKE_MAX_NEW = 256
GDN_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,
    16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30,
]
CONFIGS = {
    "fp_state": ("FP_STATE", None, None, None),
    "int8_row": ("INT8-row", 8, "row", "row"),
    "int8_column": ("INT8-column", 8, "column", "column"),
    "int4_row": ("INT4-row", 4, "row", "row"),
    "int4_column": ("INT4-column", 4, "column", "column"),
}
CONFIG_ORDER = ["fp_state", "int8_row", "int8_column", "int4_row", "int4_column"]
DEFAULT_BENCHMARKS = ["MATH-500"]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_imports():
    for p in (str(EXP), str(TRANSFORMERS_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(fh, obj):
    fh.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")
    fh.flush()


def sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def cfg_dict(key):
    q, bits, axis, gran = CONFIGS[key]
    return {
        "key": key,
        "quantizer": q,
        "bit_width": bits,
        "axis": axis,
        "bits": bits,
        "granularity": gran,
    }


def load_arrow(path):
    from datasets import Dataset
    return Dataset.from_file(path)


def make_item(benchmark, row, idx):
    if benchmark == "MATH-500":
        return {
            "benchmark": benchmark,
            "problem_id": row.get("unique_id") or "math_%03d" % idx,
            "problem_index": idx,
            "problem": row["problem"],
            "reference_answer": str(row["answer"]),
            "solution": row.get("solution"),
            "subject": row.get("subject"),
            "level": row.get("level"),
        }
    return {
        "benchmark": benchmark,
        "problem_id": str(row.get("id", idx)),
        "problem_index": idx,
        "problem": row["problem"],
        "reference_answer": str(row["answer"]),
        "solution": row.get("solution"),
        "url": row.get("url"),
        "year": row.get("year"),
    }


def load_benchmarks(smoke=False):
    math_ds = load_arrow(MATH_ARROW)
    aime_ds = load_arrow(AIME_ARROW)
    if len(math_ds) != 500:
        raise RuntimeError("MATH-500 length mismatch: %d" % len(math_ds))
    if len(aime_ds) != 30:
        raise RuntimeError("AIME-2024 length mismatch: %d" % len(aime_ds))
    rng = random.Random(SEED)
    indices = list(range(500))
    rng.shuffle(indices)
    indices = indices[:MATH_N]
    math_items = [make_item("MATH-500", math_ds[i], i) for i in indices]
    aime_items = [make_item("AIME-2024", aime_ds[i], i) for i in range(30)]
    save_json(SUBSET, {
        "task": TASK,
        "subset_seed": SEED,
        "source_arrow": MATH_ARROW,
        "n": len(math_items),
        "indices": indices,
        "records": math_items,
    })
    if smoke:
        return math_items[:2], aime_items[:2]
    return math_items, aime_items


def prompt_for(problem):
    return (
        "Solve the following math problem. Give the final answer in "
        "\\boxed{} form.\n\nProblem:\n%s\n\nSolution:" % problem.strip()
    )


def render_prompt(tokenizer, problem):
    user = prompt_for(problem)
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        messages = [{"role": "user", "content": user}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    return user


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


def extract_answer(text, benchmark):
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    for pat in [
        r"(?:final answer|answer is|answer:)\s*([^\n\.]+)",
        r"答案是\s*([^\n。]+)",
    ]:
        vals = re.findall(pat, text or "", flags=re.I)
        if vals:
            return vals[-1].strip()
    if benchmark == "AIME-2024":
        nums = re.findall(r"(?<![\d\-])\d{1,3}(?!\d)", text or "")
        return nums[-1] if nums else ""
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text or "")
    return nums[-1] if nums else ""


def normalize_answer(ans):
    ans = str(ans or "").strip()
    ans = last_boxed(ans) or ans
    ans = ans.replace("\n", " ")
    ans = re.sub(r"\\left|\\right", "", ans)
    ans = re.sub(r"\\(?:,|;|!|\s)", "", ans)
    ans = ans.replace("$", "").replace(",", "")
    ans = ans.replace(" ", "").strip(".。")
    return ans.lower()


def numeric_value(ans):
    s = normalize_answer(ans)
    try:
        if re.fullmatch(r"-?\d+/\d+", s):
            a, b = s.split("/")
            return float(a) / float(b)
        return float(s)
    except Exception:
        return None


def is_correct(pred, ref, benchmark):
    if benchmark == "AIME-2024":
        p = re.findall(r"\d+", str(pred))
        r = re.findall(r"\d+", str(ref))
        return bool(p and r and int(p[-1]) == int(r[-1]))
    pn = normalize_answer(pred)
    rn = normalize_answer(ref)
    if pn and pn == rn:
        return True
    pv = numeric_value(pred)
    rv = numeric_value(ref)
    return pv is not None and rv is not None and abs(pv - rv) <= 1e-6


def get_state(cache, layer_idx, state_idx=0):
    states = getattr(cache.layers[layer_idx], "recurrent_states")
    return states[state_idx] if not isinstance(states, dict) else states[state_idx]


def quantize_gdn_cache(torch, cache, qcfg, touched, scale_shapes):
    if qcfg["bits"] is None:
        return True, None, 0
    from state_fake_quant import fake_quant_state
    for layer_idx in GDN_LAYERS:
        state = get_state(cache, layer_idx)
        qdq, scale = fake_quant_state(
            state, bits=qcfg["bits"], granularity=qcfg["granularity"])
        touched.add(layer_idx)
        scale_shapes.setdefault(str(layer_idx), list(scale.shape))
        if not bool(torch.isfinite(qdq).all().item()):
            bad = int((~torch.isfinite(qdq)).sum().item())
            return False, layer_idx, bad
        state.copy_(qdq.to(state.dtype))
    return True, None, 0


def top_p_sample(torch, logits, temperature, top_p, generator):
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits.float() / float(temperature)
    probs = torch.softmax(logits, dim=-1)
    vals, idx = torch.sort(probs, descending=True, dim=-1)
    cdf = torch.cumsum(vals, dim=-1)
    mask = cdf > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    vals = vals.masked_fill(mask, 0.0)
    vals = vals / vals.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    pick = torch.multinomial(vals, num_samples=1, generator=generator)
    return idx.gather(-1, pick)


def generate_one(torch, model, tokenizer, item, qcfg, max_new):
    device = next(model.parameters()).device
    prompt = render_prompt(tokenizer, item["problem"])
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)
    benchmark = item["benchmark"]
    temp = 0.0 if benchmark == "MATH-500" else 0.6
    top_p = 1.0 if benchmark == "MATH-500" else 0.95
    seed = SEED + int(item["problem_index"])
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(int(tid))
    tokens = []
    touched = set()
    scale_shapes = {}
    nonfinite = False
    bad_layer = None
    bad_count = 0
    qcalls = 0
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
        past = out.past_key_values
        nxt = top_p_sample(torch, out.logits[:, -1, :], temp, top_p, gen)
        tokens.append(int(nxt.item()))
        cur = nxt.to(device)
        if int(cur.item()) not in stop_ids:
            for _ in range(1, max_new):
                out = model(input_ids=cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                ok, bad_layer, bad_count = quantize_gdn_cache(
                    torch, past, qcfg, touched, scale_shapes)
                if qcfg["bits"] is not None:
                    qcalls += 1
                if not ok:
                    nonfinite = True
                    break
                nxt = top_p_sample(torch, out.logits[:, -1, :], temp, top_p, gen)
                tokens.append(int(nxt.item()))
                cur = nxt.to(device)
                if int(cur.item()) in stop_ids:
                    break
    response = tokenizer.decode(tokens, skip_special_tokens=True)
    pred = extract_answer(response, benchmark)
    max_hit = len(tokens) >= max_new and int(tokens[-1]) not in stop_ids
    return {
        "task": TASK,
        "benchmark": benchmark,
        "problem_id": item["problem_id"],
        "problem_index": item["problem_index"],
        "quantizer": qcfg["quantizer"],
        "bit_width": qcfg["bit_width"],
        "axis": qcfg["axis"],
        "seed": seed,
        "correct": is_correct(pred, item["reference_answer"], benchmark),
        "predicted_answer": pred,
        "reference_answer": item["reference_answer"],
        "output_token_count": len(tokens),
        "completed": bool((not nonfinite) and (not max_hit)),
        "nonfinite": bool(nonfinite),
        "truncated": bool(max_hit),
        "max_token_hit": bool(max_hit),
        "response": response,
        "ref": item.get("solution"),
        "prompt_sha1": sha1(prompt),
        "prompt_format": "chat_template" if getattr(tokenizer, "chat_template", None) else "plain",
        "stop_token_ids": sorted(stop_ids),
        "all_gdn_touched": sorted(touched),
        "touched_gdn_count": len(touched) if qcfg["bits"] is not None else 0,
        "prefill_quantized": False,
        "decode_quantize_calls": qcalls,
        "scale_shapes": scale_shapes,
        "first_nonfinite_layer": bad_layer,
        "nonfinite_count": bad_count,
    }


def run_protocol():
    ensure_imports()
    import torch
    from state_fake_quant import fake_quant_state
    x = torch.randn(1, 32, 128, 128)
    rows = []
    ok_all = True
    for bits in (8, 4):
        for axis in ("row", "column"):
            qdq, scale = fake_quant_state(x, bits, axis)
            qmax = (2 ** (bits - 1)) - 1
            codes = torch.round(x / scale).clamp(-qmax, qmax)
            exp = [1, 32, 128, 1] if axis == "row" else [1, 32, 1, 128]
            rec = {
                "bits": bits,
                "axis": axis,
                "qrange": [-qmax, qmax],
                "scale_shape": list(scale.shape),
                "expected_scale_shape": exp,
                "dequant_shape": list(qdq.shape),
                "finite": bool(torch.isfinite(qdq).all().item()),
                "code_min": int(codes.min().item()),
                "code_max": int(codes.max().item()),
            }
            rec["pass"] = (
                rec["scale_shape"] == exp and
                rec["dequant_shape"] == list(x.shape) and
                rec["finite"] and
                rec["code_min"] >= -qmax and
                rec["code_max"] <= qmax
            )
            ok_all = ok_all and rec["pass"]
            rows.append(rec)
    previous = {}
    if PROTOCOL.exists():
        try:
            previous = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    obj = {
        "task": TASK,
        "time": now(),
        "QUANTIZER_PROTOCOL_GATE": "PASS" if ok_all else "FAIL",
        "canonical_signed_symmetric": True,
        "zero_point": 0,
        "rows": rows,
    }
    if previous.get("SMOKE_GATE"):
        obj["SMOKE_GATE"] = previous["SMOKE_GATE"]
    if previous.get("smoke_details"):
        obj["smoke_details"] = previous["smoke_details"]
    save_json(PROTOCOL, obj)
    print(json.dumps(obj, indent=2, sort_keys=True))
    return ok_all


def load_model():
    ensure_imports()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model_path = cfg["model_path"]
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    return torch, transformers, model, tokenizer


def planned_jobs(smoke, configs, benchmarks):
    math_items, aime_items = load_benchmarks(smoke=smoke)
    jobs = []
    for ck in configs:
        if "MATH-500" in benchmarks:
            max_new = SMOKE_MAX_NEW if smoke else MATH_MAX_NEW
            jobs += [(ck, x, max_new) for x in math_items]
        if "AIME-2024" in benchmarks:
            max_new = SMOKE_MAX_NEW if smoke else AIME_MAX_NEW
            jobs += [(ck, x, max_new) for x in aime_items]
    return jobs


def run_generation_stage(args, smoke=False):
    if not Path(MATH_ARROW).exists() or not Path(AIME_ARROW).exists():
        raise RuntimeError("offline arrow caches missing")
    if not run_protocol():
        raise RuntimeError("protocol gate failed")
    torch, _transformers, model, tokenizer = load_model()
    configs = args.configs.split(",") if args.configs else CONFIG_ORDER
    benchmarks = args.benchmarks.split(",") if args.benchmarks else DEFAULT_BENCHMARKS
    jobs = planned_jobs(smoke, configs, benchmarks)
    if args.num_shards > 1:
        jobs = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard_id]
    SHARDS.mkdir(parents=True, exist_ok=True)
    path = SHARDS / ("smoke.jsonl" if smoke else "shard_%03d.jsonl" % args.shard_id)
    done = set()
    if args.resume and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["quantizer"], r["benchmark"], r["problem_id"]))
    mode = "a" if args.resume else "w"
    t0 = time.time()
    with path.open(mode, encoding="utf-8") as fh:
        for i, (ck, item, max_new) in enumerate(jobs):
            qcfg = cfg_dict(ck)
            key = (qcfg["quantizer"], item["benchmark"], item["problem_id"])
            if key in done:
                continue
            print("[%s] %d/%d %s %s %s" % (
                now(), i + 1, len(jobs), qcfg["quantizer"],
                item["benchmark"], item["problem_id"]), flush=True)
            try:
                rec = generate_one(torch, model, tokenizer, item, qcfg, max_new)
            except Exception as exc:
                rec = {
                    "task": TASK,
                    "benchmark": item["benchmark"],
                    "problem_id": item["problem_id"],
                    "problem_index": item["problem_index"],
                    "quantizer": qcfg["quantizer"],
                    "bit_width": qcfg["bit_width"],
                    "axis": qcfg["axis"],
                    "seed": SEED + int(item["problem_index"]),
                    "correct": False,
                    "predicted_answer": "",
                    "reference_answer": item["reference_answer"],
                    "output_token_count": 0,
                    "completed": False,
                    "nonfinite": False,
                    "truncated": False,
                    "max_token_hit": False,
                    "response": "",
                    "ref": item.get("solution"),
                    "exception": repr(exc),
                }
            rec["run_elapsed_sec"] = time.time() - t0
            write_jsonl(fh, rec)
    if smoke:
        obj = verify_smoke(path)
        print(json.dumps(obj, indent=2, sort_keys=True))
        return obj["SMOKE_GATE"] == "PASS"
    return True


def read_records(paths):
    rows = []
    for p in paths:
        if Path(p).exists():
            with open(p, "r", encoding="utf-8") as f:
                rows += [json.loads(line) for line in f if line.strip()]
    return rows


def verify_smoke(path):
    rows = read_records([path])
    qnames = {cfg_dict(k)["quantizer"] for k in CONFIG_ORDER}
    seen = {r["quantizer"] for r in rows}
    benches = {r["benchmark"] for r in rows}
    quant_rows = [r for r in rows if r["quantizer"] != "FP_STATE"]
    touched_ok = all(r.get("touched_gdn_count") == 24 for r in quant_rows)
    prefill_ok = all(not r.get("prefill_quantized", False) for r in rows)
    decode_ok = all(r.get("decode_quantize_calls", 0) > 0 for r in quant_rows)
    shape_ok = True
    for r in quant_rows:
        for shape in (r.get("scale_shapes") or {}).values():
            if r["axis"] == "row" and shape[-1] != 1:
                shape_ok = False
            if r["axis"] == "column" and shape[-2] != 1:
                shape_ok = False
    extract_ok = all(str(r.get("predicted_answer", "")) != "" for r in rows)
    expected = len(qnames) * (2 if benches == {"MATH-500"} else 4)
    ok = (
        len(rows) == expected and seen == qnames and
        benches in ({"MATH-500"}, {"MATH-500", "AIME-2024"}) and
        touched_ok and prefill_ok and decode_ok and shape_ok and extract_ok
    )
    obj = {
        "task": TASK,
        "time": now(),
        "SMOKE_GATE": "PASS" if ok else "FAIL",
        "record_count": len(rows),
        "configs_seen": sorted(seen),
        "benchmarks_seen": sorted(benches),
        "all_24_gdn_touched": touched_ok,
        "prefill_not_quantized": prefill_ok,
        "decode_quantize_every_token_observed": decode_ok,
        "row_column_scale_shapes_observed": shape_ok,
        "answer_extraction_nonempty": extract_ok,
        "math_reproducibility_scope": "temperature_0 deterministic path",
        "aime_seed_reused": SEED,
    }
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8")) if PROTOCOL.exists() else {}
    proto["SMOKE_GATE"] = obj["SMOKE_GATE"]
    proto["smoke_details"] = obj
    save_json(PROTOCOL, proto)
    return obj


def summarize(rows):
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["benchmark"], r["quantizer"])].append(r)
    for (bench, quant), vals in groups.items():
        toks = sorted(int(r["output_token_count"]) for r in vals)
        out["%s|%s" % (bench, quant)] = {
            "N": len(vals),
            "correct": sum(1 for r in vals if r["correct"]),
            "accuracy": sum(1 for r in vals if r["correct"]) / max(len(vals), 1),
            "mean_tokens": sum(toks) / max(len(toks), 1),
            "median_tokens": toks[len(toks) // 2] if toks else None,
            "nonfinite": sum(1 for r in vals if r.get("nonfinite")),
            "truncated": sum(1 for r in vals if r.get("truncated")),
            "completed": sum(1 for r in vals if r.get("completed")),
        }
    return out


def paired(rows, benchmark):
    by_pid = defaultdict(dict)
    for r in rows:
        if r["benchmark"] == benchmark:
            by_pid[r["problem_id"]][r["quantizer"]] = r
    out = {}
    for ck in CONFIG_ORDER[1:]:
        q = cfg_dict(ck)["quantizer"]
        c = {
            "FP_correct_to_Q_correct": 0,
            "FP_correct_to_Q_wrong": 0,
            "FP_wrong_to_Q_correct": 0,
            "FP_wrong_to_Q_wrong": 0,
        }
        for m in by_pid.values():
            if "FP_STATE" not in m or q not in m:
                continue
            fp = bool(m["FP_STATE"]["correct"])
            qq = bool(m[q]["correct"])
            if fp and qq:
                c["FP_correct_to_Q_correct"] += 1
            elif fp and not qq:
                c["FP_correct_to_Q_wrong"] += 1
            elif (not fp) and qq:
                c["FP_wrong_to_Q_correct"] += 1
            else:
                c["FP_wrong_to_Q_wrong"] += 1
        c["FP_correct_quant_wrong_count"] = c["FP_correct_to_Q_wrong"]
        c["FP_wrong_quant_correct_count"] = c["FP_wrong_to_Q_correct"]
        c["net_correct_change"] = (
            c["FP_wrong_to_Q_correct"] - c["FP_correct_to_Q_wrong"])
        out[q] = c
    return out


def length_analysis(rows):
    bins = [
        ("<=256", lambda x: x <= 256),
        ("257-512", lambda x: 257 <= x <= 512),
        ("513-1024", lambda x: 513 <= x <= 1024),
        (">1024", lambda x: x > 1024),
    ]
    out = {}
    for bench in ("MATH-500", "AIME-2024"):
        fp_len = {
            r["problem_id"]: r["output_token_count"]
            for r in rows
            if r["benchmark"] == bench and r["quantizer"] == "FP_STATE"
        }
        out[bench] = {}
        for label, pred in bins:
            pids = {pid for pid, n in fp_len.items() if pred(n)}
            out[bench][label] = {}
            for ck in CONFIG_ORDER:
                q = cfg_dict(ck)["quantizer"]
                vals = [
                    r for r in rows
                    if r["benchmark"] == bench and
                    r["quantizer"] == q and r["problem_id"] in pids
                ]
                out[bench][label][q] = {
                    "N": len(vals),
                    "accuracy": (
                        sum(1 for r in vals if r["correct"]) / max(len(vals), 1)),
                    "underpowered": len(vals) < 10,
                }
    dist = defaultdict(list)
    for r in rows:
        dist[(r["benchmark"], r["quantizer"])].append(r["output_token_count"])
    out["quantizer_length_distribution"] = {
        "%s|%s" % k: {
            "N": len(v),
            "mean": sum(v) / max(len(v), 1),
            "min": min(v) if v else None,
            "max": max(v) if v else None,
        }
        for k, v in dist.items()
    }
    return out


def classify(summary, q):
    qm = summary.get("MATH-500|%s" % q)
    qa = summary.get("AIME-2024|%s" % q)
    fm = summary.get("MATH-500|FP_STATE")
    fa = summary.get("AIME-2024|FP_STATE")
    if not qm or not fm:
        return "INCONCLUSIVE"
    if qm["nonfinite"] or (qa and qa["nonfinite"]):
        return "NUMERICALLY_UNSTABLE"
    deltas = [qm["accuracy"] - fm["accuracy"]]
    if qa and fa:
        deltas.append(qa["accuracy"] - fa["accuracy"])
    worst = min(deltas)
    trunc_num = qm["truncated"] + (qa["truncated"] if qa else 0)
    trunc_den = qm["N"] + (qa["N"] if qa else 0)
    trunc = trunc_num / max(trunc_den, 1)
    if trunc > 0.25:
        return "LONG_HORIZON_RISK"
    if worst >= -0.01:
        return "NEAR_LOSSLESS_REFERENCE"
    if worst >= -0.05:
        return "VIABLE"
    if worst >= -0.15:
        return "VIABLE_BUT_DEGRADED"
    return "CLEAR_QUALITY_DEGRADATION"


def recommend(classes):
    good = [q for q, c in classes.items() if c in ("NEAR_LOSSLESS_REFERENCE", "VIABLE")]
    if any(q.startswith("INT4") for q in good):
        bit = "INT4"
    elif any(q.startswith("INT8") for q in good):
        bit = "INT8"
    else:
        bit = "INCONCLUSIVE"
    row = any(q.endswith("row") for q in good)
    col = any(q.endswith("column") for q in good)
    if row and col:
        axis = "MIXED_OR_ADAPTIVE"
    elif row:
        axis = "ROW"
    elif col:
        axis = "COLUMN"
    else:
        axis = "INCONCLUSIVE"
    return bit, axis


def copy_records():
    RECORDS.parent.mkdir(parents=True, exist_ok=True)
    with RECORDS.open("w", encoding="utf-8") as out:
        for p in sorted(SHARDS.glob("shard_*.jsonl")):
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line)


def analyze():
    rows = read_records(sorted(SHARDS.glob("shard_*.jsonl")))
    math_items, _aime_items = load_benchmarks(smoke=False)
    expected = len(CONFIG_ORDER) * len(math_items)
    seen = {(r["quantizer"], r["benchmark"], r["problem_id"]) for r in rows}
    summary = summarize(rows)
    classes = {
        cfg_dict(k)["quantizer"]: classify(summary, cfg_dict(k)["quantizer"])
        for k in CONFIG_ORDER[1:]
    }
    bit, axis = recommend(classes)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8")) if PROTOCOL.exists() else {}
    obj = {
        "TASK": TASK,
        "SMOKE_GATE": protocol.get("SMOKE_GATE", "UNKNOWN"),
        "QUANTIZER_PROTOCOL_GATE": protocol.get("QUANTIZER_PROTOCOL_GATE", "UNKNOWN"),
        "RUN_STATUS": "COMPLETE" if len(seen) == expected else "INCOMPLETE",
        "record_count": len(rows),
        "expected_record_count": expected,
        "MATH_200_RESULTS": {k: v for k, v in summary.items() if k.startswith("MATH-500|")},
        "AIME24_RESULTS": {k: v for k, v in summary.items() if k.startswith("AIME-2024|")},
        "PAIRED_FLIP_ANALYSIS": {
            "MATH-500": paired(rows, "MATH-500"),
            "AIME-2024": paired(rows, "AIME-2024"),
        },
        "OUTPUT_LENGTH_ANALYSIS": length_analysis(rows),
        "FAILURE_SUMMARY": {
            "exceptions": sum(1 for r in rows if r.get("exception")),
            "nonfinite": sum(1 for r in rows if r.get("nonfinite")),
            "truncated": sum(1 for r in rows if r.get("truncated")),
            "max_token_hit": sum(1 for r in rows if r.get("max_token_hit")),
            "missing_records": expected - len(seen),
        },
        "INT8_ROW_CLASSIFICATION": classes["INT8-row"],
        "INT8_COLUMN_CLASSIFICATION": classes["INT8-column"],
        "INT4_ROW_CLASSIFICATION": classes["INT4-row"],
        "INT4_COLUMN_CLASSIFICATION": classes["INT4-column"],
        "RECOMMENDED_RESEARCH_BIT": bit,
        "RECOMMENDED_RESEARCH_AXIS": axis,
        "END2END_BIT_AXIS_SCREENING_GATE": (
            "PASS" if len(seen) == expected else "INCOMPLETE"),
        "CURRENT_STRONGEST_END2END_CONCLUSION": (
            "Under identical 1x128 recurrent-state granularity, "
            "recommended bit=%s axis=%s; classes=%s." % (bit, axis, classes)
        ),
        "LIMITATIONS": [
            "Current resized run uses MATH-500 fixed subset N=30 only.",
            "AIME-2024 uses one sample per problem; one problem equals 3.33pp.",
            "Answer judging is normalized exact/numeric matching, not CAS.",
            "Mechanism status unchanged: STATE_CHANGE_SWAMPING_SIGNAL_IDENTIFIED; METHOD_DESIGN_READY=NO.",
        ],
        "NEXT_RECOMMENDED_EXPERIMENT": (
            "Do not auto-launch; review screening before choosing next experiment."),
    }
    save_json(RESULT, obj)
    write_report(obj)
    return obj


def write_report(obj):
    lines = ["# GDN End-to-End Bit/Axis Screening V1", ""]
    for key in [
        "TASK", "SMOKE_GATE", "QUANTIZER_PROTOCOL_GATE", "RUN_STATUS",
        "INT8_ROW_CLASSIFICATION", "INT8_COLUMN_CLASSIFICATION",
        "INT4_ROW_CLASSIFICATION", "INT4_COLUMN_CLASSIFICATION",
        "RECOMMENDED_RESEARCH_BIT", "RECOMMENDED_RESEARCH_AXIS",
        "END2END_BIT_AXIS_SCREENING_GATE",
        "CURRENT_STRONGEST_END2END_CONCLUSION",
    ]:
        lines.append("- **%s**: `%s`" % (key, obj.get(key)))
    for title, key in [
        ("Table A: MATH-500 fixed subset N=%d" % MATH_N, "MATH_200_RESULTS"),
        ("Table B: AIME-2024 all 30", "AIME24_RESULTS"),
        ("Table C: Paired flip analysis", "PAIRED_FLIP_ANALYSIS"),
        ("Table D: Output-length analysis", "OUTPUT_LENGTH_ANALYSIS"),
        ("Failure summary", "FAILURE_SUMMARY"),
    ]:
        lines += ["", "## " + title, "```json",
                  json.dumps(obj[key], indent=2, sort_keys=True, ensure_ascii=False),
                  "```"]
    lines += ["", "## Limitations"]
    lines += ["- " + x for x in obj["LIMITATIONS"]]
    lines += ["", "## Next Recommended Experiment", obj["NEXT_RECOMMENDED_EXPERIMENT"]]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description=TASK)
    p.add_argument("--stage", choices=["protocol", "smoke", "main", "analyze", "all"],
                   default="all")
    p.add_argument("--configs", default=None)
    p.add_argument("--benchmarks", default=None)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.stage == "protocol":
        raise SystemExit(0 if run_protocol() else 2)
    if args.stage == "smoke":
        raise SystemExit(0 if run_generation_stage(args, smoke=True) else 3)
    if args.stage == "main":
        raise SystemExit(0 if run_generation_stage(args, smoke=False) else 4)
    if args.stage == "analyze":
        copy_records()
        print(json.dumps(analyze(), indent=2, sort_keys=True, ensure_ascii=False))
        raise SystemExit(0)
    if args.stage == "all":
        if not run_generation_stage(args, smoke=True):
            raise SystemExit(3)
        if not run_generation_stage(args, smoke=False):
            raise SystemExit(4)
        copy_records()
        print(json.dumps(analyze(), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
