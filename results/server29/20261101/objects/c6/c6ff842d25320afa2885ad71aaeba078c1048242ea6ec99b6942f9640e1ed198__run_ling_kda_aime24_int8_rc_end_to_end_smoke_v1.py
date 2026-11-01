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
import urllib.request
from collections import defaultdict
from pathlib import Path


TASK = "LING_KDA_AIME24_INT8_RC_END_TO_END_SMOKE_V1"
SLUG = "ling_kda_aime24_int8_rc_end_to_end_smoke_v1"
REPO = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.environ.get("LING_MODEL_PATH", "/data01/user2/models/Ling-3.0-tiny"))
MODEL_REVISION = "e3a47d5b986e7141b6efd62597d598ebb392060d"
RUN_DIR = REPO / "runs" / SLUG
RESULT_DIR = REPO / "results" / "ling"
REPORT_DIR = REPO / "reports" / "ling"
FINAL_SUMMARY_PATH = RESULT_DIR / f"{SLUG}_final_summary.json"
FINAL_REPORT_PATH = REPORT_DIR / f"{SLUG}.md"
DATASET_API = "https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/aime_2024&config=default&split=train&offset=0&length=30"
SEED = 20260827
AIME_MAX_NEW_TOKENS = 32768
SMOKE_MAX_NEW_TOKENS = 512
CONFIG_ORDER = ["FP_STATE", "INT8_R128", "INT8_C128"]
EPS = 1e-12


def offline_env_vars():
    return [
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
    ]


def revision_source_path():
    return Path(os.environ.get("LING_ENV_SUMMARY_PATH", "/data01/user2/ling-kda-env-summary.txt"))


def configure_offline_runtime():
    for key in offline_env_vars():
        os.environ[key] = "1"


def selected_configs(configs_text):
    if not configs_text:
        return list(CONFIG_ORDER)
    configs = [x.strip() for x in configs_text.split(",") if x.strip()]
    invalid = [x for x in configs if x not in CONFIG_ORDER]
    if invalid:
        raise ValueError(f"Configs outside protocol are not allowed: {invalid}")
    return configs


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path, obj):
    path = Path(path)
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


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sh(cmd, cwd=REPO):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
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


def validate_shard_args(shard_id, num_shards):
    if int(num_shards) < 1:
        raise ValueError("--num-shards must be >= 1")
    if int(shard_id) < 0 or int(shard_id) >= int(num_shards):
        raise ValueError("--shard-id must satisfy 0 <= shard-id < num-shards")


def config_to_file(config, shard_id=None, num_shards=1):
    base = {
        "FP_STATE": RUN_DIR / "fp_state.jsonl",
        "INT8_R128": RUN_DIR / "int8_r128.jsonl",
        "INT8_C128": RUN_DIR / "int8_c128.jsonl",
    }[config]
    if int(num_shards) == 1 and shard_id in (None, 0):
        return base
    if shard_id is None:
        return base
    validate_shard_args(shard_id, num_shards)
    return base.with_name(f"{base.stem}_shard{int(shard_id):03d}-of-{int(num_shards):03d}{base.suffix}")


def config_record_paths(config):
    base = config_to_file(config)
    paths = []
    if base.exists():
        paths.append(base)
    paths.extend(p for p in sorted(RUN_DIR.glob(f"{base.stem}_shard*-of-*{base.suffix}")) if p not in paths)
    return paths


def formal_items_for_shard(rows, shard_id, num_shards):
    validate_shard_args(shard_id, num_shards)
    if int(num_shards) == 1:
        return list(rows)
    return [item for idx, item in enumerate(rows) if idx % int(num_shards) == int(shard_id)]


def quantizer_axis(config):
    if config == "INT8_R128":
        return {
            "config": config,
            "axis": "row",
            "bits": 8,
            "qmax": 127,
            "scale_amax_dims": (-1,),
            "expected_scale_shape_suffix": "Kx1",
            "canonical_meaning": "R128 = full 128-element Value-axis row group",
            "ling_semantic_axis": "Key-side / recurrent-state row orientation",
            "ling_state_shape": "[B,H,K,V]",
            "grouped_values": "for each fixed [B,H,K], group all V values",
        }
    if config == "INT8_C128":
        return {
            "config": config,
            "axis": "column",
            "bits": 8,
            "qmax": 127,
            "scale_amax_dims": (-2,),
            "expected_scale_shape_suffix": "1xV",
            "canonical_meaning": "C128 = full 128-element Key-axis column group",
            "ling_semantic_axis": "Value-side / recurrent-state column orientation",
            "ling_state_shape": "[B,H,K,V]",
            "grouped_values": "for each fixed [B,H,V], group all K values",
        }
    raise ValueError(config)


def fake_quant_ling_state(state, config):
    import torch
    q = quantizer_axis(config)
    scale = state.detach().float().abs().amax(dim=q["scale_amax_dims"], keepdim=True).clamp_min(EPS) / q["qmax"]
    codes = torch.round(state.detach().float() / scale).clamp(-q["qmax"], q["qmax"])
    qdq = codes * scale
    meta = {
        "config": config,
        "bits": 8,
        "qrange": [-q["qmax"], q["qmax"]],
        "zero_point": 0,
        "symmetric": True,
        "scale_formula": "amax(abs(group)).clamp_min(1e-12) / 127",
        "rounding": "torch.round",
        "dequantization": "codes * scale",
        "amax_dims": list(q["scale_amax_dims"]),
        "scale": scale,
        "codes": codes,
        "ling_semantic_axis": q["ling_semantic_axis"],
    }
    return qdq.to(state.dtype), meta


def small_tensor_stats(torch, state, scale, codes):
    sf = scale.detach().float()
    xf = state.detach().float()
    cf = codes.detach().float()
    qmax = 127.0
    return {
        "state_shape": list(state.shape),
        "state_dtype": str(state.dtype),
        "state_min": float(xf.min().item()),
        "state_max": float(xf.max().item()),
        "state_abs_max": float(xf.abs().max().item()),
        "scale_shape": list(scale.shape),
        "scale_min": float(sf.min().item()),
        "scale_max": float(sf.max().item()),
        "scale_finite": bool(torch.isfinite(scale).all().item()),
        "state_finite": bool(torch.isfinite(state).all().item()),
        "dequant_finite": None,
        "saturation_fraction": float((cf.abs() >= qmax).float().mean().item()),
    }


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


def prompt_for(problem):
    return (
        "Solve the following math problem. Give the final answer in "
        "\\boxed{} form.\n\nProblem:\n%s\n\nSolution:" % problem.strip()
    )


def render_prompt(tokenizer, problem):
    user = prompt_for(problem)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return user


def fetch_aime24(offline=True):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    dataset_copy = RUN_DIR / "aime24_HuggingFaceH4_aime_2024_train.json"
    meta_path = RUN_DIR / "aime24_dataset_source.json"
    if dataset_copy.exists():
        raw = dataset_copy.read_bytes()
        rows = json.loads(raw.decode("utf-8"))
        return rows, {
            "source": "HuggingFaceH4/aime_2024",
            "api_url": DATASET_API,
            "immutable_copy": str(dataset_copy),
            "sha256": sha256_bytes(raw),
            "loaded_from": "local_immutable_copy",
            "offline_mode": True,
        }
    if offline:
        raise RuntimeError(f"Offline AIME24 copy missing: {dataset_copy}")
    with urllib.request.urlopen(DATASET_API, timeout=60) as resp:
        api_obj = json.load(resp)
    if int(api_obj.get("num_rows_total", -1)) != 30:
        raise RuntimeError(f"AIME24 dataset length mismatch: {api_obj.get('num_rows_total')}")
    rows = []
    for item in api_obj["rows"]:
        row = item["row"]
        rows.append({
            "dataset": "AIME-2024",
            "problem_id": str(row.get("id", item["row_idx"])),
            "problem_index": int(item["row_idx"]),
            "problem": row["problem"],
            "reference_answer": str(row["answer"]),
            "solution": row.get("solution"),
            "url": row.get("url"),
            "year": row.get("year"),
        })
    raw = json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    dataset_copy.write_bytes(raw)
    meta = {
        "source": "HuggingFaceH4/aime_2024",
        "api_url": DATASET_API,
        "immutable_copy": str(dataset_copy),
        "sha256": sha256_bytes(raw),
        "num_rows": len(rows),
        "loaded_from": "datasets_server_api",
        "offline_mode": False,
    }
    save_json(meta_path, meta)
    return rows, meta


def kda_layers_from_config(config):
    n = int(config["num_hidden_layers"])
    group = int(config["layer_group_size"])
    cutoff = n // group * group
    return [
        i for i in range(n)
        if not (((i + 1) % group == 0) or (i >= cutoff))
    ]


def get_cache_state(cache, layer_idx):
    if cache is None or len(cache.layers) <= layer_idx:
        return None
    return getattr(cache.layers[layer_idx], "keys", None)


def quantize_kda_cache(torch, cache, config, kda_layers, collect_stats=False):
    if config == "FP_STATE":
        return {"applied": False, "touched_layers": [], "finite": True, "bad_layer": None, "bad_count": 0, "stats": {}}
    touched = []
    stats = {}
    for layer_idx in kda_layers:
        state = get_cache_state(cache, layer_idx)
        if state is None:
            continue
        qdq, meta = fake_quant_ling_state(state, config)
        finite = bool(torch.isfinite(qdq).all().item())
        codes = meta["codes"]
        if collect_stats:
            one = small_tensor_stats(torch, state, meta["scale"], codes)
            one["dequant_finite"] = finite
            one["ling_semantic_axis"] = meta["ling_semantic_axis"]
            stats[str(layer_idx)] = one
        if not finite:
            return {
                "applied": True,
                "touched_layers": touched + [layer_idx],
                "finite": False,
                "bad_layer": layer_idx,
                "bad_count": int((~torch.isfinite(qdq)).sum().item()),
                "stats": stats,
            }
        state.copy_(qdq)
        touched.append(layer_idx)
    return {"applied": True, "touched_layers": touched, "finite": True, "bad_layer": None, "bad_count": 0, "stats": stats}


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


def stop_token_ids(tokenizer):
    out = set()
    if tokenizer.eos_token_id is not None:
        out.add(int(tokenizer.eos_token_id))
    for tok in ("<|im_end|>", "<|endoftext|>", "<|end_of_sentence|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            out.add(int(tid))
    return out


def load_model_and_tokenizer(device_map=None):
    configure_offline_runtime()
    import transformers.utils.import_utils as import_utils
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    setup_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    if device_map is None:
        model.cuda()
    model.eval()
    return torch, model, tokenizer


def model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def make_stage0_prompt(tokenizer, target_tokens=96):
    text = "The capital of France is Paris. "
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    while ids.shape[1] < target_tokens:
        text += "The capital of France is Paris. "
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
    return text


def logits_diff(torch, a, b):
    af = a.detach().float()
    bf = b.detach().float()
    abs_diff = float((af - bf).abs().max().item())
    denom = bf.abs().max().clamp_min(EPS)
    rel_diff = float(((af - bf).abs().max() / denom).item())
    return abs_diff, rel_diff


def run_stage0(args):
    import torch
    import triton
    import fla
    import transformers

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config_obj = json.loads((MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = kda_layers_from_config(config_obj)
    rows, dataset_meta = fetch_aime24()
    env_gate = (
        sys.version.split()[0] == "3.11.16"
        and torch.__version__ == "2.7.1+cu128"
        and getattr(triton, "__version__", "") == "3.3.1"
        and getattr(fla, "__version__", "") == "0.5.2"
        and transformers.__version__ == "4.57.6"
    )
    revision_source = revision_source_path()
    revision_text = revision_source.read_text(encoding="utf-8") if revision_source.exists() else ""
    revision_gate = MODEL_PATH.exists() and MODEL_REVISION in revision_text
    quantizer_semantics = {
        "task": TASK,
        "canonical_source_files": [
            "docs/PROTOCOL.md",
            "experiments/orientation/run_end2end_bit_axis_screening.py",
            "experiments/orientation/run_int8_orientation_state_change_mechanism.py",
        ],
        "canonical_formula": {
            "signed_symmetric": True,
            "zero_point": 0,
            "qrange": [-127, 127],
            "scale": "amax(abs(group)).clamp_min(1e-12) / 127",
            "rounding": "torch.round",
            "dequantization": "codes * scale",
        },
        "ling_state_layout": "[B,H,K,V]",
        "INT8_R128": quantizer_axis("INT8_R128"),
        "INT8_C128": quantizer_axis("INT8_C128"),
    }
    save_json(RUN_DIR / "quantizer_semantics.json", quantizer_semantics)

    x = torch.zeros(1, 1, 3, 4, dtype=torch.float32)
    x[0, 0, 0, :] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x[0, 0, 1, :] = torch.tensor([10.0, 20.0, 30.0, 40.0])
    x[0, 0, 2, :] = torch.tensor([100.0, 200.0, 300.0, 400.0])
    _, rm = fake_quant_ling_state(x, "INT8_R128")
    _, cm = fake_quant_ling_state(x, "INT8_C128")
    r_gate = list(rm["scale"].shape) == [1, 1, 3, 1] and rm["amax_dims"] == [-1]
    c_gate = list(cm["scale"].shape) == [1, 1, 1, 4] and cm["amax_dims"] == [-2]

    evaluator_gate = (
        extract_answer("We get \\boxed{204}.") == "204"
        and extract_answer("first \\boxed{1}, final \\boxed{116}") == "116"
        and is_correct("204", "204")
        and not is_correct("205", "204")
    )

    torch2, model, tokenizer = load_model_and_tokenizer(device_map=None)
    device = model_input_device(model)
    prompt = make_stage0_prompt(tokenizer, target_tokens=96)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    short = tokenizer("The capital of France is", return_tensors="pt").to(device)
    gates = {}
    gate_details = {
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": getattr(triton, "__version__", "UNKNOWN"),
            "fla": getattr(fla, "__version__", "UNKNOWN"),
            "transformers": transformers.__version__,
            "cuda_device_count": torch.cuda.device_count(),
            "nvidia_smi": sh(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"]),
        },
        "model_revision_source": str(revision_source),
        "kda_layers": kda_layers,
        "dataset": dataset_meta,
    }
    with torch.inference_mode():
        base = model(**short, use_cache=True)
        fw = model(**short, use_cache=True)
        fp_meta = quantize_kda_cache(torch2, fw.past_key_values, "FP_STATE", kda_layers, collect_stats=True)
        abs_diff, rel_diff = logits_diff(torch2, base.logits, fw.logits)
        first_base = int(base.logits[:, -1, :].argmax(dim=-1).item())
        first_fw = int(fw.logits[:, -1, :].argmax(dim=-1).item())
        gates["FP_IDENTITY_GATE"] = "PASS" if abs_diff == 0.0 and rel_diff == 0.0 and first_base == first_fw and not fp_meta["applied"] else "FAIL"
        gate_details["fp_identity"] = {
            "max_logits_abs_diff": abs_diff,
            "max_logits_rel_diff": rel_diff,
            "baseline_first_token": first_base,
            "framework_first_token": first_fw,
            "fp_quantization_applied": fp_meta["applied"],
        }

        pre = model(**enc, use_cache=True)
        pre_r = quantize_kda_cache(torch2, pre.past_key_values, "INT8_R128", kda_layers, collect_stats=True)
        gates["PREFILL_QUANTIZATION_GATE"] = "PASS" if pre_r["finite"] and sorted(pre_r["touched_layers"]) == kda_layers else "FAIL"
        gate_details["prefill_quantization"] = {
            "prompt_tokens": int(enc["input_ids"].shape[1]),
            "touched_layers": pre_r["touched_layers"],
            "stats_sample": {k: pre_r["stats"][k] for k in sorted(pre_r["stats"])[:2]},
        }

        next_id = pre.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        dec = model(input_ids=next_id.to(device), past_key_values=pre.past_key_values, use_cache=True)
        dec_c = quantize_kda_cache(torch2, dec.past_key_values, "INT8_C128", kda_layers, collect_stats=True)
        gates["DECODE_QUANTIZATION_GATE"] = "PASS" if dec_c["finite"] and sorted(dec_c["touched_layers"]) == kda_layers else "FAIL"
        gate_details["decode_quantization"] = {
            "decode_tokens": 1,
            "touched_layers": dec_c["touched_layers"],
            "stats_sample": {k: dec_c["stats"][k] for k in sorted(dec_c["stats"])[:2]},
        }

        gates["INT8_R128_SANITY_GATE"] = "PASS" if pre_r["finite"] and all(v["scale_finite"] and v["state_finite"] for v in pre_r["stats"].values()) else "FAIL"
        gates["INT8_C128_SANITY_GATE"] = "PASS" if dec_c["finite"] and all(v["scale_finite"] and v["state_finite"] for v in dec_c["stats"].values()) else "FAIL"

    gates.update({
        "ENVIRONMENT_GATE": "PASS" if env_gate else "FAIL",
        "MODEL_REVISION_GATE": "PASS" if revision_gate else "FAIL",
        "R128_SEMANTICS_GATE": "PASS" if r_gate else "FAIL",
        "C128_SEMANTICS_GATE": "PASS" if c_gate else "FAIL",
        "AIME_DATASET_GATE": "PASS" if len(rows) == 30 and all(r.get("problem") and r.get("reference_answer") for r in rows) else "FAIL",
        "EVALUATOR_GATE": "PASS" if evaluator_gate else "FAIL",
    })
    protocol = {
        "task": TASK,
        "time": now(),
        "model_path": str(MODEL_PATH),
        "model_revision": MODEL_REVISION,
        "state_quantization_scope": "PREFILL_AND_DECODE",
        "configs": CONFIG_ORDER,
        "dataset": dataset_meta,
        "seed": SEED,
        "decoding_protocol": {
            "source": "repo canonical AIME prompt/sampling/max_new_tokens settings from run_end2end_bit_axis_screening.py",
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "max_new_tokens": args.max_new_tokens,
        },
        "git": {
            "branch": sh(["git", "branch", "--show-current"]),
            "head": sh(["git", "rev-parse", "HEAD"]),
            "status": sh(["git", "status", "--short"]),
        },
    }
    save_json(RUN_DIR / "protocol.json", protocol)
    out = {
        "task": TASK,
        "time": now(),
        **gates,
        "ALL_STAGE0_GATES": "PASS" if all(v == "PASS" for v in gates.values()) else "FAIL",
        "details": gate_details,
    }
    save_json(RUN_DIR / "stage0_gates.json", out)
    print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
    return out


def generate_one(torch, model, tokenizer, item, config, max_new_tokens):
    setup_seed(SEED)
    device = model_input_device(model)
    prompt = render_prompt(tokenizer, item["problem"])
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    stop_ids = stop_token_ids(tokenizer)
    config_obj = json.loads((MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = kda_layers_from_config(config_obj)
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED + int(item["problem_index"]))
    tokens = []
    touched = set()
    quantize_calls = 0
    prefill_quantized = False
    decode_quantized = 0
    nonfinite = False
    runtime_error = False
    error = None
    tb_tail = []
    first_bad_layer = None
    first_bad_step = None
    wall0 = time.time()
    try:
        with torch.inference_mode():
            out = model(**enc, use_cache=True)
            if not bool(torch.isfinite(out.logits).all().item()):
                nonfinite = True
                first_bad_step = "prefill_logits"
            qmeta = quantize_kda_cache(torch, out.past_key_values, config, kda_layers, collect_stats=False)
            if config != "FP_STATE":
                quantize_calls += 1
                prefill_quantized = qmeta["finite"] and sorted(qmeta["touched_layers"]) == kda_layers
                touched.update(qmeta["touched_layers"])
                if not qmeta["finite"]:
                    nonfinite = True
                    first_bad_layer = qmeta["bad_layer"]
                    first_bad_step = "prefill_state"
            past = out.past_key_values
            if not nonfinite:
                nxt = top_p_sample(torch, out.logits[:, -1, :], 0.6, 0.95, gen)
                tokens.append(int(nxt.item()))
                cur = nxt.to(device)
                if int(cur.item()) not in stop_ids:
                    for step in range(1, max_new_tokens):
                        out = model(input_ids=cur, past_key_values=past, use_cache=True)
                        if not bool(torch.isfinite(out.logits).all().item()):
                            nonfinite = True
                            first_bad_step = step
                            break
                        past = out.past_key_values
                        qmeta = quantize_kda_cache(torch, past, config, kda_layers, collect_stats=False)
                        if config != "FP_STATE":
                            quantize_calls += 1
                            decode_quantized += 1
                            touched.update(qmeta["touched_layers"])
                            if not qmeta["finite"]:
                                nonfinite = True
                                first_bad_layer = qmeta["bad_layer"]
                                first_bad_step = step
                                break
                        nxt = top_p_sample(torch, out.logits[:, -1, :], 0.6, 0.95, gen)
                        tokens.append(int(nxt.item()))
                        cur = nxt.to(device)
                        if int(cur.item()) in stop_ids:
                            break
    except Exception as exc:
        runtime_error = True
        error = repr(exc)
        tb_tail = traceback.format_exc().splitlines()[-12:]
    response = tokenizer.decode(tokens, skip_special_tokens=True) if tokens else ""
    pred = extract_answer(response)
    truncated = bool(len(tokens) >= max_new_tokens and (not tokens or int(tokens[-1]) not in stop_ids))
    return {
        "task": TASK,
        "config": config,
        "problem_id": item["problem_id"],
        "problem_index": item["problem_index"],
        "seed": SEED,
        "sample_seed": SEED + int(item["problem_index"]),
        "max_new_tokens": max_new_tokens,
        "prompt_sha256": sha256_text(prompt),
        "raw_generation": response,
        "parsed_answer": pred,
        "ground_truth_answer": item["reference_answer"],
        "correct": is_correct(pred, item["reference_answer"]),
        "generated_tokens": len(tokens),
        "truncated": truncated,
        "nonfinite": nonfinite,
        "runtime_error": runtime_error,
        "error": error,
        "traceback_tail": tb_tail,
        "first_bad_layer": first_bad_layer,
        "first_bad_step": first_bad_step,
        "wall_time_sec": time.time() - wall0,
        "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "prefill_quantized": prefill_quantized if config != "FP_STATE" else False,
        "decode_quantize_calls": decode_quantized if config != "FP_STATE" else 0,
        "total_quantize_calls": quantize_calls if config != "FP_STATE" else 0,
        "touched_kda_layers": sorted(touched),
        "completed": bool((not runtime_error) and (not nonfinite) and (not truncated)),
    }


def run_generation_stage(args, formal):
    gates_path = RUN_DIR / "stage0_gates.json"
    if not gates_path.exists():
        raise RuntimeError("Stage 0 must run before generation")
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    if gates.get("ALL_STAGE0_GATES") != "PASS":
        raise RuntimeError("Stage 0 gates did not pass")
    rows, _meta = fetch_aime24()
    items = formal_items_for_shard(rows, args.shard_id, args.num_shards) if formal else rows[:args.smoke_problems]
    torch, model, tokenizer = load_model_and_tokenizer(device_map=None)
    max_new = args.max_new_tokens if formal else args.smoke_max_new_tokens
    stage = "formal" if formal else "smoke"
    for config in selected_configs(args.configs):
        path = config_to_file(config, shard_id=args.shard_id, num_shards=args.num_shards) if formal else RUN_DIR / f"smoke_{config.lower()}.jsonl"
        done = set()
        if args.resume and path.exists():
            for r in iter_jsonl(path):
                done.add((r.get("config"), r.get("problem_id"), r.get("seed")))
        for idx, item in enumerate(items):
            key = (config, item["problem_id"], SEED)
            if key in done:
                continue
            shard_note = f" shard={args.shard_id}/{args.num_shards}" if formal and int(args.num_shards) > 1 else ""
            print(f"[{now()}] {stage} {config}{shard_note} {idx + 1}/{len(items)} problem_id={item['problem_id']}", flush=True)
            rec = generate_one(torch, model, tokenizer, item, config, max_new)
            rec["stage"] = stage
            if formal:
                rec["shard_id"] = int(args.shard_id)
                rec["num_shards"] = int(args.num_shards)
            append_jsonl(path, rec)
            if rec["runtime_error"]:
                print(f"[{now()}] runtime_error {config} problem_id={item['problem_id']} {rec['error']}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return analyze(write_report_flag=True, formal_required=formal)


def summarize_config(rows):
    toks = [int(r.get("generated_tokens", 0)) for r in rows]
    return {
        "attempted": len(rows),
        "success": sum(1 for r in rows if r.get("completed")),
        "correct": sum(1 for r in rows if r.get("correct")),
        "accuracy": sum(1 for r in rows if r.get("correct")) / max(len(rows), 1),
        "truncated": sum(1 for r in rows if r.get("truncated")),
        "nonfinite": sum(1 for r in rows if r.get("nonfinite")),
        "runtime_errors": sum(1 for r in rows if r.get("runtime_error")),
        "mean_tokens": sum(toks) / max(len(toks), 1),
        "median_tokens": statistics.median(toks) if toks else None,
        "min_tokens": min(toks) if toks else None,
        "max_tokens": max(toks) if toks else None,
    }


def paired_rows(rows):
    by_pid = defaultdict(dict)
    for r in rows:
        by_pid[r["problem_id"]][r["config"]] = r
    out = []
    for pid in sorted(by_pid, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        m = by_pid[pid]
        if not all(c in m for c in CONFIG_ORDER):
            continue
        out.append({
            "problem_id": pid,
            "FP_correct": bool(m["FP_STATE"]["correct"]),
            "R128_correct": bool(m["INT8_R128"]["correct"]),
            "C128_correct": bool(m["INT8_C128"]["correct"]),
            "FP_answer": m["FP_STATE"].get("parsed_answer"),
            "R128_answer": m["INT8_R128"].get("parsed_answer"),
            "C128_answer": m["INT8_C128"].get("parsed_answer"),
            "FP_tokens": m["FP_STATE"].get("generated_tokens"),
            "R128_tokens": m["INT8_R128"].get("generated_tokens"),
            "C128_tokens": m["INT8_C128"].get("generated_tokens"),
            "FP_truncated": bool(m["FP_STATE"].get("truncated")),
            "R_truncated": bool(m["INT8_R128"].get("truncated")),
            "C_truncated": bool(m["INT8_C128"].get("truncated")),
        })
    return out


def classify(summary, paired):
    if summary["FORMAL_STATUS"] != "COMPLETE" or any(summary.get(g) != "PASS" for g in [
        "R128_SEMANTICS_GATE", "C128_SEMANTICS_GATE", "PREFILL_QUANTIZATION_GATE",
        "DECODE_QUANTIZATION_GATE", "FP_IDENTITY_GATE", "INT8_R128_SANITY_GATE",
        "INT8_C128_SANITY_GATE", "AIME_DATASET_GATE", "EVALUATOR_GATE"]):
        return "LING_KDA_AIME24_PROTOCOL_OR_RUNTIME_INCONCLUSIVE", "INCONCLUSIVE", "NO"
    fp = summary["FP_STATE_CORRECT"]
    r = summary["INT8_R128_CORRECT"]
    c = summary["INT8_C128_CORRECT"]
    if summary["INT8_R128_NONFINITE"] or summary["INT8_C128_NONFINITE"] or summary["INT8_R128_RUNTIME_ERRORS"] or summary["INT8_C128_RUNTIME_ERRORS"]:
        return "LING_KDA_AIME24_PROTOCOL_OR_RUNTIME_INCONCLUSIVE", "INCONCLUSIVE", "NO"
    if r > c and r - c >= 2:
        return "LING_KDA_AIME24_ORIENTATION_ORDER_REVERSED", "R128_BETTER_THAN_C128", "YES"
    if c - r >= 3 and fp - r >= 3:
        return "LING_KDA_AIME24_RC_TRANSFER_STRONG_INITIAL_SIGNAL", "R128_MORE_HARMFUL_THAN_C128", "YES"
    if c > r or summary["R_WRONG_C_CORRECT"] > summary["R_CORRECT_C_WRONG"]:
        return "LING_KDA_AIME24_RC_TRANSFER_WEAK_INITIAL_SIGNAL", "R128_MORE_HARMFUL_THAN_C128_WEAK", "YES_WEAK"
    if fp - min(r, c) >= 3 and abs(r - c) <= 1:
        return "LING_KDA_AIME24_INT8_STATE_SENSITIVE_WITHOUT_CLEAR_ORIENTATION_SPLIT", "INT8_HARMFUL_NO_CLEAR_RC_SPLIT", "NO_CLEAR_RC_SPLIT"
    return "LING_KDA_AIME24_NO_CLEAR_INT8_EFFECT", "NO_CLEAR_R128_VS_C128_HARMFULNESS", "NO"


def analyze(write_report_flag=True, formal_required=True):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    gates = json.loads((RUN_DIR / "stage0_gates.json").read_text(encoding="utf-8")) if (RUN_DIR / "stage0_gates.json").exists() else {}
    protocol = json.loads((RUN_DIR / "protocol.json").read_text(encoding="utf-8")) if (RUN_DIR / "protocol.json").exists() else {}
    rows_by_key = {}
    for config in CONFIG_ORDER:
        for path in config_record_paths(config):
            for r in iter_jsonl(path):
                rows_by_key[(r.get("config"), r.get("problem_id"), r.get("seed"))] = r
    rows = list(rows_by_key.values())
    seen = set(rows_by_key.keys())
    expected = 90
    formal_status = "COMPLETE" if len(seen) == expected else ("RUNNING" if rows else "NOT_RUN")
    by_config = {c: [r for r in rows if r.get("config") == c] for c in CONFIG_ORDER}
    stats = {c: summarize_config(by_config[c]) for c in CONFIG_ORDER}
    pairs = paired_rows(rows)
    save_json(RUN_DIR / "paired_problem_comparison.json", pairs)
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": formal_status if formal_required else ("SMOKE_COMPLETE" if rows else "NOT_RUN"),
        "MODEL_PATH": str(MODEL_PATH),
        "MODEL_REVISION": MODEL_REVISION,
        "DATASET": "HuggingFaceH4/aime_2024 train",
        "N_PROBLEMS": 30,
        "N_SEEDS": 1,
        "SEED": SEED,
        "N_CONFIGS": 3,
        "EXPECTED_GENERATIONS": expected,
        "COMPLETED_GENERATIONS": len(seen),
        "STATE_QUANTIZATION_SCOPE": "PREFILL_AND_DECODE",
        "R128_SEMANTICS_GATE": gates.get("R128_SEMANTICS_GATE", "UNKNOWN"),
        "C128_SEMANTICS_GATE": gates.get("C128_SEMANTICS_GATE", "UNKNOWN"),
        "PREFILL_QUANTIZATION_GATE": gates.get("PREFILL_QUANTIZATION_GATE", "UNKNOWN"),
        "DECODE_QUANTIZATION_GATE": gates.get("DECODE_QUANTIZATION_GATE", "UNKNOWN"),
        "FP_IDENTITY_GATE": gates.get("FP_IDENTITY_GATE", "UNKNOWN"),
        "INT8_R128_SANITY_GATE": gates.get("INT8_R128_SANITY_GATE", "UNKNOWN"),
        "INT8_C128_SANITY_GATE": gates.get("INT8_C128_SANITY_GATE", "UNKNOWN"),
        "AIME_DATASET_GATE": gates.get("AIME_DATASET_GATE", "UNKNOWN"),
        "EVALUATOR_GATE": gates.get("EVALUATOR_GATE", "UNKNOWN"),
        "DECODING_PROTOCOL": "do_sample=True, temperature=0.6, top_p=0.95, repo canonical AIME protocol",
        "MAX_NEW_TOKENS": protocol.get("decoding_protocol", {}).get("max_new_tokens", AIME_MAX_NEW_TOKENS),
        "FP_STATE_SUCCESS": stats["FP_STATE"]["success"],
        "FP_STATE_CORRECT": stats["FP_STATE"]["correct"],
        "FP_STATE_ACCURACY": stats["FP_STATE"]["accuracy"],
        "FP_STATE_TRUNCATED": stats["FP_STATE"]["truncated"],
        "FP_STATE_NONFINITE": stats["FP_STATE"]["nonfinite"],
        "FP_STATE_RUNTIME_ERRORS": stats["FP_STATE"]["runtime_errors"],
        "FP_STATE_MEAN_TOKENS": stats["FP_STATE"]["mean_tokens"],
        "FP_STATE_MEDIAN_TOKENS": stats["FP_STATE"]["median_tokens"],
        "INT8_R128_SUCCESS": stats["INT8_R128"]["success"],
        "INT8_R128_CORRECT": stats["INT8_R128"]["correct"],
        "INT8_R128_ACCURACY": stats["INT8_R128"]["accuracy"],
        "INT8_R128_TRUNCATED": stats["INT8_R128"]["truncated"],
        "INT8_R128_NONFINITE": stats["INT8_R128"]["nonfinite"],
        "INT8_R128_RUNTIME_ERRORS": stats["INT8_R128"]["runtime_errors"],
        "INT8_R128_MEAN_TOKENS": stats["INT8_R128"]["mean_tokens"],
        "INT8_R128_MEDIAN_TOKENS": stats["INT8_R128"]["median_tokens"],
        "INT8_C128_SUCCESS": stats["INT8_C128"]["success"],
        "INT8_C128_CORRECT": stats["INT8_C128"]["correct"],
        "INT8_C128_ACCURACY": stats["INT8_C128"]["accuracy"],
        "INT8_C128_TRUNCATED": stats["INT8_C128"]["truncated"],
        "INT8_C128_NONFINITE": stats["INT8_C128"]["nonfinite"],
        "INT8_C128_RUNTIME_ERRORS": stats["INT8_C128"]["runtime_errors"],
        "INT8_C128_MEAN_TOKENS": stats["INT8_C128"]["mean_tokens"],
        "INT8_C128_MEDIAN_TOKENS": stats["INT8_C128"]["median_tokens"],
        "FP_CORRECT_R_WRONG": sum(1 for p in pairs if p["FP_correct"] and not p["R128_correct"]),
        "FP_CORRECT_C_WRONG": sum(1 for p in pairs if p["FP_correct"] and not p["C128_correct"]),
        "R_WRONG_C_CORRECT": sum(1 for p in pairs if not p["R128_correct"] and p["C128_correct"]),
        "R_CORRECT_C_WRONG": sum(1 for p in pairs if p["R128_correct"] and not p["C128_correct"]),
        "R_C_DISAGREEMENT_COUNT": sum(1 for p in pairs if p["R128_correct"] != p["C128_correct"]),
        "CONFIG_STATS": stats,
        "paired_problem_comparison_path": str(RUN_DIR / "paired_problem_comparison.json"),
        "protocol_path": str(RUN_DIR / "protocol.json"),
        "stage0_gates_path": str(RUN_DIR / "stage0_gates.json"),
        "quantizer_semantics_path": str(RUN_DIR / "quantizer_semantics.json"),
        "records": {c: [str(p) for p in config_record_paths(c)] or [str(config_to_file(c))] for c in CONFIG_ORDER},
        "COMMIT": None,
        "NEXT_RECOMMENDED_TASK": None,
    }
    cls, signal, phenotype = classify(summary, pairs)
    summary["FINAL_SCIENTIFIC_CLASSIFICATION"] = cls
    summary["INITIAL_RC_TRANSFER_SIGNAL"] = signal
    summary["LING_KDA_END_TO_END_INT8_PHENOTYPE"] = phenotype
    summary["NEXT_RECOMMENDED_TASK"] = (
        "LING_KDA_INT8_RC_MINIMAL_TRANSFER_VALIDATION_V1"
        if phenotype in ("YES", "YES_WEAK")
        else "teacher-forced / future-KL transfer diagnostic"
    )
    save_json(RUN_DIR / "final_summary.json", summary)
    save_json(FINAL_SUMMARY_PATH, summary)
    if write_report_flag:
        write_report(summary, pairs)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return summary


def write_report(summary, pairs):
    lines = [
        f"# {TASK}",
        "",
        "## Summary",
        f"- FORMAL_STATUS: `{summary['FORMAL_STATUS']}`",
        f"- FP_STATE: {summary['FP_STATE_CORRECT']}/30 ({summary['FP_STATE_ACCURACY']:.4f})",
        f"- INT8_R128: {summary['INT8_R128_CORRECT']}/30 ({summary['INT8_R128_ACCURACY']:.4f})",
        f"- INT8_C128: {summary['INT8_C128_CORRECT']}/30 ({summary['INT8_C128_ACCURACY']:.4f})",
        f"- FINAL_SCIENTIFIC_CLASSIFICATION: `{summary['FINAL_SCIENTIFIC_CLASSIFICATION']}`",
        f"- INITIAL_RC_TRANSFER_SIGNAL: `{summary['INITIAL_RC_TRANSFER_SIGNAL']}`",
        "",
        "## Gates",
        "",
    ]
    for key in [
        "R128_SEMANTICS_GATE", "C128_SEMANTICS_GATE", "PREFILL_QUANTIZATION_GATE",
        "DECODE_QUANTIZATION_GATE", "FP_IDENTITY_GATE", "INT8_R128_SANITY_GATE",
        "INT8_C128_SANITY_GATE", "AIME_DATASET_GATE", "EVALUATOR_GATE",
    ]:
        lines.append(f"- {key}: `{summary[key]}`")
    lines += [
        "",
        "## Health",
        "",
        "```json",
        json.dumps(summary["CONFIG_STATS"], indent=2, sort_keys=True, ensure_ascii=False),
        "```",
        "",
        "## Paired Problem Comparison",
        "",
        "| problem_id | FP | R128 | C128 | FP ans | R ans | C ans | FP tok | R tok | C tok | FP trunc | R trunc | C trunc |",
        "|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in pairs:
        lines.append(
            f"| {p['problem_id']} | {int(p['FP_correct'])} | {int(p['R128_correct'])} | {int(p['C128_correct'])} | "
            f"{p['FP_answer']} | {p['R128_answer']} | {p['C128_answer']} | {p['FP_tokens']} | {p['R128_tokens']} | {p['C128_tokens']} | "
            f"{int(p['FP_truncated'])} | {int(p['R_truncated'])} | {int(p['C_truncated'])} |"
        )
    lines += [
        "",
        "## Final Summary JSON",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False),
        "```",
    ]
    report_text = "\n".join(lines) + "\n"
    (RUN_DIR / "final_report.md").write_text(report_text, encoding="utf-8")
    FINAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT_PATH.write_text(report_text, encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description=TASK)
    p.add_argument("--stage", choices=["stage0", "smoke", "formal", "analyze", "all"], default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=AIME_MAX_NEW_TOKENS)
    p.add_argument("--smoke-max-new-tokens", type=int, default=SMOKE_MAX_NEW_TOKENS)
    p.add_argument("--smoke-problems", type=int, default=2)
    p.add_argument("--configs", default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    return p.parse_args()


def main():
    configure_offline_runtime()
    args = parse_args()
    if args.stage == "stage0":
        out = run_stage0(args)
        raise SystemExit(0 if out.get("ALL_STAGE0_GATES") == "PASS" else 2)
    if args.stage == "smoke":
        run_generation_stage(args, formal=False)
        raise SystemExit(0)
    if args.stage == "formal":
        run_generation_stage(args, formal=True)
        raise SystemExit(0)
    if args.stage == "analyze":
        analyze(write_report_flag=True, formal_required=True)
        raise SystemExit(0)
    if args.stage == "all":
        out = run_stage0(args)
        if out.get("ALL_STAGE0_GATES") != "PASS":
            raise SystemExit(2)
        run_generation_stage(args, formal=False)
        run_generation_stage(args, formal=True)
        analyze(write_report_flag=True, formal_required=True)


if __name__ == "__main__":
    main()
