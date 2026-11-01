#!/usr/bin/env python3
"""Evaluate FP32 vs INT8 GDN recurrent state fake quantization quality."""
from pathlib import Path

from common import build_arg_parser, require, resolve_cfg, save_json, timestamp_record
from state_fake_quant import quantize_cache_recurrent_states


PROMPTS = [
    "Answer briefly: What is the capital of France?",
    "Answer briefly: Why does the sky look blue during the day?",
    "Solve step by step: If a train travels 60 km in 45 minutes, what is its speed in km/h?",
    "Solve the arithmetic problem: A store has 17 boxes with 24 pencils each. How many pencils are there?",
    "Classify the sentiment as positive, negative, or neutral: The delivery was late, but the product works well.",
    "Rewrite this sentence in a more formal style: I can't make it to the meeting today.",
    "Follow the instruction exactly. Return only the third word in this sentence: alpha beta gamma delta epsilon.",
    "Extract the JSON fields from this text: name is Alice, age is 31, city is Boston.",
    "Translate to Chinese: The experiment should be reproducible and carefully documented.",
    "Summarize in one sentence: Large language models can perform many tasks, but their reliability depends on data, evaluation, and deployment constraints.",
    "Reasoning question: Tom has twice as many books as Ana. Ana has 7 books. They buy 5 more books together. How many books do they have?",
    "Multiple choice. Which gas do plants primarily absorb for photosynthesis? A. Oxygen B. Nitrogen C. Carbon dioxide D. Helium",
    "Write a Python function that returns the factorial of a non-negative integer using a loop.",
    "Explain the difference between latency and throughput in machine learning inference.",
    "Given the sequence 2, 4, 8, 16, what are the next two numbers?",
    "Instruction following: List exactly three benefits of unit tests, each under five words.",
    "Long context test. Context: The red key opens the basement door. The blue key opens the attic door. The green key opens the garden shed. Question: Which key opens the attic door?",
    "Long context test. Context: In the project notes, experiment A used FP32 state, experiment B used INT8 state, and experiment C used INT4 state. Question: Which experiment used INT8 state?",
    "Safety-neutral QA: What are two common reasons software tests can be flaky?",
    "Write a concise email telling a collaborator that the benchmark results are ready for review.",
]

PPL_CORPUS = [
    "Paris is the capital and most populous city of France.",
    "Machine learning inference systems are evaluated by latency, throughput, memory use, and accuracy.",
    "A recurrent state cache stores compressed information from previous tokens and is updated during decoding.",
    "If a train travels sixty kilometers in forty-five minutes, its speed is eighty kilometers per hour.",
    "Unit tests help detect regressions, document behavior, and make refactoring safer.",
    "Rayleigh scattering explains why shorter blue wavelengths are scattered more strongly in the atmosphere.",
    "The experiment compares FP32 recurrent state with INT8 recurrent state while keeping weights unchanged.",
    "A JSON object contains key value pairs and is commonly used for structured data exchange.",
    "Photosynthesis primarily uses carbon dioxide, water, and light to produce glucose and oxygen.",
    "Reliable benchmark reports should record the model path, software versions, prompts, and decoding settings.",
]

TASKS = [
    {"prompt": "Answer briefly: What is the capital of France?", "answer": "Paris"},
    {"prompt": "Answer briefly: 17 times 24 equals what?", "answer": "408"},
    {"prompt": "Answer briefly: Which gas do plants primarily absorb for photosynthesis?", "answer": "carbon dioxide"},
    {"prompt": "Answer briefly: What color is the sky usually on a clear day?", "answer": "blue"},
    {"prompt": "Answer briefly: If Ana has 7 books and Tom has twice as many, how many do they have together?", "answer": "21"},
    {"prompt": "Return only the third word: alpha beta gamma delta", "answer": "gamma"},
    {"prompt": "Answer briefly: What is 60 divided by 0.75?", "answer": "80"},
    {"prompt": "Answer briefly: Is water wet? Answer yes or no.", "answer": "yes"},
    {"prompt": "Answer briefly: What file format commonly stores key value pairs, JSON or JPEG?", "answer": "JSON"},
    {"prompt": "Answer briefly: In ML inference, does lower latency mean faster response?", "answer": "yes"},
]


def parse_prompt_file(path):
    if not path:
        return PROMPTS
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    prompts = [line.strip() for line in lines if line.strip()]
    if len(prompts) < 20:
        raise ValueError("prompt file must contain at least 20 non-empty prompts")
    return prompts


def token_agreement(a, b):
    total = max(len(a), len(b))
    if total == 0:
        return {"same_tokens": 0, "total_tokens": 0, "agreement": 1.0, "length_match": True}
    same = sum(1 for x, y in zip(a, b) if x == y)
    return {
        "same_tokens": same,
        "total_tokens": total,
        "agreement": same / total,
        "length_match": len(a) == len(b),
        "fp32_length": len(a),
        "int8_length": len(b),
    }


def load_model_and_tokenizer(transformers, torch, cfg):
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    return model, tokenizer


def run_greedy(torch, model, tokenizer, prompt, max_new_tokens, bits, layer_types):
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    past = None
    generated = []
    rel_errors = []
    nonfinite_records = 0
    with torch.inference_mode():
        for step in range(max_new_tokens):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            records = quantize_cache_recurrent_states(past, bits=bits, layer_types=layer_types)
            for rec in records:
                val = rec.get("relative_error")
                if val is not None:
                    if val == val and val not in (float("inf"), float("-inf")):
                        rel_errors.append(val)
                    else:
                        nonfinite_records += 1
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
            input_ids = next_token.to(device)
            attention_mask = None
    return {
        "generated_token_ids": generated,
        "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
        "relative_state_error_mean": sum(rel_errors) / len(rel_errors) if rel_errors else None,
        "relative_state_error_max": max(rel_errors) if rel_errors else None,
        "nonfinite_state_records": nonfinite_records,
    }


def aligned_logit_comparison(torch, model, tokenizer, prompt, max_new_tokens, layer_types):
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    fp_input = encoded["input_ids"].to(device)
    int8_input = fp_input.clone()
    fp_mask = encoded.get("attention_mask")
    int8_mask = fp_mask.clone().to(device) if fp_mask is not None else None
    fp_mask = fp_mask.to(device) if fp_mask is not None else None
    fp_past = None
    int8_past = None
    kl_values = []
    top1_same = 0
    generated_path = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            fp_out = model(
                input_ids=fp_input,
                attention_mask=fp_mask,
                past_key_values=fp_past,
                use_cache=True,
            )
            fp_past = fp_out.past_key_values
            int8_out = model(
                input_ids=int8_input,
                attention_mask=int8_mask,
                past_key_values=int8_past,
                use_cache=True,
            )
            int8_past = int8_out.past_key_values
            quantize_cache_recurrent_states(int8_past, bits=8, layer_types=layer_types)

            fp_logits = fp_out.logits[:, -1, :].float()
            int8_logits = int8_out.logits[:, -1, :].float()
            fp_logp = torch.nn.functional.log_softmax(fp_logits, dim=-1)
            int8_logp = torch.nn.functional.log_softmax(int8_logits, dim=-1)
            fp_p = fp_logp.exp()
            kl = torch.sum(fp_p * (fp_logp - int8_logp), dim=-1)
            kl_values.append(float(kl.item()))
            fp_top1 = int(fp_logits.argmax(dim=-1).item())
            int8_top1 = int(int8_logits.argmax(dim=-1).item())
            if fp_top1 == int8_top1:
                top1_same += 1
            generated_path.append(fp_top1)
            next_token = torch.tensor([[fp_top1]], dtype=fp_input.dtype, device=device)
            fp_input = next_token
            int8_input = next_token
            fp_mask = None
            int8_mask = None
    return {
        "kl_mean": sum(kl_values) / len(kl_values) if kl_values else None,
        "kl_max": max(kl_values) if kl_values else None,
        "top1_agreement": top1_same / len(kl_values) if kl_values else None,
        "num_steps": len(kl_values),
        "fp32_greedy_path_token_ids": generated_path,
    }


def run_generation_consistency(torch, model, tokenizer, prompts, max_new_tokens, layer_types):
    prompt_results = []
    for idx, prompt in enumerate(prompts):
        print("generation prompt %d/%d" % (idx + 1, len(prompts)), flush=True)
        fp32 = run_greedy(torch, model, tokenizer, prompt, max_new_tokens, 32, layer_types)
        int8 = run_greedy(torch, model, tokenizer, prompt, max_new_tokens, 8, layer_types)
        agree = token_agreement(fp32["generated_token_ids"], int8["generated_token_ids"])
        prompt_results.append({
            "prompt_id": idx,
            "prompt": prompt,
            "fp32": fp32,
            "int8_state": int8,
            "token_agreement": agree,
            "exact_text_match": fp32["generated_text"] == int8["generated_text"],
        })
    agreements = [r["token_agreement"]["agreement"] for r in prompt_results]
    exact = [r["exact_text_match"] for r in prompt_results]
    return {
        "summary": {
            "num_prompts": len(prompt_results),
            "max_new_tokens": max_new_tokens,
            "mean_token_agreement": sum(agreements) / len(agreements),
            "min_token_agreement": min(agreements),
            "max_token_agreement": max(agreements),
            "exact_text_match_rate": sum(1 for x in exact if x) / len(exact),
        },
        "prompts": prompt_results,
    }


def run_logit_divergence(torch, model, tokenizer, prompts, max_new_tokens, layer_types):
    prompt_results = []
    for idx, prompt in enumerate(prompts):
        print("logit prompt %d/%d" % (idx + 1, len(prompts)), flush=True)
        prompt_results.append({
            "prompt_id": idx,
            "prompt": prompt,
            **aligned_logit_comparison(torch, model, tokenizer, prompt, max_new_tokens, layer_types),
        })
    kls = [r["kl_mean"] for r in prompt_results if r["kl_mean"] is not None]
    top1 = [r["top1_agreement"] for r in prompt_results if r["top1_agreement"] is not None]
    return {
        "summary": {
            "num_prompts": len(prompt_results),
            "max_new_tokens": max_new_tokens,
            "mean_kl": sum(kls) / len(kls) if kls else None,
            "max_prompt_kl_mean": max(kls) if kls else None,
            "mean_top1_agreement": sum(top1) / len(top1) if top1 else None,
        },
        "prompts": prompt_results,
    }


def run_ppl(torch, model, tokenizer, corpus, layer_types, bits):
    device = next(model.parameters()).device
    nll_sum = 0.0
    token_count = 0
    per_text = []
    with torch.inference_mode():
        for idx, text in enumerate(corpus):
            ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
            if ids.shape[1] < 2:
                continue
            past = None
            text_nll = 0.0
            text_tokens = 0
            for pos in range(ids.shape[1] - 1):
                inp = ids[:, pos:pos + 1]
                target = ids[:, pos + 1]
                out = model(input_ids=inp, past_key_values=past, use_cache=True)
                past = out.past_key_values
                quantize_cache_recurrent_states(past, bits=bits, layer_types=layer_types)
                logp = torch.nn.functional.log_softmax(out.logits[:, -1, :].float(), dim=-1)
                nll = -logp.gather(1, target.view(1, 1)).squeeze()
                text_nll += float(nll.item())
                text_tokens += 1
            nll_sum += text_nll
            token_count += text_tokens
            per_text.append({
                "text_id": idx,
                "num_tokens": text_tokens,
                "nll": text_nll,
                "ppl": float(torch.exp(torch.tensor(text_nll / max(text_tokens, 1))).item()),
            })
    return {
        "bits": "FP32" if bits >= 32 else bits,
        "num_texts": len(per_text),
        "num_tokens": token_count,
        "nll": nll_sum,
        "ppl": float(torch.exp(torch.tensor(nll_sum / max(token_count, 1))).item()),
        "texts": per_text,
    }


def run_ppl_comparison(torch, model, tokenizer, layer_types):
    fp32 = run_ppl(torch, model, tokenizer, PPL_CORPUS, layer_types, 32)
    int8 = run_ppl(torch, model, tokenizer, PPL_CORPUS, layer_types, 8)
    return {
        "corpus": "built-in lightweight calibration corpus",
        "weight_quantization": "none",
        "activation_quantization": "none",
        "state_quantization": "FP32 vs INT8 GDN recurrent state only",
        "fp32": fp32,
        "int8_state": int8,
        "delta_ppl": int8["ppl"] - fp32["ppl"],
        "relative_delta_ppl": (int8["ppl"] - fp32["ppl"]) / fp32["ppl"] if fp32["ppl"] else None,
    }


def run_small_tasks(torch, model, tokenizer, max_new_tokens, layer_types):
    results = []
    for idx, task in enumerate(TASKS):
        print("task prompt %d/%d" % (idx + 1, len(TASKS)), flush=True)
        fp32 = run_greedy(torch, model, tokenizer, task["prompt"], max_new_tokens, 32, layer_types)
        int8 = run_greedy(torch, model, tokenizer, task["prompt"], max_new_tokens, 8, layer_types)
        expected = task["answer"].lower()
        fp32_correct = expected in fp32["generated_text"].lower()
        int8_correct = expected in int8["generated_text"].lower()
        results.append({
            "task_id": idx,
            "prompt": task["prompt"],
            "expected_answer": task["answer"],
            "fp32": fp32,
            "int8_state": int8,
            "fp32_contains_answer": fp32_correct,
            "int8_contains_answer": int8_correct,
            "token_agreement": token_agreement(fp32["generated_token_ids"], int8["generated_token_ids"]),
        })
    return {
        "summary": {
            "num_tasks": len(results),
            "max_new_tokens": max_new_tokens,
            "fp32_accuracy_contains": sum(1 for r in results if r["fp32_contains_answer"]) / len(results),
            "int8_accuracy_contains": sum(1 for r in results if r["int8_contains_answer"]) / len(results),
            "mean_token_agreement": sum(r["token_agreement"]["agreement"] for r in results) / len(results),
        },
        "tasks": results,
    }


def main():
    parser = build_arg_parser("Evaluate INT8 GDN recurrent state fake quantization quality")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit-prompts", type=int, default=20)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--run-logits", action="store_true")
    parser.add_argument("--run-ppl", action="store_true")
    parser.add_argument("--run-small-tasks", action="store_true")
    parser.add_argument("--task-max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    cfg = resolve_cfg(args)
    torch = require("torch")
    transformers = require("transformers")
    prompts = parse_prompt_file(args.prompt_file)[:args.limit_prompts]
    model, tokenizer = load_model_and_tokenizer(transformers, torch, cfg)
    text_cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    layer_types = list(getattr(text_cfg, "layer_types", []))

    if not args.skip_generation:
        result = {
            "run": timestamp_record(),
            "model_path": cfg["model_path"],
            "model_class": model.__class__.__name__,
            "state_quantization": "GDN recurrent state fake quantization only",
            "weight_quantization": "none",
            "activation_quantization": "none",
            "decoding": {"type": "greedy", "temperature": 0, "max_new_tokens": args.max_new_tokens},
            "generation_consistency": run_generation_consistency(
                torch, model, tokenizer, prompts, args.max_new_tokens, layer_types
            ),
        }
        save_json(Path(cfg["output_root"]) / "results" / "int8_generation_consistency.json", result)
        print("Wrote results/int8_generation_consistency.json", flush=True)
        summary = result["generation_consistency"]["summary"]
        print("mean_token_agreement=%.6f exact_text_match_rate=%.6f" % (
            summary["mean_token_agreement"], summary["exact_text_match_rate"]
        ))

    if args.run_logits:
        logit_result = {
            "run": timestamp_record(),
            "model_path": cfg["model_path"],
            "model_class": model.__class__.__name__,
            "comparison": "FP32 recurrent state vs INT8 recurrent state on aligned FP32 greedy path",
            "logit_divergence": run_logit_divergence(
                torch, model, tokenizer, prompts, args.max_new_tokens, layer_types
            ),
        }
        save_json(Path(cfg["output_root"]) / "results" / "int8_logit_divergence.json", logit_result)
        print("Wrote results/int8_logit_divergence.json", flush=True)

    if args.run_ppl:
        ppl_result = {
            "run": timestamp_record(),
            "model_path": cfg["model_path"],
            "model_class": model.__class__.__name__,
            "ppl_comparison": run_ppl_comparison(torch, model, tokenizer, layer_types),
        }
        save_json(Path(cfg["output_root"]) / "results" / "int8_ppl_comparison.json", ppl_result)
        print("Wrote results/int8_ppl_comparison.json", flush=True)

    if args.run_small_tasks:
        task_result = {
            "run": timestamp_record(),
            "model_path": cfg["model_path"],
            "model_class": model.__class__.__name__,
            "task_quality": run_small_tasks(
                torch, model, tokenizer, args.task_max_new_tokens, layer_types
            ),
        }
        save_json(Path(cfg["output_root"]) / "results" / "int8_task_quality.json", task_result)
        print("Wrote results/int8_task_quality.json", flush=True)


if __name__ == "__main__":
    main()
