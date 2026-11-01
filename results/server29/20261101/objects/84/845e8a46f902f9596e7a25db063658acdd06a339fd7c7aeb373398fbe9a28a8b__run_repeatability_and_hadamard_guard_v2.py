#!/usr/bin/env python3
"""Non-AIME deterministic repeatability and dynamic Hadamard regression guard."""

import importlib.util
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1")
INTEGRATED_PATH = ROOT / "scripts/run_qwen_c5_c6_formal.py"
ORIGINAL_PATH = Path("/data/zypan/consolidation/GDN-quantization-qwen3090-20260921/experiments/qwen_gdn/aime26/81920/run_qwen_aime26_formal.py")
READINESS_RUNTIME = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_GDN_48GB_FORMAL_TRAINING_READINESS_CLOSURE_V2/scripts/runtime.py")
PROBE_TOKENS = 16
PROBE_SEED = 20260928


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cache_fingerprint(runner, cache):
    return {str(layer): runner.tensor_sha256(runner.get_state(cache, layer).detach().float()) for layer in runner.GDN_LAYERS}


def one_run(runner, model, tokenizer, input_ids_cpu, patch, label):
    runner.setup_seed(PROBE_SEED)
    device = next(model.parameters()).device
    input_ids = input_ids_cpu.to(device)
    attention_mask = torch.ones_like(input_ids)
    generator = torch.Generator(device=device).manual_seed(PROBE_SEED)
    warpers = runner.build_warpers()
    stops = runner.stop_ids(tokenizer)
    quant = {"calls": 0, "layers": set(), "scale_shapes": {}, "code_min": 0, "code_max": 0}
    tokens = []
    survivors = []
    started = time.time()
    with patch as rotation_audit, torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        cache = output.past_key_values
        generated_mask = torch.zeros_like(output.logits[:, -1, :], dtype=torch.bool)
        current, count = runner.sample_next(warpers, output.logits[:, -1, :], generated_mask, generator)
        generated_mask.scatter_(1, current, True)
        tokens.append(int(current.item()))
        survivors.append(count)
        while len(tokens) < PROBE_TOKENS and int(current.item()) not in stops:
            output = model(input_ids=current, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            runner.quantize_c128(cache, quant)
            if not bool(torch.isfinite(output.logits).all()):
                raise FloatingPointError(f"{label} produced nonfinite logits")
            current, count = runner.sample_next(warpers, output.logits[:, -1, :], generated_mask, generator)
            generated_mask.scatter_(1, current, True)
            tokens.append(int(current.item()))
            survivors.append(count)
        finish = "eos" if tokens[-1] in stops else "probe_length"
        rotation_metadata = {
            "calls": rotation_audit.calls,
            "layers": sorted(rotation_audit.layers),
            "phase_counts": dict(rotation_audit.phase_counts),
        }
    return {
        "label": label,
        "AIME_used": False,
        "fixed_probe": "existing TRAIN:first-row token prefix",
        "input_token_count": int(input_ids.shape[-1]),
        "input_ids_sha256": runner.ids_sha256(input_ids_cpu[0].tolist()),
        "seed": PROBE_SEED,
        "generated_token_ids": tokens,
        "generated_token_count": len(tokens),
        "termination": finish,
        "top_k_survivors": survivors,
        "rotation_metadata": rotation_metadata,
        "quantizer_metadata": {
            "calls": quant["calls"], "layers": sorted(quant["layers"]),
            "scale_shapes": quant["scale_shapes"], "code_min": quant["code_min"], "code_max": quant["code_max"],
            "prefill_quantized": False,
        },
        "final_recurrent_state_sha256": cache_fingerprint(runner, cache),
        "runtime_version": runner.runtime_version(),
        "latency_s": time.time() - started,
    }


def repeatability(a, b):
    fields = ["generated_token_ids", "termination", "rotation_metadata", "quantizer_metadata", "final_recurrent_state_sha256", "runtime_version"]
    checks = {field: a[field] == b[field] for field in fields}
    return {"checks": checks, "exact": all(checks.values())}


def write_once(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def main():
    integrated = load_module("qwen_c5c6_integrated_runner", INTEGRATED_PATH)
    original = load_module("qwen_canonical_original_runner", ORIGINAL_PATH)
    readiness = load_module("qwen_readiness_runtime_for_fixed_prompt", READINESS_RUNTIME)
    core = readiness.load_core()
    model, tokenizer = integrated.load_model_and_tokenizer()
    ids, _ = core.tokenize(tokenizer, core.corpus_rows("TRAIN")[0])
    ids = ids[:32]
    input_ids = torch.tensor([ids], dtype=torch.long)
    if core.tensor_sha256(torch.tensor(ids, dtype=torch.long)) != "1ab158a4061d9f8bf3b75dcede826b0f5451bdcf9b58394310c0cf19e68a606d":
        raise RuntimeError("fixed non-AIME input identity mismatch")
    device = next(model.parameters()).device

    report = {"probe_seed": PROBE_SEED, "probe_new_tokens": PROBE_TOKENS, "AIME_used": False, "conditions": {}}
    for method in ("c5", "c6"):
        condition = method.upper()
        base = ROOT / "deployment" / condition
        bank, identity = integrated.load_frozen_rotation(
            base / "frozen_rotation_fp32.pt", base / "deployment_manifest.json", device, method)
        run1 = one_run(integrated, model, tokenizer, input_ids, integrated.QwenFrozenRotationPatch(model, bank, identity), f"{condition}_run1")
        run2 = one_run(integrated, model, tokenizer, input_ids, integrated.QwenFrozenRotationPatch(model, bank, identity), f"{condition}_run2")
        comparison = repeatability(run1, run2)
        dynamic = all([
            run1["rotation_metadata"]["layers"] == list(integrated.GDN_LAYERS),
            run1["rotation_metadata"]["calls"] > 0,
            run1["quantizer_metadata"]["layers"] == list(integrated.GDN_LAYERS),
            run1["quantizer_metadata"]["calls"] == max(run1["generated_token_count"] - 1, 0),
            run1["quantizer_metadata"]["prefill_quantized"] is False,
            identity["direct_frozen_load"] is True,
            identity["theta_reconstruction"] is False,
        ])
        report["conditions"][condition] = {
            "deployment_identity": identity,
            "run1": run1,
            "run2": run2,
            "repeatability": comparison,
            "dynamic_loader_semantics": "PASS" if dynamic else "FAIL",
            "REPEATABILITY_GATE": "PASS" if comparison["exact"] else "FAIL",
        }
        del bank
        torch.cuda.empty_cache()

    legacy_original = one_run(
        integrated, model, tokenizer, input_ids, original.QwenKeyHadamardPatch(model), "legacy_H_original_runner")
    legacy_integrated = one_run(
        integrated, model, tokenizer, input_ids, integrated.QwenKeyHadamardPatch(model), "legacy_H_integrated_runner")
    hadamard_compare = repeatability(legacy_original, legacy_integrated)
    report["hadamard_regression"] = {
        "original": legacy_original,
        "integrated": legacy_integrated,
        "comparison": hadamard_compare,
        "HADAMARD_REGRESSION_GUARD": "PASS" if hadamard_compare["exact"] else "FAIL",
    }
    gates = {
        "FROZEN_LOADER_INTEGRATION_DYNAMIC_GATE": "PASS" if all(x["dynamic_loader_semantics"] == "PASS" for x in report["conditions"].values()) else "FAIL",
        "C5_REPEATABILITY_GATE": report["conditions"]["C5"]["REPEATABILITY_GATE"],
        "C6_REPEATABILITY_GATE": report["conditions"]["C6"]["REPEATABILITY_GATE"],
        "HADAMARD_REGRESSION_GUARD": report["hadamard_regression"]["HADAMARD_REGRESSION_GUARD"],
    }
    report["gates"] = gates
    write_once(ROOT / "analysis/repeatability_and_hadamard_guard.json", report)
    print(json.dumps(gates, indent=2))
    if not all(value == "PASS" for value in gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
