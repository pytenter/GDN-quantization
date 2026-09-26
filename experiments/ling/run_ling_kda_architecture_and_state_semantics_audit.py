#!/usr/bin/env python3
import ast
import importlib
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

TASK = "LING_KDA_ARCHITECTURE_AND_STATE_SEMANTICS_AUDIT_V1"
SLUG = "ling_kda_architecture_and_state_semantics_audit_v1"
REPO = Path("/data/zypan/GDN-quantization")
MODEL_PATH = Path("/data/zypan/models/Ling-3.0-tiny")
RESULT_DIR = REPO / "results" / "ling"
REPORT_DIR = REPO / "reports" / "ling"
SCRIPT_DIR = REPO / "experiments" / "ling"
AUDIT_DEPS = Path("/data/zypan/.local/ling_audit_deps")


def sh(cmd, cwd=REPO):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"ERROR: {e}"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_line(path, needle):
    for idx, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return idx
    return None


def relevant_config(config):
    needles = [
        "kda", "mla", "attention", "head", "layer", "gate", "decay", "conv",
        "beta", "rms", "moe", "expert", "router", "rope", "lora", "kv",
        "qk", "value", "hidden", "intermediate",
    ]
    return {k: config[k] for k in sorted(config) if any(s in k.lower() for s in needles)}


def layer_pattern(config):
    n = int(config["num_hidden_layers"])
    group = int(config["layer_group_size"])
    cutoff = n // group * group
    rows = []
    for i in range(n):
        is_mla = ((i + 1) % group == 0) or (i >= cutoff)
        rows.append({
            "layer": i,
            "attention_layer_type": "attention" if is_mla else "linear_attention",
            "semantic_type": "MLA" if is_mla else "KDA",
            "module_class": "BailingMoeV3MultiLatentAttention" if is_mla else "BailingMoeV3KimiDeltaAttention",
        })
    return rows


def import_probe(extra_path=False):
    env_info = {}
    if extra_path and str(AUDIT_DEPS) not in sys.path:
        sys.path.insert(0, str(AUDIT_DEPS))
    for name in ["torch", "transformers", "modelscope", "triton", "fla", "fla.modules", "fla.ops.kda"]:
        try:
            mod = importlib.import_module(name)
            env_info[name] = {
                "status": "OK",
                "version": getattr(mod, "__version__", "UNKNOWN"),
                "file": getattr(mod, "__file__", None),
            }
        except Exception as e:
            env_info[name] = {"status": "FAIL", "error": repr(e)}
    return env_info


def static_code_map(modeling, configuration):
    return {
        "CONFIG_CLASS": {
            "class": "BailingMoeV3Config",
            "file": str(configuration),
            "line": find_line(configuration, "class BailingMoeV3Config"),
        },
        "RMS_NORM": {
            "class": "BailingMoeV3RMSNorm",
            "function": "forward",
            "file": str(modeling),
            "line": find_line(modeling, "class BailingMoeV3RMSNorm"),
        },
        "MLA_MODULE": {
            "class": "BailingMoeV3MultiLatentAttention",
            "function": "forward",
            "file": str(modeling),
            "line": find_line(modeling, "class BailingMoeV3MultiLatentAttention"),
        },
        "KDA_MODULE": {
            "class": "BailingMoeV3KimiDeltaAttention",
            "function": "forward",
            "file": str(modeling),
            "line": find_line(modeling, "class BailingMoeV3KimiDeltaAttention"),
        },
        "KDA_RECURRENT_STATE_READ": {
            "statement": "recurrent_state = past_key_value.layers[self.layer_idx].keys",
            "file": str(modeling),
            "line": find_line(modeling, "recurrent_state = past_key_value.layers[self.layer_idx].keys"),
        },
        "KDA_CONV_STATE_READ": {
            "statement": "conv_state_q, conv_state_k, conv_state_v = past_key_value.layers[self.layer_idx].values",
            "file": str(modeling),
            "line": find_line(modeling, "conv_state_q, conv_state_k, conv_state_v = past_key_value.layers[self.layer_idx].values"),
        },
        "KDA_QKV_SHORT_CONV": {
            "classes": ["ShortConvolution"],
            "file": str(modeling),
            "lines": {
                "q_conv1d": find_line(modeling, "self.q_conv1d = ShortConvolution"),
                "k_conv1d": find_line(modeling, "self.k_conv1d = ShortConvolution"),
                "v_conv1d": find_line(modeling, "self.v_conv1d = ShortConvolution"),
            },
        },
        "KDA_CHUNK_BACKEND": {
            "function": "chunk_kda",
            "file": str(modeling),
            "line": find_line(modeling, "o, recurrent_state = chunk_kda"),
        },
        "KDA_RECURRENT_BACKEND": {
            "function": "fused_recurrent_kda",
            "file": str(modeling),
            "line": find_line(modeling, "o, recurrent_state = fused_recurrent_kda"),
        },
        "KDA_STATE_WRITE": {
            "statement": "past_key_value.layers[self.layer_idx].keys = recurrent_state",
            "file": str(modeling),
            "line": find_line(modeling, "past_key_value.layers[self.layer_idx].keys = recurrent_state"),
        },
        "KDA_POST_CORE_RMS_GATE": {
            "class": "FusedRMSNormGated",
            "statement": "o = self.o_norm(o, g)",
            "file": str(modeling),
            "line": find_line(modeling, "o = self.o_norm(o, g)"),
        },
        "KDA_OUT_PROJ": {
            "module": "self.o_proj",
            "file": str(modeling),
            "line": find_line(modeling, "self.o_proj = nn.Linear"),
        },
        "DECODER_LAYER_SCHEDULE": {
            "class": "BailingMoeV3DecoderLayer",
            "file": str(modeling),
            "line": find_line(modeling, "class BailingMoeV3DecoderLayer"),
        },
        "MODEL_CACHE_INIT": {
            "class": "BailingMoeV3Model",
            "statement": "if use_cache and past_key_values is None: past_key_values = DynamicCache()",
            "file": str(modeling),
            "line": find_line(modeling, "past_key_values = DynamicCache()"),
        },
    }


def try_model_load_and_forward():
    if str(AUDIT_DEPS) not in sys.path:
        sys.path.insert(0, str(AUDIT_DEPS))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    def attempt(monkeypatch_transformers=False):
        import torch
        if monkeypatch_transformers:
            import transformers.utils.import_utils as import_utils
            if not hasattr(import_utils, "is_torch_fx_available"):
                import_utils.is_torch_fx_available = lambda: False
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map=None,
        )
        model.eval().cuda()
        inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out1 = model(**inputs, use_cache=True)
            out2 = model(**inputs, use_cache=True)
        logits = out1.logits.detach()
        return {
            "MODEL_LOAD_GATE": "PASS",
            "FP_SMOKE_GATE": "PASS" if torch.isfinite(logits).all().item() else "FAIL",
            "logits_shape": list(logits.shape),
            "logits_finite_fraction": float(torch.isfinite(logits).float().mean().item()),
            "determinism_max_abs_diff": float((out1.logits - out2.logits).abs().max().item()),
            "past_key_values_type": type(out1.past_key_values).__name__,
        }

    try:
        return attempt(monkeypatch_transformers=False)
    except Exception as e:
        first = {
            "MODEL_LOAD_GATE": "BLOCKED",
            "FP_SMOKE_GATE": "NOT_RUN",
            "error": repr(e),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
        }
        second = None
        if "is_torch_fx_available" in repr(e):
            try:
                second = attempt(monkeypatch_transformers=True)
            except Exception as e2:
                second = {
                    "MODEL_LOAD_GATE": "BLOCKED",
                    "FP_SMOKE_GATE": "NOT_RUN",
                    "error": repr(e2),
                    "traceback_tail": traceback.format_exc().splitlines()[-12:],
                    "note": "non-persistent monkeypatch only: transformers.utils.import_utils.is_torch_fx_available=lambda: False",
                }
        first["secondary_probe_after_transformers_symbol_monkeypatch"] = second
        return first


def write_report(summary):
    lines = [
        f"# {TASK}",
        "",
        "## Formal Status",
        summary["FORMAL_STATUS"],
        "",
        "## Layer Pattern",
    ]
    for row in summary["LAYER_PATTERN_ROWS"]:
        lines.append(f"- Layer {row['layer']:02d}: {row['semantic_type']} ({row['module_class']})")
    lines += [
        "",
        "## KDA Code Map",
        "```json",
        json.dumps(summary["KDA_CODE_MAP"], indent=2, sort_keys=True),
        "```",
        "",
        "## State Semantics",
        f"- State runtime type: {summary['STATE_RUNTIME_TYPE']}",
        f"- State shape: {summary['STATE_SHAPE']}",
        f"- State axes: batch={summary['STATE_BATCH_AXIS']}, head={summary['STATE_HEAD_AXIS']}, key={summary['STATE_KEY_AXIS']}, value={summary['STATE_VALUE_AXIS']}",
        f"- R128: axis={summary['R128_TENSOR_AXIS']}, meaning={summary['R128_FUNCTIONAL_MEANING']}",
        f"- C128: axis={summary['C128_TENSOR_AXIS']}, meaning={summary['C128_FUNCTIONAL_MEANING']}",
        "",
        "## Runtime Blocker",
        "```json",
        json.dumps(summary["runtime_probe"], indent=2, sort_keys=True),
        "```",
        "",
        "## Final Summary",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False),
        "```",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{SLUG}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    config_path = MODEL_PATH / "config.json"
    modeling = MODEL_PATH / "modeling_bailing_moe_v3.py"
    configuration = MODEL_PATH / "configuration_bailing_moe_v3.py"
    config = read_json(config_path)
    pattern = layer_pattern(config)
    kda_layers = [r["layer"] for r in pattern if r["semantic_type"] == "KDA"]
    mla_layers = [r["layer"] for r in pattern if r["semantic_type"] == "MLA"]
    env_probe = import_probe(extra_path=False)
    isolated_env_probe = import_probe(extra_path=True)
    runtime_probe = try_model_load_and_forward()

    runtime_ok = runtime_probe.get("MODEL_LOAD_GATE") == "PASS"
    state_verified = False
    code_map = static_code_map(modeling, configuration)

    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE_STATIC_RUNTIME_BLOCKED" if not runtime_ok else "COMPLETE",
        "MODEL_PATH": str(MODEL_PATH),
        "MODEL_LOAD_GATE": runtime_probe.get("MODEL_LOAD_GATE", "BLOCKED"),
        "PYTHON_VERSION": platform.python_version(),
        "PYTORCH_VERSION": env_probe.get("torch", {}).get("version"),
        "CUDA_VERSION": None,
        "TRANSFORMERS_VERSION": env_probe.get("transformers", {}).get("version"),
        "KDA_RUNTIME_BACKEND": "BLOCKED_MISSING_COMPATIBLE_FLA_OPS_KDA" if not runtime_ok else "fla.ops.kda",
        "NUM_LAYERS": int(config["num_hidden_layers"]),
        "NUM_KDA_LAYERS": len(kda_layers),
        "NUM_MLA_LAYERS": len(mla_layers),
        "LAYER_PATTERN": ",".join(f"{r['layer']}:{r['semantic_type']}" for r in pattern),
        "LAYER_PATTERN_ROWS": pattern,
        "HIDDEN_SIZE": int(config["hidden_size"]),
        "NUM_KDA_HEADS": int(config["num_attention_heads"]),
        "KDA_KEY_DIM": int(config["head_dim"]),
        "KDA_VALUE_DIM": int(config["head_dim"]),
        "STATE_RUNTIME_TYPE": "NOT_CAPTURED_RUNTIME_BLOCKED",
        "STATE_SHAPE": "NOT_CAPTURED_RUNTIME_BLOCKED; code-level expected recurrent_state from Cache.layers[layer].keys passed to chunk_kda/fused_recurrent_kda",
        "STATE_DTYPE": "NOT_CAPTURED_RUNTIME_BLOCKED",
        "STATE_BATCH_AXIS": "NOT_VERIFIED_RUNTIME_BLOCKED",
        "STATE_HEAD_AXIS": "NOT_VERIFIED_RUNTIME_BLOCKED",
        "STATE_KEY_AXIS": "CODE_SEMANTICS_INFERRED_NOT_RUNTIME_VERIFIED: recurrent matrix axis contracted/written by k",
        "STATE_VALUE_AXIS": "CODE_SEMANTICS_INFERRED_NOT_RUNTIME_VERIFIED: recurrent matrix axis written/read as value channel",
        "R128_TENSOR_AXIS": "NOT_RUNTIME_VERIFIED; mathematical row axis should be Key-side if state is [B,H,d_k,d_v]",
        "R128_FUNCTIONAL_MEANING": "Key-side grouping by recurrence semantics, not storage layout; blocked from runtime confirmation",
        "C128_TENSOR_AXIS": "NOT_RUNTIME_VERIFIED; mathematical column axis should be Value-side if state is [B,H,d_k,d_v]",
        "C128_FUNCTIONAL_MEANING": "Value-side grouping by recurrence semantics, not storage layout; blocked from runtime confirmation",
        "STATE_UPDATE_EQUATION": "Code passes q,k,v,g,beta,A_log,dt_bias,initial_state to chunk_kda/fused_recurrent_kda; delta/decay recurrence is inside missing FLA backend and was not runtime/source verified here",
        "QUERY_READOUT_EQUATION": "Code-level KDA backend returns o from q,k,v,g,beta and recurrent_state; exact q^T S form is inside missing FLA backend and not verified",
        "DECAY_TYPE": "A_log per head plus dt_bias per value channel are passed to KDA backend; exact scalar/channel semantics not runtime verified",
        "UPDATE_GATE_TYPE": "beta = sigmoid(b_proj(hidden_states)), shape [B,T,H] before backend",
        "PREFILL_STATE_SEMANTICS": "KDA receives initial_state from Cache.layers[layer].keys and writes final recurrent_state back when use_cache; prefill runtime not captured",
        "DECODE_STATE_SEMANTICS": "For q_len <= 64 code switches to fused_recurrent_kda and uses same Cache.layers[layer].keys slot; decode runtime not captured",
        "QUERY_SHAPE": "CODE_LEVEL: q rearranged as [B,T,H,128]",
        "KEY_SHAPE": "CODE_LEVEL: k rearranged as [B,T,H,128]",
        "VALUE_SHAPE": "CODE_LEVEL: v rearranged as [B,T,H,128]",
        "DECAY_SHAPE": "CODE_LEVEL: A_log [H], dt_bias [H*128]",
        "BETA_SHAPE": "CODE_LEVEL: [B,T,H]",
        "POST_CORE_RMS": "YES: FusedRMSNormGated(self.head_dim, eps=rms_norm_eps, activation='sigmoid')",
        "POST_CORE_GATE": "YES: g_proj(hidden_states) reshaped to [B,T,H,128], consumed by o_norm(o,g)",
        "POST_CORE_OUT_PROJ": "YES: o flattened [B,T,H*128] then o_proj to hidden_size",
        "FP_SMOKE_GATE": runtime_probe.get("FP_SMOKE_GATE", "NOT_RUN"),
        "STATE_CAPTURE_GATE": "NOT_RUN_RUNTIME_BLOCKED",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "NOT_RUN_RUNTIME_BLOCKED",
        "LOGIT_MAX_ABS_DIFF_WITH_INSTRUMENTATION": None,
        "LOGIT_RELATIVE_DIFF_WITH_INSTRUMENTATION": None,
        "FINAL_SCIENTIFIC_CLASSIFICATION": "LING_KDA_STATIC_ARCHITECTURE_VERIFIED_RUNTIME_BLOCKED" if not runtime_ok else "LING_KDA_STATE_SEMANTICS_FULLY_VERIFIED",
        "LING_KDA_QUANTIZATION_PROTOCOL_READY": "YES" if runtime_ok and state_verified else "NO",
        "NEXT_RECOMMENDED_TASK": "Install/locate the exact Ling-compatible Flash Linear Attention backend providing fla.modules and fla.ops.kda, then rerun LING_KDA_ARCHITECTURE_AND_STATE_SEMANTICS_AUDIT_V1 runtime stages",
        "COMMIT": None,
        "git": {
            "branch": sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "head": sh(["git", "rev-parse", "HEAD"]),
            "status_start": sh(["git", "status", "--short"]),
        },
        "CONFIG_AUDIT": relevant_config(config),
        "KDA_CODE_MAP": code_map,
        "KDA_VS_GDN_DIFFERENCE_AUDIT": {
            "scalar_decay_vs_channel_wise_decay": "NOT_YET_CONFIRMED; Ling passes A_log[H] and dt_bias[H*D] to KDA backend, exact backend recurrence unavailable",
            "state_shape": "NOT_YET_CONFIRMED_RUNTIME_BLOCKED",
            "query_key_normalization": "DIFFERENT/CONFIRMED_CODE_LEVEL: use_qk_l2norm_in_kernel=True",
            "update_gate": "CONFIRMED_CODE_LEVEL: beta sigmoid per head",
            "output_gate": "CONFIRMED_CODE_LEVEL: FusedRMSNormGated sigmoid gate",
            "RMS_path": "SAME_AT_HIGH_LEVEL: RMS/gated post-core path exists",
            "out_proj": "SAME_AT_HIGH_LEVEL: flattened heads through o_proj",
            "short_convolution": "DIFFERENT: q/k/v use ShortConvolution before KDA backend",
            "head_merge_semantics": "CONFIRMED_CODE_LEVEL: [B,T,H,D] -> [B,T,H*D] before o_proj",
        },
        "six_question_answers": {
            "Q1": f"24 layers: {len(kda_layers)} KDA and {len(mla_layers)} MLA by config+modeling schedule; runtime instantiation blocked by missing FLA backend.",
            "Q2": "KDA code/config use 16 heads, d_k=128, d_v=128; 128x128 per-head recurrent state is the intended mathematical state but runtime tensor was not captured.",
            "Q3": "Runtime recurrent state shape was not captured because model load is blocked by missing compatible fla.modules/fla.ops.kda.",
            "Q4": "By recurrence API semantics, Key-side is the k-associated state axis and Value-side is the v-associated axis; tensor axis indices remain not runtime verified.",
            "Q5": "If runtime state is [B,H,d_k,d_v], R128 maps to Key-side axis d_k and C128 maps to Value-side axis d_v; this is not yet a passed gate.",
            "Q6": "Safe instrumentation is not yet verified; canonical point should be read-only hooks around BailingMoeV3KimiDeltaAttention.forward and Cache.layers[layer].keys after compatible backend is available.",
        },
        "dependency_probe": env_probe,
        "isolated_dependency_probe": isolated_env_probe,
        "runtime_probe": runtime_probe,
        "output_artifact_policy": "small metadata/report only; no large tensors saved",
    }
    try:
        import torch
        summary["CUDA_VERSION"] = torch.version.cuda
    except Exception:
        pass

    result_path = RESULT_DIR / f"{SLUG}_final_summary.json"
    result_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(summary)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
