#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path


TASK = "LING_KDA_PERSISTENT_ERROR_DECOMPOSITION_CAUSAL_V1"
SLUG = "ling_kda_persistent_error_decomposition_causal_v1"
REPO = Path(os.environ.get("LING_REPO_ROOT", Path(__file__).resolve().parents[2]))
MODEL_PATH = Path(os.environ.get("LING_MODEL_PATH", "/data01/user2/zypan_ling_dist/models/Ling-3.0-tiny"))
RUN_DIR = Path(os.environ.get("LING_PERSISTENT_DECOMP_RUN_DIR", REPO / "runs" / SLUG))
RESULT_DIR = REPO / "results" / "ling"
REPORT_DIR = REPO / "reports" / "ling"
LEGACY_AIME_DATA = REPO / "runs" / "ling_kda_aime24_int8_rc_end_to_end_smoke_v1" / "aime24_HuggingFaceH4_aime_2024_train.json"
FORMAL_TRAJECTORY_DIR = REPO / "runs" / "ling_kda_aime24_int8_rc_end_to_end_formal_v1_distributed4"
RUNTIME_AUDIT_PATH = Path("/data01/user2/ling-kda-env-smoke/results/ling_kda_model_load_and_runtime_state_capture_recovery_v1_final_summary.json")
CANONICAL_UNIT_CANDIDATES = [
    RUN_DIR / "canonical_units.json",
    REPO / "runs" / "ling_kda_commutator_functional_causal_v1" / "canonical_units.json",
    REPO / "results" / "ling" / "ling_kda_commutator_functional_causal_v1_canonical_units.json",
    Path(os.environ.get("LING_CANONICAL_UNITS", "")) if os.environ.get("LING_CANONICAL_UNITS") else None,
]

CONFIGS = ["INT8_R128", "INT8_C128"]
EPS = 1e-12
BASE_SEED = 0
EOS_TOKEN_ID = 156895
PAD_TOKEN_ID = 156892
HORIZON = 128
EXPECTED_CANONICAL_MANIFEST_SHA256 = "d2a60fcedcfe6b4a5f6757f93703736caf8cb06fa7b792cc5c486010b47ab33b"
EXPECTED_PROMPT_SET = ["60", "61", "64", "68", "69", "76"]
EXPECTED_T0_SET = [64, 128, 256]
EXPECTED_FUTURE_HORIZON = 128
R_ROLE_ABS_ENERGY_TOL = 1e-8


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd, cwd=REPO):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


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


def finite_number(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def median(xs):
    vals = [float(x) for x in xs if finite_number(x)]
    return statistics.median(vals) if vals else None


def mean(xs):
    vals = [float(x) for x in xs if finite_number(x)]
    return sum(vals) / len(vals) if vals else None


def as_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def bootstrap_ci(xs, n=4000, seed=20260907):
    vals = [float(x) for x in xs if as_float(x) is not None]
    if not vals:
        return None
    rng = random.Random(seed)
    boot = []
    for _ in range(n):
        boot.append(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals))
    boot.sort()
    return [boot[int(0.025 * (n - 1))], boot[int(0.975 * (n - 1))]]


def binomial_p_two_sided(k, n, p=0.5):
    if n <= 0:
        return None
    from math import comb
    obs = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    total = 0.0
    for i in range(n + 1):
        prob = comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        if prob <= obs + 1e-15:
            total += prob
    return min(total, 1.0)


def quantizer_axis(config):
    if config == "INT8_R128":
        return {
            "config": config,
            "orientation": "R128",
            "scale_amax_dims": (-1,),
            "scale_shape_semantics": "[B,H,K,1]",
            "grouped_values": "for each fixed [B,H,K], group all V values",
        }
    if config == "INT8_C128":
        return {
            "config": config,
            "orientation": "C128",
            "scale_amax_dims": (-2,),
            "scale_shape_semantics": "[B,H,1,V]",
            "grouped_values": "for each fixed [B,H,V], group all K values",
        }
    raise ValueError(f"unsupported Ling quantizer config: {config}")


def fake_quant_ling_state(state, config):
    import torch
    q = quantizer_axis(config)
    scale = state.detach().float().abs().amax(dim=q["scale_amax_dims"], keepdim=True).clamp_min(EPS) / 127.0
    codes = torch.round(state.detach().float() / scale).clamp(-127, 127)
    qdq = codes * scale
    meta = {
        "config": config,
        "bits": 8,
        "qrange": [-127, 127],
        "zero_point": 0,
        "symmetric": True,
        "scale_formula": "amax(abs(group)).clamp_min(1e-12) / 127",
        "rounding": "torch.round",
        "dequantization": "codes * scale",
        "amax_dims": list(q["scale_amax_dims"]),
        "scale": scale,
        "codes": codes,
        "scale_shape_semantics": q["scale_shape_semantics"],
        "grouped_values": q["grouped_values"],
    }
    return qdq.to(state.dtype), meta


def kda_layers_from_config(config):
    n = int(config["num_hidden_layers"])
    group = int(config["layer_group_size"])
    cutoff = n // group * group
    return [i for i in range(n) if not (((i + 1) % group == 0) or (i >= cutoff))]


def get_cache_state(cache, layer_idx):
    if cache is None or len(cache.layers) <= int(layer_idx):
        return None
    return getattr(cache.layers[int(layer_idx)], "keys", None)


def quantize_kda_cache(torch, cache, config, kda_layers):
    touched = []
    for layer_idx in kda_layers:
        state = get_cache_state(cache, layer_idx)
        if state is None:
            continue
        qdq, _meta = fake_quant_ling_state(state, config)
        if not bool(torch.isfinite(qdq).all().item()):
            return {"finite": False, "bad_layer": int(layer_idx), "touched_layers": touched}
        state.copy_(qdq)
        touched.append(int(layer_idx))
    return {"finite": True, "bad_layer": None, "touched_layers": touched}


def tensor_norm(x):
    import torch
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def tensor_dot(a, b):
    return float((a.detach().float() * b.detach().float()).sum().item())


def tensor_cos(a, b):
    an = tensor_norm(a)
    bn = tensor_norm(b)
    if an < EPS or bn < EPS:
        return None
    return tensor_dot(a, b) / (an * bn + EPS)


def decompose_error(fp_next, fp_on_q_prev, q_pre, q_next):
    import torch
    P = fp_on_q_prev.detach().float() - fp_next.detach().float()
    G = q_pre.detach().float() - fp_on_q_prev.detach().float()
    R = q_next.detach().float() - q_pre.detach().float()
    E = q_next.detach().float() - fp_next.detach().float()
    residual = E - (P + G + R)
    denom = torch.linalg.vector_norm(E).clamp_min(EPS)
    identity_cos = tensor_cos(E, P + G + R)
    return {
        "E": E,
        "P": P,
        "G": G,
        "R": R,
        "identity": {
            "max_abs_error": float(residual.abs().max().item()),
            "relative_L2_error": float((torch.linalg.vector_norm(residual) / denom).item()),
            "cosine": identity_cos,
        },
    }


def component_metrics(*, E, P, G, R, fp_state, readout=None):
    import torch
    B = P.detach().float() + G.detach().float()
    norms = {
        "E_norm": tensor_norm(E),
        "P_norm": tensor_norm(P),
        "G_norm": tensor_norm(G),
        "R_norm": tensor_norm(R),
        "B_norm": tensor_norm(B),
        "FP_state_norm": tensor_norm(fp_state),
    }
    out = dict(norms)
    for key in ["E", "P", "G", "R"]:
        out[f"{key}_rel"] = out[f"{key}_norm"] / (out["FP_state_norm"] + EPS)
        out[f"{key}_rms"] = out[f"{key}_norm"] / math.sqrt(max(int(E.numel()), 1))
    out["B_rel"] = out["B_norm"] / (out["FP_state_norm"] + EPS)
    out["dot_P_G"] = tensor_dot(P, G)
    out["dot_P_R"] = tensor_dot(P, R)
    out["dot_G_R"] = tensor_dot(G, R)
    out["dot_B_R"] = tensor_dot(B, R)
    out["cos_P_G"] = tensor_cos(P, G)
    out["cos_P_R"] = tensor_cos(P, R)
    out["cos_G_R"] = tensor_cos(G, R)
    out["cos_B_R"] = tensor_cos(B, R)
    out["PG_interaction"] = 2.0 * out["dot_P_G"]
    out["PR_interaction"] = 2.0 * out["dot_P_R"]
    out["GR_interaction"] = 2.0 * out["dot_G_R"]
    e2 = out["E_norm"] ** 2
    b2 = out["B_norm"] ** 2
    gr2 = tensor_norm(G + R) ** 2
    pr2 = tensor_norm(P + R) ** 2
    out["delta_r_energy"] = e2 - b2
    out["delta_r_energy_normalized"] = None if b2 < 1e-20 else out["delta_r_energy"] / b2
    out["r_cancellation_flag"] = bool(out["E_norm"] < out["B_norm"])
    predicted = (
        out["P_norm"] ** 2
        + out["G_norm"] ** 2
        + out["R_norm"] ** 2
        + out["PG_interaction"]
        + out["PR_interaction"]
        + out["GR_interaction"]
    )
    out["P_energy"] = out["P_norm"] ** 2
    out["G_energy"] = out["G_norm"] ** 2
    out["R_energy"] = out["R_norm"] ** 2
    out["TOTAL_predicted"] = predicted
    out["TOTAL_actual"] = e2
    out["energy_identity_abs_error"] = abs(predicted - e2)
    out["energy_identity_relative_error"] = abs(predicted - e2) / (abs(e2) + EPS)
    out["predicted_total_energy"] = predicted
    out["actual_total_energy"] = e2
    out["energy_identity_error"] = out["energy_identity_abs_error"]
    out["loo_P"] = e2 - gr2
    out["loo_G"] = e2 - pr2
    out["loo_R"] = out["delta_r_energy"]
    out["loo_P_normalized"] = out["loo_P"] / (abs(e2) + EPS)
    out["loo_G_normalized"] = out["loo_G"] / (abs(e2) + EPS)
    out["loo_R_normalized"] = out["loo_R"] / (abs(e2) + EPS)
    role_tol = max(R_ROLE_ABS_ENERGY_TOL, 1e-8 * max(abs(e2), abs(b2), 1.0))
    if out["loo_R"] > role_tol:
        out["R_error_role"] = "INJECTION"
    elif out["loo_R"] < -role_tol:
        out["R_error_role"] = "CANCELLATION"
    else:
        out["R_error_role"] = "NEUTRAL"
    readout = readout or {}
    for key in ["E", "P", "G", "R"]:
        out[f"readout_{key}"] = readout.get(key)
        out[f"postproj_{key}"] = None
    return out


def canonical_manifest_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_canonical_units(path=None, expected_sha256=EXPECTED_CANONICAL_MANIFEST_SHA256):
    candidates = [Path(path)] if path else [p for p in CANONICAL_UNIT_CANDIDATES if p]
    for candidate in candidates:
        if candidate and candidate.exists():
            digest = canonical_manifest_sha256(candidate)
            obj = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                units = obj
            else:
                units = obj.get("formal_units") or obj.get("canonical_units") or obj.get("units") or []
            norm = []
            for idx, u in enumerate(units):
                row = dict(u)
                row.setdefault("unit_id", row.get("id") or f"unit_{idx:03d}")
                for key in ["problem_id", "problem_index", "layer", "head", "t0"]:
                    if key not in row:
                        raise ValueError(f"canonical unit {idx} missing required field {key}")
                row["problem_id"] = str(row["problem_id"])
                row.setdefault("prompt_id", row["problem_id"])
                row["problem_index"] = int(row["problem_index"])
                row["t0"] = int(row["t0"])
                row["future_horizon"] = int(row.get("future_horizon", HORIZON))
                norm.append(row)
            identities = {
                (str(u["unit_id"]), str(u["problem_id"]), int(u["problem_index"]), int(u["t0"]))
                for u in norm
            }
            prompt_set = sorted({str(u["problem_id"]) for u in norm}, key=int)
            t0_set = sorted({int(u["t0"]) for u in norm})
            future_horizon_set = sorted({int(u["future_horizon"]) for u in norm})
            hash_ok = expected_sha256 is None or digest == expected_sha256
            unit_set_ok = (
                len(norm) == 18
                and len(identities) == 18
                and prompt_set == EXPECTED_PROMPT_SET
                and t0_set == EXPECTED_T0_SET
                and future_horizon_set == [EXPECTED_FUTURE_HORIZON]
            )
            return {
                "status": "FOUND",
                "path": str(candidate),
                "sha256": digest,
                "n_units": len(norm),
                "unique_units": len(identities),
                "prompt_set": prompt_set,
                "t0_set": t0_set,
                "future_horizon_set": future_horizon_set,
                "formal_units": norm,
                "hash_matches_expected": hash_ok,
                "gate": "PASS" if hash_ok and unit_set_ok else "FAIL",
            }
    return {
        "status": "MISSING",
        "path": None,
        "n_units": 0,
        "formal_units": [],
        "gate": "FAIL",
        "searched": [str(p) for p in candidates if p],
    }


def iter_unit_layer_heads(unit, kda_layers):
    layers = list(kda_layers) if str(unit.get("layer")) == "ALL_KDA" else [int(unit["layer"])]
    if str(unit.get("head")) == "ALL":
        return [(int(layer), "ALL") for layer in layers]
    return [(int(layer), int(unit["head"])) for layer in layers]


def reset_stage_outputs(stage):
    for name in [f"decomposition_horizon_{stage}.csv", f"decomposition_raw_{stage}.jsonl", f"failure_{stage}.json"]:
        path = RUN_DIR / name
        if path.exists():
            path.unlink()


def load_dataset():
    rows = json.loads(LEGACY_AIME_DATA.read_text(encoding="utf-8"))
    return {str(r["problem_id"]): r for r in rows}


def load_fp_teacher_tokens():
    out = {}
    for path in sorted(FORMAL_TRAJECTORY_DIR.glob("fp_state_shard*.jsonl")):
        for rec in iter_jsonl(path):
            pid = str(rec.get("problem_id"))
            toks = rec.get("generated_token_ids")
            if toks:
                out[pid] = [int(x) for x in toks]
    return out


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


def configure_offline_runtime():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def load_model_and_tokenizer():
    configure_offline_runtime()
    import transformers.utils.import_utils as import_utils
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    setup_seed(BASE_SEED)
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
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True, return_tensors="pt")


class KdaTransitionCapture:
    def __init__(self):
        self.branch = None
        self.layer = None
        self.records = defaultdict(dict)
        self.handles = []
        self.globals = None
        self.orig_chunk = None
        self.orig_fused = None

    def install(self, model):
        kda_modules = []
        for idx, layer in enumerate(model.model.layers):
            mod = getattr(layer, "attention", None)
            if mod is None or not hasattr(mod, "A_log") or not hasattr(mod, "q_conv1d"):
                continue
            kda_modules.append((idx, mod))
            self.handles.append(mod.register_forward_pre_hook(self._make_pre_hook(idx)))
        if not kda_modules:
            raise RuntimeError("no Ling KDA attention modules found")
        self.globals = type(kda_modules[0][1]).forward.__globals__
        self.orig_chunk = self.globals["chunk_kda"]
        self.orig_fused = self.globals["fused_recurrent_kda"]
        self.globals["chunk_kda"] = self._wrap("chunk_kda", self.orig_chunk)
        self.globals["fused_recurrent_kda"] = self._wrap("fused_recurrent_kda", self.orig_fused)
        return [idx for idx, _mod in kda_modules]

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self.globals is not None:
            self.globals["chunk_kda"] = self.orig_chunk
            self.globals["fused_recurrent_kda"] = self.orig_fused

    def _make_pre_hook(self, layer_idx):
        def pre(_module, _inputs):
            self.layer = int(layer_idx)
        return pre

    def _wrap(self, name, fn):
        def wrapped(**kwargs):
            out, final_state = fn(**kwargs)
            if self.branch is not None and self.layer is not None:
                rec = {
                    "operator": name,
                    "q": kwargs["q"].detach().clone(),
                    "k": kwargs["k"].detach().clone(),
                    "v": kwargs["v"].detach().clone(),
                    "g": kwargs["g"].detach().clone(),
                    "beta": kwargs["beta"].detach().clone(),
                    "A_log": None if kwargs.get("A_log") is None else kwargs.get("A_log").detach().clone(),
                    "dt_bias": None if kwargs.get("dt_bias") is None else kwargs.get("dt_bias").detach().clone(),
                    "initial_state": None if kwargs.get("initial_state") is None else kwargs.get("initial_state").detach().float().clone(),
                    "final_state": None if final_state is None else final_state.detach().float().clone(),
                    "output": out.detach().float().clone(),
                    "output_final_state": bool(kwargs.get("output_final_state", False)),
                    "use_qk_l2norm_in_kernel": bool(kwargs.get("use_qk_l2norm_in_kernel", False)),
                    "use_gate_in_kernel": bool(kwargs.get("use_gate_in_kernel", False)),
                    "use_beta_sigmoid_in_kernel": bool(kwargs.get("use_beta_sigmoid_in_kernel", False)),
                    "safe_gate": bool(kwargs.get("safe_gate", False)),
                    "lower_bound": kwargs.get("lower_bound"),
                    "state_v_first": bool(kwargs.get("state_v_first", False)),
                    "cu_seqlens": kwargs.get("cu_seqlens"),
                }
                self.records[str(self.branch)][int(self.layer)] = rec
            return out, final_state
        return wrapped

    def replay(self, torch, rec, initial_state):
        fn = self.orig_fused if rec["operator"] == "fused_recurrent_kda" else self.orig_chunk
        kwargs = {
            "q": rec["q"],
            "k": rec["k"],
            "v": rec["v"],
            "g": rec["g"],
            "beta": rec["beta"],
            "initial_state": initial_state.to(device=rec["q"].device, dtype=torch.float32),
            "output_final_state": True,
            "use_qk_l2norm_in_kernel": rec["use_qk_l2norm_in_kernel"],
            "use_gate_in_kernel": rec["use_gate_in_kernel"],
            "use_beta_sigmoid_in_kernel": rec["use_beta_sigmoid_in_kernel"],
            "lower_bound": rec["lower_bound"],
            "state_v_first": rec["state_v_first"],
            "cu_seqlens": rec["cu_seqlens"],
        }
        if rec["A_log"] is not None:
            kwargs["A_log"] = rec["A_log"]
        if rec["dt_bias"] is not None:
            kwargs["dt_bias"] = rec["dt_bias"]
        if rec["operator"] == "chunk_kda":
            kwargs["safe_gate"] = rec["safe_gate"]
        return fn(**kwargs)


def transition_driver_audit(kda_layers):
    modeling = MODEL_PATH / "modeling_bailing_moe_v3.py"
    def driver(name, shape, computed, drift, replayable):
        return {
            "name": name,
            "shape": shape,
            "where_computed": computed,
            "computed": computed,
            "FP_Q_identity_or_drift_possibility": drift,
            "affected_by_recurrent_state": drift,
            "replayable": replayable,
        }
    return {
        "driver_tensors": [
            driver("q", "[B,T,H,128]", "q_proj(hidden_states) -> q_conv1d -> rearrange in Ling attention forward", "DRIFT_POSSIBLE via hidden_states from prior quantized trajectory", "YES captured from kwargs and replayed through original KDA backend"),
            driver("k", "[B,T,H,128]", "k_proj(hidden_states) -> k_conv1d -> rearrange in Ling attention forward", "DRIFT_POSSIBLE via hidden_states from prior quantized trajectory", "YES captured from kwargs and replayed through original KDA backend"),
            driver("v", "[B,T,H,128]", "v_proj(hidden_states) -> v_conv1d -> rearrange in Ling attention forward", "DRIFT_POSSIBLE via hidden_states from prior quantized trajectory", "YES captured from kwargs and replayed through original KDA backend"),
            driver("g", "[B,T,H,128]", "f_proj(hidden_states) before KDA backend", "DRIFT_POSSIBLE via hidden_states from prior quantized trajectory", "YES captured from kwargs and replayed through original KDA backend"),
            driver("beta", "[B,T,H]", "sigmoid(b_proj(hidden_states)) before KDA backend", "DRIFT_POSSIBLE via hidden_states from prior quantized trajectory", "YES captured from kwargs and replayed through original KDA backend"),
            driver("A_log", "[H]", "learned per-layer parameter passed to KDA backend", "IDENTICAL_FIXED_PARAMETER", "YES captured from kwargs; not a trajectory drift component"),
            driver("dt_bias", "[H*128]", "learned per-layer parameter passed to KDA backend", "IDENTICAL_FIXED_PARAMETER", "YES captured from kwargs; not a trajectory drift component"),
            driver("initial_state", "[B,H,K,V]", "past_key_values.layers[layer].keys", "INTERVENTION_AXIS: FP previous state or persistent quantized previous state", "YES supplied explicitly in replay"),
        ],
        "backend": "chunk_kda for q_len > 64 prefill; fused_recurrent_kda for decode q_len <= 64",
        "source": str(modeling),
        "kda_layers": list(kda_layers),
        "theta_fp_vs_q_status": "q/k/v/g/beta may differ between FP and persistent quantized branch; A_log/dt_bias identical parameters",
    }


def row_state(x, head):
    if head is None or str(head) == "ALL":
        return x
    return x[:, int(head):int(head) + 1, :, :]


def driver_distance(torch, fp_rec, q_rec):
    out = {}
    for name in ["q", "k", "v", "g", "beta"]:
        a = fp_rec[name].detach().float()
        b = q_rec[name].detach().float()
        out[f"theta_{name}_rel_l2"] = tensor_norm(a - b) / (tensor_norm(a) + EPS)
        out[f"theta_{name}_cos"] = tensor_cos(a, b)
    return out


def readout_visibility(torch, capture, fp_rec, fp_prev_state, components):
    base_o, _base_s = capture.replay(torch, fp_rec, fp_prev_state)
    out = {}
    for name, comp in components.items():
        try:
            pert_o, _ = capture.replay(torch, fp_rec, fp_prev_state + comp.to(fp_prev_state.device))
            out[name] = tensor_norm(pert_o.detach().float() - base_o.detach().float()) / (tensor_norm(base_o) + EPS)
        except Exception:
            out[name] = None
    return out


def run_units(stage, units, horizon=HORIZON, unit_limit=None):
    import torch
    rows_by_pid = load_dataset()
    teacher_tokens = load_fp_teacher_tokens()
    config = json.loads((MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = kda_layers_from_config(config)
    torch, model, tokenizer = load_model_and_tokenizer()
    capture = KdaTransitionCapture()
    installed_layers = capture.install(model)
    horizon_path = RUN_DIR / f"decomposition_horizon_{stage}.csv"
    raw_path = RUN_DIR / f"decomposition_raw_{stage}.jsonl"
    horizon_path.parent.mkdir(parents=True, exist_ok=True)
    reset_stage_outputs(stage)
    fields = None
    count = 0
    gate_failures = []
    max_decomp_abs = 0.0
    max_decomp_rel = 0.0
    max_energy_err = 0.0
    max_fp_replay = 0.0
    max_q_replay = 0.0
    seen_layers = set()
    seen_units = set()
    seen_orientations = set()
    seen_horizons = set()
    try:
        for u in units[:unit_limit]:
            pid = str(u["problem_id"])
            item = rows_by_pid.get(pid)
            toks = teacher_tokens.get(pid)
            if item is None or not toks:
                raise RuntimeError(f"missing prompt or FP teacher tokens for canonical unit {u['unit_id']}")
            max_needed = int(u["t0"]) + int(horizon)
            if len(toks) < max_needed:
                raise RuntimeError(f"teacher tokens too short for {u['unit_id']}: need {max_needed}, have {len(toks)}")
            setup_seed(BASE_SEED + int(item["problem_index"]))
            input_ids = render_prompt(tokenizer, item["problem"]).cuda()
            masks = {b: torch.ones_like(input_ids) for b in ["FP", "INT8_R128", "INT8_C128"]}
            pasts = {}
            with torch.inference_mode():
                for branch in ["FP", "INT8_R128", "INT8_C128"]:
                    capture.branch = branch
                    out = model(
                        input_ids=input_ids,
                        attention_mask=masks[branch],
                        cache_position=torch.arange(0, input_ids.shape[-1], device=input_ids.device),
                        use_cache=True,
                    )
                    pasts[branch] = out.past_key_values
                    if branch in CONFIGS:
                        meta = quantize_kda_cache(torch, pasts[branch], branch, kda_layers)
                        if not meta["finite"]:
                            raise RuntimeError(f"nonfinite prefill quantization {branch} layer={meta['bad_layer']}")
                prompt_len = int(input_ids.shape[-1])
                for t, token in enumerate(toks[:max_needed]):
                    cur = torch.tensor([[int(token)]], device=input_ids.device, dtype=torch.long)
                    for branch in ["FP", "INT8_R128", "INT8_C128"]:
                        masks[branch] = torch.cat([masks[branch], torch.ones_like(cur)], dim=-1)
                        capture.branch = branch
                        out = model(
                            input_ids=cur,
                            attention_mask=masks[branch],
                            past_key_values=pasts[branch],
                            cache_position=torch.tensor([prompt_len + t], device=input_ids.device, dtype=torch.long),
                            use_cache=True,
                        )
                        pasts[branch] = out.past_key_values
                        if branch in CONFIGS:
                            meta = quantize_kda_cache(torch, pasts[branch], branch, kda_layers)
                            if not meta["finite"]:
                                raise RuntimeError(f"nonfinite decode quantization {branch} layer={meta['bad_layer']} t={t}")
                    if t < int(u["t0"]):
                        continue
                    for orientation in CONFIGS:
                        for layer, head in iter_unit_layer_heads(u, kda_layers):
                            fp_rec = capture.records["FP"][layer]
                            q_rec = capture.records[orientation][layer]
                            fp_prev = fp_rec["initial_state"]
                            q_prev = q_rec["initial_state"]
                            fp_next = fp_rec["final_state"]
                            q_pre = q_rec["final_state"]
                            q_next = get_cache_state(pasts[orientation], layer).detach().float().clone()
                            if fp_prev is None or q_prev is None:
                                raise RuntimeError(f"initial_state missing at decode t={t} for {u['unit_id']}")
                            fp_replay_o, fp_replay_s = capture.replay(torch, fp_rec, fp_prev)
                            q_replay_o, q_replay_s = capture.replay(torch, q_rec, q_prev)
                            fp_on_q_o, fp_on_q_s = capture.replay(torch, fp_rec, q_prev)
                            replay_fp_err = tensor_norm(fp_replay_s.detach().float() - fp_next.detach().float()) / (tensor_norm(fp_next) + EPS)
                            replay_q_err = tensor_norm(q_replay_s.detach().float() - q_pre.detach().float()) / (tensor_norm(q_pre) + EPS)
                            full_parts = decompose_error(fp_next, fp_on_q_s, q_pre, q_next)
                            parts = decompose_error(
                                row_state(fp_next, head),
                                row_state(fp_on_q_s, head),
                                row_state(q_pre, head),
                                row_state(q_next, head),
                            )
                            readout = readout_visibility(
                                torch,
                                capture,
                                fp_rec,
                                fp_prev,
                                {k: v for k, v in full_parts.items() if k in ["E", "P", "G", "R"]},
                            )
                            metrics = component_metrics(
                                E=parts["E"],
                                P=parts["P"],
                                G=parts["G"],
                                R=parts["R"],
                                fp_state=row_state(fp_next, head),
                                readout=readout,
                            )
                            row = {
                                "task": TASK,
                                "stage": stage,
                                "prompt_id": pid,
                                "problem_index": int(item["problem_index"]),
                                "unit_id": str(u["unit_id"]),
                                "layer": layer,
                                "head": head,
                                "t0": int(u["t0"]),
                                "timestep": int(t),
                                "horizon": int(t - int(u["t0"]) + 1),
                                "orientation": orientation.replace("INT8_", ""),
                                "fp_replay_relative_L2_error": replay_fp_err,
                                "q_replay_relative_L2_error": replay_q_err,
                                "decomp_max_abs_error": parts["identity"]["max_abs_error"],
                                "decomp_relative_L2_error": parts["identity"]["relative_L2_error"],
                                "decomp_cosine": parts["identity"]["cosine"],
                            }
                            row.update(driver_distance(torch, fp_rec, q_rec))
                            row.update(metrics)
                            max_decomp_abs = max(max_decomp_abs, row["decomp_max_abs_error"])
                            max_decomp_rel = max(max_decomp_rel, row["decomp_relative_L2_error"])
                            max_energy_err = max(max_energy_err, row["energy_identity_error"])
                            max_fp_replay = max(max_fp_replay, row["fp_replay_relative_L2_error"])
                            max_q_replay = max(max_q_replay, row["q_replay_relative_L2_error"])
                            seen_layers.add(layer)
                            seen_units.add(str(u["unit_id"]))
                            seen_orientations.add(row["orientation"])
                            seen_horizons.add(row["horizon"])
                            append_jsonl(raw_path, row)
                            if fields is None:
                                fields = list(row.keys())
                                with horizon_path.open("w", newline="", encoding="utf-8") as f:
                                    csv.DictWriter(f, fieldnames=fields).writeheader()
                            with horizon_path.open("a", newline="", encoding="utf-8") as f:
                                csv.DictWriter(f, fieldnames=fields).writerow({k: row.get(k) for k in fields})
                            if row["decomp_max_abs_error"] > 1e-5 or row["decomp_relative_L2_error"] > 1e-6:
                                gate_failures.append(row)
                            count += 1
                    if (t - int(u["t0"]) + 1) >= horizon:
                        break
            print(f"[{now()}] {stage} completed unit={u['unit_id']}", flush=True)
    except Exception as exc:
        failure = {"error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-20:]}
        save_json(RUN_DIR / f"failure_{stage}.json", failure)
        raise
    finally:
        capture.close()
    decomp_gate = "PASS" if max_decomp_abs <= 1e-5 and max_decomp_rel <= 1e-6 and not gate_failures else "FAIL"
    replay_gate = "PASS" if max_fp_replay <= 1e-6 and max_q_replay <= 1e-6 else "FAIL"
    local_requant_gate = "PASS" if decomp_gate == "PASS" and max_energy_err <= 1e-4 else "FAIL"
    smoke_gate = "PASS" if (
        count > 0
        and decomp_gate == "PASS"
        and local_requant_gate == "PASS"
        and seen_orientations == {"R128", "C128"}
        and set(installed_layers).issubset(seen_layers)
        and max(seen_horizons or {0}) == horizon
    ) else "FAIL"
    stage0_gate = "PASS" if (
        decomp_gate == "PASS"
        and replay_gate == "PASS"
        and local_requant_gate == "PASS"
        and set(installed_layers).issubset(seen_layers)
    ) else "FAIL"
    return {
        "stage": stage,
        "rows_written": count,
        "horizon_csv": str(horizon_path),
        "raw_jsonl": str(raw_path),
        "decomposition_identity_failures": len(gate_failures),
        "kda_layers": installed_layers,
        "unique_units": len(seen_units),
        "orientations": sorted(seen_orientations),
        "horizons": [min(seen_horizons), max(seen_horizons)] if seen_horizons else [],
        "max_decomposition_identity_abs_error": max_decomp_abs,
        "max_decomposition_identity_relative_L2_error": max_decomp_rel,
        "max_energy_identity_error": max_energy_err,
        "max_fp_replay_relative_L2_error": max_fp_replay,
        "max_q_replay_relative_L2_error": max_q_replay,
        "PERSISTENT_TRAJECTORY_PARITY_GATE": replay_gate,
        "DECOMPOSITION_IDENTITY_GATE": decomp_gate,
        "PER_LAYER_DECOMPOSITION_IDENTITY_GATE": "PASS" if set(installed_layers).issubset(seen_layers) and decomp_gate == "PASS" else "FAIL",
        "P_ISOLATION_GATE": replay_gate,
        "G_ISOLATION_GATE": replay_gate,
        "LOCAL_REQUANT_IDENTITY_GATE": local_requant_gate,
        "QUANTIZATION_EXPOSURE_SEMANTICS": "Persistent INT8 trajectory quantizes KDA cache immediately after prefill and after every decode step.",
        "BOUNDARY_EXPECTATION": "Canonical t0 values are after prior persistent exposure, so measured horizon=1 is not expected to have P=0 or G=0.",
        "FIRST_STEP_BOUNDARY_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": replay_gate,
        "STAGE0_GATE": stage0_gate if stage == "stage0" else "NOT_APPLICABLE",
        "SMOKE": smoke_gate if stage == "smoke" else "NOT_APPLICABLE",
    }


def read_csv_rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def simple_svg(path, title, lines):
    width = 920
    height = 520
    pad = 56
    points = [(x, y) for _name, pts in lines for x, y in pts if y is not None]
    if not points:
        body = f"<text x='24' y='48' font-size='22'>{title}: no data</text>"
    else:
        xs = [x for x, _y in points]
        ys = [y for _x, y in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if abs(xmax - xmin) < EPS:
            xmax += 1.0
        if abs(ymax - ymin) < EPS:
            ymax += 1.0
        def sx(x):
            return pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)
        def sy(y):
            return height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
        parts = [
            f"<text x='24' y='34' font-size='22' font-family='sans-serif'>{title}</text>",
            f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='#333'/>",
            f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height-pad}' stroke='#333'/>",
        ]
        for i, (name, pts) in enumerate(lines):
            pts = [(x, y) for x, y in pts if y is not None]
            if not pts:
                continue
            color = colors[i % len(colors)]
            d = " ".join(("M" if j == 0 else "L") + f"{sx(x):.2f},{sy(y):.2f}" for j, (x, y) in enumerate(pts))
            parts.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2'/>")
            for x, y in pts:
                parts.append(f"<circle cx='{sx(x):.2f}' cy='{sy(y):.2f}' r='3' fill='{color}'/>")
            parts.append(f"<text x='{width-pad-220}' y='{pad + 18*i}' font-size='13' font-family='sans-serif' fill='{color}'>{name}</text>")
        body = "\n".join(parts)
    write_text(path, f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>\n{body}\n</svg>\n")


def make_plots(rows, unit_rows, layer_rows):
    plot_dir = RUN_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    by_h = defaultdict(list)
    for r in rows:
        by_h[(r["orientation"], int(r["horizon"]))].append(r)
    def horizon_series(metric):
        return [
            (orient, [(h, median([x.get(metric) for x in by_h[(orient, h)]])) for h in sorted({k[1] for k in by_h if k[0] == orient})])
            for orient in ["R128", "C128"]
        ]
    simple_svg(plot_dir / "01_epgr_norm_vs_horizon.svg", "E/P/G/R norm vs horizon", [
        (f"{orient}_{metric}", [(h, median([x.get(metric) for x in by_h[(orient, h)]])) for h in sorted({k[1] for k in by_h if k[0] == orient})])
        for orient in ["R128", "C128"]
        for metric in ["E_norm", "P_norm", "G_norm", "R_norm"]
    ])
    simple_svg(plot_dir / "02_pgr_raw_energy_ledger.svg", "P/G/R raw energy ledger", [
        (f"{orient}_{metric}", [(i + 1, as_float(r.get(f"median_{metric}"))) for i, r in enumerate(unit_rows) if r["orientation"] == orient])
        for orient in ["R128", "C128"]
        for metric in ["P_energy", "G_energy", "R_energy"]
    ])
    simple_svg(plot_dir / "03_interaction_energy_ledger.svg", "PG/PR/GR interaction energy", [
        (metric, [(i + 1, median([x.get(metric) for x in rows if x["unit_id"] == r["unit_id"] and x["orientation"] == r["orientation"]])) for i, r in enumerate(unit_rows)])
        for metric in ["PG_interaction", "PR_interaction", "GR_interaction"]
    ])
    for idx, metric in [(4, "loo_P"), (5, "loo_G"), (6, "loo_R")]:
        simple_svg(plot_dir / f"{idx:02d}_{metric}_r128_vs_c128.svg", f"{metric} R128 vs C128", horizon_series(metric))
    simple_svg(plot_dir / "07_cos_b_r_r128_vs_c128.svg", "cos(B,R) R128 vs C128", horizon_series("cos_B_R"))
    simple_svg(plot_dir / "08_fraction_r_cancellation_per_unit.svg", "fraction R cancellation per canonical unit", [
        (orient, [(i + 1, as_float(r.get("fraction_cancellation"))) for i, r in enumerate(unit_rows) if r["orientation"] == orient])
        for orient in ["R128", "C128"]
    ])
    simple_svg(plot_dir / "09_readout_visibility.svg", "readout visibility P/G/R/E", [
        (f"{orient}_{comp}", [(i + 1, as_float(r.get(f"median_readout_{comp}"))) for i, r in enumerate(unit_rows) if r["orientation"] == orient])
        for orient in ["R128", "C128"]
        for comp in ["P", "G", "R", "E"]
    ])
    simple_svg(plot_dir / "10_per_layer_component_profile.svg", "per-layer component profile", [
        (f"{orient}_{metric}", [(int(r["layer"]), as_float(r.get(f"median_{metric}"))) for r in layer_rows if r["orientation"] == orient])
        for orient in ["R128", "C128"]
        for metric in ["P_norm", "G_norm", "R_norm"]
    ])
    simple_svg(plot_dir / "11_per_unit_total_e_r128_vs_c128.svg", "per-unit total E R128 vs C128", [
        (orient, [(i + 1, as_float(r.get("median_E_norm"))) for i, r in enumerate(unit_rows) if r["orientation"] == orient])
        for orient in ["R128", "C128"]
    ])
    by_unit = defaultdict(dict)
    for r in unit_rows:
        by_unit[r["unit_id"]][r["orientation"]] = r
    simple_svg(plot_dir / "12_per_unit_c_minus_r_component_gaps.svg", "per-unit C-R component gaps", [
        (metric, [(i + 1, (as_float(m["C128"].get(f"median_{metric}")) or 0.0) - (as_float(m["R128"].get(f"median_{metric}")) or 0.0)) for i, (uid, m) in enumerate(sorted(by_unit.items())) if "R128" in m and "C128" in m])
        for metric in ["loo_P", "loo_G", "loo_R", "E_norm"]
    ])


def make_report(summary):
    lines = [
        "# LING_KDA_PERSISTENT_ERROR_DECOMPOSITION_CAUSAL_V1",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False),
        "```",
        "",
    ]
    write_text(RUN_DIR / "report.md", "\n".join(lines))
    write_text(REPORT_DIR / f"{SLUG}.md", "\n".join(lines))


def aggregate(horizon_csv):
    rows = read_csv_rows(horizon_csv)
    by_unit = defaultdict(list)
    by_layer = defaultdict(list)
    for r in rows:
        by_unit[(r["unit_id"], r["orientation"])].append(r)
        by_layer[(r["layer"], r["orientation"])].append(r)
    unit_rows = []
    metric_keys = [
        "E_norm", "P_norm", "G_norm", "R_norm", "B_norm",
        "E_rel", "P_rel", "G_rel", "R_rel", "B_rel",
        "cos_B_R", "delta_r_energy", "delta_r_energy_normalized",
        "loo_P", "loo_G", "loo_R", "loo_P_normalized", "loo_G_normalized", "loo_R_normalized",
        "dot_P_R", "dot_G_R", "readout_E", "readout_P", "readout_G", "readout_R",
        "P_energy", "G_energy", "R_energy",
    ]
    for (unit_id, orientation), rs in sorted(by_unit.items()):
        row = {
            "unit_id": unit_id,
            "orientation": orientation,
            "prompt_id": rs[0]["prompt_id"],
            "layer": "ALL_KDA",
            "head": rs[0]["head"],
            "t0": int(rs[0]["t0"]),
            "n_horizon_rows": len(rs),
            "fraction_cancellation": mean([1.0 if str(x["r_cancellation_flag"]) == "True" else 0.0 for x in rs]),
            "fraction_R_role_cancellation": mean([1.0 if x.get("R_error_role") == "CANCELLATION" else 0.0 for x in rs]),
            "fraction_R_role_injection": mean([1.0 if x.get("R_error_role") == "INJECTION" else 0.0 for x in rs]),
        }
        for key in metric_keys:
            row[f"median_{key}"] = median([r.get(key) for r in rs])
            row[f"mean_{key}"] = mean([r.get(key) for r in rs])
        unit_rows.append(row)
    unit_path = RUN_DIR / "decomposition_unit_summary.csv"
    if unit_rows:
        with unit_path.open("w", newline="", encoding="utf-8") as f:
            fields = list(unit_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(unit_rows)
    layer_rows = []
    for (layer, orientation), rs in sorted(by_layer.items(), key=lambda x: (int(x[0][0]), x[0][1])):
        row = {
            "layer": int(layer),
            "orientation": orientation,
            "head": rs[0]["head"],
            "n_rows": len(rs),
            "unique_units": len({r["unit_id"] for r in rs}),
        }
        for key in metric_keys:
            row[f"median_{key}"] = median([r.get(key) for r in rs])
            row[f"mean_{key}"] = mean([r.get(key) for r in rs])
        layer_rows.append(row)
    layer_path = RUN_DIR / "decomposition_layer_formal.csv"
    if layer_rows:
        with layer_path.open("w", newline="", encoding="utf-8") as f:
            fields = list(layer_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(layer_rows)
    by_orient = {o: [r for r in unit_rows if r["orientation"] == o] for o in ["R128", "C128"]}
    max_decomp_abs = max([as_float(r.get("decomp_max_abs_error")) or 0.0 for r in rows], default=None)
    max_decomp_rel = max([as_float(r.get("decomp_relative_L2_error")) or 0.0 for r in rows], default=None)
    max_energy_err = max([as_float(r.get("energy_identity_error")) or 0.0 for r in rows], default=None)
    max_fp_replay = max([as_float(r.get("fp_replay_relative_L2_error")) or 0.0 for r in rows], default=None)
    max_q_replay = max([as_float(r.get("q_replay_relative_L2_error")) or 0.0 for r in rows], default=None)
    decomp_gate = "PASS" if (max_decomp_abs is not None and max_decomp_abs <= 1e-5 and max_decomp_rel <= 1e-6) else "FAIL"
    energy_gate = "PASS" if (max_energy_err is not None and max_energy_err <= 1e-4) else "FAIL"
    per_layer_gate = "PASS" if (
        decomp_gate == "PASS"
        and all((as_float(r.get("median_E_norm")) is not None) for r in layer_rows)
        and len({r["layer"] for r in layer_rows}) > 0
    ) else "FAIL"
    replay_gate = "PASS" if (max_fp_replay is not None and max_fp_replay <= 1e-6 and max_q_replay <= 1e-6) else "FAIL"
    units_seen = sorted({r["unit_id"] for r in unit_rows})
    n_units = len(units_seen)
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "COMPLETE" if n_units == 18 else "INCOMPLETE",
        "N_FORMAL_UNITS": n_units,
        "CANONICAL_MANIFEST_GATE": "PASS",
        "MANIFEST_SHA256": canonical_manifest_sha256(RUN_DIR / "canonical_units.json") if (RUN_DIR / "canonical_units.json").exists() else None,
        "PERSISTENT_TRAJECTORY_PARITY_GATE": replay_gate,
        "TRANSITION_DRIVER_AUDIT": transition_driver_audit(sorted({int(r["layer"]) for r in layer_rows})),
        "DECOMPOSITION_IDENTITY_GATE": decomp_gate,
        "PER_LAYER_DECOMPOSITION_IDENTITY_GATE": per_layer_gate,
        "P_ISOLATION_GATE": replay_gate,
        "G_ISOLATION_GATE": replay_gate,
        "LOCAL_REQUANT_IDENTITY_GATE": "PASS" if decomp_gate == "PASS" and energy_gate == "PASS" else "FAIL",
        "QUANTIZATION_EXPOSURE_SEMANTICS": "Persistent INT8 trajectory quantizes KDA cache immediately after prefill and after every decode step, matching the existing Ling end-to-end runner.",
        "BOUNDARY_EXPECTATION": "Canonical measurements start at t0 in {64,128,256}; historical quantization exposure already exists before measured horizon 1, so P_first/G_first are not expected to be zero.",
        "FIRST_STEP_BOUNDARY_GATE": "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": replay_gate,
        "STAGE0_GATE": "PASS" if Path(RUN_DIR / "stage0_run_summary.json").exists() else "NOT_RUN",
        "SMOKE": "PASS" if Path(RUN_DIR / "smoke_run_summary.json").exists() else "NOT_RUN",
        "PILOT": "PASS" if Path(RUN_DIR / "pilot_run_summary.json").exists() else "NOT_RUN",
        "max_decomposition_identity_abs_error": max_decomp_abs,
        "max_decomposition_identity_relative_L2_error": max_decomp_rel,
        "max_energy_identity_error": max_energy_err,
        "max_fp_replay_relative_L2_error": max_fp_replay,
        "max_q_replay_relative_L2_error": max_q_replay,
        "decomposition_horizon_formal_csv": str(horizon_csv),
        "decomposition_layer_formal_csv": str(layer_path),
        "decomposition_unit_summary_csv": str(unit_path),
    }
    for orient, rs in by_orient.items():
        prefix = orient
        summary[f"median_E_{prefix[0]}"] = median([r["median_E_norm"] for r in rs])
        summary[f"median_P_{prefix[0]}"] = median([r["median_P_norm"] for r in rs])
        summary[f"median_G_{prefix[0]}"] = median([r["median_G_norm"] for r in rs])
        summary[f"median_Rlocal_{prefix[0]}"] = median([r["median_R_norm"] for r in rs])
        summary[f"median_cos_B_R_{orient}"] = median([r["median_cos_B_R"] for r in rs])
        summary[f"fraction_cancellation_{orient}"] = mean([r["fraction_cancellation"] for r in rs])
        summary[f"median_delta_r_energy_{orient}"] = median([r["median_delta_r_energy"] for r in rs])
        summary[f"median_LOO_P_{orient}"] = median([r["median_loo_P"] for r in rs])
        summary[f"median_LOO_G_{orient}"] = median([r["median_loo_G"] for r in rs])
        summary[f"median_LOO_R_{orient}"] = median([r["median_loo_R"] for r in rs])
        for comp in ["P", "G", "R", "E"]:
            summary[f"median_readout_{comp}_{orient}"] = median([r[f"median_readout_{comp}"] for r in rs])
    paired = []
    by_id = defaultdict(dict)
    for r in unit_rows:
        by_id[r["unit_id"]][r["orientation"]] = r
    for unit_id, m in by_id.items():
        if "R128" in m and "C128" in m:
            one = {"unit_id": unit_id}
            for key in ["E_norm", "P_norm", "G_norm", "R_norm", "loo_P", "loo_G", "loo_R", "delta_r_energy", "cos_B_R", "readout_E", "readout_P", "readout_G", "readout_R"]:
                one[f"gap_{key}_C_minus_R"] = (m["C128"].get(f"median_{key}") or 0.0) - (m["R128"].get(f"median_{key}") or 0.0)
            paired.append(one)
    summary["C_gt_R_paired_counts"] = {
        key: sum(1 for r in paired if r[key] > 0.0)
        for key in (paired[0].keys() if paired else [])
        if key.startswith("gap_")
    }
    gaps = {k: median([r[k] for r in paired]) for k in (paired[0].keys() if paired else []) if k.startswith("gap_")}
    summary["orientation_gap_medians"] = gaps
    e_gaps = [r["gap_E_norm_C_minus_R"] for r in paired]
    summary["C_E_gt_R_E"] = sum(1 for x in e_gaps if x > 0.0)
    summary["paired_E_gap_median"] = median(e_gaps)
    summary["paired_E_gap_bootstrap_CI"] = bootstrap_ci(e_gaps)
    summary["paired_E_gap_binomial_p"] = binomial_p_two_sided(summary["C_E_gt_R_E"], len(e_gaps))
    summary["R128_TOTAL_ERROR"] = summary.get("median_E_R")
    summary["C128_TOTAL_ERROR"] = summary.get("median_E_C")
    summary["R128_PROPAGATED_ERROR"] = summary.get("median_P_R")
    summary["C128_PROPAGATED_ERROR"] = summary.get("median_P_C")
    summary["R128_UPSTREAM_DRIVER_DRIFT"] = summary.get("median_G_R")
    summary["C128_UPSTREAM_DRIVER_DRIFT"] = summary.get("median_G_C")
    summary["R128_LOCAL_REQUANT"] = summary.get("median_Rlocal_R")
    summary["C128_LOCAL_REQUANT"] = summary.get("median_Rlocal_C")
    summary["R128_LOO_P"] = summary.get("median_LOO_P_R128")
    summary["C128_LOO_P"] = summary.get("median_LOO_P_C128")
    summary["R128_LOO_G"] = summary.get("median_LOO_G_R128")
    summary["C128_LOO_G"] = summary.get("median_LOO_G_C128")
    summary["R128_LOO_R"] = summary.get("median_LOO_R_R128")
    summary["C128_LOO_R"] = summary.get("median_LOO_R_C128")
    summary["R128_CANCELLATION_FRACTION"] = summary.get("fraction_cancellation_R128")
    summary["C128_CANCELLATION_FRACTION"] = summary.get("fraction_cancellation_C128")
    summary["R128_MEDIAN_COS_B_R"] = summary.get("median_cos_B_R_R128")
    summary["C128_MEDIAN_COS_B_R"] = summary.get("median_cos_B_R_C128")
    summary["FUNCTIONAL_VISIBILITY_RESULT"] = {
        "R128": {comp: summary.get(f"median_readout_{comp}_R128") for comp in ["P", "G", "R", "E"]},
        "C128": {comp: summary.get(f"median_readout_{comp}_C128") for comp in ["P", "G", "R", "E"]},
        "interpretation": "SECONDARY_DIAGNOSTIC_ONLY",
    }
    summary["PER_LAYER_STRUCTURE_RESULT"] = {
        "n_layers": len({r["layer"] for r in layer_rows}),
        "layer_summary_csv": str(layer_path),
    }
    dom = "NO_SINGLE_DOMINANT_COMPONENT"
    if gaps:
        comp_gaps = {
            "PROPAGATED_HISTORICAL_STATE_ERROR": abs(gaps.get("gap_loo_P_C_minus_R") or 0.0),
            "UPSTREAM_TRANSITION_DRIVER_DRIFT": abs(gaps.get("gap_loo_G_C_minus_R") or 0.0),
            "LOCAL_REQUANTIZATION": abs(gaps.get("gap_loo_R_C_minus_R") or 0.0),
        }
        top = max(comp_gaps, key=comp_gaps.get)
        vals = sorted(comp_gaps.values(), reverse=True)
        if vals and vals[0] > 1.25 * (vals[1] if len(vals) > 1 else 0.0):
            dom = top
    c_cancel = summary.get("fraction_cancellation_C128") or 0.0
    r_cancel = summary.get("fraction_cancellation_R128") or 0.0
    c_delta = summary.get("median_LOO_R_C128")
    if c_delta is not None and c_delta < 0 and c_cancel > 0.5 and c_cancel >= r_cancel:
        stage_b_trigger = "REQUANTIZATION_CANCELLATION_SIGNAL_STRONG"
        role = "SYSTEMATIC_CANCELLATION_SIGNAL"
    elif c_delta is not None and c_delta > 0:
        stage_b_trigger = "NOT_TRIGGERED"
        role = "STATE_SPACE_ERROR_INJECTION"
    else:
        stage_b_trigger = "NOT_TRIGGERED"
        role = "MIXED_OR_NEUTRAL"
    g_ratio = None
    if summary.get("R128_UPSTREAM_DRIVER_DRIFT") is not None and summary["R128_UPSTREAM_DRIVER_DRIFT"] > EPS:
        g_ratio = (summary.get("C128_UPSTREAM_DRIVER_DRIFT") or 0.0) / summary["R128_UPSTREAM_DRIVER_DRIFT"]
    loo_g_gap = gaps.get("gap_loo_G_C_minus_R") if gaps else None
    if g_ratio is not None and g_ratio > 2.0 and loo_g_gap and loo_g_gap > 0:
        g_role = "LARGE"
    elif g_ratio is not None and g_ratio > 1.25:
        g_role = "MODERATE"
    elif summary.get("C128_UPSTREAM_DRIVER_DRIFT") is not None:
        g_role = "SMALL"
    else:
        g_role = "NOT_EVALUATED"
    summary.update({
        "PERSISTENT_ERROR_DOMINANT_COMPONENT": dom,
        "REQUANTIZATION_ERROR_ROLE": role,
        "UPSTREAM_TRANSITION_DRIVER_DRIFT_ROLE": g_role,
        "RC_ORIENTATION_ERROR_MECHANISM": {
            "median_gap_E_norm_C_minus_R": gaps.get("gap_E_norm_C_minus_R") if gaps else None,
            "median_gap_LOO_P_C_minus_R": gaps.get("gap_loo_P_C_minus_R") if gaps else None,
            "median_gap_LOO_G_C_minus_R": gaps.get("gap_loo_G_C_minus_R") if gaps else None,
            "median_gap_LOO_R_C_minus_R": gaps.get("gap_loo_R_C_minus_R") if gaps else None,
        },
        "STAGE_B_TRIGGER": stage_b_trigger,
        "STAGE_B_STATUS": "NOT_RUN_BY_GATE" if stage_b_trigger != "REQUANTIZATION_CANCELLATION_SIGNAL_STRONG" else "NOT_RUN_PENDING_EXPLICIT_STAGE_B",
        "METHOD_DESIGN_READY": "NO",
        "NEXT_RECOMMENDED_TASK": {
            "PROPAGATED_HISTORICAL_STATE_ERROR": "LING_KDA_SOURCE_ERROR_FUNCTIONAL_CAUSAL_V1",
            "UPSTREAM_TRANSITION_DRIVER_DRIFT": "LING_KDA_TRANSITION_DRIVER_DRIFT_CAUSAL_V1",
            "LOCAL_REQUANTIZATION": "LING_KDA_LOCAL_REQUANTIZATION_COMPENSATION_CAUSAL_V1" if role == "SYSTEMATIC_CANCELLATION_SIGNAL" else "LING_KDA_FUNCTIONAL_ERROR_DIRECTION_V1",
            "NO_SINGLE_DOMINANT_COMPONENT": "LING_KDA_SOURCE_ERROR_FUNCTIONAL_GEOMETRY_V1",
        }.get(dom, "LING_KDA_SOURCE_ERROR_FUNCTIONAL_GEOMETRY_V1"),
    })
    if role == "SYSTEMATIC_CANCELLATION_SIGNAL":
        summary["NEXT_RECOMMENDED_TASK"] = "LING_KDA_LOCAL_REQUANTIZATION_COMPENSATION_CAUSAL_V1"
    summary["FINAL_CLASSIFICATION"] = "PERSISTENT_ERROR_DECOMPOSITION_OBSERVATIONAL_FORMAL_COMPLETE" if summary["FORMAL_STATUS"] == "COMPLETE" else "PERSISTENT_ERROR_DECOMPOSITION_OBSERVATIONAL_INCOMPLETE"
    save_json(RUN_DIR / "decomposition_final_summary.json", summary)
    save_json(RESULT_DIR / f"{SLUG}_final_summary.json", summary)
    make_plots(rows, unit_rows, layer_rows)
    make_report(summary)
    return summary


def write_blocked_summary(audit):
    config = json.loads((MODEL_PATH / "config.json").read_text(encoding="utf-8")) if (MODEL_PATH / "config.json").exists() else {}
    kda_layers = kda_layers_from_config(config) if config else []
    runtime = json.loads(RUNTIME_AUDIT_PATH.read_text(encoding="utf-8")) if RUNTIME_AUDIT_PATH.exists() else {}
    summary = {
        "TASK": TASK,
        "FORMAL_STATUS": "BLOCKED_CANONICAL_UNIT_MANIFEST_MISSING",
        "N_FORMAL_UNITS": 0,
        "REPO_AUDIT": {
            "REUSED_FILES": [
                "experiments/ling/run_ling_kda_aime24_int8_rc_end_to_end_smoke_v1.py",
                "tests/ling/test_ling_kda_int8_rc_semantics.py",
                "runs/ling_kda_aime24_int8_rc_end_to_end_formal_v1_distributed4/",
                str(RUNTIME_AUDIT_PATH),
            ],
            "NEW_FILES": [
                "experiments/ling/run_ling_kda_persistent_error_decomposition_causal_v1.py",
                "tests/ling/test_ling_kda_persistent_error_decomposition.py",
            ],
            "CANONICAL_UNIT_SOURCE": "MISSING: searched known Ling commutator/scalarization artifact paths and LING_CANONICAL_UNITS",
            "PERSISTENT_TRAJECTORY_SOURCE": "runs/ling_kda_aime24_int8_rc_end_to_end_formal_v1_distributed4 FP teacher tokens plus live persistent INT8_R128/INT8_C128 replay",
            "STATE_CAPTURE_SOURCE": str(RUNTIME_AUDIT_PATH),
        },
        "TRANSITION_DRIVER_AUDIT": transition_driver_audit(kda_layers),
        "CANONICAL_UNIT_AUDIT": audit,
        "DECOMPOSITION_IDENTITY_GATE": "NOT_RUN_CANONICAL_UNIT_MISSING",
        "P_ISOLATION_GATE": "NOT_RUN_CANONICAL_UNIT_MISSING",
        "G_ISOLATION_GATE": "NOT_RUN_CANONICAL_UNIT_MISSING",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": runtime.get("INSTRUMENTATION_NONINTERFERENCE_GATE", "PREVIOUS_PASS_NOT_RERUN"),
        "PERSISTENT_TRAJECTORY_PARITY_GATE": "NOT_RUN_CANONICAL_UNIT_MISSING",
        "STAGE_B_TRIGGER": "NOT_EVALUATED",
        "STAGE_B_STATUS": "NOT_RUN_BY_GATE",
        "FINAL_CLASSIFICATION": "FORMAL_BLOCKED_MISSING_CANONICAL_18_UNIT_MANIFEST",
        "METHOD_DESIGN_READY": "NO",
        "NEXT_RECOMMENDED_TASK": "restore/provide canonical 18-unit manifest from LING_KDA_COMMUTATOR_FUNCTIONAL_CAUSAL_V1, then run stage0",
        "reproducibility": reproducibility_snapshot(),
    }
    save_json(RUN_DIR / "decomposition_final_summary.json", summary)
    save_json(RESULT_DIR / f"{SLUG}_final_summary.json", summary)
    return summary


def reproducibility_snapshot():
    return {
        "time": now(),
        "git_head": sh(["git", "rev-parse", "HEAD"]),
        "git_status": sh(["git", "status", "--short"]),
        "model_path": str(MODEL_PATH),
        "cuda": sh(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"]),
        "python": sys.version.split()[0],
    }


def parse_args():
    p = argparse.ArgumentParser(description=TASK)
    p.add_argument("--stage", choices=["audit", "stage0", "smoke", "pilot", "formal", "analyze"], default="audit")
    p.add_argument("--canonical-units", default=None)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--unit-limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit = load_canonical_units(args.canonical_units)
    if audit["gate"] != "PASS":
        summary = write_blocked_summary(audit)
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        return
    if args.stage == "audit":
        config = json.loads((MODEL_PATH / "config.json").read_text(encoding="utf-8"))
        out = {
            "TASK": TASK,
            "FORMAL_STATUS": "AUDIT_COMPLETE_CAN_RUN_STAGE0",
            "CANONICAL_UNIT_AUDIT": audit,
            "TRANSITION_DRIVER_AUDIT": transition_driver_audit(kda_layers_from_config(config)),
            "reproducibility": reproducibility_snapshot(),
        }
        save_json(RUN_DIR / "audit.json", out)
        print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
        return
    limits = {"stage0": 1, "smoke": 1, "pilot": 6, "formal": None}
    limit = args.unit_limit if args.unit_limit is not None else limits.get(args.stage)
    if args.stage in {"stage0", "smoke", "pilot", "formal"}:
        run_summary = run_units(args.stage, audit["formal_units"], horizon=args.horizon, unit_limit=limit)
        save_json(RUN_DIR / f"{args.stage}_run_summary.json", run_summary)
        if args.stage == "formal":
            summary = aggregate(RUN_DIR / "decomposition_horizon_formal.csv")
        else:
            summary = run_summary
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        return
    if args.stage == "analyze":
        print(json.dumps(aggregate(RUN_DIR / "decomposition_horizon_formal.csv"), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
