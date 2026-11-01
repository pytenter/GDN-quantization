#!/usr/bin/env python3
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch


TASK = "KDA_ROTATION_POST_READOUT_GAIN_RETENTION_V1"
SLUG = "kda_rotation_post_readout_gain_retention_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
KDA_PYTHON = "/data01/user2/.conda/envs/ling-kda/bin/python"
MODEL_PATH = Path(os.environ.get("LING_MODEL_PATH", "/data01/user2/zypan_ling_dist/models/Ling-3.0-tiny"))
EXACT_REPLAY = REPO / "experiments" / "rotation" / "run_gdn_kda_rotation_exact_frozen_driver_replay_v1.py"
KDA_CANONICAL_UNITS = REPO / "runs" / "ling_kda_persistent_error_decomposition_causal_v1" / "canonical_units.json"
HORIZONS = [16, 32, 64]
PRIMARY_HORIZON = 64
EPS = 1e-12
RHT_SEED = 0
EXPECTED_CANONICAL_N = 18
BOOTSTRAP_SEED = 20260914

STAGE_ORDER = [
    "core_readout",
    "o_norm_gated_rms",
    "head_merge",
    "o_proj",
    "attention_residual_add",
    "post_attention_layernorm",
    "mlp",
    "layer_output",
    "model_norm",
    "lm_head_logits",
]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd, cwd=REPO):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXACT = None


def exact():
    global _EXACT
    if _EXACT is None:
        _EXACT = import_file(EXACT_REPLAY, "exact_frozen_driver_for_kda_post_readout_v1")
    return _EXACT


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=True) + "\n")
        f.flush()


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    if not fields:
        fields = ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def as_float(x, default=None):
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def mean(xs):
    vals = [float(x) for x in xs if finite(x)]
    return sum(vals) / len(vals) if vals else None


def median(xs):
    vals = [float(x) for x in xs if finite(x)]
    return statistics.median(vals) if vals else None


def std(xs):
    vals = [float(x) for x in xs if finite(x)]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def bootstrap_ci(xs, n=4000, seed=BOOTSTRAP_SEED):
    vals = [float(x) for x in xs if finite(x)]
    if not vals:
        return None
    rng = random.Random(int(seed))
    out = []
    for _ in range(int(n)):
        out.append(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals))
    out.sort()
    return [out[int(0.025 * (n - 1))], out[int(0.975 * (n - 1))]]


def tensor_norm(x):
    return float(torch.linalg.vector_norm(x.detach().double()).item())


def tensor_hash(x):
    y = x.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(y.shape)).encode("ascii"))
    h.update(str(y.dtype).encode("ascii"))
    if y.dtype in {torch.bfloat16, torch.float16}:
        h.update(y.float().numpy().tobytes())
    else:
        h.update(y.numpy().tobytes())
    return h.hexdigest()


def stack_hash(stack):
    h = hashlib.sha256()
    for key in sorted(stack):
        h.update(str(key).encode("ascii"))
        h.update(tensor_hash(stack[key]).encode("ascii"))
    return h.hexdigest()


def stack_norm(stack):
    return math.sqrt(sum(tensor_norm(v) ** 2 for v in stack.values()))


def readout_perturbation(error, query):
    return torch.einsum("bhkv,bhk->bhv", error.detach().float(), query.detach().float())


def normalized_stage_risk(fp, delta, eps=EPS):
    f = fp.detach().double()
    d = delta.detach().double()
    return float((torch.linalg.vector_norm(d) ** 2 / (torch.linalg.vector_norm(f) ** 2 + eps)).item())


def relative_l2(fp, delta, eps=EPS):
    f = fp.detach().double()
    d = delta.detach().double()
    return float((torch.linalg.vector_norm(d) / (torch.linalg.vector_norm(f) + eps)).item())


def stage_gain(native_risk, rotated_risk, eps=EPS):
    return 1.0 - float(rotated_risk) / (float(native_risk) + eps)


def delta_gain(current_gain, previous_gain):
    return float(current_gain) - float(previous_gain)


def operator_transfer(prev_native, next_native, prev_rotated, next_rotated, eps=EPS):
    g_native = tensor_norm(next_native) / (tensor_norm(prev_native) + eps)
    g_rotated = tensor_norm(next_rotated) / (tensor_norm(prev_rotated) + eps)
    return {
        "g_native": g_native,
        "g_rotated": g_rotated,
        "Lambda": g_rotated / (g_native + eps),
    }


def scale_safe_operator_transfer(prev_fp, next_fp, prev_native, next_native, prev_rotated, next_rotated, eps=EPS):
    prev_n = tensor_norm(prev_native) / (tensor_norm(prev_fp) + eps)
    next_n = tensor_norm(next_native) / (tensor_norm(next_fp) + eps)
    prev_r = tensor_norm(prev_rotated) / (tensor_norm(prev_fp) + eps)
    next_r = tensor_norm(next_rotated) / (tensor_norm(next_fp) + eps)
    g_native = next_n / (prev_n + eps)
    g_rotated = next_r / (prev_r + eps)
    return {
        "scale_safe_g_native": g_native,
        "scale_safe_g_rotated": g_rotated,
        "scale_safe_Lambda": g_rotated / (g_native + eps),
    }


def check_matched_pair_provenance(native, rotated):
    differing = []
    for key in sorted(set(native) | set(rotated)):
        if native.get(key) != rotated.get(key):
            differing.append(key)
    allowed = {"branch", "delta_o_sha256"}
    bad = [key for key in differing if key not in allowed]
    return {"gate": "PASS" if not bad else "FAIL", "differing_fields": differing, "bad_fields": bad}


def zero_perturbation_noninterference(fp_stages, zero_stages):
    max_abs = 0.0
    rels = []
    for key, fp in fp_stages.items():
        if key not in zero_stages or not torch.is_tensor(fp) or not torch.is_tensor(zero_stages[key]):
            continue
        diff = zero_stages[key].detach().float() - fp.detach().float()
        max_abs = max(max_abs, float(diff.abs().max().item()))
        rels.append(relative_l2(fp, diff))
    max_rel = max(rels, default=0.0)
    return {
        "gate": "PASS" if max_abs <= 1e-7 and max_rel <= 1e-7 else "FAIL",
        "max_abs_error": max_abs,
        "max_relative_l2": max_rel,
    }


def cache_modification_gate(before_sha256, after_sha256):
    return "PASS" if before_sha256 == after_sha256 else "FAIL"


def build_static_kda_post_readout_audit():
    source = MODEL_PATH / "modeling_bailing_moe_v3.py"
    stages = [
        {
            "stage_name": "core_readout",
            "module_function": "fla.ops.kda.chunk_kda or fused_recurrent_kda",
            "source_file_function_identifier": f"{source}:BailingMoeV3KimiDeltaAttention.forward lines 863-896",
            "input_shape": "q/k/v/g [B,T,H,128], recurrent state [B,H,K,V]",
            "output_shape": "o [B,T,H,V]",
            "dtype": "model runtime dtype, captured as float32",
            "linear_or_nonlinear": "linear in recurrent state at fixed q",
            "uses_another_branch": True,
            "auxiliary_values_shared_in_matched_intervention": "q/k/v/g/beta/A_log/dt_bias fixed by clean FP context at hook boundary",
            "hook_location": "patched chunk_kda/fused_recurrent_kda return output before cache assignment",
        },
        {
            "stage_name": "o_norm_gated_rms",
            "module_function": "BailingMoeV3KimiDeltaAttention.o_norm = FusedRMSNormGated",
            "source_file_function_identifier": f"{source}:BailingMoeV3KimiDeltaAttention.forward lines 909-918",
            "input_shape": "o [B,T,H,V], g [B,T,H,V]",
            "output_shape": "[B,T,H,V]",
            "dtype": "model runtime dtype",
            "linear_or_nonlinear": "nonlinear normalization/gate",
            "uses_another_branch": True,
            "auxiliary_values_shared_in_matched_intervention": "target-layer gate g is computed from clean FP hidden before injected o",
            "hook_location": "o_norm module forward",
        },
        {
            "stage_name": "head_merge",
            "module_function": "einops.rearrange(o, 'b t h d -> b t (h d)')",
            "source_file_function_identifier": f"{source}:BailingMoeV3KimiDeltaAttention.forward line 919",
            "input_shape": "[B,T,H,V]",
            "output_shape": "[B,T,H*V]",
            "dtype": "model runtime dtype",
            "linear_or_nonlinear": "linear reshape",
            "uses_another_branch": False,
            "auxiliary_values_shared_in_matched_intervention": True,
            "hook_location": "o_proj pre-hook",
        },
        {
            "stage_name": "o_proj",
            "module_function": "BailingMoeV3KimiDeltaAttention.o_proj",
            "source_file_function_identifier": f"{source}:BailingMoeV3KimiDeltaAttention.forward line 920",
            "input_shape": "[B,T,H*V]",
            "output_shape": "[B,T,hidden_size]",
            "dtype": "model runtime dtype",
            "linear_or_nonlinear": "linear",
            "uses_another_branch": False,
            "auxiliary_values_shared_in_matched_intervention": True,
            "hook_location": "o_proj module forward",
        },
        {
            "stage_name": "attention_residual_add",
            "module_function": "BailingMoeV3DecoderLayer.forward residual + attention output",
            "source_file_function_identifier": f"{source}:BailingMoeV3DecoderLayer.forward line 1089",
            "input_shape": "[B,T,hidden_size]",
            "output_shape": "[B,T,hidden_size]",
            "dtype": "model runtime dtype",
            "linear_or_nonlinear": "affine add with fixed residual at target layer",
            "uses_another_branch": True,
            "auxiliary_values_shared_in_matched_intervention": "target-layer residual is clean FP",
            "hook_location": "post_attention_layernorm pre-hook",
        },
        {
            "stage_name": "post_attention_layernorm",
            "module_function": "BailingMoeV3DecoderLayer.post_attention_layernorm",
            "source_file_function_identifier": f"{source}:BailingMoeV3DecoderLayer.forward line 1093",
            "input_shape": "[B,T,hidden_size]",
            "output_shape": "[B,T,hidden_size]",
            "dtype": "model runtime dtype",
            "linear_or_nonlinear": "nonlinear RMSNorm",
            "uses_another_branch": False,
            "auxiliary_values_shared_in_matched_intervention": "normalization statistics are branch-dependent by definition",
            "hook_location": "post_attention_layernorm module forward",
        },
        {
            "stage_name": "mlp",
            "module_function": "BailingMoeV3DecoderLayer.mlp",
            "source_file_function_identifier": f"{source}:BailingMoeV3DecoderLayer.forward line 1094",
            "input_shape": "[B,T,hidden_size]",
            "output_shape": "[B,T,hidden_size] or tuple(output, router_logits)",
            "dtype": "model runtime dtype",
            "linear_or_nonlinear": "nonlinear MoE/MLP",
            "uses_another_branch": True,
            "auxiliary_values_shared_in_matched_intervention": "router/gate values may be branch-dependent after the target residual",
            "hook_location": "mlp module forward",
        },
        {
            "stage_name": "layer_output",
            "module_function": "BailingMoeV3DecoderLayer.forward residual + mlp output",
            "source_file_function_identifier": f"{source}:BailingMoeV3DecoderLayer.forward line 1100",
            "input_shape": "[B,T,hidden_size]",
            "output_shape": "[B,T,hidden_size]",
            "dtype": "model runtime dtype",
            "linear_or_nonlinear": "affine add",
            "uses_another_branch": True,
            "auxiliary_values_shared_in_matched_intervention": "MLP residual is branch-dependent after target perturbation",
            "hook_location": "decoder layer post-hook",
        },
        {
            "stage_name": "model_norm",
            "module_function": "BailingMoeV3Model.norm",
            "source_file_function_identifier": f"{source}:BailingMoeV3Model.forward line 1410",
            "input_shape": "[B,T,hidden_size]",
            "output_shape": "[B,T,hidden_size]",
            "dtype": "model runtime dtype",
            "linear_or_nonlinear": "nonlinear RMSNorm",
            "uses_another_branch": False,
            "auxiliary_values_shared_in_matched_intervention": "same-token hidden may be branch-dependent",
            "hook_location": "model.model.norm forward",
        },
        {
            "stage_name": "lm_head_logits",
            "module_function": "BailingMoeV3ForCausalLM.lm_head",
            "source_file_function_identifier": f"{source}:BailingMoeV3ForCausalLM.forward lines 1577-1578",
            "input_shape": "[B,T,hidden_size]",
            "output_shape": "[B,T,vocab_size]",
            "dtype": "float32 logits",
            "linear_or_nonlinear": "linear projection",
            "uses_another_branch": False,
            "auxiliary_values_shared_in_matched_intervention": True,
            "hook_location": "lm_head forward",
        },
    ]
    return {
        "task": TASK,
        "model_path": str(MODEL_PATH),
        "KDA_POST_READOUT_PATH_AUDIT": "PASS" if source.exists() else "FAIL",
        "source_file": str(source),
        "stages": stages,
    }


class KdaPostReadoutProbe:
    def __init__(self):
        self.branch = None
        self.target_layer = None
        self.injections = {}
        self.current_layer = None
        self.records = defaultdict(dict)
        self.handles = []
        self.globals = None
        self.orig_chunk = None
        self.orig_fused = None
        self.kda_layers = []

    def install(self, model):
        kda_modules = []
        for idx, layer in enumerate(model.model.layers):
            attn = getattr(layer, "attention", None)
            if attn is None or not hasattr(attn, "A_log") or not hasattr(attn, "q_conv1d"):
                continue
            kda_modules.append((idx, layer, attn))
            self.handles.append(attn.register_forward_pre_hook(self._make_attention_pre(idx)))
            self.handles.append(attn.o_norm.register_forward_pre_hook(self._make_stage_pre(idx, "o_norm_gated_rms_input")))
            self.handles.append(attn.o_norm.register_forward_hook(self._make_stage_post(idx, "o_norm_gated_rms")))
            self.handles.append(attn.o_proj.register_forward_pre_hook(self._make_stage_pre(idx, "head_merge")))
            self.handles.append(attn.o_proj.register_forward_hook(self._make_stage_post(idx, "o_proj")))
            self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(self._make_stage_pre(idx, "attention_residual_add")))
            self.handles.append(layer.post_attention_layernorm.register_forward_hook(self._make_stage_post(idx, "post_attention_layernorm")))
            self.handles.append(layer.mlp.register_forward_hook(self._make_stage_post(idx, "mlp")))
            self.handles.append(layer.register_forward_hook(self._make_layer_post(idx)))
        if not kda_modules:
            raise RuntimeError("no Ling KDA attention modules found")
        self.kda_layers = [idx for idx, _layer, _attn in kda_modules]
        self.handles.append(model.model.norm.register_forward_hook(self._make_global_post("model_norm")))
        self.handles.append(model.lm_head.register_forward_hook(self._make_global_post("lm_head_logits")))
        self.globals = type(kda_modules[0][2]).forward.__globals__
        self.orig_chunk = self.globals["chunk_kda"]
        self.orig_fused = self.globals["fused_recurrent_kda"]
        self.globals["chunk_kda"] = self._wrap_backend("chunk_kda", self.orig_chunk)
        self.globals["fused_recurrent_kda"] = self._wrap_backend("fused_recurrent_kda", self.orig_fused)
        return list(self.kda_layers)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self.globals is not None:
            self.globals["chunk_kda"] = self.orig_chunk
            self.globals["fused_recurrent_kda"] = self.orig_fused
        self.globals = None

    def begin(self, branch, target_layer=None, injections=None):
        self.branch = str(branch)
        self.target_layer = None if target_layer is None else int(target_layer)
        self.injections = injections or {}
        self.current_layer = None
        self.records[str(branch)] = {}

    def end(self):
        self.branch = None
        self.target_layer = None
        self.injections = {}
        self.current_layer = None

    def _record(self, stage_name, tensor, layer=None):
        if self.branch is None or not torch.is_tensor(tensor):
            return
        key = stage_name if layer is None else f"L{int(layer)}:{stage_name}"
        self.records[str(self.branch)][key] = tensor.detach().float().cpu().clone()
        if layer is not None and self.target_layer is not None and int(layer) == int(self.target_layer):
            self.records[str(self.branch)][stage_name] = tensor.detach().float().cpu().clone()

    def _make_attention_pre(self, layer_idx):
        def pre(_module, _inputs):
            self.current_layer = int(layer_idx)
        return pre

    def _first_tensor(self, obj):
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, (tuple, list)):
            for item in obj:
                if torch.is_tensor(item):
                    return item
        return None

    def _make_stage_pre(self, layer_idx, stage_name):
        def pre(_module, inputs):
            if self.target_layer is None or int(layer_idx) == int(self.target_layer):
                x = self._first_tensor(inputs)
                if x is not None:
                    self._record(stage_name, x, layer_idx)
        return pre

    def _make_stage_post(self, layer_idx, stage_name):
        def post(_module, _inputs, output):
            if self.target_layer is None or int(layer_idx) == int(self.target_layer):
                x = self._first_tensor(output)
                if x is not None:
                    self._record(stage_name, x, layer_idx)
        return post

    def _make_layer_post(self, layer_idx):
        def post(_module, _inputs, output):
            if self.target_layer is None or int(layer_idx) == int(self.target_layer):
                x = self._first_tensor(output)
                if x is not None:
                    self._record("layer_output", x, layer_idx)
        return post

    def _make_global_post(self, stage_name):
        def post(_module, _inputs, output):
            x = self._first_tensor(output)
            if x is not None:
                self._record(stage_name, x, None)
        return post

    def _wrap_backend(self, name, fn):
        def wrapped(**kwargs):
            out, final_state = fn(**kwargs)
            layer = self.current_layer
            out2 = out
            if self.branch is not None and layer is not None:
                delta = self.injections.get(int(layer))
                if delta is not None:
                    d = delta.to(device=out.device, dtype=out.dtype)
                    if d.ndim == 3:
                        d = d.unsqueeze(1)
                    out2 = out + d
                if self.target_layer is None or int(layer) == int(self.target_layer):
                    self._record("core_readout", out2, layer)
            return out2, final_state
        return wrapped


def clean_stage_subset(records):
    return {name: value for name, value in records.items() if name in STAGE_ORDER and torch.is_tensor(value)}


def load_kda_context():
    mod = import_file(REPO / "experiments" / "ling" / "run_ling_kda_int8_rc_granularity_phase_diagram_v1.py", "kda_granularity_for_post_readout_v1")
    torch_mod, model, tokenizer = mod.P.load_model_and_tokenizer()
    config_obj = json.loads((mod.P.MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = mod.P.kda_layers_from_config(config_obj)
    rows_by_pid = mod.P.load_dataset()
    teacher_tokens = mod.P.load_fp_teacher_tokens()
    return mod, torch_mod, model, tokenizer, kda_layers, rows_by_pid, teacher_tokens


def load_units(scope, max_units=None):
    units_all, manifest_hash = exact().load_kda_units()
    units = exact().select_units(units_all, scope, max_units)
    return units, manifest_hash, units_all


def get_kda_stack(mod, past, layers):
    return {int(layer): mod.P.get_cache_state(past, layer).detach().float().cpu().clone() for layer in layers}


def clone_cache(past):
    return copy.deepcopy(past)


def prepare_clean_context(mod, torch_mod, model, tokenizer, unit, row, tokens, horizon):
    device = next(model.parameters()).device
    input_ids = mod.P.render_prompt(tokenizer, row["problem"]).to(device)
    mask = torch_mod.ones_like(input_ids)
    probe_off = None
    with torch_mod.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=mask,
            cache_position=torch_mod.arange(0, input_ids.shape[-1], device=device),
            use_cache=True,
        )
    past = out.past_key_values
    prompt_len = int(input_ids.shape[-1])
    n_before = int(unit["t0"]) + int(horizon) - 1
    for t, tok in enumerate(tokens[:n_before]):
        cur = torch_mod.tensor([[int(tok)]], device=device, dtype=torch_mod.long)
        mask = torch_mod.cat([mask, torch_mod.ones_like(cur)], dim=-1)
        with torch_mod.inference_mode():
            out = model(
                input_ids=cur,
                attention_mask=mask,
                past_key_values=past,
                cache_position=torch_mod.tensor([prompt_len + t], device=device, dtype=torch_mod.long),
                use_cache=True,
            )
        past = out.past_key_values
    tok = int(tokens[n_before])
    cur = torch_mod.tensor([[tok]], device=device, dtype=torch_mod.long)
    run_mask = torch_mod.cat([mask, torch_mod.ones_like(cur)], dim=-1)
    cache_position = torch_mod.tensor([prompt_len + n_before], device=device, dtype=torch_mod.long)
    return {"input_ids": cur, "attention_mask": run_mask, "past": past, "cache_position": cache_position, "token_id": tok}


def run_branch_forward(torch_mod, model, probe, context, branch, target_layer, delta):
    past_copy = clone_cache(context["past"])
    probe.begin(branch, target_layer=target_layer, injections={int(target_layer): delta})
    try:
        with torch_mod.inference_mode():
            out = model(
                input_ids=context["input_ids"],
                attention_mask=context["attention_mask"],
                past_key_values=past_copy,
                cache_position=context["cache_position"],
                use_cache=True,
            )
    finally:
        probe.end()
    return out, dict(probe.records[str(branch)])


def replay_error_stacks(tape, capture, rotation):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states_native = {layer: value.to(device) for layer, value in tape["state0"].items()}
    states_rot = {layer: exact().transform_state_to_branch(value, "kda", "rotated", rotation).to(device) for layer, value in tape["state0"].items()}
    out = {}
    readout_rows = []
    for step in tape["steps"]:
        e_native = {}
        e_rot = {}
        delta_native = {}
        delta_rot = {}
        q_sha = {}
        for layer, rec in step["records"].items():
            _core_n, pre_n = exact().replay_kda_record(capture, torch, rec, states_native[layer], rotated=False)
            qstate_n = exact().quantize_branch_state(pre_n, "kda", "native", quantizer_off=False)
            states_native[layer] = qstate_n
            native_q = exact().inverse_branch_state(qstate_n.detach().cpu(), "kda", "native", rotation)
            e_native[layer] = native_q - step["fp_states"][layer]

            _core_r, pre_r = exact().replay_kda_record(capture, torch, rec, states_rot[layer], rotated=True, rotation=rotation)
            qstate_r = exact().quantize_branch_state(pre_r, "kda", "rotated", quantizer_off=False)
            states_rot[layer] = qstate_r
            rot_q = exact().inverse_branch_state(qstate_r.detach().cpu(), "kda", "rotated", rotation)
            e_rot[layer] = rot_q - step["fp_states"][layer]

            q = step["readout_queries"][layer]
            delta_native[layer] = readout_perturbation(e_native[layer], q)
            delta_rot[layer] = readout_perturbation(e_rot[layer], q)
            q_sha[layer] = tensor_hash(q)
        h = int(step["t"])
        read_n = stack_norm(delta_native)
        read_r = stack_norm(delta_rot)
        row = {
            "unit_id": tape["unit_id"],
            "horizon": h,
            "native_readout_error_norm": read_n,
            "rotated_readout_error_norm": read_r,
            "G_read": 1.0 - read_r / (read_n + EPS),
            "driver_tape_sha256": tape["driver_tape_sha256"],
        }
        readout_rows.append(row)
        if h in HORIZONS:
            out[h] = {
                "E_N": e_native,
                "E_R": e_rot,
                "delta_N": delta_native,
                "delta_R": delta_rot,
                "q_sha256": q_sha,
                "readout_row": row,
            }
    return out, readout_rows


def stage_pair_metrics(unit_id, horizon, target_layer, fp_records, native_records, rotated_records):
    rows = []
    tensors = {}
    for stage in STAGE_ORDER:
        if stage not in fp_records or stage not in native_records or stage not in rotated_records:
            continue
        fp = fp_records[stage]
        dn = native_records[stage] - fp
        dr = rotated_records[stage] - fp
        rn = normalized_stage_risk(fp, dn)
        rr = normalized_stage_risk(fp, dr)
        row = {
            "unit_id": unit_id,
            "horizon": horizon,
            "target_layer": int(target_layer),
            "stage": stage,
            "native_risk": rn,
            "rotated_risk": rr,
            "G_j": stage_gain(rn, rr),
            "native_rel_l2": relative_l2(fp, dn),
            "rotated_rel_l2": relative_l2(fp, dr),
            "fp_norm": tensor_norm(fp),
            "native_delta_norm": tensor_norm(dn),
            "rotated_delta_norm": tensor_norm(dr),
        }
        rows.append(row)
        tensors[stage] = {"fp": fp, "native_delta": dn, "rotated_delta": dr}
    return rows, tensors


def transfer_rows_from_tensors(unit_id, horizon, target_layer, tensors):
    rows = []
    present = [s for s in STAGE_ORDER if s in tensors]
    for prev, cur in zip(present, present[1:]):
        raw = operator_transfer(
            tensors[prev]["native_delta"],
            tensors[cur]["native_delta"],
            tensors[prev]["rotated_delta"],
            tensors[cur]["rotated_delta"],
        )
        safe = scale_safe_operator_transfer(
            tensors[prev]["fp"],
            tensors[cur]["fp"],
            tensors[prev]["native_delta"],
            tensors[cur]["native_delta"],
            tensors[prev]["rotated_delta"],
            tensors[cur]["rotated_delta"],
        )
        rows.append({
            "unit_id": unit_id,
            "horizon": horizon,
            "target_layer": int(target_layer),
            "from_stage": prev,
            "to_stage": cur,
            **raw,
            **safe,
        })
    return rows


def aggregate_stage_rows(stage_rows):
    grouped = defaultdict(list)
    for row in stage_rows:
        grouped[(row["unit_id"], int(row["horizon"]), row["stage"])].append(row)
    unit_rows = []
    for (unit_id, horizon, stage), rows in sorted(grouped.items()):
        n_risk = sum(float(r["native_risk"]) for r in rows)
        r_risk = sum(float(r["rotated_risk"]) for r in rows)
        unit_rows.append({
            "unit_id": unit_id,
            "horizon": horizon,
            "stage": stage,
            "n_target_layers": len({int(r["target_layer"]) for r in rows}),
            "native_risk_sum": n_risk,
            "rotated_risk_sum": r_risk,
            "G_j": stage_gain(n_risk, r_risk),
            "native_rel_l2_rms": math.sqrt(mean([float(r["native_rel_l2"]) ** 2 for r in rows]) or 0.0),
            "rotated_rel_l2_rms": math.sqrt(mean([float(r["rotated_rel_l2"]) ** 2 for r in rows]) or 0.0),
        })
    return unit_rows


def aggregate_transfer_rows(transfer_rows):
    grouped = defaultdict(list)
    for row in transfer_rows:
        grouped[(row["unit_id"], int(row["horizon"]), row["from_stage"], row["to_stage"])].append(row)
    out = []
    for (unit_id, horizon, frm, to), rows in sorted(grouped.items()):
        out.append({
            "unit_id": unit_id,
            "horizon": horizon,
            "from_stage": frm,
            "to_stage": to,
            "n_target_layers": len({int(r["target_layer"]) for r in rows}),
            "median_Lambda": median([r["Lambda"] for r in rows]),
            "median_scale_safe_Lambda": median([r["scale_safe_Lambda"] for r in rows]),
            "wins_Lambda_gt_1": sum(float(r["scale_safe_Lambda"]) > 1.0 for r in rows if finite(r["scale_safe_Lambda"])),
        })
    return out


def summarize_stage_gains(unit_stage_rows):
    summary_rows = []
    by = defaultdict(list)
    for row in unit_stage_rows:
        by[(int(row["horizon"]), row["stage"])].append(row)
    for horizon in HORIZONS:
        for stage in STAGE_ORDER:
            rows = by.get((horizon, stage), [])
            vals = [float(r["G_j"]) for r in rows if finite(r["G_j"])]
            summary_rows.append({
                "horizon": horizon,
                "stage": stage,
                "n_units": len({r["unit_id"] for r in rows}),
                "median_G_j": median(vals),
                "mean_G_j": mean(vals),
                "std_G_j": std(vals),
                "wins_G_j_gt_0": sum(v > 0 for v in vals),
                "bootstrap95_G_j": json.dumps(bootstrap_ci(vals), sort_keys=True),
            })
    return summary_rows


def summarize_delta_g(unit_stage_rows):
    by_unit = defaultdict(dict)
    for row in unit_stage_rows:
        by_unit[(row["unit_id"], int(row["horizon"]))][row["stage"]] = row
    out = []
    for horizon in HORIZONS:
        for prev, cur in zip(STAGE_ORDER, STAGE_ORDER[1:]):
            vals = []
            for (unit_id, h), stages in by_unit.items():
                if h != horizon or prev not in stages or cur not in stages:
                    continue
                vals.append(float(stages[cur]["G_j"]) - float(stages[prev]["G_j"]))
            out.append({
                "horizon": horizon,
                "from_stage": prev,
                "to_stage": cur,
                "n_units": len(vals),
                "median_DeltaG": median(vals),
                "mean_DeltaG": mean(vals),
                "wins_DeltaG_lt_0": sum(v < 0 for v in vals),
                "bootstrap95_DeltaG": json.dumps(bootstrap_ci(vals), sort_keys=True),
            })
    return out


def summarize_transfer(unit_transfer_rows):
    by = defaultdict(list)
    for row in unit_transfer_rows:
        by[(int(row["horizon"]), row["from_stage"], row["to_stage"])].append(row)
    out = []
    for horizon in HORIZONS:
        for prev, cur in zip(STAGE_ORDER, STAGE_ORDER[1:]):
            rows = by.get((horizon, prev, cur), [])
            vals = [float(r["median_scale_safe_Lambda"]) for r in rows if finite(r["median_scale_safe_Lambda"])]
            out.append({
                "horizon": horizon,
                "from_stage": prev,
                "to_stage": cur,
                "n_units": len({r["unit_id"] for r in rows}),
                "median_Lambda": median(vals),
                "mean_Lambda": mean(vals),
                "wins_Lambda_gt_1": sum(v > 1.0 for v in vals),
                "bootstrap95_Lambda": json.dumps(bootstrap_ci(vals), sort_keys=True),
            })
    return out


def first_collapse(stage_summary_rows, delta_rows, horizon=PRIMARY_HORIZON):
    stage_by = {r["stage"]: r for r in stage_summary_rows if int(r["horizon"]) == int(horizon)}
    delta_by = {(r["from_stage"], r["to_stage"]): r for r in delta_rows if int(r["horizon"]) == int(horizon)}
    for prev, cur in zip(STAGE_ORDER, STAGE_ORDER[1:]):
        p = as_float(stage_by.get(prev, {}).get("median_G_j"))
        c = as_float(stage_by.get(cur, {}).get("median_G_j"))
        d = as_float(delta_by.get((prev, cur), {}).get("median_DeltaG"))
        wins = int(float(delta_by.get((prev, cur), {}).get("wins_DeltaG_lt_0") or 0))
        if p is None or c is None or d is None:
            continue
        if p > 0.10 and (c <= 0.03 or p - c >= 0.10) and d < 0.0 and wins >= 10:
            return cur
    logits = as_float(stage_by.get("lm_head_logits", {}).get("median_G_j"))
    if logits is not None and logits > 0.05:
        return "AFTER_SAME_TOKEN"
    return "NO_CLEAR_COLLAPSE"


def responsible_operator(stage):
    mapping = {
        "o_norm_gated_rms": "o_norm_gated_rms",
        "head_merge": "head_merge",
        "o_proj": "o_proj",
        "attention_residual_add": "attention_residual_add",
        "post_attention_layernorm": "post_attention_layernorm",
        "mlp": "mlp",
        "layer_output": "mlp_residual_block",
        "model_norm": "model_norm",
        "lm_head_logits": "lm_head",
        "AFTER_SAME_TOKEN": "NONE_WITHIN_SAME_TOKEN",
        "NO_CLEAR_COLLAPSE": "UNRESOLVED",
    }
    return mapping.get(stage, "UNRESOLVED")


def classify_logit_gain(stage_summary_rows):
    row = next((r for r in stage_summary_rows if int(r["horizon"]) == PRIMARY_HORIZON and r["stage"] == "lm_head_logits"), None)
    val = as_float(row.get("median_G_j") if row else None)
    if val is None:
        return "NOT_RUN"
    if val >= 0.30:
        return "STRONG"
    if val >= 0.10:
        return "POSITIVE"
    if val >= 0.03:
        return "WEAK"
    if val >= -0.03:
        return "NONE"
    return "REVERSED"


def classify_failure_localization(collapse_stage, delta_rows):
    if collapse_stage in {"AFTER_SAME_TOKEN", "NO_CLEAR_COLLAPSE"}:
        return "PARTIAL" if collapse_stage == "AFTER_SAME_TOKEN" else "NOT_SUPPORTED"
    row = next((r for r in delta_rows if int(r["horizon"]) == PRIMARY_HORIZON and r["to_stage"] == collapse_stage), None)
    if not row:
        return "PARTIAL"
    ci = json.loads(row.get("bootstrap95_DeltaG") or "null")
    wins = int(float(row.get("wins_DeltaG_lt_0") or 0))
    if ci and ci[1] < 0.0 and wins >= 13:
        return "SUPPORTED"
    return "PARTIAL"


def environment_info():
    info = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "expected_kda_python": KDA_PYTHON,
        "model_path": str(MODEL_PATH),
        "model_config_sha256": hashlib.sha256((MODEL_PATH / "config.json").read_bytes()).hexdigest() if (MODEL_PATH / "config.json").exists() else None,
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
    }
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception as exc:
        info["transformers"] = f"IMPORT_ERROR: {exc!r}"
    try:
        import fla
        info["fla"] = getattr(fla, "__version__", None)
        info["fla_file"] = getattr(fla, "__file__", None)
    except Exception as exc:
        info["fla"] = f"IMPORT_ERROR: {exc!r}"
    info["torch"] = torch.__version__
    info["torch_cuda"] = torch.version.cuda
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


def reset_outputs(outdir):
    for rel in [
        "readout_capture/readout_capture.jsonl",
        "stage_metrics/stage_metrics.jsonl",
        "operator_transfer/operator_transfer.jsonl",
        "conditional_closure/conditional_operator_closure.jsonl",
    ]:
        p = outdir / rel
        if p.exists():
            p.unlink()


def run_experiment(args):
    outdir = Path(args.output_dir)
    if args.overwrite:
        reset_outputs(outdir)
    for name in ["architecture_audit", "readout_capture", "stage_metrics", "operator_transfer", "conditional_closure", "figures"]:
        (outdir / name).mkdir(parents=True, exist_ok=True)

    audit = build_static_kda_post_readout_audit()
    save_json(outdir / "architecture_audit" / "kda_post_readout_path.json", audit)
    if audit["KDA_POST_READOUT_PATH_AUDIT"] != "PASS":
        raise RuntimeError("KDA post-readout path audit failed")

    env = environment_info()
    save_json(outdir / "provenance.json", env)
    units, manifest_hash, units_all = load_units(args.scope, args.max_units)
    config = {
        "task": TASK,
        "scope": args.scope,
        "horizons": HORIZONS,
        "primary_horizon": PRIMARY_HORIZON,
        "rotation": "KDA Value-side RHT seed 0",
        "formal_unit": "canonical recurrent unit",
        "expected_formal_units": EXPECTED_CANONICAL_N,
        "canonical_manifest": str(KDA_CANONICAL_UNITS),
        "canonical_manifest_sha256": manifest_hash,
        "target_layer_mode": "layerwise",
        "model_path": str(MODEL_PATH),
    }
    save_json(outdir / "config.json", config)

    mod, torch_mod, model, tokenizer, kda_layers, rows_by_pid, teacher_tokens = load_kda_context()
    if args.target_layer_limit:
        kda_layers = list(kda_layers)[: int(args.target_layer_limit)]
    stage_rows = []
    transfer_rows = []
    readout_rows_all = []
    readout_unit_rows = []
    closure_rows = []
    failures = []
    unit_done = set()
    provenance_rows = []
    try:
        rotation = exact().make_experiment_rotation("kda", 128, "rht", seed=RHT_SEED, dtype=torch.float32)
        for idx, unit in enumerate(units, 1):
            print(f"[{now()}] KDA post-readout unit {idx}/{len(units)} {unit['unit_id']}", flush=True)
            exact_capture = None
            probe = None
            try:
                pid = str(unit["problem_id"])
                item = rows_by_pid.get(pid)
                tokens = [int(x) for x in teacher_tokens.get(pid, unit.get("teacher_forced_token_ids") or [])]
                if item is None or len(tokens) <= int(unit["t0"]) + max(HORIZONS):
                    raise RuntimeError(f"missing prompt/tokens for {unit['unit_id']}")

                exact_capture = mod.P.KdaTransitionCapture()
                exact_capture.install(model)
                ctx_exact = (mod, torch_mod, model, tokenizer, exact_capture, kda_layers, rows_by_pid, teacher_tokens)
                tape = exact().capture_driver_tape("kda", unit, ctx_exact, max(HORIZONS))
                err_by_h, readout_rows = replay_error_stacks(tape, exact_capture, rotation)
                exact_capture.close()
                exact_capture = None

                probe = KdaPostReadoutProbe()
                installed_layers = probe.install(model)
                missing_layers = sorted(set(map(int, kda_layers)) - set(map(int, installed_layers)))
                if missing_layers:
                    raise RuntimeError(f"post-readout probe missing KDA layers: {missing_layers}")
                for row in readout_rows:
                    append_jsonl(outdir / "readout_capture" / "readout_capture.jsonl", row)
                readout_rows_all.extend(readout_rows)
                for horizon in HORIZONS:
                    if horizon in err_by_h:
                        rrow = dict(err_by_h[horizon]["readout_row"])
                        readout_unit_rows.append(rrow)

                for horizon in HORIZONS:
                    context = prepare_clean_context(mod, torch_mod, model, tokenizer, unit, item, tokens, horizon)
                    original_cache_hash = stack_hash(get_kda_stack(mod, context["past"], kda_layers))
                    for layer in kda_layers:
                        if layer not in err_by_h[horizon]["delta_N"] or layer not in err_by_h[horizon]["delta_R"]:
                            continue
                        zero = torch.zeros_like(err_by_h[horizon]["delta_N"][layer])
                        fp_out, fp_records = run_branch_forward(torch_mod, model, probe, context, f"FP_{unit['unit_id']}_{horizon}_{layer}", layer, zero)
                        zero_out, zero_records = run_branch_forward(torch_mod, model, probe, context, f"ZERO_{unit['unit_id']}_{horizon}_{layer}", layer, zero)
                        n_out, n_records = run_branch_forward(torch_mod, model, probe, context, f"N_{unit['unit_id']}_{horizon}_{layer}", layer, err_by_h[horizon]["delta_N"][layer])
                        r_out, r_records = run_branch_forward(torch_mod, model, probe, context, f"R_{unit['unit_id']}_{horizon}_{layer}", layer, err_by_h[horizon]["delta_R"][layer])
                        after_cache_hash = stack_hash(get_kda_stack(mod, context["past"], kda_layers))
                        cache_gate = cache_modification_gate(original_cache_hash, after_cache_hash)
                        zero_gate = zero_perturbation_noninterference(clean_stage_subset(fp_records), clean_stage_subset(zero_records))
                        if cache_gate != "PASS" or zero_gate["gate"] != "PASS":
                            raise RuntimeError(f"noninterference/cache gate failed unit={unit['unit_id']} h={horizon} layer={layer}: {cache_gate} {zero_gate}")

                        n_prov = {
                            "unit_id": str(unit["unit_id"]),
                            "t": int(horizon),
                            "token_id": int(context["token_id"]),
                            "layer": int(layer),
                            "q_sha256": err_by_h[horizon]["q_sha256"][layer],
                            "cache_sha256": original_cache_hash,
                            "branch": "native",
                            "delta_o_sha256": tensor_hash(err_by_h[horizon]["delta_N"][layer]),
                        }
                        r_prov = dict(n_prov, branch="rotated", delta_o_sha256=tensor_hash(err_by_h[horizon]["delta_R"][layer]))
                        prov_gate = check_matched_pair_provenance(n_prov, r_prov)
                        if prov_gate["gate"] != "PASS":
                            raise RuntimeError(f"provenance gate failed unit={unit['unit_id']} h={horizon} layer={layer}: {prov_gate}")
                        provenance_rows.append({**n_prov, "provenance_gate": prov_gate["gate"]})

                        rows, tensors = stage_pair_metrics(str(unit["unit_id"]), horizon, layer, clean_stage_subset(fp_records), clean_stage_subset(n_records), clean_stage_subset(r_records))
                        trs = transfer_rows_from_tensors(str(unit["unit_id"]), horizon, layer, tensors)
                        for row in rows:
                            row["cache_nonmodification_gate"] = cache_gate
                            row["zero_noninterference_gate"] = zero_gate["gate"]
                            append_jsonl(outdir / "stage_metrics" / "stage_metrics.jsonl", row)
                        for row in trs:
                            append_jsonl(outdir / "operator_transfer" / "operator_transfer.jsonl", row)
                        stage_rows.extend(rows)
                        transfer_rows.extend(trs)
                    del context
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                unit_done.add(str(unit["unit_id"]))
            except Exception as exc:
                failure = {"unit": unit, "error": repr(exc), "traceback": traceback.format_exc(limit=20), "timestamp": now()}
                failures.append(failure)
                save_json(outdir / "failures.json", failures)
                print(f"[{now()}] FAILED unit={unit.get('unit_id')} {exc!r}", flush=True)
                if not args.keep_going:
                    raise
            finally:
                if probe is not None:
                    probe.close()
                if exact_capture is not None:
                    exact_capture.close()
    finally:
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    write_rows(outdir / "provenance_pairs.csv", provenance_rows)
    write_rows(outdir / "readout_capture" / "readout_unit_aggregate.csv", readout_unit_rows)
    unit_stage_rows = aggregate_stage_rows(stage_rows)
    unit_transfer_rows = aggregate_transfer_rows(transfer_rows)
    stage_summary_rows = summarize_stage_gains(unit_stage_rows)
    delta_rows = summarize_delta_g(unit_stage_rows)
    transfer_summary_rows = summarize_transfer(unit_transfer_rows)
    write_rows(outdir / "stage_metrics" / "stage_metrics_unit_aggregate.csv", unit_stage_rows)
    write_rows(outdir / "stage_metrics" / "stage_gain_summary.csv", stage_summary_rows)
    write_rows(outdir / "stage_metrics" / "delta_gain_summary.csv", delta_rows)
    write_rows(outdir / "operator_transfer" / "operator_transfer_unit_aggregate.csv", unit_transfer_rows)
    write_rows(outdir / "operator_transfer" / "operator_transfer_summary.csv", transfer_summary_rows)

    collapse = first_collapse(stage_summary_rows, delta_rows)
    operator = responsible_operator(collapse)
    if collapse == "o_proj":
        closure = "SUPPORTED" if any(r["to_stage"] == "o_proj" and int(r["horizon"]) == PRIMARY_HORIZON and as_float(r.get("median_Lambda"), 1.0) > 1.0 for r in transfer_summary_rows) else "PARTIAL"
    elif collapse in {"AFTER_SAME_TOKEN", "NO_CLEAR_COLLAPSE"}:
        closure = "NOT_SUPPORTED"
    else:
        closure = "PARTIAL"
    closure_rows.append({
        "first_collapse_stage": collapse,
        "responsible_operator": operator,
        "operator_specific_closure": closure,
        "status": "CONDITIONAL_SUMMARY",
    })
    append_jsonl(outdir / "conditional_closure" / "conditional_operator_closure.jsonl", closure_rows[0])
    write_rows(outdir / "conditional_closure" / "conditional_operator_closure_unit_aggregate.csv", closure_rows)

    horizon_rows = []
    readout_by_h = defaultdict(list)
    for row in readout_unit_rows:
        readout_by_h[int(row["horizon"])].append(float(row["G_read"]))
    for horizon in HORIZONS:
        vals = readout_by_h[horizon]
        logits_row = next((r for r in stage_summary_rows if int(r["horizon"]) == horizon and r["stage"] == "lm_head_logits"), {})
        horizon_rows.append({
            "horizon": horizon,
            "n_units": len(vals),
            "median_core_readout_gain": median(vals),
            "mean_core_readout_gain": mean(vals),
            "wins_core_readout_gain_gt_0": sum(v > 0 for v in vals),
            "median_logit_gain": logits_row.get("median_G_j"),
            "collapse_stage": first_collapse(stage_summary_rows, delta_rows, horizon=horizon),
        })
    write_rows(outdir / "horizon_summary.csv", horizon_rows)

    t64_read_vals = readout_by_h[PRIMARY_HORIZON]
    readout_reproduced = (
        len(t64_read_vals) == len(units)
        and median(t64_read_vals) is not None
        and median(t64_read_vals) >= 0.30
    )
    localization = classify_failure_localization(collapse, delta_rows)
    logit_gain_cls = classify_logit_gain(stage_summary_rows)
    if collapse == "AFTER_SAME_TOKEN":
        same_token_retention = "GAIN_PRESERVED"
        final_cls = "KDA_SAME_TOKEN_GAIN_PRESERVED_CLOSED_LOOP_FAILURE_LIKELY"
        next_exp = "KDA_ROTATION_CLOSED_LOOP_DRIVER_DRIFT_CAUSAL_V1"
        mechanism = "SUFFICIENT_FOR_NEXT_STAGE"
    elif collapse == "NO_CLEAR_COLLAPSE":
        same_token_retention = "MIXED_OR_NOISY"
        final_cls = "KDA_POST_READOUT_LOCALIZATION_INCONCLUSIVE"
        next_exp = "KDA_POST_READOUT_GAIN_RETENTION_REASSESSMENT_V2"
        mechanism = "NOT_CLOSED"
    else:
        same_token_retention = "COLLAPSES_WITHIN_SAME_TOKEN"
        final_cls = f"KDA_POST_READOUT_GAIN_COLLAPSE_AT_{collapse}"
        next_exp = f"KDA_{operator.upper()}_CAUSAL_CLOSURE_V1"
        mechanism = "SUFFICIENT_FOR_NEXT_STAGE" if localization == "SUPPORTED" else "PARTIAL"

    summary = {
        "task": TASK,
        "KDA_POST_READOUT_PATH_AUDIT": audit["KDA_POST_READOUT_PATH_AUDIT"],
        "KDA_FROZEN_READOUT_GAIN_REPRODUCED": "YES" if readout_reproduced else "NO",
        "KDA_CORE_READOUT_GAIN": median(t64_read_vals),
        "KDA_STAGE_GAIN_RETENTION": same_token_retention,
        "KDA_FIRST_POST_READOUT_GAIN_COLLAPSE_STAGE": collapse,
        "KDA_RESPONSIBLE_OPERATOR": operator,
        "KDA_OPERATOR_CONTEXT": "BRANCH_DEPENDENT_AFTER_TARGET_RESIDUAL; TARGET_GATE_FIXED",
        "KDA_OPERATOR_SPECIFIC_CLOSURE": closure,
        "KDA_SAME_TOKEN_LOGIT_ROTATION_GAIN": logit_gain_cls,
        "KDA_POST_READOUT_FAILURE_LOCALIZATION": localization,
        "KDA_MECHANISM_CLOSURE_STATUS": mechanism,
        "FINAL_CLASSIFICATION": final_cls,
        "NEXT_EXPERIMENT": next_exp,
        "n_units_completed": len(unit_done),
        "expected_units": len(units),
        "kda_layers": list(map(int, kda_layers)),
        "n_stage_rows": len(stage_rows),
        "n_transfer_rows": len(transfer_rows),
        "failures": failures,
        "environment": env,
    }
    save_json(outdir / "summary.json", summary)
    save_json(outdir / "manifest.json", {
        "task": TASK,
        "created_at": now(),
        "artifact_paths": [
            str(outdir / "config.json"),
            str(outdir / "manifest.json"),
            str(outdir / "provenance.json"),
            str(outdir / "architecture_audit" / "kda_post_readout_path.json"),
            str(outdir / "readout_capture" / "readout_capture.jsonl"),
            str(outdir / "readout_capture" / "readout_unit_aggregate.csv"),
            str(outdir / "stage_metrics" / "stage_metrics.jsonl"),
            str(outdir / "stage_metrics" / "stage_metrics_unit_aggregate.csv"),
            str(outdir / "stage_metrics" / "stage_gain_summary.csv"),
            str(outdir / "operator_transfer" / "operator_transfer.jsonl"),
            str(outdir / "operator_transfer" / "operator_transfer_unit_aggregate.csv"),
            str(outdir / "conditional_closure" / "conditional_operator_closure.jsonl"),
            str(outdir / "conditional_closure" / "conditional_operator_closure_unit_aggregate.csv"),
            str(outdir / "horizon_summary.csv"),
            str(outdir / "summary.json"),
            str(outdir / "report.md"),
        ],
    })
    make_plots(outdir, stage_summary_rows, delta_rows, transfer_summary_rows)
    write_report(outdir, summary, audit, horizon_rows, stage_summary_rows, delta_rows, transfer_summary_rows)
    return summary


def make_plots(outdir, stage_summary_rows, delta_rows, transfer_summary_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        save_json(outdir / "figures" / "manifest.json", {"status": "SKIPPED", "reason": repr(exc), "plots": []})
        return
    made = []
    idx = {s: i for i, s in enumerate(STAGE_ORDER)}
    for horizon in HORIZONS:
        rows = [r for r in stage_summary_rows if int(r["horizon"]) == horizon]
        rows = sorted(rows, key=lambda r: idx.get(r["stage"], 999))
        xs = [r["stage"] for r in rows]
        ys = [as_float(r["median_G_j"], 0.0) for r in rows]
        plt.figure(figsize=(9, 3))
        plt.plot(range(len(xs)), ys, marker="o")
        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(range(len(xs)), xs, rotation=35, ha="right", fontsize=7)
        plt.ylabel("median G_j")
        plt.title(f"KDA gain retention t{horizon}")
        plt.tight_layout()
        p = outdir / "figures" / f"stage_gain_t{horizon}.png"
        plt.savefig(p, dpi=160)
        plt.close()
        made.append(str(p))
    # 1. Verified downstream path.
    plt.figure(figsize=(11, 2.8))
    plt.plot(range(len(STAGE_ORDER)), [0] * len(STAGE_ORDER), marker="o", linewidth=1)
    for i, stage in enumerate(STAGE_ORDER):
        plt.text(i, 0.04 if i % 2 == 0 else -0.08, stage, rotation=35, ha="right", va="center", fontsize=7)
    plt.yticks([])
    plt.xticks([])
    plt.title("KDA verified downstream path")
    plt.tight_layout()
    p = outdir / "figures" / "verified_downstream_path.png"
    plt.savefig(p, dpi=160)
    plt.close()
    made.append(str(p))

    # 6. Wins/18 per stage at t64.
    rows = [r for r in stage_summary_rows if int(r["horizon"]) == PRIMARY_HORIZON]
    rows = sorted(rows, key=lambda r: idx.get(r["stage"], 999))
    plt.figure(figsize=(8, 3))
    plt.bar(range(len(rows)), [int(float(r.get("wins_G_j_gt_0") or 0)) for r in rows])
    plt.axhline(EXPECTED_CANONICAL_N / 2, color="black", linewidth=1)
    plt.xticks(range(len(rows)), [r["stage"] for r in rows], rotation=35, ha="right", fontsize=7)
    plt.ylabel("wins / 18")
    plt.title("Stage-wise positive-gain wins at t64")
    plt.tight_layout()
    p = outdir / "figures" / "wins_per_stage_t64.png"
    plt.savefig(p, dpi=160)
    plt.close()
    made.append(str(p))

    # 7. Core readout to logits gain curve at primary horizon.
    plt.figure(figsize=(8, 3))
    plt.plot(range(len(rows)), [as_float(r.get("median_G_j"), 0.0) for r in rows], marker="o")
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(range(len(rows)), [r["stage"] for r in rows], rotation=35, ha="right", fontsize=7)
    plt.ylabel("median G_j")
    plt.title("Core readout to same-token logits gain curve")
    plt.tight_layout()
    p = outdir / "figures" / "core_to_logits_gain_curve_t64.png"
    plt.savefig(p, dpi=160)
    plt.close()
    made.append(str(p))

    # 8. First collapse visualization.
    deltas = [r for r in delta_rows if int(r["horizon"]) == PRIMARY_HORIZON]
    if deltas:
        min_i = min(range(len(deltas)), key=lambda i: as_float(deltas[i].get("median_DeltaG"), 0.0))
        colors = ["#4c78a8"] * len(deltas)
        colors[min_i] = "#d62728"
        labels = [f"{r['from_stage']}->{r['to_stage']}" for r in deltas]
        vals = [as_float(r.get("median_DeltaG"), 0.0) for r in deltas]
        plt.figure(figsize=(9, 3))
        plt.bar(range(len(vals)), vals, color=colors)
        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(range(len(vals)), labels, rotation=35, ha="right", fontsize=6)
        plt.ylabel("median DeltaG")
        plt.title("First collapse stage visualization")
        plt.tight_layout()
        p = outdir / "figures" / "first_collapse_stage_t64.png"
        plt.savefig(p, dpi=160)
        plt.close()
        made.append(str(p))

    # 9. Same-token logit Native vs Rotated perturbation proxy.
    unit_rows = read_rows(outdir / "stage_metrics" / "stage_metrics_unit_aggregate.csv")
    logit_rows = [r for r in unit_rows if int(float(r.get("horizon") or 0)) == PRIMARY_HORIZON and r.get("stage") == "lm_head_logits"]
    if logit_rows:
        xs = [as_float(r.get("native_rel_l2_rms"), 0.0) for r in logit_rows]
        ys = [as_float(r.get("rotated_rel_l2_rms"), 0.0) for r in logit_rows]
        plt.figure(figsize=(4, 4))
        plt.scatter(xs, ys, s=18)
        lim = max(xs + ys + [1e-12])
        plt.plot([0, lim], [0, lim], color="black", linewidth=1)
        plt.xlabel("Native logit rel-L2")
        plt.ylabel("Rotated logit rel-L2")
        plt.title("Same-token logit perturbation")
        plt.tight_layout()
        p = outdir / "figures" / "same_token_logit_native_vs_rotated_t64.png"
        plt.savefig(p, dpi=160)
        plt.close()
        made.append(str(p))

    # 10. Responsible operator scatter.
    raw_rows = read_rows(outdir / "stage_metrics" / "stage_metrics_unit_aggregate.csv")
    op_rows = [
        r for r in raw_rows
        if int(float(r.get("horizon") or 0)) == PRIMARY_HORIZON and r.get("stage") in {"core_readout", "o_norm_gated_rms"}
    ]
    by_unit = defaultdict(dict)
    for r in op_rows:
        by_unit[r["unit_id"]][r["stage"]] = r
    xs, ys = [], []
    for stages in by_unit.values():
        if "core_readout" in stages and "o_norm_gated_rms" in stages:
            xs.append(as_float(stages["core_readout"].get("G_j"), 0.0))
            ys.append(as_float(stages["o_norm_gated_rms"].get("G_j"), 0.0))
    if xs:
        plt.figure(figsize=(4, 4))
        plt.scatter(xs, ys, s=18)
        plt.axhline(0, color="black", linewidth=1)
        plt.axvline(0, color="black", linewidth=1)
        plt.xlabel("Core readout G_j")
        plt.ylabel("o_norm G_j")
        plt.title("Responsible operator scatter")
        plt.tight_layout()
        p = outdir / "figures" / "responsible_operator_scatter_t64.png"
        plt.savefig(p, dpi=160)
        plt.close()
        made.append(str(p))

    rows = [r for r in delta_rows if int(r["horizon"]) == PRIMARY_HORIZON]
    labels = [f"{r['from_stage']}->{r['to_stage']}" for r in rows]
    vals = [as_float(r["median_DeltaG"], 0.0) for r in rows]
    plt.figure(figsize=(9, 3))
    plt.bar(range(len(vals)), vals)
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(range(len(vals)), labels, rotation=35, ha="right", fontsize=6)
    plt.ylabel("median DeltaG")
    plt.tight_layout()
    p = outdir / "figures" / "deltaG_t64.png"
    plt.savefig(p, dpi=160)
    plt.close()
    made.append(str(p))
    rows = [r for r in transfer_summary_rows if int(r["horizon"]) == PRIMARY_HORIZON]
    labels = [f"{r['from_stage']}->{r['to_stage']}" for r in rows]
    vals = [as_float(r["median_Lambda"], 1.0) for r in rows]
    plt.figure(figsize=(9, 3))
    plt.bar(range(len(vals)), vals)
    plt.axhline(1, color="black", linewidth=1)
    plt.xticks(range(len(vals)), labels, rotation=35, ha="right", fontsize=6)
    plt.ylabel("median scale-safe Lambda")
    plt.tight_layout()
    p = outdir / "figures" / "lambda_t64.png"
    plt.savefig(p, dpi=160)
    plt.close()
    made.append(str(p))
    save_json(outdir / "figures" / "manifest.json", {"status": "OK", "plots": made})


def markdown_table(rows, fields, limit=None):
    rows = rows[:limit] if limit else rows
    out = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        out.append("|" + "|".join(str(row.get(f, "")) for f in fields) + "|")
    return "\n".join(out)


def write_report(outdir, summary, audit, horizon_rows, stage_summary_rows, delta_rows, transfer_summary_rows):
    lines = [
        f"# {TASK}",
        "",
        "## Classification",
        "",
        "```text",
        f"KDA_POST_READOUT_PATH_AUDIT = {summary['KDA_POST_READOUT_PATH_AUDIT']}",
        f"KDA_FROZEN_READOUT_GAIN_REPRODUCED = {summary['KDA_FROZEN_READOUT_GAIN_REPRODUCED']}",
        f"KDA_CORE_READOUT_GAIN = {summary['KDA_CORE_READOUT_GAIN']}",
        f"KDA_STAGE_GAIN_RETENTION = {summary['KDA_STAGE_GAIN_RETENTION']}",
        f"KDA_FIRST_POST_READOUT_GAIN_COLLAPSE_STAGE = {summary['KDA_FIRST_POST_READOUT_GAIN_COLLAPSE_STAGE']}",
        f"KDA_RESPONSIBLE_OPERATOR = {summary['KDA_RESPONSIBLE_OPERATOR']}",
        f"KDA_OPERATOR_CONTEXT = {summary['KDA_OPERATOR_CONTEXT']}",
        f"KDA_OPERATOR_SPECIFIC_CLOSURE = {summary['KDA_OPERATOR_SPECIFIC_CLOSURE']}",
        f"KDA_SAME_TOKEN_LOGIT_ROTATION_GAIN = {summary['KDA_SAME_TOKEN_LOGIT_ROTATION_GAIN']}",
        f"KDA_POST_READOUT_FAILURE_LOCALIZATION = {summary['KDA_POST_READOUT_FAILURE_LOCALIZATION']}",
        f"KDA_MECHANISM_CLOSURE_STATUS = {summary['KDA_MECHANISM_CLOSURE_STATUS']}",
        f"FINAL_CLASSIFICATION = {summary['FINAL_CLASSIFICATION']}",
        f"NEXT_EXPERIMENT = {summary['NEXT_EXPERIMENT']}",
        "```",
        "",
        f"Canonical units completed: {summary['n_units_completed']} / {summary['expected_units']}.",
        "",
        "## Verified Path",
        "",
        " -> ".join([stage["stage_name"] for stage in audit["stages"]]),
        "",
        "## Horizon Summary",
        "",
        markdown_table(horizon_rows, ["horizon", "n_units", "median_core_readout_gain", "median_logit_gain", "collapse_stage"]),
        "",
        "## Stage Gain Summary",
        "",
        markdown_table([r for r in stage_summary_rows if int(r["horizon"]) == PRIMARY_HORIZON], ["stage", "n_units", "median_G_j", "wins_G_j_gt_0", "bootstrap95_G_j"]),
        "",
        "## DeltaG Summary",
        "",
        markdown_table([r for r in delta_rows if int(r["horizon"]) == PRIMARY_HORIZON], ["from_stage", "to_stage", "median_DeltaG", "wins_DeltaG_lt_0", "bootstrap95_DeltaG"]),
        "",
        "## Lambda Summary",
        "",
        markdown_table([r for r in transfer_summary_rows if int(r["horizon"]) == PRIMARY_HORIZON], ["from_stage", "to_stage", "median_Lambda", "wins_Lambda_gt_1", "bootstrap95_Lambda"]),
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(summary["environment"], indent=2, sort_keys=True),
        "```",
    ]
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scope", choices=["smoke", "pilot", "formal"], default="formal")
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--target-layer-limit", type=int, default=None)
    p.add_argument("--output-dir", default=str(RESULT_DIR))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = run_experiment(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    formal_gate_applies = (
        args.scope == "formal"
        and args.max_units is None
        and args.target_layer_limit is None
        and summary.get("expected_units") == EXPECTED_CANONICAL_N
    )
    if formal_gate_applies and summary.get("KDA_FROZEN_READOUT_GAIN_REPRODUCED") != "YES":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
