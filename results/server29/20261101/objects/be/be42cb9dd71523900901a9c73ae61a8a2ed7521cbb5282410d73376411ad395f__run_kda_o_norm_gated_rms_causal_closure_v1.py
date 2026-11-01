#!/usr/bin/env python3
import argparse
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


TASK = "KDA_O_NORM_GATED_RMS_CAUSAL_CLOSURE_V1"
SLUG = "kda_o_norm_gated_rms_causal_closure_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
POST_RUNNER = REPO / "experiments" / "rotation" / "run_kda_rotation_post_readout_gain_retention_v1.py"
KDA_PYTHON = "/data01/user2/.conda/envs/ling-kda/bin/python"
MODEL_PATH = Path(os.environ.get("LING_MODEL_PATH", "/data01/user2/zypan_ling_dist/models/Ling-3.0-tiny"))
KDA_CANONICAL_UNITS = REPO / "runs" / "ling_kda_persistent_error_decomposition_causal_v1" / "canonical_units.json"
HORIZONS = [16, 32, 64]
PRIMARY_HORIZON = 64
EXPECTED_CANONICAL_N = 18
RHT_SEED = 0
EPS = 1e-12
BOOTSTRAP_SEED = 20260914
IDENTITY_TOL = 1e-8

INTERNAL_STAGE_ORDER = [
    "core_readout",
    "pure_rms",
    "learned_scale",
    "effective_gate",
    "module_output",
]

DOWNSTREAM_STAGE_ORDER = [
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


_POST = None


def post():
    global _POST
    if _POST is None:
        _POST = import_file(POST_RUNNER, "kda_post_readout_for_o_norm_closure_v1")
    return _POST


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


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


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def percentile(xs, q):
    vals = sorted(float(x) for x in xs if finite(x))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


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


def relative_l2(fp, delta, eps=EPS):
    return tensor_norm(delta) / (tensor_norm(fp) + eps)


def tensor_hash(x):
    y = x.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(y.shape)).encode("ascii"))
    h.update(str(y.dtype).encode("ascii"))
    if y.dtype in {torch.float16, torch.bfloat16}:
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


def stage_gain_from_norms(native_norm, rotated_norm, eps=EPS):
    r = float(rotated_norm) / (float(native_norm) + eps)
    return 1.0 - r * r


def atomic_stage_metric(fp, native_delta, rotated_delta, eps=EPS):
    del fp
    n = tensor_norm(native_delta)
    r = tensor_norm(rotated_delta)
    ratio = r / (n + eps)
    return {
        "native_delta_norm": n,
        "rotated_delta_norm": r,
        "r_j": ratio,
        "G_j": 1.0 - ratio * ratio,
    }


def atomic_transfer_metric(prev_metric, cur_metric, eps=EPS):
    prev_r = float(prev_metric["r_j"])
    cur_r = float(cur_metric["r_j"])
    lam = cur_r / prev_r if abs(prev_r) > eps else cur_r / (prev_r + eps)
    lhs = 1.0 - float(cur_metric["G_j"])
    rhs = lam * lam * (1.0 - float(prev_metric["G_j"]))
    abs_err = abs(lhs - rhs)
    return {
        "g_native": float(cur_metric["native_delta_norm"]) / (float(prev_metric["native_delta_norm"]) + eps),
        "g_rotated": float(cur_metric["rotated_delta_norm"]) / (float(prev_metric["rotated_delta_norm"]) + eps),
        "Lambda_j": lam,
        "identity_lhs": lhs,
        "identity_rhs": rhs,
        "identity_abs_error": abs_err,
        "identity_rel_error": abs_err / (abs(lhs) + eps),
    }


def aggregation_reconciliation(prev_rows, cur_rows, eps=EPS):
    cur_by = {(r.get("unit_id"), int(r.get("horizon", 0))): r for r in cur_rows}
    lambdas = []
    paired_prev_r = []
    paired_cur_r = []
    for row in prev_rows:
        key = (row.get("unit_id"), int(row.get("horizon", 0)))
        cur = cur_by.get(key)
        if not cur:
            continue
        pr = float(row["r_j"])
        cr = float(cur["r_j"])
        paired_prev_r.append(pr)
        paired_cur_r.append(cr)
        lambdas.append(cr / (pr + eps))
    med_prev_r = median(paired_prev_r)
    med_cur_r = median(paired_cur_r)
    return {
        "median_atomic_G_prev": median([r["G_j"] for r in prev_rows]),
        "median_atomic_G_cur": median([r["G_j"] for r in cur_rows]),
        "G_prev_from_median_r": None if med_prev_r is None else 1.0 - med_prev_r * med_prev_r,
        "G_cur_from_median_r": None if med_cur_r is None else 1.0 - med_cur_r * med_cur_r,
        "median_atomic_Lambda": median(lambdas),
        "Lambda_from_median_r": None if med_prev_r in (None, 0.0) or med_cur_r is None else med_cur_r / (med_prev_r + eps),
        "n_pairs": len(lambdas),
    }


def _rms(x, eps=EPS):
    xd = x.detach().double()
    return torch.sqrt(torch.mean(xd * xd, dim=-1, keepdim=True) + float(eps))


def _gate_multiplier(g, activation):
    if activation == "sigmoid":
        return torch.sigmoid(g)
    if activation in {"swish", "silu"}:
        return g * torch.sigmoid(g)
    raise ValueError(f"unsupported o_norm gate activation: {activation}")


def compute_o_norm_internal_stages(x, gate_logits, weight, activation="sigmoid", eps=EPS):
    out_dtype = x.dtype
    xd = x.detach().double()
    gd = gate_logits.detach().double()
    pure = xd / _rms(xd, eps=eps)
    if weight is None:
        learned = pure
    else:
        w = weight.detach().double()
        view = [1] * pure.ndim
        view[-1] = int(w.numel())
        learned = pure * w.view(*view)
    gate = _gate_multiplier(gd, activation)
    out = learned * gate
    return {
        "input_core_readout": xd.to(dtype=out_dtype),
        "pure_rms": pure.to(dtype=out_dtype),
        "learned_scale": learned.to(dtype=out_dtype),
        "effective_gate": gate.to(dtype=out_dtype),
        "module_output": out.to(dtype=out_dtype),
    }


def frozen_denominator_o_norm_stages(x, delta, gate_logits, weight, activation="sigmoid", eps=EPS):
    xd = x.detach().double()
    dd = delta.detach().double()
    pure = (xd + dd) / _rms(xd, eps=eps)
    if weight is None:
        learned = pure
    else:
        w = weight.detach().double()
        view = [1] * pure.ndim
        view[-1] = int(w.numel())
        learned = pure * w.view(*view)
    gate = _gate_multiplier(gate_logits.detach().double(), activation)
    return {
        "frozen_rms": pure,
        "frozen_learned_scale": learned,
        "frozen_module_output": learned * gate,
    }


def radial_tangential_decomposition(x, delta, eps=EPS):
    out_dtype = delta.dtype
    xd = x.detach().double()
    dd = delta.detach().double()
    dot = torch.sum(xd * dd, dim=-1, keepdim=True)
    x2 = torch.sum(xd * xd, dim=-1, keepdim=True)
    parallel = dot / (x2 + eps) * xd
    tangential = dd - parallel
    pn2 = float(torch.sum(parallel * parallel).item())
    tn2 = float(torch.sum(tangential * tangential).item())
    total = pn2 + tn2 + eps
    return {
        "parallel": parallel.to(dtype=out_dtype),
        "tangential": tangential.to(dtype=out_dtype),
        "radial_fraction": pn2 / total,
        "tangential_fraction": tn2 / total,
        "delta_norm": tensor_norm(dd),
        "parallel_norm": tensor_norm(parallel),
        "tangential_norm": tensor_norm(tangential),
    }


def rms_normalize(x, eps=EPS):
    return x.detach().double() / _rms(x, eps=eps)


def rms_jacobian_apply(x, delta, eps=EPS):
    xd = x.detach().double()
    dd = delta.detach().double()
    d = xd.shape[-1]
    rms = _rms(xd, eps=eps)
    dot = torch.sum(xd * dd, dim=-1, keepdim=True)
    return dd / rms - xd * dot / (float(d) * (rms ** 3))


def rms_jacobian_audit(x, native_delta, rotated_delta, eps=EPS):
    x = x.detach().double()
    dn = native_delta.detach().double()
    dr = rotated_delta.detach().double()
    jn = rms_jacobian_apply(x, dn, eps=eps)
    jr = rms_jacobian_apply(x, dr, eps=eps)
    exact_n = rms_normalize(x + dn, eps=eps) - rms_normalize(x, eps=eps)
    exact_r = rms_normalize(x + dr, eps=eps) - rms_normalize(x, eps=eps)
    g_n = tensor_norm(jn) / (tensor_norm(dn) + eps)
    g_r = tensor_norm(jr) / (tensor_norm(dr) + eps)
    return {
        "g_native_J": g_n,
        "g_rotated_J": g_r,
        "Lambda_RMS_J": g_r / (g_n + eps),
        "native_linearization_relative_error": tensor_norm(jn - exact_n) / (tensor_norm(exact_n) + eps),
        "rotated_linearization_relative_error": tensor_norm(jr - exact_r) / (tensor_norm(exact_r) + eps),
    }


def frozen_denominator_rms_delta(x, native_delta, rotated_delta, eps=EPS):
    x = x.detach().double()
    dn = native_delta.detach().double()
    dr = rotated_delta.detach().double()
    denom = _rms(x, eps=eps)
    n_out = dn / denom
    r_out = dr / denom
    dn_norm = tensor_norm(dn)
    dr_norm = tensor_norm(dr)
    g_n = tensor_norm(n_out) / dn_norm if dn_norm > eps else tensor_norm(n_out) / (dn_norm + eps)
    g_r = tensor_norm(r_out) / dr_norm if dr_norm > eps else tensor_norm(r_out) / (dr_norm + eps)
    return {
        "g_native_frozen": g_n,
        "g_rotated_frozen": g_r,
        "Lambda_frozen": g_r / g_n if abs(g_n) > eps else g_r / (g_n + eps),
        "native_output_norm": tensor_norm(n_out),
        "rotated_output_norm": tensor_norm(r_out),
    }


def isotropic_multiplier(multiplier):
    out_dtype = multiplier.dtype
    m = multiplier.detach().double()
    c = torch.sqrt(torch.mean(m * m))
    return (torch.ones_like(m) * c).to(dtype=out_dtype)


def diagonal_transfer(native_delta, rotated_delta, multiplier, eps=EPS):
    m = multiplier.detach().double()
    view = [1] * native_delta.ndim
    view[-1] = int(m.numel())
    n0 = tensor_norm(native_delta)
    r0 = tensor_norm(rotated_delta)
    n1 = tensor_norm(native_delta.detach().double() * m.view(*view))
    r1 = tensor_norm(rotated_delta.detach().double() * m.view(*view))
    g_n = n1 / (n0 + eps)
    g_r = r1 / (r0 + eps)
    return {"g_native": g_n, "g_rotated": g_r, "Lambda": g_r / (g_n + eps)}


def zero_perturbation_noninterference(fp_stages, zero_stages, eps=EPS):
    max_abs = 0.0
    rels = []
    for key, fp in fp_stages.items():
        if key not in zero_stages or not torch.is_tensor(fp) or not torch.is_tensor(zero_stages[key]):
            continue
        diff = zero_stages[key].detach().double() - fp.detach().double()
        max_abs = max(max_abs, float(diff.abs().max().item()))
        rels.append(tensor_norm(diff) / (tensor_norm(fp) + eps))
    max_rel = max(rels, default=0.0)
    return {
        "gate": "PASS" if max_abs <= 1e-7 and max_rel <= 1e-7 else "FAIL",
        "max_abs_error": max_abs,
        "max_relative_l2": max_rel,
    }


def check_matched_context(native, rotated):
    differing = []
    for key in sorted(set(native) | set(rotated)):
        if native.get(key) != rotated.get(key):
            differing.append(key)
    allowed = {"branch", "delta_o_sha256"}
    bad = [key for key in differing if key not in allowed]
    return {"gate": "PASS" if not bad else "FAIL", "differing_fields": differing, "bad_fields": bad}


def environment_info():
    info = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "expected_kda_python": KDA_PYTHON,
        "model_path": str(MODEL_PATH),
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
    if (MODEL_PATH / "config.json").exists():
        info["model_config_sha256"] = hashlib.sha256((MODEL_PATH / "config.json").read_bytes()).hexdigest()
    return info


class KdaONormProbe:
    def __init__(self):
        self.branch = None
        self.target_layer = None
        self.injections = {}
        self.overrides = {}
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
            self.handles.append(attn.o_norm.register_forward_pre_hook(self._make_o_norm_pre(idx)))
            self.handles.append(attn.o_norm.register_forward_hook(self._make_o_norm_post(idx)))
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
        self.globals["chunk_kda"] = self._wrap_backend(self.orig_chunk)
        self.globals["fused_recurrent_kda"] = self._wrap_backend(self.orig_fused)
        return list(self.kda_layers)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self.globals is not None:
            self.globals["chunk_kda"] = self.orig_chunk
            self.globals["fused_recurrent_kda"] = self.orig_fused
        self.globals = None

    def begin(self, branch, target_layer=None, injections=None, overrides=None):
        self.branch = str(branch)
        self.target_layer = None if target_layer is None else int(target_layer)
        self.injections = injections or {}
        self.overrides = overrides or {}
        self.current_layer = None
        self.records[str(branch)] = {}

    def end(self):
        self.branch = None
        self.target_layer = None
        self.injections = {}
        self.overrides = {}
        self.current_layer = None

    def _record(self, stage_name, tensor, layer=None):
        if self.branch is None or not torch.is_tensor(tensor):
            return
        key = stage_name if layer is None else f"L{int(layer)}:{stage_name}"
        value = tensor.detach().float().cpu().clone()
        self.records[str(self.branch)][key] = value
        if layer is not None and self.target_layer is not None and int(layer) == int(self.target_layer):
            self.records[str(self.branch)][stage_name] = value

    def _first_tensor(self, obj):
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, (tuple, list)):
            for item in obj:
                if torch.is_tensor(item):
                    return item
        return None

    def _make_attention_pre(self, layer_idx):
        def pre(_module, _inputs):
            self.current_layer = int(layer_idx)
        return pre

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

    def _make_o_norm_pre(self, layer_idx):
        def pre(_module, inputs):
            if self.target_layer is None or int(layer_idx) == int(self.target_layer):
                if len(inputs) >= 1 and torch.is_tensor(inputs[0]):
                    self._record("o_norm_input", inputs[0], layer_idx)
                if len(inputs) >= 2 and torch.is_tensor(inputs[1]):
                    self._record("o_norm_gate_logits", inputs[1], layer_idx)
        return pre

    def _make_o_norm_post(self, layer_idx):
        def post(_module, _inputs, output):
            out = self._first_tensor(output)
            replacement = None
            if self.branch is not None:
                replacement = self.overrides.get(int(layer_idx))
            if replacement is not None:
                replacement = replacement.to(device=out.device, dtype=out.dtype)
                self._record("o_norm_gated_rms", replacement, layer_idx)
                return replacement
            if self.target_layer is None or int(layer_idx) == int(self.target_layer):
                if out is not None:
                    self._record("o_norm_gated_rms", out, layer_idx)
            return None
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

    def _wrap_backend(self, fn):
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


def run_branch_forward(torch_mod, model, probe, context, branch, target_layer, delta, override=None):
    past_copy = post().clone_cache(context["past"])
    overrides = {}
    if override is not None:
        overrides[int(target_layer)] = override
    probe.begin(branch, target_layer=target_layer, injections={int(target_layer): delta}, overrides=overrides)
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


def build_o_norm_path_audit(model=None, target_layer=None):
    source = MODEL_PATH / "modeling_bailing_moe_v3.py"
    activation = "sigmoid"
    eps = None
    hidden_size = 128
    weight_shape = [128]
    if model is not None and target_layer is not None:
        attn = model.model.layers[int(target_layer)].attention
        mod = attn.o_norm
        activation = getattr(mod, "activation", activation)
        eps = float(getattr(mod, "eps", 0.0))
        hidden_size = int(getattr(mod, "hidden_size", hidden_size))
        weight = getattr(mod, "weight", None)
        weight_shape = list(weight.shape) if weight is not None else None
    stages = [
        {
            "stage_name": "input_core_readout",
            "source_function": "BailingMoeV3KimiDeltaAttention.forward -> FusedRMSNormGated.forward input x",
            "operation": "core readout tensor o after KDA recurrent readout, before o_norm",
            "input_shape": "[B,T,H,V]",
            "output_shape": "[B,T,H,V]",
            "dtype": "runtime model dtype, captured as float32 and analyzed in float64",
            "linear_or_nonlinear": "linear in recurrent state at fixed query",
            "learned_parameter_involvement": False,
            "gate_dependence": False,
            "broadcast_semantics": "none",
        },
        {
            "stage_name": "pure_rms",
            "source_function": "fla.modules.fused_norm_gate.rms_norm_gated",
            "operation": "x / sqrt(mean(x^2, dim=-1, keepdim=True) + eps)",
            "input_shape": "[B,T,H,V]",
            "output_shape": "[B,T,H,V]",
            "dtype": "float64 analysis reconstruction",
            "linear_or_nonlinear": "nonlinear RMS normalization",
            "learned_parameter_involvement": False,
            "gate_dependence": False,
            "broadcast_semantics": "normalization over V/head_dim",
        },
        {
            "stage_name": "learned_scale",
            "source_function": "fla.modules.fused_norm_gate.rms_norm_gated",
            "operation": "pure_rms * weight",
            "input_shape": "[B,T,H,V]",
            "output_shape": "[B,T,H,V]",
            "dtype": "float64 analysis reconstruction",
            "linear_or_nonlinear": "linear diagonal scale after RMS",
            "learned_parameter_involvement": True,
            "gate_dependence": False,
            "broadcast_semantics": f"weight shape {weight_shape} broadcast over B,T,H",
        },
        {
            "stage_name": "effective_gate",
            "source_function": "fla.modules.fused_norm_gate.rms_norm_gated",
            "operation": "sigmoid(g) because Ling uses activation='sigmoid'",
            "input_shape": "[B,T,H,V]",
            "output_shape": "[B,T,H,V]",
            "dtype": "float64 analysis reconstruction",
            "linear_or_nonlinear": "elementwise nonlinear gate multiplier",
            "learned_parameter_involvement": "g comes from g_proj(hidden_states)",
            "gate_dependence": True,
            "broadcast_semantics": "elementwise gate multiplier, no reduction",
        },
        {
            "stage_name": "module_output",
            "source_function": "FusedRMSNormGated.forward",
            "operation": "learned_scale * effective_gate",
            "input_shape": "[B,T,H,V]",
            "output_shape": "[B,T,H,V]",
            "dtype": "runtime model dtype; reconstructed in float64",
            "linear_or_nonlinear": "composition of nonlinear RMS and elementwise gate",
            "learned_parameter_involvement": True,
            "gate_dependence": True,
            "broadcast_semantics": "weight over V, gate elementwise",
        },
    ]
    return {
        "task": TASK,
        "KDA_O_NORM_GATED_RMS_PATH_AUDIT": "PASS" if source.exists() and activation == "sigmoid" else "FAIL",
        "source_file": str(source),
        "module_class": "fla.modules.fused_norm_gate.FusedRMSNormGated",
        "activation": activation,
        "eps": eps,
        "hidden_size": hidden_size,
        "stages": stages,
    }


def reset_outputs(outdir):
    for rel in [
        "metric_consistency/atomic_identity.jsonl",
        "internal_stage_metrics/internal_stage_metrics.jsonl",
        "rms_geometry/rms_geometry.jsonl",
        "rms_jacobian/rms_jacobian.jsonl",
        "frozen_denominator/frozen_denominator_metrics.jsonl",
        "learned_scale/learned_scale_metrics.jsonl",
        "gate/gate_metrics.jsonl",
        "downstream/downstream_stage_metrics.jsonl",
        "downstream/downstream_operator_transfer.jsonl",
    ]:
        p = outdir / rel
        if p.exists():
            p.unlink()


def internal_stage_tensors(x, gate_logits, weight, activation, eps):
    stages = compute_o_norm_internal_stages(x, gate_logits, weight, activation=activation, eps=eps)
    return {
        "core_readout": stages["input_core_readout"],
        "pure_rms": stages["pure_rms"],
        "learned_scale": stages["learned_scale"],
        "effective_gate": stages["module_output"],
        "module_output": stages["module_output"],
    }, stages


def stage_rows_from_tensors(unit_id, horizon, target_layer, fp_tensors, native_tensors, rotated_tensors, order, prefix=""):
    rows = []
    tensors = {}
    for stage in order:
        if stage not in fp_tensors or stage not in native_tensors or stage not in rotated_tensors:
            continue
        fp = fp_tensors[stage].detach().double()
        dn = native_tensors[stage].detach().double() - fp
        dr = rotated_tensors[stage].detach().double() - fp
        met = atomic_stage_metric(fp, dn, dr)
        row = {
            "unit_id": unit_id,
            "horizon": int(horizon),
            "target_layer": int(target_layer),
            "stage": prefix + stage,
            "native_delta_norm": met["native_delta_norm"],
            "rotated_delta_norm": met["rotated_delta_norm"],
            "r_j": met["r_j"],
            "G_j": met["G_j"],
            "native_rel_l2": relative_l2(fp, dn),
            "rotated_rel_l2": relative_l2(fp, dr),
            "fp_norm": tensor_norm(fp),
        }
        rows.append(row)
        tensors[stage] = {"fp": fp, "native_delta": dn, "rotated_delta": dr, "metric": met}
    return rows, tensors


def transfer_rows_from_stage_tensors(unit_id, horizon, target_layer, tensors, order):
    rows = []
    present = [s for s in order if s in tensors]
    for prev, cur in zip(present, present[1:]):
        tr = atomic_transfer_metric(tensors[prev]["metric"], tensors[cur]["metric"])
        rows.append({
            "unit_id": unit_id,
            "horizon": int(horizon),
            "target_layer": int(target_layer),
            "from_stage": prev,
            "to_stage": cur,
            **tr,
        })
    return rows


def summarize_metric_consistency(rows):
    vals = [abs(float(r["identity_abs_error"])) for r in rows if finite(r.get("identity_abs_error"))]
    rels = [abs(float(r["identity_rel_error"])) for r in rows if finite(r.get("identity_rel_error"))]
    return {
        "n_atomic_transfers": len(vals),
        "max_absolute_identity_error": max(vals, default=None),
        "max_relative_identity_error": max(rels, default=None),
        "median_absolute_identity_error": median(vals),
        "p95_absolute_identity_error": percentile(vals, 0.95),
        "p99_absolute_identity_error": percentile(vals, 0.99),
        "KDA_STAGE_TRANSFER_METRIC_CONSISTENCY": "PASS" if vals and max(vals) <= IDENTITY_TOL and max(rels, default=0.0) <= 1e-6 else "FAIL",
    }


def aggregate_stage_rows(stage_rows):
    grouped = defaultdict(list)
    for row in stage_rows:
        grouped[(row["unit_id"], int(row["horizon"]), row["stage"])].append(row)
    out = []
    for (unit_id, horizon, stage), rows in sorted(grouped.items()):
        n2 = sum(float(r["native_delta_norm"]) ** 2 for r in rows)
        r2 = sum(float(r["rotated_delta_norm"]) ** 2 for r in rows)
        ratio = math.sqrt(r2) / (math.sqrt(n2) + EPS)
        out.append({
            "unit_id": unit_id,
            "horizon": horizon,
            "stage": stage,
            "n_target_layers": len({int(r["target_layer"]) for r in rows}),
            "native_delta_norm_rss": math.sqrt(n2),
            "rotated_delta_norm_rss": math.sqrt(r2),
            "r_j": ratio,
            "G_j": 1.0 - ratio * ratio,
            "native_rel_l2_rms": math.sqrt(mean([float(r["native_rel_l2"]) ** 2 for r in rows]) or 0.0),
            "rotated_rel_l2_rms": math.sqrt(mean([float(r["rotated_rel_l2"]) ** 2 for r in rows]) or 0.0),
        })
    return out


def summarize_stage_rows(unit_stage_rows, order):
    by = defaultdict(list)
    for row in unit_stage_rows:
        by[(int(row["horizon"]), row["stage"])].append(row)
    out = []
    for horizon in HORIZONS:
        for stage in order:
            rows = by.get((horizon, stage), [])
            vals = [float(r["G_j"]) for r in rows if finite(r.get("G_j"))]
            out.append({
                "horizon": horizon,
                "stage": stage,
                "n_units": len({r["unit_id"] for r in rows}),
                "median_G_j": median(vals),
                "mean_G_j": mean(vals),
                "std_G_j": std(vals),
                "wins_G_j_gt_0": sum(v > 0 for v in vals),
                "bootstrap95_G_j": json.dumps(bootstrap_ci(vals), sort_keys=True),
            })
    return out


def summarize_delta_g(unit_stage_rows, order):
    by_unit = defaultdict(dict)
    for row in unit_stage_rows:
        by_unit[(row["unit_id"], int(row["horizon"]))][row["stage"]] = row
    out = []
    for horizon in HORIZONS:
        for prev, cur in zip(order, order[1:]):
            vals = []
            for (_unit, h), stages in by_unit.items():
                if h == horizon and prev in stages and cur in stages:
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


def aggregate_transfer_rows(transfer_rows):
    grouped = defaultdict(list)
    for row in transfer_rows:
        grouped[(row["unit_id"], int(row["horizon"]), row["from_stage"], row["to_stage"])].append(row)
    out = []
    for (unit_id, horizon, frm, to), rows in sorted(grouped.items()):
        vals = [float(r["Lambda_j"]) for r in rows if finite(r.get("Lambda_j"))]
        out.append({
            "unit_id": unit_id,
            "horizon": horizon,
            "from_stage": frm,
            "to_stage": to,
            "n_target_layers": len({int(r["target_layer"]) for r in rows}),
            "median_Lambda": median(vals),
            "wins_Lambda_gt_1": sum(v > 1.0 for v in vals),
        })
    return out


def summarize_transfer_rows(unit_transfer_rows, order):
    by = defaultdict(list)
    for row in unit_transfer_rows:
        by[(int(row["horizon"]), row["from_stage"], row["to_stage"])].append(row)
    out = []
    for horizon in HORIZONS:
        for prev, cur in zip(order, order[1:]):
            rows = by.get((horizon, prev, cur), [])
            vals = [float(r["median_Lambda"]) for r in rows if finite(r.get("median_Lambda"))]
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


def classify_first_internal_attenuation(stage_summary, delta_summary):
    stage_by = {r["stage"]: r for r in stage_summary if int(r["horizon"]) == PRIMARY_HORIZON}
    delta_by = {(r["from_stage"], r["to_stage"]): r for r in delta_summary if int(r["horizon"]) == PRIMARY_HORIZON}
    for prev, cur in zip(INTERNAL_STAGE_ORDER, INTERNAL_STAGE_ORDER[1:]):
        p = as_float(stage_by.get(prev, {}).get("median_G_j"))
        c = as_float(stage_by.get(cur, {}).get("median_G_j"))
        d = as_float(delta_by.get((prev, cur), {}).get("median_DeltaG"))
        wins = int(float(delta_by.get((prev, cur), {}).get("wins_DeltaG_lt_0") or 0))
        if p is None or c is None or d is None:
            continue
        if p > 0.05 and p - c >= 0.03 and d < 0.0 and wins >= 10:
            if cur == "pure_rms":
                return "pure_rms"
            if cur == "learned_scale":
                return "learned_scale"
            if cur in {"effective_gate", "module_output"}:
                return "gate"
    return "none"


def summarize_scalar_rows(rows, group_fields, value_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(f) for f in group_fields)].append(row)
    out = []
    for key, rs in sorted(grouped.items()):
        row = {f: v for f, v in zip(group_fields, key)}
        row["n"] = len(rs)
        for field in value_fields:
            vals = [float(r[field]) for r in rs if finite(r.get(field))]
            row[f"median_{field}"] = median(vals)
            row[f"mean_{field}"] = mean(vals)
            row[f"wins_{field}_gt_1"] = sum(v > 1.0 for v in vals)
            row[f"bootstrap95_{field}"] = json.dumps(bootstrap_ci(vals), sort_keys=True)
        out.append(row)
    return out


def classify_logit_gain(downstream_summary):
    row = next((r for r in downstream_summary if int(r["horizon"]) == PRIMARY_HORIZON and r["stage"] == "lm_head_logits"), None)
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


def run_experiment(args):
    outdir = Path(args.output_dir)
    if args.overwrite:
        reset_outputs(outdir)
    for name in [
        "metric_consistency",
        "architecture_audit",
        "internal_stage_metrics",
        "rms_geometry",
        "rms_jacobian",
        "frozen_denominator",
        "learned_scale",
        "gate",
        "interaction",
        "downstream",
        "figures",
    ]:
        (outdir / name).mkdir(parents=True, exist_ok=True)

    env = environment_info()
    save_json(outdir / "provenance.json", env)
    units, manifest_hash, units_all = post().load_units(args.scope, args.max_units)
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
        "model_path": str(MODEL_PATH),
    }
    save_json(outdir / "config.json", config)

    mod, torch_mod, model, tokenizer, kda_layers, rows_by_pid, teacher_tokens = post().load_kda_context()
    if args.target_layer_limit:
        kda_layers = list(kda_layers)[: int(args.target_layer_limit)]
    audit = build_o_norm_path_audit(model, kda_layers[0])
    save_json(outdir / "architecture_audit" / "o_norm_gated_rms_internal_path.json", audit)
    if audit["KDA_O_NORM_GATED_RMS_PATH_AUDIT"] != "PASS":
        raise RuntimeError("o_norm_gated_rms internal path audit failed")

    rotation = post().exact().make_experiment_rotation("kda", 128, "rht", seed=RHT_SEED, dtype=torch.float32)
    internal_rows = []
    internal_transfer_rows = []
    identity_rows = []
    reconciliation_rows = []
    downstream_rows = []
    downstream_transfer_rows = []
    rms_geometry_rows = []
    rms_jacobian_rows = []
    frozen_rows = []
    learned_rows = []
    gate_rows = []
    provenance_rows = []
    failures = []
    unit_done = set()
    module_parity_errors = []
    frozen_downstream_rows = []

    try:
        for idx, unit in enumerate(units, 1):
            print(f"[{now()}] KDA o_norm closure unit {idx}/{len(units)} {unit['unit_id']}", flush=True)
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
                tape = post().exact().capture_driver_tape("kda", unit, ctx_exact, max(HORIZONS))
                err_by_h, _readout_rows = post().replay_error_stacks(tape, exact_capture, rotation)
                exact_capture.close()
                exact_capture = None

                probe = KdaONormProbe()
                installed_layers = probe.install(model)
                missing_layers = sorted(set(map(int, kda_layers)) - set(map(int, installed_layers)))
                if missing_layers:
                    raise RuntimeError(f"o_norm probe missing KDA layers: {missing_layers}")

                for horizon in HORIZONS:
                    context = post().prepare_clean_context(mod, torch_mod, model, tokenizer, unit, item, tokens, horizon)
                    original_cache_hash = stack_hash(post().get_kda_stack(mod, context["past"], kda_layers))
                    for layer in kda_layers:
                        if layer not in err_by_h[horizon]["delta_N"] or layer not in err_by_h[horizon]["delta_R"]:
                            continue
                        delta_n = err_by_h[horizon]["delta_N"][layer]
                        delta_r = err_by_h[horizon]["delta_R"][layer]
                        zero = torch.zeros_like(delta_n)
                        fp_out, fp_records = run_branch_forward(torch_mod, model, probe, context, f"FP_{unit['unit_id']}_{horizon}_{layer}", layer, zero)
                        zero_out, zero_records = run_branch_forward(torch_mod, model, probe, context, f"ZERO_{unit['unit_id']}_{horizon}_{layer}", layer, zero)
                        n_out, n_records = run_branch_forward(torch_mod, model, probe, context, f"N_{unit['unit_id']}_{horizon}_{layer}", layer, delta_n)
                        r_out, r_records = run_branch_forward(torch_mod, model, probe, context, f"R_{unit['unit_id']}_{horizon}_{layer}", layer, delta_r)
                        after_cache_hash = stack_hash(post().get_kda_stack(mod, context["past"], kda_layers))
                        cache_gate = "PASS" if original_cache_hash == after_cache_hash else "FAIL"
                        zero_gate = zero_perturbation_noninterference(
                            {k: v for k, v in fp_records.items() if torch.is_tensor(v)},
                            {k: v for k, v in zero_records.items() if torch.is_tensor(v)},
                        )
                        if cache_gate != "PASS" or zero_gate["gate"] != "PASS":
                            raise RuntimeError(f"noninterference/cache gate failed unit={unit['unit_id']} h={horizon} layer={layer}: {cache_gate} {zero_gate}")

                        attn = model.model.layers[int(layer)].attention
                        norm_mod = attn.o_norm
                        weight = getattr(norm_mod, "weight", None)
                        activation = getattr(norm_mod, "activation", "sigmoid")
                        eps = float(getattr(norm_mod, "eps", EPS))
                        fp_x = fp_records["o_norm_input"]
                        fp_gate = fp_records["o_norm_gate_logits"]
                        n_gate = n_records["o_norm_gate_logits"]
                        r_gate = r_records["o_norm_gate_logits"]
                        gate_context = "FIXED_SHARED_GATE" if tensor_hash(fp_gate) == tensor_hash(n_gate) == tensor_hash(r_gate) else "BRANCH_DEPENDENT_GATE"
                        fp_it, fp_aux = internal_stage_tensors(fp_x, fp_gate, weight.cpu() if weight is not None else None, activation, eps)
                        n_it, n_aux = internal_stage_tensors(n_records["o_norm_input"], n_gate, weight.cpu() if weight is not None else None, activation, eps)
                        r_it, r_aux = internal_stage_tensors(r_records["o_norm_input"], r_gate, weight.cpu() if weight is not None else None, activation, eps)
                        for branch_name, records, aux in [("fp", fp_records, fp_aux), ("native", n_records, n_aux), ("rotated", r_records, r_aux)]:
                            actual = records["o_norm_gated_rms"].detach().double()
                            err = tensor_norm(actual - aux["module_output"]) / (tensor_norm(actual) + EPS)
                            module_parity_errors.append(err)
                        irows, itensors = stage_rows_from_tensors(str(unit["unit_id"]), horizon, layer, fp_it, n_it, r_it, INTERNAL_STAGE_ORDER)
                        trows = transfer_rows_from_stage_tensors(str(unit["unit_id"]), horizon, layer, itensors, INTERNAL_STAGE_ORDER)
                        for row in irows:
                            row["gate_context"] = gate_context
                            append_jsonl(outdir / "internal_stage_metrics" / "internal_stage_metrics.jsonl", row)
                        for row in trows:
                            append_jsonl(outdir / "metric_consistency" / "atomic_identity.jsonl", row)
                        internal_rows.extend(irows)
                        internal_transfer_rows.extend(trows)
                        identity_rows.extend(trows)

                        prev_by_stage = {r["stage"]: r for r in irows}
                        for prev, cur in zip(INTERNAL_STAGE_ORDER, INTERNAL_STAGE_ORDER[1:]):
                            reconciliation_rows.append({
                                "unit_id": str(unit["unit_id"]),
                                "horizon": horizon,
                                "target_layer": int(layer),
                                "from_stage": prev,
                                "to_stage": cur,
                                **aggregation_reconciliation([prev_by_stage[prev]], [prev_by_stage[cur]]),
                            })

                        fp_down = {s: fp_records[s] for s in DOWNSTREAM_STAGE_ORDER if s in fp_records}
                        n_down = {s: n_records[s] for s in DOWNSTREAM_STAGE_ORDER if s in n_records}
                        r_down = {s: r_records[s] for s in DOWNSTREAM_STAGE_ORDER if s in r_records}
                        drows, dtensors = stage_rows_from_tensors(str(unit["unit_id"]), horizon, layer, fp_down, n_down, r_down, DOWNSTREAM_STAGE_ORDER)
                        dtrs = transfer_rows_from_stage_tensors(str(unit["unit_id"]), horizon, layer, dtensors, DOWNSTREAM_STAGE_ORDER)
                        for row in drows:
                            append_jsonl(outdir / "downstream" / "downstream_stage_metrics.jsonl", row)
                        for row in dtrs:
                            append_jsonl(outdir / "downstream" / "downstream_operator_transfer.jsonl", row)
                        downstream_rows.extend(drows)
                        downstream_transfer_rows.extend(dtrs)

                        n_input_delta = n_it["core_readout"] - fp_it["core_readout"]
                        r_input_delta = r_it["core_readout"] - fp_it["core_readout"]
                        n_dec = radial_tangential_decomposition(fp_it["core_readout"], n_input_delta)
                        r_dec = radial_tangential_decomposition(fp_it["core_readout"], r_input_delta)
                        geom = {
                            "unit_id": str(unit["unit_id"]),
                            "horizon": horizon,
                            "target_layer": int(layer),
                            "native_radial_fraction": n_dec["radial_fraction"],
                            "native_tangential_fraction": n_dec["tangential_fraction"],
                            "rotated_radial_fraction": r_dec["radial_fraction"],
                            "rotated_tangential_fraction": r_dec["tangential_fraction"],
                            "radial_fraction_delta_R_minus_N": r_dec["radial_fraction"] - n_dec["radial_fraction"],
                        }
                        rms_geometry_rows.append(geom)
                        append_jsonl(outdir / "rms_geometry" / "rms_geometry.jsonl", geom)
                        jac = {
                            "unit_id": str(unit["unit_id"]),
                            "horizon": horizon,
                            "target_layer": int(layer),
                            **rms_jacobian_audit(fp_it["core_readout"], n_input_delta, r_input_delta, eps=eps),
                        }
                        rms_jacobian_rows.append(jac)
                        append_jsonl(outdir / "rms_jacobian" / "rms_jacobian.jsonl", jac)
                        frozen_local = {
                            "unit_id": str(unit["unit_id"]),
                            "horizon": horizon,
                            "target_layer": int(layer),
                            **frozen_denominator_rms_delta(fp_it["core_readout"], n_input_delta, r_input_delta, eps=eps),
                        }

                        fr_n = frozen_denominator_o_norm_stages(fp_it["core_readout"], n_input_delta, fp_gate, weight.cpu() if weight is not None else None, activation, eps)
                        fr_r = frozen_denominator_o_norm_stages(fp_it["core_readout"], r_input_delta, fp_gate, weight.cpu() if weight is not None else None, activation, eps)
                        frozen_tensors_fp = {
                            "frozen_rms": fp_aux["pure_rms"],
                            "frozen_learned_scale": fp_aux["learned_scale"],
                            "frozen_module_output": fp_aux["module_output"],
                        }
                        frows, _ftensors = stage_rows_from_tensors(
                            str(unit["unit_id"]),
                            horizon,
                            layer,
                            frozen_tensors_fp,
                            fr_n,
                            fr_r,
                            ["frozen_rms", "frozen_learned_scale", "frozen_module_output"],
                        )
                        for row in frows:
                            frozen_rows.append({**frozen_local, **{f"stage_{k}": v for k, v in row.items()}})
                        append_jsonl(outdir / "frozen_denominator" / "frozen_denominator_metrics.jsonl", {**frozen_local, "stage_rows": frows})

                        fn_out, fn_records = run_branch_forward(torch_mod, model, probe, context, f"FN_{unit['unit_id']}_{horizon}_{layer}", layer, delta_n, override=fr_n["frozen_module_output"])
                        fr_out, fr_records = run_branch_forward(torch_mod, model, probe, context, f"FR_{unit['unit_id']}_{horizon}_{layer}", layer, delta_r, override=fr_r["frozen_module_output"])
                        fn_down = {s: fn_records[s] for s in DOWNSTREAM_STAGE_ORDER if s in fn_records}
                        fr_down = {s: fr_records[s] for s in DOWNSTREAM_STAGE_ORDER if s in fr_records}
                        fdrows, _ = stage_rows_from_tensors(
                            str(unit["unit_id"]), horizon, layer, fp_down, fn_down, fr_down, DOWNSTREAM_STAGE_ORDER, prefix="frozen_downstream:"
                        )
                        frozen_downstream_rows.extend(fdrows)

                        if weight is not None:
                            gamma = weight.detach().cpu().double()
                            learned = {
                                "unit_id": str(unit["unit_id"]),
                                "horizon": horizon,
                                "target_layer": int(layer),
                                **diagonal_transfer(n_aux["pure_rms"] - fp_aux["pure_rms"], r_aux["pure_rms"] - fp_aux["pure_rms"], gamma),
                            }
                            iso_gamma = isotropic_multiplier(gamma)
                            iso = diagonal_transfer(n_aux["pure_rms"] - fp_aux["pure_rms"], r_aux["pure_rms"] - fp_aux["pure_rms"], iso_gamma)
                            learned["isotropic_Lambda"] = iso["Lambda"]
                            learned_rows.append(learned)
                            append_jsonl(outdir / "learned_scale" / "learned_scale_metrics.jsonl", learned)
                        gate_mult = fp_aux["effective_gate"]
                        gate_flat = gate_mult.reshape(-1, gate_mult.shape[-1]).mean(dim=0)
                        gate_met = {
                            "unit_id": str(unit["unit_id"]),
                            "horizon": horizon,
                            "target_layer": int(layer),
                            "KDA_GATE_CONTEXT": gate_context,
                            **diagonal_transfer(n_aux["learned_scale"] - fp_aux["learned_scale"], r_aux["learned_scale"] - fp_aux["learned_scale"], gate_flat),
                        }
                        iso_gate = isotropic_multiplier(gate_flat)
                        iso_g = diagonal_transfer(n_aux["learned_scale"] - fp_aux["learned_scale"], r_aux["learned_scale"] - fp_aux["learned_scale"], iso_gate)
                        gate_met["isotropic_Lambda"] = iso_g["Lambda"]
                        gate_rows.append(gate_met)
                        append_jsonl(outdir / "gate" / "gate_metrics.jsonl", gate_met)

                        n_prov = {
                            "unit_id": str(unit["unit_id"]),
                            "horizon": int(horizon),
                            "token_id": int(context["token_id"]),
                            "layer": int(layer),
                            "fp_core_sha256": tensor_hash(fp_records["core_readout"]),
                            "gate_sha256": tensor_hash(fp_gate),
                            "cache_sha256": original_cache_hash,
                            "branch": "native",
                            "delta_o_sha256": tensor_hash(delta_n),
                        }
                        r_prov = dict(n_prov, branch="rotated", delta_o_sha256=tensor_hash(delta_r))
                        prov = check_matched_context(n_prov, r_prov)
                        if prov["gate"] != "PASS":
                            raise RuntimeError(f"INVALID_O_NORM_MATCHED_COMPARISON {prov}")
                        provenance_rows.append({**n_prov, "provenance_gate": prov["gate"], "gate_context": gate_context})
                    del context
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                unit_done.add(str(unit["unit_id"]))
            except Exception as exc:
                failure = {"unit": unit, "error": repr(exc), "traceback": traceback.format_exc(limit=25), "timestamp": now()}
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

    consistency = summarize_metric_consistency(identity_rows)
    save_json(outdir / "metric_consistency" / "atomic_identity_summary.json", consistency)
    write_rows(outdir / "metric_consistency" / "aggregation_reconciliation.csv", reconciliation_rows)
    if consistency["KDA_STAGE_TRANSFER_METRIC_CONSISTENCY"] != "PASS":
        summary = {
            "task": TASK,
            **consistency,
            "KDA_O_NORM_GATED_RMS_PATH_AUDIT": audit["KDA_O_NORM_GATED_RMS_PATH_AUDIT"],
            "FINAL_CLASSIFICATION": "STOPPED_METRIC_CONSISTENCY_FAIL",
            "NEXT_EXPERIMENT": "KDA_O_NORM_GATED_RMS_REASSESSMENT_V2",
            "n_units_completed": len(unit_done),
            "failures": failures,
            "environment": env,
        }
        save_json(outdir / "summary.json", summary)
        write_report(outdir, summary, audit, [], [], [], [], [], [], [], [])
        return summary

    unit_internal = aggregate_stage_rows(internal_rows)
    internal_summary = summarize_stage_rows(unit_internal, INTERNAL_STAGE_ORDER)
    internal_delta = summarize_delta_g(unit_internal, INTERNAL_STAGE_ORDER)
    unit_internal_transfer = aggregate_transfer_rows(internal_transfer_rows)
    internal_transfer_summary = summarize_transfer_rows(unit_internal_transfer, INTERNAL_STAGE_ORDER)
    write_rows(outdir / "internal_stage_metrics" / "internal_stage_unit_aggregate.csv", unit_internal)
    write_rows(outdir / "internal_stage_metrics" / "internal_stage_summary.csv", internal_summary)
    write_rows(outdir / "internal_stage_metrics" / "internal_delta_summary.csv", internal_delta)
    write_rows(outdir / "internal_stage_metrics" / "internal_transfer_unit_aggregate.csv", unit_internal_transfer)
    write_rows(outdir / "internal_stage_metrics" / "internal_transfer_summary.csv", internal_transfer_summary)

    unit_downstream = aggregate_stage_rows(downstream_rows)
    downstream_summary = summarize_stage_rows(unit_downstream, DOWNSTREAM_STAGE_ORDER)
    downstream_delta = summarize_delta_g(unit_downstream, DOWNSTREAM_STAGE_ORDER)
    unit_downstream_transfer = aggregate_transfer_rows(downstream_transfer_rows)
    downstream_transfer_summary = summarize_transfer_rows(unit_downstream_transfer, DOWNSTREAM_STAGE_ORDER)
    write_rows(outdir / "downstream" / "downstream_stage_unit_aggregate.csv", unit_downstream)
    write_rows(outdir / "downstream" / "downstream_stage_summary.csv", downstream_summary)
    write_rows(outdir / "downstream" / "downstream_delta_summary.csv", downstream_delta)
    write_rows(outdir / "downstream" / "downstream_transfer_unit_aggregate.csv", unit_downstream_transfer)
    write_rows(outdir / "downstream" / "downstream_transfer_summary.csv", downstream_transfer_summary)

    write_rows(outdir / "provenance_pairs.csv", provenance_rows)
    write_rows(outdir / "rms_geometry" / "rms_geometry_unit_aggregate.csv", summarize_scalar_rows(rms_geometry_rows, ["unit_id", "horizon"], ["native_radial_fraction", "rotated_radial_fraction", "radial_fraction_delta_R_minus_N"]))
    write_rows(outdir / "rms_jacobian" / "rms_jacobian_unit_aggregate.csv", summarize_scalar_rows(rms_jacobian_rows, ["unit_id", "horizon"], ["Lambda_RMS_J", "native_linearization_relative_error", "rotated_linearization_relative_error"]))
    write_rows(outdir / "frozen_denominator" / "frozen_denominator_unit_aggregate.csv", summarize_scalar_rows(frozen_rows, ["stage_unit_id", "stage_horizon", "stage_stage"], ["stage_G_j"]))
    write_rows(outdir / "frozen_denominator" / "frozen_downstream_unit_aggregate.csv", aggregate_stage_rows(frozen_downstream_rows))
    write_rows(outdir / "learned_scale" / "learned_scale_unit_aggregate.csv", summarize_scalar_rows(learned_rows, ["unit_id", "horizon"], ["Lambda", "isotropic_Lambda"]))
    write_rows(outdir / "gate" / "gate_unit_aggregate.csv", summarize_scalar_rows(gate_rows, ["unit_id", "horizon"], ["Lambda", "isotropic_Lambda"]))

    first = classify_first_internal_attenuation(internal_summary, internal_delta)
    t64_delta = {(r["from_stage"], r["to_stage"]): r for r in internal_delta if int(r["horizon"]) == PRIMARY_HORIZON}
    rms_delta = as_float(t64_delta.get(("core_readout", "pure_rms"), {}).get("median_DeltaG"), 0.0)
    learned_delta = as_float(t64_delta.get(("pure_rms", "learned_scale"), {}).get("median_DeltaG"), 0.0)
    gate_delta = as_float(t64_delta.get(("learned_scale", "effective_gate"), {}).get("median_DeltaG"), 0.0)
    loss_rows = [
        {"component": "pure_rms", "median_DeltaG": rms_delta},
        {"component": "learned_scale", "median_DeltaG": learned_delta},
        {"component": "gate", "median_DeltaG": gate_delta},
    ]
    write_rows(outdir / "loss_budget.csv", loss_rows)

    rms_geom_rows_t64 = [r for r in rms_geometry_rows if int(r["horizon"]) == PRIMARY_HORIZON]
    jac_rows_t64 = [r for r in rms_jacobian_rows if int(r["horizon"]) == PRIMARY_HORIZON]
    geom_delta = median([r["radial_fraction_delta_R_minus_N"] for r in rms_geom_rows_t64])
    jac_lam = median([r["Lambda_RMS_J"] for r in jac_rows_t64])
    jac_err = median([max(r["native_linearization_relative_error"], r["rotated_linearization_relative_error"]) for r in jac_rows_t64])
    rms_geometry_cls = "PRESENT" if geom_delta is not None and abs(geom_delta) >= 0.01 else "WEAK_OR_ABSENT"
    jac_cls = "SUPPORTED" if jac_lam is not None and jac_lam > 1.01 and jac_err is not None and jac_err < 0.25 else ("PARTIAL" if jac_lam is not None and jac_lam > 1.0 else "NOT_SUPPORTED")
    frozen_unit = aggregate_stage_rows(frozen_downstream_rows)
    frozen_summary = summarize_stage_rows(frozen_unit, [f"frozen_downstream:{s}" for s in DOWNSTREAM_STAGE_ORDER])
    write_rows(outdir / "frozen_denominator" / "frozen_downstream_summary.csv", frozen_summary)
    real_pure = next((r for r in internal_summary if int(r["horizon"]) == PRIMARY_HORIZON and r["stage"] == "pure_rms"), {})
    real_core = next((r for r in internal_summary if int(r["horizon"]) == PRIMARY_HORIZON and r["stage"] == "core_readout"), {})
    frozen_rms_rows = [r for r in frozen_rows if int(r.get("stage_horizon", 0)) == PRIMARY_HORIZON and r.get("stage_stage") == "frozen_rms"]
    frozen_module_rows = [r for r in frozen_rows if int(r.get("stage_horizon", 0)) == PRIMARY_HORIZON and r.get("stage_stage") == "frozen_module_output"]
    frozen_rms_gain = median([r["stage_G_j"] for r in frozen_rms_rows])
    frozen_module_gain = median([r["stage_G_j"] for r in frozen_module_rows])
    real_pure_gain = as_float(real_pure.get("median_G_j"))
    real_core_gain = as_float(real_core.get("median_G_j"))
    real_module = next((r for r in internal_summary if int(r["horizon"]) == PRIMARY_HORIZON and r["stage"] == "module_output"), {})
    real_module_gain = as_float(real_module.get("median_G_j"))
    frozen_rescue = (
        frozen_rms_gain is not None
        and real_pure_gain is not None
        and real_core_gain is not None
        and frozen_rms_gain > real_pure_gain + 0.03
        and abs(frozen_rms_gain - real_core_gain) < abs(real_pure_gain - real_core_gain)
    )
    frozen_partial = (
        frozen_module_gain is not None
        and real_module_gain is not None
        and frozen_module_gain > real_module_gain + 0.03
    )
    frozen_cls = "SUPPORTED" if frozen_rescue else ("PARTIAL" if frozen_partial else "NOT_SUPPORTED")

    learned_lam = median([r["Lambda"] for r in learned_rows if int(r["horizon"]) == PRIMARY_HORIZON])
    learned_iso_lam = median([r["isotropic_Lambda"] for r in learned_rows if int(r["horizon"]) == PRIMARY_HORIZON])
    learned_sel = "PRESENT" if learned_lam is not None and abs(learned_lam - 1.0) >= 0.02 else "ABSENT"
    learned_closure = "SUPPORTED" if learned_sel == "PRESENT" and learned_iso_lam is not None and abs(learned_iso_lam - 1.0) < abs(learned_lam - 1.0) else "NOT_SUPPORTED"
    gate_context_vals = sorted({r["KDA_GATE_CONTEXT"] for r in gate_rows})
    gate_context = gate_context_vals[0] if gate_context_vals else "NOT_RUN"
    gate_lam = median([r["Lambda"] for r in gate_rows if int(r["horizon"]) == PRIMARY_HORIZON])
    gate_iso_lam = median([r["isotropic_Lambda"] for r in gate_rows if int(r["horizon"]) == PRIMARY_HORIZON])
    gate_sel = "PRESENT" if gate_lam is not None and abs(gate_lam - 1.0) >= 0.02 else "ABSENT"
    gate_closure = "SUPPORTED" if gate_sel == "PRESENT" and gate_iso_lam is not None and abs(gate_iso_lam - 1.0) < abs(gate_lam - 1.0) else "NOT_SUPPORTED"

    rms_real_robust = first == "pure_rms" or (rms_delta is not None and rms_delta < -0.03)
    rms_closure = "SUPPORTED" if rms_real_robust and jac_cls in {"SUPPORTED", "PARTIAL"} and frozen_cls == "SUPPORTED" else ("PARTIAL" if rms_real_robust else "NOT_SUPPORTED")
    factors = []
    if rms_closure in {"SUPPORTED", "PARTIAL"}:
        factors.append("RMS_NORMALIZATION")
    if learned_sel == "PRESENT" and learned_delta < -0.02:
        factors.append("LEARNED_SCALE")
    if gate_sel == "PRESENT" and gate_delta < -0.02:
        factors.append("DYNAMIC_GATE")
    if len(factors) > 1:
        primary = "MULTI_FACTOR"
    elif factors:
        primary = factors[0]
    else:
        primary = "UNRESOLVED"
    causal_closure = "SUPPORTED" if primary != "UNRESOLVED" and (rms_closure == "SUPPORTED" or learned_closure == "SUPPORTED" or gate_closure == "SUPPORTED") else ("PARTIAL" if primary != "UNRESOLVED" else "NOT_SUPPORTED")
    model_norm_delta = next((r for r in downstream_delta if int(r["horizon"]) == PRIMARY_HORIZON and r["from_stage"] == "layer_output" and r["to_stage"] == "model_norm"), {})
    model_norm_secondary = "PRESENT" if as_float(model_norm_delta.get("median_DeltaG"), 0.0) < -0.05 else "ABSENT"
    same_token = classify_logit_gain(downstream_summary)
    if causal_closure in {"SUPPORTED", "PARTIAL"} and same_token in {"STRONG", "POSITIVE"}:
        mechanism_status = "SUFFICIENT_FOR_CLOSED_LOOP_STAGE"
        final_cls = "KDA_O_NORM_GATED_RMS_FIRST_ATTENUATION_PARTLY_CLOSED_SAME_TOKEN_GAIN_REMAINS"
        next_exp = "KDA_ROTATION_CLOSED_LOOP_DRIVER_DRIFT_CAUSAL_V1"
    elif causal_closure == "NOT_SUPPORTED":
        mechanism_status = "NOT_CLOSED"
        final_cls = "KDA_O_NORM_GATED_RMS_FIRST_ATTENUATION_UNRESOLVED"
        next_exp = "KDA_O_NORM_GATED_RMS_REASSESSMENT_V2"
    else:
        mechanism_status = "PARTIAL"
        final_cls = "KDA_O_NORM_GATED_RMS_CAUSAL_CLOSURE_PARTIAL"
        next_exp = "KDA_O_NORM_GATED_RMS_REASSESSMENT_V2"

    horizon_rows = []
    for horizon in HORIZONS:
        core = next((r for r in internal_summary if int(r["horizon"]) == horizon and r["stage"] == "core_readout"), {})
        modrow = next((r for r in internal_summary if int(r["horizon"]) == horizon and r["stage"] == "module_output"), {})
        logit = next((r for r in downstream_summary if int(r["horizon"]) == horizon and r["stage"] == "lm_head_logits"), {})
        horizon_rows.append({
            "horizon": horizon,
            "n_units": core.get("n_units"),
            "median_core_readout_G": core.get("median_G_j"),
            "median_o_norm_module_G": modrow.get("median_G_j"),
            "median_logit_G": logit.get("median_G_j"),
        })
    write_rows(outdir / "horizon_summary.csv", horizon_rows)

    summary = {
        "task": TASK,
        "KDA_STAGE_TRANSFER_METRIC_CONSISTENCY": consistency["KDA_STAGE_TRANSFER_METRIC_CONSISTENCY"],
        "metric_consistency": consistency,
        "KDA_O_NORM_GATED_RMS_PATH_AUDIT": audit["KDA_O_NORM_GATED_RMS_PATH_AUDIT"],
        "KDA_O_NORM_MANUAL_MODULE_PARITY_MAX_REL": max(module_parity_errors, default=None),
        "KDA_O_NORM_FIRST_INTERNAL_ATTENUATION_STAGE": first,
        "KDA_RMS_GEOMETRY_DIFFERENCE": rms_geometry_cls,
        "KDA_RMS_JACOBIAN_PREDICTS_ATTENUATION": jac_cls,
        "KDA_FROZEN_DENOMINATOR_RMS_RESCUE": frozen_cls,
        "KDA_RMS_NORMALIZATION_CAUSAL_CLOSURE": rms_closure,
        "KDA_LEARNED_SCALE_SELECTIVITY": learned_sel,
        "KDA_LEARNED_SCALE_CAUSAL_CLOSURE": learned_closure,
        "KDA_GATE_CONTEXT": gate_context,
        "KDA_GATE_SELECTIVITY": gate_sel,
        "KDA_GATE_CAUSAL_CLOSURE": gate_closure,
        "KDA_O_NORM_GATED_RMS_PRIMARY_CAUSE": primary,
        "KDA_O_NORM_GATED_RMS_CAUSAL_CLOSURE": causal_closure,
        "KDA_MODEL_NORM_SECONDARY_ATTENUATION": model_norm_secondary,
        "KDA_SAME_TOKEN_LOGIT_ROTATION_GAIN": same_token,
        "KDA_MECHANISM_CLOSURE_STATUS": mechanism_status,
        "FINAL_CLASSIFICATION": final_cls,
        "NEXT_EXPERIMENT": next_exp,
        "n_units_completed": len(unit_done),
        "expected_units": len(units),
        "kda_layers": list(map(int, kda_layers)),
        "n_internal_rows": len(internal_rows),
        "n_downstream_rows": len(downstream_rows),
        "failures": failures,
        "environment": env,
        "primary_values": {
            "t64_rms_median_DeltaG": rms_delta,
            "t64_learned_scale_median_DeltaG": learned_delta,
            "t64_gate_median_DeltaG": gate_delta,
            "t64_jacobian_median_Lambda": jac_lam,
            "t64_jacobian_median_max_linearization_error": jac_err,
            "t64_frozen_rms_gain": frozen_rms_gain,
            "t64_frozen_module_gain": frozen_module_gain,
            "t64_real_pure_rms_gain": real_pure_gain,
            "t64_real_module_gain": real_module_gain,
            "t64_real_core_gain": real_core_gain,
            "t64_learned_scale_Lambda": learned_lam,
            "t64_gate_Lambda": gate_lam,
        },
    }
    save_json(outdir / "summary.json", summary)
    write_rows(outdir / "loss_budget.csv", loss_rows)
    save_json(outdir / "manifest.json", {
        "task": TASK,
        "created_at": now(),
        "artifact_paths": sorted(str(p) for p in outdir.rglob("*") if p.is_file()),
    })
    make_plots(outdir, consistency, internal_summary, internal_delta, internal_transfer_summary, rms_geometry_rows, rms_jacobian_rows, frozen_summary, loss_rows, downstream_summary)
    write_report(outdir, summary, audit, horizon_rows, internal_summary, internal_delta, internal_transfer_summary, downstream_summary, rms_geometry_rows, rms_jacobian_rows, loss_rows)
    return summary


def make_plots(outdir, consistency, internal_summary, internal_delta, internal_transfer, rms_geom, rms_jac, frozen_summary, loss_rows, downstream_summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        save_json(outdir / "figures" / "manifest.json", {"status": "SKIPPED", "reason": repr(exc), "plots": []})
        return
    made = []
    rows = read_jsonl(outdir / "metric_consistency" / "atomic_identity.jsonl")
    if rows:
        xs = [as_float(r.get("identity_lhs"), 0.0) for r in rows]
        ys = [as_float(r.get("identity_rhs"), 0.0) for r in rows]
        plt.figure(figsize=(4, 4))
        plt.scatter(xs, ys, s=8)
        lim = max(xs + ys + [1e-12])
        plt.plot([0, lim], [0, lim], color="black", linewidth=1)
        plt.xlabel("1 - G_j")
        plt.ylabel("Lambda^2(1 - G_prev)")
        plt.tight_layout()
        p = outdir / "figures" / "atomic_identity_closure.png"
        plt.savefig(p, dpi=150)
        plt.close()
        made.append(str(p))
    for horizon in HORIZONS:
        srows = [r for r in internal_summary if int(r["horizon"]) == horizon]
        xs = [r["stage"] for r in srows]
        ys = [as_float(r["median_G_j"], 0.0) for r in srows]
        plt.figure(figsize=(7, 3))
        plt.plot(range(len(xs)), ys, marker="o")
        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(range(len(xs)), xs, rotation=25, ha="right", fontsize=7)
        plt.ylabel("median G")
        plt.tight_layout()
        p = outdir / "figures" / f"internal_gain_t{horizon}.png"
        plt.savefig(p, dpi=150)
        plt.close()
        made.append(str(p))
    drows = [r for r in internal_delta if int(r["horizon"]) == PRIMARY_HORIZON]
    if drows:
        labels = [f"{r['from_stage']}->{r['to_stage']}" for r in drows]
        vals = [as_float(r["median_DeltaG"], 0.0) for r in drows]
        plt.figure(figsize=(7, 3))
        plt.bar(range(len(vals)), vals)
        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(range(len(vals)), labels, rotation=25, ha="right", fontsize=7)
        plt.ylabel("median DeltaG")
        plt.tight_layout()
        p = outdir / "figures" / "internal_deltaG_t64.png"
        plt.savefig(p, dpi=150)
        plt.close()
        made.append(str(p))
    trows = [r for r in internal_transfer if int(r["horizon"]) == PRIMARY_HORIZON]
    if trows:
        labels = [f"{r['from_stage']}->{r['to_stage']}" for r in trows]
        vals = [as_float(r["median_Lambda"], 1.0) for r in trows]
        plt.figure(figsize=(7, 3))
        plt.bar(range(len(vals)), vals)
        plt.axhline(1, color="black", linewidth=1)
        plt.xticks(range(len(vals)), labels, rotation=25, ha="right", fontsize=7)
        plt.ylabel("median Lambda")
        plt.tight_layout()
        p = outdir / "figures" / "internal_lambda_t64.png"
        plt.savefig(p, dpi=150)
        plt.close()
        made.append(str(p))
    vals_n = [r["native_radial_fraction"] for r in rms_geom if int(r["horizon"]) == PRIMARY_HORIZON]
    vals_r = [r["rotated_radial_fraction"] for r in rms_geom if int(r["horizon"]) == PRIMARY_HORIZON]
    if vals_n:
        plt.figure(figsize=(5, 3))
        plt.boxplot([vals_n, vals_r], labels=["Native", "Rotated"])
        plt.ylabel("radial fraction")
        plt.tight_layout()
        p = outdir / "figures" / "radial_fraction_t64.png"
        plt.savefig(p, dpi=150)
        plt.close()
        made.append(str(p))
    vals_n = [r["native_tangential_fraction"] for r in rms_geom if int(r["horizon"]) == PRIMARY_HORIZON]
    vals_r = [r["rotated_tangential_fraction"] for r in rms_geom if int(r["horizon"]) == PRIMARY_HORIZON]
    if vals_n:
        plt.figure(figsize=(5, 3))
        plt.boxplot([vals_n, vals_r], labels=["Native", "Rotated"])
        plt.ylabel("tangential fraction")
        plt.tight_layout()
        p = outdir / "figures" / "tangential_fraction_t64.png"
        plt.savefig(p, dpi=150)
        plt.close()
        made.append(str(p))
    vals = [r["Lambda_RMS_J"] for r in rms_jac if int(r["horizon"]) == PRIMARY_HORIZON]
    if vals:
        plt.figure(figsize=(5, 3))
        plt.hist(vals, bins=20)
        plt.axvline(1, color="black", linewidth=1)
        plt.xlabel("RMS Jacobian Lambda")
        plt.tight_layout()
        p = outdir / "figures" / "rms_jacobian_lambda_t64.png"
        plt.savefig(p, dpi=150)
        plt.close()
        made.append(str(p))
    frows = [r for r in frozen_summary if int(r["horizon"]) == PRIMARY_HORIZON]
    if frows:
        plt.figure(figsize=(8, 3))
        plt.bar(range(len(frows)), [as_float(r["median_G_j"], 0.0) for r in frows])
        plt.xticks(range(len(frows)), [r["stage"] for r in frows], rotation=25, ha="right", fontsize=6)
        plt.ylabel("median G")
        plt.tight_layout()
        p = outdir / "figures" / "frozen_denominator_downstream_t64.png"
        plt.savefig(p, dpi=150)
        plt.close()
        made.append(str(p))
    if loss_rows:
        plt.figure(figsize=(5, 3))
        plt.bar([r["component"] for r in loss_rows], [as_float(r["median_DeltaG"], 0.0) for r in loss_rows])
        plt.axhline(0, color="black", linewidth=1)
        plt.ylabel("median DeltaG")
        plt.tight_layout()
        p = outdir / "figures" / "first_attenuation_loss_budget.png"
        plt.savefig(p, dpi=150)
        plt.close()
        made.append(str(p))
    lrows = [r for r in downstream_summary if int(r["horizon"]) == PRIMARY_HORIZON and r["stage"] in {"layer_output", "model_norm", "lm_head_logits"}]
    if lrows:
        plt.figure(figsize=(5, 3))
        plt.plot(range(len(lrows)), [as_float(r["median_G_j"], 0.0) for r in lrows], marker="o")
        plt.xticks(range(len(lrows)), [r["stage"] for r in lrows], rotation=20, ha="right", fontsize=7)
        plt.ylabel("median G")
        plt.tight_layout()
        p = outdir / "figures" / "secondary_model_norm_and_logits_t64.png"
        plt.savefig(p, dpi=150)
        plt.close()
        made.append(str(p))
    save_json(outdir / "figures" / "manifest.json", {"status": "OK", "plots": made})


def markdown_table(rows, fields, limit=None):
    rows = rows[:limit] if limit else rows
    out = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        out.append("|" + "|".join(str(row.get(f, "")) for f in fields) + "|")
    return "\n".join(out)


def write_report(outdir, summary, audit, horizon_rows, internal_summary, internal_delta, internal_transfer, downstream_summary, rms_geom, rms_jac, loss_rows):
    lines = [
        f"# {TASK}",
        "",
        "## Classifications",
        "",
        "```text",
        f"KDA_STAGE_TRANSFER_METRIC_CONSISTENCY = {summary.get('KDA_STAGE_TRANSFER_METRIC_CONSISTENCY')}",
        f"KDA_O_NORM_GATED_RMS_PATH_AUDIT = {summary.get('KDA_O_NORM_GATED_RMS_PATH_AUDIT')}",
        f"KDA_O_NORM_FIRST_INTERNAL_ATTENUATION_STAGE = {summary.get('KDA_O_NORM_FIRST_INTERNAL_ATTENUATION_STAGE')}",
        f"KDA_RMS_GEOMETRY_DIFFERENCE = {summary.get('KDA_RMS_GEOMETRY_DIFFERENCE')}",
        f"KDA_RMS_JACOBIAN_PREDICTS_ATTENUATION = {summary.get('KDA_RMS_JACOBIAN_PREDICTS_ATTENUATION')}",
        f"KDA_FROZEN_DENOMINATOR_RMS_RESCUE = {summary.get('KDA_FROZEN_DENOMINATOR_RMS_RESCUE')}",
        f"KDA_RMS_NORMALIZATION_CAUSAL_CLOSURE = {summary.get('KDA_RMS_NORMALIZATION_CAUSAL_CLOSURE')}",
        f"KDA_LEARNED_SCALE_SELECTIVITY = {summary.get('KDA_LEARNED_SCALE_SELECTIVITY')}",
        f"KDA_LEARNED_SCALE_CAUSAL_CLOSURE = {summary.get('KDA_LEARNED_SCALE_CAUSAL_CLOSURE')}",
        f"KDA_GATE_CONTEXT = {summary.get('KDA_GATE_CONTEXT')}",
        f"KDA_GATE_SELECTIVITY = {summary.get('KDA_GATE_SELECTIVITY')}",
        f"KDA_GATE_CAUSAL_CLOSURE = {summary.get('KDA_GATE_CAUSAL_CLOSURE')}",
        f"KDA_O_NORM_GATED_RMS_PRIMARY_CAUSE = {summary.get('KDA_O_NORM_GATED_RMS_PRIMARY_CAUSE')}",
        f"KDA_O_NORM_GATED_RMS_CAUSAL_CLOSURE = {summary.get('KDA_O_NORM_GATED_RMS_CAUSAL_CLOSURE')}",
        f"KDA_MODEL_NORM_SECONDARY_ATTENUATION = {summary.get('KDA_MODEL_NORM_SECONDARY_ATTENUATION')}",
        f"KDA_SAME_TOKEN_LOGIT_ROTATION_GAIN = {summary.get('KDA_SAME_TOKEN_LOGIT_ROTATION_GAIN')}",
        f"KDA_MECHANISM_CLOSURE_STATUS = {summary.get('KDA_MECHANISM_CLOSURE_STATUS')}",
        f"FINAL_CLASSIFICATION = {summary.get('FINAL_CLASSIFICATION')}",
        f"NEXT_EXPERIMENT = {summary.get('NEXT_EXPERIMENT')}",
        "```",
        "",
        f"Canonical units completed: {summary.get('n_units_completed')} / {summary.get('expected_units')}.",
        f"Manual o_norm parity max relative error: {summary.get('KDA_O_NORM_MANUAL_MODULE_PARITY_MAX_REL')}.",
        "",
        "## Internal Path",
        "",
        " -> ".join(stage["stage_name"] for stage in audit.get("stages", [])),
        "",
        "## Atomic Metric Consistency",
        "",
        "```json",
        json.dumps(summary.get("metric_consistency", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Horizon Summary",
        "",
        markdown_table(horizon_rows, ["horizon", "n_units", "median_core_readout_G", "median_o_norm_module_G", "median_logit_G"]),
        "",
        "## Internal Stage Gains t64",
        "",
        markdown_table([r for r in internal_summary if int(r["horizon"]) == PRIMARY_HORIZON], ["stage", "n_units", "median_G_j", "wins_G_j_gt_0", "bootstrap95_G_j"]),
        "",
        "## Internal DeltaG t64",
        "",
        markdown_table([r for r in internal_delta if int(r["horizon"]) == PRIMARY_HORIZON], ["from_stage", "to_stage", "median_DeltaG", "wins_DeltaG_lt_0", "bootstrap95_DeltaG"]),
        "",
        "## Internal Lambda t64",
        "",
        markdown_table([r for r in internal_transfer if int(r["horizon"]) == PRIMARY_HORIZON], ["from_stage", "to_stage", "median_Lambda", "wins_Lambda_gt_1", "bootstrap95_Lambda"]),
        "",
        "## Loss Budget",
        "",
        markdown_table(loss_rows, ["component", "median_DeltaG"]),
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(summary.get("environment", {}), indent=2, sort_keys=True),
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
    formal_gate_applies = args.scope == "formal" and args.max_units is None and args.target_layer_limit is None
    if formal_gate_applies:
        if summary.get("KDA_STAGE_TRANSFER_METRIC_CONSISTENCY") != "PASS":
            return 2
        if summary.get("KDA_O_NORM_GATED_RMS_PATH_AUDIT") != "PASS":
            return 2
        if summary.get("n_units_completed") != summary.get("expected_units"):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
