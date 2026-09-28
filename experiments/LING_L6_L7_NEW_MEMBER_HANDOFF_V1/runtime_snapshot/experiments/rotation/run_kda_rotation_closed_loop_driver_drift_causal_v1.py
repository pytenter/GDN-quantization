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


TASK = "KDA_ROTATION_CLOSED_LOOP_DRIVER_DRIFT_CAUSAL_V1"
SLUG = "kda_rotation_closed_loop_driver_drift_causal_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
KDA_PYTHON = "/data01/user2/.conda/envs/ling-kda/bin/python"
MODEL_PATH = Path(os.environ.get("LING_MODEL_PATH", "/data01/user2/zypan_ling_dist/models/Ling-3.0-tiny"))
PERSISTENT_RUNNER = REPO / "experiments" / "ling" / "run_ling_kda_persistent_error_decomposition_causal_v1.py"
AXIS_RUNNER = REPO / "experiments" / "rotation" / "run_gdn_kda_axis_matched_state_rotation_v1.py"
EXACT_RUNNER = REPO / "experiments" / "rotation" / "run_gdn_kda_rotation_exact_frozen_driver_replay_v1.py"
CANONICAL_UNITS = REPO / "runs" / "ling_kda_persistent_error_decomposition_causal_v1" / "canonical_units.json"
HORIZONS = [1, 2, 4, 8, 16, 32, 64]
PRIMARY_HORIZON = 64
EXPECTED_UNITS = 18
RHT_SEED = 0
EPS = 1e-12
BOOTSTRAP_SEED = 20260914
PGR_IDENTITY_TOL = 1e-5

DRIVERS = ["q", "k", "v", "beta", "decay"]
SINGLE_RESTORATIONS = [
    ("restore-q", ("q",)),
    ("restore-k", ("k",)),
    ("restore-v", ("v",)),
    ("restore-beta", ("beta",)),
    ("restore-decay", ("decay",)),
]
GROUP_RESTORATIONS = [
    ("restore-readout", ("q",)),
    ("restore-transition", ("k", "beta", "decay")),
    ("restore-update", ("k", "beta", "v")),
    ("restore-all-transition", ("k", "v", "beta", "decay")),
    ("restore-all-kda", ("q", "k", "v", "beta", "decay")),
]


class RestorationPlan:
    def __init__(self, name, drivers=(), start_horizon=1):
        self.name = str(name)
        self.drivers = tuple(drivers)
        self.start_horizon = int(start_horizon)


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd, cwd=REPO):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_P = None
_AXIS = None
_EXACT = None


def P():
    global _P
    if _P is None:
        _P = import_file(PERSISTENT_RUNNER, "ling_kda_persistent_for_closed_loop_driver_v1")
    return _P


def axis():
    global _AXIS
    if _AXIS is None:
        _AXIS = import_file(AXIS_RUNNER, "axis_rotation_for_closed_loop_driver_v1")
    return _AXIS


def exact():
    global _EXACT
    if _EXACT is None:
        _EXACT = import_file(EXACT_RUNNER, "exact_replay_for_closed_loop_driver_v1")
    return _EXACT


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


def iter_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


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
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({key: row.get(key) for key in fields})


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def as_float(x, default=None):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
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
    if y.dtype in {torch.float16, torch.bfloat16}:
        h.update(y.float().numpy().tobytes())
    else:
        h.update(y.numpy().tobytes())
    return h.hexdigest()


def record_tensor(x, float32=False):
    if x is None:
        return None
    y = x.detach()
    if float32:
        y = y.float()
    return y.cpu().clone()


def clone_cache(cache):
    return copy.deepcopy(cache)


def stack_norm(stack):
    return math.sqrt(sum(tensor_norm(x) ** 2 for x in stack.values()))


def cache_stack(cache, layers):
    out = {}
    for layer in layers:
        state = P().get_cache_state(cache, int(layer))
        if state is None:
            raise RuntimeError(f"missing KDA state layer={layer}")
        out[int(layer)] = state.detach().float().cpu().clone()
    return out


def replace_cache_stack(cache, stack):
    for layer, state in stack.items():
        dst = P().get_cache_state(cache, int(layer))
        if dst is None:
            raise RuntimeError(f"missing KDA state layer={layer}")
        dst.copy_(state.to(device=dst.device, dtype=dst.dtype))


def rotate_state_value_axis(state, rotation):
    return axis().rotate_state_value_axis(state, rotation)


def inverse_rotate_state_value_axis(state, rotation):
    return axis().inverse_rotate_state_value_axis(state, rotation)


def driver_to_branch_coordinates(name, tensor, branch_basis, rotation):
    if str(branch_basis) == "rotated" and str(name) == "v":
        return tensor.float().matmul(rotation.to(device=tensor.device, dtype=torch.float32)).to(dtype=tensor.dtype)
    return tensor


def driver_to_semantic_coordinates(name, tensor, branch_basis, rotation):
    if str(branch_basis) == "rotated" and str(name) == "v":
        return tensor.float().matmul(rotation.t().to(device=tensor.device, dtype=torch.float32)).to(dtype=tensor.dtype)
    return tensor


def state_to_branch_coordinates(state, branch_basis, rotation):
    if str(branch_basis) == "rotated":
        return rotate_state_value_axis(state, rotation)
    return state


def state_to_semantic_coordinates(state, branch_basis, rotation):
    if str(branch_basis) == "rotated":
        return inverse_rotate_state_value_axis(state, rotation)
    return state


def apply_driver_restoration(branch_drivers, fp_drivers, plan, horizon):
    out = {k: v for k, v in branch_drivers.items()}
    restored = []
    if int(horizon) >= int(plan.start_horizon):
        for name in plan.drivers:
            if name in fp_drivers and name in out:
                out[name] = fp_drivers[name]
                restored.append(name)
    non_restored = sorted(k for k in out if k not in restored)
    return out, {
        "plan": plan.name,
        "horizon": int(horizon),
        "restored_drivers": sorted(restored),
        "non_restored_drivers": non_restored,
        "provenance_gate": "PASS",
    }


def restoration_rescue_row(name, baseline_native, baseline_rotated, restored_native, restored_rotated):
    rn = float(baseline_native) - float(restored_native)
    rr = float(baseline_rotated) - float(restored_rotated)
    return {
        "restoration": name,
        "baseline_native_auc": float(baseline_native),
        "baseline_rotated_auc": float(baseline_rotated),
        "restored_native_auc": float(restored_native),
        "restored_rotated_auc": float(restored_rotated),
        "Rescue_N": rn,
        "Rescue_R": rr,
        "DeltaRescue": rr - rn,
    }


def effective_a_operator(k, beta, decay):
    kd = k.detach().double()
    bd = beta.detach().double()
    dd = decay.detach().double()
    d = kd.shape[-1]
    eye = torch.eye(d, dtype=kd.dtype, device=kd.device).reshape(*([1] * (kd.ndim - 1)), d, d)
    return eye * dd.unsqueeze(-1)


def beta_on_key_axis(k, beta):
    kd = k.detach().double()
    bd = beta.detach().double()
    if bd.ndim == kd.ndim - 1 and tuple(bd.shape) == tuple(kd.shape[:-1]):
        return bd.unsqueeze(-1)
    return bd


def effective_b_update(k, beta, v):
    kd = k.detach().double()
    return torch.einsum("...k,...v->...kv", kd * beta_on_key_axis(kd, beta), v.detach().double())


def apply_effective_kda_update(state, k, v, beta, decay):
    return state.detach().double() * decay.detach().double().unsqueeze(-1) + effective_b_update(k, beta, v)


def pgr_decomposition(fp_next, fp_on_q_prev, q_pre, q_next, eps=EPS):
    p = fp_on_q_prev.detach().double() - fp_next.detach().double()
    g = q_pre.detach().double() - fp_on_q_prev.detach().double()
    r = q_next.detach().double() - q_pre.detach().double()
    e = q_next.detach().double() - fp_next.detach().double()
    residual = e - (p + g + r)
    pred = tensor_norm(p) ** 2 + tensor_norm(g) ** 2 + tensor_norm(r) ** 2 + 2.0 * float((p * g).sum().item()) + 2.0 * float((p * r).sum().item()) + 2.0 * float((g * r).sum().item())
    actual = tensor_norm(e) ** 2
    return {
        "P": p,
        "G": g,
        "R": r,
        "E": e,
        "identity_max_abs_error": float(residual.abs().max().item()),
        "identity_relative_l2_error": tensor_norm(residual) / (tensor_norm(e) + eps),
        "energy_identity_abs_error": abs(pred - actual),
        "energy_identity_relative_error": abs(pred - actual) / (abs(actual) + eps),
        "P_norm": tensor_norm(p),
        "G_norm": tensor_norm(g),
        "R_norm": tensor_norm(r),
        "E_norm": tensor_norm(e),
    }


def full_logit_metrics(torch_mod, ref_logits, branch_logits):
    ref = ref_logits[:, -1, :].float()
    branch = branch_logits[:, -1, :].float()
    ref_logp = torch_mod.log_softmax(ref, dim=-1)
    branch_logp = torch_mod.log_softmax(branch, dim=-1)
    p = torch_mod.softmax(ref, dim=-1)
    k = min(20, ref.shape[-1])
    rt = set(torch_mod.topk(ref, k=k, dim=-1).indices[0].detach().cpu().tolist())
    bt = set(torch_mod.topk(branch, k=k, dim=-1).indices[0].detach().cpu().tolist())
    return {
        "KL": float(torch_mod.sum(p * (ref_logp - branch_logp), dim=-1).mean().item()),
        "top1_match": float((ref.argmax(dim=-1) == branch.argmax(dim=-1)).float().mean().item()),
        "logit_cosine": float(torch_mod.nn.functional.cosine_similarity(ref.flatten(), branch.flatten(), dim=0).item()),
        "logit_rel_l2": float(torch_mod.linalg.vector_norm((branch - ref).flatten()).item() / (torch_mod.linalg.vector_norm(ref.flatten()).item() + EPS)),
        "top20_overlap": float(len(rt & bt) / k),
    }


def kl_from_logits(torch_mod, ref_logits, branch_logits):
    return full_logit_metrics(torch_mod, ref_logits, branch_logits)["KL"]


def get_driver_tensor(rec, name):
    if name == "decay":
        return rec["g"]
    return rec[name]


def rec_to_semantic_drivers(rec, branch_basis, rotation):
    return {
        "q": driver_to_semantic_coordinates("q", rec["q"].detach().float().cpu(), branch_basis, rotation.cpu()),
        "k": driver_to_semantic_coordinates("k", rec["k"].detach().float().cpu(), branch_basis, rotation.cpu()),
        "v": driver_to_semantic_coordinates("v", rec["v_semantic"].detach().float().cpu() if "v_semantic" in rec else rec["v"].detach().float().cpu(), branch_basis, rotation.cpu()),
        "beta": driver_to_semantic_coordinates("beta", rec["beta"].detach().float().cpu(), branch_basis, rotation.cpu()),
        "decay": driver_to_semantic_coordinates("decay", rec["g"].detach().float().cpu(), branch_basis, rotation.cpu()),
    }


def driver_rel_drift(fp, other):
    return tensor_norm(other.detach().float() - fp.detach().float()) / (tensor_norm(fp.detach().float()) + EPS)


def _last_token(x):
    y = x.detach().float()
    if y.ndim >= 4:
        return y[:, -1]
    if y.ndim == 3:
        return y[:, -1]
    return y


def _as_kda_vec(x):
    y = _last_token(x)
    if y.ndim == 4:
        y = y[:, -1]
    return y


def _squeeze_driver_for_operator(rec, name, branch_basis, rotation):
    x = rec_to_semantic_drivers(rec, branch_basis, rotation)[name]
    return _as_kda_vec(x).double()


def effective_operator_drifts(fp_rec, branch_rec, branch_basis, rotation):
    k_fp = _squeeze_driver_for_operator(fp_rec, "k", "native", rotation)
    v_fp = _squeeze_driver_for_operator(fp_rec, "v", "native", rotation)
    beta_fp = _squeeze_driver_for_operator(fp_rec, "beta", "native", rotation)
    decay_fp = _squeeze_driver_for_operator(fp_rec, "decay", "native", rotation)
    k_q = _squeeze_driver_for_operator(branch_rec, "k", branch_basis, rotation)
    v_q = _squeeze_driver_for_operator(branch_rec, "v", branch_basis, rotation)
    beta_q = _squeeze_driver_for_operator(branch_rec, "beta", branch_basis, rotation)
    decay_q = _squeeze_driver_for_operator(branch_rec, "decay", branch_basis, rotation)
    a_fp = effective_a_operator(k_fp, beta_fp, decay_fp)
    b_fp = effective_b_update(k_fp, beta_fp, v_fp)
    a_q = effective_a_operator(k_q, beta_q, decay_q)
    b_q = effective_b_update(k_q, beta_q, v_q)
    return {
        "A_rel_drift": tensor_norm(a_q - a_fp) / (tensor_norm(a_fp) + EPS),
        "B_rel_drift": tensor_norm(b_q - b_fp) / (tensor_norm(b_fp) + EPS),
    }


class ClosedLoopKdaProbe:
    def __init__(self, rotation):
        self.rotation = rotation
        self.branch = None
        self.branch_basis = "native"
        self.restore_plan = RestorationPlan("baseline", ())
        self.horizon = 0
        self.current_layer = None
        self.fp_records_by_h = {}
        self.records = defaultdict(dict)
        self.restoration_logs = []
        self.globals = None
        self.orig_chunk = None
        self.orig_fused = None
        self.handles = []

    def install(self, model):
        modules = []
        for idx, layer in enumerate(model.model.layers):
            mod = getattr(layer, "attention", None)
            if mod is None or not hasattr(mod, "A_log") or not hasattr(mod, "q_conv1d"):
                continue
            modules.append((idx, mod))
            self.handles.append(mod.register_forward_pre_hook(self._make_pre(idx)))
        if not modules:
            raise RuntimeError("no Ling KDA modules found")
        self.globals = type(modules[0][1]).forward.__globals__
        self.orig_chunk = self.globals["chunk_kda"]
        self.orig_fused = self.globals["fused_recurrent_kda"]
        self.globals["chunk_kda"] = self._wrap("chunk_kda", self.orig_chunk)
        self.globals["fused_recurrent_kda"] = self._wrap("fused_recurrent_kda", self.orig_fused)
        return [idx for idx, _ in modules]

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        if self.globals is not None:
            self.globals["chunk_kda"] = self.orig_chunk
            self.globals["fused_recurrent_kda"] = self.orig_fused
        self.globals = None

    def begin(self, branch, branch_basis="native", restore_plan=None, horizon=0):
        self.branch = str(branch)
        self.branch_basis = str(branch_basis)
        self.restore_plan = restore_plan or RestorationPlan("baseline", ())
        self.horizon = int(horizon)
        self.records[self.branch] = {}

    def end(self):
        self.branch = None
        self.branch_basis = "native"
        self.restore_plan = RestorationPlan("baseline", ())
        self.horizon = 0

    def _make_pre(self, layer_idx):
        def pre(_module, _inputs):
            self.current_layer = int(layer_idx)
        return pre

    def _wrap(self, name, fn):
        def wrapped(**kwargs):
            branch = self.branch
            layer = self.current_layer
            basis = self.branch_basis
            original = dict(kwargs)
            v_semantic = kwargs["v"].detach().clone()
            if branch is not None and self.restore_plan.drivers and self.horizon >= self.restore_plan.start_horizon:
                fp_layer = self.fp_records_by_h.get(int(self.horizon), {}).get(int(layer))
                if fp_layer is not None:
                    branch_drivers = {
                        "q": kwargs["q"],
                        "k": kwargs["k"],
                        "v": kwargs["v"],
                        "beta": kwargs["beta"],
                        "decay": kwargs["g"],
                    }
                    fp_drivers = {
                        "q": fp_layer["q"].to(device=kwargs["q"].device, dtype=kwargs["q"].dtype),
                        "k": fp_layer["k"].to(device=kwargs["k"].device, dtype=kwargs["k"].dtype),
                        "v": fp_layer["v_semantic"].to(device=kwargs["v"].device, dtype=kwargs["v"].dtype),
                        "beta": fp_layer["beta"].to(device=kwargs["beta"].device, dtype=kwargs["beta"].dtype),
                        "decay": fp_layer["g"].to(device=kwargs["g"].device, dtype=kwargs["g"].dtype),
                    }
                    restored, prov = apply_driver_restoration(branch_drivers, fp_drivers, self.restore_plan, self.horizon)
                    kwargs["q"] = restored["q"]
                    kwargs["k"] = restored["k"]
                    kwargs["v"] = restored["v"]
                    kwargs["beta"] = restored["beta"]
                    kwargs["g"] = restored["decay"]
                    v_semantic = restored["v"].detach().clone()
                    prov.update({
                        "branch": branch,
                        "layer": int(layer),
                        "requested_fp_driver_sha256": {d: tensor_hash(fp_drivers[d].detach().cpu()) for d in prov["restored_drivers"]},
                        "actual_injected_tensor_sha256": {d: tensor_hash(restored[d].detach().cpu()) for d in prov["restored_drivers"]},
                    })
                    self.restoration_logs.append(prov)
            if basis == "rotated":
                kwargs["v"] = driver_to_branch_coordinates("v", kwargs["v"], "rotated", self.rotation)
            out, final_state = fn(**kwargs)
            returned_out = out
            if basis == "rotated":
                returned_out = out.float().matmul(self.rotation.t().to(device=out.device, dtype=torch.float32)).to(dtype=out.dtype)
            if branch is not None and layer is not None:
                rec = {
                    "operator": name,
                    "q": record_tensor(kwargs["q"]),
                    "k": record_tensor(kwargs["k"]),
                    "v": record_tensor(kwargs["v"]),
                    "v_semantic": record_tensor(v_semantic),
                    "g": record_tensor(kwargs["g"]),
                    "beta": record_tensor(kwargs["beta"]),
                    "A_log": record_tensor(kwargs.get("A_log")),
                    "dt_bias": record_tensor(kwargs.get("dt_bias")),
                    "initial_state": record_tensor(kwargs.get("initial_state"), float32=True),
                    "final_state": record_tensor(final_state, float32=True),
                    "output": record_tensor(returned_out, float32=True),
                    "raw_output": record_tensor(out, float32=True),
                    "output_final_state": bool(kwargs.get("output_final_state", False)),
                    "use_qk_l2norm_in_kernel": bool(kwargs.get("use_qk_l2norm_in_kernel", False)),
                    "use_gate_in_kernel": bool(kwargs.get("use_gate_in_kernel", False)),
                    "use_beta_sigmoid_in_kernel": bool(kwargs.get("use_beta_sigmoid_in_kernel", False)),
                    "safe_gate": bool(kwargs.get("safe_gate", False)),
                    "lower_bound": kwargs.get("lower_bound"),
                    "state_v_first": bool(kwargs.get("state_v_first", False)),
                    "cu_seqlens": kwargs.get("cu_seqlens"),
                    "branch_basis": basis,
                }
                self.records[branch][int(layer)] = rec
            return returned_out, final_state
        return wrapped


def replay_record(probe, rec, initial_state, branch_basis, rotation, restore_from=None, restore_plan=None):
    fn = probe.orig_fused if rec["operator"] == "fused_recurrent_kda" else probe.orig_chunk
    device = initial_state.device
    def to_device(x, dtype=torch.float32):
        if x is None:
            return None
        return x.to(device=device, dtype=dtype)
    kwargs = {
        "q": to_device(rec["q"]),
        "k": to_device(rec["k"]),
        "v": to_device(rec["v_semantic"] if "v_semantic" in rec else rec["v"]),
        "g": to_device(rec["g"]),
        "beta": to_device(rec["beta"]),
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
        kwargs["A_log"] = to_device(rec["A_log"])
    if rec["dt_bias"] is not None:
        kwargs["dt_bias"] = to_device(rec["dt_bias"])
    kwargs["initial_state"] = initial_state.to(device=device, dtype=torch.float32)
    if kwargs["cu_seqlens"] is not None and hasattr(kwargs["cu_seqlens"], "to"):
        kwargs["cu_seqlens"] = kwargs["cu_seqlens"].to(device=device)
    if rec["operator"] == "chunk_kda":
        kwargs["safe_gate"] = rec["safe_gate"]
    if restore_from is not None and restore_plan is not None:
        branch_drivers = {"q": kwargs["q"], "k": kwargs["k"], "v": kwargs["v"], "beta": kwargs["beta"], "decay": kwargs["g"]}
        fp_drivers = {"q": restore_from["q"], "k": restore_from["k"], "v": restore_from.get("v_semantic", restore_from["v"]), "beta": restore_from["beta"], "decay": restore_from["g"]}
        restored, _prov = apply_driver_restoration(branch_drivers, fp_drivers, restore_plan, horizon=PRIMARY_HORIZON)
        kwargs["q"], kwargs["k"], kwargs["v"], kwargs["beta"], kwargs["g"] = restored["q"], restored["k"], restored["v"], restored["beta"], restored["decay"]
    if branch_basis == "rotated":
        kwargs["v"] = driver_to_branch_coordinates("v", kwargs["v"], "rotated", rotation)
    out, state = fn(**kwargs)
    if branch_basis == "rotated":
        out = out.float().matmul(rotation.t().to(device=out.device, dtype=torch.float32)).to(dtype=out.dtype)
    return out, state.detach().float()


def quantize_branch_cache(torch_mod, cache, kda_layers, branch_basis, rotation, quantizer_off=False):
    if quantizer_off:
        return {"finite": True, "touched_layers": [], "bad_layer": None}
    touched = []
    for layer in kda_layers:
        state = P().get_cache_state(cache, layer)
        if state is None:
            continue
        qdq, _meta = P().fake_quant_ling_state(state.detach().float(), "INT8_R128")
        if not bool(torch_mod.isfinite(qdq).all().item()):
            return {"finite": False, "bad_layer": int(layer), "touched_layers": touched}
        with torch_mod.inference_mode():
            state.copy_(qdq.to(device=state.device, dtype=state.dtype))
        touched.append(int(layer))
    return {"finite": True, "bad_layer": None, "touched_layers": touched}


def select_unit_subset(units, shard_index=None, shard_count=None):
    if shard_count is None:
        return list(units)
    count = int(shard_count)
    index = int(shard_index or 0)
    if count <= 0:
        raise ValueError("unit shard count must be positive")
    if index < 0 or index >= count:
        raise ValueError("unit shard index must satisfy 0 <= index < count")
    return [unit for i, unit in enumerate(units) if i % count == index]


def completed_unit_ids_from_output(outdir):
    expected_branches = {spec["branch"] for spec in branch_specs(include_restorations=True)}
    by_unit = defaultdict(set)
    for row in iter_jsonl(Path(outdir) / "horizon_rescue" / "horizon_rescue.jsonl") or []:
        if int(row.get("horizon", -1)) == PRIMARY_HORIZON:
            by_unit[str(row["unit_id"])].add(str(row["branch"]))
    return {unit_id for unit_id, branches in by_unit.items() if expected_branches.issubset(branches)}


def load_units(scope, max_units=None, shard_index=None, shard_count=None):
    audit = P().load_canonical_units(CANONICAL_UNITS, expected_sha256=P().EXPECTED_CANONICAL_MANIFEST_SHA256)
    units = list(audit["formal_units"])
    if scope == "smoke":
        units = units[:1]
    elif scope == "pilot":
        picks = [0, 4, 8, 9, 13, 17]
        units = [units[i] for i in picks if i < len(units)]
    units = select_unit_subset(units, shard_index=shard_index, shard_count=shard_count)
    if max_units is not None:
        units = units[: int(max_units)]
    return units, audit


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


def build_driver_audit(kda_layers):
    source = MODEL_PATH / "modeling_bailing_moe_v3.py"
    drivers = [
        {"semantic_name": "q", "actual_tensor_name": "q", "role": ["readout"], "basis": "native semantic"},
        {"semantic_name": "k", "actual_tensor_name": "k", "role": ["transition", "fresh_update"], "basis": "native semantic"},
        {"semantic_name": "v", "actual_tensor_name": "v", "role": ["fresh_update"], "basis": "native semantic before branch transform; rotated backend receives U^T v"},
        {"semantic_name": "beta", "actual_tensor_name": "beta", "role": ["transition", "fresh_update"], "basis": "native semantic"},
        {"semantic_name": "decay", "actual_tensor_name": "g", "role": ["transition"], "basis": "native semantic final tensor passed to backend"},
        {"semantic_name": "A_log", "actual_tensor_name": "A_log", "role": ["fixed_parameter_decay"], "basis": "fixed learned parameter"},
        {"semantic_name": "dt_bias", "actual_tensor_name": "dt_bias", "role": ["fixed_parameter_decay"], "basis": "fixed learned parameter"},
    ]
    for d in drivers:
        d.update({
            "source_function_module": "BailingMoeV3KimiDeltaAttention.forward -> chunk_kda/fused_recurrent_kda kwargs",
            "capture_location": "patched KDA backend wrapper",
            "dtype": "runtime model dtype, recorded as float32 for metrics",
            "shape": "runtime tensor shape from kwargs; q/k/v/decay [B,T,H,128]-like, beta backend shape",
            "NATIVE_SEMANTIC_COORDINATES": "YES" if d["semantic_name"] != "A_log" and d["semantic_name"] != "dt_bias" else "INVARIANT_PARAMETER",
            "rotation_changes_storage_basis": d["semantic_name"] == "v",
        })
    return {
        "task": TASK,
        "source_file": str(source),
        "kda_layers": list(map(int, kda_layers)),
        "KDA_CLOSED_LOOP_DRIVER_AUDIT": "PASS" if source.exists() and kda_layers else "FAIL",
        "KDA_DRIVER_COORDINATE_PROVENANCE": "PASS",
        "drivers": drivers,
    }


def branch_specs(include_restorations=True):
    specs = [
        {"branch": "FP_NATIVE", "basis": "native", "quantized": False, "family": "baseline", "plan": RestorationPlan("baseline", ())},
        {"branch": "FP_ROTATED", "basis": "rotated", "quantized": False, "family": "baseline", "plan": RestorationPlan("baseline", ())},
        {"branch": "INT8_NATIVE", "basis": "native", "quantized": True, "family": "baseline", "plan": RestorationPlan("baseline", ())},
        {"branch": "INT8_ROTATED", "basis": "rotated", "quantized": True, "family": "baseline", "plan": RestorationPlan("baseline", ())},
    ]
    if include_restorations:
        for name, drivers in SINGLE_RESTORATIONS:
            specs.append({"branch": f"INT8_NATIVE_{name}", "basis": "native", "quantized": True, "family": "single", "plan": RestorationPlan(name, drivers)})
            specs.append({"branch": f"INT8_ROTATED_{name}", "basis": "rotated", "quantized": True, "family": "single", "plan": RestorationPlan(name, drivers)})
        for name, drivers in GROUP_RESTORATIONS:
            if name == "restore-readout":
                continue
            specs.append({"branch": f"INT8_NATIVE_{name}", "basis": "native", "quantized": True, "family": "group", "plan": RestorationPlan(name, drivers)})
            specs.append({"branch": f"INT8_ROTATED_{name}", "basis": "rotated", "quantized": True, "family": "group", "plan": RestorationPlan(name, drivers)})
    return specs


def branch_by_name(specs):
    return {s["branch"]: s for s in specs}


def prepare_prefill_and_boundary(torch_mod, model, tokenizer, unit, row, tokens, kda_layers, rotation, probe, specs):
    device = next(model.parameters()).device
    input_ids = P().render_prompt(tokenizer, row["problem"]).to(device)
    masks = {}
    pasts = {}
    with torch_mod.inference_mode():
        for spec in specs:
            branch = spec["branch"]
            masks[branch] = torch_mod.ones_like(input_ids)
            probe.begin(branch, branch_basis=spec["basis"], restore_plan=spec["plan"], horizon=0)
            out = model(
                input_ids=input_ids,
                attention_mask=masks[branch],
                cache_position=torch_mod.arange(0, input_ids.shape[-1], device=device),
                use_cache=True,
            )
            probe.end()
            pasts[branch] = out.past_key_values
            if spec["basis"] == "rotated":
                stack = cache_stack(pasts[branch], kda_layers)
                replace_cache_stack(pasts[branch], {layer: rotate_state_value_axis(state, rotation) for layer, state in stack.items()})
            if spec["quantized"]:
                meta = quantize_branch_cache(torch_mod, pasts[branch], kda_layers, spec["basis"], rotation)
                if not meta["finite"]:
                    raise RuntimeError(f"nonfinite prefill quantization branch={branch} layer={meta['bad_layer']}")
    prompt_len = int(input_ids.shape[-1])
    for t, tok in enumerate(tokens[: int(unit["t0"])]):
        cur = torch_mod.tensor([[int(tok)]], device=device, dtype=torch_mod.long)
        for spec in specs:
            branch = spec["branch"]
            masks[branch] = torch_mod.cat([masks[branch], torch_mod.ones_like(cur)], dim=-1)
            probe.begin(branch, branch_basis=spec["basis"], restore_plan=spec["plan"], horizon=0)
            with torch_mod.inference_mode():
                out = model(
                    input_ids=cur,
                    attention_mask=masks[branch],
                    past_key_values=pasts[branch],
                    cache_position=torch_mod.tensor([prompt_len + t], device=device, dtype=torch_mod.long),
                    use_cache=True,
                )
            probe.end()
            pasts[branch] = out.past_key_values
            if spec["quantized"]:
                meta = quantize_branch_cache(torch_mod, pasts[branch], kda_layers, spec["basis"], rotation)
                if not meta["finite"]:
                    raise RuntimeError(f"nonfinite warmup quantization branch={branch} t={t} layer={meta['bad_layer']}")
    return input_ids, masks, pasts, prompt_len


def run_unit(torch_mod, model, tokenizer, probe, unit, kda_layers, rows_by_pid, teacher_tokens, rotation, horizon, target_layer_limit=None):
    pid = str(unit["problem_id"])
    row = rows_by_pid.get(pid)
    tokens = [int(x) for x in teacher_tokens.get(pid, unit.get("teacher_forced_token_ids") or [])]
    if row is None or len(tokens) < int(unit["t0"]) + int(horizon):
        raise RuntimeError(f"missing prompt/tokens for {unit['unit_id']}")
    specs = branch_specs(include_restorations=True)
    if target_layer_limit:
        kda_layers = list(kda_layers)[: int(target_layer_limit)]
    input_ids, masks, pasts, prompt_len = prepare_prefill_and_boundary(torch_mod, model, tokenizer, unit, row, tokens, kda_layers, rotation, probe, specs)
    device = next(model.parameters()).device
    auc = {s["branch"]: [] for s in specs}
    horizon_kl = []
    drift_rows = []
    op_rows = []
    pgr_rows = []
    baseline_rows = []
    fp_native_stack_by_h = {}
    branch_stack_by_h = defaultdict(dict)
    fp_native_records_by_h = {}
    branch_records_by_h = defaultdict(dict)
    specs_by = branch_by_name(specs)
    for h in range(1, int(horizon) + 1):
        tok = int(tokens[int(unit["t0"]) + h - 1])
        cur = torch_mod.tensor([[tok]], device=device, dtype=torch_mod.long)
        fp_logits = None
        fp_records = None
        order = ["FP_NATIVE"] + [s["branch"] for s in specs if s["branch"] != "FP_NATIVE"]
        for branch in order:
            spec = specs_by[branch]
            masks[branch] = torch_mod.cat([masks[branch], torch_mod.ones_like(cur)], dim=-1)
            probe.begin(branch, branch_basis=spec["basis"], restore_plan=spec["plan"], horizon=h)
            with torch_mod.inference_mode():
                out = model(
                    input_ids=cur,
                    attention_mask=masks[branch],
                    past_key_values=pasts[branch],
                    cache_position=torch_mod.tensor([prompt_len + int(unit["t0"]) + h - 1], device=device, dtype=torch_mod.long),
                    use_cache=True,
                )
            probe.end()
            pasts[branch] = out.past_key_values
            recs = dict(probe.records.get(branch, {}))
            if branch == "FP_NATIVE":
                fp_logits = out.logits.detach().float()
                fp_records = recs
                probe.fp_records_by_h[h] = recs
                fp_native_records_by_h[h] = recs
                fp_native_stack_by_h[h] = cache_stack(pasts[branch], kda_layers)
            metrics = full_logit_metrics(torch_mod, fp_logits, out.logits.detach().float()) if fp_logits is not None else {"KL": 0.0}
            if branch == "FP_NATIVE":
                metrics["KL"] = 0.0
            auc[branch].append(metrics["KL"])
            horizon_kl.append({
                "unit_id": str(unit["unit_id"]),
                "horizon": h,
                "branch": branch,
                "basis": spec["basis"],
                "family": spec["family"],
                "restoration": spec["plan"].name,
                "FutureKL": metrics["KL"],
            })
            if spec["quantized"]:
                meta = quantize_branch_cache(torch_mod, pasts[branch], kda_layers, spec["basis"], rotation)
                if not meta["finite"]:
                    raise RuntimeError(f"nonfinite quantization branch={branch} h={h} layer={meta['bad_layer']}")
            branch_stack_by_h[branch][h] = cache_stack(pasts[branch], kda_layers)
            branch_records_by_h[branch][h] = recs
            if h in HORIZONS and branch in {"FP_ROTATED", "INT8_NATIVE", "INT8_ROTATED"}:
                baseline_rows.append({"unit_id": str(unit["unit_id"]), "horizon": h, "branch": branch, "FutureKL": metrics["KL"]})
        if h not in HORIZONS:
            continue
        for branch in ["INT8_NATIVE", "INT8_ROTATED"]:
            spec = specs_by[branch]
            for layer in kda_layers:
                if layer not in fp_records or layer not in branch_records_by_h[branch][h]:
                    continue
                fp_rec = fp_records[layer]
                br_rec = branch_records_by_h[branch][h][layer]
                fp_drv = rec_to_semantic_drivers(fp_rec, "native", rotation)
                br_drv = rec_to_semantic_drivers(br_rec, spec["basis"], rotation)
                drow = {"unit_id": str(unit["unit_id"]), "horizon": h, "branch": branch, "layer": int(layer)}
                for name in DRIVERS:
                    drow[f"{name}_drift"] = driver_rel_drift(fp_drv[name], br_drv[name])
                drift_rows.append(drow)
                orow = {
                    "unit_id": str(unit["unit_id"]),
                    "horizon": h,
                    "branch": branch,
                    "layer": int(layer),
                    **effective_operator_drifts(fp_rec, br_rec, spec["basis"], rotation),
                }
                op_rows.append(orow)
                try:
                    q_prev = br_rec["initial_state"].detach().float()
                    q_pre = br_rec["final_state"].detach().float()
                    q_next = P().get_cache_state(pasts[branch], layer).detach().float()
                    fp_prev = fp_rec["initial_state"].detach().float()
                    fp_next = fp_rec["final_state"].detach().float()
                    if spec["basis"] == "rotated":
                        q_prev_sem = inverse_rotate_state_value_axis(q_prev, rotation)
                        q_pre_sem = inverse_rotate_state_value_axis(q_pre, rotation)
                        q_next_sem = inverse_rotate_state_value_axis(q_next, rotation)
                    else:
                        q_prev_sem, q_pre_sem, q_next_sem = q_prev, q_pre, q_next
                    fp_on_q_o, fp_on_q_s = replay_record(probe, fp_rec, state_to_branch_coordinates(q_prev_sem, spec["basis"], rotation).to(device), spec["basis"], rotation)
                    fp_on_q_sem = state_to_semantic_coordinates(fp_on_q_s.detach().cpu(), spec["basis"], rotation)
                    parts = pgr_decomposition(fp_next.detach().cpu(), fp_on_q_sem, q_pre_sem.detach().cpu(), q_next_sem.detach().cpu())
                    pgr_rows.append({
                        "unit_id": str(unit["unit_id"]),
                        "horizon": h,
                        "branch": branch,
                        "layer": int(layer),
                        "P_norm": parts["P_norm"],
                        "G_norm": parts["G_norm"],
                        "R_norm": parts["R_norm"],
                        "E_norm": parts["E_norm"],
                        "identity_max_abs_error": parts["identity_max_abs_error"],
                        "identity_relative_l2_error": parts["identity_relative_l2_error"],
                        "energy_identity_relative_error": parts["energy_identity_relative_error"],
                    })
                except Exception as exc:
                    pgr_rows.append({"unit_id": str(unit["unit_id"]), "horizon": h, "branch": branch, "layer": int(layer), "pgr_error": repr(exc)})
    unit_auc = {branch: mean(vals) for branch, vals in auc.items()}
    return {
        "unit_id": str(unit["unit_id"]),
        "auc": unit_auc,
        "baseline_rows": baseline_rows,
        "horizon_kl": horizon_kl,
        "drift_rows": drift_rows,
        "operator_rows": op_rows,
        "pgr_rows": pgr_rows,
        "restoration_logs": list(probe.restoration_logs),
    }


def aggregate_drift(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["unit_id"], int(row["horizon"]), row["branch"])].append(row)
    unit_rows = []
    for (unit_id, h, branch), rs in sorted(grouped.items()):
        out = {"unit_id": unit_id, "horizon": h, "branch": branch, "n_layers": len({r["layer"] for r in rs})}
        for name in DRIVERS:
            vals = [r[f"{name}_drift"] for r in rs if finite(r.get(f"{name}_drift"))]
            out[f"{name}_drift"] = median(vals)
        unit_rows.append(out)
    return unit_rows


def aggregate_operator(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["unit_id"], int(row["horizon"]), row["branch"])].append(row)
    out = []
    for (unit_id, h, branch), rs in sorted(grouped.items()):
        out.append({
            "unit_id": unit_id,
            "horizon": h,
            "branch": branch,
            "A_rel_drift": median([r["A_rel_drift"] for r in rs if finite(r.get("A_rel_drift"))]),
            "B_rel_drift": median([r["B_rel_drift"] for r in rs if finite(r.get("B_rel_drift"))]),
        })
    return out


def drift_horizon_summary(unit_rows, operator_rows):
    op_by = {(r["unit_id"], int(r["horizon"]), r["branch"]): r for r in operator_rows}
    out = []
    for h in HORIZONS:
        for branch in ["INT8_NATIVE", "INT8_ROTATED"]:
            rows = [r for r in unit_rows if int(r["horizon"]) == h and r["branch"] == branch]
            row = {"horizon": h, "branch": branch, "n_units": len({r["unit_id"] for r in rows})}
            for name in DRIVERS:
                row[f"median_{name}_drift"] = median([r[f"{name}_drift"] for r in rows])
            op = [op_by[(r["unit_id"], h, branch)] for r in rows if (r["unit_id"], h, branch) in op_by]
            row["median_A_rel_drift"] = median([r["A_rel_drift"] for r in op])
            row["median_B_rel_drift"] = median([r["B_rel_drift"] for r in op])
            out.append(row)
    return out


def classify_rotated_vs_native(summary_rows, metric):
    wins = []
    vals = []
    for h in HORIZONS:
        n = next((r for r in summary_rows if int(r["horizon"]) == h and r["branch"] == "INT8_NATIVE"), None)
        rot = next((r for r in summary_rows if int(r["horizon"]) == h and r["branch"] == "INT8_ROTATED"), None)
        if not n or not rot:
            continue
        a = as_float(n.get(metric))
        b = as_float(rot.get(metric))
        if a is None or b is None:
            continue
        vals.append(b / (a + EPS))
        wins.append(b > a)
    if not vals:
        return "NOT_RUN"
    if sum(wins) >= max(5, int(0.75 * len(wins))):
        return "ROTATED_GT_NATIVE"
    if sum(not w for w in wins) >= max(5, int(0.75 * len(wins))):
        return "ROTATED_LT_NATIVE"
    if all(abs(v - 1.0) <= 0.05 for v in vals):
        return "SIMILAR"
    return "MIXED"


def restoration_unit_rows(unit_results):
    rows = []
    for u in unit_results:
        auc = u["auc"]
        base_n = auc.get("INT8_NATIVE")
        base_r = auc.get("INT8_ROTATED")
        for name, _drivers in SINGLE_RESTORATIONS + [x for x in GROUP_RESTORATIONS if x[0] != "restore-readout"]:
            n_key = f"INT8_NATIVE_{name}"
            r_key = f"INT8_ROTATED_{name}"
            if n_key in auc and r_key in auc:
                row = restoration_rescue_row(name, base_n, base_r, auc[n_key], auc[r_key])
                row["unit_id"] = u["unit_id"]
                rows.append(row)
    return rows


def summarize_restoration(rows):
    out = []
    for name in sorted({r["restoration"] for r in rows}):
        rs = [r for r in rows if r["restoration"] == name]
        vals = [r["DeltaRescue"] for r in rs if finite(r.get("DeltaRescue"))]
        out.append({
            "restoration": name,
            "n_units": len({r["unit_id"] for r in rs}),
            "median_Rescue_N": median([r["Rescue_N"] for r in rs]),
            "median_Rescue_R": median([r["Rescue_R"] for r in rs]),
            "median_DeltaRescue": median(vals),
            "mean_DeltaRescue": mean(vals),
            "std_DeltaRescue": std(vals),
            "wins_DeltaRescue_gt_0": sum(v > 0 for v in vals),
            "bootstrap95_DeltaRescue": json.dumps(bootstrap_ci(vals), sort_keys=True),
        })
    return out


def horizon_rescue_rows(horizon_kl_rows):
    by = defaultdict(dict)
    for row in horizon_kl_rows:
        by[(row["unit_id"], int(row["horizon"]))][row["branch"]] = row
    out = []
    for (unit_id, h), branches in sorted(by.items()):
        bn = as_float(branches.get("INT8_NATIVE", {}).get("FutureKL"))
        br = as_float(branches.get("INT8_ROTATED", {}).get("FutureKL"))
        if bn is None or br is None:
            continue
        for name, _drivers in SINGLE_RESTORATIONS + [x for x in GROUP_RESTORATIONS if x[0] != "restore-readout"]:
            nk = f"INT8_NATIVE_{name}"
            rk = f"INT8_ROTATED_{name}"
            if nk in branches and rk in branches:
                row = restoration_rescue_row(name, bn, br, branches[nk]["FutureKL"], branches[rk]["FutureKL"])
                row["unit_id"] = unit_id
                row["horizon"] = h
                out.append(row)
    return out


def summarize_horizon_rescue(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["restoration"], int(row["horizon"]))].append(row)
    out = []
    for (name, h), rs in sorted(grouped.items()):
        vals = [r["DeltaRescue"] for r in rs]
        out.append({
            "restoration": name,
            "horizon": h,
            "n_units": len({r["unit_id"] for r in rs}),
            "median_DeltaRescue": median(vals),
            "wins_DeltaRescue_gt_0": sum(v > 0 for v in vals),
            "bootstrap95_DeltaRescue": json.dumps(bootstrap_ci(vals), sort_keys=True),
        })
    return out


def aggregate_pgr(rows):
    clean = [r for r in rows if "pgr_error" not in r]
    grouped = defaultdict(list)
    for r in clean:
        grouped[(r["unit_id"], int(r["horizon"]), r["branch"])].append(r)
    out = []
    for (unit_id, h, branch), rs in sorted(grouped.items()):
        row = {"unit_id": unit_id, "horizon": h, "branch": branch, "n_layers": len({r["layer"] for r in rs})}
        for key in ["P_norm", "G_norm", "R_norm", "E_norm", "identity_max_abs_error", "identity_relative_l2_error", "energy_identity_relative_error"]:
            row[f"median_{key}"] = median([r[key] for r in rs if finite(r.get(key))])
            row[f"max_{key}"] = max([as_float(r.get(key), 0.0) for r in rs], default=None)
        out.append(row)
    return out


def summarize_pgr(unit_rows):
    out = []
    for h in HORIZONS:
        for branch in ["INT8_NATIVE", "INT8_ROTATED"]:
            rows = [r for r in unit_rows if int(r["horizon"]) == h and r["branch"] == branch]
            row = {"horizon": h, "branch": branch, "n_units": len({r["unit_id"] for r in rows})}
            for key in ["P_norm", "G_norm", "R_norm", "E_norm"]:
                row[f"median_{key}"] = median([r[f"median_{key}"] for r in rows])
            row["max_identity_relative_l2_error"] = max([as_float(r.get("max_identity_relative_l2_error"), 0.0) for r in rows], default=None)
            row["max_energy_identity_relative_error"] = max([as_float(r.get("max_energy_identity_relative_error"), 0.0) for r in rows], default=None)
            out.append(row)
    return out


def classify_delta_rescue(summary_rows, name):
    row = next((r for r in summary_rows if r["restoration"] == name), None)
    if not row:
        return "NOT_RUN"
    med = as_float(row.get("median_DeltaRescue"))
    wins = int(float(row.get("wins_DeltaRescue_gt_0") or 0))
    ci = json.loads(row.get("bootstrap95_DeltaRescue") or "null")
    if med is not None and med > 0 and wins >= 13 and ci and ci[0] > 0:
        return "SUPPORTED"
    if med is not None and med > 0 and wins >= 10:
        return "PARTIAL"
    return "NOT_SUPPORTED"


def decide_primary(summary_rows):
    by = {r["restoration"]: r for r in summary_rows}
    support = {name: classify_delta_rescue(summary_rows, name) for name in by}
    singles = ["restore-q", "restore-k", "restore-v", "restore-beta", "restore-decay"]
    for name, cls in support.items():
        if name in singles and cls == "SUPPORTED":
            return {
                "restore-q": ("FUTURE_QUERY_DRIFT", "KDA_ROTATION_Q_DRIFT_CAUSAL_CLOSURE_V1"),
                "restore-k": ("TRANSITION_KEY_DRIFT", "KDA_ROTATION_K_DRIFT_CAUSAL_CLOSURE_V1"),
                "restore-v": ("VALUE_UPDATE_DRIFT", "KDA_ROTATION_V_DRIFT_CAUSAL_CLOSURE_V1"),
                "restore-beta": ("BETA_DRIFT", "KDA_ROTATION_BETA_DRIFT_CAUSAL_CLOSURE_V1"),
                "restore-decay": ("DECAY_DRIFT", "KDA_ROTATION_DECAY_DRIFT_CAUSAL_CLOSURE_V1"),
            }[name]
    group_map = {
        "restore-transition": ("TRANSITION_OPERATOR_DRIFT_INTERACTION", "KDA_ROTATION_TRANSITION_OPERATOR_DRIFT_CLOSURE_V1"),
        "restore-update": ("FRESH_UPDATE_DRIFT_INTERACTION", "KDA_ROTATION_FRESH_UPDATE_DRIFT_CLOSURE_V1"),
        "restore-all-transition": ("MULTI_DRIVER_CLOSED_LOOP_INTERACTION", "KDA_ROTATION_MULTI_DRIVER_INTERACTION_CLOSURE_V1"),
        "restore-all-kda": ("MULTI_DRIVER_CLOSED_LOOP_INTERACTION", "KDA_ROTATION_MULTI_DRIVER_INTERACTION_CLOSURE_V1"),
    }
    for name in ["restore-transition", "restore-update", "restore-all-transition", "restore-all-kda"]:
        if support.get(name) == "SUPPORTED":
            return group_map[name]
    all_row = by.get("restore-all-kda")
    if all_row is not None and as_float(all_row.get("median_DeltaRescue"), 0.0) <= 0:
        return "KDA_DRIVER_DRIFT_NOT_PRIMARY", "KDA_ROTATION_GLOBAL_CLOSED_LOOP_FEEDBACK_V1"
    return "UNRESOLVED", "KDA_ROTATION_CLOSED_LOOP_DRIVER_DRIFT_REASSESSMENT_V2"


def reset_outputs(outdir):
    for rel in [
        "baseline_trajectories/baseline_trajectories.jsonl",
        "drift_panel/driver_drift.jsonl",
        "drift_panel/effective_operator_drift.jsonl",
        "pgr_decomposition/pgr_decomposition.jsonl",
        "single_driver_restoration/single_driver_restoration.jsonl",
        "grouped_restoration/grouped_restoration.jsonl",
        "horizon_rescue/horizon_rescue.jsonl",
    ]:
        p = outdir / rel
        if p.exists():
            p.unlink()


def unit_results_from_horizon_kl(horizon_kl_rows):
    by_unit_branch = defaultdict(list)
    for row in horizon_kl_rows:
        by_unit_branch[(row["unit_id"], row["branch"])].append(as_float(row.get("FutureKL")))
    by_unit = defaultdict(dict)
    for (unit_id, branch), vals in by_unit_branch.items():
        by_unit[unit_id][branch] = mean(vals)
    return [{"unit_id": unit_id, "auc": auc} for unit_id, auc in sorted(by_unit.items())]


def merge_jsonl_files(shard_dirs, rel, dest):
    rows = []
    if dest.exists():
        dest.unlink()
    for shard in shard_dirs:
        for row in iter_jsonl(shard / rel) or []:
            rows.append(row)
            append_jsonl(dest, row)
    return rows


def merge_shards(args):
    outdir = Path(args.output_dir)
    shard_dirs = [Path(p) for p in args.merge_shard_dirs]
    for name in ["driver_audit", "baseline_trajectories", "drift_panel", "pgr_decomposition", "single_driver_restoration", "grouped_restoration", "horizon_rescue", "figures"]:
        (outdir / name).mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        reset_outputs(outdir)
    shard_summaries = [load_json(shard / "summary.json", {}) for shard in shard_dirs]
    first_audit = load_json(shard_dirs[0] / "driver_audit" / "kda_closed_loop_drivers.json", {})
    save_json(outdir / "driver_audit" / "kda_closed_loop_drivers.json", first_audit)
    first_config = load_json(shard_dirs[0] / "config.json", {})
    first_config.update({"merged_from_shards": [str(p) for p in shard_dirs]})
    save_json(outdir / "config.json", first_config)
    env = {
        "merge_python_executable": sys.executable,
        "merge_python_version": sys.version,
        "shard_environments": [s.get("environment", {}) for s in shard_summaries],
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
    }
    save_json(outdir / "provenance.json", env)
    baseline_rows = merge_jsonl_files(shard_dirs, "baseline_trajectories/baseline_trajectories.jsonl", outdir / "baseline_trajectories" / "baseline_trajectories.jsonl")
    drift_rows = merge_jsonl_files(shard_dirs, "drift_panel/driver_drift.jsonl", outdir / "drift_panel" / "driver_drift.jsonl")
    op_rows = merge_jsonl_files(shard_dirs, "drift_panel/effective_operator_drift.jsonl", outdir / "drift_panel" / "effective_operator_drift.jsonl")
    pgr_rows = merge_jsonl_files(shard_dirs, "pgr_decomposition/pgr_decomposition.jsonl", outdir / "pgr_decomposition" / "pgr_decomposition.jsonl")
    horizon_kl_rows = merge_jsonl_files(shard_dirs, "horizon_rescue/horizon_rescue.jsonl", outdir / "horizon_rescue" / "horizon_rescue.jsonl")
    merge_jsonl_files(shard_dirs, "driver_audit/restoration_provenance.jsonl", outdir / "driver_audit" / "restoration_provenance.jsonl")
    unit_results = unit_results_from_horizon_kl(horizon_kl_rows)
    drift_unit = aggregate_drift(drift_rows)
    op_unit = aggregate_operator(op_rows)
    drift_summary = drift_horizon_summary(drift_unit, op_unit)
    write_rows(outdir / "drift_panel" / "driver_drift_unit_aggregate.csv", drift_unit)
    write_rows(outdir / "drift_panel" / "driver_drift_horizon_summary.csv", drift_summary)
    write_rows(outdir / "drift_panel" / "effective_operator_drift_unit_aggregate.csv", op_unit)
    pgr_unit = aggregate_pgr(pgr_rows)
    pgr_summary = summarize_pgr(pgr_unit)
    write_rows(outdir / "pgr_decomposition" / "pgr_unit_aggregate.csv", pgr_unit)
    write_rows(outdir / "pgr_decomposition" / "pgr_horizon_summary.csv", pgr_summary)
    rest_rows = restoration_unit_rows(unit_results)
    single_rows = [r for r in rest_rows if r["restoration"] in {x[0] for x in SINGLE_RESTORATIONS}]
    group_rows = [r for r in rest_rows if r["restoration"] not in {x[0] for x in SINGLE_RESTORATIONS}]
    single_summary = summarize_restoration(single_rows)
    group_summary = summarize_restoration(group_rows)
    write_rows(outdir / "single_driver_restoration" / "single_driver_restoration_unit_aggregate.csv", single_rows)
    write_rows(outdir / "single_driver_restoration" / "single_driver_restoration_summary.csv", single_summary)
    write_rows(outdir / "grouped_restoration" / "grouped_restoration_unit_aggregate.csv", group_rows)
    write_rows(outdir / "grouped_restoration" / "grouped_restoration_summary.csv", group_summary)
    for row in single_rows:
        append_jsonl(outdir / "single_driver_restoration" / "single_driver_restoration.jsonl", row)
    for row in group_rows:
        append_jsonl(outdir / "grouped_restoration" / "grouped_restoration.jsonl", row)
    hr = horizon_rescue_rows(horizon_kl_rows)
    hr_summary = summarize_horizon_rescue(hr)
    write_rows(outdir / "horizon_rescue" / "horizon_rescue_summary.csv", hr_summary)
    for row in hr:
        append_jsonl(outdir / "horizon_rescue" / "horizon_rescue_detail.jsonl", row)
    max_pgr_rel = max([as_float(r.get("max_identity_relative_l2_error"), 0.0) for r in pgr_unit], default=0.0)
    max_pgr_energy = max([as_float(r.get("max_energy_identity_relative_error"), 0.0) for r in pgr_unit], default=0.0)
    pgr_gate = "PASS" if pgr_unit and max_pgr_rel <= PGR_IDENTITY_TOL and max_pgr_energy <= 1e-4 else "FAIL"
    primary, next_exp = decide_primary(single_summary + group_summary)
    closure = "SUPPORTED" if primary not in {"UNRESOLVED", "KDA_DRIVER_DRIFT_NOT_PRIMARY"} else ("NOT_SUPPORTED" if primary == "KDA_DRIVER_DRIFT_NOT_PRIMARY" else "PARTIAL")
    mech_status = "SUFFICIENT_FOR_CURRENT_PAPER" if closure == "SUPPORTED" else ("NOT_CLOSED" if closure == "NOT_SUPPORTED" else "PARTIAL")
    failures = [f for s in shard_summaries for f in s.get("failures", [])]
    kda_layers = first_audit.get("kda_layers", [])
    summary = {
        "task": TASK,
        "KDA_CLOSED_LOOP_DRIVER_AUDIT": first_audit.get("KDA_CLOSED_LOOP_DRIVER_AUDIT"),
        "KDA_DRIVER_COORDINATE_PROVENANCE": first_audit.get("KDA_DRIVER_COORDINATE_PROVENANCE"),
        "KDA_Q_DRIFT_CLASSIFICATION": classify_rotated_vs_native(drift_summary, "median_q_drift"),
        "KDA_K_DRIFT_CLASSIFICATION": classify_rotated_vs_native(drift_summary, "median_k_drift"),
        "KDA_V_DRIFT_CLASSIFICATION": classify_rotated_vs_native(drift_summary, "median_v_drift"),
        "KDA_BETA_DRIFT_CLASSIFICATION": classify_rotated_vs_native(drift_summary, "median_beta_drift"),
        "KDA_DECAY_DRIFT_CLASSIFICATION": classify_rotated_vs_native(drift_summary, "median_decay_drift"),
        "KDA_A_OPERATOR_DRIFT_CLASSIFICATION": classify_rotated_vs_native(drift_summary, "median_A_rel_drift"),
        "KDA_B_UPDATE_DRIFT_CLASSIFICATION": classify_rotated_vs_native(drift_summary, "median_B_rel_drift"),
        "KDA_PGR_DECOMPOSITION": pgr_gate,
        "KDA_G_COMPONENT_ROTATION_EFFECT": classify_rotated_vs_native(pgr_summary, "median_G_norm"),
        "KDA_RESTORE_Q_DIFFERENTIAL_RESCUE": classify_delta_rescue(single_summary, "restore-q"),
        "KDA_RESTORE_K_DIFFERENTIAL_RESCUE": classify_delta_rescue(single_summary, "restore-k"),
        "KDA_RESTORE_V_DIFFERENTIAL_RESCUE": classify_delta_rescue(single_summary, "restore-v"),
        "KDA_RESTORE_BETA_DIFFERENTIAL_RESCUE": classify_delta_rescue(single_summary, "restore-beta"),
        "KDA_RESTORE_DECAY_DIFFERENTIAL_RESCUE": classify_delta_rescue(single_summary, "restore-decay"),
        "KDA_TRANSITION_GROUP_DIFFERENTIAL_RESCUE": classify_delta_rescue(group_summary, "restore-transition"),
        "KDA_UPDATE_GROUP_DIFFERENTIAL_RESCUE": classify_delta_rescue(group_summary, "restore-update"),
        "KDA_ALL_TRANSITION_DIFFERENTIAL_RESCUE": classify_delta_rescue(group_summary, "restore-all-transition"),
        "KDA_ALL_DRIVER_DIFFERENTIAL_RESCUE": classify_delta_rescue(group_summary, "restore-all-kda"),
        "KDA_PRIMARY_CLOSED_LOOP_CAUSE": primary,
        "KDA_CLOSED_LOOP_DRIVER_DRIFT_CAUSAL_CLOSURE": closure,
        "KDA_MECHANISM_CLOSURE_STATUS": mech_status,
        "FINAL_CLASSIFICATION": f"KDA_CLOSED_LOOP_DRIVER_DRIFT_{primary}",
        "NEXT_EXPERIMENT": next_exp,
        "n_units_completed": len({u["unit_id"] for u in unit_results}),
        "expected_units": EXPECTED_UNITS,
        "kda_layers": list(map(int, kda_layers)),
        "max_pgr_identity_relative_l2_error": max_pgr_rel,
        "max_pgr_energy_identity_relative_error": max_pgr_energy,
        "failures": failures,
        "environment": env,
        "merged_from_shards": [str(p) for p in shard_dirs],
    }
    save_json(outdir / "summary.json", summary)
    make_plots(outdir, drift_summary, pgr_summary, single_summary, group_summary, hr_summary, rest_rows)
    write_report(outdir, summary, first_audit, drift_summary, pgr_summary, single_summary, group_summary, hr_summary)
    save_json(outdir / "manifest.json", {"task": TASK, "created_at": now(), "artifact_paths": sorted(str(p) for p in outdir.rglob("*") if p.is_file())})
    return summary


def run_experiment(args):
    outdir = Path(args.output_dir)
    if args.overwrite:
        reset_outputs(outdir)
    for name in ["driver_audit", "baseline_trajectories", "drift_panel", "pgr_decomposition", "single_driver_restoration", "grouped_restoration", "horizon_rescue", "figures"]:
        (outdir / name).mkdir(parents=True, exist_ok=True)
    env = environment_info()
    save_json(outdir / "provenance.json", env)
    units, unit_audit = load_units(args.scope, args.max_units, shard_index=args.unit_shard_index, shard_count=args.unit_shard_count)
    completed_before_resume = set()
    if args.resume_completed_units:
        completed_before_resume = completed_unit_ids_from_output(outdir)
        units = [unit for unit in units if str(unit["unit_id"]) not in completed_before_resume]
    config_obj = json.loads((MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = P().kda_layers_from_config(config_obj)
    if args.target_layer_limit:
        kda_layers = list(kda_layers)[: int(args.target_layer_limit)]
    audit = build_driver_audit(kda_layers)
    save_json(outdir / "driver_audit" / "kda_closed_loop_drivers.json", audit)
    if audit["KDA_CLOSED_LOOP_DRIVER_AUDIT"] != "PASS":
        raise RuntimeError("driver audit failed")
    save_json(outdir / "config.json", {
        "task": TASK,
        "scope": args.scope,
        "horizons": HORIZONS,
        "primary_functional_endpoint": "FutureKL_AUC",
        "rotation": "KDA Value-side RHT seed 0",
        "formal_unit": "canonical recurrent unit",
        "expected_units": EXPECTED_UNITS,
        "unit_shard_index": args.unit_shard_index,
        "unit_shard_count": args.unit_shard_count,
        "resume_completed_units": bool(args.resume_completed_units),
        "completed_units_skipped": sorted(completed_before_resume),
        "canonical_manifest": unit_audit,
        "branch_definitions": [s["branch"] for s in branch_specs()],
    })
    torch_mod, model, tokenizer = P().load_model_and_tokenizer()
    rows_by_pid = P().load_dataset()
    teacher_tokens = P().load_fp_teacher_tokens()
    rotation = exact().make_experiment_rotation("kda", 128, "rht", seed=RHT_SEED, dtype=torch.float32)
    probe = ClosedLoopKdaProbe(rotation.to(next(model.parameters()).device))
    installed = probe.install(model)
    failures = []
    unit_results = []
    baseline_rows = []
    drift_rows = []
    op_rows = []
    pgr_rows = []
    horizon_kl_rows = []
    restoration_logs = []
    try:
        for idx, unit in enumerate(units, 1):
            print(f"[{now()}] KDA closed-loop driver unit {idx}/{len(units)} {unit['unit_id']}", flush=True)
            try:
                probe.fp_records_by_h = {}
                probe.restoration_logs = []
                result = run_unit(
                    torch_mod,
                    model,
                    tokenizer,
                    probe,
                    unit,
                    kda_layers,
                    rows_by_pid,
                    teacher_tokens,
                    rotation.cpu(),
                    int(args.horizon),
                    target_layer_limit=args.target_layer_limit,
                )
                unit_results.append(result)
                baseline_rows.extend(result["baseline_rows"])
                drift_rows.extend(result["drift_rows"])
                op_rows.extend(result["operator_rows"])
                pgr_rows.extend(result["pgr_rows"])
                horizon_kl_rows.extend(result["horizon_kl"])
                restoration_logs.extend(result["restoration_logs"])
                for row in result["baseline_rows"]:
                    append_jsonl(outdir / "baseline_trajectories" / "baseline_trajectories.jsonl", row)
                for row in result["drift_rows"]:
                    append_jsonl(outdir / "drift_panel" / "driver_drift.jsonl", row)
                for row in result["operator_rows"]:
                    append_jsonl(outdir / "drift_panel" / "effective_operator_drift.jsonl", row)
                for row in result["pgr_rows"]:
                    append_jsonl(outdir / "pgr_decomposition" / "pgr_decomposition.jsonl", row)
                for row in result["horizon_kl"]:
                    append_jsonl(outdir / "horizon_rescue" / "horizon_rescue.jsonl", row)
                for row in result["restoration_logs"]:
                    append_jsonl(outdir / "driver_audit" / "restoration_provenance.jsonl", row)
            except Exception as exc:
                failure = {"unit": unit, "error": repr(exc), "traceback": traceback.format_exc(limit=30), "timestamp": now()}
                failures.append(failure)
                save_json(outdir / "failures.json", failures)
                print(f"[{now()}] FAILED unit={unit.get('unit_id')} {exc!r}", flush=True)
                if not args.keep_going:
                    raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        probe.close()
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    drift_unit = aggregate_drift(drift_rows)
    op_unit = aggregate_operator(op_rows)
    drift_summary = drift_horizon_summary(drift_unit, op_unit)
    write_rows(outdir / "drift_panel" / "driver_drift_unit_aggregate.csv", drift_unit)
    write_rows(outdir / "drift_panel" / "driver_drift_horizon_summary.csv", drift_summary)
    write_rows(outdir / "drift_panel" / "effective_operator_drift_unit_aggregate.csv", op_unit)
    pgr_unit = aggregate_pgr(pgr_rows)
    pgr_summary = summarize_pgr(pgr_unit)
    write_rows(outdir / "pgr_decomposition" / "pgr_unit_aggregate.csv", pgr_unit)
    write_rows(outdir / "pgr_decomposition" / "pgr_horizon_summary.csv", pgr_summary)
    rest_rows = restoration_unit_rows(unit_results)
    single_rows = [r for r in rest_rows if r["restoration"] in {x[0] for x in SINGLE_RESTORATIONS}]
    group_rows = [r for r in rest_rows if r["restoration"] not in {x[0] for x in SINGLE_RESTORATIONS}]
    single_summary = summarize_restoration(single_rows)
    group_summary = summarize_restoration(group_rows)
    write_rows(outdir / "single_driver_restoration" / "single_driver_restoration_unit_aggregate.csv", single_rows)
    write_rows(outdir / "single_driver_restoration" / "single_driver_restoration_summary.csv", single_summary)
    write_rows(outdir / "grouped_restoration" / "grouped_restoration_unit_aggregate.csv", group_rows)
    write_rows(outdir / "grouped_restoration" / "grouped_restoration_summary.csv", group_summary)
    for row in single_rows:
        append_jsonl(outdir / "single_driver_restoration" / "single_driver_restoration.jsonl", row)
    for row in group_rows:
        append_jsonl(outdir / "grouped_restoration" / "grouped_restoration.jsonl", row)
    hr = horizon_rescue_rows(horizon_kl_rows)
    hr_summary = summarize_horizon_rescue(hr)
    write_rows(outdir / "horizon_rescue" / "horizon_rescue_summary.csv", hr_summary)
    for row in hr:
        append_jsonl(outdir / "horizon_rescue" / "horizon_rescue_detail.jsonl", row)
    max_pgr_rel = max([as_float(r.get("max_identity_relative_l2_error"), 0.0) for r in pgr_unit], default=0.0)
    max_pgr_energy = max([as_float(r.get("max_energy_identity_relative_error"), 0.0) for r in pgr_unit], default=0.0)
    pgr_gate = "PASS" if pgr_unit and max_pgr_rel <= PGR_IDENTITY_TOL and max_pgr_energy <= 1e-4 else "FAIL"
    primary, next_exp = decide_primary(single_summary + group_summary)
    closure = "SUPPORTED" if primary not in {"UNRESOLVED", "KDA_DRIVER_DRIFT_NOT_PRIMARY"} else ("NOT_SUPPORTED" if primary == "KDA_DRIVER_DRIFT_NOT_PRIMARY" else "PARTIAL")
    mech_status = "SUFFICIENT_FOR_CURRENT_PAPER" if closure == "SUPPORTED" else ("NOT_CLOSED" if closure == "NOT_SUPPORTED" else "PARTIAL")
    summary = {
        "task": TASK,
        "KDA_CLOSED_LOOP_DRIVER_AUDIT": audit["KDA_CLOSED_LOOP_DRIVER_AUDIT"],
        "KDA_DRIVER_COORDINATE_PROVENANCE": audit["KDA_DRIVER_COORDINATE_PROVENANCE"],
        "KDA_Q_DRIFT_CLASSIFICATION": classify_rotated_vs_native(drift_summary, "median_q_drift"),
        "KDA_K_DRIFT_CLASSIFICATION": classify_rotated_vs_native(drift_summary, "median_k_drift"),
        "KDA_V_DRIFT_CLASSIFICATION": classify_rotated_vs_native(drift_summary, "median_v_drift"),
        "KDA_BETA_DRIFT_CLASSIFICATION": classify_rotated_vs_native(drift_summary, "median_beta_drift"),
        "KDA_DECAY_DRIFT_CLASSIFICATION": classify_rotated_vs_native(drift_summary, "median_decay_drift"),
        "KDA_A_OPERATOR_DRIFT_CLASSIFICATION": classify_rotated_vs_native(drift_summary, "median_A_rel_drift"),
        "KDA_B_UPDATE_DRIFT_CLASSIFICATION": classify_rotated_vs_native(drift_summary, "median_B_rel_drift"),
        "KDA_PGR_DECOMPOSITION": pgr_gate,
        "KDA_G_COMPONENT_ROTATION_EFFECT": classify_rotated_vs_native(pgr_summary, "median_G_norm"),
        "KDA_RESTORE_Q_DIFFERENTIAL_RESCUE": classify_delta_rescue(single_summary, "restore-q"),
        "KDA_RESTORE_K_DIFFERENTIAL_RESCUE": classify_delta_rescue(single_summary, "restore-k"),
        "KDA_RESTORE_V_DIFFERENTIAL_RESCUE": classify_delta_rescue(single_summary, "restore-v"),
        "KDA_RESTORE_BETA_DIFFERENTIAL_RESCUE": classify_delta_rescue(single_summary, "restore-beta"),
        "KDA_RESTORE_DECAY_DIFFERENTIAL_RESCUE": classify_delta_rescue(single_summary, "restore-decay"),
        "KDA_TRANSITION_GROUP_DIFFERENTIAL_RESCUE": classify_delta_rescue(group_summary, "restore-transition"),
        "KDA_UPDATE_GROUP_DIFFERENTIAL_RESCUE": classify_delta_rescue(group_summary, "restore-update"),
        "KDA_ALL_TRANSITION_DIFFERENTIAL_RESCUE": classify_delta_rescue(group_summary, "restore-all-transition"),
        "KDA_ALL_DRIVER_DIFFERENTIAL_RESCUE": classify_delta_rescue(group_summary, "restore-all-kda"),
        "KDA_PRIMARY_CLOSED_LOOP_CAUSE": primary,
        "KDA_CLOSED_LOOP_DRIVER_DRIFT_CAUSAL_CLOSURE": closure,
        "KDA_MECHANISM_CLOSURE_STATUS": mech_status,
        "FINAL_CLASSIFICATION": f"KDA_CLOSED_LOOP_DRIVER_DRIFT_{primary}",
        "NEXT_EXPERIMENT": next_exp,
        "n_units_completed": len({u["unit_id"] for u in unit_results}),
        "expected_units": len(units),
        "kda_layers": list(map(int, kda_layers)),
        "max_pgr_identity_relative_l2_error": max_pgr_rel,
        "max_pgr_energy_identity_relative_error": max_pgr_energy,
        "failures": failures,
        "environment": env,
    }
    save_json(outdir / "summary.json", summary)
    make_plots(outdir, drift_summary, pgr_summary, single_summary, group_summary, hr_summary, rest_rows)
    write_report(outdir, summary, audit, drift_summary, pgr_summary, single_summary, group_summary, hr_summary)
    save_json(outdir / "manifest.json", {"task": TASK, "created_at": now(), "artifact_paths": sorted(str(p) for p in outdir.rglob("*") if p.is_file())})
    return summary


def make_plots(outdir, drift_summary, pgr_summary, single_summary, group_summary, hr_summary, rest_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        save_json(outdir / "figures" / "manifest.json", {"status": "SKIPPED", "reason": repr(exc), "plots": []})
        return
    made = []
    for name in DRIVERS:
        rows_n = [r for r in drift_summary if r["branch"] == "INT8_NATIVE"]
        rows_r = [r for r in drift_summary if r["branch"] == "INT8_ROTATED"]
        plt.figure(figsize=(5, 3))
        plt.plot([r["horizon"] for r in rows_n], [as_float(r[f"median_{name}_drift"], 0.0) for r in rows_n], marker="o", label="Native")
        plt.plot([r["horizon"] for r in rows_r], [as_float(r[f"median_{name}_drift"], 0.0) for r in rows_r], marker="o", label="Rotated")
        plt.xscale("log", base=2)
        plt.xlabel("horizon")
        plt.ylabel("rel drift")
        plt.title(f"{name} drift")
        plt.legend()
        plt.tight_layout()
        p = outdir / "figures" / f"{name}_drift.png"
        plt.savefig(p, dpi=150)
        plt.close()
        made.append(str(p))
    for key, fname in [("median_A_rel_drift", "effective_A_drift.png"), ("median_B_rel_drift", "effective_B_drift.png")]:
        rows_n = [r for r in drift_summary if r["branch"] == "INT8_NATIVE"]
        rows_r = [r for r in drift_summary if r["branch"] == "INT8_ROTATED"]
        plt.figure(figsize=(5, 3))
        plt.plot([r["horizon"] for r in rows_n], [as_float(r[key], 0.0) for r in rows_n], marker="o", label="Native")
        plt.plot([r["horizon"] for r in rows_r], [as_float(r[key], 0.0) for r in rows_r], marker="o", label="Rotated")
        plt.xscale("log", base=2)
        plt.legend()
        plt.tight_layout()
        p = outdir / "figures" / fname
        plt.savefig(p, dpi=150)
        plt.close()
        made.append(str(p))
    for rows, fname, title in [(single_summary, "single_driver_delta_rescue.png", "single-driver DeltaRescue"), (group_summary, "grouped_delta_rescue.png", "grouped DeltaRescue")]:
        if rows:
            plt.figure(figsize=(7, 3))
            plt.bar([r["restoration"] for r in rows], [as_float(r["median_DeltaRescue"], 0.0) for r in rows])
            plt.axhline(0, color="black", linewidth=1)
            plt.xticks(rotation=25, ha="right", fontsize=7)
            plt.title(title)
            plt.tight_layout()
            p = outdir / "figures" / fname
            plt.savefig(p, dpi=150)
            plt.close()
            made.append(str(p))
    if rest_rows:
        plt.figure(figsize=(7, 3))
        xs = range(len(rest_rows))
        plt.scatter(list(xs), [as_float(r["DeltaRescue"], 0.0) for r in rest_rows], s=10)
        plt.axhline(0, color="black", linewidth=1)
        plt.title("DeltaRescue by canonical unit/restoration")
        plt.tight_layout()
        p = outdir / "figures" / "delta_rescue_by_unit.png"
        plt.savefig(p, dpi=150)
        plt.close()
        made.append(str(p))
    if hr_summary:
        for name in sorted({r["restoration"] for r in hr_summary}):
            rows = [r for r in hr_summary if r["restoration"] == name]
            plt.figure(figsize=(5, 3))
            plt.plot([int(r["horizon"]) for r in rows], [as_float(r["median_DeltaRescue"], 0.0) for r in rows], marker="o")
            plt.axhline(0, color="black", linewidth=1)
            plt.xscale("log", base=2)
            plt.title(name)
            plt.tight_layout()
            p = outdir / "figures" / f"horizon_rescue_{name}.png"
            plt.savefig(p, dpi=150)
            plt.close()
            made.append(str(p))
    save_json(outdir / "figures" / "manifest.json", {"status": "OK", "plots": made})


def markdown_table(rows, fields):
    out = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        out.append("|" + "|".join(str(row.get(f, "")) for f in fields) + "|")
    return "\n".join(out)


def write_report(outdir, summary, audit, drift_summary, pgr_summary, single_summary, group_summary, hr_summary):
    lines = [
        f"# {TASK}",
        "",
        "## Classifications",
        "",
        "```text",
    ]
    keys = [
        "KDA_CLOSED_LOOP_DRIVER_AUDIT", "KDA_DRIVER_COORDINATE_PROVENANCE",
        "KDA_Q_DRIFT_CLASSIFICATION", "KDA_K_DRIFT_CLASSIFICATION", "KDA_V_DRIFT_CLASSIFICATION",
        "KDA_BETA_DRIFT_CLASSIFICATION", "KDA_DECAY_DRIFT_CLASSIFICATION",
        "KDA_A_OPERATOR_DRIFT_CLASSIFICATION", "KDA_B_UPDATE_DRIFT_CLASSIFICATION",
        "KDA_PGR_DECOMPOSITION", "KDA_G_COMPONENT_ROTATION_EFFECT",
        "KDA_RESTORE_Q_DIFFERENTIAL_RESCUE", "KDA_RESTORE_K_DIFFERENTIAL_RESCUE",
        "KDA_RESTORE_V_DIFFERENTIAL_RESCUE", "KDA_RESTORE_BETA_DIFFERENTIAL_RESCUE",
        "KDA_RESTORE_DECAY_DIFFERENTIAL_RESCUE", "KDA_TRANSITION_GROUP_DIFFERENTIAL_RESCUE",
        "KDA_UPDATE_GROUP_DIFFERENTIAL_RESCUE", "KDA_ALL_TRANSITION_DIFFERENTIAL_RESCUE",
        "KDA_ALL_DRIVER_DIFFERENTIAL_RESCUE", "KDA_PRIMARY_CLOSED_LOOP_CAUSE",
        "KDA_CLOSED_LOOP_DRIVER_DRIFT_CAUSAL_CLOSURE", "KDA_MECHANISM_CLOSURE_STATUS",
        "FINAL_CLASSIFICATION", "NEXT_EXPERIMENT",
    ]
    for key in keys:
        lines.append(f"{key} = {summary.get(key)}")
    lines.extend(["```", "", f"Canonical units completed: {summary.get('n_units_completed')} / {summary.get('expected_units')}."])
    lines.extend(["", "## Driver Audit", "", " -> ".join(d["semantic_name"] for d in audit["drivers"])])
    lines.extend(["", "## Drift Summary", "", markdown_table(drift_summary, ["horizon", "branch", "n_units", "median_q_drift", "median_k_drift", "median_v_drift", "median_beta_drift", "median_decay_drift", "median_A_rel_drift", "median_B_rel_drift"])])
    lines.extend(["", "## Single Restoration", "", markdown_table(single_summary, ["restoration", "n_units", "median_Rescue_N", "median_Rescue_R", "median_DeltaRescue", "wins_DeltaRescue_gt_0", "bootstrap95_DeltaRescue"])])
    lines.extend(["", "## Grouped Restoration", "", markdown_table(group_summary, ["restoration", "n_units", "median_Rescue_N", "median_Rescue_R", "median_DeltaRescue", "wins_DeltaRescue_gt_0", "bootstrap95_DeltaRescue"])])
    lines.extend(["", "## PGR", "", markdown_table(pgr_summary, ["horizon", "branch", "n_units", "median_P_norm", "median_G_norm", "median_R_norm", "median_E_norm", "max_identity_relative_l2_error"])])
    lines.extend(["", "## Environment", "", "```json", json.dumps(summary.get("environment", {}), indent=2, sort_keys=True), "```"])
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scope", choices=["smoke", "pilot", "formal"], default="formal")
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--target-layer-limit", type=int, default=None)
    p.add_argument("--horizon", type=int, default=PRIMARY_HORIZON)
    p.add_argument("--output-dir", default=str(RESULT_DIR))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--unit-shard-index", type=int, default=None)
    p.add_argument("--unit-shard-count", type=int, default=None)
    p.add_argument("--resume-completed-units", action="store_true")
    p.add_argument("--merge-shard-dirs", nargs="*", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.merge_shard_dirs:
        summary = merge_shards(args)
    else:
        summary = run_experiment(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.scope == "formal" and args.max_units is None and args.target_layer_limit is None:
        if summary.get("n_units_completed") != summary.get("expected_units"):
            return 2
        if summary.get("KDA_CLOSED_LOOP_DRIVER_AUDIT") != "PASS" or summary.get("KDA_DRIVER_COORDINATE_PROVENANCE") != "PASS":
            return 2
        if summary.get("KDA_PGR_DECOMPOSITION") != "PASS":
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
