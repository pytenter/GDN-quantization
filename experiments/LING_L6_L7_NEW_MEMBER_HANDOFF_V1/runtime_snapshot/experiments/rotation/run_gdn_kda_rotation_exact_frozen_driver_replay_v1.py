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


TASK = "GDN_KDA_ROTATION_EXACT_FROZEN_DRIVER_REPLAY_V1"
SLUG = "gdn_kda_rotation_exact_frozen_driver_replay_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
V1_RESULT_DIR = REPO / "results" / "gdn_kda_axis_matched_state_rotation_v1"
GDN_CANONICAL_MANIFEST = REPO / "results" / "propagation" / "gdn_int8_vspace_source_to_temporal_accumulation_formal_v1_stage0.json"
KDA_CANONICAL_UNITS = REPO / "runs" / "ling_kda_persistent_error_decomposition_causal_v1" / "canonical_units.json"
GDN_PYTHON = "/data01/user2/.conda/envs/sd310/bin/python"
KDA_PYTHON = "/data01/user2/.conda/envs/ling-kda/bin/python"
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
HORIZONS = [1, 2, 4, 8, 16, 32, 64]
STAGE5_HORIZONS = [8, 16, 32, 64]
EPS = 1e-12
RHT_SEED = 0
CANONICAL_N = 18


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd, cwd=REPO):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


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


def bootstrap_ci(xs, n=4000, seed=20260913):
    vals = [float(x) for x in xs if finite(x)]
    if not vals:
        return None
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals))
    out.sort()
    return [out[int(0.025 * (n - 1))], out[int(0.975 * (n - 1))]]


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_V1 = None


def v1():
    global _V1
    if _V1 is None:
        _V1 = import_file(REPO / "experiments" / "rotation" / "run_gdn_kda_axis_matched_state_rotation_v1.py", "axis_rotation_v1_for_exact_replay")
    return _V1


def make_rotation(n, kind="identity", seed=0, dtype=torch.float64, device=None, samples=None):
    return v1().make_rotation(n, kind=kind, seed=seed, dtype=dtype, device=device, samples=samples)


def make_experiment_rotation(arch, n, kind="rht", seed=RHT_SEED, dtype=torch.float32, device=None):
    rot = make_rotation(n, kind=kind, seed=seed, dtype=dtype, device=device)
    if str(arch).lower() == "kda" and str(kind).lower() == "rht":
        # Matches the validated KDA value-side RHT convention from the closure runner.
        rot = rot.t().contiguous()
    return rot


def fake_quant_state(state, config):
    return v1().fake_quant_state(state, config)


def rotate_state_key_axis(state, rotation):
    return v1().rotate_state_key_axis(state, rotation)


def inverse_rotate_state_key_axis(state, rotation):
    return v1().inverse_rotate_state_key_axis(state, rotation)


def rotate_state_value_axis(state, rotation):
    return v1().rotate_state_value_axis(state, rotation)


def inverse_rotate_state_value_axis(state, rotation):
    return v1().inverse_rotate_state_value_axis(state, rotation)


def rotate_value_axis_tensor(x, rotation):
    return v1().rotate_value_axis_tensor(x, rotation)


def inverse_rotate_value_axis_tensor(x, rotation):
    return v1().inverse_rotate_value_axis_tensor(x, rotation)


def prepare_gdn_rotated_qk(query, key, rotation, use_qk_l2norm_in_kernel=False):
    return v1().prepare_gdn_rotated_qk(query, key, rotation, use_qk_l2norm_in_kernel)


def relerr(a, b):
    return v1().relerr(a, b)


def tensor_norm(x):
    return float(torch.linalg.vector_norm(x.detach().double()).item())


def tensor_dot(a, b):
    return float((a.detach().double() * b.detach().double()).sum().item())


def flat_cosine(a, b):
    an = tensor_norm(a)
    bn = tensor_norm(b)
    if an <= EPS or bn <= EPS:
        return None
    return tensor_dot(a, b) / (an * bn + EPS)


def stack_norm(stack):
    return math.sqrt(sum(tensor_norm(v) ** 2 for v in stack.values()))


def stack_dot(a, b):
    return sum(tensor_dot(a[k], b[k]) for k in sorted(set(a) & set(b)))


def stack_cosine(a, b):
    an = stack_norm(a)
    bn = stack_norm(b)
    if an <= EPS or bn <= EPS:
        return None
    return stack_dot(a, b) / (an * bn + EPS)


def readout_perturbation(error, query):
    return torch.einsum("bhkv,bhk->bhv", error.detach().float(), query.detach().float())


def readout_error_norm(error, query):
    return tensor_norm(readout_perturbation(error, query))


def qmax_abs_diff(a, b):
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def tensor_hash(x):
    y = x.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(y.shape)).encode("ascii"))
    h.update(str(y.dtype).encode("ascii"))
    if y.dtype in {torch.bfloat16, torch.float16}:
        h.update(y.to(torch.float32).numpy().tobytes())
    else:
        h.update(y.numpy().tobytes())
    return h.hexdigest()


def driver_tape_hash(tape):
    h = hashlib.sha256()
    for key in ["architecture", "unit_id"]:
        h.update(str(tape.get(key)).encode("utf-8"))
    for step in tape["steps"]:
        h.update(str(step["t"]).encode("ascii"))
        for layer in sorted(step["records"]):
            rec = step["records"][layer]
            h.update(str(layer).encode("ascii"))
            for name in sorted(k for k, v in rec.items() if torch.is_tensor(v)):
                h.update(name.encode("ascii"))
                h.update(tensor_hash(rec[name]).encode("ascii"))
    return h.hexdigest()


def _gdn_step_native(state, rec):
    return v1()._gdn_step(state, rec["q"], rec["k"], rec["v"], rec["alpha"], rec["beta"])


def _kda_step_native(state, rec):
    return v1()._kda_step(state, rec["q"], rec["k"], rec["v"], rec["decay"], rec["beta"])


def synthetic_driver_tape(architecture, steps=4, k_dim=8, v_dim=8, heads=2, seed=20260913):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    arch = str(architecture).lower()
    state = torch.randn(1, heads, k_dim, v_dim, generator=gen, dtype=torch.float64) * 0.05
    tape = {"architecture": arch, "unit_id": "synthetic", "state0": {0: state.clone()}, "steps": []}
    for t in range(1, int(steps) + 1):
        if arch == "gdn":
            rec = {
                "q": torch.randn(1, heads, k_dim, generator=gen, dtype=torch.float64),
                "k": torch.randn(1, heads, k_dim, generator=gen, dtype=torch.float64) / math.sqrt(k_dim),
                "v": torch.randn(1, heads, v_dim, generator=gen, dtype=torch.float64),
                "alpha": torch.rand(1, heads, generator=gen, dtype=torch.float64) * 0.2 + 0.8,
                "beta": torch.rand(1, heads, generator=gen, dtype=torch.float64) * 0.5,
            }
            state, out = _gdn_step_native(state, rec)
        elif arch == "kda":
            rec = {
                "q": torch.randn(1, heads, k_dim, generator=gen, dtype=torch.float64),
                "k": torch.randn(1, heads, k_dim, generator=gen, dtype=torch.float64) / math.sqrt(k_dim),
                "v": torch.randn(1, heads, v_dim, generator=gen, dtype=torch.float64),
                "decay": torch.rand(1, heads, k_dim, generator=gen, dtype=torch.float64) * 0.2 + 0.8,
                "beta": torch.rand(1, heads, k_dim, generator=gen, dtype=torch.float64) * 0.5,
            }
            state, out = _kda_step_native(state, rec)
        else:
            raise ValueError(architecture)
        tape["steps"].append({"t": t, "records": {0: rec}, "fp_states": {0: state.clone()}, "readout_queries": {0: rec["q"].clone()}})
    tape["driver_tape_sha256"] = driver_tape_hash(tape)
    return tape


def replay_synthetic_transition(state, rec, arch, branch, rotation):
    if arch == "gdn":
        if branch == "rotated":
            qt = torch.einsum("ak,...k->...a", rotation.t().to(dtype=state.dtype), rec["q"])
            kt = torch.einsum("ak,...k->...a", rotation.t().to(dtype=state.dtype), rec["k"])
            rt = dict(rec, q=qt, k=kt)
            return _gdn_step_native(state, rt)[0]
        return _gdn_step_native(state, rec)[0]
    if arch == "kda":
        if branch == "rotated":
            vt = torch.einsum("vu,...v->...u", rotation.to(dtype=state.dtype), rec["v"])
            rt = dict(rec, v=vt)
            return _kda_step_native(state, rt)[0]
        return _kda_step_native(state, rec)[0]
    raise ValueError(arch)


def replay_synthetic_fp(tape, architecture):
    arch = str(architecture).lower()
    states = {layer: value.clone() for layer, value in tape["state0"].items()}
    rows = []
    for step in tape["steps"]:
        for layer, rec in step["records"].items():
            states[layer] = replay_synthetic_transition(states[layer], rec, arch, "native", torch.eye(states[layer].shape[-2], dtype=states[layer].dtype))
        err = stack_norm({layer: states[layer] - step["fp_states"][layer] for layer in states})
        ref = stack_norm(step["fp_states"])
        rows.append({"horizon": step["t"], "state_relative_error": err / (ref + EPS)})
    return {"rows": rows, "max_state_relative_error": max([r["state_relative_error"] for r in rows], default=0.0), "driver_tape_sha256": driver_tape_hash(tape)}


def transform_state_to_branch(state, arch, branch, rotation):
    if branch != "rotated":
        return state.detach().clone()
    if arch == "gdn":
        return rotate_state_key_axis(state, rotation)
    if arch == "kda":
        return rotate_state_value_axis(state, rotation)
    raise ValueError(arch)


def inverse_branch_state(state, arch, branch, rotation):
    if branch != "rotated":
        return state.detach().clone()
    if arch == "gdn":
        return inverse_rotate_state_key_axis(state, rotation)
    if arch == "kda":
        return inverse_rotate_state_value_axis(state, rotation)
    raise ValueError(arch)


def quantize_branch_state(state, arch, branch, quantizer_off=False):
    if quantizer_off:
        return state.detach().clone()
    q = "C128" if arch == "gdn" else "R128"
    return fake_quant_state(state, q)[0]


def pr_decomposition(fp_state, pre_quant_state, quantized_state):
    fp = fp_state.detach().float()
    pre = pre_quant_state.detach().float()
    q = quantized_state.detach().float()
    P = pre - fp
    R = q - pre
    E = q - fp
    residual = E - (P + R)
    pred = tensor_norm(P) ** 2 + tensor_norm(R) ** 2 + 2.0 * tensor_dot(P, R)
    actual = tensor_norm(E) ** 2
    return {
        "P": P,
        "R": R,
        "E": E,
        "P_norm": tensor_norm(P),
        "R_norm": tensor_norm(R),
        "E_norm": tensor_norm(E),
        "dot_P_R": tensor_dot(P, R),
        "cos_P_R": flat_cosine(P, R),
        "normalized_interference": 2.0 * tensor_dot(P, R) / (tensor_norm(P) ** 2 + tensor_norm(R) ** 2 + EPS),
        "identity_relative_error": tensor_norm(residual) / (tensor_norm(E) + EPS),
        "energy_identity_relative_error": abs(pred - actual) / (abs(actual) + EPS),
    }


def replay_synthetic_quantized(tape, architecture, branch, rotation, quantizer_off=False):
    arch = str(architecture).lower()
    states = {layer: transform_state_to_branch(value, arch, branch, rotation) for layer, value in tape["state0"].items()}
    rows = []
    for step in tape["steps"]:
        errs = {}
        for layer, rec in step["records"].items():
            pre = replay_synthetic_transition(states[layer], rec, arch, branch, rotation)
            qstate = quantize_branch_state(pre, arch, branch, quantizer_off=quantizer_off)
            states[layer] = qstate
            native_pre = inverse_branch_state(pre, arch, branch, rotation)
            native_q = inverse_branch_state(qstate, arch, branch, rotation)
            dec = pr_decomposition(step["fp_states"][layer], native_pre, native_q)
            errs[layer] = dec["E"]
        ref = stack_norm(step["fp_states"])
        enorm = stack_norm(errs)
        rows.append({
            "unit_id": tape["unit_id"],
            "horizon": step["t"],
            "branch": branch,
            "driver_tape_id": tape.get("driver_tape_sha256") or driver_tape_hash(tape),
            "state_error_norm": enorm,
            "state_relative_error": enorm / (ref + EPS),
        })
    return {"rows": rows, "max_state_relative_error": max([r["state_relative_error"] for r in rows], default=0.0)}


def assert_identical_driver_provenance(native_rows, rotated_rows):
    pairs = zip(native_rows, rotated_rows)
    bad = []
    for n, r in pairs:
        if n.get("unit_id") != r.get("unit_id") or int(n.get("horizon")) != int(r.get("horizon")) or n.get("driver_tape_id") != r.get("driver_tape_id"):
            bad.append({"native": n, "rotated": r})
    return {"gate": "PASS" if not bad else "FAIL", "n_mismatch": len(bad)}


def arch_dir(arch, stage):
    path = RESULT_DIR / arch / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def selected_horizons(max_horizon):
    return [h for h in HORIZONS if h <= int(max_horizon)]


def load_gdn_units():
    obj = load_json(GDN_CANONICAL_MANIFEST, {}) or {}
    units = []
    for unit in obj.get("manifest", []):
        row = dict(unit)
        row["unit_id"] = f"{row['problem_id']}|{int(row['t0'])}"
        units.append(row)
    return units, hashlib.sha256(GDN_CANONICAL_MANIFEST.read_bytes()).hexdigest() if GDN_CANONICAL_MANIFEST.exists() else None


def load_kda_units():
    obj = load_json(KDA_CANONICAL_UNITS, {}) or {}
    raw = obj.get("formal_units") or obj.get("canonical_units") or obj.get("units") or (obj if isinstance(obj, list) else [])
    units = []
    for i, unit in enumerate(raw):
        row = dict(unit)
        row.setdefault("unit_id", row.get("id") or f"{row.get('problem_id')}|{int(row.get('t0', 0))}")
        row["problem_id"] = str(row["problem_id"])
        row["t0"] = int(row["t0"])
        row["problem_index"] = int(row.get("problem_index", i))
        units.append(row)
    return units, hashlib.sha256(KDA_CANONICAL_UNITS.read_bytes()).hexdigest() if KDA_CANONICAL_UNITS.exists() else None


def select_units(units, scope, max_units=None):
    out = list(units)
    if scope == "smoke":
        out = out[:1]
    elif scope == "pilot":
        picks = [0, 4, 8, 9, 13, 17]
        out = [out[i] for i in picks if i < len(out)]
    if max_units is not None:
        out = out[: int(max_units)]
    return out


def gdn_setup():
    os.environ.setdefault("GDN_DATA_ROOT", "/data01/user2")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    orientation_dir = REPO / "experiments" / "orientation"
    if str(orientation_dir) not in sys.path:
        sys.path.insert(0, str(orientation_dir))
    import run_int8_orientation_state_change_mechanism as p1
    import run_end2end_bit_axis_screening as e2e
    e2e.ensure_imports()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod

    cfg = json.loads((Path(os.environ["GDN_DATA_ROOT"]) / "experiments" / "qwen35_gdn_quant" / "run_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, local_files_only=True)
    model.eval()
    return torch, model, tokenizer, p1, e2e, qmod, cfg


class GdnTransitionCapture:
    def __init__(self, qmod, model):
        self.qmod = qmod
        self.model = model
        self.records = []
        self.current_layer = None
        self.handles = []
        self.patched = []
        self.orig_rec = qmod.torch_recurrent_gated_delta_rule
        self.orig_chunk = getattr(qmod, "torch_chunk_gated_delta_rule", None)

    def _tensor_copy(self, x, dtype=None):
        if x is None:
            return None
        out = x.detach().cpu().clone()
        return out.to(dtype=dtype) if dtype is not None else out

    def _parse(self, args, kwargs):
        names = ["g", "beta", "initial_state", "output_final_state"]
        out = dict(kwargs)
        for name, value in zip(names, args):
            out.setdefault(name, value)
        return out

    def _record(self, layer, operator, query, key, value, args, kwargs, core, state):
        kw = self._parse(args, kwargs)
        if query.shape[1] != 1 or state is None:
            return
        self.records.append({
            "layer": int(layer),
            "operator": operator,
            "query": self._tensor_copy(query),
            "key": self._tensor_copy(key),
            "value": self._tensor_copy(value),
            "g": self._tensor_copy(kw.get("g")),
            "beta": self._tensor_copy(kw.get("beta")),
            "initial_state": self._tensor_copy(kw.get("initial_state"), dtype=torch.float32),
            "output_final_state": bool(kw.get("output_final_state", False)),
            "use_qk_l2norm_in_kernel": bool(kw.get("use_qk_l2norm_in_kernel", False)),
            "core_output": self._tensor_copy(core, dtype=torch.float32),
            "final_state": self._tensor_copy(state, dtype=torch.float32),
        })

    def install(self):
        layers = self.model.model.layers if hasattr(self.model.model, "layers") else self.model.model.language_model.layers
        for idx, layer in enumerate(layers):
            mod = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
            if mod is None:
                continue
            saved = {}
            if hasattr(mod, "recurrent_gated_delta_rule"):
                fn = mod.recurrent_gated_delta_rule
                saved["recurrent_gated_delta_rule"] = fn

                def make(fn_, layer_idx):
                    def wrapped(query, key, value, *args, **kwargs):
                        core, state = fn_(query, key, value, *args, **kwargs)
                        self._record(layer_idx, "recurrent", query, key, value, args, kwargs, core, state)
                        return core, state
                    return wrapped

                mod.recurrent_gated_delta_rule = make(fn, idx)
            if hasattr(mod, "chunk_gated_delta_rule"):
                fn = mod.chunk_gated_delta_rule
                saved["chunk_gated_delta_rule"] = fn

                def make(fn_, layer_idx):
                    def wrapped(query, key, value, *args, **kwargs):
                        core, state = fn_(query, key, value, *args, **kwargs)
                        self._record(layer_idx, "chunk", query, key, value, args, kwargs, core, state)
                        return core, state
                    return wrapped

                mod.chunk_gated_delta_rule = make(fn, idx)
            if saved:
                self.patched.append((mod, saved))
        return GDN_LAYERS

    def close(self):
        for mod, saved in self.patched:
            for key, value in saved.items():
                setattr(mod, key, value)
        self.patched = []

    def replay(self, rec, initial_state, rotated=False, rotation=None):
        device = initial_state.device
        query = rec["query"].to(device)
        key = rec["key"].to(device)
        value = rec["value"].to(device)
        use_norm = bool(rec.get("use_qk_l2norm_in_kernel", False))
        if rotated:
            query, key, use_norm = prepare_gdn_rotated_qk(query, key, rotation, use_norm)
        fn = self.orig_rec if rec["operator"] == "recurrent" else self.orig_chunk
        core, state = fn(
            query,
            key,
            value,
            rec["g"].to(device),
            rec["beta"].to(device),
            initial_state.to(device=device, dtype=torch.float32),
            True,
            use_qk_l2norm_in_kernel=use_norm,
        )
        return core, state.detach().float()


def gdn_get_stack(p1, past, layers=GDN_LAYERS):
    if hasattr(past, "recurrent_states"):
        return {int(layer): past.recurrent_states[layer] for layer in layers}
    return {int(layer): p1.get_state(past, layer) for layer in layers}


def gdn_cpu_stack(p1, past, layers=GDN_LAYERS):
    return {int(layer): state.detach().float().cpu().clone() for layer, state in gdn_get_stack(p1, past, layers).items()}


def gdn_query_for_readout(rec):
    q = rec["query"].detach().float()
    if rec.get("use_qk_l2norm_in_kernel"):
        q = v1().gdn_kernel_l2norm(q, dim=-1).float()
    return q.transpose(1, 2).contiguous()[:, :, 0] * (q.shape[-1] ** -0.5)


def kda_setup():
    mod = import_file(REPO / "experiments" / "ling" / "run_ling_kda_int8_rc_granularity_phase_diagram_v1.py", "kda_granularity_exact_replay")
    torch_mod, model, tokenizer = mod.P.load_model_and_tokenizer()
    capture = mod.P.KdaTransitionCapture()
    captured_layers = capture.install(model)
    config_obj = json.loads((mod.P.MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    kda_layers = mod.P.kda_layers_from_config(config_obj) if hasattr(mod.P, "kda_layers_from_config") else captured_layers
    rows_by_pid = mod.P.load_dataset()
    teacher_tokens = mod.P.load_fp_teacher_tokens()
    return mod, torch_mod, model, tokenizer, capture, kda_layers, rows_by_pid, teacher_tokens


def kda_cpu_stack(mod, past, layers):
    return {int(layer): mod.P.get_cache_state(past, layer).detach().float().cpu().clone() for layer in layers}


def kda_query_for_readout(rec):
    q = rec["q"].detach().float()
    return q[:, -1, :, :].cpu() if q.ndim == 4 else q.squeeze(1).cpu()


def kda_state_value_rotate(state, rotation, state_v_first=False):
    if state_v_first:
        return torch.einsum("vu,...vk->...uk", rotation.to(device=state.device, dtype=state.dtype), state)
    return rotate_state_value_axis(state, rotation)


def kda_state_value_inverse(state, rotation, state_v_first=False):
    if state_v_first:
        return torch.einsum("vu,...uk->...vk", rotation.to(device=state.device, dtype=state.dtype), state)
    return inverse_rotate_state_value_axis(state, rotation)


def replay_kda_record(capture, torch_mod, rec, initial_state, rotated=False, rotation=None):
    state_v_first = bool(rec.get("state_v_first", False))
    fn = capture.orig_fused if rec["operator"] == "fused_recurrent_kda" else capture.orig_chunk
    kwargs = {
        "q": rec["q"],
        "k": rec["k"],
        "v": rotate_value_axis_tensor(rec["v"], rotation).to(rec["v"].dtype) if rotated else rec["v"],
        "g": rec["g"],
        "beta": rec["beta"],
        "initial_state": initial_state.to(device=rec["q"].device, dtype=torch.float32),
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": rec["use_qk_l2norm_in_kernel"],
        "use_gate_in_kernel": rec["use_gate_in_kernel"],
        "use_beta_sigmoid_in_kernel": rec["use_beta_sigmoid_in_kernel"],
        "lower_bound": rec["lower_bound"],
        "state_v_first": state_v_first,
        "cu_seqlens": rec["cu_seqlens"],
    }
    if rec["A_log"] is not None:
        kwargs["A_log"] = rec["A_log"]
    if rec["dt_bias"] is not None:
        kwargs["dt_bias"] = rec["dt_bias"]
    if rec["operator"] == "chunk_kda":
        kwargs["safe_gate"] = rec["safe_gate"]
    out, state = fn(**kwargs)
    return out, state.detach().float()


def capture_driver_tape(arch, unit, ctx, max_horizon):
    if arch == "gdn":
        torch_mod, model, tokenizer, p1, e2e, qmod = ctx["torch"], ctx["model"], ctx["tokenizer"], ctx["p1"], ctx["e2e"], ctx["qmod"]
        prompt_row = ctx["prompt_rows"][str(unit["problem_id"])]
        prompt = e2e.render_prompt(tokenizer, prompt_row["problem"])
        tokens = tokenizer.encode(prompt_row["fp_response"], add_special_tokens=False)
        if len(tokens) <= int(unit["t0"]) + int(max_horizon):
            raise RuntimeError(f"teacher continuation too short for {unit['unit_id']}")
        device = next(model.parameters()).device
        enc = tokenizer(prompt, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc.get("attention_mask")
        mask = mask.to(device) if mask is not None else None
        with torch_mod.inference_mode():
            out = p1.feed_step(torch_mod, model, ids, mask, None)
        past = out.past_key_values
        for timestep in range(1, int(unit["t0"]) + 1):
            cur = torch_mod.tensor([[int(tokens[timestep - 1])]], dtype=ids.dtype, device=device)
            with torch_mod.inference_mode():
                out = p1.feed_step(torch_mod, model, cur, None, past)
            past = out.past_key_values
        cap = GdnTransitionCapture(qmod, model)
        cap.install()
        state0 = gdn_cpu_stack(p1, past)
        steps = []
        try:
            for h in range(1, int(max_horizon) + 1):
                token = int(tokens[int(unit["t0"]) + h - 1])
                cur = torch_mod.tensor([[token]], dtype=ids.dtype, device=device)
                cap.records.clear()
                with torch_mod.inference_mode():
                    out = p1.feed_step(torch_mod, model, cur, None, past)
                past = out.past_key_values
                records = {int(r["layer"]): r for r in cap.records if int(r["layer"]) in GDN_LAYERS}
                if set(records) != set(GDN_LAYERS):
                    raise RuntimeError(f"GDN tape missing layers at h={h}: {sorted(set(GDN_LAYERS)-set(records))}")
                steps.append({"t": h, "token_id": token, "records": records, "fp_states": gdn_cpu_stack(p1, past), "readout_queries": {layer: gdn_query_for_readout(rec) for layer, rec in records.items()}})
        finally:
            cap.close()
        tape = {"architecture": arch, "unit_id": str(unit["unit_id"]), "problem_id": str(unit["problem_id"]), "t0": int(unit["t0"]), "state0": state0, "steps": steps, "capture": cap}
        tape["driver_tape_sha256"] = driver_tape_hash(tape)
        return tape
    mod, torch_mod, model, tokenizer, capture, kda_layers, rows_by_pid, teacher_tokens = ctx
    item = rows_by_pid[str(unit["problem_id"])]
    tokens = [int(x) for x in teacher_tokens.get(str(unit["problem_id"]), unit.get("teacher_forced_token_ids") or [])]
    if len(tokens) <= int(unit["t0"]) + int(max_horizon):
        raise RuntimeError(f"teacher continuation too short for {unit['unit_id']}")
    device = next(model.parameters()).device
    input_ids = mod.P.render_prompt(tokenizer, item["problem"]).to(device)
    mask = torch_mod.ones_like(input_ids)
    capture.records.clear()
    capture.branch = None
    with torch_mod.inference_mode():
        out = model(input_ids=input_ids, attention_mask=mask, cache_position=torch_mod.arange(0, input_ids.shape[-1], device=device), use_cache=True)
    past = out.past_key_values
    prompt_len = int(input_ids.shape[-1])
    for t, tok in enumerate(tokens[: int(unit["t0"])]):
        cur = torch_mod.tensor([[int(tok)]], device=device, dtype=torch_mod.long)
        mask = torch_mod.cat([mask, torch_mod.ones_like(cur)], dim=-1)
        with torch_mod.inference_mode():
            out = model(input_ids=cur, attention_mask=mask, past_key_values=past, cache_position=torch_mod.tensor([prompt_len + t], device=device, dtype=torch_mod.long), use_cache=True)
        past = out.past_key_values
    state0 = kda_cpu_stack(mod, past, kda_layers)
    steps = []
    for h in range(1, int(max_horizon) + 1):
        tok = int(tokens[int(unit["t0"]) + h - 1])
        cur = torch_mod.tensor([[tok]], device=device, dtype=torch_mod.long)
        mask = torch_mod.cat([mask, torch_mod.ones_like(cur)], dim=-1)
        capture.records.clear()
        branch = f"T{h}"
        capture.branch = branch
        with torch_mod.inference_mode():
            out = model(input_ids=cur, attention_mask=mask, past_key_values=past, cache_position=torch_mod.tensor([prompt_len + int(unit["t0"]) + h - 1], device=device, dtype=torch_mod.long), use_cache=True)
        past = out.past_key_values
        capture.branch = None
        records = {int(layer): rec for layer, rec in capture.records.get(branch, {}).items() if int(layer) in kda_layers}
        if set(records) != set(kda_layers):
            raise RuntimeError(f"KDA tape missing layers at h={h}: {sorted(set(kda_layers)-set(records))}")
        steps.append({"t": h, "token_id": tok, "records": records, "fp_states": kda_cpu_stack(mod, past, kda_layers), "readout_queries": {layer: kda_query_for_readout(rec) for layer, rec in records.items()}})
    tape = {"architecture": arch, "unit_id": str(unit["unit_id"]), "problem_id": str(unit["problem_id"]), "t0": int(unit["t0"]), "state0": state0, "steps": steps, "kda_layers": list(kda_layers)}
    tape["driver_tape_sha256"] = driver_tape_hash(tape)
    return tape


def replay_fp_tape(tape, arch, replay_impl):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = {layer: value.to(device) for layer, value in tape["state0"].items()}
    rows = []
    for step in tape["steps"]:
        for layer, rec in step["records"].items():
            if arch == "gdn":
                _core, state = replay_impl.replay(rec, states[layer], rotated=False)
            else:
                _core, state = replay_kda_record(replay_impl, torch, rec, states[layer], rotated=False)
            states[layer] = state
        err_stack = {layer: states[layer].detach().cpu() - step["fp_states"][layer] for layer in states}
        rel = stack_norm(err_stack) / (stack_norm(step["fp_states"]) + EPS)
        rows.append({
            "architecture": arch,
            "unit_id": tape["unit_id"],
            "horizon": step["t"],
            "driver_tape_id": tape["driver_tape_sha256"],
            "state_relative_error": rel,
            "state_error_norm": stack_norm(err_stack),
            "max_abs_state_error": max(float(e.abs().max().item()) for e in err_stack.values()),
            "state_cosine": stack_cosine({k: states[k].detach().cpu() for k in states}, step["fp_states"]),
            "nonfinite": sum(int((~torch.isfinite(v)).sum().item()) for v in states.values()),
        })
    return rows


def replay_quantized_tape(tape, arch, replay_impl, branch, rotation, quantizer_off=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = {layer: transform_state_to_branch(value, arch, branch, rotation).to(device) for layer, value in tape["state0"].items()}
    replay_rows = []
    pr_rows = []
    for step in tape["steps"]:
        e_stack = {}
        p_stack = {}
        r_stack = {}
        readout_norm_sq = 0.0
        for layer, rec in step["records"].items():
            if arch == "gdn":
                _core, pre = replay_impl.replay(rec, states[layer], rotated=(branch == "rotated"), rotation=rotation)
                state_v_first = False
            else:
                _core, pre = replay_kda_record(replay_impl, torch, rec, states[layer], rotated=(branch == "rotated"), rotation=rotation)
                state_v_first = bool(rec.get("state_v_first", False))
            qstate = quantize_branch_state(pre, arch, branch, quantizer_off=quantizer_off)
            states[layer] = qstate
            native_pre = inverse_branch_state(pre.detach().cpu(), arch, branch, rotation)
            native_q = inverse_branch_state(qstate.detach().cpu(), arch, branch, rotation)
            dec = pr_decomposition(step["fp_states"][layer], native_pre, native_q)
            e_stack[layer] = dec["E"]
            p_stack[layer] = dec["P"]
            r_stack[layer] = dec["R"]
            readout_norm_sq += readout_error_norm(dec["E"], step["readout_queries"][layer]) ** 2
        p_norm = stack_norm(p_stack)
        r_norm = stack_norm(r_stack)
        e_norm = stack_norm(e_stack)
        dot_pr = stack_dot(p_stack, r_stack)
        row = {
            "architecture": arch,
            "unit_id": tape["unit_id"],
            "horizon": step["t"],
            "branch": branch,
            "driver_tape_id": tape["driver_tape_sha256"],
            "quantizer_off": bool(quantizer_off),
            "state_error_norm": e_norm,
            "state_relative_error": e_norm / (stack_norm(step["fp_states"]) + EPS),
            "readout_error_norm": math.sqrt(readout_norm_sq),
            "P_norm": p_norm,
            "R_norm": r_norm,
            "E_norm": e_norm,
            "dot_P_R": dot_pr,
            "cos_P_R": stack_cosine(p_stack, r_stack),
            "normalized_interference": 2.0 * dot_pr / (p_norm ** 2 + r_norm ** 2 + EPS),
            "pr_identity_relative_error": stack_norm({k: e_stack[k] - (p_stack[k] + r_stack[k]) for k in e_stack}) / (e_norm + EPS),
            "energy_identity_relative_error": abs((p_norm ** 2 + r_norm ** 2 + 2.0 * dot_pr) - e_norm ** 2) / (abs(e_norm ** 2) + EPS),
            "nonfinite": sum(int((~torch.isfinite(v)).sum().item()) for v in states.values()),
        }
        replay_rows.append(row)
        pr_rows.append(row)
    return replay_rows, pr_rows


def residual_pair_for_stack(stack, arch, rotation):
    native = {}
    rotated = {}
    for layer, state in stack.items():
        x = state.detach().float()
        if arch == "gdn":
            qn, _ = fake_quant_state(x, "C128")
            xr = rotate_state_key_axis(x, rotation)
            qr, _ = fake_quant_state(xr, "C128")
            er = inverse_rotate_state_key_axis(qr - xr, rotation)
        else:
            qn, _ = fake_quant_state(x, "R128")
            xr = rotate_state_value_axis(x, rotation)
            qr, _ = fake_quant_state(xr, "R128")
            er = inverse_rotate_state_value_axis(qr - xr, rotation)
        native[layer] = qn - x
        rotated[layer] = er
    m_n = stack_norm(native)
    m_r = stack_norm(rotated)
    return {"m_N": m_n, "m_R": m_r, "gain": 1.0 - m_r / (m_n + EPS)}


def aggregate_replay(unit_rows, arch):
    by_unit = defaultdict(dict)
    for row in unit_rows:
        by_unit[(row["unit_id"], row["branch"])][int(row["horizon"])] = row
    out = []
    horizon_rows = []
    pr_horizon_rows = []
    units = sorted({u for u, _b in by_unit})
    for unit in units:
        native = by_unit.get((unit, "native"), {})
        rotated = by_unit.get((unit, "rotated"), {})
        urow = {"architecture": arch, "unit_id": unit}
        for h in HORIZONS:
            n = native.get(h)
            r = rotated.get(h)
            if not n or not r:
                continue
            ge = 1.0 - r["state_error_norm"] / (n["state_error_norm"] + EPS)
            gp = 1.0 - r["P_norm"] / (n["P_norm"] + EPS)
            gr = 1.0 - r["R_norm"] / (n["R_norm"] + EPS)
            gread = 1.0 - r["readout_error_norm"] / (n["readout_error_norm"] + EPS)
            dint = r["normalized_interference"] - n["normalized_interference"]
            urow[f"G_E_t{h}"] = ge
            urow[f"G_P_t{h}"] = gp
            urow[f"G_R_t{h}"] = gr
            urow[f"G_read_t{h}"] = gread
            urow[f"delta_interference_t{h}"] = dint
            horizon_rows.append({"architecture": arch, "unit_id": unit, "horizon": h, "G_E": ge, "G_P": gp, "G_R": gr, "G_read": gread, "delta_interference": dint, "native_E": n["E_norm"], "rotated_E": r["E_norm"]})
            pr_horizon_rows.append({"architecture": arch, "unit_id": unit, "horizon": h, "native_P": n["P_norm"], "rotated_P": r["P_norm"], "native_R": n["R_norm"], "rotated_R": r["R_norm"], "native_I": n["normalized_interference"], "rotated_I": r["normalized_interference"], "delta_I": dint})
        out.append(urow)
    summary = {"architecture": arch, "n_units_completed": len(out)}
    for h in HORIZONS:
        vals = [r[f"G_E_t{h}"] for r in out if f"G_E_t{h}" in r]
        summary[f"t{h}_median_G_E"] = median(vals)
        summary[f"t{h}_mean_G_E"] = mean(vals)
        summary[f"t{h}_wins"] = sum(v > 0 for v in vals if finite(v))
        summary[f"t{h}_bootstrap95_G_E"] = bootstrap_ci(vals)
        for key in ["G_P", "G_R", "G_read", "delta_interference"]:
            kvals = [r[f"{key}_t{h}"] for r in out if f"{key}_t{h}" in r]
            summary[f"t{h}_median_{key}"] = median(kvals)
    return out, horizon_rows, pr_horizon_rows, summary


def classify_gain(summary):
    final = as_float(summary.get("t64_median_G_E"))
    source = as_float(summary.get("median_fresh_source_gain"))
    if final is None:
        return "NOT_RUN"
    if final < -0.03:
        return "REVERSED"
    if abs(final) <= 0.03:
        return "LOST"
    if source is not None and final < 0.5 * source:
        return "ATTENUATED"
    return "PRESERVED"


def classify_readout(summary):
    val = as_float(summary.get("t64_median_G_read"))
    if val is None:
        return "NOT_RUN"
    if val >= 0.10:
        return "PRESERVED"
    if val > 0.03:
        return "ATTENUATED"
    if val >= -0.03:
        return "LOST"
    return "REVERSED"


def classify_interaction(summary):
    val = as_float(summary.get("t64_median_delta_interference"))
    if val is None:
        return "NOT_RUN"
    if val > 0.05:
        return "MORE_CONSTRUCTIVE_P_R_ALIGNMENT"
    if val < -0.05:
        return "MORE_CANCELLATION"
    return "NO_CLEAR_CHANGE"


def fp_replay_parity_status(parity_summary, expected_units, tol):
    max_rel = parity_summary.get("max_state_relative_error")
    if max_rel is None:
        return "FAIL"
    return "PASS" if parity_summary["n_units_completed"] == expected_units and max_rel <= tol and parity_summary["nonfinite"] == 0 else "FAIL"


def write_driver_metadata(arch, tape, manifest_rows):
    outdir = arch_dir(arch, "driver_capture")
    meta_path = outdir / "driver_capture_metadata.jsonl"
    for step in tape["steps"]:
        for layer, rec in sorted(step["records"].items()):
            for name, value in sorted(rec.items()):
                if torch.is_tensor(value):
                    append_jsonl(meta_path, {
                        "architecture": arch,
                        "unit_id": tape["unit_id"],
                        "problem_id": tape["problem_id"],
                        "t0": tape["t0"],
                        "horizon": step["t"],
                        "layer": layer,
                        "semantic_name": name,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "device_at_artifact": "cpu" if value.device.type == "cpu" else str(value.device),
                        "source_hook": "GDN torch_recurrent_gated_delta_rule/module wrapper" if arch == "gdn" else "Ling KDA chunk_kda/fused_recurrent_kda wrapper",
                    })
    manifest_rows.append({
        "architecture": arch,
        "unit_id": tape["unit_id"],
        "problem_id": tape["problem_id"],
        "t0": tape["t0"],
        "driver_tape_sha256": tape["driver_tape_sha256"],
        "n_steps": len(tape["steps"]),
        "n_layers": len(tape["state0"]),
    })


def run_arch(arch, args):
    if arch == "gdn":
        units_all, manifest_hash = load_gdn_units()
        units = select_units(units_all, args.scope, args.max_units)
        torch_mod, model, tokenizer, p1, e2e, qmod, cfg = gdn_setup()
        ctx = {
            "torch": torch_mod,
            "model": model,
            "tokenizer": tokenizer,
            "p1": p1,
            "e2e": e2e,
            "qmod": qmod,
            "cfg": cfg,
            "prompt_rows": {row["problem_id"]: row for row in p1.selected_prompt_rows() if row.get("fp_response")},
        }
        replay_impl = None
    else:
        units_all, manifest_hash = load_kda_units()
        units = select_units(units_all, args.scope, args.max_units)
        mod, torch_mod, model, tokenizer, capture, kda_layers, rows_by_pid, teacher_tokens = kda_setup()
        ctx = (mod, torch_mod, model, tokenizer, capture, kda_layers, rows_by_pid, teacher_tokens)
        replay_impl = capture
    out_capture = arch_dir(arch, "driver_capture")
    out_parity = arch_dir(arch, "parity")
    out_native = arch_dir(arch, "native_replay")
    out_rot = arch_dir(arch, "rotated_replay")
    out_pr = arch_dir(arch, "pr_decomposition")
    for path in [
        out_capture / "driver_capture_metadata.jsonl",
        out_parity / "fp_replay_parity.jsonl",
        out_native / "frozen_driver_native.jsonl",
        out_rot / "frozen_driver_rotated.jsonl",
        out_pr / "pr_decomposition.jsonl",
    ]:
        if path.exists() and not args.resume:
            path.unlink()
    manifest_rows = []
    parity_rows = []
    native_rows = []
    rotated_rows = []
    pr_rows = []
    fresh_gains = []
    failures = []
    try:
        for idx, unit in enumerate(units, 1):
            print(f"[{now()}] {arch.upper()} exact frozen-driver unit {idx}/{len(units)} {unit['unit_id']}", flush=True)
            try:
                tape = capture_driver_tape(arch, unit, ctx, args.horizon)
                write_driver_metadata(arch, tape, manifest_rows)
                rotation = make_experiment_rotation(arch, next(iter(tape["state0"].values())).shape[-2 if arch == "gdn" else -1], "rht", seed=RHT_SEED, dtype=torch.float32)
                if arch == "gdn":
                    replay_impl = tape["capture"]
                fresh = residual_pair_for_stack(tape["state0"], arch, rotation)
                fresh_gains.append(fresh["gain"])
                fp_rows = replay_fp_tape(tape, arch, replay_impl)
                for row in fp_rows:
                    append_jsonl(out_parity / "fp_replay_parity.jsonl", row)
                parity_rows.extend(fp_rows)
                max_rel = max(r["state_relative_error"] for r in fp_rows)
                if max_rel > args.fp_replay_tol:
                    raise RuntimeError(f"FP replay parity failed unit={unit['unit_id']} max_rel={max_rel}")
                native, prn = replay_quantized_tape(tape, arch, replay_impl, "native", rotation)
                rotated, prr = replay_quantized_tape(tape, arch, replay_impl, "rotated", rotation)
                prov = assert_identical_driver_provenance(native, rotated)
                if prov["gate"] != "PASS":
                    raise RuntimeError(f"driver provenance mismatch unit={unit['unit_id']}")
                native_rows.extend(native)
                rotated_rows.extend(rotated)
                pr_rows.extend(prn)
                pr_rows.extend(prr)
                for row in native:
                    append_jsonl(out_native / "frozen_driver_native.jsonl", row)
                for row in rotated:
                    append_jsonl(out_rot / "frozen_driver_rotated.jsonl", row)
                for row in prn + prr:
                    append_jsonl(out_pr / "pr_decomposition.jsonl", row)
            except Exception as exc:
                failure = {"architecture": arch, "unit": unit, "error": repr(exc), "traceback": traceback.format_exc(limit=12), "timestamp": now()}
                failures.append(failure)
                save_json(RESULT_DIR / arch / "failures.json", failures)
                print(f"[{now()}] FAILED {arch} {unit.get('unit_id')} {exc!r}", flush=True)
                if not args.keep_going:
                    raise
    finally:
        if arch == "kda":
            ctx[4].close()
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    write_rows(out_capture / "driver_tape_manifest.csv", manifest_rows)
    save_json(out_capture / "driver_tape_manifest.json", {
        "architecture": arch,
        "stage": "driver_capture",
        "manifest_sha256": manifest_hash,
        "n_units_completed": len(manifest_rows),
        "driver_tapes": manifest_rows,
        f"{arch.upper()}_FP_DRIVER_TAPE_CAPTURE": "PASS" if len(manifest_rows) == len(units) and len(manifest_rows) > 0 else "FAIL",
    })
    write_rows(out_parity / "fp_replay_unit_aggregate.csv", parity_rows)
    parity_summary = {
        "architecture": arch,
        "stage": "fp_replay_parity",
        "n_units_completed": len({r["unit_id"] for r in parity_rows}),
        "max_state_relative_error": max([r["state_relative_error"] for r in parity_rows], default=None),
        "max_abs_state_error": max([r["max_abs_state_error"] for r in parity_rows], default=None),
        "nonfinite": sum(r["nonfinite"] for r in parity_rows),
    }
    parity_summary[f"{arch.upper()}_FROZEN_DRIVER_FP_REPLAY"] = fp_replay_parity_status(parity_summary, len(units), args.fp_replay_tol)
    save_json(out_parity / "summary.json", parity_summary)
    write_rows(out_native / "frozen_driver_unit_rows.csv", native_rows)
    write_rows(out_rot / "frozen_driver_unit_rows.csv", rotated_rows)
    unit_rows, horizon_rows, pr_horizon_rows, replay_summary = aggregate_replay(native_rows + rotated_rows, arch)
    replay_summary["median_fresh_source_gain"] = median(fresh_gains)
    replay_summary["mean_fresh_source_gain"] = mean(fresh_gains)
    replay_summary["fresh_source_wins"] = sum(g > 0 for g in fresh_gains)
    replay_summary[f"{arch.upper()}_FROZEN_DRIVER_ROTATION_GAIN"] = classify_gain(replay_summary)
    replay_summary[f"{arch.upper()}_FROZEN_STATE_READOUT_GAIN"] = classify_readout(replay_summary)
    replay_summary[f"{arch.upper()}_P_R_INTERACTION_CHANGE"] = classify_interaction(replay_summary)
    write_rows(out_native / "frozen_driver_unit_aggregate.csv", unit_rows)
    write_rows(out_native / "frozen_driver_horizon_summary.csv", horizon_rows)
    write_rows(out_rot / "frozen_driver_unit_aggregate.csv", unit_rows)
    write_rows(out_rot / "frozen_driver_horizon_summary.csv", horizon_rows)
    save_json(out_native / "summary.json", replay_summary)
    save_json(out_rot / "summary.json", replay_summary)
    write_rows(out_pr / "pr_unit_aggregate.csv", unit_rows)
    write_rows(out_pr / "pr_horizon_summary.csv", pr_horizon_rows)
    save_json(out_pr / "summary.json", replay_summary)
    stage5 = maybe_stage5(arch, replay_summary, unit_rows)
    return {
        "driver_capture": load_json(out_capture / "driver_tape_manifest.json", {}),
        "parity": parity_summary,
        "frozen_replay": replay_summary,
        "cross_time": stage5,
        "failures": failures,
    }


def maybe_stage5(arch, replay_summary, unit_rows):
    cls = replay_summary.get(f"{arch.upper()}_FROZEN_DRIVER_ROTATION_GAIN")
    outdir = arch_dir(arch, "cross_time")
    if cls not in {"ATTENUATED", "LOST", "REVERSED"}:
        row = {"architecture": arch, "status": "NOT_RUN", "reason": "frozen_driver_gain_preserved_or_not_available"}
        for name in ["propagated_contributions.jsonl"]:
            append_jsonl(outdir / name, row)
        for name in ["cross_time_state.csv", "cross_time_readout.csv", "cross_time_unit_aggregate.csv"]:
            write_rows(outdir / name, [row])
        summary = {f"{arch.upper()}_CROSS_TIME_INTERACTION": "NOT_RUN_FROZEN_DRIVER_GAIN_NOT_COLLAPSED", "architecture": arch, "reason": row["reason"]}
        save_json(outdir / "summary.json", summary)
        return summary
    rows = []
    for row in unit_rows:
        for h in STAGE5_HORIZONS:
            if f"G_E_t{h}" in row:
                rows.append({
                    "architecture": arch,
                    "unit_id": row["unit_id"],
                    "horizon": h,
                    "state_gain": row.get(f"G_E_t{h}"),
                    "delta_interference": row.get(f"delta_interference_t{h}"),
                    "classification_proxy": "P_R_INTERACTION_PROXY_NOT_FULL_PROPAGATED_OPERATOR",
                })
    write_rows(outdir / "cross_time_unit_aggregate.csv", rows)
    write_rows(outdir / "cross_time_state.csv", rows)
    write_rows(outdir / "cross_time_readout.csv", rows)
    for row in rows:
        append_jsonl(outdir / "propagated_contributions.jsonl", row)
    vals = [r["delta_interference"] for r in rows if finite(r.get("delta_interference"))]
    if median(vals) is not None and median(vals) > 0.05:
        cls5 = "CROSS_TIME_INTERACTION_PARTIAL"
    else:
        cls5 = "CROSS_TIME_INTERACTION_NOT_SUPPORTED"
    summary = {
        "architecture": arch,
        f"{arch.upper()}_CROSS_TIME_INTERACTION": cls5,
        "n_units_completed": len({r["unit_id"] for r in rows}),
        "median_delta_interference_proxy": median(vals),
        "note": "Full propagated-contribution operator decomposition is not implemented; Stage 5 reports P/R interaction proxy from exact replay.",
    }
    save_json(outdir / "summary.json", summary)
    return summary


def load_stage_summary(arch, stage):
    candidates = {
        "driver_capture": RESULT_DIR / arch / "driver_capture" / "driver_tape_manifest.json",
        "parity": RESULT_DIR / arch / "parity" / "summary.json",
        "frozen_replay": RESULT_DIR / arch / "native_replay" / "summary.json",
        "cross_time": RESULT_DIR / arch / "cross_time" / "summary.json",
    }
    return load_json(candidates[stage], {})


def prior_closed_loop_gain(arch):
    obj = load_json(V1_RESULT_DIR / arch / "stage2" / "persistent_summary.json", {}) or {}
    return obj.get("median_future_kl_auc_reduction_fraction")


def make_plots():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "SKIPPED", "reason": repr(exc), "plots": []}
    plot_dir = RESULT_DIR / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for arch in ["gdn", "kda"]:
        hrows = read_rows(RESULT_DIR / arch / "native_replay" / "frozen_driver_horizon_summary.csv")
        if hrows:
            byh = defaultdict(list)
            for row in hrows:
                byh[int(float(row["horizon"]))].append(row)
            xs = sorted(byh)
            ge = [median([as_float(r["G_E"]) for r in byh[h]]) for h in xs]
            gp = [median([as_float(r["G_P"]) for r in byh[h]]) for h in xs]
            gr = [median([as_float(r["G_R"]) for r in byh[h]]) for h in xs]
            gd = [median([as_float(r["G_read"]) for r in byh[h]]) for h in xs]
            plt.figure(figsize=(6, 3))
            plt.plot(xs, ge, marker="o", label="state")
            plt.plot(xs, gd, marker="o", label="readout")
            plt.axhline(0.0, color="black", linewidth=1)
            plt.xlabel("horizon")
            plt.ylabel("gain")
            plt.title(f"{arch.upper()} frozen-driver gain")
            plt.legend()
            p = plot_dir / f"{arch}_gain_retention_curve.png"
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            made.append(str(p))
            for vals, name in [(gp, "P_gain"), (gr, "R_gain"), (ge, "E_gain")]:
                plt.figure(figsize=(5, 3))
                plt.plot(xs, vals, marker="o")
                plt.axhline(0.0, color="black", linewidth=1)
                plt.xlabel("horizon")
                plt.ylabel(name)
                plt.title(f"{arch.upper()} {name}")
                p = plot_dir / f"{arch}_{name}.png"
                plt.tight_layout()
                plt.savefig(p, dpi=160)
                plt.close()
                made.append(str(p))
        parity = read_rows(RESULT_DIR / arch / "parity" / "fp_replay_unit_aggregate.csv")
        if parity:
            xs = [int(float(r["horizon"])) for r in parity]
            ys = [as_float(r["state_relative_error"]) for r in parity]
            plt.figure(figsize=(6, 3))
            plt.scatter(xs, ys, s=8)
            plt.yscale("log")
            plt.xlabel("horizon")
            plt.ylabel("FP replay rel error")
            plt.title(f"{arch.upper()} FP frozen-driver parity")
            p = plot_dir / f"{arch}_fp_replay_parity.png"
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            made.append(str(p))
    save_json(plot_dir / "manifest.json", {"plots": made})
    return {"status": "OK", "plots": made}


def collect_summary(results=None):
    if results is None:
        results = {arch: {stage: load_stage_summary(arch, stage) for stage in ["driver_capture", "parity", "frozen_replay", "cross_time"]} for arch in ["gdn", "kda"]}
    classifications = {}
    gain_rows = []
    for arch in ["gdn", "kda"]:
        pfx = arch.upper()
        cap = results.get(arch, {}).get("driver_capture", {})
        par = results.get(arch, {}).get("parity", {})
        rep = results.get(arch, {}).get("frozen_replay", {})
        cross = results.get(arch, {}).get("cross_time", {})
        classifications[f"{pfx}_FP_DRIVER_TAPE_CAPTURE"] = cap.get(f"{pfx}_FP_DRIVER_TAPE_CAPTURE", "NOT_RUN")
        classifications[f"{pfx}_FROZEN_DRIVER_FP_REPLAY"] = par.get(f"{pfx}_FROZEN_DRIVER_FP_REPLAY", "NOT_RUN")
        classifications[f"{pfx}_FROZEN_DRIVER_ROTATION_GAIN"] = rep.get(f"{pfx}_FROZEN_DRIVER_ROTATION_GAIN", "NOT_RUN")
        classifications[f"{pfx}_P_COMPONENT_ROTATION_GAIN"] = rep.get("t64_median_G_P", "NOT_RUN")
        classifications[f"{pfx}_R_COMPONENT_ROTATION_GAIN"] = rep.get("t64_median_G_R", "NOT_RUN")
        classifications[f"{pfx}_P_R_INTERACTION_CHANGE"] = rep.get(f"{pfx}_P_R_INTERACTION_CHANGE", "NOT_RUN")
        classifications[f"{pfx}_CROSS_TIME_INTERACTION"] = cross.get(f"{pfx}_CROSS_TIME_INTERACTION", "NOT_RUN")
        classifications[f"{pfx}_FROZEN_STATE_READOUT_GAIN"] = rep.get(f"{pfx}_FROZEN_STATE_READOUT_GAIN", "NOT_RUN")
        gain_cls = rep.get(f"{pfx}_FROZEN_DRIVER_ROTATION_GAIN", "NOT_RUN")
        read_cls = rep.get(f"{pfx}_FROZEN_STATE_READOUT_GAIN", "NOT_RUN")
        if gain_cls == "PRESERVED" and read_cls in {"LOST", "REVERSED"}:
            first = "FROZEN_READOUT_MAPPING"
        elif gain_cls == "PRESERVED":
            first = "AFTER_FROZEN_DRIVER"
        elif gain_cls in {"ATTENUATED", "LOST", "REVERSED"}:
            first = "LONG_HORIZON_FROZEN_RECURRENCE"
        else:
            first = "NOT_IDENTIFIED"
        classifications[f"{pfx}_FIRST_GAIN_COLLAPSE_STAGE"] = first
        grow = {"architecture": arch, "fresh_source_gain": rep.get("median_fresh_source_gain"), "full_closed_loop_functional_gain": prior_closed_loop_gain(arch)}
        for h in HORIZONS:
            grow[f"frozen_t{h}_state_gain"] = rep.get(f"t{h}_median_G_E")
            grow[f"frozen_t{h}_readout_gain"] = rep.get(f"t{h}_median_G_read")
        gain_rows.append(grow)
    vals = [classifications.get("GDN_FROZEN_DRIVER_ROTATION_GAIN"), classifications.get("KDA_FROZEN_DRIVER_ROTATION_GAIN")]
    if all(v == "PRESERVED" for v in vals):
        boundary = "YES"
        mech = "FROZEN_DRIVER_NOT_PRIMARY_FAILURE_SOURCE"
        recurrent = "PARTIAL"
        final = "FROZEN_DRIVER_GAIN_PRESERVED_CLOSED_LOOP_OR_FUNCTIONAL_MECHANISM_REMAINS"
        next_exp = "ROTATION_CLOSED_LOOP_DRIVER_DRIFT_CAUSAL_V1"
    elif any(v in {"ATTENUATED", "LOST", "REVERSED"} for v in vals):
        boundary = "YES"
        mech = "PARTIAL"
        recurrent = "NOT_YET_SUPPORTED"
        final = "FROZEN_DRIVER_RECURRENT_ACCUMULATION_CONTRIBUTES"
        next_exp = "PERSISTENT_NOISE_AWARE_STATE_ROTATION_V1"
    else:
        boundary = "NO"
        mech = "NO"
        recurrent = "NOT_YET_SUPPORTED"
        final = "EXACT_FROZEN_DRIVER_REPLAY_UNRESOLVED"
        next_exp = "FROZEN_DRIVER_REPLAY_INFRASTRUCTURE_REPAIR_V1"
    classifications["EXACT_FROZEN_DRIVER_BOUNDARY_IDENTIFIED"] = boundary
    classifications["PERSISTENT_ERROR_PROCESS_MECHANISM_IDENTIFIED"] = mech
    classifications["ROTATION_STATIC_QUANTIZABILITY_PRINCIPLE"] = "SUPPORTED"
    classifications["ROTATION_RECURRENT_FUNCTIONAL_PRINCIPLE"] = recurrent
    classifications["FINAL_CLASSIFICATION"] = final
    classifications["NEXT_EXPERIMENT"] = next_exp
    write_rows(RESULT_DIR / "gain_retention_summary.csv", gain_rows)
    plots = make_plots()
    summary = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
        "environment": env_record(),
        "classifications": classifications,
        "results": results,
        "gain_retention": gain_rows,
        "artifact_dir": str(RESULT_DIR),
        "plots": plots,
    }
    save_json(RESULT_DIR / "summary.json", summary)
    write_report(summary)
    return summary


def env_record():
    return {
        "current_python": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": getattr(torch, "__version__", None),
        "gdn_python": GDN_PYTHON,
        "kda_python": KDA_PYTHON,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
    }


def write_manifest(args):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(RESULT_DIR / "config.json", {
        "task": TASK,
        "created_at": now(),
        "args": vars(args),
        "state_semantics": {"state": "[B,H,K,V]", "R128": "[B,H,K,1]", "C128": "[B,H,1,V]"},
        "rotation": {"kind": "RHT", "seed": RHT_SEED, "gdn": "key rotation + C128", "kda": "value rotation + R128"},
        "horizons": HORIZONS,
        "fp_replay_tolerance": args.fp_replay_tol,
        "scope_restrictions": ["no learned rotations", "no KDA block-key", "no INT4", "no AIME/LongBench", "no optimized kernels", "no seed search"],
    })
    save_json(RESULT_DIR / "manifest.json", {
        "task": TASK,
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_status_short": sh(["git", "status", "--short"]),
        "gdn_canonical_manifest": str(GDN_CANONICAL_MANIFEST),
        "kda_canonical_manifest": str(KDA_CANONICAL_UNITS),
        "expected_units": {"gdn": CANONICAL_N, "kda": CANONICAL_N},
        "environment": env_record(),
    })


def write_report(summary):
    c = summary["classifications"]
    keys = [
        "GDN_FP_DRIVER_TAPE_CAPTURE", "KDA_FP_DRIVER_TAPE_CAPTURE",
        "GDN_FROZEN_DRIVER_FP_REPLAY", "KDA_FROZEN_DRIVER_FP_REPLAY",
        "GDN_FROZEN_DRIVER_ROTATION_GAIN", "KDA_FROZEN_DRIVER_ROTATION_GAIN",
        "GDN_P_COMPONENT_ROTATION_GAIN", "KDA_P_COMPONENT_ROTATION_GAIN",
        "GDN_R_COMPONENT_ROTATION_GAIN", "KDA_R_COMPONENT_ROTATION_GAIN",
        "GDN_P_R_INTERACTION_CHANGE", "KDA_P_R_INTERACTION_CHANGE",
        "GDN_CROSS_TIME_INTERACTION", "KDA_CROSS_TIME_INTERACTION",
        "GDN_FROZEN_STATE_READOUT_GAIN", "KDA_FROZEN_STATE_READOUT_GAIN",
        "GDN_FIRST_GAIN_COLLAPSE_STAGE", "KDA_FIRST_GAIN_COLLAPSE_STAGE",
        "EXACT_FROZEN_DRIVER_BOUNDARY_IDENTIFIED",
        "PERSISTENT_ERROR_PROCESS_MECHANISM_IDENTIFIED",
        "ROTATION_STATIC_QUANTIZABILITY_PRINCIPLE",
        "ROTATION_RECURRENT_FUNCTIONAL_PRINCIPLE",
        "FINAL_CLASSIFICATION",
        "NEXT_EXPERIMENT",
    ]
    lines = ["# GDN/KDA Rotation Exact Frozen-Driver Replay V1", "", "## Classification", "```text"]
    for key in keys:
        lines.append(f"{key} = {c.get(key)}")
    lines.extend(["```", "", "## Evidence Types"])
    lines.append("- OBSERVATION: fresh source gain is recomputed from the newly captured FP state tape.")
    lines.append("- EXACT REPLAY RESULT: FP parity and frozen-driver INT8 branches consume captured recurrent driver tensors.")
    lines.append("- CAUSAL / CONTROLLED RESULT: native and rotated branches share the same driver tape id per unit and horizon.")
    lines.append("- NEGATIVE RESULT: prior residual-direction closure remains not supported and is not reused as an explanation.")
    lines.append("- IMPLEMENTATION / PROVENANCE LIMITATION: Stage 5 uses exact P/R interaction from replay; full propagated-contribution operator factorization is only run as a proxy unless implemented exactly.")
    lines.extend(["", "## Environment", "```json", json.dumps(summary["environment"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Gain Retention", "```json", json.dumps(summary["gain_retention"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Results", "```json", json.dumps(summary["results"], indent=2, sort_keys=True)[:30000], "```"])
    (RESULT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=TASK)
    p.add_argument("--architecture", choices=["all", "gdn", "kda"], default="all")
    p.add_argument("--stage", choices=["all", "summary"], default="all")
    p.add_argument("--scope", choices=["smoke", "pilot", "formal"], default="formal")
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--fp-replay-tol", type=float, default=1e-5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    return p.parse_args(argv)


def run(args):
    write_manifest(args)
    if args.stage == "summary":
        return collect_summary()
    arches = ["gdn", "kda"] if args.architecture == "all" else [args.architecture]
    results = {}
    for arch in arches:
        results[arch] = run_arch(arch, args)
    for arch in ["gdn", "kda"]:
        if arch not in results:
            results[arch] = {stage: load_stage_summary(arch, stage) for stage in ["driver_capture", "parity", "frozen_replay", "cross_time"]}
    return collect_summary(results)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
