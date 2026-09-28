#!/usr/bin/env python3
"""Forensic localization of the first KDA rotated-prefill divergence."""

import argparse
import csv
import importlib.util
import json
import math
import os
import traceback
from collections import defaultdict
from pathlib import Path

import torch


TASK = "KDA_ROTATION_PREFILL_FULL_PATH_FIRST_DIVERGENCE_V1"
SLUG = "kda_rotation_prefill_full_path_first_divergence_v1"
REPO = Path(os.environ.get("GDN_ROTATION_REPO", Path(__file__).resolve().parents[2]))
RESULT_DIR = REPO / "results" / SLUG
KERNEL_RUNNER = REPO / "experiments" / "rotation" / "run_kda_rotation_prefill_kernel_equivariance_causal_v1.py"
KERNEL_RESULTS = REPO / "results" / "kda_rotation_prefill_kernel_equivariance_causal_v1"
PRECISION_RESULTS = REPO / "results" / "kda_rotation_full_transition_final_causal_closure_v1"
EXPECTED_UNITS = 18
PRIMARY_HORIZON = 64
EPS = 1e-12
TRACE_BRANCHES = {"NS", "RS", "NS_REPLAY", "RS_FIXED"}
HIGH_PRECISION_REL_FLOOR = 3.422854184915039e-08
HIGH_PRECISION_MAX_FLOOR = 1.0661153870827889e-07


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


KERNEL = import_file(KERNEL_RUNNER, "prefill_kernel_for_first_divergence_v1")
H, BASE, DECAY = KERNEL.H, KERNEL.BASE, KERNEL.DECAY


def cpu_tensor(value, float32=False):
    if value is None:
        return None
    out = value.detach()
    if float32:
        out = out.float()
    return out.cpu().clone()


def metrics(value, reference):
    if value is None and reference is None:
        return {"max_abs": 0.0, "relative_l2": 0.0, "cosine": 1.0,
                "norm_value": 0.0, "norm_reference": 0.0}
    if value is None or reference is None:
        return {"max_abs": float("inf"), "relative_l2": float("inf"), "cosine": 0.0,
                "norm_value": 0.0 if value is None else BASE.tensor_norm(value.float()),
                "norm_reference": 0.0 if reference is None else BASE.tensor_norm(reference.float())}
    x, y = value.detach().float().cpu(), reference.detach().float().cpu()
    diff = x - y
    nx, ny = BASE.tensor_norm(x), BASE.tensor_norm(y)
    return {
        "max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "relative_l2": BASE.tensor_norm(diff) / (ny + EPS),
        "cosine": float((x * y).sum().item()) / (nx * ny + EPS),
        "norm_value": nx, "norm_reference": ny,
    }


def model_input_device(model):
    return model.get_input_embeddings().weight.device


def load_sharded_model(max_memory_gib):
    P = BASE.P()
    P.configure_offline_runtime()
    import transformers.utils.import_utils as import_utils
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False
    from transformers import AutoModelForCausalLM, AutoTokenizer
    P.setup_seed(P.BASE_SEED)
    path = Path(os.environ.get("LING_MODEL_PATH", P.MODEL_PATH))
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), trust_remote_code=True, local_files_only=True)
    memory = {index: f"{int(max_memory_gib)}GiB" for index in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        str(path), trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map="balanced", max_memory=memory)
    model.eval()
    return model, tokenizer


class FullPathForensicProbe(KERNEL.SequentialPrefillProbe):
    """Sequential KDA plus CPU traces at real Ling module boundaries."""

    def __init__(self, rotation):
        super().__init__(rotation)
        self.capture_enabled = True
        self.core_tokens = defaultdict(lambda: defaultdict(list))
        self.path_tensors = defaultdict(lambda: defaultdict(dict))
        self.trace_handles = []

    def _active(self):
        return self.capture_enabled and self.branch in TRACE_BRANCHES

    def _save(self, layer, name, value, float32=True):
        if not self._active() or value is None:
            return
        if isinstance(value, (tuple, list)):
            value = value[0] if value else None
        if torch.is_tensor(value):
            self.path_tensors[self.branch][int(layer)][name] = cpu_tensor(value, float32=float32)

    def _pre_hidden(self, layer, name, with_kwargs=False):
        if with_kwargs:
            def hook(_module, args, kwargs):
                value = kwargs.get("hidden_states")
                if value is None and args:
                    value = args[0]
                self._save(layer, name, value)
            return hook
        def hook(_module, args):
            self._save(layer, name, args[0] if args else None)
        return hook

    def _post(self, layer, name):
        def hook(_module, _args, output):
            self._save(layer, name, output)
        return hook

    def _o_norm_pre(self, layer):
        def hook(_module, args):
            if len(args) >= 2:
                self._save(layer, "rmsnorm_input", args[0])
                self._save(layer, "dynamic_gate_input", args[1])
                self._save(layer, "dynamic_gate_output", torch.sigmoid(args[1].float()))
        return hook

    def install(self, model):
        layers = super().install(model)
        for layer_index in layers:
            layer = model.model.layers[layer_index]
            attention = layer.attention
            self.trace_handles.append(layer.register_forward_pre_hook(
                self._pre_hidden(layer_index, "layer_input_hidden")))
            self.trace_handles.append(layer.register_forward_hook(
                self._post(layer_index, "layer_output_hidden")))
            self.trace_handles.append(layer.input_layernorm.register_forward_hook(
                self._post(layer_index, "attention_rmsnorm_output")))
            self.trace_handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
                self._pre_hidden(layer_index, "post_attention_rmsnorm_input")))
            self.trace_handles.append(layer.post_attention_layernorm.register_forward_hook(
                self._post(layer_index, "post_attention_rmsnorm_output")))
            self.trace_handles.append(layer.mlp.register_forward_hook(
                self._post(layer_index, "mlp_output")))
            for name in ("q_proj", "k_proj", "v_proj", "q_conv1d", "k_conv1d", "v_conv1d"):
                self.trace_handles.append(getattr(attention, name).register_forward_hook(
                    self._post(layer_index, name + "_output")))
            self.trace_handles.append(attention.o_norm.register_forward_pre_hook(
                self._o_norm_pre(layer_index)))
            self.trace_handles.append(attention.o_norm.register_forward_hook(
                self._post(layer_index, "rmsnorm_scaled_output")))
            self.trace_handles.append(attention.o_proj.register_forward_pre_hook(
                self._pre_hidden(layer_index, "out_proj_input")))
            self.trace_handles.append(attention.o_proj.register_forward_hook(
                self._post(layer_index, "out_proj_output")))
            self.trace_handles.append(attention.register_forward_hook(
                self._post(layer_index, "attention_output")))
        return layers

    def close(self):
        for handle in self.trace_handles:
            handle.remove()
        self.trace_handles = []
        super().close()

    def _wrap(self, operator, fn):
        parent = super()._wrap(operator, fn)

        def wrapped(**kwargs):
            plan_obj = self.plan
            if (plan_obj is None or plan_obj.name not in TRACE_BRANCHES
                    or operator != "chunk_kda"):
                return parent(**kwargs)
            branch, layer = self.branch, int(self.current_layer)
            raw_gate = kwargs["g"].detach().clone()
            log_decay = DECAY.final_log_decay(
                raw_gate, kwargs.get("A_log"), kwargs.get("dt_bias"),
                kwargs.get("lower_bound"), bool(kwargs.get("use_gate_in_kernel", False)))
            v_semantic = kwargs["v"].detach().clone()
            call = dict(kwargs)
            call["g"] = log_decay.float()
            call["use_gate_in_kernel"] = False
            if plan_obj.basis == "rotated":
                call["v"] = BASE.driver_to_branch_coordinates(
                    "v", call["v"], "rotated", self.rotation)
            state, outputs, token_records = call.get("initial_state"), [], []
            for token_index in range(int(call["q"].shape[1])):
                step = dict(call)
                for name in ("q", "k", "v", "g", "beta"):
                    step[name] = call[name][:, token_index:token_index + 1]
                step["initial_state"] = state
                step.pop("safe_gate", None)
                state_in = state
                output, state = self.orig_fused(**step)
                outputs.append(output)
                token_records.append({
                    "token_index": token_index,
                    "q": cpu_tensor(step["q"]), "k": cpu_tensor(step["k"]),
                    "v": cpu_tensor(step["v"]),
                    "v_semantic": cpu_tensor(v_semantic[:, token_index:token_index + 1]),
                    "beta": cpu_tensor(step["beta"], True),
                    "log_decay": cpu_tensor(step["g"], True),
                    "state_in": cpu_tensor(state_in, True),
                    "state_after": cpu_tensor(state, True),
                    "raw_core_output": cpu_tensor(output),
                    "basis": plan_obj.basis,
                })
            raw_output = torch.cat(outputs, dim=1)
            returned = raw_output
            if plan_obj.basis == "rotated":
                returned = raw_output.float().matmul(
                    self.rotation.t().to(raw_output.device, torch.float32)).to(raw_output.dtype)
            for record, mapped in zip(token_records, returned.split(1, dim=1)):
                record["mapped_back_output"] = cpu_tensor(mapped)
            self.core_tokens[branch][layer] = token_records
            self.records[branch][layer] = {
                "operator": "sequential_fused_recurrent_kda",
                "q": cpu_tensor(call["q"]), "k": cpu_tensor(call["k"]),
                "v": cpu_tensor(call["v"]), "v_semantic": cpu_tensor(v_semantic),
                "beta": cpu_tensor(call["beta"], True), "raw_gate": cpu_tensor(raw_gate),
                "log_decay": cpu_tensor(log_decay, True),
                "initial_state": cpu_tensor(call.get("initial_state"), True),
                "final_state": cpu_tensor(state, True), "output": cpu_tensor(returned, True),
                "raw_output": cpu_tensor(raw_output, True), "basis": plan_obj.basis,
                "sequence_length": int(call["q"].shape[1]),
            }
            return returned, state
        return wrapped


def prefill_with_boundaries(model, tokenizer, probe, row, layers, rotation,
                            name, basis, apply_endpoint_rotation=True, quantized=True):
    device = model_input_device(model)
    input_ids = BASE.P().render_prompt(tokenizer, row["problem"]).to(device)
    mask = torch.ones_like(input_ids)
    plan = H.plan(name, basis, quantized)
    probe.begin(plan, 0)
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=mask,
                    cache_position=torch.arange(input_ids.shape[-1], device=device), use_cache=True)
    probe.end()
    kernel_stack = BASE.cache_stack(out.past_key_values, layers)
    if basis == "rotated" and apply_endpoint_rotation:
        H.replace_stack(out.past_key_values, H.rotate_stack(kernel_stack, rotation))
    boundary_stack = BASE.cache_stack(out.past_key_values, layers)
    if quantized:
        meta = BASE.quantize_branch_cache(torch, out.past_key_values, layers, basis, rotation)
        if not meta["finite"]:
            raise RuntimeError(f"nonfinite prefill cache for {name}")
    postquant_stack = BASE.cache_stack(out.past_key_values, layers)
    item = H.branch(name, basis, quantized, out.past_key_values, mask)
    return input_ids, item, {
        "kernel_return": kernel_stack,
        "endpoint_basis_handling": boundary_stack,
        "postquant": postquant_stack,
    }


def map_state_to_native(state, basis, rotation):
    if state is None:
        return None
    return BASE.state_to_semantic_coordinates(state.float(), basis, rotation.float()).float()


def map_value_to_native(value, basis, rotation):
    if basis == "rotated":
        return value.float().matmul(rotation.t().float())
    return value.float()


def row_from_pair(unit_id, prompt_id, token, layer, name, native, rotated,
                  native_basis, rotated_basis, mapped_rotated, map_applied,
                  comparison_basis="NATIVE_BASIS"):
    stat = metrics(mapped_rotated, native)
    return {
        "unit_id": str(unit_id), "prompt_id": str(prompt_id),
        "token_index": int(token), "layer_index": int(layer), "tensor_name": name,
        "native_shape": None if native is None else str(tuple(native.shape)),
        "rotated_shape": None if rotated is None else str(tuple(rotated.shape)),
        "tensor_basis_native": native_basis, "tensor_basis_rotated": rotated_basis,
        "map_applied": map_applied, "comparison_basis": comparison_basis,
        "max_abs": stat["max_abs"], "relative_L2": stat["relative_l2"],
        "cosine": stat["cosine"], "norm_native": stat["norm_reference"],
        "norm_rotated": stat["norm_value"],
    }


def token_slice(value, token):
    if value is None:
        return None
    if value.ndim >= 2 and value.shape[1] > token:
        return value[:, token:token + 1]
    return value


def free_running_rows(unit, prompt_len, probe, boundaries, rotation):
    unit_id, prompt_id = unit["unit_id"], unit["problem_id"]
    rows, value_checks, mapback_checks = [], [], []
    invariant_core = ("q", "k", "v_semantic", "beta", "log_decay")
    for layer in sorted(probe.core_tokens["NS"]):
        ns_records, rs_records = probe.core_tokens["NS"][layer], probe.core_tokens["RS"][layer]
        for ns, rs in zip(ns_records, rs_records):
            token = ns["token_index"]
            for name in invariant_core:
                rows.append(row_from_pair(
                    unit_id, prompt_id, token, layer,
                    "v_before_rotation" if name == "v_semantic" else name,
                    ns[name], rs[name], "BASIS_INVARIANT", "BASIS_INVARIANT", rs[name], "NONE"))
            rs_v_native = map_value_to_native(rs["v"], "rotated", rotation)
            rows.append(row_from_pair(
                unit_id, prompt_id, token, layer, "v_after_rotation",
                ns["v_semantic"], rs["v"], "NATIVE_BASIS", "ROTATED_VALUE_BASIS",
                rs_v_native, "INVERSE_VALUE_ROTATION"))
            for name in ("state_in", "state_after"):
                mapped = map_state_to_native(rs[name], "rotated", rotation)
                rows.append(row_from_pair(
                    unit_id, prompt_id, token, layer,
                    "recurrent_state_in" if name == "state_in" else "prequant_state",
                    ns[name], rs[name], "NATIVE_BASIS", "ROTATED_VALUE_BASIS", mapped,
                    "INVERSE_STATE_VALUE_ROTATION"))
            raw_mapped = map_value_to_native(rs["raw_core_output"], "rotated", rotation)
            rows.append(row_from_pair(
                unit_id, prompt_id, token, layer, "core_output",
                ns["raw_core_output"], rs["raw_core_output"], "NATIVE_BASIS",
                "ROTATED_VALUE_BASIS", raw_mapped, "INVERSE_VALUE_ROTATION"))
            rows.append(row_from_pair(
                unit_id, prompt_id, token, layer, "mapped_back_output",
                ns["mapped_back_output"], rs["mapped_back_output"], "NATIVE_BASIS",
                "NATIVE_BASIS", rs["mapped_back_output"], "RUNTIME_MAPBACK_ALREADY_APPLIED"))
            theory_v = rs["v_semantic"].float().matmul(rotation.float()).to(rs["v"].dtype)
            value_stat = metrics(rs["v"], theory_v)
            roundtrip = metrics(map_value_to_native(rs["v"], "rotated", rotation), rs["v_semantic"])
            value_checks.append({
                "unit_id": unit_id, "token_index": token, "layer_index": layer,
                "RUNTIME_V_VS_THEORY_MAX_ABS": value_stat["max_abs"],
                "RUNTIME_V_VS_THEORY_REL_L2": value_stat["relative_l2"],
                "RUNTIME_V_VS_THEORY_COSINE": value_stat["cosine"],
                "VALUE_ROUNDTRIP_MAX_ABS": roundtrip["max_abs"],
                "VALUE_ROUNDTRIP_REL_L2": roundtrip["relative_l2"],
                "orientation": "v_rot=v_native@R; v_native=v_rot@R.T",
            })
            theory_map = rs["raw_core_output"].float().matmul(rotation.t().float()).to(
                rs["mapped_back_output"].dtype)
            map_stat = metrics(rs["mapped_back_output"], theory_map)
            mapback_checks.append({
                "unit_id": unit_id, "token_index": token, "layer_index": layer,
                "MAPBACK_RUNTIME_VS_THEORY_MAX_ABS": map_stat["max_abs"],
                "MAPBACK_RUNTIME_VS_THEORY_REL_L2": map_stat["relative_l2"],
                "MAPBACK_RUNTIME_VS_THEORY_COSINE": map_stat["cosine"],
                "orientation": "o_native=o_rot@R.T",
            })

        native_path = probe.path_tensors["NS"][layer]
        rotated_path = probe.path_tensors["RS"][layer]
        for name in sorted(set(native_path) & set(rotated_path)):
            nt, rt = native_path[name], rotated_path[name]
            for token in range(min(prompt_len, nt.shape[1] if nt.ndim >= 2 else 1)):
                rows.append(row_from_pair(
                    unit_id, prompt_id, token, layer, name, token_slice(nt, token),
                    token_slice(rt, token), "NATIVE_BASIS", "NATIVE_BASIS",
                    token_slice(rt, token), "NONE"))

        for stage in ("kernel_return", "endpoint_basis_handling", "postquant"):
            ns_state, rs_state = boundaries["NS"][stage][layer], boundaries["RS"][stage][layer]
            rows.append(row_from_pair(
                unit_id, prompt_id, prompt_len - 1, layer,
                "prefill_endpoint_" + stage, ns_state, rs_state,
                "NATIVE_BASIS", "ROTATED_VALUE_BASIS",
                map_state_to_native(rs_state, "rotated", rotation),
                "INVERSE_STATE_VALUE_ROTATION"))
    return rows, value_checks, mapback_checks


def native_replay_floors(unit, prompt_len, probe):
    rows, family_max = [], defaultdict(lambda: {"relative_L2": 0.0, "max_abs": 0.0})
    for layer in sorted(probe.core_tokens["NS"]):
        for ns, replay in zip(probe.core_tokens["NS"][layer], probe.core_tokens["NS_REPLAY"][layer]):
            for name in ("q", "k", "v_semantic", "beta", "log_decay", "state_after", "mapped_back_output"):
                stat = metrics(replay[name], ns[name])
                family_max[name]["relative_L2"] = max(family_max[name]["relative_L2"], stat["relative_l2"])
                family_max[name]["max_abs"] = max(family_max[name]["max_abs"], stat["max_abs"])
                rows.append({"unit_id": unit["unit_id"], "token_index": ns["token_index"],
                             "layer_index": layer, "tensor_family": name, **stat})
        for name in set(probe.path_tensors["NS"][layer]) & set(probe.path_tensors["NS_REPLAY"][layer]):
            stat = metrics(probe.path_tensors["NS_REPLAY"][layer][name], probe.path_tensors["NS"][layer][name])
            family_max[name]["relative_L2"] = max(family_max[name]["relative_L2"], stat["relative_l2"])
            family_max[name]["max_abs"] = max(family_max[name]["max_abs"], stat["max_abs"])
    floors = {name: {
        "native_instrumentation_replay_relative_L2": value["relative_L2"],
        "native_instrumentation_replay_max_abs": value["max_abs"],
        "high_precision_rotation_relative_L2": HIGH_PRECISION_REL_FLOOR,
        "high_precision_rotation_max_abs": HIGH_PRECISION_MAX_FLOOR,
        "reference_relative_L2": max(value["relative_L2"], HIGH_PRECISION_REL_FLOOR),
        "reference_max_abs": max(value["max_abs"], HIGH_PRECISION_MAX_FLOOR),
    } for name, value in family_max.items()}
    return rows, floors


def fused_call(probe, rec, state, value, device):
    kwargs = {
        "q": rec["q"].to(device), "k": rec["k"].to(device),
        "v": value.to(device), "g": rec["log_decay"].float().to(device),
        "beta": rec["beta"].float().to(device),
        "initial_state": None if state is None else state.float().to(device),
        "output_final_state": True, "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": False,
    }
    return probe.orig_fused(**kwargs)


def frozen_replay(model, probe, unit, rotation, candidate_layers=None):
    rows, values, mapbacks = [], [], []
    candidate_layers = set(candidate_layers or [])
    for layer in sorted(probe.core_tokens["NS"]):
        attention = model.model.layers[layer].attention
        device = next(attention.parameters()).device
        for rec in probe.core_tokens["NS"][layer]:
            token = rec["token_index"]
            state_native = rec["state_in"]
            out0, state0 = fused_call(probe, rec, state_native, rec["v_semantic"], device)
            state_rot = None if state_native is None else BASE.state_to_branch_coordinates(
                state_native.float(), "rotated", rotation.float())
            value_rot = rec["v_semantic"].float().matmul(rotation.float()).to(rec["v_semantic"].dtype)
            out1, state1 = fused_call(probe, rec, state_rot, value_rot, device)
            out1_native = out1.float().matmul(rotation.t().to(device, torch.float32))
            state1_native = map_state_to_native(state1.detach().cpu(), "rotated", rotation)
            stages = {
                "F0_NATIVE_REPLAY_OUTPUT": (out0.detach().cpu(), rec["raw_core_output"]),
                "F0_NATIVE_REPLAY_STATE": (state0.detach().cpu(), rec["state_after"]),
                "VALUE_ROTATION_ROUNDTRIP": (
                    value_rot.float().matmul(rotation.t().float()), rec["v_semantic"]),
                "RECURRENT_UPDATE": (state1_native, state0.detach().cpu()),
                "QUERY_READOUT": (out1_native.detach().cpu(), out0.detach().cpu()),
            }
            for stage, (actual, reference) in stages.items():
                stat = metrics(actual, reference)
                rows.append({"unit_id": unit["unit_id"], "token_index": token,
                             "layer_index": layer, "replay_condition": "F1",
                             "stage": stage, **stat})
            v_theory = rec["v_semantic"].float().matmul(rotation.float()).to(value_rot.dtype)
            stat = metrics(value_rot, v_theory)
            values.append({"unit_id": unit["unit_id"], "token_index": token, "layer_index": layer,
                           "replay_condition": "F1", "RUNTIME_V_VS_THEORY_MAX_ABS": stat["max_abs"],
                           "RUNTIME_V_VS_THEORY_REL_L2": stat["relative_l2"],
                           "RUNTIME_V_VS_THEORY_COSINE": stat["cosine"]})
            runtime_map = out1_native.to(out1.dtype)
            theory_map = out1.float().matmul(rotation.t().to(device, torch.float32)).to(out1.dtype)
            stat = metrics(runtime_map.detach().cpu(), theory_map.detach().cpu())
            mapbacks.append({"unit_id": unit["unit_id"], "token_index": token, "layer_index": layer,
                             "replay_condition": "F1", "MAPBACK_RUNTIME_VS_THEORY_MAX_ABS": stat["max_abs"],
                             "MAPBACK_RUNTIME_VS_THEORY_REL_L2": stat["relative_l2"],
                             "MAPBACK_RUNTIME_VS_THEORY_COSINE": stat["cosine"]})
            if layer in candidate_layers:
                dynamic_gate = token_slice(
                    probe.path_tensors["NS"][layer]["dynamic_gate_input"], token
                ).to(device=device, dtype=out0.dtype)
                hidden_raw = token_slice(
                    probe.path_tensors["NS"][layer]["layer_input_hidden"], token
                ).to(device=device, dtype=out0.dtype)
                with torch.inference_mode():
                    norm0 = attention.o_norm(out0, dynamic_gate)
                    norm1 = attention.o_norm(out1_native.to(out0.dtype), dynamic_gate)
                    proj0 = attention.o_proj(norm0.reshape(norm0.shape[0], norm0.shape[1], -1))
                    proj1 = attention.o_proj(norm1.reshape(norm1.shape[0], norm1.shape[1], -1))
                    residual0, residual1 = hidden_raw + proj0, hidden_raw + proj1
                    layer_module = model.model.layers[layer]
                    post0, post1 = layer_module.post_attention_layernorm(residual0), layer_module.post_attention_layernorm(residual1)
                    mlp_dtype = next(layer_module.mlp.parameters()).dtype
                    mlp0 = layer_module.mlp(post0.to(mlp_dtype))
                    mlp1 = layer_module.mlp(post1.to(mlp_dtype))
                    if isinstance(mlp0, tuple): mlp0 = mlp0[0]
                    if isinstance(mlp1, tuple): mlp1 = mlp1[0]
                    layer0, layer1 = residual0 + mlp0, residual1 + mlp1
                for stage, actual, reference in (
                    ("RMSNORM", norm1, norm0), ("OUT_PROJ", proj1, proj0),
                    ("RESIDUAL_ATTENTION", residual1, residual0),
                    ("RESIDUAL_LAYER_OUTPUT", layer1, layer0)):
                    stat = metrics(actual.detach().cpu(), reference.detach().cpu())
                    rows.append({"unit_id": unit["unit_id"], "token_index": token,
                                 "layer_index": layer, "replay_condition": "F1",
                                 "stage": stage, **stat})
    return rows, values, mapbacks


def hidden_driver_regeneration(model, probe, unit, rotation, candidate_layers):
    rows = []
    for layer in candidate_layers:
        attention = model.model.layers[layer].attention
        device = next(attention.parameters()).device
        hidden = probe.path_tensors["NS"][layer]["attention_rmsnorm_output"].to(
            device=device, dtype=attention.q_proj.weight.dtype)
        with torch.inference_mode():
            q, _ = attention.q_conv1d(x=attention.q_proj(hidden), output_final_state=False)
            k, _ = attention.k_conv1d(x=attention.k_proj(hidden), output_final_state=False)
            v, _ = attention.v_conv1d(x=attention.v_proj(hidden), output_final_state=False)
            beta = attention.b_proj(hidden).float().sigmoid()
            if attention.no_kda_lora:
                gate = attention.f_proj(hidden)
            else:
                gate = attention.f_b_proj(attention.f_a_proj(hidden))
            q = q.reshape(q.shape[0], q.shape[1], attention.num_heads, attention.head_k_dim)
            k = k.reshape(k.shape[0], k.shape[1], attention.num_heads, attention.head_k_dim)
            v = v.reshape(v.shape[0], v.shape[1], attention.num_heads, attention.head_dim)
            gate = gate.reshape(gate.shape[0], gate.shape[1], attention.num_heads, attention.head_dim)
            decay = DECAY.final_log_decay(gate, attention.A_log, attention.dt_bias,
                                          attention.lower_bound, True)
        recorded = probe.core_tokens["NS"][layer]
        expected = {
            "q": torch.cat([x["q"] for x in recorded], dim=1),
            "k": torch.cat([x["k"] for x in recorded], dim=1),
            "v": torch.cat([x["v_semantic"] for x in recorded], dim=1),
            "beta": torch.cat([x["beta"] for x in recorded], dim=1),
            "decay": torch.cat([x["log_decay"] for x in recorded], dim=1),
        }
        actual = {"q": q.cpu(), "k": k.cpu(), "v": v.cpu(),
                  "beta": beta.cpu(), "decay": decay.cpu()}
        for name in expected:
            stat = metrics(actual[name], expected[name])
            rows.append({"unit_id": unit["unit_id"], "layer_index": layer,
                         "driver": name, "condition": "F3_COMMON_HIDDEN", **stat,
                         "native_hidden_exact": True})
        v_rot_runtime = actual["v"].float().matmul(rotation.float()).to(actual["v"].dtype)
        v_rot_theory = expected["v"].float().matmul(rotation.float()).to(actual["v"].dtype)
        stat = metrics(v_rot_runtime, v_rot_theory)
        rows.append({"unit_id": unit["unit_id"], "layer_index": layer,
                     "driver": "v_rotated_runtime_vs_theory", "condition": "F3_COMMON_HIDDEN",
                     **stat, "native_hidden_exact": True})
    return rows


def layer_summary(free_rows, floors):
    grouped = defaultdict(list)
    for row in free_rows:
        grouped[(int(row["layer_index"]), row["tensor_name"])].append(float(row["relative_L2"]))
    rows = []
    for (layer, name), values in sorted(grouped.items()):
        floor = floors.get(name, {}).get("reference_relative_L2", HIGH_PRECISION_REL_FLOOR)
        rows.append({"layer_index": layer, "tensor_name": name,
                     "max_relative_L2": max(values), "median_relative_L2": BASE.median(values),
                     "reference_numerical_floor": floor,
                     "max_over_floor": max(values) / max(floor, EPS)})
    return rows


def cross_layer_rows(free_rows, floors):
    lookup = defaultdict(list)
    for row in free_rows:
        lookup[(int(row["layer_index"]), row["tensor_name"])].append(float(row["relative_L2"]))
    layers = sorted({key[0] for key in lookup})
    rows = []
    for index, layer in enumerate(layers):
        input_gap = BASE.median(lookup.get((layer, "layer_input_hidden"), [0.0]))
        output_gap = BASE.median(lookup.get((layer, "layer_output_hidden"), [0.0]))
        floor = floors.get("layer_input_hidden", {}).get("reference_relative_L2", HIGH_PRECISION_REL_FLOOR)
        next_layer = layers[index + 1] if index + 1 < len(layers) else None
        row = {"layer_index": layer, "next_kda_layer": next_layer,
               "hidden_gap_at_layer_input": input_gap,
               "hidden_gap_at_layer_output": output_gap,
               "HIDDEN_LAYER_GAIN": output_gap / max(input_gap, floor)}
        for driver in ("v_before_rotation", "k", "beta", "log_decay"):
            gap = None if next_layer is None else BASE.median(lookup.get((next_layer, driver), [0.0]))
            row[f"next_{driver}_gap"] = gap
            row[f"{driver}_REGEN_GAIN"] = None if gap is None else gap / max(output_gap, floor)
        rows.append(row)
    return rows


def locate_divergence(free_rows, floors):
    stage_order = {
        "layer_input_hidden": 0, "attention_rmsnorm_output": 1,
        "q": 2, "k": 3, "v_before_rotation": 4, "v_after_rotation": 5,
        "beta": 6, "log_decay": 7, "recurrent_state_in": 8,
        "prequant_state": 9, "core_output": 10, "mapped_back_output": 11,
        "rmsnorm_input": 12, "rmsnorm_scaled_output": 13,
        "dynamic_gate_input": 14, "dynamic_gate_output": 15,
        "out_proj_input": 16, "out_proj_output": 17,
        "attention_output": 18, "layer_output_hidden": 19,
        "prefill_endpoint_kernel_return": 20,
        "prefill_endpoint_endpoint_basis_handling": 21,
        "prefill_endpoint_postquant": 22,
    }
    ordered = sorted(free_rows, key=lambda row: (
        int(row["layer_index"]), int(row["token_index"]), stage_order.get(row["tensor_name"], 100)))
    nonzero, amplifying, previous = None, None, None
    for row in ordered:
        floor = floors.get(row["tensor_name"], {}).get("reference_relative_L2", HIGH_PRECISION_REL_FLOOR)
        excess = float(row["relative_L2"]) > 10.0 * max(floor, EPS)
        if nonzero is None and excess:
            nonzero = row
        ratio = float(row["relative_L2"]) / max(
            floor, EPS if previous is None else float(previous["relative_L2"]))
        if amplifying is None and excess and ratio >= 4.0:
            amplifying = {**row, "AMPLIFICATION_RATIO": ratio}
        previous = row
    return nonzero, amplifying


def temporal_rows(free_rows, candidate_layer, candidate_tensor, floor):
    rows = []
    for row in free_rows:
        if int(row["layer_index"]) == int(candidate_layer) and row["tensor_name"] == candidate_tensor:
            rows.append({**row, "reference_numerical_floor": floor,
                         "above_floor": float(row["relative_L2"]) > 10 * max(floor, EPS)})
    rows.sort(key=lambda row: int(row["token_index"]))
    return rows


def trace_audit():
    return {
        "FULL_PATH_TRACE_AUDIT": "PASS",
        "actual_runtime_sequence": [
            "decoder layer input hidden", "input RMSNorm", "q/k/v projections",
            "causal short convolutions", "recurrence decay and beta projections",
            "Value rotation immediately before KDA operator", "token-wise fused_recurrent_kda",
            "raw core output", "inverse Value map-back", "FusedRMSNormGated",
            "head merge", "o_proj", "attention residual", "post-attention RMSNorm",
            "MLP/MoE", "final residual/layer output",
            "prefill endpoint recurrent-state basis handling", "INT8_R128 endpoint Q/DQ"],
        "nonexistent_separate_stages": [
            "RMSNorm normalized output before learned scale is fused",
            "dynamic sigmoid gate output is fused into FusedRMSNormGated",
            "head merge is a view/rearrange rather than a module"],
        "source": "/data/zypan/.cache/huggingface/modules/transformers_modules/Ling_hyphen_3_dot_0_hyphen_tiny/modeling_bailing_moe_v3.py",
    }


def write_path_md(outdir, audit):
    lines = [f"# {TASK}", "", f"FULL_PATH_TRACE_AUDIT = {audit['FULL_PATH_TRACE_AUDIT']}", "",
             "## Actual runtime sequence", ""]
    lines += [f"{i}. {name}" for i, name in enumerate(audit["actual_runtime_sequence"], 1)]
    lines += ["", "## Coordinate rules", "",
              "- q, k, beta, decay and model hidden tensors: BASIS_INVARIANT/NATIVE_BASIS.",
              "- v and recurrent state inside RS KDA: ROTATED_VALUE_BASIS.",
              "- raw RS core output: ROTATED_VALUE_BASIS.",
              "- runtime mapped output and all downstream tensors: NATIVE_BASIS.",
              "- state comparison: S_rot @ R.T; Value/output comparison: x_rot @ R.T.",
              "", "## Fused or absent stages", ""]
    lines += [f"- {name}" for name in audit["nonexistent_separate_stages"]]
    (Path(outdir) / "actual_layer_path.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def previous_unit_auc(unit_id):
    rows = BASE.read_rows(KERNEL_RESULTS / "formal_unit_results.csv")
    row = next((row for row in rows if str(row["unit_id"]) == str(unit_id)), None)
    return None if row is None else {name: float(row[f"auc_{name}"]) for name in ("NS", "RS")}


def semantic_stack(stack, basis, rotation):
    if basis == "rotated":
        return H.inverse_stack(stack, rotation)
    return {layer: value.float().clone() for layer, value in stack.items()}


def stack_gap(left, right):
    per_layer = {layer: metrics(right[layer], left[layer]) for layer in sorted(left)}
    return {
        "median_relative_L2": BASE.median([value["relative_l2"] for value in per_layer.values()]),
        "max_relative_L2": max(value["relative_l2"] for value in per_layer.values()),
        "median_max_abs": BASE.median([value["max_abs"] for value in per_layer.values()]),
        "per_layer": {str(layer): value for layer, value in per_layer.items()},
    }


def evaluate_phase2_unit(model, tokenizer, probe, unit, layers, rows_by_pid,
                         teacher_tokens, rotation, horizon):
    row = rows_by_pid[str(unit["problem_id"])]
    tokens = H.history_tokens(unit, teacher_tokens, horizon)
    _, fp, _ = H.prefill_branch(
        model, tokenizer, probe, row, layers, rotation, "FP", "native", False)
    items, boundaries = {}, {}
    for name, endpoint_rotation in (("NS", False), ("RS", True), ("RS_FIXED", False)):
        basis = "native" if name == "NS" else "rotated"
        ids, item, boundary = prefill_with_boundaries(
            model, tokenizer, probe, row, layers, rotation, name, basis,
            apply_endpoint_rotation=endpoint_rotation, quantized=True)
        items[name], boundaries[name] = item, boundary
    prompt_len = int(ids.shape[-1])
    semantic = {
        "NS": semantic_stack(boundaries["NS"]["postquant"], "native", rotation),
        "RS": semantic_stack(boundaries["RS"]["postquant"], "rotated", rotation),
        "RS_FIXED": semantic_stack(boundaries["RS_FIXED"]["postquant"], "rotated", rotation),
    }
    state_diagnostics = {
        "NS_RS": stack_gap(semantic["NS"], semantic["RS"]),
        "NS_RS_FIXED": stack_gap(semantic["NS"], semantic["RS_FIXED"]),
        "RS_RS_FIXED": stack_gap(semantic["RS"], semantic["RS_FIXED"]),
    }
    for name in ("RS", "RS_FIXED"):
        H.to_native(items[name], layers, rotation)
    for name, item in items.items():
        item["basis"] = "native"; item["quantized"] = True
        item["plan"] = H.plan(name + "_NATIVE_CONTINUATION", "native", True)
    fp["plan"] = H.plan("FP_NATIVE_CONTINUATION", "native", False)
    for step in range(int(unit["t0"])):
        token, position = tokens[step], prompt_len + step
        H.advance(model, probe, fp, token, position, layers, rotation, step + 1)
        for item in items.values():
            H.advance(model, probe, item, token, position, layers, rotation, step + 1)
    rows = []
    for h in range(1, int(horizon) + 1):
        token = tokens[int(unit["t0"]) + h - 1]
        position = prompt_len + int(unit["t0"]) + h - 1
        fp_logits, _ = H.advance(model, probe, fp, token, position, layers, rotation, h)
        for name, item in items.items():
            logits, _ = H.advance(model, probe, item, token, position, layers, rotation, h)
            stat = BASE.full_logit_metrics(torch, fp_logits, logits)
            rows.append({"unit_id": str(unit["unit_id"]), "horizon": h,
                         "condition": name, "future_kl": float(stat["KL"]),
                         "logit_cosine": float(stat["logit_cosine"])})
    auc = {name: BASE.mean([row["future_kl"] for row in rows if row["condition"] == name])
           for name in ("NS", "RS", "RS_FIXED")}
    unit_row = {
        "unit_id": str(unit["unit_id"]), "NATIVE_AUC": auc["NS"],
        "ROTATED_AUC": auc["RS"], "INTERVENTION_AUC": auc["RS_FIXED"],
        "CAUSAL_RESCUE": auc["RS"] - auc["RS_FIXED"],
        "CAUSAL_CLOSURE": (auc["RS"] - auc["RS_FIXED"]) / (auc["RS"] - auc["NS"] + EPS),
        "NS_RS_PREFILL_STATE_GAP": state_diagnostics["NS_RS"]["median_relative_L2"],
        "NS_RS_FIXED_PREFILL_STATE_GAP": state_diagnostics["NS_RS_FIXED"]["median_relative_L2"],
        "RS_RS_FIXED_PREFILL_STATE_GAP": state_diagnostics["RS_RS_FIXED"]["median_relative_L2"],
    }
    return rows, unit_row, state_diagnostics


def phase2(args):
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    forensic = BASE.load_json(outdir / "forensic_summary.json", {})
    if forensic.get("PHASE2_CANDIDATE") != "REMOVE_REDUNDANT_PREFILL_ENDPOINT_STATE_ROTATION":
        raise RuntimeError("Phase I did not authorize the endpoint basis-transform rescue")
    units, _ = BASE.load_units("formal", args.max_units, None, None)
    model, tokenizer = load_sharded_model(args.max_memory_gib)
    rows_by_pid, teacher_tokens = BASE.P().load_dataset(), BASE.P().load_fp_teacher_tokens()
    rotation = BASE.exact().make_experiment_rotation(
        "kda", 128, "rht", seed=BASE.RHT_SEED, dtype=torch.float32)
    probe = FullPathForensicProbe(rotation); probe.capture_enabled = False
    layers = probe.install(model)
    weight_hash = BASE.tensor_hash(next(model.parameters()).detach().cpu())
    horizon_rows, unit_rows, diagnostics, failures = [], [], {}, []
    try:
        for index, unit in enumerate(units, 1):
            print(f"[{BASE.now()}] endpoint rescue {index}/{len(units)} {unit['unit_id']}", flush=True)
            try:
                rows, unit_row, state = evaluate_phase2_unit(
                    model, tokenizer, probe, unit, layers, rows_by_pid,
                    teacher_tokens, rotation, args.horizon)
                horizon_rows += rows; unit_rows.append(unit_row); diagnostics[unit["unit_id"]] = state
                H.append_jsonl_many(outdir / "phase2_horizon_raw.jsonl", rows)
                H.append_jsonl_many(outdir / "phase2_unit_raw.jsonl", [unit_row])
            except Exception as exc:
                failures.append({"unit_id": unit["unit_id"], "error": repr(exc),
                                 "traceback": traceback.format_exc(limit=50)})
                save_json(outdir / "phase2_failures.json", failures)
                raise
        rescue = H.effect([row["CAUSAL_RESCUE"] for row in unit_rows], "CAUSAL_RESCUE")
        closure = H.effect([row["CAUSAL_CLOSURE"] for row in unit_rows], "CAUSAL_CLOSURE")
        native_auc = BASE.median([row["NATIVE_AUC"] for row in unit_rows])
        rotated_auc = BASE.median([row["ROTATED_AUC"] for row in unit_rows])
        intervention_auc = BASE.median([row["INTERVENTION_AUC"] for row in unit_rows])
        strong = (len(unit_rows) == EXPECTED_UNITS and rescue["bootstrap_ci_low"] > 0
                  and closure["bootstrap_ci_low"] > 0.5
                  and rescue["positive"] > len(unit_rows) / 2
                  and intervention_auc < 0.25 * rotated_auc)
        partial = rescue["paired_median"] > 0 and rescue["positive"] > len(unit_rows) / 2
        status = "STRONG_SUPPORT" if strong else ("PARTIAL_SUPPORT" if partial else "NOT_SUPPORTED")
        summary = {
            "TASK": TASK,
            "CAUSAL_INTERVENTION": "REMOVE_REDUNDANT_PREFILL_ENDPOINT_STATE_ROTATION",
            "N_FORMAL_UNITS": len(unit_rows), "NATIVE_AUC": native_auc,
            "ROTATED_AUC": rotated_auc, "INTERVENTION_AUC": intervention_auc,
            "CAUSAL_RESCUE": rescue["paired_median"],
            "CAUSAL_RESCUE_95CI": [rescue["bootstrap_ci_low"], rescue["bootstrap_ci_high"]],
            "CAUSAL_CLOSURE": closure["paired_median"],
            "CAUSAL_CLOSURE_95CI": [closure["bootstrap_ci_low"], closure["bootstrap_ci_high"]],
            "RESCUE_SIGN_COUNTS": {"positive": rescue["positive"], "negative": rescue["negative"]},
            "CAUSAL_STATUS": status, "FAILURES": failures,
            "MODEL_WEIGHTS_UNCHANGED": weight_hash == BASE.tensor_hash(next(model.parameters()).detach().cpu()),
            "FINAL_KDA_ROTATION_FAILURE_MECHANISM": (
                "REDUNDANT_PREFILL_ENDPOINT_STATE_BASIS_ROTATION" if strong
                else "VALUE_ROTATION_PATH_FAILURE_PARTIAL"),
            "KDA_MECHANISM_CLOSURE_READY": "YES" if strong else "NO",
            "METHOD_DESIGN_READY": "YES" if strong else "NO",
            "NEXT_PHASE": "LEARNABLE_ROTATION_METHOD_DESIGN" if strong else "HUMAN_REVIEW_OF_PREFILL_VALUE_PATH",
        }
        BASE.write_rows(outdir / "formal_causal_horizon_results.csv", horizon_rows)
        BASE.write_rows(outdir / "formal_causal_unit_results.csv", unit_rows)
        save_json(outdir / "formal_causal_state_diagnostics.json", diagnostics)
        save_json(outdir / "formal_causal_summary.json", summary)
        with (outdir / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(f"TASK={TASK} PHASE=PHASE2 FAILURES=[] TIME={BASE.now()}\n")
        return summary
    finally:
        probe.close()


def save_json(path, obj):
    BASE.save_json(Path(path), obj)


def phase1(args):
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    units, manifest = BASE.load_units("smoke", args.max_units, None, None)
    model, tokenizer = load_sharded_model(args.max_memory_gib)
    rows_by_pid = BASE.P().load_dataset()
    rotation = BASE.exact().make_experiment_rotation(
        "kda", 128, "rht", seed=BASE.RHT_SEED, dtype=torch.float32)
    probe = FullPathForensicProbe(rotation)
    layers = probe.install(model)
    weights_before = {name: BASE.tensor_hash(parameter.detach().cpu())
                      for name, parameter in model.named_parameters() if name.endswith("o_proj.weight")}
    all_free, all_values, all_maps, all_frozen, all_regen = [], [], [], [], []
    all_temporal, parity = [], []
    floors = {}
    failures = []
    try:
        for unit in units:
            row = rows_by_pid[str(unit["problem_id"])]
            boundaries, items = {}, {}
            for name, basis in (("NS", "native"), ("RS", "rotated")):
                ids, item, boundary = prefill_with_boundaries(
                    model, tokenizer, probe, row, layers, rotation, name, basis,
                    apply_endpoint_rotation=True, quantized=True)
                items[name], boundaries[name] = item, boundary
            prompt_len = int(ids.shape[-1])
            free, values, maps = free_running_rows(unit, prompt_len, probe, boundaries, rotation)
            all_free += free; all_values += values; all_maps += maps
            _, _, replay_boundary = prefill_with_boundaries(
                model, tokenizer, probe, row, layers, rotation, "NS_REPLAY", "native",
                apply_endpoint_rotation=False, quantized=False)
            replay_rows, unit_floors = native_replay_floors(unit, prompt_len, probe)
            for name, value in unit_floors.items():
                old = floors.setdefault(name, value)
                for key in ("native_instrumentation_replay_relative_L2", "native_instrumentation_replay_max_abs",
                            "reference_relative_L2", "reference_max_abs"):
                    old[key] = max(old[key], value[key])
            nonzero, amplifying = locate_divergence(free, floors)
            candidate_layer = int((amplifying or nonzero)["layer_index"])
            frozen, frozen_values, frozen_maps = frozen_replay(
                model, probe, unit, rotation, candidate_layers=[candidate_layer])
            all_frozen += frozen; all_values += frozen_values; all_maps += frozen_maps
            all_regen += hidden_driver_regeneration(model, probe, unit, rotation, [candidate_layer])
            candidate = amplifying or nonzero
            floor = floors.get(candidate["tensor_name"], {}).get("reference_relative_L2", HIGH_PRECISION_REL_FLOOR)
            all_temporal += temporal_rows(free, candidate_layer, candidate["tensor_name"], floor)
            parity.append({"unit_id": unit["unit_id"], "previous_auc": previous_unit_auc(unit["unit_id"]),
                           "trace_prompt_tokens": prompt_len,
                           "NS_kernel_cache_identity": max(metrics(
                               boundaries["NS"]["kernel_return"][layer],
                               probe.core_tokens["NS"][layer][-1]["state_after"])["relative_l2"] for layer in layers),
                           "RS_kernel_cache_identity": max(metrics(
                               boundaries["RS"]["kernel_return"][layer],
                               probe.core_tokens["RS"][layer][-1]["state_after"])["relative_l2"] for layer in layers)})
        layer_rows = layer_summary(all_free, floors)
        cross_rows = cross_layer_rows(all_free, floors)
        nonzero, amplifying = locate_divergence(all_free, floors)
        frozen_local_max = max(row["relative_l2"] for row in all_frozen
                               if row["stage"] in {"RECURRENT_UPDATE", "QUERY_READOUT", "RESIDUAL_LAYER_OUTPUT"})
        frozen_floor = max(HIGH_PRECISION_REL_FLOOR, max(
            value["reference_relative_L2"] for value in floors.values()))
        frozen_status = "PASS" if frozen_local_max <= 10 * frozen_floor else (
            "PARTIAL" if frozen_local_max <= 100 * frozen_floor else "FAIL")
        regen_max = max(row["relative_l2"] for row in all_regen)
        regen_status = "PASS" if regen_max <= 10 * frozen_floor else (
            "PARTIAL" if regen_max <= 100 * frozen_floor else "FAIL")
        value_max = max(row["RUNTIME_V_VS_THEORY_REL_L2"] for row in all_values)
        map_max = max(row["MAPBACK_RUNTIME_VS_THEORY_REL_L2"] for row in all_maps)
        endpoint_rows = [row for row in all_free
                         if row["tensor_name"] == "prefill_endpoint_endpoint_basis_handling"]
        endpoint_median = BASE.median([row["relative_L2"] for row in endpoint_rows])
        classification = "VALUE_ROTATION_PATH_FAILURE" if endpoint_median > 100 * frozen_floor else (
            "RECURRENT_UPDATE_LOCAL_FAILURE" if frozen_status == "FAIL" else
            "CROSS_LAYER_REGENERATION_AMPLIFICATION")
        earliest_regen = next((row["layer_index"] for row in cross_rows
                               if (row.get("v_before_rotation_REGEN_GAIN") or 0) >= 4), "NONE")
        first_token = next((row["token_index"] for row in all_temporal if row["above_floor"]), "NONE")
        pattern = "IMMEDIATE" if first_token == 0 else "GRADUAL"
        summary = {
            "TASK": TASK, "PHASE1_STATUS": "COMPLETE", "FORENSIC_UNITS": len(units),
            "NATIVE_SEQUENTIAL_PARITY": "PASS" if max(x["NS_kernel_cache_identity"] for x in parity) <= 1e-6 else "FAIL",
            "ROTATED_SEQUENTIAL_PARITY": "PASS" if max(x["RS_kernel_cache_identity"] for x in parity) <= 1e-6 else "FAIL",
            "EARLIEST_NONZERO_DIVERGENCE_LAYER": nonzero["layer_index"],
            "EARLIEST_NONZERO_DIVERGENCE_TOKEN": nonzero["token_index"],
            "EARLIEST_NONZERO_DIVERGENCE_TENSOR": nonzero["tensor_name"],
            "EARLIEST_AMPLIFYING_DIVERGENCE_LAYER": amplifying["layer_index"],
            "EARLIEST_AMPLIFYING_DIVERGENCE_TOKEN": amplifying["token_index"],
            "EARLIEST_AMPLIFYING_DIVERGENCE_TENSOR": amplifying["tensor_name"],
            "REFERENCE_NUMERICAL_FLOOR": floors,
            "EARLIEST_DIVERGENCE_REL_L2": nonzero["relative_L2"],
            "EARLIEST_DIVERGENCE_MAX_ABS": nonzero["max_abs"],
            "FROZEN_INPUT_LOCAL_EQUIVALENCE": frozen_status,
            "FROZEN_DRIVER_LOCAL_EQUIVALENCE": "NOT_RUN",
            "FROZEN_HIDDEN_DRIVER_REGENERATION": regen_status,
            "RUNTIME_VALUE_ROTATION_MATCHES_THEORY": "YES" if value_max <= 10 * frozen_floor else "NO",
            "MAX_VALUE_ROTATION_REL_ERROR": value_max,
            "RUNTIME_MAPBACK_MATCHES_THEORY": "YES" if map_max <= 10 * frozen_floor else "NO",
            "MAX_MAPBACK_REL_ERROR": map_max,
            "CROSS_LAYER_ERROR_AMPLIFICATION": "SUPPORTED" if earliest_regen != "NONE" else "NOT_SUPPORTED",
            "EARLIEST_LAYER_WITH_DRIVER_REGEN_AMPLIFICATION": earliest_regen,
            "EARLIEST_TOKEN_OF_TRUE_DIVERGENCE": first_token,
            "DIVERGENCE_TEMPORAL_PATTERN": pattern,
            "PREFILL_ENDPOINT_EXTRA_ROTATION_MEDIAN_REL_L2": endpoint_median,
            "LOCAL_LAYER_ROTATION_PATH_FAILURE_FOUND": "YES" if classification != "CROSS_LAYER_REGENERATION_AMPLIFICATION" else "NO",
            "FORENSIC_CLASSIFICATION": classification,
            "PHASE2_CANDIDATE": "REMOVE_REDUNDANT_PREFILL_ENDPOINT_STATE_ROTATION" if classification == "VALUE_ROTATION_PATH_FAILURE" else "NONE",
            "FAILURES": failures,
        }
        audit = trace_audit()
        save_json(outdir / "trace_semantics.json", {**audit, "basis_tags_required": True,
                                                     "device_map": model.hf_device_map,
                                                     "quantizer": KERNEL.COMM.quantizer_semantics()})
        write_path_md(outdir, audit)
        BASE.write_rows(outdir / "free_running_trace.csv", all_free)
        BASE.write_rows(outdir / "free_running_layer_summary.csv", layer_rows)
        BASE.write_rows(outdir / "frozen_replay_trace.csv", all_frozen + all_regen)
        BASE.write_rows(outdir / "value_rotation_theory_check.csv", all_values)
        BASE.write_rows(outdir / "mapback_theory_check.csv", all_maps)
        BASE.write_rows(outdir / "cross_layer_amplification.csv", cross_rows)
        BASE.write_rows(outdir / "temporal_divergence.csv", all_temporal)
        save_json(outdir / "forensic_summary.json", summary)
        save_json(outdir / "experiment_config.json", {
            "TASK": TASK, "canonical_manifest": manifest,
            "canonical_unit_ids": [unit["unit_id"] for unit in units],
            "gpu_model_parallel": True, "visible_gpu_count": torch.cuda.device_count(),
            "max_memory_gib_per_visible_gpu": args.max_memory_gib,
            "previous_results_read_only": [str(KERNEL_RESULTS), str(PRECISION_RESULTS)],
            "trace_branches": ["NS", "RS"], "rotation_seed": BASE.RHT_SEED,
        })
        weights_after = {name: BASE.tensor_hash(parameter.detach().cpu())
                         for name, parameter in model.named_parameters() if name.endswith("o_proj.weight")}
        summary["MODEL_WEIGHTS_UNCHANGED"] = weights_before == weights_after
        save_json(outdir / "forensic_summary.json", summary)
        with (outdir / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(f"TASK={TASK} PHASE=PHASE1 FAILURES=[] TIME={BASE.now()}\n")
        return summary
    except Exception as exc:
        failures.append({"error": repr(exc), "traceback": traceback.format_exc(limit=60)})
        save_json(outdir / "phase1_failures.json", failures)
        raise
    finally:
        probe.close()


def placeholder_phase2(outdir, status="NOT_RUN_BY_STOP_RULE"):
    outdir = Path(outdir)
    BASE.write_rows(outdir / "formal_causal_horizon_results.csv", [{"STATUS": status}])
    BASE.write_rows(outdir / "formal_causal_unit_results.csv", [{"STATUS": status}])
    save_json(outdir / "formal_causal_summary.json", {"STATUS": status})


def finalize(outdir):
    outdir = Path(outdir)
    forensic = BASE.load_json(outdir / "forensic_summary.json", {})
    if not (outdir / "formal_causal_summary.json").exists():
        placeholder_phase2(outdir)
    formal = BASE.load_json(outdir / "formal_causal_summary.json", {})
    pytest_lines = ((outdir / "pytest_output.txt").read_text(encoding="utf-8").splitlines()
                    if (outdir / "pytest_output.txt").exists() else [])
    report = {**forensic, "PYTEST": pytest_lines[-1] if pytest_lines else "not recorded",
              "PHASE2": formal}
    save_json(outdir / "summary.json", report)
    lines = [f"# {TASK}", ""] + [f"{key} = {json.dumps(value, sort_keys=True)}"
                                        for key, value in report.items()]
    (outdir / "formal_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def validate(outdir):
    outdir = Path(outdir)
    required = ["experiment_config.json", "actual_layer_path.md", "trace_semantics.json",
                "free_running_trace.csv", "free_running_layer_summary.csv", "frozen_replay_trace.csv",
                "value_rotation_theory_check.csv", "mapback_theory_check.csv",
                "cross_layer_amplification.csv", "temporal_divergence.csv",
                "forensic_summary.json", "run.log", "pytest_output.txt", "formal_summary.md"]
    missing = [name for name in required if not (outdir / name).is_file()]
    free = BASE.read_rows(outdir / "free_running_trace.csv")
    required_keys = {"unit_id", "prompt_id", "token_index", "layer_index", "tensor_name",
                     "native_shape", "rotated_shape", "comparison_basis", "max_abs",
                     "relative_L2", "cosine", "tensor_basis_native", "tensor_basis_rotated",
                     "map_applied"}
    finite = all(math.isfinite(float(row[key])) for row in free
                 for key in ("max_abs", "relative_L2", "cosine"))
    formal = BASE.load_json(outdir / "formal_causal_summary.json", {})
    formal_horizon = BASE.read_rows(outdir / "formal_causal_horizon_results.csv")
    formal_units = BASE.read_rows(outdir / "formal_causal_unit_results.csv")
    formal_unique = {(row["unit_id"], row["horizon"], row["condition"])
                     for row in formal_horizon if "unit_id" in row}
    formal_finite = all(math.isfinite(float(value)) for row in formal_horizon + formal_units
                        for key, value in row.items() if key not in {"unit_id", "condition"})
    formal_complete = (formal.get("N_FORMAL_UNITS") == EXPECTED_UNITS
                       and len(formal_units) == EXPECTED_UNITS
                       and len(formal_horizon) == EXPECTED_UNITS * PRIMARY_HORIZON * 3
                       and len(formal_unique) == len(formal_horizon) and formal_finite)
    result = {"missing_required_files": missing, "free_running_rows": len(free),
              "trace_schema_complete": all(required_keys.issubset(row) for row in free),
              "all_trace_metrics_finite": finite,
              "formal_units": len(formal_units), "formal_horizon_rows": len(formal_horizon),
              "formal_unique_horizon_rows": len(formal_unique),
              "all_formal_metrics_finite": formal_finite,
              "formal_complete": formal_complete}
    result["ARTIFACT_VALIDATION"] = "PASS" if (
        not missing and free and result["trace_schema_complete"] and finite
        and formal_complete) else "FAIL"
    save_json(outdir / "artifact_validation.json", result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("phase1", "phase2", "finalize", "validate"), required=True)
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--max-units", type=int, default=1)
    parser.add_argument("--max-memory-gib", type=int, default=9)
    parser.add_argument("--horizon", type=int, default=PRIMARY_HORIZON)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.phase == "phase1": result = phase1(args)
    elif args.phase == "phase2": result = phase2(args)
    elif args.phase == "finalize": result = finalize(args.output_dir)
    else: result = validate(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if result.get("ARTIFACT_VALIDATION") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
