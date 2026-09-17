#!/usr/bin/env python3
"""Runtime-only KDA R128/Value-H hooks for pinned SGLang v0.5.19.

The patch is injected before the SGLang server starts.  It does not modify the
installed package.  The hook point is the KDA kernel dispatcher: v has already
passed ShortConv+SiLU, and the returned core has not yet reached output norm,
dynamic gating, merge, or o_proj.
"""

import json
import math
import os
from pathlib import Path

import torch


MODE = os.environ.get("AIME26_KDA_MODE", "fp_state")
QUANTIZE = MODE in {"int8_r128", "int8_r128_value_h"}
ROTATE = MODE in {"fp_state_value_h", "int8_r128_value_h"}
AUDIT_PATH = os.environ.get("AIME26_KDA_AUDIT_JSONL")
FORCE_PATH = os.environ.get("AIME26_KDA_FORCE_JSON")
SELECTED_LAYERS = {
    int(value) for value in os.environ.get("AIME26_KDA_DIAGNOSTIC_LAYERS", "0,12,22").split(",") if value
}
EPS = 1e-12
_CURRENT_LAYER = None
_AUDITED = set()
_H_CACHE = {}
_FORCE = {"active": False, "mtime_ns": None, "request_id": None, "step": 0, "tokens": [], "dump_dir": None}


def _hadamard(device):
    key = str(device)
    if key not in _H_CACHE:
        matrix = torch.ones((1, 1), dtype=torch.float32, device=device)
        while matrix.shape[0] < 128:
            matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
        _H_CACHE[key] = matrix / math.sqrt(128)
    return _H_CACHE[key]


def _append_audit(record):
    if not AUDIT_PATH:
        return
    path = Path(AUDIT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _refresh_force():
    if not FORCE_PATH:
        _FORCE["active"] = False
        return _FORCE
    path = Path(FORCE_PATH)
    if not path.exists():
        _FORCE["active"] = False
        return _FORCE
    stat = path.stat()
    if _FORCE["mtime_ns"] != stat.st_mtime_ns:
        payload = json.loads(path.read_text(encoding="utf-8"))
        request_id = str(payload["request_id"])
        if request_id != _FORCE["request_id"]:
            _FORCE["step"] = 0
        _FORCE.update(
            {
                "active": True,
                "mtime_ns": stat.st_mtime_ns,
                "request_id": request_id,
                "tokens": [int(value) for value in payload["tokens"]],
                "dump_dir": str(payload["dump_dir"]),
            }
        )
        Path(_FORCE["dump_dir"]).mkdir(parents=True, exist_ok=True)
    return _FORCE


def _rotate_v(value):
    # FP32 dense H followed by cast-back is required by the optimized KDA kernels.
    return (value.float() @ _hadamard(value.device)).to(value.dtype)


def _inverse_core(value):
    return (value.float() @ _hadamard(value.device).transpose(0, 1)).to(value.dtype)


def _map_core(result, function):
    if isinstance(result, tuple):
        return (function(result[0]),) + result[1:]
    return function(result)


def _dump_state(active_state, phase, layer):
    force = _refresh_force()
    if not force["active"] or layer not in SELECTED_LAYERS or active_state.shape[0] != 1:
        return
    step = int(force["step"])
    path = Path(force["dump_dir"]) / f"state_step{step:04d}_layer{layer:02d}_{phase}.pt"
    torch.save(active_state[0].detach().cpu(), path)


def _dump_basis_state(kind, active_state, layer):
    """Capture the corrected prefill-endpoint V2 state continuity evidence."""
    force = _refresh_force()
    if not ROTATE or not force["active"] or layer not in SELECTED_LAYERS or active_state.shape[0] != 1:
        return
    path = Path(force["dump_dir"]) / f"basis_{kind}_layer{layer:02d}.pt"
    if path.exists():
        return
    torch.save(active_state[0].detach().cpu(), path)
    _append_audit(
        {
            "event": "kda_prefill_endpoint_basis_capture",
            "semantics_version": "CORRECTED_PREFILL_ENDPOINT_V2",
            "kind": kind,
            "layer": layer,
            "request_id": force["request_id"],
            "redundant_prefill_endpoint_rotation": False,
        }
    )


def _backend_active_state(backend, layer):
    layer_cache = backend.req_to_token_pool.mamba2_layer_cache(int(layer.layer_id))
    indices = backend.forward_metadata.mamba_cache_indices.to(torch.int64)
    return layer_cache.temporal.index_select(0, indices)


def _quantize_state(ssm_states, cache_indices, phase, layer):
    indices = cache_indices.to(torch.int64)
    active = ssm_states.index_select(0, indices)
    source = active.detach().float()
    # SGLang stores KDA cache physically as [B,H,V,K], while the canonical
    # experiment convention is [B,H,K,V].  R128 shares one scale across the
    # 128 Value elements for each fixed Key row, hence reduction over physical
    # axis -2 and a physical scale layout [B,H,1,K].
    scale = source.abs().amax(dim=-2, keepdim=True).clamp_min(EPS) / 127.0
    codes = torch.round(source / scale).clamp(-127, 127)
    dequant = (codes * scale).to(active.dtype)
    ssm_states.index_copy_(0, indices, dequant)
    event_key = (phase, layer)
    if event_key not in _AUDITED:
        _AUDITED.add(event_key)
        _append_audit(
            {
                "event": "kda_state_qdq",
                "mode": MODE,
                "phase": phase,
                "layer": layer,
                "state_shape": list(active.shape),
                "scale_shape": list(scale.shape),
                "qrange_observed": [int(codes.min().item()), int(codes.max().item())],
                "grouping": "canonical R128 scale=[B,H,K,1]; SGLang physical scale=[B,H,1,K]",
                "timing": "after KDA state update, before the next token",
            }
        )
        dump_dir = os.environ.get("AIME26_KDA_GATE_C_DUMP_DIR")
        if dump_dir and active.shape[0] > 0:
            path = Path(dump_dir)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "prequant_state": source[0, 0].transpose(-1, -2).cpu(),
                    "scale": scale[0, 0].transpose(-1, -2).cpu(),
                    "qcode": codes[0, 0].transpose(-1, -2).to(torch.int8).cpu(),
                    "dequant_state": dequant[0, 0].transpose(-1, -2).float().cpu(),
                    "canonical_layout": "[K,V]",
                    "runtime_physical_layout": "[V,K]",
                    "layer": layer,
                    "head": 0,
                    "phase": phase,
                },
                path / f"runtime_{phase}_layer{layer:02d}_head00.pt",
            )
    return ssm_states.index_select(0, indices)


def _after_kernel(result, kwargs, phase):
    layer = int(_CURRENT_LAYER)
    ssm_states = kwargs["ssm_states"]
    cache_indices = kwargs["cache_indices"]
    if QUANTIZE:
        active = _quantize_state(ssm_states, cache_indices, phase, layer)
    else:
        active = ssm_states.index_select(0, cache_indices.to(torch.int64))
    _dump_state(active, phase, layer)
    if ROTATE:
        result = _map_core(result, _inverse_core)
    event_key = ("rotation", phase, layer)
    if ROTATE and event_key not in _AUDITED:
        _AUDITED.add(event_key)
        _append_audit(
            {
                "event": "kda_value_hadamard",
                "mode": MODE,
                "phase": phase,
                "layer": layer,
                "rotation": "dense normalized Sylvester H128, FP32 matmul cast back",
                "forward_placement": "final v after ShortConv+SiLU, immediately before KDA kernel",
                "inverse_placement": "once after KDA core, before RMSNorm/learned weight/dynamic gate/merge/o_proj",
                "cache_basis": "S H retained across prefill and decode",
            }
        )
    return result


def install_patch():
    from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend, KDAKernelDispatcher
    from sglang.srt.layers.sampler import Sampler

    if getattr(KDAKernelDispatcher, "_aime26_patch_installed", False):
        return

    original_backend_decode = KDAAttnBackend.forward_decode
    original_backend_extend = KDAAttnBackend.forward_extend
    original_decode = KDAKernelDispatcher.decode
    original_extend = KDAKernelDispatcher.extend
    original_packed = KDAKernelDispatcher.packed_decode
    original_sampler = Sampler.forward

    def backend_decode(self, layer, *args, **kwargs):
        global _CURRENT_LAYER
        _CURRENT_LAYER = int(layer.layer_id)
        _dump_basis_state("first_decode_input", _backend_active_state(self, layer), _CURRENT_LAYER)
        return original_backend_decode(self, layer, *args, **kwargs)

    def backend_extend(self, layer, *args, **kwargs):
        global _CURRENT_LAYER
        _CURRENT_LAYER = int(layer.layer_id)
        result = original_backend_extend(self, layer, *args, **kwargs)
        # The KDA kernel has already written the rotated-basis state to cache.
        # V2 semantics write/retain it directly: no H/H^T endpoint transform.
        _dump_basis_state("prefill_end_cache", _backend_active_state(self, layer), _CURRENT_LAYER)
        return result

    def decode(self, *args, **kwargs):
        if ROTATE:
            kwargs["v"] = _rotate_v(kwargs["v"])
        result = original_decode(self, *args, **kwargs)
        return _after_kernel(result, kwargs, "decode")

    def extend(self, *args, **kwargs):
        if ROTATE:
            kwargs["v"] = _rotate_v(kwargs["v"])
        result = original_extend(self, *args, **kwargs)
        if ROTATE:
            indices = kwargs["cache_indices"].to(torch.int64)
            kernel_state = kwargs["ssm_states"].index_select(0, indices)
            _dump_basis_state("kernel_return", kernel_state, int(_CURRENT_LAYER))
        return _after_kernel(result, kwargs, "prefill")

    def packed_decode(self, *args, **kwargs):
        if ROTATE:
            mixed = kwargs["mixed_qkv"]
            v_dim = int(kwargs["num_v_heads"]) * int(kwargs["head_v_dim"])
            value = mixed[..., -v_dim:].unflatten(-1, (int(kwargs["num_v_heads"]), int(kwargs["head_v_dim"])))
            value = _rotate_v(value).flatten(-2)
            kwargs["mixed_qkv"] = torch.cat((mixed[..., :-v_dim], value), dim=-1)
        result = original_packed(self, *args, **kwargs)
        return _after_kernel(result, kwargs, "decode")

    def sampler_forward(self, logits_output, *args, **kwargs):
        force = _refresh_force()
        step = int(force["step"])
        if force["active"] and step < len(force["tokens"]):
            logits = logits_output.next_token_logits
            if logits is None or logits.shape[0] != 1:
                raise RuntimeError(f"AIME26 forced diagnostic requires batch=1 logits, got {None if logits is None else tuple(logits.shape)}")
            torch.save(logits.detach().float().cpu(), Path(force["dump_dir"]) / f"logits_step{step:04d}.pt")
        result = original_sampler(self, logits_output, *args, **kwargs)
        if force["active"] and step < len(force["tokens"]):
            result.fill_(int(force["tokens"][step]))
            _FORCE["step"] = step + 1
        return result

    KDAAttnBackend.forward_decode = backend_decode
    KDAAttnBackend.forward_extend = backend_extend
    KDAKernelDispatcher.decode = decode
    KDAKernelDispatcher.extend = extend
    KDAKernelDispatcher.packed_decode = packed_decode
    KDAKernelDispatcher._aime26_patch_installed = True
    Sampler.forward = sampler_forward
    _append_audit(
        {
            "event": "patch_install",
            "mode": MODE,
            "quantize": QUANTIZE,
            "rotate": ROTATE,
            "pid": os.getpid(),
        }
    )
