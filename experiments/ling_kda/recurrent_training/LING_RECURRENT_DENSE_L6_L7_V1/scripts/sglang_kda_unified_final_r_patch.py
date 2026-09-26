#!/usr/bin/env python3
"""Unified materialized-final-R KDA hooks for pinned SGLang v0.5.19.

The patch is injected before the SGLang server starts.  It does not modify the
installed package.  The hook point is the KDA kernel dispatcher: v has already
passed ShortConv+SiLU, and the returned core has not yet reached output norm,
dynamic gating, merge, or o_proj.
"""

import hashlib
import json
import math
import os
from pathlib import Path

import torch


MODE = os.environ.get("AIME26_KDA_MODE", "fp_state")
FINAL_R_MODES = {"fp_state_unified_final_r", "int8_r128_unified_final_r"}
SUPPORTED_MODES = {"fp_state", "int8_r128"} | FINAL_R_MODES
if MODE not in SUPPORTED_MODES:
    raise RuntimeError(f"unsupported AIME26_KDA_MODE={MODE!r}")
QUANTIZE = MODE.startswith("int8_r128")
ROTATE = MODE not in {"fp_state", "int8_r128"}
AUDIT_PATH = os.environ.get("AIME26_KDA_AUDIT_JSONL")
FORCE_PATH = os.environ.get("AIME26_KDA_FORCE_JSON")
FINAL_ROTATION_PATH = os.environ.get("AIME26_KDA_FINAL_ROTATION")
EXPECTED_LAYERS = tuple(
    int(value)
    for value in os.environ.get(
        "AIME26_KDA_EXPECTED_LAYERS",
        "0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22",
    ).split(",")
    if value
)
SELECTED_LAYERS = {
    int(value) for value in os.environ.get("AIME26_KDA_DIAGNOSTIC_LAYERS", "0,12,22").split(",") if value
}
EPS = 1e-12
_CURRENT_LAYER = None
_AUDITED = set()
_H_CACHE = {}
_ROTATION_CPU = {}
_ROTATION_DEVICE = {}
_ROTATION_MANIFEST = {}
_FORCE = {"active": False, "mtime_ns": None, "request_id": None, "step": 0, "tokens": [], "dump_dir": None}


def _hadamard(device):
    key = str(device)
    if key not in _H_CACHE:
        matrix = torch.ones((1, 1), dtype=torch.float32, device=device)
        while matrix.shape[0] < 128:
            matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
        _H_CACHE[key] = matrix / math.sqrt(128)
    return _H_CACHE[key]


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value):
    source = value.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(source.numpy().tobytes()).hexdigest()


def _matrix_audit(matrix):
    identity = torch.eye(128, dtype=torch.float32)
    residual = matrix.transpose(0, 1).matmul(matrix) - identity
    sign, logabsdet = torch.linalg.slogdet(matrix)
    return {
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "sha256_fp32_le": _tensor_sha256(matrix),
        "max_abs_rt_r_minus_i": float(residual.abs().max()),
        "frobenius_rt_r_minus_i": float(torch.linalg.vector_norm(residual)),
        "determinant_sign": float(sign),
        "log_abs_determinant": float(logabsdet),
        "finite": bool(torch.isfinite(matrix).all()),
    }


def _initialize_rotation_bank():
    if MODE in {"fp_state", "int8_r128"}:
        return
    if MODE not in FINAL_R_MODES:
        raise AssertionError(MODE)
    if not FINAL_ROTATION_PATH:
        raise RuntimeError(f"{MODE} requires AIME26_KDA_FINAL_ROTATION")
    rotation_path = Path(FINAL_ROTATION_PATH).resolve()
    if not rotation_path.is_file():
        raise RuntimeError(f"missing materialized final-R file: {rotation_path}")
    payload = torch.load(rotation_path, map_location="cpu", weights_only=False)
    checkpoint_layers = tuple(int(value) for value in payload.get("layer_ids", ()))
    if checkpoint_layers != EXPECTED_LAYERS:
        raise RuntimeError(
            f"final-R layer mapping mismatch: expected {EXPECTED_LAYERS}, found {checkpoint_layers}"
        )
    rotations = payload.get("rotations", {})
    layer_manifest = {}
    for layer in EXPECTED_LAYERS:
        if layer not in rotations and str(layer) not in rotations:
            raise RuntimeError(f"final-R payload is missing layer {layer}")
        total = rotations.get(layer, rotations.get(str(layer))).detach().cpu().float().contiguous()
        if tuple(total.shape) != (128, 128):
            raise RuntimeError(f"layer {layer} final-R shape is {tuple(total.shape)}, expected (128, 128)")
        audit = _matrix_audit(total)
        if not audit["finite"] or audit["max_abs_rt_r_minus_i"] > 1.0e-4:
            raise RuntimeError(f"orthogonality gate failed for layer {layer}: {audit}")
        _ROTATION_CPU[layer] = {"total": total}
        layer_manifest[str(layer)] = {
            "representation": "materialized_final_R_single_dense_GEMM",
            "total_R": audit,
        }

    _ROTATION_MANIFEST.update(
        {
            "mode": MODE,
            "final_rotation_path": str(rotation_path),
            "final_rotation_sha256": _sha256_file(rotation_path),
            "condition": payload.get("condition"),
            "source_checkpoint": payload.get("source_checkpoint"),
            "source_checkpoint_sha256": payload.get("source_checkpoint_sha256"),
            "layer_ids": list(EXPECTED_LAYERS),
            "head_sharing": "one 128x128 Value rotation per KDA layer, shared across heads",
            "forward_order": "one FP32 value @ materialized_R_final GEMM; then cast back",
            "inverse_order": "one FP32 core @ materialized_R_final.T GEMM; then cast back",
            "factorized_h_delta_runtime": False,
            "layers": layer_manifest,
        }
    )


def _device_rotation(layer, device):
    layer = int(layer)
    if layer not in _ROTATION_CPU:
        raise RuntimeError(f"no rotation is loaded for SGLang global layer {layer}")
    key = (layer, str(device))
    if key not in _ROTATION_DEVICE:
        _ROTATION_DEVICE[key] = {
            name: value.to(device=device, dtype=torch.float32)
            for name, value in _ROTATION_CPU[layer].items()
        }
    return _ROTATION_DEVICE[key]


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
    layer = int(_CURRENT_LAYER)
    source = value.float()
    if MODE not in FINAL_R_MODES:
        raise AssertionError(MODE)
    rotated = source @ _device_rotation(layer, value.device)["total"]
    return rotated.to(value.dtype)


def _inverse_core(value):
    layer = int(_CURRENT_LAYER)
    source = value.float()
    if MODE not in FINAL_R_MODES:
        raise AssertionError(MODE)
    restored = source @ _device_rotation(layer, value.device)["total"].transpose(0, 1)
    return restored.to(value.dtype)


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
                "event": "kda_value_rotation",
                "mode": MODE,
                "phase": phase,
                "layer": layer,
                "rotation": (
                    _ROTATION_MANIFEST["layers"][str(layer)]["representation"]
                ),
                "matrix_sha256_fp32_le": (
                    _ROTATION_MANIFEST["layers"][str(layer)]["total_R"]["sha256_fp32_le"]
                ),
                "forward_placement": "final v after ShortConv+SiLU, immediately before KDA kernel",
                "inverse_placement": "once after KDA core, before RMSNorm/learned weight/dynamic gate/merge/o_proj",
                "cache_basis": "S @ R retained across prefill and decode",
            }
        )
    return result


def install_patch():
    from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend, KDAKernelDispatcher
    from sglang.srt.layers.sampler import Sampler

    if getattr(KDAKernelDispatcher, "_aime26_patch_installed", False):
        return

    _initialize_rotation_bank()

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
            "rotation_manifest": _ROTATION_MANIFEST or None,
            "pid": os.getpid(),
        }
    )
