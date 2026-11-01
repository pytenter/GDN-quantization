#!/usr/bin/env python3
"""Fake quantization utilities for Qwen3.5 GDN recurrent state cache."""
from quant_utils import error_record, tensor_stats


STATE_GRANULARITY_ALIASES = {
    "layer": "per-layer",
    "head": "per-head",
    "row": "per-row",
    "column": "per-column",
    "per-layer": "per-layer",
    "per-head": "per-head",
    "per-row": "per-row",
    "per-column": "per-column",
}
VALID_STATE_GRANULARITIES = ("per-layer", "per-head", "per-row", "per-column")


def normalize_granularity(granularity):
    try:
        return STATE_GRANULARITY_ALIASES[granularity]
    except KeyError as exc:
        raise ValueError(
            "unknown granularity %r, expected one of %r"
            % (granularity, tuple(STATE_GRANULARITY_ALIASES))
        ) from exc


def _scale_dims_for_state(state, granularity):
    granularity = normalize_granularity(granularity)
    ndim = len(state.shape)
    if ndim < 3:
        raise ValueError("expected recurrent state with at least 3 dims, got shape %r" % (tuple(state.shape),))
    if granularity == "per-layer":
        return tuple(range(ndim))
    if granularity == "per-head":
        return (-2, -1)
    if granularity == "per-row":
        return (-1,)
    if granularity == "per-column":
        return (-2,)
    raise ValueError("unknown granularity %r, expected one of %r" % (granularity, VALID_STATE_GRANULARITIES))


def logical_scale_shape(state, granularity):
    granularity = normalize_granularity(granularity)
    if granularity == "per-layer":
        return []
    if len(state.shape) < 4:
        return list(state.shape)
    heads, key_dim, value_dim = state.shape[-3], state.shape[-2], state.shape[-1]
    if granularity == "per-head":
        return [heads]
    if granularity == "per-row":
        return [heads, key_dim]
    if granularity == "per-column":
        return [heads, value_dim]
    raise ValueError("unknown granularity %r" % (granularity,))


def fake_quant_state(state, bits, granularity="per-layer", eps=1e-12):
    """Symmetric uniform fake quantization for GDN recurrent state.

    For the runtime state shape [B, H, K, V]:
    - per-layer: one scalar scale per full tensor
    - per-head: one scale per [K, V] matrix, shape [B, H, 1, 1]
    - per-row: one scale per row, shape [B, H, K, 1]
    - per-column: one scale per column, shape [B, H, 1, V]
    """
    import torch

    granularity = normalize_granularity(granularity)
    if bits is None or bits >= 16:
        return state, torch.ones((), device=state.device, dtype=torch.float32)
    if granularity not in VALID_STATE_GRANULARITIES:
        raise ValueError("unknown granularity %r, expected one of %r" % (granularity, VALID_STATE_GRANULARITIES))
    qmax = (2 ** (bits - 1)) - 1
    if granularity == "per-layer":
        scale = state.detach().abs().amax().clamp_min(eps) / qmax
    else:
        reduce_dims = _scale_dims_for_state(state, granularity)
        scale = state.detach().abs().amax(dim=reduce_dims, keepdim=True).clamp_min(eps) / qmax
    q = torch.round(state / scale).clamp(-qmax, qmax)
    return q * scale, scale


def iter_recurrent_state_tensors(cache, layer_types=None):
    if cache is None or not hasattr(cache, "layers"):
        return
    for layer_idx, layer in enumerate(cache.layers):
        layer_type = layer_types[layer_idx] if layer_types and layer_idx < len(layer_types) else "unknown"
        states = getattr(layer, "recurrent_states", None)
        if states is None:
            continue
        iterator = states.items() if isinstance(states, dict) else enumerate(states)
        for state_idx, state in iterator:
            if hasattr(state, "shape"):
                yield layer_idx, layer_type, state_idx, state


def quantize_cache_recurrent_states(cache, bits, layer_types=None, granularity="per-layer"):
    granularity = normalize_granularity(granularity)
    records = []
    if bits is None or bits >= 32:
        for layer_idx, layer_type, state_idx, state in iter_recurrent_state_tensors(cache, layer_types):
            stats = tensor_stats(state)
            rec = {
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "state_idx": state_idx,
                "bits": "FP32",
                "granularity": "none",
                "state": stats,
                "state_norm_l2": stats["norm_l2"],
                "quant_error_l2": 0.0,
                "relative_error": 0.0,
                "scale_shape": [],
                "broadcast_scale_shape": [],
            }
            records.append(rec)
        return records
    for layer_idx, layer_type, state_idx, state in iter_recurrent_state_tensors(cache, layer_types):
        qdq, scale = fake_quant_state(state, bits=bits, granularity=granularity)
        rec = error_record(state, qdq)
        rec.update({
            "layer_idx": layer_idx,
            "layer_type": layer_type,
            "state_idx": state_idx,
            "bits": bits,
            "granularity": granularity,
            "state": tensor_stats(state),
            "scale_shape": logical_scale_shape(state, granularity),
            "broadcast_scale_shape": list(scale.shape),
        })
        state.copy_(qdq.to(state.dtype))
        records.append(rec)
    return records
