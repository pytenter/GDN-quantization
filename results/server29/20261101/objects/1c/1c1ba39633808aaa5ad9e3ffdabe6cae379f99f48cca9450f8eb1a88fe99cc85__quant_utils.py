#!/usr/bin/env python3
"""Quantization helpers. Torch is imported lazily by call sites."""


def symmetric_quant_dequant(x, bits, axis=None, eps=1e-12):
    import torch
    if bits >= 16:
        return x, torch.ones((), device=x.device, dtype=torch.float32)
    qmax = (2 ** (bits - 1)) - 1
    if axis is None:
        scale = x.detach().abs().amax().clamp_min(eps) / qmax
    else:
        scale = x.detach().abs().amax(dim=axis, keepdim=True).clamp_min(eps) / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q * scale, scale


def groupwise_weight_quant_dequant(w, bits=4, group_size=128):
    import torch
    if bits >= 16:
        return w
    orig_shape = w.shape
    flat = w.reshape(w.shape[0], -1)
    pad = (group_size - flat.shape[1] % group_size) % group_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    grouped = flat.reshape(flat.shape[0], -1, group_size)
    q, _ = symmetric_quant_dequant(grouped, bits=bits, axis=2)
    q = q.reshape(flat.shape[0], -1)
    if pad:
        q = q[:, :-pad]
    return q.reshape(orig_shape)


def tensor_stats(x):
    import torch
    x32 = x.detach().float()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "norm_l2": float(torch.linalg.vector_norm(x32).item()),
        "max_abs": float(x32.abs().amax().item()),
        "mean_abs": float(x32.abs().mean().item()),
    }


def error_record(fp, qdq):
    import torch
    fp32 = fp.detach().float()
    q32 = qdq.detach().float()
    err = fp32 - q32
    norm = torch.linalg.vector_norm(fp32).clamp_min(1e-12)
    err_norm = torch.linalg.vector_norm(err)
    return {
        "state_norm_l2": float(norm.item()),
        "quant_error_l2": float(err_norm.item()),
        "relative_error": float((err_norm / norm).item()),
    }
