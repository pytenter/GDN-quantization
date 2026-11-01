#!/usr/bin/env python3
from dataclasses import dataclass

import torch
import torch.nn.functional as F


EPS = 1e-12
L2NORM_EPS = 1e-6


@dataclass
class KdaReferenceResult:
    output: torch.Tensor
    final_state: torch.Tensor
    stage_tensors: dict


def kernel_stage_ids():
    return [
        "K0_INPUT_ERROR",
        "K1_DECAYED_ERROR",
        "K2_UPDATE_ERROR",
        "K3_READOUT_ERROR",
        "K4_CORE_OUTPUT",
    ]


def _as_kv_state(state, state_v_first=False):
    if state_v_first:
        return state.transpose(-1, -2).contiguous()
    return state


def _from_kv_state(state, state_v_first=False):
    if state_v_first:
        return state.transpose(-1, -2).contiguous()
    return state


def _expand_qk_to_value_heads(q, k, hv):
    h = q.shape[2]
    if hv % h != 0:
        raise ValueError(f"HV={hv} must be divisible by H={h}")
    repeat = hv // h
    return q.repeat_interleave(repeat, dim=2), k.repeat_interleave(repeat, dim=2)


def _normalize_qk(q, k, enabled):
    if not enabled:
        return q.float(), k.float()
    qf = q.float()
    kf = k.float()
    qf = qf / torch.sqrt(torch.sum(qf * qf, dim=-1, keepdim=True) + L2NORM_EPS)
    kf = kf / torch.sqrt(torch.sum(kf * kf, dim=-1, keepdim=True) + L2NORM_EPS)
    return qf, kf


def _decay_log(g, A_log=None, dt_bias=None, use_gate_in_kernel=False, lower_bound=None):
    gf = g.float()
    if not use_gate_in_kernel:
        return gf
    if A_log is None:
        raise ValueError("A_log is required when use_gate_in_kernel=True")
    if dt_bias is not None:
        bias = dt_bias.float().reshape(1, 1, gf.shape[2], gf.shape[3])
        gf = gf + bias
    a = torch.exp(A_log.float()).reshape(1, 1, -1, 1)
    if lower_bound is not None:
        return float(lower_bound) * torch.sigmoid(a * gf)
    return -a * F.softplus(gf)


def _beta_value(beta, v, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False):
    bf = beta.float()
    if use_beta_sigmoid_in_kernel:
        bf = torch.sigmoid(bf)
        if allow_neg_eigval:
            bf = bf * 2.0
    if bf.ndim == v.ndim - 1:
        bf = bf.unsqueeze(-1)
    return bf


def _prepared_drivers(
    q,
    k,
    v,
    g,
    beta,
    A_log=None,
    dt_bias=None,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    lower_bound=None,
):
    hv = v.shape[2]
    qx, kx = _expand_qk_to_value_heads(q, k, hv)
    qx, kx = _normalize_qk(qx, kx, use_qk_l2norm_in_kernel)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    qx = qx * float(scale)
    decay = _decay_log(g, A_log=A_log, dt_bias=dt_bias, use_gate_in_kernel=use_gate_in_kernel, lower_bound=lower_bound)
    beta_x = _beta_value(beta, v, use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel, allow_neg_eigval=allow_neg_eigval)
    return qx, kx, v.float(), decay, beta_x


def kda_one_step_reference(
    q,
    k,
    v,
    g,
    beta,
    A_log=None,
    dt_bias=None,
    initial_state=None,
    scale=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    lower_bound=None,
    state_v_first=False,
):
    if q.shape[1] != 1:
        raise ValueError("kda_one_step_reference only supports one recurrent decode token")
    qx, kx, vf, decay_log, beta_x = _prepared_drivers(
        q,
        k,
        v,
        g,
        beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        lower_bound=lower_bound,
    )
    bq = qx[:, 0]
    bk = kx[:, 0]
    bv = vf[:, 0]
    bg = decay_log[:, 0]
    bbeta = beta_x[:, 0]
    if initial_state is None:
        state = torch.zeros(q.shape[0], v.shape[2], k.shape[-1], v.shape[-1], device=q.device, dtype=torch.float32)
    else:
        state = _as_kv_state(initial_state.float(), state_v_first=state_v_first)

    decayed = state * torch.exp(bg).unsqueeze(-1)
    projected_v = bv - torch.sum(decayed * bk.unsqueeze(-1), dim=-2)
    update_value = projected_v * bbeta
    updated = decayed + bk.unsqueeze(-1) * update_value.unsqueeze(-2)
    readout = torch.sum(updated * bq.unsqueeze(-1), dim=-2).unsqueeze(1)
    final_state = _from_kv_state(updated, state_v_first=state_v_first) if output_final_state else None
    stages = {
        "prepared_q": bq,
        "prepared_k": bk,
        "decay_log": bg,
        "decay_multiplier": torch.exp(bg),
        "projected_v": projected_v,
        "update_value": update_value,
        "decayed_state": _from_kv_state(decayed, state_v_first=state_v_first),
        "updated_state": _from_kv_state(updated, state_v_first=state_v_first),
        "readout_float": readout,
    }
    return KdaReferenceResult(output=readout.to(dtype=q.dtype), final_state=final_state, stage_tensors=stages)


def kda_exact_state_error_transfer(
    q,
    k,
    v,
    g,
    beta,
    error_state,
    A_log=None,
    dt_bias=None,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    lower_bound=None,
    state_v_first=False,
):
    qx, kx, vf, decay_log, beta_x = _prepared_drivers(
        q,
        k,
        v,
        g,
        beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        lower_bound=lower_bound,
    )
    bq = qx[:, 0]
    bk = kx[:, 0]
    bg = decay_log[:, 0]
    bbeta = beta_x[:, 0]
    err = _as_kv_state(error_state.float(), state_v_first=state_v_first)
    decayed_err = err * torch.exp(bg).unsqueeze(-1)
    correction = torch.sum(decayed_err * bk.unsqueeze(-1), dim=-2)
    update_err = decayed_err - bk.unsqueeze(-1) * (correction * bbeta).unsqueeze(-2)
    readout_err = torch.sum(update_err * bq.unsqueeze(-1), dim=-2).unsqueeze(1)
    stage_tensors = {
        "K0_INPUT_ERROR": error_state.float(),
        "K1_DECAYED_ERROR": _from_kv_state(decayed_err, state_v_first=state_v_first),
        "K2_UPDATE_ERROR": _from_kv_state(update_err, state_v_first=state_v_first),
        "K3_READOUT_ERROR": readout_err,
        "K4_CORE_OUTPUT": readout_err.to(dtype=q.dtype),
        "K2_KEY_CORRECTION": correction,
    }
    return KdaReferenceResult(
        output=stage_tensors["K4_CORE_OUTPUT"],
        final_state=stage_tensors["K2_UPDATE_ERROR"],
        stage_tensors=stage_tensors,
    )
