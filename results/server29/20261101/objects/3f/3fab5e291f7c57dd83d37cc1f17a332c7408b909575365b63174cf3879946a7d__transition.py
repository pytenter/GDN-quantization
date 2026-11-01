"""Actual FLA KDA fused recurrence, with zero value for the shadow path.

No decay-only approximation is used here. Inputs are frozen pre-kernel q/k/v,
already canonical log_decay and processed beta, with in-kernel q/k L2 norm.
"""
import torch
from fla.ops.kda import fused_recurrent_kda

def update_pre_q(state, record, value=None):
    _, s = fused_recurrent_kda(q=record['q'],k=record['k'],
        v=record['v'] if value is None else value,
        g=record['log_decay'].float(), beta=record['beta'].float(),
        initial_state=state, output_final_state=True,
        use_qk_l2norm_in_kernel=True,use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,state_v_first=False)
    return s

def propagate_zero_write(state, record):
    return update_pre_q(state,record,torch.zeros_like(record['v']))

def qdq_r128(state):
    # Matches the actual Phase2 persistent-cache boundary: FP32 recurrent state,
    # symmetric rowwise Value-dimension INT8, dequant written to the cache dtype.
    source=state.float()
    scale=source.abs().amax(dim=-1,keepdim=True).clamp_min(1e-12)/127.
    code=torch.round(source/scale).clamp(-127,127)
    return (code*scale).to(state.dtype), code.to(torch.int8), scale

