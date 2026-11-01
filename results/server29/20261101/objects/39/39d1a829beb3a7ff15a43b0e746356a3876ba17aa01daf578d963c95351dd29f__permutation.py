"""Fixed-axis permutation primitives only; no unconditional candidate search."""
from __future__ import annotations
import torch
from reference_codec import cpu_state,architecture_name,owned


def validate_permutation(p,n):
    p=torch.tensor(p,dtype=torch.int64,device="cpu") if not isinstance(p,torch.Tensor) else p
    if p.device.type!="cpu" or p.dtype!=torch.int64 or p.ndim!=1 or len(p)!=n:
        raise ValueError("Permutation must be CPU INT64 vector of the axis length")
    if not torch.equal(torch.sort(p).values,torch.arange(n,dtype=torch.int64,device="cpu")):
        raise ValueError("Permutation is not a bijection")
    return p


def inverse_permutation(p):
    p=validate_permutation(p,len(p));return owned(torch.argsort(p))


def permute_state(x,p,architecture):
    x=cpu_state(x);architecture=architecture_name(architecture)
    axis=-2 if architecture=="gdn" else -1;p=validate_permutation(p,x.shape[axis])
    return owned(torch.index_select(x,axis,p))


def inverse_permute_state(x,p,architecture):
    return permute_state(x,inverse_permutation(p),architecture)


def bit_reverse_permutation(n=128):
    if n<1 or n&(n-1):raise ValueError("Bit reversal requires a positive power of two")
    bits=(n-1).bit_length()
    return torch.tensor([int(format(i,f"0{bits}b")[::-1],2) if bits else 0 for i in range(n)],dtype=torch.int64,device="cpu")


def fixed_random_permutation(n,seed):
    if seed not in (17,29,43):raise ValueError("Only preregistered random seeds 17/29/43")
    generator=torch.Generator(device="cpu");generator.manual_seed(seed)
    return torch.randperm(n,generator=generator,device="cpu")


def within_group_reversal(n=128,group_size=64):
    if n%group_size:raise ValueError("group_size must divide axis")
    return torch.arange(n,dtype=torch.int64,device="cpu").reshape(-1,group_size).flip(-1).reshape(-1)


def require_search_gate(gate):
    required=("real_calibration_states", "aligned_real_future_query", "state_improvement_ge_5pct",
              "readout_improvement_ge_5pct", "tail_gate_pass", "validation_not_used_for_selection")
    if not isinstance(gate,dict) or not all(gate.get(k) is True for k in required):
        raise RuntimeError("NOT_RUN_BY_PRE_REGISTERED_GATE")
    return dict(status="GATE_PASSED",random_seeds=[17,29,43],max_cross_group_swaps=32,
                search_performed_here=False)

