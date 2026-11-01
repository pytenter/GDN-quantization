"""Fixed-order inclusive scan. No cumsum, CPU transfer, random operations or promotion."""
import torch

def deterministic_cumsum_fixed_order(x, dim=-1):
    dim = dim % x.ndim
    if x.shape[dim] == 0:
        return torch.empty_like(x)
    values = []
    acc = x.select(dim, 0)
    values.append(acc)
    for i in range(1, x.shape[dim]):
        acc = acc + x.select(dim, i)
        values.append(acc)
    return torch.stack(values, dim=dim)
