@overload
def tensordot(
    a,
    b,
    dims: int = 2,
    out: Optional[torch.Tensor] = None,
):
    pass
