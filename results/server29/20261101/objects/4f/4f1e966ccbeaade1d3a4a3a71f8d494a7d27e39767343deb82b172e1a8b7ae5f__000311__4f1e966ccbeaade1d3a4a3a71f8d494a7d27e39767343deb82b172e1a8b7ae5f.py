@overload
def tensordot(  # noqa: F811
    a,
    b,
    dims: torch.Tensor,
    out: Optional[torch.Tensor] = None,
):
    pass
