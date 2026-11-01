@overload
def tensordot(  # noqa: F811
    a,
    b,
    dims: list[list[int]],
    out: Optional[torch.Tensor] = None,
):
    pass
