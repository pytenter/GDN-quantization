@overload
def tensordot(  # noqa: F811
    a,
    b,
    dims: tuple[list[int], list[int]],
    out: Optional[torch.Tensor] = None,
):
    pass
