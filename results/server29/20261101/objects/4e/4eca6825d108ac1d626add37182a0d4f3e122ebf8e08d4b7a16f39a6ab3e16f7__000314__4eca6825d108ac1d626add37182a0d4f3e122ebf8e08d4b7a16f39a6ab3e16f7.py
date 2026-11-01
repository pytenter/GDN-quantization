@overload
def norm(  # noqa: F811
    input,
    p="fro",
    dim=None,
    keepdim=False,
    out=None,
    dtype=None,
):
    # type: (Tensor, Optional[number], Optional[int], bool, Optional[Tensor], Optional[int]) -> Tensor
    pass
