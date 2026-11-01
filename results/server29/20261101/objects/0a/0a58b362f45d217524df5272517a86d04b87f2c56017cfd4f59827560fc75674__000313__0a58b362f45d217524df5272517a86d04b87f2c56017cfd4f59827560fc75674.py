@overload
def norm(  # noqa: F811
    input,
    p="fro",
    dim=None,
    keepdim=False,
    out=None,
    dtype=None,
):
    # type: (Tensor, Optional[number], Optional[List[int]], bool, Optional[Tensor], Optional[int]) -> Tensor
    pass
