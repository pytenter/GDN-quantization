@overload
def norm(
    input,
    p="fro",
    dim=None,
    keepdim=False,
    out=None,
    dtype=None,
):
    # type: (Tensor, str, Optional[List[int]], bool, Optional[Tensor], Optional[int]) -> Tensor
    pass
