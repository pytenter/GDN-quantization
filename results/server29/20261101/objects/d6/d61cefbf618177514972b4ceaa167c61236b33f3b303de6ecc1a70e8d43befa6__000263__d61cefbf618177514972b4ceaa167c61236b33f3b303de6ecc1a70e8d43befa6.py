@_overload
def upsample(  # noqa: F811
    input: Tensor,
    size: Optional[list[int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
) -> Tensor:  # noqa: B950
    pass
