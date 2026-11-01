@_overload
def interpolate(  # noqa: F811
    input: Tensor,
    size: Optional[list[int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> Tensor:
    pass
