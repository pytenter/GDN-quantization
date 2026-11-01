@overload  # type: ignore[override]
@torch._jit_internal._overload_method  # noqa: F811
def forward(
    self, input: Tensor, hx: Optional[tuple[Tensor, Tensor]] = None
) -> tuple[Tensor, tuple[Tensor, Tensor]]:  # noqa: F811
    pass
