@overload
@torch._jit_internal._overload_method  # noqa: F811
def forward(
    self, input: Tensor, hx: Optional[Tensor] = None
) -> tuple[Tensor, Tensor]:
    pass
