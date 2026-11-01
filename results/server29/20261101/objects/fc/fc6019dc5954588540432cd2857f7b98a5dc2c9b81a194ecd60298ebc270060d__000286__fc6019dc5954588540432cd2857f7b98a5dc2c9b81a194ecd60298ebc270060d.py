@overload
@torch._jit_internal._overload_method  # noqa: F811
def forward(
    self, input: PackedSequence, hx: Optional[Tensor] = None
) -> tuple[PackedSequence, Tensor]:  # noqa: F811
    pass
