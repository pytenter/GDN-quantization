@overload
@torch._jit_internal._overload_method  # noqa: F811
def forward(
    self, input: PackedSequence, hx: Optional[tuple[Tensor, Tensor]] = None
) -> tuple[PackedSequence, tuple[Tensor, Tensor]]:  # noqa: F811
    pass
