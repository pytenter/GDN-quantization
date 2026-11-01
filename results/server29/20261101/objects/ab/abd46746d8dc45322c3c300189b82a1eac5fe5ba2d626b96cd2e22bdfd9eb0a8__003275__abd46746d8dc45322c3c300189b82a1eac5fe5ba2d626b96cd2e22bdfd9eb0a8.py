def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(var={self.var!r}, size={self.size!r}, offset={self.offset!r}, tiled_size={self.tiled_size!r}, steps={self.steps!r}, parallel={self.parallel!r}, simd_omp={self.simd_omp!r}, simd_vec={self.simd_vec!r}, collapsed={self.collapsed!r}, is_reduction={self.is_reduction!r})"
 return __repr__