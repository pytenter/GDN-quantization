def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(device={self.device!r}, dtype={self.dtype!r}, inner_fn={self.inner_fn!r}, ranges={self.ranges!r}, reduction_ranges={self.reduction_ranges!r}, reduction_type={self.reduction_type!r}, src_dtype={self.src_dtype!r}, reduction_hint={self.reduction_hint!r})"
 return __repr__