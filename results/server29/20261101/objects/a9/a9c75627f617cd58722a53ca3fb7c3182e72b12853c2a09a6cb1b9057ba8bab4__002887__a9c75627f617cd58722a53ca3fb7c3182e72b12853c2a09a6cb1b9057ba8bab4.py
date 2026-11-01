def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(device={self.device!r}, dtype={self.dtype!r}, inner_fn={self.inner_fn!r}, ranges={self.ranges!r}, scan_ranges={self.scan_ranges!r}, size={self.size!r}, combine_fn={self.combine_fn!r}, reindex={self.reindex!r}, reduction_hint={self.reduction_hint!r}, output_index={self.output_index!r}, dtypes={self.dtypes!r}, inner_fns={self.inner_fns!r})"
 return __repr__