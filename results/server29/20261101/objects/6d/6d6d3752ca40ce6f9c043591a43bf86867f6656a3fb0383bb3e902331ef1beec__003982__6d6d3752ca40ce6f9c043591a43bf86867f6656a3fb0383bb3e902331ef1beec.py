def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(use_autocast_during_eval={self.use_autocast_during_eval!r}, margin={self.margin!r}, interval={self.interval!r}, fp8_format={self.fp8_format!r}, amax_history_len={self.amax_history_len!r}, amax_compute_algo={self.amax_compute_algo!r}, override_linear_precision={self.override_linear_precision!r}, use_mxfp8_block_scaling={self.use_mxfp8_block_scaling!r})"
 return __repr__