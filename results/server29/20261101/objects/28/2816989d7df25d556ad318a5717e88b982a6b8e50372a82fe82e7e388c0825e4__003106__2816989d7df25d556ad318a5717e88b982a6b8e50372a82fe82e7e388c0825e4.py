def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(dtype={self.dtype!r}, quant_min_lower_bound={self.quant_min_lower_bound!r}, quant_max_upper_bound={self.quant_max_upper_bound!r}, scale_min_lower_bound={self.scale_min_lower_bound!r}, scale_max_upper_bound={self.scale_max_upper_bound!r}, scale_exact_match={self.scale_exact_match!r}, zero_point_exact_match={self.zero_point_exact_match!r})"
 return __repr__