def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(dtype={self.dtype!r}, scale={self.scale!r}, zero_point={self.zero_point!r}, quant_min={self.quant_min!r}, quant_max={self.quant_max!r}, qscheme={self.qscheme!r}, is_dynamic={self.is_dynamic!r})"
 return __repr__