def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(derived_from={self.derived_from!r}, derive_qparams_fn={self.derive_qparams_fn!r}, dtype={self.dtype!r}, quant_min={self.quant_min!r}, quant_max={self.quant_max!r}, qscheme={self.qscheme!r}, ch_axis={self.ch_axis!r}, is_dynamic={self.is_dynamic!r})"
 return __repr__