def __create_fn__():
 def __hash__(self):
  return hash((self.dtype,self.scale,self.zero_point,self.quant_min,self.quant_max,self.qscheme,self.is_dynamic,))
 return __hash__