def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.derived_from,self.derive_qparams_fn,self.dtype,self.quant_min,self.quant_max,self.qscheme,self.ch_axis,self.is_dynamic,)==(other.derived_from,other.derive_qparams_fn,other.dtype,other.quant_min,other.quant_max,other.qscheme,other.ch_axis,other.is_dynamic,)
  return NotImplemented
 return __eq__