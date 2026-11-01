def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.dtype,self.scale,self.zero_point,self.quant_min,self.quant_max,self.qscheme,self.is_dynamic,)==(other.dtype,other.scale,other.zero_point,other.quant_min,other.quant_max,other.qscheme,other.is_dynamic,)
  return NotImplemented
 return __eq__