def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.quant_method,)==(other.quant_method,)
  return NotImplemented
 return __eq__