def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.codegen_low_precision,)==(other.codegen_low_precision,)
  return NotImplemented
 return __eq__