def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.dtype,self.is_scalar,)==(other.dtype,other.is_scalar,)
  return NotImplemented
 return __eq__