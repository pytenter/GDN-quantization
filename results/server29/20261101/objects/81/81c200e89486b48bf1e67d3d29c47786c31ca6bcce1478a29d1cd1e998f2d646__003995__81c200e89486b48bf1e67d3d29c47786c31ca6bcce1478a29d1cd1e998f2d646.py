def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.shape,self.dtype,)==(other.shape,other.dtype,)
  return NotImplemented
 return __eq__