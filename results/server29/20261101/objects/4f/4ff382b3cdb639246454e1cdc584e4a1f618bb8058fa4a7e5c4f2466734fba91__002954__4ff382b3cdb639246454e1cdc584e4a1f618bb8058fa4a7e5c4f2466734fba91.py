def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.dtype,self.device,)==(other.dtype,other.device,)
  return NotImplemented
 return __eq__