def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.dtype,self.layout,self.device,self.pin_memory,)==(other.dtype,other.layout,other.device,other.pin_memory,)
  return NotImplemented
 return __eq__