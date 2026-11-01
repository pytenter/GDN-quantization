def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.dtype,self.layout,self.requires_grad,self.memory_format,self.pin_memory,)==(other.dtype,other.layout,other.requires_grad,other.memory_format,other.pin_memory,)
  return NotImplemented
 return __eq__