def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.device,self.dtype,self.sizes,self.strides,self.offset,self.name,)==(other.device,other.dtype,other.sizes,other.strides,other.offset,other.name,)
  return NotImplemented
 return __eq__