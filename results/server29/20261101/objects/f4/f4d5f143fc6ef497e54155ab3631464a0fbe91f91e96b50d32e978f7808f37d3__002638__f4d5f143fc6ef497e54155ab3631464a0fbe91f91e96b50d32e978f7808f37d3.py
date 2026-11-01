def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.dtype,self.sizes,self.requires_grad,self.device,self.strides,self.storage_offset,self.layout,)==(other.dtype,other.sizes,other.requires_grad,other.device,other.strides,other.storage_offset,other.layout,)
  return NotImplemented
 return __eq__