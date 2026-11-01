def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.shape,self.block_shape,self.strides,self.offsets,)==(other.shape,other.block_shape,other.strides,other.offsets,)
  return NotImplemented
 return __eq__