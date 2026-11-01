def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.reduce_op,self.mask_buffer,self.offset_shape,self.offset_dim,)==(other.reduce_op,other.mask_buffer,other.offset_shape,other.offset_dim,)
  return NotImplemented
 return __eq__