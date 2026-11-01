def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.tensor_size,self.tensor_stride,self.tensor_storage_offset,self.is_nested,)==(other.tensor_size,other.tensor_stride,other.tensor_storage_offset,other.is_nested,)
  return NotImplemented
 return __eq__