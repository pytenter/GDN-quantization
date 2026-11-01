def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.size,self.layout,self.is_sparse,self.stride,self.storage_offset,self.is_conj,self.is_neg,)==(other.size,other.layout,other.is_sparse,other.stride,other.storage_offset,other.is_conj,other.is_neg,)
  return NotImplemented
 return __eq__