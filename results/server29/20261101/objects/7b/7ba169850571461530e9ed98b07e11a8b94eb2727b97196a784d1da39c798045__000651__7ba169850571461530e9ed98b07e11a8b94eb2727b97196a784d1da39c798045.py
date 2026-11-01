def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.size,self.stride,self.storage_offset,)==(other.size,other.stride,other.storage_offset,)
  return NotImplemented
 return __eq__