def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.ptr,self.allocation_id,)==(other.ptr,other.allocation_id,)
  return NotImplemented
 return __eq__