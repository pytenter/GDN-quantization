def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.base,self.index,self.index_is_slice,)==(other.base,other.index,other.index_is_slice,)
  return NotImplemented
 return __eq__