def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.inplace_idx,self.metadata,self.view_idx,)==(other.inplace_idx,other.metadata,other.view_idx,)
  return NotImplemented
 return __eq__