def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.node,self.can_inplace,self.is_weak,)==(other.node,other.can_inplace,other.is_weak,)
  return NotImplemented
 return __eq__