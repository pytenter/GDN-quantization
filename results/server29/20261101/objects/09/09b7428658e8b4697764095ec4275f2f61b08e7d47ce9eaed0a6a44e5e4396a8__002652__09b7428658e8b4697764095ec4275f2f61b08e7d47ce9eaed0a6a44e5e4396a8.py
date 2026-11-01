def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.as_tensor,self.as_none,)==(other.as_tensor,other.as_none,)
  return NotImplemented
 return __eq__