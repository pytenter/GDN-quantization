def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.tensor,self.metadata,)==(other.tensor,other.metadata,)
  return NotImplemented
 return __eq__