def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.scalar,self.size,self.stride,)==(other.scalar,other.size,other.stride,)
  return NotImplemented
 return __eq__