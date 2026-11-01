def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.persistent,self.looped,)==(other.persistent,other.looped,)
  return NotImplemented
 return __eq__