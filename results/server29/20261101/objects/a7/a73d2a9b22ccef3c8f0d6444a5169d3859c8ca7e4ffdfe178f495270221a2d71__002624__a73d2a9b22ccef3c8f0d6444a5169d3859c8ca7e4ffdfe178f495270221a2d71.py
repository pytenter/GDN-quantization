def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.type,self.index,)==(other.type,other.index,)
  return NotImplemented
 return __eq__