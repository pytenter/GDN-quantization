def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.entries,self.index,)==(other.entries,other.index,)
  return NotImplemented
 return __eq__