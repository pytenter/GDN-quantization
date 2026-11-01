def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.__hash__,)==(other.__hash__,)
  return NotImplemented
 return __eq__