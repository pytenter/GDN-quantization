def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.pin_memory,)==(other.pin_memory,)
  return NotImplemented
 return __eq__