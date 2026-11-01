def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.value,)==(other.value,)
  return NotImplemented
 return __eq__