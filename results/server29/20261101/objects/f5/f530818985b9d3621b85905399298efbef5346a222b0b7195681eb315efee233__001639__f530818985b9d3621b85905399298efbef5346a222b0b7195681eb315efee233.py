def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.automatic_dynamic,)==(other.automatic_dynamic,)
  return NotImplemented
 return __eq__