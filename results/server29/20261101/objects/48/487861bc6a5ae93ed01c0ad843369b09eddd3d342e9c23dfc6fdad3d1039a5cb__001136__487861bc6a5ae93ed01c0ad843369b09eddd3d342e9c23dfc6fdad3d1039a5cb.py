def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.autotune_key,)==(other.autotune_key,)
  return NotImplemented
 return __eq__