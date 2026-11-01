def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.norm_type,)==(other.norm_type,)
  return NotImplemented
 return __eq__