def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.group_size,)==(other.group_size,)
  return NotImplemented
 return __eq__