def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.alias_set,self.is_write,self.alias_set_after,)==(other.alias_set,other.is_write,other.alias_set_after,)
  return NotImplemented
 return __eq__