def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.alias_set,self.is_write,self.name,)==(other.alias_set,other.is_write,other.name,)
  return NotImplemented
 return __eq__