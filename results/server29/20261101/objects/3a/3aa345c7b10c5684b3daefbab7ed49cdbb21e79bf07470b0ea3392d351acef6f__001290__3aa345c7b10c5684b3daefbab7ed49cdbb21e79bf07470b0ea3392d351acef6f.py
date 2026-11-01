def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.scope,self.name,)==(other.scope,other.name,)
  return NotImplemented
 return __eq__