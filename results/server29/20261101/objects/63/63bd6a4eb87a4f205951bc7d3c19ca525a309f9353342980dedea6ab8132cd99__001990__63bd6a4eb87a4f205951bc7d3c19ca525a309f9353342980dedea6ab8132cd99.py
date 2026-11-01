def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.enabled,self.level,self.parameters,self.properties,self.rank,)==(other.enabled,other.level,other.parameters,other.properties,other.rank,)
  return NotImplemented
 return __eq__