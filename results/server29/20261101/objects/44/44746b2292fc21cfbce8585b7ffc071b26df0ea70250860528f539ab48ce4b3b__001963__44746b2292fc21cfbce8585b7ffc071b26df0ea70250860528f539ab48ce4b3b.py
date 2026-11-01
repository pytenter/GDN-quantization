def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.guid,self.index,self.name,self.properties,)==(other.guid,other.index,other.name,other.properties,)
  return NotImplemented
 return __eq__