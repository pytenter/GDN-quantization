def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.target,self.description,self.kinds,self.properties,)==(other.target,other.description,other.kinds,other.properties,)
  return NotImplemented
 return __eq__