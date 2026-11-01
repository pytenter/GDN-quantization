def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.driver,self.extensions,self.properties,)==(other.driver,other.extensions,other.properties,)
  return NotImplemented
 return __eq__