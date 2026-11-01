def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.display_base,self.properties,)==(other.display_base,other.properties,)
  return NotImplemented
 return __eq__