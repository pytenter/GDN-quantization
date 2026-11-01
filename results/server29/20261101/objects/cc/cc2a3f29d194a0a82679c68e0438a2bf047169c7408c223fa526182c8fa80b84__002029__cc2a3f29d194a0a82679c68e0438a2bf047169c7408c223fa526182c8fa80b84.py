def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.id,self.children,self.label,self.location,self.properties,)==(other.id,other.children,other.label,other.location,other.properties,)
  return NotImplemented
 return __eq__