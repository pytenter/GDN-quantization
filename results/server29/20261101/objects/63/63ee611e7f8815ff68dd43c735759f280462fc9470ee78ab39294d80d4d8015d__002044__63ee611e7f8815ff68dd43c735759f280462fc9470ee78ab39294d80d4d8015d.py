def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.kind,self.guid,self.justification,self.location,self.properties,self.state,)==(other.kind,other.guid,other.justification,other.location,other.properties,other.state,)
  return NotImplemented
 return __eq__