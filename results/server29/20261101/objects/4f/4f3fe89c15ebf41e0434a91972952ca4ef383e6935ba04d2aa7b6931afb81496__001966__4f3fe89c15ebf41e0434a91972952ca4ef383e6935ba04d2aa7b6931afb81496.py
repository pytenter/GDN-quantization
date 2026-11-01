def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.guid,self.id,self.index,self.properties,self.tool_component,)==(other.guid,other.id,other.index,other.properties,other.tool_component,)
  return NotImplemented
 return __eq__