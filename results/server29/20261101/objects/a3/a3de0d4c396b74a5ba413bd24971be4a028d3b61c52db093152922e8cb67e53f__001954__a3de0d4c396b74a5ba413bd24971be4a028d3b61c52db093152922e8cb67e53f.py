def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.decorated_name,self.fully_qualified_name,self.index,self.kind,self.name,self.parent_index,self.properties,)==(other.decorated_name,other.fully_qualified_name,other.index,other.kind,other.name,other.parent_index,other.properties,)
  return NotImplemented
 return __eq__