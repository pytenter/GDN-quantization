def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.absolute_address,self.fully_qualified_name,self.index,self.kind,self.length,self.name,self.offset_from_parent,self.parent_index,self.properties,self.relative_address,)==(other.absolute_address,other.fully_qualified_name,other.index,other.kind,other.length,other.name,other.offset_from_parent,other.parent_index,other.properties,other.relative_address,)
  return NotImplemented
 return __eq__