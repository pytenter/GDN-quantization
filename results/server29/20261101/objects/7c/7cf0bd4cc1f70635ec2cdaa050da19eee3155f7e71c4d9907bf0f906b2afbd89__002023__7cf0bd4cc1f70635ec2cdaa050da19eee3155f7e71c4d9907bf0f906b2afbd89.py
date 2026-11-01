def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.id,self.source_node_id,self.target_node_id,self.label,self.properties,)==(other.id,other.source_node_id,other.target_node_id,other.label,other.properties,)
  return NotImplemented
 return __eq__